"""本体 DAG の継続学習を実行する（v1.4）。"""

from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurograph.lateral_v13b import UnifiedBrainV13b
from neurograph.run_v13_unified import build_shuffled_multidomain_bytes, fmt_3domains
from neurograph.utf8_dag_v88 import UTF8Corpus


def load_utf8_slice(
    data_dir: str,
    name: str,
    start_char: int = 0,
    train_chars: int = 20_000_000,
    val_chars: int = 120_000,
) -> UTF8Corpus:
    base = Path(data_dir)
    mm_tr = np.memmap(base / f"{name}.train", dtype=np.uint32, mode="r")
    end_char = min(len(mm_tr), start_char + train_chars)
    tr_u32 = mm_tr[start_char:end_char]
    va_u32 = np.memmap(base / f"{name}.val", dtype=np.uint32, mode="r")[:val_chars]

    tr_str = np.array(tr_u32, dtype="<u4").tobytes().decode("utf-32-le", errors="ignore")
    va_str = np.array(va_u32, dtype="<u4").tobytes().decode("utf-32-le", errors="ignore")
    tr_b = np.frombuffer(tr_str.encode("utf-8"), dtype=np.uint8).copy()
    va_b = np.frombuffer(va_str.encode("utf-8"), dtype=np.uint8).copy()
    return UTF8Corpus(
        train_bytes=tr_b,
        val_bytes=va_b,
        train_num_chars=len(tr_str),
        val_num_chars=len(va_str),
        bytes_per_char_train=len(tr_b) / max(1, len(tr_str)),
        bytes_per_char_val=len(va_b) / max(1, len(va_str)),
    )


def eval_3domains(brain: UnifiedBrainV13b, vals: dict, val_streams: int, use_exec: bool, use_lat: bool) -> dict:
    out = {}
    for k, corp in vals.items():
        out[k] = brain.eval_with_idf_lateral_wave(
            corp.val_bytes, batch_streams=val_streams, bytes_per_char=corp.bytes_per_char_val,
            use_executive=use_exec, use_lateral=use_lat,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="brain_v13_unified_6sys.pt")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("ENNE_DATA_DIR", "data"))
    parser.add_argument("--start-char", type=int, default=10_000_000)
    parser.add_argument("--chars-per-domain", type=int, default=20_000_000)
    parser.add_argument("--val-chars", type=int, default=120_000)
    parser.add_argument("--pool-capacity", type=int, default=3_600_000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--batch-streams", type=int, default=3072)
    parser.add_argument("--val-streams", type=int, default=128)
    parser.add_argument("--save-ckpt", type=str, default="brain_v14_90m_6sys.pt")
    parser.add_argument("--out", type=str, default="results_v14_90m_6sys.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    t0 = time.time()
    brain = UnifiedBrainV13b.from_checkpoint(
        args.ckpt,
        device,
        new_pool_capacity=args.pool_capacity,
        cfg_overrides={
            "lr_feat": 0.20,
            "lr_basis": 0.015,
            "lr_sharp": 0.26,
            "lr_slot_read": 0.065,
            "lr_slot_mod": 0.038,
            "lr_slot_gate": 0.024,
        },
    )
    print(
        f"Transplanted {args.ckpt} into {brain.P:,}-neuron bf16 vessel in {time.time()-t0:.1f}s | "
        f"Alive={brain.num_hidden_alive:,}N / {brain.edges.count():,}E | LTM={brain.ltm.count_active()/1e6:.2f}M",
        flush=True,
    )

    c_easy = load_utf8_slice(args.data_dir, "easy_ja", start_char=args.start_char, train_chars=args.chars_per_domain, val_chars=args.val_chars)
    c_aozora = load_utf8_slice(args.data_dir, "aozora", start_char=args.start_char, train_chars=args.chars_per_domain, val_chars=args.val_chars)
    c_wiki = load_utf8_slice(args.data_dir, "wikipedia", start_char=args.start_char, train_chars=args.chars_per_domain, val_chars=args.val_chars)
    vals = {"easy_ja": c_easy, "aozora": c_aozora, "wikipedia": c_wiki}

    mixed_unseen = build_shuffled_multidomain_bytes(c_easy.train_bytes, c_aozora.train_bytes, c_wiki.train_bytes, block_bytes=4096, seed=140)
    avg_bpc_tr = (c_easy.bytes_per_char_train + c_aozora.bytes_per_char_train + c_wiki.bytes_per_char_train) / 3.0
    total_new_ch = c_easy.train_num_chars + c_aozora.train_num_chars + c_wiki.train_num_chars
    print(
        f"Loaded UNSEEN 3-Domain Raw UTF-8 Stream: {len(mixed_unseen):,}B ({total_new_ch/1e6:.1f}M new chars: "
        f"easy={c_easy.train_num_chars/1e6:.1f}M, aozora={c_aozora.train_num_chars/1e6:.1f}M, wiki={c_wiki.train_num_chars/1e6:.1f}M)",
        flush=True,
    )

    ev0 = eval_3domains(brain, vals, args.val_streams, use_exec=True, use_lat=False)
    print(f"[S0 Transplanted 30M Baseline] {fmt_3domains(ev0)}", flush=True)

    seg_len = len(mixed_unseen) // args.rounds
    history = []
    sleep_logs = []

    for r in range(args.rounds):
        rt = time.time()
        seg = mixed_unseen[r * seg_len : (r + 1) * seg_len]
        lr_s = 0.72 / (1.0 + 0.16 * r)
        tr = brain.run_stream(
            seg, train=True, batch_streams=args.batch_streams, lr_scale=lr_s,
            bytes_per_char=avg_bpc_tr, use_slots=True, use_ltm=False, use_stm=False, use_executive=False, populate_ltm=True,
        )
        ap = brain.apoptosis()
        fu = brain.fuse_similar_neurons(seed=400 + r * 23)

        sp_r = brain.ltm.sleep_prune()
        sleep_logs.append({"stage": f"after_round_{r+1}", **sp_r})
        print(
            f"  [Nightly Sleep R{r+1}] Pruned {sp_r['pruned']:,}/{sp_r['before']:,} ({sp_r['prune_pct']}%) -> {sp_r['after']/1e6:.2f}M LTM",
            flush=True,
        )

        ev_r = eval_3domains(brain, vals, args.val_streams, use_exec=False, use_lat=False)
        dt = time.time() - rt
        row = {
            "round": r + 1,
            "cum_chars_m": round(30.0 + (r + 1) * (total_new_ch / 1e6 / args.rounds), 1),
            "sec": round(dt, 1),
            "births": tr["births"], "deaths": ap["died"], "merged": fu["merged"], "divorces": tr["divorces"],
            "alive_hidden": fu["alive_hidden"], "edges": fu["edges"], "ltm_active": brain.ltm.count_active(),
            "domains": ev_r,
        }
        history.append(row)
        print(
            f"[round {r+1}/{args.rounds} | {row['cum_chars_m']:4.1f}Mch] {dt:5.1f}s | {fu['alive_hidden']:,}N / {fu['edges']:,}E | "
            f"LTM={brain.ltm.count_active()/1e6:.2f}M | {fmt_3domains(ev_r)}",
            flush=True,
        )
        if r == (args.rounds // 2 - 1):
            brain.save_full_checkpoint(args.save_ckpt.replace(".pt", "_mid.pt"))

    print(f"\n[LTM Post-Sleep Final] {brain.ltm.count_active()/1e6:.2f}M consolidated LTM entries", flush=True)

    rt = time.time()
    exec_fit = brain.fit_executive_modes(mixed_unseen[:8_000_000], passage_len=320, max_passages=16_000)
    print(f"[System 5 Executive] Refitted {exec_fit['num_modes']} discourse modes in {time.time()-rt:.1f}s", flush=True)

    rt = time.time()
    lat_stats = brain.build_idf_lateral_graph(
        mixed_unseen, doc_len=2048, num_docs_idf=6144, cooc_scan_bytes=20_000_000, cooc_batch_streams=2048
    )
    print(
        f"[System 6 IDF Lateral Graph] {lat_stats['num_concepts']:,} content concepts "
        f"(clean UTF-8: {lat_stats['clean_utf8_pct']}%), {lat_stats['num_edges']:,} PPMI synapses in {time.time()-rt:.1f}s",
        flush=True,
    )

    probes = ["猫", "世界大戦", "東京", "ドイツ", "日本", "学校", "江戸時代", "フランス"]
    waves_log = {}
    print("\n--- Live 2-Hop IDF Lateral Semantic Thought Waves (90M) ---", flush=True)
    for pr in probes:
        w_res = brain.trace_semantic_wave(pr, top_k=8)
        waves_log[pr] = w_res
        print(f"  [{pr}] ==wave==> " + ", ".join(f"{t}({s:.2f})" for t, s in w_res), flush=True)

    ev_4sys = eval_3domains(brain, vals, args.val_streams, use_exec=False, use_lat=False)
    ev_5sys = eval_3domains(brain, vals, args.val_streams, use_exec=True, use_lat=False)
    ev_6sys = eval_3domains(brain, vals, args.val_streams, use_exec=True, use_lat=True)

    c_easy_full = load_utf8_slice(args.data_dir, "easy_ja", start_char=0, train_chars=1000, val_chars=306_000)
    ev_easy_306k = brain.eval_with_idf_lateral_wave(
        c_easy_full.val_bytes, batch_streams=args.val_streams, bytes_per_char=c_easy_full.bytes_per_char_val,
        use_executive=True, use_lateral=True,
    )

    print("\n================ NEUROGRAPH v1.4 (90M CHARS / 3.6M VESSEL / 6-SYSTEM UNIFIED) ================", flush=True)
    print(f"v1.2 Ref (52M seq, D=128)     | easy_ja=4.3983(70.6%) | aozora=5.3076(66.5%) | wikipedia=4.9649(66.5%) | MEAN=4.8903", flush=True)
    print(f"v1.3 Ref (30M shuf, D=256)    | easy_ja=4.3019(70.8%) | aozora=5.1329(66.6%) | wikipedia=4.6887(66.7%) | MEAN=4.7078", flush=True)
    print(f"v1.4 4-Sys (90M, 3.6M vessel) | {fmt_3domains(ev_4sys)}", flush=True)
    print(f"v1.4 5-Sys (+24-Mode Exec)    | {fmt_3domains(ev_5sys)}", flush=True)
    print(f"v1.4 6-Sys (+IDF Lateral)     | {fmt_3domains(ev_6sys)}", flush=True)
    print(
        f"v1.4 Full easy_ja 306k Val    | bpb={ev_easy_306k['bpb']:.4f} | bpc_lossless={ev_easy_306k['bpc_lossless']:.4f} "
        f"(ppl={ev_easy_306k['ppl']:.2f}) | byte_top1={ev_easy_306k['byte_top1']*100:.1f}% | top5={ev_easy_306k['byte_top5']*100:.1f}%",
        flush=True,
    )

    brain.save_full_checkpoint(args.save_ckpt)
    print(f"Saved full bit-exact 6-system checkpoint: {args.save_ckpt}", flush=True)

    prompts = [
        ("Wikipedia歴史", "第二次世界大戦（だいにじせかいたいせん）は、1939年から1945年にかけて"),
        ("青空文庫文学", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"),
        ("東京・都市", "日本の東京は、江戸時代から続く歴史ある都市であり、"),
        ("ドイツ工業", "ドイツの自動車産業とエンジン技術は世界的に知られており、"),
        ("無限拡張🚀", "薔薇と憂鬱と🚀について、"),
    ]
    gens = {}
    print("\n--- v1.4 (90M) 4-Sys OFF vs 6-Sys IDF Semantic Wave ON Generations ---", flush=True)
    for tag, p in prompts:
        torch.manual_seed(42)
        t_off, ok_off, _ = brain.generate_semantic_utf8_v13b(p, max_new_bytes=180, rollout_k=1, use_semantic_exec=False)
        torch.manual_seed(42)
        t_on, ok_on, info_on = brain.generate_semantic_utf8_v13b(p, max_new_bytes=180, rollout_k=4, use_semantic_exec=True)
        gens[tag] = {"prompt": p, "off": {"text": t_off, "valid": ok_off}, "on": {"text": t_on, "valid": ok_on, "info": info_on}}
        print(f"  [{tag} | OFF valid={ok_off}] {t_off.replace(chr(10), ' ')}", flush=True)
        print(f"  [{tag} |  ON valid={ok_on} seeds={info_on['prompt_seeds'][:4]} wave={info_on['wave_lit'][:4]} hits={info_on['wave_hits_in_gen'][:4]}] {t_on.replace(chr(10), ' ')}", flush=True)

    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(f"\nTotal Elapsed: {time.time()-t0:.1f}s | Peak VRAM: {vram_mb:.1f} MB", flush=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "s0_baseline": ev0, "history": history, "sleep_logs": sleep_logs,
                "exec_fit": exec_fit, "lat_stats": lat_stats, "waves": waves_log,
                "ablation": {"sys4": ev_4sys, "sys5": ev_5sys, "sys6": ev_6sys, "easy_ja_306k": ev_easy_306k},
                "generations": gens, "peak_vram_mb": round(vram_mb, 1), "total_sec": round(time.time() - t0, 1),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
