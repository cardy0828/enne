"""Run NeuroGraph v1.3: Unified 6-System Raw UTF-8 Living Brain (D=256, S=16 Slots, 3M bf16 DAG,
33.6M 8-Bank LTM, 1-Shot STM, 24-Mode Executive, 2-Hop Lateral Semantic Thought Wave)
trained on a FULLY SHUFFLED 3-Domain Balance (10M easy_ja + 10M aozora + 10M wikipedia = 30M chars / ~86M raw bytes)!"""

from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurograph.lateral_brain_v13 import UnifiedBrainV13, UnifiedBrainV13Config
from neurograph.utf8_dag_v88 import load_utf8_from_utf32


def build_shuffled_multidomain_bytes(
    c_easy: np.ndarray,
    c_aozora: np.ndarray,
    c_wiki: np.ndarray,
    block_bytes: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    """Slice all 3 domains into 4 KB passage blocks and shuffle them completely -> ZERO recency domain drift!"""
    blocks = []
    for arr in (c_easy, c_aozora, c_wiki):
        n_b = len(arr) // block_bytes
        blocks.append(arr[: n_b * block_bytes].reshape(n_b, block_bytes))
    all_b = np.concatenate(blocks, axis=0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(all_b.shape[0])
    return all_b[perm].reshape(-1).copy()


def eval_3domains(brain: UnifiedBrainV13, vals: dict, val_streams: int, use_exec: bool, use_lat: bool) -> dict:
    out = {}
    for k, corp in vals.items():
        out[k] = brain.eval_with_lateral_wave(
            corp.val_bytes, batch_streams=val_streams, bytes_per_char=corp.bytes_per_char_val,
            use_executive=use_exec, use_lateral=use_lat,
        )
    return out


def fmt_3domains(m: dict) -> str:
    avg_bpc = sum(m[k]["bpc_lossless"] for k in ("easy_ja", "aozora", "wikipedia")) / 3.0
    parts = [f"{k}={m[k]['bpc_lossless']:.4f}({m[k]['byte_top1']*100:4.1f}%)" for k in ("easy_ja", "aozora", "wikipedia")]
    return " | ".join(parts) + f" | MEAN={avg_bpc:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.environ.get("ENNE_DATA_DIR", "data"))
    parser.add_argument("--chars-per-domain", type=int, default=10_000_000)
    parser.add_argument("--val-chars", type=int, default=120_000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-streams", type=int, default=3072)
    parser.add_argument("--val-streams", type=int, default=128)
    parser.add_argument("--out", type=str, default="results_v13_unified_6sys.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    c_easy = load_utf8_from_utf32(args.data_dir, "easy_ja", train_chars=args.chars_per_domain, val_chars=args.val_chars)
    c_aozora = load_utf8_from_utf32(args.data_dir, "aozora", train_chars=args.chars_per_domain, val_chars=args.val_chars)
    c_wiki = load_utf8_from_utf32(args.data_dir, "wikipedia", train_chars=args.chars_per_domain, val_chars=args.val_chars)
    vals = {"easy_ja": c_easy, "aozora": c_aozora, "wikipedia": c_wiki}

    mixed_train = build_shuffled_multidomain_bytes(c_easy.train_bytes, c_aozora.train_bytes, c_wiki.train_bytes, block_bytes=4096, seed=42)
    avg_bpc_tr = (c_easy.bytes_per_char_train + c_aozora.bytes_per_char_train + c_wiki.bytes_per_char_train) / 3.0
    print(
        f"Built SHUFFLED 3-Domain Raw UTF-8 Stream (ZERO <UNK>, V=256): {len(mixed_train):,}B "
        f"({args.chars_per_domain*3/1e6:.1f}M chars across easy_ja + aozora + wikipedia, 4KB shuffled blocks!)",
        flush=True,
    )

    torch.manual_seed(0)
    t0 = time.time()
    brain = UnifiedBrainV13(UnifiedBrainV13Config(), device)
    brain.init_sensory_basis(mixed_train)
    print(
        f"Born from ZERO hidden neurons! Specs: D={brain.D}, S={brain.cfg.num_slots} slots, "
        f"Pool={brain.P:,} (bf16), LTM={sum(b*4 for b in brain.ltm.bank_buckets):,} (8 banks), ByteSTM={brain.cfg.stm_slots}",
        flush=True,
    )

    seg_len = len(mixed_train) // args.rounds
    history = []

    for r in range(args.rounds):
        rt = time.time()
        seg = mixed_train[r * seg_len : (r + 1) * seg_len]
        lr_s = 1.0 / (1.0 + 0.20 * r)
        tr = brain.run_stream(
            seg, train=True, batch_streams=args.batch_streams, lr_scale=lr_s,
            bytes_per_char=avg_bpc_tr, use_slots=True, use_ltm=False, use_stm=False, use_executive=False, populate_ltm=True,
        )
        ap = brain.apoptosis()
        fu = brain.fuse_similar_neurons(seed=200 + r * 17)
        ev_r = eval_3domains(brain, vals, args.val_streams, use_exec=False, use_lat=False)
        dt = time.time() - rt
        row = {
            "round": r + 1, "sec": round(dt, 1),
            "births": tr["births"], "deaths": ap["died"], "merged": fu["merged"], "divorces": tr["divorces"],
            "alive_hidden": fu["alive_hidden"], "edges": fu["edges"], "ltm_active": brain.ltm.count_active(),
            "domains": ev_r,
        }
        history.append(row)
        print(
            f"[round {r+1}/{args.rounds}] {dt:5.1f}s | {fu['alive_hidden']:,}N / {fu['edges']:,}E | LTM={brain.ltm.count_active()/1e6:.2f}M | "
            f"{fmt_3domains(ev_r)}",
            flush=True,
        )

    # Sleep Pruning on 8-bank MegaByteLTM
    sleep_stats = brain.ltm.sleep_prune()
    print(f"\n[Sleep Pruning] Pruned {sleep_stats['pruned']:,}/{sleep_stats['before']:,} ({sleep_stats['prune_pct']}%) -> {sleep_stats['after']/1e6:.2f}M LTM!", flush=True)

    # Fit System 5 (24-Mode Unsupervised Executive Critic)
    rt = time.time()
    exec_fit = brain.fit_executive_modes(mixed_train[:6_000_000], passage_len=320, max_passages=12_000)
    print(f"[System 5 Executive] Fitted {exec_fit['num_modes']} discourse modes in {time.time()-rt:.1f}s", flush=True)

    # Build System 6 (Lateral Multi-Byte Semantic Association Graph — ZERO tokenizer!)
    rt = time.time()
    lat_stats = brain.build_lateral_semantic_graph(mixed_train, scan_bytes=16_000_000, batch_streams=2048)
    print(
        f"[System 6 Lateral Semantic Graph] Extracted {lat_stats['num_concepts']:,} emergent multi-byte concepts, "
        f"{lat_stats['num_edges']:,} PPMI semantic synapses in {time.time()-rt:.1f}s!",
        flush=True,
    )

    # Inspect live 2-hop Lateral Semantic Thought Waves!
    probes = ["世界大戦", "東京", "ドイツ", "日本", "猫", "学校"]
    waves_log = {}
    print("\n--- Live 2-Hop Lateral Semantic Thought Waves (Zero Tokenizer, Pure Raw UTF-8!) ---", flush=True)
    for pr in probes:
        w_res = brain.trace_semantic_wave(pr, top_k=8)
        waves_log[pr] = w_res
        w_str = ", ".join(f"{t}({s:.2f})" for t, s in w_res)
        print(f"  [{pr}] ==wave==> {w_str}", flush=True)

    # Clean 3-Stage Causal Ablation across ALL 3 Domains!
    ev_4sys = eval_3domains(brain, vals, args.val_streams, use_exec=False, use_lat=False)
    ev_5sys = eval_3domains(brain, vals, args.val_streams, use_exec=True, use_lat=False)
    ev_6sys = eval_3domains(brain, vals, args.val_streams, use_exec=True, use_lat=True)

    print("\n================ NEUROGRAPH v1.3 UNIFIED 6-SYSTEM RESULTS (SHUFFLED 3-DOMAIN, ZERO <UNK>) ================", flush=True)
    print(f"v1.2 Ref (sequential 52M)   | easy_ja=4.3983(70.6%) | aozora=5.3076(66.5%) | wikipedia=4.9649(66.5%) | MEAN=4.8903", flush=True)
    print(f"v1.3 4-Sys (DAG+16Sl+LTM+STM)| {fmt_3domains(ev_4sys)}", flush=True)
    print(f"v1.3 5-Sys (+24-Mode Exec)  | {fmt_3domains(ev_5sys)}", flush=True)
    print(f"v1.3 6-Sys (+Lateral Wave!) | {fmt_3domains(ev_6sys)}", flush=True)
    lat_hits_str = " | ".join(f"{k}_lat_hit={ev_6sys[k]['lat_hit_rate']*100:.1f}%" for k in ("easy_ja", "aozora", "wikipedia"))
    print(f"  Proactive Lateral Semantic Priming Hits: {lat_hits_str}", flush=True)

    # Save v1.3 Unified 6-System Checkpoint!
    torch.save(
        {
            "cfg": brain.cfg.__dict__,
            "alive": brain.alive.cpu(), "parent": brain.parent.cpu(), "cond_kind": brain.cond_kind.cpu(),
            "cond_id": brain.cond_id.cpu(), "depth": brain.depth.cpu(), "local_span": brain.local_span.cpu(),
            "fires": brain.fires.cpu(), "upd_cnt": brain.upd_cnt.cpu(),
            "E": brain.E.cpu(), "bias": brain.bias.cpu(),
            "W_node": brain.W_node.cpu(), "sharp_dst": brain.sharp_dst.cpu(), "sharp_w": brain.sharp_w.cpu(),
            "G_write": brain.G_write.cpu(), "G_keep": brain.G_keep.cpu(),
            "write_bias": brain.write_bias.cpu(), "keep_bias": brain.keep_bias.cpu(),
            "W_read": brain.W_read.cpu(), "W_mod": brain.W_mod.cpu(),
            "mode_centroids": brain.mode_centroids.cpu(), "mode_llr": brain.mode_llr.cpu(),
            "ltm_sig": [x.cpu() for x in brain.ltm.sig],
            "ltm_dst": [x.cpu() for x in brain.ltm.dst],
            "ltm_w": [x.cpu() for x in brain.ltm.w],
            "sem_to_node": brain.sem_to_node, "sem_texts": brain.sem_texts,
            "lat_dst": brain.lat_dst.cpu(), "lat_w": brain.lat_w.cpu(),
            "sem_prime_k3": brain.sem_prime_k3.cpu(), "sem_prime_y": brain.sem_prime_y.cpu(),
        },
        "brain_v13_unified_6sys.pt",
    )
    print("Saved 6-system brain: brain_v13_unified_6sys.pt", flush=True)

    # Semantic Wave + Executive Rollout A/B Generations!
    prompts = [
        ("Wikipedia歴史", "第二次世界大戦（だいにじせかいたいせん）は、1939年から1945年にかけて"),
        ("青空文庫文学", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"),
        ("東京・都市", "日本の東京は、江戸時代から続く歴史ある都市であり、"),
        ("無限拡張🚀", "薔薇と憂鬱と🚀について、"),
    ]
    gens = {}
    print("\n--- 6-System Lateral Semantic Wave + Executive Rollout A/B Generations ---", flush=True)
    for tag, p in prompts:
        torch.manual_seed(42)
        t_off, ok_off, _ = brain.generate_semantic_utf8(p, max_new_bytes=180, rollout_k=1, use_semantic_exec=False)
        torch.manual_seed(42)
        t_on, ok_on, info_on = brain.generate_semantic_utf8(p, max_new_bytes=180, rollout_k=4, use_semantic_exec=True)
        gens[tag] = {"prompt": p, "off": {"text": t_off, "valid": ok_off}, "on": {"text": t_on, "valid": ok_on, "info": info_on}}
        print(f"  [{tag} | 4-Sys OFF valid={ok_off}] {t_off.replace(chr(10), ' ')}", flush=True)
        print(f"  [{tag} | 6-Sys  ON valid={ok_on} wave={info_on['wave_lit'][:4]} hits={info_on['wave_hits_in_gen'][:4]}] {t_on.replace(chr(10), ' ')}", flush=True)

    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(f"\nTotal Elapsed: {time.time()-t0:.1f}s | Peak VRAM: {vram_mb:.1f} MB", flush=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "history": history, "sleep_prune": sleep_stats, "exec_fit": exec_fit, "lat_stats": lat_stats,
                "waves": waves_log, "ablation": {"sys4": ev_4sys, "sys5": ev_5sys, "sys6": ev_6sys},
                "generations": gens, "peak_vram_mb": round(vram_mb, 1), "total_sec": round(time.time() - t0, 1),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
