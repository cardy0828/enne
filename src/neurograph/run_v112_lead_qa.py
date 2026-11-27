"""NeuroGraph v11.2 (Stage 2): Build Persistent Full-Wikipedia (1.67B chars!) Article-Lead Hippocampus
+ Strict 1-Factor QA Ablation on our SINGLE Merged v11.1 Brain!

Runs in ~2 minutes total (< 6 GB GPU VRAM, < 2 GB RAM for the persistent SQLite Lead Hippocampus!):
  1. Streams ALL 1.67B chars of `wikipedia.train` + 31M chars of `easy_ja.train` in 16M-char chunks
     -> builds `wiki_lead_v112/wiki_leads.sqlite` (persistent on disk, < 50 MB RAM at query time!).
  2. Loads ONE single `merged_v111_broca_core.pt` + `merged_v111_adapter.pkl`.
  3. Runs strict 1-factor A/B inside the EXACT SAME Single Merged Brain & Decoder:
       - `H0_Flat6gram`: Legacy flat 6-gram Hippocampus (30M wiki + 31M easy_ja)
       - `H1_WikiLead`:  NEW Persistent Full-Wikipedia Article-Lead Hippocampus!
  4. Evaluates ALL 6 open-world strict novel-fact QA tests + Greeting check + 2 Custom Zero-Shot `アストラX-99` QA!
"""

from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from neurograph.broca_cortex_v100 import BrocaSyntacticParserV100
from neurograph.engram_v20 import DecoupledBrainV20, EngramCartridge, EngramContextEncoder
from neurograph.episodic_v21 import EpisodicRack
from neurograph.instruct_cortex_v80 import InstructAdapterV80, _mem_snapshot
from neurograph.lateral_v13b import UnifiedBrainV13b
from neurograph.reasoning_graph_v90 import PrefrontalReasoningGraphV90
from neurograph.role_cortex_v60 import RoleBoundThoughtCortexV60
from neurograph.run_v14_scale90m import load_utf8_slice
from neurograph.speculative_decode_v31 import SentenceAlignedEpisodicCartridge
from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50
from neurograph.settings import DATA_DIR, require_gpu
from neurograph.wiki_lead_hippocampus_v112 import SingleMergedBrainV112, WikiLeadHippocampusV112


def load_brain(build_index=False):
    """統合コアと記憶をロードする。評価時だけ索引を再構築し旧検索器も用意する。"""
    import os
    require_gpu()
    device = torch.device("cuda")
    # 1) Mount Single Merged v11.1 Brain + L2 Assets (~5s, ~5.0 GB GPU VRAM!)
    t0 = time.perf_counter()
    with open("merged_v111_adapter.pkl", "rb") as f:
        ad111 = pickle.load(f)
    core = UnifiedBrainV13b.from_checkpoint("merged_v111_broca_core.pt", device)
    v20 = DecoupledBrainV20(core)
    v20.encoder = EngramContextEncoder(torch.from_numpy(ad111["E_base"]).to(device), device)
    for folder in ("engrams_v20/easy_ja", "engrams_v20/aozora", "engrams_v22/wikipedia_full_1b"):
        c = EngramCartridge.load(folder, mmap_mode="r")
        c.cfg.gate_min = 0.02; c.cfg.gate_alpha = 7.5; c.cfg.gate_tau = 0.28
        c.cfg.bank_scales = (0.08, 0.28, 0.72, 1.15, 1.50, 1.85)
        c.cfg.min_infer_w = (4, 3, 2, 2, 1, 1)
        v20.rack.mount(c)
    sft_eng = EngramCartridge.load("engrams_v80/instruct_sft", mmap_mode="r")
    mc = ContinuousThoughtCortexV50.load("macro_v70")
    rc = RoleBoundThoughtCortexV60(mc, num_roles=64, seed=60)
    rc.load_weights("macro_v70/role_hrr_v60.pkl")
    adapter_v80 = InstructAdapterV80.load("instruct_v80", engram_dir=None)
    rgraph = PrefrontalReasoningGraphV90.load("reasoning_v90", mc=mc)
    parser = BrocaSyntacticParserV100.load("broca_v100")
    print(f"[Mounted Single Merged v11.1 Core ({core.num_hidden_alive:,}N)] in {time.perf_counter()-t0:.2f}s | mem={_mem_snapshot(device)}", flush=True)

    # 2) Build Persistent Full-Wikipedia (1.67B chars!) Article-Lead Hippocampus (ZERO hardcoded language rules!)
    # Learn terminator & subject-head delimiters directly from our unsupervised models!
    db_path = "wiki_lead_v112/wiki_leads.sqlite"
    flat_hip = None
    lead_stat = {}
    if build_index:
        term_mid = max(
            ((m, p) for m, p in adapter_v80.eot_after_motor.items() if m >= 256 and len(mc.motor_bytes[m]) <= 4),
            key=lambda x: x[1],
            default=(-1, 0.0),
        )[0]
        term_str = mc.motor_bytes[term_mid].decode("utf-8", errors="ignore") if term_mid >= 256 else ""
        # Use unsupervised role markers discovered by RoleBoundThoughtCortexV60 as subject-head boundary delimiters!
        head_delims = []
        for mid in rc.role_motor_ids.tolist():
            s = mc.motor_bytes[int(mid)].decode("utf-8", errors="ignore")
            if 1 <= len(s) <= 2 and s not in head_delims:
                head_delims.append(s)
        print(f"\n=== 1. BUILDING PERSISTENT FULL-WIKIPEDIA ARTICLE-LEAD HIPPOCAMPUS (term='{term_str}', {len(head_delims)} learned delims) ===", flush=True)
        db_path = "wiki_lead_v112/wiki_leads.sqlite"
        lead_hip, lead_stat = WikiLeadHippocampusV112.build_from_utf32_corpus(
            corpus_paths=[
                ("wikipedia", str(DATA_DIR / "wikipedia.train"), 2.0),
                ("easy_ja", str(DATA_DIR / "easy_ja.train"), 1.2),
            ],
            db_path=db_path,
            head_delims=head_delims,
            term_str=term_str,
            chunk_chars=16_000_000,
        )
        print(f"  [Persistent WikiLead Index Built] {lead_stat} | mem={_mem_snapshot(device)}", flush=True)

        # Mount lightweight legacy flat 6-gram Hippocampus (50M wiki + 31M easy_ja) for fair 1-factor H0 baseline
        learned_bounds = [mc.motor_bytes[int(m)] for m in rc.role_motor_ids[:3]]
        flat_hip = EpisodicRack()
        for dname, nch, bkts, prio in (("wikipedia", 50_000_000, 1 << 20, 1.3), ("easy_ja", 31_000_000, 1 << 18, 0.95)):
            c_sl = load_utf8_slice(str(DATA_DIR), dname, start_char=0, train_chars=nch, val_chars=1000)
            flat_hip.mount(SentenceAlignedEpisodicCartridge(
                name=dname, raw_bytes=c_sl.train_bytes, target_ep_len=280,
                num_buckets=bkts, max_df_ratio=0.004, stride=3, priority_boost=prio, boundary_patterns=learned_bounds,
            ))
            del c_sl
        gc.collect()

    else:
        if Path("wiki_lead_v112/v124.sqlite").is_file():
            db_path = "wiki_lead_v112/v124.sqlite"
        if not Path(db_path).is_file():
            raise FileNotFoundError(db_path)
        lead_hip = WikiLeadHippocampusV112(db_path)
    brain = SingleMergedBrainV112(
        v20_merged=v20,
        W_res=ad111["W_res"],
        b_res=ad111["b_res"],
        readout_scale=ad111["readout_scale"],
        sft_engram=sft_eng,
        mc=mc,
        rc=rc,
        adapter_v80=adapter_v80,
        rgraph=rgraph,
        parser=parser,
        lead_hip=lead_hip,
        flat_hip=flat_hip,
    )

    return brain, lead_stat


def main() -> None:
    """同一デコーダで検索器を切り替え、回答と計測値を保存する。"""
    t_all = time.perf_counter()
    brain, lead_stat = load_brain(build_index=True)
    device = brain.device
    lead_hip = brain.lead_hip
    # 3) Strict 1-Factor QA Ablation (H0 Flat 6-gram vs H1 Full-Wiki Lead Hippocampus!)
    test_suite = [
        ("挨拶", "こんにちは！調子はどうですか？", []),
        ("用語説明(鉄道)", "東海道新幹線とは何ですか？簡単に教えてください。", ["東京", "新大阪", "高速", "JR"]),
        ("科学説明(物理)", "量子力学と一般相対性理論の違いをわかりやすく説明してください。", ["アインシュタイン", "重力", "時空", "確率"]),
        ("歴史質問", "第二次世界大戦はいつ始まり、いつ終わりましたか？", ["1939", "1945"]),
        ("文学質問", "夏目漱石の『吾輩は猫である』はどのような小説ですか？", ["明治", "長編", "風刺", "教師"]),
        ("日常質問", "日本の首都はどこですか？", ["東京"]),
    ]
    for _, q, gks in test_suite:
        for kw in gks:
            assert kw not in q

    print("\n=== 2. STRICT 1-FACTOR QA ABLATION ON SINGLE MERGED v11.1 BRAIN (H0 Flat vs H1 WikiLead) ===", flush=True)
    ab = {}
    totals = {
        "H0_Flat6gram": {"ev_h": 0, "an_h": 0, "gt": 0, "eot": 0, "bpcs": [], "ms": []},
        "H1_WikiLead":  {"ev_h": 0, "an_h": 0, "gt": 0, "eot": 0, "bpcs": [], "ms": []},
    }
    for tag, q, gks in test_suite:
        ab[tag] = {"q": q, "gold": gks}
        print(f"\n  [{tag}] Q: {q} (novel_gold={gks})", flush=True)
        for cname, use_l in (("H0_Flat6gram", False), ("H1_WikiLead", True)):
            ans, ok, inf = brain.respond(q, use_lead_hip=use_l, use_memory_gate=False)
            assert ok, f"Invalid UTF-8 on {tag} {cname}"
            ev_txt = " ".join(inf["agenda_preview"])
            ev_h = sum(1 for k in gks if k in ev_txt) if gks else 0
            an_h = sum(1 for k in gks if k in ans) if gks else 0
            if gks:
                totals[cname]["ev_h"] += ev_h; totals[cname]["an_h"] += an_h; totals[cname]["gt"] += len(gks)
            totals[cname]["eot"] += int(inf["stopped_eot"])
            totals[cname]["bpcs"].append(inf["fluency_bpc"])
            totals[cname]["ms"].append(inf["hop23_ms"])
            ab[tag][cname] = {"ans": ans, "ev_h": ev_h, "an_h": an_h, "info": inf}
            print(
                f"    [{cname:12s}] mode={inf['intent_mode']:11s} hip={str(inf['want_hip']):5s} ({inf['hop23_ms']:4.1f}ms) "
                f"ev={ev_h}/{len(gks)} ans={an_h}/{len(gks)} BPC={inf['fluency_bpc']:.2f} eot={inf['stopped_eot']} ({inf['chars_per_sec']:.1f}ch/s) >> {ans}",
                flush=True,
            )
            if use_l and inf["agenda_preview"]:
                print(f"      [Lead Evid] ents={inf['entities']} heads={inf['agenda_heads']} | {inf['agenda_preview']}", flush=True)

    print("\n=== 3. 1-FACTOR HIPPOCAMPUS ABLATION SUMMARY ===", flush=True)
    for cname, st in totals.items():
        gt = max(1, st["gt"])
        print(
            f"  [{cname:12s}] Evid-Recall={st['ev_h']}/{gt} ({100*st['ev_h']/gt:5.1f}%) | "
            f"Ans-Novel-Recall={st['an_h']}/{gt} ({100*st['an_h']/gt:5.1f}%) | "
            f"EOT={st['eot']}/{len(test_suite)} | MeanBPC={np.mean(st['bpcs']):.3f} | MeanQuery={np.mean(st['ms']):.1f}ms",
            flush=True,
        )

    # 4) Hot-Mounted Custom Zero-Shot Fact QA (アストラX-99)
    fact_txt = (
        "量子重力エンジン「アストラX-99」は2042年に日本のつくば研究所で開発された次世代推進装置である。\n"
        "アストラX-99の最大推力は従来のイオンエンジンの約400倍であり、火星までの航行時間をわずか18日間に短縮できる。\n"
    )
    lead_hip.mount_custom("astra", fact_txt)
    custom_out = {}
    for ctag, cp, gk in (
        ("Custom Direct QA", "量子重力エンジン「アストラX-99」の特徴を教えてください。", ["2042", "つくば", "推進"]),
        ("Custom Indirect QA", "アストラX-99の最大推力はどのくらいですか？", ["400", "イオン", "18"]),
    ):
        a, ok, inf = brain.respond(cp, use_lead_hip=True, use_memory_gate=False)
        assert ok
        ev_h = sum(1 for k in gk if k in " ".join(inf["agenda_preview"]))
        an_h = sum(1 for k in gk if k in a)
        custom_out[ctag] = {"q": cp, "ev_h": ev_h, "an_h": an_h, "ans": a, "info": inf}
        print(f"\n  [{ctag}] ents={inf['entities']} | evid={ev_h}/{len(gk)} ans={an_h}/{len(gk)} BPC={inf['fluency_bpc']:.2f} eot={inf['stopped_eot']} ({inf['chars_per_sec']:.1f}ch/s) >> {a}", flush=True)
    lead_hip.unmount_custom("astra")
    lead_hip.close()

    out = {
        "lead_index_stats": lead_stat,
        "ablation_summary": {k: {**v, "mean_bpc": round(float(np.mean(v["bpcs"])), 3), "mean_ms": round(float(np.mean(v["ms"])), 2)} for k, v in totals.items()},
        "ablation_details": ab,
        "custom": custom_out,
        "final_mem": _mem_snapshot(device),
        "peak_cuda_alloc_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "total_sec": round(time.perf_counter() - t_all, 1),
    }
    Path("results_v112_lead_qa.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== v11.2 STAGE 2 COMPLETE in {out['total_sec']}s | final_mem={out['final_mem']} ===", flush=True)


if __name__ == "__main__":
    main()
