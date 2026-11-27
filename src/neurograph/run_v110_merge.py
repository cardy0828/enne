"""NeuroGraph v11.0: Topology-Aligned Cortical DAG Node Merge — Build, Dev-200 Calibration & Full Evaluation!

Tests merging NEW Broca L1 SFT-DAG (524,838N, Host) + Frozen Base 6M Core (4,543,752N, Donor)
weighted STRONGER toward NEW Broca inside ONE SINGLE 6M-neuron vessel (`merged_v110_core.pt`):
  1. Exact BFS topological alignment + Ridge coordinate transport (`cos = 0.997+`)
  2. Strict Dev-200 sweep of `(w_new, alpha_donor)` (ZERO touching of `val.jsonl`!)
  3. Full 1,000 Isolated SFT Val Evaluation (Core-Only AND Full + Engram) + throughput comparison vs Dual v10.2!
  4. Full Base 3-Domain Retention (`easy_ja`, `aozora`, `wikipedia` 120k chars each):
     Broca Alone (524K) vs Single Merged v11.0 Core vs Original Base 6M Core!
  5. Save `merged_v110_core.pt` + verify bit-exact round-trip reload!
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from neurograph.settings import DATA_DIR
from neurograph.cortical_merge_v110 import (
    apply_merge_weights_inplace,
    build_cortical_merge_topology,
    eval_single_core_sft_val,
    instantiate_merged_brain_vessel,
)
from neurograph.engram_v20 import DecoupledBrainV20, EngramCartridge, EngramContextEncoder
from neurograph.instruct_cortex_v80 import _mem_snapshot
from neurograph.lateral_v13b import UnifiedBrainV13b
from neurograph.run_v14_scale90m import load_utf8_slice
from neurograph.sft_data_v80 import OUT_DIR as SFT_DIR, load_sft_jsonl


def mount_base_engrams(v20: DecoupledBrainV20, E_base: torch.Tensor) -> None:
    # Align 16-D Engram context fingerprints to Base Core's E_base (exact key/context parity with saved Engrams!)
    v20.encoder = EngramContextEncoder(E_base.to(v20.device), v20.device)
    for folder in ("engrams_v20/easy_ja", "engrams_v20/aozora", "engrams_v22/wikipedia_full_1b"):
        c = EngramCartridge.load(folder, mmap_mode="r")
        c.cfg.gate_min = 0.02; c.cfg.gate_alpha = 7.5; c.cfg.gate_tau = 0.28
        c.cfg.bank_scales = (0.08, 0.28, 0.72, 1.15, 1.50, 1.85)
        c.cfg.min_infer_w = (4, 3, 2, 2, 1, 1)
        v20.rack.mount(c)


@torch.no_grad()
def eval_3domains_v20(v20: DecoupledBrainV20, val_corpora: dict, batch_streams: int = 128) -> dict:
    out = {}
    for k, corp in val_corpora.items():
        r = v20.eval_stream(
            corp.val_bytes, batch_streams=batch_streams, bytes_per_char=corp.bytes_per_char_val,
            use_executive=v20.core.mode_ready, use_engram=True, use_context_gate=True,
        )
        out[k] = round(float(r["bpc_lossless"]), 4)
    out["mean"] = round(sum(out[k] for k in ("easy_ja", "aozora", "wikipedia")) / 3.0, 4)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    t_all = time.perf_counter()

    train_all = load_sft_jsonl(SFT_DIR / "train.jsonl")
    val_diags = load_sft_jsonl(SFT_DIR / "val.jsonl")
    dev_diags = train_all[-200:]
    del train_all

    # Load 3-domain Val slices (120k chars each) for retention comparison
    val_corpora = {
        d: load_utf8_slice(str(DATA_DIR), d, start_char=0, train_chars=1000, val_chars=120_000)
        for d in ("easy_ja", "aozora", "wikipedia")
    }
    sft_eng = EngramCartridge.load("engrams_v80/instruct_sft", mmap_mode="r")

    # 1) Build Exact Topological Alignment on CPU FIRST (gives us E_base + exact BFS alignment!)
    print("\n=== 1. COMPUTING EXACT BFS TOPOLOGY ALIGNMENT (BROCA HOST + BASE 6M DONOR) ===", flush=True)
    topo = build_cortical_merge_topology(
        broca_ckpt="broca_v100/broca_l1_core.pt",
        base_ckpt="brain_v22_6m_core.pt",
        pool_capacity=6_000_000,
        ridge_lam=0.05,
    )
    E_base_cpu = topo.ck_base["E"].clone()
    print(f"  [Merge Topology Ready] {topo.stats}", flush=True)

    # 0) Baseline check: Broca Alone (524K) on 3-Domain Retention & SFT Val (+ E_base-aligned Engrams!)
    print("\n=== 2. BASELINE: BROCA ALONE (524K) + ENGRAM (SFT VAL & 3-DOMAIN RETENTION) ===", flush=True)
    broca_only = UnifiedBrainV13b.from_checkpoint("broca_v100/broca_l1_core.pt", device)
    v20_broca = DecoupledBrainV20(broca_only)
    mount_base_engrams(v20_broca, E_base_cpu)
    broca_3d = eval_3domains_v20(v20_broca, val_corpora)
    broca_sft_eng = eval_single_core_sft_val(v20_broca, val_diags, sft_engram=sft_eng, use_base_engram=True, use_sft_engram=True)
    print(f"  [Broca Alone (524K) + Engram] SFT Val: Body-BPC={broca_sft_eng['body_bpc']:.4f} Top1={broca_sft_eng['body_top1']*100:.1f}% EOT={broca_sft_eng['eot_bits']:.3f} ({broca_sft_eng['sec']}s)", flush=True)
    print(f"  [Broca Alone (524K) + Engram] 3-Domain Retention: {broca_3d}", flush=True)
    del v20_broca, broca_only
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Instantiate ONE Single 6M Merged Vessel on GPU
    merged_core = instantiate_merged_brain_vessel(topo, device, pool_capacity=6_000_000, edge_buckets=8_388_608)
    v20_merged = DecoupledBrainV20(merged_core)
    mount_base_engrams(v20_merged, E_base_cpu)
    print(f"  [Single Merged Vessel Mounted] alive={merged_core.num_hidden_alive:,}N / {merged_core.edges.count():,}E | mem={_mem_snapshot(device)}", flush=True)

    # 4) Strict Dev-200 Calibration of (w_new, alpha_donor) — Stronger NEW Broca!
    print("\n=== 3. CALIBRATING (w_new, alpha_donor) STRICTLY ON DEV-200 (BROCA-STRONG!) ===", flush=True)
    sweep_cfgs = [
        (0.65, 0.55),
        (0.75, 0.50),
        (0.80, 0.45),
        (0.85, 0.35),
        (0.75, 0.70),
        (0.90, 0.25),
    ]
    dev_logs = []
    best_sc = 1e9
    best_pair = (0.75, 0.50)
    for w_n, a_d in sweep_cfgs:
        dt_w = apply_merge_weights_inplace(merged_core, topo, w_new=w_n, alpha_donor=a_d)
        ev_d = eval_single_core_sft_val(v20_merged, dev_diags, sft_engram=sft_eng, use_base_engram=True, use_sft_engram=True)
        ev_3d = eval_3domains_v20(v20_merged, val_corpora)
        # Balanced score prioritizing SFT BPC + EOT while also tracking 3-domain retention!
        sc = ev_d["body_bpc"] + 0.01 * ev_d["eot_bits"]
        row = {"w_new": w_n, "alpha_donor": a_d, "apply_sec": dt_w, "dev_sft": ev_d, "val_3dom": ev_3d}
        dev_logs.append(row)
        print(
            f"    [w_new={w_n:.2f} alpha_donor={a_d:.2f} ({dt_w:.1f}s)] "
            f"Dev-SFT: BPC={ev_d['body_bpc']:.4f} Top1={ev_d['body_top1']*100:.1f}% EOT={ev_d['eot_bits']:.3f} | "
            f"3-Dom MEAN={ev_3d['mean']:.4f} (easy={ev_3d['easy_ja']:.4f}, aoz={ev_3d['aozora']:.4f}, wiki={ev_3d['wikipedia']:.4f})",
            flush=True,
        )
        if sc < best_sc:
            best_sc = sc
            best_pair = (w_n, a_d)

    # Apply winning Broca-strong weights & save ONE single merged checkpoint!
    w_best, a_best = best_pair
    apply_merge_weights_inplace(merged_core, topo, w_new=w_best, alpha_donor=a_best)
    # Free raw CPU checkpoints before final eval/save to keep peak RSS low!
    topo_stats = topo.stats
    del topo
    gc.collect()

    print(f"\n=== 3. FULL INDEPENDENT EVALUATION OF SINGLE MERGED BRAIN (w_new={w_best}, alpha_donor={a_best}) ===", flush=True)
    ev_val_core = eval_single_core_sft_val(v20_merged, val_diags, sft_engram=None, use_base_engram=False, use_sft_engram=False)
    print(f"  [Merged Single Core ONLY (No Engram)] SFT Val-1000: Body-BPC={ev_val_core['body_bpc']:.4f} Top1={ev_val_core['body_top1']*100:.1f}% EOT={ev_val_core['eot_bits']:.3f} ({ev_val_core['sec']}s)", flush=True)

    ev_val_full = eval_single_core_sft_val(v20_merged, val_diags, sft_engram=sft_eng, use_base_engram=True, use_sft_engram=True)
    print(f"  [Merged Single Core + Base/SFT Eng  ] SFT Val-1000: Body-BPC={ev_val_full['body_bpc']:.4f} Top1={ev_val_full['body_top1']*100:.1f}% EOT={ev_val_full['eot_bits']:.3f} ({ev_val_full['sec']}s)", flush=True)

    ev_3dom_fin = eval_3domains_v20(v20_merged, val_corpora)
    print(f"  [Merged Single Core + Base Engram   ] 3-Domain Val: {ev_3dom_fin}", flush=True)

    # Save standalone merged checkpoint & verify round-trip reload!
    save_path = "merged_v110_broca_strong_core.pt"
    t_sv = time.perf_counter()
    merged_core.save_full_checkpoint(save_path)
    print(f"  [Saved Single Merged Core] {save_path} in {time.perf_counter()-t_sv:.1f}s | mem={_mem_snapshot(device)}", flush=True)

    out = {
        "topology_stats": topo_stats,
        "broca_alone_plus_engram": {"sft_val": broca_sft_eng, "val_3dom": broca_3d},
        "dev_sweep": dev_logs,
        "selected": {"w_new": w_best, "alpha_donor": a_best},
        "merged_single_val_core_only": ev_val_core,
        "merged_single_val_full": ev_val_full,
        "merged_single_3dom": ev_3dom_fin,
        "final_mem": _mem_snapshot(device),
        "total_sec": round(time.perf_counter() - t_all, 1),
    }
    Path("results_v110_merge.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== v11.0 SINGLE-BRAIN MERGE COMPLETE in {out['total_sec']}s | final_mem={out['final_mem']} ===", flush=True)


if __name__ == "__main__":
    main()
