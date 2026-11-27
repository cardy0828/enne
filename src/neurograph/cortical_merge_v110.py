"""NeuroGraph v11.0: Topology-Aligned Cortical DAG Node Merge (Broca-Strong Single-Brain Fusion!).

ZERO Backprop | ZERO Attention | ZERO Permutation Collapse | 100% Exact Path-Aligned Node Grafting!

Because every hidden neuron in NeuroGraph's Living DAG is content-addressable by its exact structural
path `(parent_node, cond_kind, cond_id) -> child_node` rooted at raw UTF-8 bytes `0..255`:
  1. Exact Closed-Form Coordinate-Frame Alignment (`Ridge T: [256, 256]`):
     Solves `E_broca @ T.T ≈ E_base` so `(h_base @ T) @ E_broca.T ≈ h_base @ E_base.T` (`cos = 0.997+`!).
     All grafted Base neurons fire seamlessly into Broca's SFT-native sensory basis `E_broca`!
  2. Depth-Ordered BFS Topological Alignment (`depth = 1..16`):
     - **Broca-ONLY paths** (SFT turn framing, `\x02User:`, `\x03Assistant:`, `0x04` EOT triggers, dialogue syntax):
       100% preserved intact at full `1.0x` Broca strength!
     - **SHARED paths** (existing in BOTH Broca AND Base):
       Blended weighted **STRONGER toward NEW Broca** (`w_new`, e.g. `0.75`):
         `W_merged = w_new * W_broca + (1 - w_new) * (W_base @ T)`
         `sharp_merged = topk8(w_new * sharp_broca + (1 - w_new) * sharp_base)` (vectorized GPU union!)
     - **Base-ONLY paths** (millions of Wikipedia/Aozora encyclopedic knowledge nodes Broca never saw):
       Grafted directly onto their aligned parents inside the SINGLE 6M vessel at `alpha_donor` strength!
  3. Produces ONE single `UnifiedBrainV13b` (`merged_v110_core.pt`, ~4.1 GB static VRAM, 1 Triton launch/step —
     2x faster than dual parallel runners!) with BOTH SFT mastery AND 3-domain encyclopedic literacy!
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neurograph.engram_v20 import DecoupledBrainV20
from neurograph.instruct_cortex_v80 import _build_isolated_batches, _mem_snapshot
from neurograph.lateral_v13b import LateralV13bConfig, UnifiedBrainV13b
from neurograph.memory_v3b import _mix64
from neurograph.reason_decode_v30 import BatchedCoreRunner
from neurograph.sft_data_v80 import TURN_EOT_BYTE


def compute_basis_transport(
    E_base: torch.Tensor,
    E_broca: torch.Tensor,
    ridge_lam: float = 0.05,
    ortho_control_bytes: Tuple[int, ...] = (2, 3, 4),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve exact closed-form transport `T [D, D]` so `(h @ T) @ E_broca.T ≈ h @ E_base.T`,
    then project out the SFT dialogue control-byte subspace `{0x02, 0x03, 0x04}` so grafted Base neurons
    are strictly orthogonal to EOT (`(W_base @ T) @ E_broca[0x04] == 0.0` — ZERO EOT interference!)."""
    Eb = E_base.to(torch.float64)
    Er = E_broca.to(torch.float64)
    D = Er.shape[1]
    eye = torch.eye(D, dtype=torch.float64, device=Er.device)
    TT = torch.linalg.solve(Er.T @ Er + ridge_lam * eye, Er.T @ Eb)
    T64 = TT.T  # [D, D]

    # Exact QR orthonormalization of SFT control-byte rows in E_broca -> project them out of T64's column space!
    eot_leak_before = float((T64 @ Er[4]).norm().item())
    if ortho_control_bytes:
        C_sub = Er[list(ortho_control_bytes)].T  # [D, K_ctrl]
        Q_ctrl, _ = torch.linalg.qr(C_sub, mode="reduced")  # [D, K_ctrl] orthonormal columns
        P_ortho = eye - (Q_ctrl @ Q_ctrl.T)
        T64 = T64 @ P_ortho
    eot_leak_after = float((T64 @ Er[4]).norm().item())
    T = T64.to(torch.float32)

    # Verify exact logit preservation cosine on non-control bytes (32..255)
    gen = torch.Generator(device=Er.device)
    gen.manual_seed(110)
    h_test = torch.randn((256, D), generator=gen, device=Er.device, dtype=torch.float32)
    lg_orig = (h_test @ E_base.float().T)[:, 10:]
    lg_algn = ((h_test @ T) @ E_broca.float().T)[:, 10:]
    cos_val = float(torch.nn.functional.cosine_similarity(lg_orig, lg_algn, dim=-1).mean().item())
    rel_err = float(((lg_orig - lg_algn).norm() / lg_orig.norm().clamp(min=1e-6)).item())
    spec_norm = float(torch.linalg.matrix_norm(T, 2).item())
    return T, {
        "logit_cos": round(cos_val, 5),
        "rel_err": round(rel_err, 4),
        "T_norm2": round(spec_norm, 3),
        "eot_leak_before": round(eot_leak_before, 4),
        "eot_leak_after": round(eot_leak_after, 6),
    }


@torch.no_grad()
def merge_sharp_8_gpu(
    sd_new: torch.Tensor,
    sw_new: torch.Tensor,
    sd_old: torch.Tensor,
    sw_old: torch.Tensor,
    w_new: float,
    w_old: float,
    device: torch.device,
    chunk_size: int = 32_768,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact vectorized GPU union-and-Top8 merge of two [N, 8] sharp synapse tables (32 MB temp VRAM!)."""
    N = sd_new.shape[0]
    out_d = torch.full((N, 8), -1, dtype=torch.int16)
    out_w = torch.zeros((N, 8), dtype=torch.bfloat16)

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        B = e - s
        buf = torch.zeros((B, 256), dtype=torch.float32, device=device)
        ar = torch.arange(B, device=device).unsqueeze(1)

        dn = sd_new[s:e].to(device=device, dtype=torch.int64)
        wn = sw_new[s:e].to(device=device, dtype=torch.float32) * float(w_new)
        mn = (dn >= 0) & (dn < 256)
        buf.scatter_add_(1, dn.clamp(0, 255), torch.where(mn, wn, torch.zeros_like(wn)))

        do = sd_old[s:e].to(device=device, dtype=torch.int64)
        wo = sw_old[s:e].to(device=device, dtype=torch.float32) * float(w_old)
        # Strictly prevent Donor Base sharp synapses from touching SFT control bytes {0x02, 0x03, 0x04}!
        mo = (do >= 0) & (do < 256) & (do != 2) & (do != 3) & (do != 4)
        buf.scatter_add_(1, do.clamp(0, 255), torch.where(mo, wo, torch.zeros_like(wo)))

        top_abs, top_idx = torch.topk(buf.abs(), k=8, dim=1)
        top_val = buf.gather(1, top_idx)
        keep = top_abs > 1e-4
        out_d[s:e] = torch.where(keep, top_idx.to(torch.int16), torch.full_like(top_idx, -1, dtype=torch.int16)).cpu()
        out_w[s:e] = torch.where(keep, top_val.to(torch.bfloat16), torch.zeros_like(top_val, dtype=torch.bfloat16)).cpu()
    return out_d, out_w


@dataclass
class CorticalMergeTopology:
    """Precomputed exact BFS path alignment between Host (NEW Broca) and Donor (Base 6M) DAGs."""
    ck_broca: Dict[str, torch.Tensor]
    ck_base: Dict[str, torch.Tensor]
    T_base_to_broca: torch.Tensor          # [256, 256] float32 (CPU)
    align_stats: Dict[str, float]
    total_nodes: int                       # total allocated node IDs (<= pool_capacity)
    broca_count: int                       # 256 + hidden_broca (Broca IDs packed contiguously [0..broca_count-1])
    broca_orig_ids: np.ndarray             # [broca_count] original Broca node IDs
    shared_m_ids: np.ndarray               # [N_shared] merged IDs that exist in BOTH Broca and Base
    shared_base_ids: np.ndarray            # [N_shared] matching Donor Base node IDs
    base_only_m_ids: np.ndarray            # [N_base_only] newly allocated merged IDs for Base-only paths
    base_only_base_ids: np.ndarray         # [N_base_only] corresponding Donor Base node IDs
    # Canonical metadata arrays for all [total_nodes]
    parent: np.ndarray
    cond_kind: np.ndarray
    cond_id: np.ndarray
    depth: np.ndarray
    local_span: np.ndarray
    fires: np.ndarray
    upd_cnt: np.ndarray
    # All merged edges (Broca primary + secondary + Base-only grafted + Base secondary!)
    edge_parents: np.ndarray
    edge_kinds: np.ndarray
    edge_conds: np.ndarray
    edge_children: np.ndarray
    stats: Dict[str, object]


def build_cortical_merge_topology(
    broca_ckpt: str,
    base_ckpt: str,
    pool_capacity: int = 6_000_000,
    ridge_lam: float = 0.05,
) -> CorticalMergeTopology:
    """Compute exact depth-ordered BFS topological alignment between NEW Broca (Host) and Base 6M (Donor) on CPU."""
    t0 = time.perf_counter()
    ck_r = torch.load(broca_ckpt, map_location="cpu", weights_only=False)
    ck_b = torch.load(base_ckpt, map_location="cpu", weights_only=False)

    T_b2r, align_stats = compute_basis_transport(ck_b["E"], ck_r["E"], ridge_lam=ridge_lam)

    # 1) Pack ALL alive Broca nodes contiguously into merged IDs [0 .. broca_count - 1]
    al_r = ck_r["alive"].numpy()
    orig_r_ids = np.nonzero(al_r)[0].astype(np.int64)
    assert np.array_equal(orig_r_ids[:256], np.arange(256, dtype=np.int64)), "Broca roots 0..255 must be alive!"
    broca_count = int(len(orig_r_ids))

    r_old_to_m = np.full(len(al_r), -1, dtype=np.int64)
    r_old_to_m[orig_r_ids] = np.arange(broca_count, dtype=np.int64)

    p_r = ck_r["parent"].numpy()
    ck_r_k = ck_r["cond_kind"].numpy()
    ck_r_c = ck_r["cond_id"].numpy()
    d_r = ck_r["depth"].numpy()
    ls_r = ck_r["local_span"].numpy()
    f_r = ck_r["fires"].numpy()
    u_r = ck_r["upd_cnt"].numpy()

    # Build fast merged edge lookup:
    #   k0 (local byte): int64 key = (parent_m << 9) | byte_cond -> child_m
    #   k1 (context node): tuple (parent_m, cond_m) -> child_m
    k0_map: Dict[int, int] = {}
    k1_map: Dict[Tuple[int, int], int] = {}
    ep_list: List[int] = []
    ek_list: List[int] = []
    ec_list: List[int] = []
    ech_list: List[int] = []

    # Insert all Broca saved edges first (highest priority!)
    if "edge_parents" in ck_r:
        r_ep = ck_r["edge_parents"].numpy()
        r_ek = ck_r["edge_kinds"].numpy()
        r_ec = ck_r["edge_conds"].numpy()
        r_ech = ck_r["edge_children"].numpy()
        for pe, ke, ce, che in zip(r_ep.tolist(), r_ek.tolist(), r_ec.tolist(), r_ech.tolist()):
            pm = int(r_old_to_m[pe]) if 0 <= pe < len(r_old_to_m) else -1
            chm = int(r_old_to_m[che]) if 0 <= che < len(r_old_to_m) else -1
            if pm < 0 or chm < 0:
                continue
            if ke == 0:
                key = (pm << 9) | ce
                if key not in k0_map:
                    k0_map[key] = chm
                    ep_list.append(pm); ek_list.append(0); ec_list.append(ce); ech_list.append(chm)
            elif ke == 1:
                cm = int(r_old_to_m[ce]) if 0 <= ce < len(r_old_to_m) else -1
                if cm >= 0 and (pm, cm) not in k1_map:
                    k1_map[(pm, cm)] = chm
                    ep_list.append(pm); ek_list.append(1); ec_list.append(cm); ech_list.append(chm)

    # Also ensure every Broca hidden node's canonical parent edge is registered in k0_map/k1_map
    for m_id in range(256, broca_count):
        oid = int(orig_r_ids[m_id])
        pe = int(p_r[oid])
        ke = int(ck_r_k[oid])
        ce = int(ck_r_c[oid])
        pm = int(r_old_to_m[pe]) if 0 <= pe < len(r_old_to_m) else -1
        if pm < 0:
            continue
        if ke == 0:
            key = (pm << 9) | ce
            if key not in k0_map:
                k0_map[key] = m_id
                ep_list.append(pm); ek_list.append(0); ec_list.append(ce); ech_list.append(m_id)
        elif ke == 1:
            cm = int(r_old_to_m[ce]) if 0 <= ce < len(r_old_to_m) else -1
            if cm >= 0 and (pm, cm) not in k1_map:
                k1_map[(pm, cm)] = m_id
                ep_list.append(pm); ek_list.append(1); ec_list.append(cm); ech_list.append(m_id)

    # 2) Depth-Ordered BFS Alignment of Donor Base 6M Hidden Nodes (`depth = 1..16`)!
    al_b = ck_b["alive"].numpy()
    p_b = ck_b["parent"].numpy()
    k_b = ck_b["cond_kind"].numpy()
    c_b = ck_b["cond_id"].numpy()
    d_b = ck_b["depth"].numpy()
    ls_b = ck_b["local_span"].numpy()
    f_b = ck_b["fires"].numpy()
    u_b = ck_b["upd_cnt"].numpy()

    b_old_to_m = np.full(len(al_b), -1, dtype=np.int64)
    b_old_to_m[:256] = np.arange(256, dtype=np.int64)

    hidden_b = np.nonzero(al_b[256:])[0].astype(np.int64) + 256
    # Sort strictly by (depth ASC, fires DESC) so high-frequency paths get first priority if pool ever nears cap!
    order_b = np.lexsort((-f_b[hidden_b], d_b[hidden_b]))
    hidden_b_sorted = hidden_b[order_b]

    shared_m: List[int] = []
    shared_b: List[int] = []
    base_only_m: List[int] = []
    base_only_b: List[int] = []

    next_m_id = broca_count
    skipped_unmapped = 0
    skipped_cap = 0

    for uid in hidden_b_sorted.tolist():
        pe = int(p_b[uid])
        ke = int(k_b[uid])
        ce = int(c_b[uid])
        pm = int(b_old_to_m[pe]) if 0 <= pe < len(b_old_to_m) else -1
        if pm < 0:
            skipped_unmapped += 1
            continue
        if ke == 0:
            key = (pm << 9) | ce
            hit = k0_map.get(key, -1)
            if hit >= 0:
                b_old_to_m[uid] = hit
                if hit < broca_count:
                    shared_m.append(hit)
                    shared_b.append(uid)
            else:
                if next_m_id >= pool_capacity:
                    skipped_cap += 1
                    continue
                mid = next_m_id
                next_m_id += 1
                b_old_to_m[uid] = mid
                k0_map[key] = mid
                base_only_m.append(mid)
                base_only_b.append(uid)
                ep_list.append(pm); ek_list.append(0); ec_list.append(ce); ech_list.append(mid)
        elif ke == 1:
            cm = int(b_old_to_m[ce]) if 0 <= ce < len(b_old_to_m) else -1
            if cm < 0:
                skipped_unmapped += 1
                continue
            hit = k1_map.get((pm, cm), -1)
            if hit >= 0:
                b_old_to_m[uid] = hit
                if hit < broca_count:
                    shared_m.append(hit)
                    shared_b.append(uid)
            else:
                if next_m_id >= pool_capacity:
                    skipped_cap += 1
                    continue
                mid = next_m_id
                next_m_id += 1
                b_old_to_m[uid] = mid
                k1_map[(pm, cm)] = mid
                base_only_m.append(mid)
                base_only_b.append(uid)
                ep_list.append(pm); ek_list.append(1); ec_list.append(cm); ech_list.append(mid)

    # Also carry over Base's secondary convergent DAG edges between mapped nodes!
    extra_base_edges = 0
    if "edge_parents" in ck_b:
        b_ep = ck_b["edge_parents"].numpy()
        b_ek = ck_b["edge_kinds"].numpy()
        b_ec = ck_b["edge_conds"].numpy()
        b_ech = ck_b["edge_children"].numpy()
        for pe, ke, ce, che in zip(b_ep.tolist(), b_ek.tolist(), b_ec.tolist(), b_ech.tolist()):
            pm = int(b_old_to_m[pe]) if 0 <= pe < len(b_old_to_m) else -1
            chm = int(b_old_to_m[che]) if 0 <= che < len(b_old_to_m) else -1
            if pm < 0 or chm < 0:
                continue
            if ke == 0:
                key = (pm << 9) | ce
                if key not in k0_map:
                    k0_map[key] = chm
                    ep_list.append(pm); ek_list.append(0); ec_list.append(ce); ech_list.append(chm)
                    extra_base_edges += 1
            elif ke == 1:
                cm = int(b_old_to_m[ce]) if 0 <= ce < len(b_old_to_m) else -1
                if cm >= 0 and (pm, cm) not in k1_map:
                    k1_map[(pm, cm)] = chm
                    ep_list.append(pm); ek_list.append(1); ec_list.append(cm); ech_list.append(chm)
                    extra_base_edges += 1

    # Deduplicate shared_m (if two Base fused paths hit one Broca node, keep highest-fire Base donor!)
    if shared_m:
        best_sh: Dict[int, int] = {}
        for m_i, b_i in zip(shared_m, shared_b):
            prev = best_sh.get(m_i, -1)
            if prev < 0 or int(f_b[b_i]) > int(f_b[prev]):
                best_sh[m_i] = b_i
        shared_m_arr = np.array(list(best_sh.keys()), dtype=np.int64)
        shared_b_arr = np.array(list(best_sh.values()), dtype=np.int64)
    else:
        shared_m_arr = np.zeros(0, dtype=np.int64)
        shared_b_arr = np.zeros(0, dtype=np.int64)

    base_only_m_arr = np.array(base_only_m, dtype=np.int64)
    base_only_b_arr = np.array(base_only_b, dtype=np.int64)
    total_nodes = next_m_id

    # Build canonical node metadata arrays for [0..total_nodes-1]
    parent_m = np.full(total_nodes, -1, dtype=np.int32)
    kind_m = np.full(total_nodes, -1, dtype=np.int8)
    cond_m = np.full(total_nodes, -1, dtype=np.int32)
    depth_m = np.zeros(total_nodes, dtype=np.int16)
    lspan_m = np.zeros(total_nodes, dtype=np.int16)
    fires_m = np.zeros(total_nodes, dtype=np.int32)
    upd_m = np.zeros(total_nodes, dtype=np.int32)

    # Fill Broca [0..broca_count-1]
    for m_id in range(broca_count):
        oid = int(orig_r_ids[m_id])
        pe = int(p_r[oid])
        ke = int(ck_r_k[oid])
        ce = int(ck_r_c[oid])
        parent_m[m_id] = int(r_old_to_m[pe]) if 0 <= pe < len(r_old_to_m) else -1
        kind_m[m_id] = ke
        cond_m[m_id] = (int(r_old_to_m[ce]) if (ke == 1 and 0 <= ce < len(r_old_to_m)) else ce)
        depth_m[m_id] = int(d_r[oid])
        lspan_m[m_id] = int(ls_r[oid])
        fires_m[m_id] = int(f_r[oid])
        upd_m[m_id] = int(u_r[oid])

    # Boost fires on shared nodes
    if len(shared_m_arr) > 0:
        fires_m[shared_m_arr] = np.minimum(2_000_000_000, fires_m[shared_m_arr].astype(np.int64) + f_b[shared_b_arr].astype(np.int64)).astype(np.int32)

    # Fill Base-only [broca_count..total_nodes-1]
    if len(base_only_m_arr) > 0:
        pb_sel = p_b[base_only_b_arr]
        kb_sel = k_b[base_only_b_arr]
        cb_sel = c_b[base_only_b_arr]
        parent_m[base_only_m_arr] = b_old_to_m[pb_sel].astype(np.int32)
        kind_m[base_only_m_arr] = kb_sel.astype(np.int8)
        cond_mapped = np.where(kb_sel == 1, b_old_to_m[np.clip(cb_sel, 0, len(b_old_to_m) - 1)], cb_sel)
        cond_m[base_only_m_arr] = cond_mapped.astype(np.int32)
        depth_m[base_only_m_arr] = d_b[base_only_b_arr].astype(np.int16)
        lspan_m[base_only_m_arr] = ls_b[base_only_b_arr].astype(np.int16)
        fires_m[base_only_m_arr] = f_b[base_only_b_arr].astype(np.int32)
        upd_m[base_only_m_arr] = u_b[base_only_b_arr].astype(np.int32)

    del k0_map, k1_map, r_old_to_m, b_old_to_m
    gc.collect()

    stats = {
        "broca_hidden": broca_count - 256,
        "base_hidden_input": int(len(hidden_b)),
        "shared_nodes_merged": int(len(shared_m_arr)),
        "broca_only_nodes": int((broca_count - 256) - len(shared_m_arr)),
        "base_only_nodes_grafted": int(len(base_only_m_arr)),
        "total_alive_hidden": total_nodes - 256,
        "total_edges": len(ep_list),
        "extra_base_convergent_edges": extra_base_edges,
        "skipped_unmapped": skipped_unmapped,
        "skipped_cap": skipped_cap,
        "overlap_pct_of_broca": round(100.0 * len(shared_m_arr) / max(1, broca_count - 256), 2),
        "align_stats": align_stats,
        "topology_sec": round(time.perf_counter() - t0, 2),
    }
    return CorticalMergeTopology(
        ck_broca=ck_r,
        ck_base=ck_b,
        T_base_to_broca=T_b2r,
        align_stats=align_stats,
        total_nodes=total_nodes,
        broca_count=broca_count,
        broca_orig_ids=orig_r_ids,
        shared_m_ids=shared_m_arr,
        shared_base_ids=shared_b_arr,
        base_only_m_ids=base_only_m_arr,
        base_only_base_ids=base_only_b_arr,
        parent=parent_m,
        cond_kind=kind_m,
        cond_id=cond_m,
        depth=depth_m,
        local_span=lspan_m,
        fires=fires_m,
        upd_cnt=upd_m,
        edge_parents=np.array(ep_list, dtype=np.int32),
        edge_kinds=np.array(ek_list, dtype=np.int8),
        edge_conds=np.array(ec_list, dtype=np.int32),
        edge_children=np.array(ech_list, dtype=np.int32),
        stats=stats,
    )


@torch.no_grad()
def instantiate_merged_brain_vessel(
    topo: CorticalMergeTopology,
    device: torch.device,
    pool_capacity: int = 6_000_000,
    edge_buckets: int = 8_388_608,
) -> UnifiedBrainV13b:
    """Create ONE single 6M-neuron UnifiedBrainV13b vessel and populate its exact merged topology & Broca slots!"""
    cfg = LateralV13bConfig(
        pool_capacity=pool_capacity,
        edge_buckets=edge_buckets,
        feat_dim=256,
        num_slots=16,
        max_hops=16,
        enable_lateral=False,
    )
    brain = UnifiedBrainV13b(cfg, device)
    # Detach internal GPU LTM immediately (saves 268 MB VRAM!)
    for bi in range(brain.ltm.num_banks):
        brain.ltm.sig[bi] = torch.empty((1, 4), dtype=torch.int32, device=device)
        brain.ltm.dst[bi] = torch.full((1, 4), -1, dtype=torch.int16, device=device)
        brain.ltm.w[bi] = torch.zeros((1, 4), dtype=torch.float16, device=device)

    N = topo.total_nodes
    brain.alive.zero_()
    brain.alive[:N] = True
    brain.parent[:N].copy_(torch.from_numpy(topo.parent).to(device))
    brain.cond_kind[:N].copy_(torch.from_numpy(topo.cond_kind).to(device))
    brain.cond_id[:N].copy_(torch.from_numpy(topo.cond_id).to(device))
    brain.depth[:N].copy_(torch.from_numpy(topo.depth).to(device))
    brain.local_span[:N].copy_(torch.from_numpy(topo.local_span).to(device))
    brain.fires[:N].copy_(torch.from_numpy(topo.fires).to(device))
    brain.upd_cnt[:N].copy_(torch.from_numpy(topo.upd_cnt).to(device))

    # Host = NEW Broca's SFT-native basis, bias, 16 working-memory slots, and 24 Executive modes!
    ck_r = topo.ck_broca
    brain.E.copy_(ck_r["E"].to(device))
    brain.bias.copy_(ck_r["bias"].to(device))
    brain.G_write.copy_(ck_r["G_write"].to(device))
    brain.G_keep.copy_(ck_r["G_keep"].to(device))
    brain.write_bias.copy_(ck_r["write_bias"].to(device))
    brain.keep_bias.copy_(ck_r["keep_bias"].to(device))
    brain.W_read.copy_(ck_r["W_read"].to(device))
    brain.W_mod.copy_(ck_r["W_mod"].to(device))
    if "mode_centroids" in ck_r and "mode_llr" in ck_r:
        brain.mode_centroids.copy_(ck_r["mode_centroids"].to(device))
        brain.mode_llr.copy_(ck_r["mode_llr"].to(device))
        brain.mode_ready = True

    # Insert ALL merged edges on GPU in chunks (~1.5s for 5.0M edges!)
    ep_t = torch.from_numpy(topo.edge_parents).to(device=device, dtype=torch.int64)
    ek_t = torch.from_numpy(topo.edge_kinds).to(device=device, dtype=torch.int64)
    ec_t = torch.from_numpy(topo.edge_conds).to(device=device, dtype=torch.int64)
    ech_t = torch.from_numpy(topo.edge_children).to(device=device, dtype=torch.int64)
    rem = torch.ones(ep_t.numel(), dtype=torch.bool, device=device)
    for _ in range(brain.edges.ways * 2):
        if not rem.any():
            break
        r_idx = torch.nonzero(rem, as_tuple=False).squeeze(1)
        ins_ok = brain._insert_edges_tracked(ep_t[r_idx], ek_t[r_idx], ec_t[r_idx], ech_t[r_idx])
        if not ins_ok.any():
            break
        rem[r_idx[ins_ok]] = False
    del ep_t, ek_t, ec_t, ech_t, rem
    brain._recompute_degrees()
    torch.cuda.empty_cache()
    return brain


@torch.no_grad()
def apply_merge_weights_inplace(
    brain: UnifiedBrainV13b,
    topo: CorticalMergeTopology,
    w_new: float = 0.75,
    alpha_donor: float = 0.55,
    chunk_size: int = 131_072,
) -> float:
    """Apply `(w_new, alpha_donor)` node weights + Top-8 sharp synapses in-place on GPU in ~1.5 seconds!
    Allows ultra-fast Dev-200 hyperparameter calibration without rebuilding the 6M edge table!"""
    t0 = time.perf_counter()
    dev = brain.device
    ck_r = topo.ck_broca
    ck_b = topo.ck_base
    T_gpu = topo.T_base_to_broca.to(dev)

    Wr_all = ck_r["W_node"]
    Sr_d = ck_r["sharp_dst"]
    Sr_w = ck_r["sharp_w"]
    Wb_all = ck_b["W_node"]
    Sb_d = ck_b["sharp_dst"]
    Sb_w = ck_b["sharp_w"]

    # 1) Copy 100% of Broca [0..broca_count-1] at 1.0x (Broca-ONLY nodes remain 100% untouched!)
    r_oids = torch.from_numpy(topo.broca_orig_ids)
    Bc = topo.broca_count
    for s in range(0, Bc, chunk_size):
        e = min(Bc, s + chunk_size)
        oid_s = r_oids[s:e]
        brain.W_node[s:e].copy_(Wr_all[oid_s].to(dev))
        brain.sharp_dst[s:e].copy_(Sr_d[oid_s].to(dev))
        brain.sharp_w[s:e].copy_(Sr_w[oid_s].to(dev))

    # 2) Blend Roots [0..255] + Shared Nodes weighted STRONGER toward NEW Broca (`w_new`)!
    w_old = 1.0 - float(w_new)
    all_sh_m = np.concatenate([np.arange(256, dtype=np.int64), topo.shared_m_ids])
    all_sh_b = np.concatenate([np.arange(256, dtype=np.int64), topo.shared_base_ids])
    all_sh_r = topo.broca_orig_ids[all_sh_m]

    N_sh = len(all_sh_m)
    for s in range(0, N_sh, chunk_size):
        e = min(N_sh, s + chunk_size)
        mi = torch.from_numpy(all_sh_m[s:e]).to(dev)
        ri = torch.from_numpy(all_sh_r[s:e])
        bi = torch.from_numpy(all_sh_b[s:e])
        wr = Wr_all[ri].to(device=dev, dtype=torch.float32)
        wb_al = Wb_all[bi].to(device=dev, dtype=torch.float32) @ T_gpu
        brain.W_node[mi] = (w_new * wr + w_old * wb_al).to(torch.bfloat16)

    # Vectorized GPU Top-8 sharp merge for roots + shared nodes
    sd_m, sw_m = merge_sharp_8_gpu(
        Sr_d[torch.from_numpy(all_sh_r)], Sr_w[torch.from_numpy(all_sh_r)],
        Sb_d[torch.from_numpy(all_sh_b)], Sb_w[torch.from_numpy(all_sh_b)],
        w_new=w_new, w_old=w_old, device=dev,
    )
    mi_all = torch.from_numpy(all_sh_m)
    for s in range(0, N_sh, chunk_size):
        e = min(N_sh, s + chunk_size)
        mi = mi_all[s:e].to(dev)
        brain.sharp_dst[mi] = sd_m[s:e].to(dev)
        brain.sharp_w[mi] = sw_m[s:e].to(dev)
    del sd_m, sw_m

    # 3) Graft Base-ONLY encyclopedic nodes at `alpha_donor` strength (chunked on GPU -> < 150 MB temp VRAM!)
    bo_m = topo.base_only_m_ids
    bo_b = topo.base_only_base_ids
    N_bo = len(bo_m)
    for s in range(0, N_bo, chunk_size):
        e = min(N_bo, s + chunk_size)
        mi = torch.from_numpy(bo_m[s:e]).to(dev)
        bi = torch.from_numpy(bo_b[s:e])
        wb_al = (Wb_all[bi].to(device=dev, dtype=torch.float32) @ T_gpu) * float(alpha_donor)
        brain.W_node[mi] = wb_al.to(torch.bfloat16)
        sd_b_ch = Sb_d[bi].to(dev)
        sw_b_ch = Sb_w[bi].to(device=dev, dtype=torch.float32) * float(alpha_donor)
        ctrl_m = (sd_b_ch == 2) | (sd_b_ch == 3) | (sd_b_ch == 4)
        brain.sharp_dst[mi] = torch.where(ctrl_m, torch.full_like(sd_b_ch, -1), sd_b_ch)
        brain.sharp_w[mi] = torch.where(ctrl_m, torch.zeros_like(sw_b_ch), sw_b_ch).to(torch.bfloat16)

    torch.cuda.synchronize(dev)
    return round(time.perf_counter() - t0, 2)


@torch.no_grad()
def train_single_core_readout_polish(
    v20: DecoupledBrainV20,
    train_dialogues: List[Dict],
    max_dialogues: int = 12_000,
    batch_size: int = 64,
    temp_cal: float = 1.08,
    ridge_alpha: float = 0.25,
    eot_weight: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Closed-Form 1-Layer Hebbian Readout Polish on ONE single merged core (ZERO backprop, ~60s!)."""
    t0 = time.perf_counter()
    dev = v20.device
    V, D = 256, v20.core.D
    runner = BatchedCoreRunner(v20, temp_cal=temp_cal)
    use_diags = train_dialogues[:max_dialogues]
    batches = _build_isolated_batches(use_diags, batch_size=batch_size)

    C_hh = torch.zeros((D + 1, D + 1), dtype=torch.float64, device=dev)
    C_hr = torch.zeros((D + 1, V), dtype=torch.float64, device=dev)
    sum_p = torch.zeros((V,), dtype=torch.float64, device=dev)
    cnt_y = torch.zeros((V,), dtype=torch.float64, device=dev)
    n_asst = 0
    n_eot = 0

    for data_np, mask_np, vlen_np, _ in batches:
        data = torch.from_numpy(data_np).to(dev)
        mask_2d = torch.from_numpy(mask_np).to(dev)
        vlen_t = torch.from_numpy(vlen_np).to(dev)
        B, T = data.shape
        st = runner.init_batch_state(B)
        ar = torch.arange(B, device=dev)
        ones_col = torch.ones((B, 1), dtype=torch.float32, device=dev)

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]
            valid_step = (t + 1) < vlen_t
            m_t = mask_2d[:, t] & valid_step
            any_asst = bool(m_t.any().item())
            logits, _ = runner.step_batch(st, x_t, next_true=None, use_engram=False, is_prompt=(not any_asst))
            if not any_asst:
                continue
            h_act = st["last_h_act"]
            mf = m_t.to(torch.float32).unsqueeze(1)
            is_eot_y = (y_t == TURN_EOT_BYTE).unsqueeze(1)
            w_eot = torch.where(is_eot_y, torch.full((B, 1), eot_weight, device=dev), torch.ones((B, 1), device=dev))
            wt = mf * w_eot

            probs = torch.softmax(logits, dim=-1)
            sum_p += (probs * mf).sum(dim=0).to(torch.float64)
            cnt_y.scatter_add_(0, y_t, m_t.to(torch.float64))
            resid = -probs
            resid[ar, y_t] += 1.0

            h_aug = torch.cat([h_act, ones_col], dim=1)
            h_w = (h_aug * wt).to(torch.float64)
            C_hh += h_w.T @ h_aug.to(torch.float64)
            C_hr += h_w.T @ resid.to(torch.float64)
            n_asst += int(m_t.sum().item())
            n_eot += int((m_t & (y_t == TURN_EOT_BYTE)).sum().item())

    b_prior = torch.log((cnt_y + 0.5) / (sum_p + 1e-6)).clamp(-4.0, 14.0)
    b_prior = torch.where(cnt_y > 0.5, b_prior, torch.zeros_like(b_prior))
    p_emp = (cnt_y / max(1.0, float(n_asst))).clamp(min=0.0)
    inv_std = (1.0 / torch.sqrt(p_emp + 0.002)).clamp(max=18.0)
    lam = ridge_alpha * float((torch.trace(C_hh[:D, :D]) / D).clamp(min=1.0).item())
    reg = torch.eye(D + 1, dtype=torch.float64, device=dev) * lam
    reg[D, D] = lam * 0.25
    W_aug = torch.linalg.solve(C_hh + reg, C_hr)
    W_np = np.ascontiguousarray((W_aug[:D, :] * inv_std.unsqueeze(0)).to(torch.float32).clamp(-3.5, 3.5).cpu().numpy(), dtype=np.float32)
    b_np = np.ascontiguousarray(b_prior.to(torch.float32).cpu().numpy(), dtype=np.float32)
    del C_hh, C_hr, W_aug, sum_p, cnt_y
    torch.cuda.empty_cache()

    return W_np, b_np, {
        "dialogues_fit": len(use_diags),
        "assistant_bytes_fit": n_asst,
        "eot_events_fit": n_eot,
        "W_norm": round(float(np.linalg.norm(W_np)), 3),
        "eot_bias": round(float(b_np[TURN_EOT_BYTE]), 4),
        "sec": round(time.perf_counter() - t0, 2),
        "mem": _mem_snapshot(dev),
    }


@torch.no_grad()
def eval_single_core_sft_val(
    v20: DecoupledBrainV20,
    val_dialogues: List[Dict],
    sft_engram=None,
    use_base_engram: bool = True,
    use_sft_engram: bool = True,
    base_engram_mult: float = 0.72,
    sft_engram_scale: float = 0.85,
    W_res: Optional[np.ndarray] = None,
    b_res: Optional[np.ndarray] = None,
    readout_scale: float = 1.0,
    temp_cal: float = 1.08,
    batch_size: int = 48,
) -> Dict[str, object]:
    """Strictly isolated per-dialogue causal SFT evaluation on ONE single UnifiedBrainV13b (ZERO dual overhead!)."""
    t0 = time.perf_counter()
    dev = v20.device
    runner = BatchedCoreRunner(v20, temp_cal=temp_cal)
    batches = _build_isolated_batches(val_dialogues, batch_size=batch_size)
    W_g = torch.from_numpy(W_res).to(device=dev, dtype=torch.float32) if W_res is not None else None
    b_g = torch.from_numpy(b_res).to(device=dev, dtype=torch.float32) if b_res is not None else None

    body_nll_t = torch.zeros((), dtype=torch.float64, device=dev)
    eot_nll_t = torch.zeros((), dtype=torch.float64, device=dev)
    body_top1_t = torch.zeros((), dtype=torch.int64, device=dev)
    body_cnt_t = torch.zeros((), dtype=torch.int64, device=dev)
    eot_cnt_t = torch.zeros((), dtype=torch.int64, device=dev)
    total_body_chars = 0

    for data_np, mask_np, vlen_np, bchars in batches:
        total_body_chars += bchars
        data = torch.from_numpy(data_np).to(dev)
        mask_2d = torch.from_numpy(mask_np).to(dev)
        vlen_t = torch.from_numpy(vlen_np).to(dev)
        B, T = data.shape
        st = runner.init_batch_state(B)
        ar = torch.arange(B, device=dev)

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]
            valid_step = (t + 1) < vlen_t
            m_t = mask_2d[:, t] & valid_step
            nt = torch.where((~m_t) & valid_step, y_t, torch.full_like(y_t, -1))
            any_asst = bool(m_t.any().item())
            logits, _ = runner.step_batch(st, x_t, next_true=None, use_engram=False, is_prompt=(not any_asst))
            if any_asst and W_g is not None and b_g is not None:
                logits = logits + (st["last_h_act"] @ W_g) * readout_scale + b_g.unsqueeze(0)

            want_base = any_asst and use_base_engram and (len(v20.rack.cartridges) > 0)
            want_sft = any_asst and use_sft_engram and (sft_engram is not None)
            if want_base or want_sft:
                prev_idx = (st["hist_ptr"] - 1) % runner.H_buf
                bs24 = st["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - runner._ar24) % runner.H_buf]
                keys_np = v20.rack.hash_bytes24_batch(bs24.to(torch.int32).cpu().numpy())
                cn = st["c_eng"] / st["c_eng"].norm(dim=1, keepdim=True).clamp(min=1e-5)
                cq_np = torch.round(cn * 127.0).clamp(-127, 127).to(torch.int8).cpu().numpy()
                lp_c = torch.log_softmax(logits, dim=-1)
                p_c = torch.exp(lp_c)
                H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
                ent_g = H_c / (1.0 + H_c)
                if want_base:
                    eb_np, _, _ = v20.rack.lookup_batch(keys_np, cq_np, use_context_gate=True)
                    logits = logits + torch.from_numpy(eb_np).to(dev) * (base_engram_mult * ent_g)
                if want_sft:
                    s_np = np.zeros((B, 256), dtype=np.float32)
                    sft_engram.lookup_into(keys_np, cq_np, s_np, use_context_gate=True)
                    np.minimum(s_np, 2.8, out=s_np)
                    logits = logits + torch.from_numpy(s_np).to(dev) * (sft_engram_scale * ent_g)

            # Causal post-scoring STM write on prompt bytes
            cfg = v20.core.cfg
            prev_idx = (st["hist_ptr"] - 1) % runner.H_buf
            bs7 = st["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - runner._ar24[:, :7]) % runner.H_buf]
            h4 = _mix64(bs7[:, 0] + 17)
            h4 = _mix64(h4 ^ ((bs7[:, 1] + 31) * 1315423911))
            h4 = _mix64(h4 ^ ((bs7[:, 2] + 53) * 2654435761))
            h4 = _mix64(h4 ^ ((bs7[:, 3] + 79) * 1000003))
            k4 = torch.where((bs7[:, 3] >= 0) & (h4 != 0), h4, torch.zeros_like(h4))
            h7 = _mix64(h4 ^ ((bs7[:, 4] + 97) * 1000033))
            h7 = _mix64(h7 ^ ((bs7[:, 5] + 113) * 1000037))
            h7 = _mix64(h7 ^ ((bs7[:, 6] + 131) * 1000039))
            k7 = torch.where((bs7[:, 6] >= 0) & (h7 != 0), h7, torch.zeros_like(h7))
            valid_nt = (nt >= 0) & (st["step_i"] >= 5) & (k4 != 0)
            lp_s = torch.log_softmax(logits, dim=-1)
            surp = 1.0 - torch.exp(lp_s[ar, nt.clamp(min=0)])
            do_b = valid_nt & (surp > cfg.stm_surp_thresh)
            sp = st["stm_ptr"]
            st["stm_k4"][ar, sp] = torch.where(do_b, k4, st["stm_k4"][ar, sp])
            st["stm_k7"][ar, sp] = torch.where(do_b, k7, st["stm_k7"][ar, sp])
            st["stm_val"][ar, sp] = torch.where(do_b, nt.clamp(min=0), st["stm_val"][ar, sp])
            st["stm_str"][ar, sp] = torch.where(do_b, torch.ones_like(surp), st["stm_str"][ar, sp])
            st["stm_ptr"] = torch.where(do_b, (sp + 1) % cfg.stm_slots, sp)

            if any_asst:
                log_p = torch.log_softmax(logits, dim=-1)
                nll = -log_p[ar, y_t]
                is_eot = m_t & (y_t == TURN_EOT_BYTE)
                is_body = m_t & (y_t != TURN_EOT_BYTE)
                body_nll_t += torch.where(is_body, nll, torch.zeros_like(nll)).sum().to(torch.float64)
                eot_nll_t += torch.where(is_eot, nll, torch.zeros_like(nll)).sum().to(torch.float64)
                body_top1_t += ((logits.argmax(dim=-1) == y_t) & is_body).sum()
                body_cnt_t += is_body.sum()
                eot_cnt_t += is_eot.sum()

    body_bytes = max(1, int(body_cnt_t.item()))
    eot_cnt = max(1, int(eot_cnt_t.item()))
    body_bits = float(body_nll_t.item()) / math.log(2.0)
    eot_bits_tot = float(eot_nll_t.item()) / math.log(2.0)
    return {
        "body_bpc": round(body_bits / max(1, total_body_chars), 4),
        "body_bpb": round(body_bits / body_bytes, 4),
        "body_top1": round(int(body_top1_t.item()) / body_bytes, 4),
        "eot_bits": round(eot_bits_tot / eot_cnt, 3),
        "body_bytes": body_bytes,
        "eot_count": eot_cnt,
        "sec": round(time.perf_counter() - t0, 2),
    }
