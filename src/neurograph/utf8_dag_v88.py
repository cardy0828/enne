"""NeuroGraph v0.8.8: Pure Raw UTF-8 Byte-Level Living Convergent DAG (V = 256, Zero Tokenizer, Zero <UNK>).

Infinite extensibility:
- Atomic sensory neurons = ONLY 256 raw byte neurons (0x00 .. 0xFF).
- ZERO pre-trained tokenizer, ZERO vocabulary cap, ZERO <UNK> forever!
  Japanese, English, rare kanji, Emoji, math symbols, source code, or arbitrary binary streams
  flow natively through the exact same 256 sensory neurons.
- Starts with ZERO hidden neurons:
  1. Multi-byte UTF-8 characters (3 bytes in Japanese) crystallize organically at depth 1..2.
  2. Multi-character words & syntax branch up to depth 16 (5+ full Japanese chars / 16 ASCII chars).
  3. Emergent Multi-Byte Salient Context (kind=1): rare crystallized hidden neurons (depth 2..5)
     automatically serve as long-range semantic context keys over ~90 bytes—NO Unicode decoder needed!
  4. Selective Asymmetric Synaptic Fusion + Pain-Triggered Auto-Divorce + DAG Apoptosis operate
     directly across byte depths, naturally merging synonymous multi-byte characters & morphemes!
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import math
import numpy as np
import torch

from neurograph.growing_graph_v8 import DynamicEdgeTable, _edge_sig


@dataclass
class UTF8Corpus:
    train_bytes: np.ndarray      # uint8 [N_train_bytes]
    val_bytes: np.ndarray        # uint8 [N_val_bytes]
    train_num_chars: int
    val_num_chars: int
    bytes_per_char_train: float
    bytes_per_char_val: float


def load_utf8_from_utf32(
    data_dir: str,
    name: str = "easy_ja",
    train_chars: int = 2_400_000,
    val_chars: int = 80_000,
) -> UTF8Corpus:
    """Losslessly convert UTF-32-LE corpus files directly into raw UTF-8 byte streams (0..255, zero <UNK>)."""
    base = Path(data_dir)
    tr_u32 = np.memmap(base / f"{name}.train", dtype=np.uint32, mode="r")[:train_chars]
    va_u32 = np.memmap(base / f"{name}.val", dtype=np.uint32, mode="r")[:val_chars]

    # Fast exact UTF-32-LE -> str -> UTF-8 bytes conversion
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


@dataclass
class UTF8DAGConfig:
    vocab_size: int = 256            # Fixed 256 raw byte sensory neurons (0x00..0xFF)!
    pool_capacity: int = 280_000     # Starts at 0 hidden neurons; grows/merges/prunes organically
    feat_dim: int = 96               # Shared byte feature basis E [256, 96] (only 98 KB!)
    max_hops: int = 16               # Spans 5+ Japanese chars (15 bytes) or 16 ASCII chars!
    enable_long_branch: bool = True
    enable_apoptosis: bool = True
    enable_merge: bool = True
    enable_divorce: bool = True
    # Hash table for unbounded dynamic fan-out
    edge_buckets: int = 524_288
    edge_ways: int = 4
    # Neurogenesis (faster spawning since bytes are 2.8x denser than chars)
    birth_surprise_thresh: float = 0.40
    birth_pain_thresh: float = 1.8
    max_births_per_step: int = 128
    # Apoptosis
    min_fires_survive: int = 4
    min_utility_survive: float = 0.015
    # Selective Asymmetric Synaptic Fusion (v0.8.6)
    merge_min_src_fires: int = 4
    merge_max_src_fires: int = 96
    merge_min_lead_fires: int = 16
    merge_lead_ratio: float = 1.5
    merge_min_norm: float = 0.25
    merge_cos_thresh: float = 0.87
    merge_w_cos_thresh: float = 0.83
    merge_max_norm_ratio: float = 2.4
    merge_lsh_bits: int = 10
    merge_rounds: int = 3
    merge_min_depth: int = 2         # Merge emergent char/morpheme nodes (depth >= 2), keep raw 1-byte prefixes crisp!
    # Pain-triggered reversible auto-divorce
    divorce_surprise_thresh: float = 0.45
    divorce_align_thresh: float = -0.02
    divorce_pain_thresh: float = 2.0
    max_divorces_per_step: int = 16
    # Local plasticity (zero backprop)
    lr_feat: float = 0.22
    lr_basis: float = 0.04
    lr_sharp: float = 0.28
    k_sharp: int = 8
    metaplasticity: float = 0.015


class UTF8LivingDAG:
    """Pure Raw UTF-8 Byte-Level Living Convergent DAG (0 hidden neurons -> emergent chars -> words -> DAG)."""

    def __init__(self, cfg: UTF8DAGConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size  # 256
        self.P = cfg.pool_capacity
        self.D = cfg.feat_dim

        # Only 256 atomic byte neurons alive at birth; 0 hidden neurons!
        self.alive = torch.zeros(self.P, dtype=torch.bool, device=device)
        self.alive[: self.V] = True

        self.parent = torch.full((self.P,), -1, dtype=torch.int32, device=device)
        self.cond_kind = torch.full((self.P,), -1, dtype=torch.int8, device=device)
        self.cond_id = torch.full((self.P,), -1, dtype=torch.int32, device=device)
        self.depth = torch.zeros(self.P, dtype=torch.int16, device=device)
        self.local_span = torch.zeros(self.P, dtype=torch.int16, device=device)

        self.pain = torch.zeros(self.P, dtype=torch.float32, device=device)
        self.fires = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.recent_fires = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.utility = torch.zeros(self.P, dtype=torch.float32, device=device)
        self.age = torch.zeros(self.P, dtype=torch.int32, device=device)

        # Edge hash table + lossless DAG metadata + per-edge divorce pain
        self.edges = DynamicEdgeTable(cfg.edge_buckets, cfg.edge_ways, device)
        self.edge_parents = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int32, device=device)
        self.edge_kinds = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int8, device=device)
        self.edge_conds = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int32, device=device)
        self.edge_pain = torch.zeros((cfg.edge_buckets, cfg.edge_ways), dtype=torch.float32, device=device)

        self.in_degree = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.out_degree = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.divorce_immune = torch.zeros(self.P, dtype=torch.bool, device=device)

        # Ultra-tiny byte feature basis E [256, D] + bias [256]
        self.E = torch.zeros((self.V, self.D), dtype=torch.float32, device=device)
        self.bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        self.W_node = torch.zeros((self.P, self.D), dtype=torch.float32, device=device)
        self.upd_cnt = torch.zeros(self.P, dtype=torch.float32, device=device)
        self.sharp_dst = torch.full((self.P, cfg.k_sharp), -1, dtype=torch.int32, device=device)
        self.sharp_w = torch.zeros((self.P, cfg.k_sharp), dtype=torch.float32, device=device)

        self.merged_aliases: Dict[int, List[str]] = {}
        self.total_births = 0
        self.total_deaths = 0
        self.total_merges = 0
        self.total_divorces = 0

    @property
    def num_hidden_alive(self) -> int:
        return int(self.alive[self.V :].sum().item())

    @torch.no_grad()
    def init_sensory_basis(self, train_bytes: np.ndarray, sample_len: int = 3_000_000) -> None:
        """Self-organize byte feature basis E [256, D] purely from raw byte co-occurrence distances (1..3)."""
        V, D = self.V, self.D
        seq = torch.from_numpy(train_bytes[: min(sample_len, len(train_bytes))].astype(np.int64)).to(self.device)

        cooc = torch.zeros((V, V), dtype=torch.float32, device=self.device)
        for d, w in ((1, 1.0), (2, 0.55), (3, 0.35)):
            a, b = seq[:-d], seq[d:]
            vals = torch.full((a.numel(),), w, dtype=torch.float32, device=self.device)
            cooc.view(-1).scatter_add_(0, a * V + b, vals)
            cooc.view(-1).scatter_add_(0, b * V + a, vals)

        tot = cooc.sum().clamp(min=1.0)
        rs = cooc.sum(dim=1, keepdim=True).clamp(min=1.0)
        cs = cooc.sum(dim=0, keepdim=True).clamp(min=1.0)
        ppmi = torch.relu(torch.log((cooc * tot) / (rs * cs) + 1e-8))
        # Also add tiny identity so unseen bytes (e.g. rare ASCII/Emoji lead bytes) get clean orthogonal directions!
        ppmi.add_(torch.eye(V, device=self.device) * 0.25)
        U, S, _ = torch.svd_lowrank(ppmi, q=min(D + 12, V - 1), niter=4)
        E_init = U[:, :D] * torch.sqrt(S[:D].clamp(min=1e-6)).unsqueeze(0)
        self.E.copy_((E_init / E_init.norm(dim=1, keepdim=True).clamp(min=1e-5)) * 1.35)

        counts = torch.bincount(seq, minlength=V).to(torch.float32)
        self.bias.copy_(torch.log((counts + 1.0) / (counts.sum() + V)))

    def describe_byte_neuron(self, nid: int) -> str:
        """Render a grown byte neuron's lineage as UTF-8 text (with hex fallback for partial byte sequences)."""
        if nid < 0 or nid >= self.P or not bool(self.alive[nid].item()):
            return "<DEAD>"
        if nid < self.V:
            b = bytes([nid])
            try:
                s = b.decode("utf-8")
                return repr(s) if s.isprintable() else f"0x{nid:02x}"
            except UnicodeDecodeError:
                return f"0x{nid:02x}"

        local_bytes: List[int] = []
        ctx_tags: List[str] = []
        cur = nid
        guard = 0
        while cur >= self.V and guard < 24:
            k = int(self.cond_kind[cur].item())
            cid = int(self.cond_id[cur].item())
            if k == 0 and 0 <= cid < 256:
                local_bytes.append(cid)
            else:
                ctx_tags.append(f"@N{cid}")
            cur = int(self.parent[cur].item())
            guard += 1
        if 0 <= cur < 256:
            # local_bytes were added from furthest-left (deepest) to closest-left; cur is the rightmost (current) byte!
            raw = bytes(local_bytes + [cur])
        else:
            raw = bytes(local_bytes)

        # Try clean UTF-8 decode; if partial leading/trailing UTF-8 continuation bytes, decode valid interior cleanly!
        try:
            txt = raw.decode("utf-8")
            label = repr(txt)
        except UnicodeDecodeError:
            # Find longest valid UTF-8 subslice for human readability
            dec = raw.decode("utf-8", errors="ignore")
            label = f"«{dec}»({raw.hex()[:12]})" if dec else f"0x{raw.hex()}"

        if ctx_tags:
            return f"{label}[{','.join(ctx_tags[:2])}]"
        return label

    @torch.no_grad()
    def _recompute_degrees(self) -> None:
        self.in_degree.zero_()
        self.out_degree.zero_()
        valid = self.edges.child >= 0
        if valid.any():
            ch = self.edges.child[valid].to(torch.int64)
            pa = self.edge_parents[valid].to(torch.int64)
            ones = torch.ones(ch.numel(), dtype=torch.int32, device=self.device)
            self.in_degree.scatter_add_(0, ch, ones)
            self.out_degree.scatter_add_(0, pa, ones)

    @torch.no_grad()
    def _lookup_with_loc(
        self, parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.edges.mask).to(torch.int64)
        b_sig = self.edges.sig[bkts]
        b_ch = self.edges.child[bkts]
        hit = (b_sig == sigs.unsqueeze(1)) & (b_ch >= 0)
        has = hit.any(dim=1)
        slot = hit.to(torch.int32).argmax(dim=1).to(torch.int64)
        found = b_ch[torch.arange(parent.numel(), device=self.device), slot].to(torch.int64)
        child = torch.where(has, found, torch.full_like(found, -1))
        return child, bkts, slot

    @torch.no_grad()
    def _insert_edges_tracked(
        self, parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor, child_id: torch.Tensor
    ) -> torch.Tensor:
        n = parent.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.edges.mask).to(torch.int64)

        order = torch.argsort(bkts, stable=True)
        s_bkts = bkts[order]
        is_first = torch.ones(n, dtype=torch.bool, device=self.device)
        if n > 1:
            is_first[1:] = s_bkts[1:] != s_bkts[:-1]

        cand_idx = order[is_first]
        c_bkts = bkts[cand_idx]
        c_sigs = sigs[cand_idx]
        c_child = child_id[cand_idx].to(torch.int32)

        b_sig = self.edges.sig[c_bkts]
        b_ch = self.edges.child[c_bkts]
        already = ((b_sig == c_sigs.unsqueeze(1)) & (b_ch >= 0)).any(dim=1)
        empty = b_ch < 0
        ok = (~already) & empty.any(dim=1)

        success = torch.zeros(n, dtype=torch.bool, device=self.device)
        if ok.any():
            ok_pos = cand_idx[ok]
            ok_bkts = c_bkts[ok]
            ok_slots = empty[ok].to(torch.int32).argmax(dim=1)
            self.edges.sig[ok_bkts, ok_slots] = c_sigs[ok]
            self.edges.child[ok_bkts, ok_slots] = c_child[ok]
            self.edge_parents[ok_bkts, ok_slots] = parent[ok_pos].to(torch.int32)
            self.edge_kinds[ok_bkts, ok_slots] = kind[ok_pos].to(torch.int8)
            self.edge_conds[ok_bkts, ok_slots] = cond[ok_pos].to(torch.int32)
            success[ok_pos] = True
        return success

    @torch.no_grad()
    def _allocate_free_ids(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        free = torch.nonzero(~self.alive[self.V :], as_tuple=False).squeeze(1) + self.V
        return free[:count] if free.numel() > 0 else torch.empty(0, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def _spawn_children(self, parents: torch.Tensor, kinds: torch.Tensor, conds: torch.Tensor) -> int:
        M = parents.numel()
        if M == 0:
            return 0
        if M > self.cfg.max_births_per_step:
            parents, kinds, conds = parents[: self.cfg.max_births_per_step], kinds[: self.cfg.max_births_per_step], conds[: self.cfg.max_births_per_step]
            M = parents.numel()
        new_ids = self._allocate_free_ids(M)
        n_alloc = new_ids.numel()
        if n_alloc == 0:
            return 0
        parents, kinds, conds = parents[:n_alloc], kinds[:n_alloc], conds[:n_alloc]

        ok = self._insert_edges_tracked(parents, kinds, conds, new_ids)
        if not ok.any():
            return 0
        born, bp, bk, bc = new_ids[ok], parents[ok], kinds[ok], conds[ok]
        self.alive[born] = True
        self.parent[born] = bp.to(torch.int32)
        self.cond_kind[born] = bk.to(torch.int8)
        self.cond_id[born] = bc.to(torch.int32)
        self.depth[born] = self.depth[bp] + 1
        self.local_span[born] = self.local_span[bp] + (bk == 0).to(torch.int16)

        self.pain[born] = 0.0
        self.fires[born] = 1
        self.recent_fires[born] = 1
        self.utility[born] = 0.05
        self.age[born] = 0
        self.W_node[born] = 0.0
        self.upd_cnt[born] = 0.0
        self.sharp_dst[born] = -1
        self.sharp_w[born] = 0.0
        self.divorce_immune[born] = False

        self.pain[bp] *= 0.35
        n_born = int(born.numel())
        self.total_births += n_born
        return n_born

    @torch.no_grad()
    def _divorce_edges(self, bkts: torch.Tensor, slots: torch.Tensor) -> int:
        if bkts.numel() == 0:
            return 0
        lin = bkts * self.edges.ways + slots
        uniq_lin = torch.unique(lin)
        u_bkt = uniq_lin // self.edges.ways
        u_slot = uniq_lin % self.edges.ways

        hub = self.edges.child[u_bkt, u_slot].to(torch.int64)
        valid = (hub >= self.V) & self.alive[hub.clamp(min=0)] & (self.in_degree[hub.clamp(min=0)] > 1)
        if not valid.any():
            return 0
        u_bkt, u_slot, hub = u_bkt[valid], u_slot[valid], hub[valid]
        if u_bkt.numel() > self.cfg.max_divorces_per_step:
            u_bkt, u_slot, hub = u_bkt[: self.cfg.max_divorces_per_step], u_slot[: self.cfg.max_divorces_per_step], hub[: self.cfg.max_divorces_per_step]

        new_ids = self._allocate_free_ids(u_bkt.numel())
        n_alloc = new_ids.numel()
        if n_alloc == 0:
            return 0
        u_bkt, u_slot, hub = u_bkt[:n_alloc], u_slot[:n_alloc], hub[:n_alloc]

        self.alive[new_ids] = True
        self.parent[new_ids] = self.edge_parents[u_bkt, u_slot]
        self.cond_kind[new_ids] = self.edge_kinds[u_bkt, u_slot]
        self.cond_id[new_ids] = self.edge_conds[u_bkt, u_slot]
        self.depth[new_ids] = self.depth[hub]
        self.local_span[new_ids] = self.local_span[hub]
        self.W_node[new_ids] = self.W_node[hub] * 0.75
        self.sharp_dst[new_ids] = self.sharp_dst[hub]
        self.sharp_w[new_ids] = self.sharp_w[hub] * 0.70
        self.upd_cnt[new_ids] = 4.0
        self.fires[new_ids] = 4
        self.recent_fires[new_ids] = 4
        self.utility[new_ids] = 0.10
        self.pain[new_ids] = 0.0
        self.age[new_ids] = 0
        self.divorce_immune[new_ids] = True

        self.edges.child[u_bkt, u_slot] = new_ids.to(torch.int32)
        self.edge_pain[u_bkt, u_slot] = 0.0
        ones = torch.ones(n_alloc, dtype=torch.int32, device=self.device)
        self.in_degree.scatter_add_(0, hub, -ones)
        self.in_degree.clamp_(min=0)
        self.in_degree[new_ids] = 1
        self.out_degree[new_ids] = 0
        self.total_divorces += n_alloc
        return n_alloc

    @torch.no_grad()
    def _update_sharp_synapses(
        self,
        nodes: torch.Tensor,
        y_t: torch.Tensor,
        surp: torch.Tensor,
        top_bad: torch.Tensor,
        p_bad: torch.Tensor,
        lr: float,
    ) -> None:
        if nodes.numel() == 0:
            return
        n = nodes.numel()
        order = torch.argsort(nodes, stable=True)
        sn = nodes[order]
        is_first = torch.ones(n, dtype=torch.bool, device=self.device)
        if n > 1:
            is_first[1:] = sn[1:] != sn[:-1]
        idx = order[is_first]

        u = nodes[idx]
        yt = y_t[idx].to(torch.int32)
        sp = surp[idx]
        tb = top_bad[idx].to(torch.int32)
        pb = p_bad[idx]

        dst = self.sharp_dst[u]
        w = self.sharp_w[u]
        hit = dst == yt.unsqueeze(1)
        has = hit.any(dim=1)
        if has.any():
            hu = u[has]
            hs = hit[has].to(torch.int32).argmax(dim=1)
            self.sharp_w[hu, hs] = (self.sharp_w[hu, hs] + lr * sp[has]).clamp(-3.5, 4.0)

        need = (~has) & (sp > 0.25)
        if need.any():
            nu = u[need]
            nd = dst[need]
            nw = w[need]
            emp = nd < 0
            slot = torch.where(emp.any(dim=1), emp.to(torch.int32).argmax(dim=1), nw.abs().argmin(dim=1))
            self.sharp_dst[nu, slot] = yt[need]
            self.sharp_w[nu, slot] = (lr * sp[need]).clamp(0.05, 1.5)

        bad_valid = (tb != yt) & (pb > 0.08)
        if bad_valid.any():
            bu, bt, bp = u[bad_valid], tb[bad_valid], pb[bad_valid]
            bhit = self.sharp_dst[bu] == bt.unsqueeze(1)
            bhas = bhit.any(dim=1)
            if bhas.any():
                bhu = bu[bhas]
                bhs = bhit[bhas].to(torch.int32).argmax(dim=1)
                self.sharp_w[bhu, bhs] = (self.sharp_w[bhu, bhs] - (lr * 0.8) * bp[bhas]).clamp(-3.5, 4.0)

    @torch.no_grad()
    def apoptosis(self) -> Dict[str, int]:
        """DAG-aware metabolic cell death & orphan pruning."""
        if not self.cfg.enable_apoptosis:
            return {"died": 0, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

        hidden_alive = self.alive.clone()
        hidden_alive[: self.V] = False
        self.age[hidden_alive] += 1

        w_norm = self.W_node.norm(dim=1)
        sharp_norm = self.sharp_w.abs().sum(dim=1)
        weak = (self.recent_fires < self.cfg.min_fires_survive) | (
            (self.utility < self.cfg.min_utility_survive) & (w_norm < 0.15) & (sharp_norm < 0.15)
        )
        dead = hidden_alive & (self.age >= 1) & weak

        valid_e = self.edges.child >= 0
        if valid_e.any() and dead.any():
            e_ch = self.edges.child.clamp(min=0).to(torch.int64)
            e_pa = self.edge_parents.clamp(min=0).to(torch.int64)
            kill_e = valid_e & (dead[e_ch] | dead[e_pa])
            if kill_e.any():
                self.edges.sig[kill_e] = 0
                self.edges.child[kill_e] = -1
                self.edge_parents[kill_e] = -1
                self.edge_kinds[kill_e] = -1
                self.edge_conds[kill_e] = -1
                self.edge_pain[kill_e] = 0.0

            for _ in range(self.cfg.max_hops):
                still_e = self.edges.child >= 0
                has_in = torch.zeros(self.P, dtype=torch.bool, device=self.device)
                has_in[: self.V] = True
                if still_e.any():
                    has_in[self.edges.child[still_e].to(torch.int64)] = True
                orphans = hidden_alive & (~dead) & (~has_in)
                if not orphans.any():
                    break
                dead |= orphans
                e_pa2 = self.edge_parents.clamp(min=0).to(torch.int64)
                kill_e2 = still_e & dead[e_pa2]
                if kill_e2.any():
                    self.edges.sig[kill_e2] = 0
                    self.edges.child[kill_e2] = -1
                    self.edge_parents[kill_e2] = -1
                    self.edge_kinds[kill_e2] = -1
                    self.edge_conds[kill_e2] = -1
                    self.edge_pain[kill_e2] = 0.0

        n_died = int(dead.sum().item())
        if n_died > 0:
            for did in torch.nonzero(dead, as_tuple=False).squeeze(1).cpu().tolist():
                self.merged_aliases.pop(did, None)
            self.alive[dead] = False
            self.parent[dead] = -1
            self.cond_kind[dead] = -1
            self.cond_id[dead] = -1
            self.depth[dead] = 0
            self.local_span[dead] = 0
            self.pain[dead] = 0.0
            self.fires[dead] = 0
            self.utility[dead] = 0.0
            self.age[dead] = 0
            self.W_node[dead] = 0.0
            self.upd_cnt[dead] = 0.0
            self.sharp_dst[dead] = -1
            self.sharp_w[dead] = 0.0
            self.divorce_immune[dead] = False
            self.total_deaths += n_died

        self.recent_fires.zero_()
        self.utility.mul_(0.85)
        self._recompute_degrees()
        return {"died": n_died, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

    @torch.no_grad()
    def _build_full_pred_signature(self, node_ids: torch.Tensor) -> torch.Tensor:
        P = self.W_node[node_ids].clone()
        sd = self.sharp_dst[node_ids]
        sw = self.sharp_w[node_ids]
        valid = (sd >= 0) & (sw.abs() > 1e-4)
        if valid.any():
            e_sd = self.E[sd.clamp(min=0).to(torch.int64)]
            w_m = torch.where(valid, sw, torch.zeros_like(sw)).unsqueeze(-1)
            P.add_((e_sd * w_m).sum(dim=1), alpha=0.55)
        return P

    @torch.no_grad()
    def fuse_similar_neurons(self, seed: int = 100) -> Dict[str, int]:
        """Selective Asymmetric Synaptic Fusion on emergent multi-byte neurons (depth >= merge_min_depth)."""
        cfg = self.cfg
        if not cfg.enable_merge:
            return {"merged": 0, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

        device = self.device
        self._recompute_degrees()
        redirect = torch.arange(self.P, dtype=torch.int64, device=device)
        total_merged_now = 0
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        for _ in range(cfg.merge_rounds):
            idx_all = torch.arange(self.P, device=device)
            w_norm = self.W_node.norm(dim=1)
            base_alive = (
                self.alive
                & (idx_all >= self.V)
                & (self.depth >= cfg.merge_min_depth)
                & (redirect == idx_all)
                & (w_norm >= cfg.merge_min_norm)
            )
            is_src_elig = (
                base_alive
                & (~self.divorce_immune)
                & (self.fires >= cfg.merge_min_src_fires)
                & (self.fires <= cfg.merge_max_src_fires)
                & (self.out_degree == 0)
            )
            is_lead_elig = base_alive & (self.fires >= cfg.merge_min_lead_fires)
            cand_ids = torch.nonzero(is_src_elig | is_lead_elig, as_tuple=False).squeeze(1)
            M = cand_ids.numel()
            if M < 2:
                break

            P_sig = self._build_full_pred_signature(cand_ids)
            U_p = P_sig / P_sig.norm(dim=1, keepdim=True).clamp(min=1e-6)
            w_c_norm = w_norm[cand_ids].clamp(min=1e-6)
            U_w = self.W_node[cand_ids] / w_c_norm.unsqueeze(1)

            planes = torch.randn((self.D, cfg.merge_lsh_bits), generator=gen, device=device)
            bits = (U_p @ planes) > 0
            powers = 1 << torch.arange(cfg.merge_lsh_bits, dtype=torch.int64, device=device)
            lsh_code = (bits.to(torch.int64) * powers.unsqueeze(0)).sum(dim=1)

            strata_key = (
                (self.depth[cand_ids].to(torch.int64) << 28)
                | (self.local_span[cand_ids].to(torch.int64) << 20)
                | lsh_code
            )

            fires_c = self.fires[cand_ids].to(torch.int64)
            f_ord = torch.argsort(fires_c, descending=True)
            s_ord = torch.argsort(strata_key[f_ord], stable=True)
            perm = f_ord[s_ord]

            s_ids = cand_ids[perm]
            s_keys = strata_key[perm]
            s_Up = U_p[perm]
            s_Uw = U_w[perm]
            s_wn = w_c_norm[perm]
            s_fires = fires_c[perm]

            is_new_seg = torch.ones(M, dtype=torch.bool, device=device)
            is_new_seg[1:] = s_keys[1:] != s_keys[:-1]
            seg_starts = torch.nonzero(is_new_seg, as_tuple=False).squeeze(1)
            seg_id = torch.cumsum(is_new_seg.to(torch.int64), dim=0) - 1
            leader_pos = seg_starts[seg_id]

            not_leader = torch.arange(M, device=device) != leader_pos
            if not not_leader.any():
                continue

            lead_ids = s_ids[leader_pos]
            asym_ok = (
                not_leader
                & is_src_elig[s_ids]
                & is_lead_elig[lead_ids]
                & (s_fires[leader_pos].to(torch.float32) >= s_fires.to(torch.float32) * cfg.merge_lead_ratio)
            )
            cos_p = (s_Up * s_Up[leader_pos]).sum(dim=1)
            cos_w = (s_Uw * s_Uw[leader_pos]).sum(dim=1)
            norm_ratio = s_wn[leader_pos] / s_wn.clamp(min=1e-4)
            norm_ok = (norm_ratio >= 0.45) & (norm_ratio <= cfg.merge_max_norm_ratio)

            w_st = self.sharp_w[s_ids].argmax(dim=1)
            w_lt = self.sharp_w[lead_ids].argmax(dim=1)
            st_dst = self.sharp_dst[s_ids, w_st]
            lt_dst = self.sharp_dst[lead_ids, w_lt]
            st_w = self.sharp_w[s_ids, w_st]
            lt_w = self.sharp_w[lead_ids, w_lt]
            lead_has_st = (self.sharp_dst[lead_ids] == st_dst.unsqueeze(1)).any(dim=1)
            sharp_conflict = (
                ((st_dst >= 0) & (lt_dst >= 0) & (st_w > 0.35) & (lt_w > 0.35) & (st_dst != lt_dst))
                | ((st_dst >= 0) & (st_w > 0.55) & (~lead_has_st))
            )

            do_merge = asym_ok & (cos_p >= cfg.merge_cos_thresh) & (cos_w >= cfg.merge_w_cos_thresh) & norm_ok & (~sharp_conflict)
            if not do_merge.any():
                continue

            src_m = s_ids[do_merge]
            dst_m = lead_ids[do_merge]

            for s_id, d_id in zip(src_m.cpu().tolist(), dst_m.cpu().tolist()):
                aliases = self.merged_aliases.setdefault(d_id, [self.describe_byte_neuron(d_id)])
                src_aliases = self.merged_aliases.pop(s_id, [self.describe_byte_neuron(s_id)])
                for a in src_aliases:
                    if a not in aliases and len(aliases) < 12:
                        aliases.append(a)

            f_src = self.fires[src_m].to(torch.float32)
            u_dst_uniq = torch.unique(dst_m)
            w_sum = torch.zeros_like(self.W_node)
            f_sum = torch.zeros(self.P, dtype=torch.float32, device=device)
            w_sum[u_dst_uniq] = self.W_node[u_dst_uniq] * self.fires[u_dst_uniq].to(torch.float32).unsqueeze(1)
            f_sum[u_dst_uniq] = self.fires[u_dst_uniq].to(torch.float32)
            w_sum.index_add_(0, dst_m, self.W_node[src_m] * f_src.unsqueeze(1))
            f_sum.scatter_add_(0, dst_m, f_src)
            self.W_node[u_dst_uniq] = w_sum[u_dst_uniq] / f_sum[u_dst_uniq].clamp(min=1.0).unsqueeze(1)

            s_sd, s_sw, d_sd = self.sharp_dst[src_m], self.sharp_w[src_m], self.sharp_dst[dst_m]
            match = (s_sd.unsqueeze(2) == d_sd.unsqueeze(1)) & (s_sd.unsqueeze(2) >= 0)
            if match.any():
                alpha = (f_src / (self.fires[dst_m].to(torch.float32) + f_src).clamp(min=1.0)).clamp(max=0.25)
                matched_sw = (match.to(torch.float32) * s_sw.unsqueeze(2)).sum(dim=1)
                delta = torch.where(match.any(dim=1), (matched_sw - self.sharp_w[dst_m]) * alpha.unsqueeze(1), torch.zeros_like(matched_sw))
                self.sharp_w.index_add_(0, dst_m, delta)
                self.sharp_w[u_dst_uniq].clamp_(-3.5, 4.0)

            self.fires.scatter_add_(0, dst_m, self.fires[src_m])
            self.recent_fires.scatter_add_(0, dst_m, self.recent_fires[src_m])
            self.utility.scatter_add_(0, dst_m, self.utility[src_m])
            self.upd_cnt.scatter_add_(0, dst_m, self.upd_cnt[src_m])

            redirect[src_m] = dst_m
            self.alive[src_m] = False
            alive_h = self.alive & (idx_all >= self.V)
            if alive_h.any():
                self.parent[alive_h] = redirect[self.parent[alive_h].clamp(min=0).to(torch.int64)].to(torch.int32)
            self.parent[src_m] = -1
            self.cond_kind[src_m] = -1
            self.cond_id[src_m] = -1
            self.depth[src_m] = 0
            self.local_span[src_m] = 0
            self.pain[src_m] = 0.0
            self.fires[src_m] = 0
            self.recent_fires[src_m] = 0
            self.utility[src_m] = 0.0
            self.age[src_m] = 0
            self.W_node[src_m] = 0.0
            self.upd_cnt[src_m] = 0.0
            self.sharp_dst[src_m] = -1
            self.sharp_w[src_m] = 0.0
            self.divorce_immune[src_m] = False
            total_merged_now += int(src_m.numel())

        if total_merged_now > 0:
            for _ in range(4):
                redirect = redirect[redirect]
            valid_e = self.edges.child >= 0
            if valid_e.any():
                old_p = self.edge_parents[valid_e].to(torch.int64)
                old_k = self.edge_kinds[valid_e].to(torch.int64)
                old_c = self.edge_conds[valid_e].to(torch.int64)
                old_ch = self.edges.child[valid_e].to(torch.int64)
                new_p = redirect[old_p]
                new_ch = redirect[old_ch]
                keep = self.alive[new_p] & self.alive[new_ch] & (new_p != new_ch)
                new_p, old_k, old_c, new_ch = new_p[keep], old_k[keep], old_c[keep], new_ch[keep]
                ord_f = torch.argsort(self.fires[new_ch], descending=True)
                new_p, old_k, old_c, new_ch = new_p[ord_f], old_k[ord_f], old_c[ord_f], new_ch[ord_f]

                self.edges.sig.zero_()
                self.edges.child.fill_(-1)
                self.edge_parents.fill_(-1)
                self.edge_kinds.fill_(-1)
                self.edge_conds.fill_(-1)
                self.edge_pain.zero_()

                e_sigs = _edge_sig(new_p, old_k, old_c)
                s_ord = torch.argsort(e_sigs, stable=True)
                ss = e_sigs[s_ord]
                is_f = torch.ones(new_p.numel(), dtype=torch.bool, device=device)
                if new_p.numel() > 1:
                    is_f[1:] = ss[1:] != ss[:-1]
                u_idx = s_ord[is_f]
                new_p, old_k, old_c, new_ch = new_p[u_idx], old_k[u_idx], old_c[u_idx], new_ch[u_idx]

                rem = torch.ones(new_p.numel(), dtype=torch.bool, device=device)
                for _ in range(self.edges.ways * 2):
                    if not rem.any():
                        break
                    r_idx = torch.nonzero(rem, as_tuple=False).squeeze(1)
                    ins_ok = self._insert_edges_tracked(new_p[r_idx], old_k[r_idx], old_c[r_idx], new_ch[r_idx])
                    if not ins_ok.any():
                        break
                    rem[r_idx[ins_ok]] = False

            self.total_merges += total_merged_now
            self._recompute_degrees()

        return {"merged": total_merged_now, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

    @torch.no_grad()
    def run_stream(
        self,
        byte_arr: np.ndarray,
        train: bool = True,
        batch_streams: int = 2048,
        lr_scale: float = 1.0,
        bytes_per_char: float = 2.8,
    ) -> Dict[str, float]:
        """Event-driven 16-hop raw byte traversal with emergent multi-byte context & reversible DAG."""
        cfg = self.cfg
        device = self.device
        V, D = self.V, self.D
        self._recompute_degrees()

        total_len = len(byte_arr)
        stream_len = total_len // batch_streams
        if stream_len < 48:
            batch_streams = max(1, total_len // 48)
            stream_len = total_len // batch_streams

        data = torch.from_numpy(
            byte_arr[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(device)
        B, T = data.shape

        H_buf = cfg.max_hops + 4
        byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0

        # Emergent multi-byte long-range context: delay ring of 12 bytes storing crystallized hidden neuron IDs!
        D_ctx = 12
        node_delay = torch.full((B, D_ctx), -1, dtype=torch.int64, device=device)
        nd_ptr = 0
        ctx_node = torch.full((B, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((B, 2), dtype=torch.float32, device=device)

        arange_B = torch.arange(B, device=device)
        lr_feat = cfg.lr_feat * lr_scale
        lr_basis = cfg.lr_basis * lr_scale
        lr_sharp = cfg.lr_sharp * lr_scale

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0
        total_depth_sum = 0.0
        total_visited_sum = 0.0
        max_depth_seen = 0
        births_this_pass = 0
        divorces_this_pass = 0

        MAX_FRONT = 4
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=device)
        track_divorce = train and cfg.enable_divorce and bool((self.in_degree[self.V :] > 1).any().item())

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]

            byte_hist[:, hist_ptr] = x_t
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            # Update emergent multi-byte salient long-range context (purely from grown hidden nodes, zero decoder!)
            ctx_score.mul_(0.985)
            if t >= D_ctx:
                u_old = node_delay[:, nd_ptr]
                ok_u = (u_old >= V) & self.alive[u_old.clamp(min=0)]
                if ok_u.any():
                    # Rare crystallized multi-byte neurons = salient content kanji/words!
                    f_old = self.fires[u_old.clamp(min=0)].to(torch.float32)
                    sal_old = torch.where(ok_u, 1.8 / torch.sqrt(1.0 + f_old * 0.02), torch.zeros_like(f_old))
                    already = (ctx_node[:, 0] == u_old) | (ctx_node[:, 1] == u_old)
                    weak_slot = ctx_score.argmin(dim=1)
                    weak_val = ctx_score[arange_B, weak_slot]
                    rep = ok_u & (~already) & (sal_old > weak_val) & (sal_old > 0.55)
                    if rep.any():
                        rb = torch.nonzero(rep, as_tuple=False).squeeze(1)
                        rs = weak_slot[rb]
                        ctx_node[rb, rs] = u_old[rb]
                        ctx_score[rb, rs] = sal_old[rb]

            visited_nodes: List[torch.Tensor] = [x_t]
            visited_gains: List[float] = [1.0]
            traversed_conv_edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

            frontier = x_t.unsqueeze(1)
            stream_max_depth = torch.zeros(B, dtype=torch.int32, device=device)
            deepest_leaf = x_t.clone()
            deepest_span = torch.zeros(B, dtype=torch.int64, device=device)
            # Track emergent char/morpheme node (depth 2..5) for long-range context memory
            emergent_ctx_cand = torch.full((B,), -1, dtype=torch.int64, device=device)

            for hop in range(1, cfg.max_hops + 1):
                valid_f = frontier >= 0
                if not valid_f.any():
                    break

                fb, ff = torch.nonzero(valid_f, as_tuple=True)
                p_ids = frontier[fb, ff]
                p_span = self.local_span[p_ids].to(torch.int64)
                next_dist = (p_span + 1).clamp(max=H_buf - 1)
                c_loc = byte_hist[fb, (cur_idx - next_dist) % H_buf]

                k0 = torch.zeros_like(p_ids)
                if track_divorce:
                    ch_loc, loc_bkt, loc_slot = self._lookup_with_loc(p_ids, k0, c_loc)
                else:
                    ch_loc = self.edges.lookup(p_ids, k0, c_loc)
                ch_loc = torch.where(
                    (c_loc >= 0) & (ch_loc >= V) & self.alive[ch_loc.clamp(min=0)],
                    ch_loc,
                    torch.full_like(ch_loc, -1),
                )

                ch_ctx = torch.full_like(p_ids, -1)
                ctx_bkt = ctx_slot = None
                if cfg.enable_long_branch and 2 <= hop <= 6:
                    top_ctx = ctx_node[fb, ctx_score[fb].argmax(dim=1)]
                    k1 = torch.ones_like(p_ids)
                    if track_divorce:
                        raw_ctx, ctx_bkt, ctx_slot = self._lookup_with_loc(p_ids, k1, top_ctx)
                    else:
                        raw_ctx = self.edges.lookup(p_ids, k1, top_ctx)
                    ch_ctx = torch.where(
                        (top_ctx >= V) & (raw_ctx >= V) & self.alive[raw_ctx.clamp(min=0)],
                        raw_ctx,
                        torch.full_like(raw_ctx, -1),
                    )

                next_front = torch.full((B, MAX_FRONT), -1, dtype=torch.int64, device=device)
                hit_loc = ch_loc >= 0
                if hit_loc.any():
                    hl_b = fb[hit_loc]
                    hl_c = ch_loc[hit_loc]
                    hl_slot = (ff[hit_loc] * 2) % MAX_FRONT
                    next_front[hl_b, hl_slot] = hl_c
                    stream_max_depth[hl_b] = hop
                    nsp = self.local_span[hl_c].to(torch.int64)
                    deeper = nsp > deepest_span[hl_b]
                    if deeper.any():
                        deepest_leaf[hl_b[deeper]] = hl_c[deeper]
                        deepest_span[hl_b[deeper]] = nsp[deeper]
                    if 2 <= hop <= 4:
                        emergent_ctx_cand[hl_b] = hl_c
                    if track_divorce:
                        is_conv = self.in_degree[hl_c] > 1
                        if is_conv.any():
                            traversed_conv_edges.append(
                                (hl_b[is_conv], loc_bkt[hit_loc][is_conv], loc_slot[hit_loc][is_conv], hl_c[is_conv])
                            )

                hit_ctx = ch_ctx >= 0
                if hit_ctx.any():
                    hc_b = fb[hit_ctx]
                    hc_c = ch_ctx[hit_ctx]
                    hc_slot = (ff[hit_ctx] * 2 + 1) % MAX_FRONT
                    next_front[hc_b, hc_slot] = hc_c
                    stream_max_depth[hc_b] = hop
                    if track_divorce and ctx_bkt is not None:
                        is_conv_c = self.in_degree[hc_c] > 1
                        if is_conv_c.any():
                            traversed_conv_edges.append(
                                (hc_b[is_conv_c], ctx_bkt[hit_ctx][is_conv_c], ctx_slot[hit_ctx][is_conv_c], hc_c[is_conv_c])
                            )

                if not (hit_loc.any() or hit_ctx.any()):
                    break

                gain = 1.0 / (1.0 + 0.10 * hop)
                for col in range(MAX_FRONT):
                    col_v = next_front[:, col]
                    if (col_v >= 0).any():
                        visited_nodes.append(col_v)
                        visited_gains.append(gain)

                frontier = next_front

            # Push emergent multi-byte neuron into delay buffer
            node_delay[:, nd_ptr] = emergent_ctx_cand
            nd_ptr = (nd_ptr + 1) % D_ctx

            d_mean = float(stream_max_depth.to(torch.float32).mean().item())
            total_depth_sum += d_mean * B
            cur_md = int(stream_max_depth.max().item())
            if cur_md > max_depth_seen:
                max_depth_seen = cur_md

            # Hierarchical residual readout into 256-byte space (blazing fast!)
            h = torch.zeros((B, D), dtype=torch.float32, device=device)
            logits_sharp.zero_()
            n_vis_step = 0
            for v_ids, v_g in zip(visited_nodes, visited_gains):
                ok = v_ids >= 0
                if not ok.any():
                    continue
                n_vis_step += int(ok.sum().item())
                u_ok = v_ids[ok]
                h[ok] += self.W_node[u_ok] * v_g
                sd = self.sharp_dst[u_ok]
                sw = self.sharp_w[u_ok]
                sm = sd >= 0
                if sm.any():
                    b_ok = torch.nonzero(ok, as_tuple=False).squeeze(1)
                    b_rep = b_ok.unsqueeze(1).expand_as(sd)[sm]
                    d_rep = sd[sm].to(torch.int64)
                    w_rep = sw[sm] * (v_g * 0.65)
                    logits_sharp.index_put_((b_rep, d_rep), w_rep, accumulate=True)

            total_visited_sum += n_vis_step
            h_act = torch.tanh(h * 0.35) * 2.85
            logits = h_act @ self.E.T + self.bias.unsqueeze(0) + logits_sharp

            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]
            total_nll += float(nll.sum().item())

            top5 = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5[:, 0] == y_t).sum().item())
            total_top5 += int((top5 == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if not train:
                continue

            probs = torch.exp(log_p)
            p_true = probs[arange_B, y_t]
            surprise = 1.0 - p_true

            e_feat = self.E[y_t] - (probs @ self.E)
            sat = (1.0 - (h_act / 2.85).pow(2)).clamp(min=0.15)
            e_local = e_feat * sat

            top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
            top_bad = torch.where(top2_i[:, 0] != y_t, top2_i[:, 0], top2_i[:, 1])
            p_bad = torch.where(top2_i[:, 0] != y_t, top2_p[:, 0], top2_p[:, 1])

            if track_divorce and traversed_conv_edges:
                div_bkts_list: List[torch.Tensor] = []
                div_slots_list: List[torch.Tensor] = []
                for cb, cbkt, cslot, cnode in traversed_conv_edges:
                    c_surp = surprise[cb]
                    c_align = (self.W_node[cnode] * e_local[cb]).sum(dim=-1) / math.sqrt(D)
                    hurt = (c_surp > cfg.divorce_surprise_thresh) & (c_align < cfg.divorce_align_thresh)
                    help_ok = (c_surp < 0.30) & (c_align > 0.02)
                    if hurt.any():
                        hb, hs = cbkt[hurt], cslot[hurt]
                        self.edge_pain.index_put_((hb, hs), c_surp[hurt] * 0.65, accumulate=True)
                        over = self.edge_pain[hb, hs] >= cfg.divorce_pain_thresh
                        if over.any():
                            div_bkts_list.append(hb[over])
                            div_slots_list.append(hs[over])
                    if help_ok.any():
                        gb, gs = cbkt[help_ok], cslot[help_ok]
                        self.edge_pain[gb, gs] = (self.edge_pain[gb, gs] * 0.7).clamp(min=0.0)
                if div_bkts_list:
                    divorces_this_pass += self._divorce_edges(torch.cat(div_bkts_list), torch.cat(div_slots_list))

            for v_ids, v_g in zip(visited_nodes, visited_gains):
                ok = v_ids >= 0
                if not ok.any():
                    continue
                u_ok = v_ids[ok]
                ones_u = torch.ones(u_ok.numel(), dtype=torch.int32, device=device)
                self.fires.scatter_add_(0, u_ok, ones_u)
                self.recent_fires.scatter_add_(0, u_ok, ones_u)

                align = (self.W_node[u_ok] * e_local[ok]).sum(dim=-1) / math.sqrt(D)
                self.utility.scatter_add_(0, u_ok, torch.relu(align + 0.05) * 0.25)

                self.upd_cnt.scatter_add_(0, u_ok, torch.ones(u_ok.numel(), dtype=torch.float32, device=device))
                lr_u = (lr_feat * v_g) / torch.sqrt(1.0 + cfg.metaplasticity * self.upd_cnt[u_ok])
                self.W_node.index_add_(0, u_ok, e_local[ok] * lr_u.unsqueeze(1))

                self._update_sharp_synapses(u_ok, y_t[ok], surprise[ok], top_bad[ok], p_bad[ok], lr_sharp * v_g)

            bias_g = torch.zeros(V, dtype=torch.float32, device=device)
            bias_g.scatter_add_(0, y_t, torch.ones(B, dtype=torch.float32, device=device))
            bias_g.sub_(probs.sum(dim=0))
            self.bias.add_(bias_g * (lr_basis * 0.4 / B))
            step_e = lr_basis / B
            self.E.index_add_(0, y_t, h_act * step_e)
            self.E.addmm_(probs.T, h_act, alpha=-step_e)
            if t % 16 == 0:
                self.E.mul_(0.9995)

            high_surp = surprise > cfg.birth_surprise_thresh
            if high_surp.any():
                hs_b = torch.nonzero(high_surp, as_tuple=False).squeeze(1)
                leaves = deepest_leaf[hs_b]
                self.pain.scatter_add_(0, leaves, surprise[hs_b])

                can_split = (self.pain[leaves] >= cfg.birth_pain_thresh) & (self.depth[leaves] < cfg.max_hops)
                if can_split.any():
                    sb = hs_b[can_split]
                    sp = leaves[can_split]
                    l_span = self.local_span[sp].to(torch.int64)
                    next_d = (l_span + 1).clamp(max=H_buf - 1)
                    c_loc = byte_hist[sb, (cur_idx - next_d) % H_buf]

                    use_long = (
                        cfg.enable_long_branch
                        & (l_span >= 3)
                        & (ctx_score[sb].max(dim=1).values > 0.75)
                        & ((t % 4) == 0)
                    )
                    top_ctx = ctx_node[sb, ctx_score[sb].argmax(dim=1)]
                    kinds = torch.where(use_long, torch.ones_like(sp), torch.zeros_like(sp))
                    conds = torch.where(use_long, top_ctx, c_loc)
                    vb = torch.where(use_long, conds >= V, conds >= 0)
                    if vb.any():
                        births_this_pass += self._spawn_children(sp[vb], kinds[vb], conds[vb])

        avg_nll = total_nll / max(1, total_steps)
        bpb = avg_nll / math.log(2.0)
        bpc_equiv = bpb * bytes_per_char
        return {
            "bpb": round(bpb, 4),
            "bpc_lossless": round(bpc_equiv, 4),
            "ppl_char_equiv": round(2.0 ** min(bpc_equiv, 20.0), 2),
            "byte_top1": round(total_top1 / max(1, total_steps), 4),
            "byte_top5": round(total_top5 / max(1, total_steps), 4),
            "avg_depth_bytes": round(total_depth_sum / max(1, total_steps), 2),
            "avg_visited": round(total_visited_sum / max(1, total_steps), 2),
            "max_depth_bytes": max_depth_seen,
            "births": births_this_pass,
            "divorces": divorces_this_pass,
            "alive_hidden": self.num_hidden_alive,
            "edges": self.edges.count(),
        }

    @torch.no_grad()
    def generate_utf8(self, prompt: str, max_new_bytes: int = 180, temperature: float = 0.72, top_k: int = 24) -> Tuple[str, bool]:
        """Generate raw UTF-8 bytes directly from ANY prompt (Japanese, English, Emoji, Code — zero <UNK>!)."""
        cfg = self.cfg
        device = self.device
        raw_in = list(prompt.encode("utf-8"))
        H_buf = cfg.max_hops + 4
        byte_hist = torch.full((1, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0
        D_ctx = 12
        node_delay = torch.full((1, D_ctx), -1, dtype=torch.int64, device=device)
        nd_ptr = 0
        ctx_node = torch.full((1, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((1, 2), dtype=torch.float32, device=device)
        out_bytes = list(raw_in)

        def step_one(b_val: int, step_i: int) -> torch.Tensor:
            nonlocal hist_ptr, nd_ptr
            x_t = torch.tensor([b_val], dtype=torch.int64, device=device)
            byte_hist[0, hist_ptr] = b_val
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            ctx_score.mul_(0.985)
            if step_i >= D_ctx:
                u_old = int(node_delay[0, nd_ptr].item())
                if u_old >= self.V and bool(self.alive[u_old].item()):
                    sal = 1.8 / math.sqrt(1.0 + float(self.fires[u_old].item()) * 0.02)
                    if sal > 0.55 and u_old not in ctx_node[0].tolist():
                        ws = int(ctx_score[0].argmin().item())
                        if sal > float(ctx_score[0, ws].item()):
                            ctx_node[0, ws] = u_old
                            ctx_score[0, ws] = sal

            h = self.W_node[x_t[0]].clone()
            logits_s = torch.zeros(self.V, dtype=torch.float32, device=device)
            sd0, sw0 = self.sharp_dst[x_t[0]], self.sharp_w[x_t[0]]
            m0 = sd0 >= 0
            if m0.any():
                logits_s.index_add_(0, sd0[m0].to(torch.int64), sw0[m0] * 0.65)

            frontier = [b_val]
            emergent_cand = -1
            for hop in range(1, cfg.max_hops + 1):
                nxt: List[int] = []
                gain = 1.0 / (1.0 + 0.10 * hop)
                for p in frontier:
                    sp = int(self.local_span[p].item())
                    c_loc = int(byte_hist[0, (cur_idx - min(sp + 1, H_buf - 1)) % H_buf].item())
                    if c_loc >= 0:
                        ch = int(self.edges.lookup(
                            torch.tensor([p], device=device), torch.zeros(1, dtype=torch.int64, device=device), torch.tensor([c_loc], device=device)
                        ).item())
                        if ch >= self.V and bool(self.alive[ch].item()):
                            nxt.append(ch)
                            if 2 <= hop <= 4:
                                emergent_cand = ch
                    if cfg.enable_long_branch and 2 <= hop <= 6:
                        tc = int(ctx_node[0, int(ctx_score[0].argmax().item())].item())
                        if tc >= self.V:
                            chc = int(self.edges.lookup(
                                torch.tensor([p], device=device), torch.ones(1, dtype=torch.int64, device=device), torch.tensor([tc], device=device)
                            ).item())
                            if chc >= self.V and bool(self.alive[chc].item()):
                                nxt.append(chc)
                if not nxt:
                    break
                nxt = list(dict.fromkeys(nxt))[:4]
                for u in nxt:
                    h.add_(self.W_node[u], alpha=gain)
                    sdu, swu = self.sharp_dst[u], self.sharp_w[u]
                    mu = sdu >= 0
                    if mu.any():
                        logits_s.index_add_(0, sdu[mu].to(torch.int64), swu[mu] * (gain * 0.65))
                frontier = nxt

            node_delay[0, nd_ptr] = emergent_cand
            nd_ptr = (nd_ptr + 1) % D_ctx
            h_act = torch.tanh(h * 0.35) * 2.85
            return h_act @ self.E.T + self.bias + logits_s

        logits = None
        for i, b in enumerate(raw_in):
            logits = step_one(b, i)

        gen_only: List[int] = []
        for s in range(max_new_bytes):
            scaled = logits / max(temperature, 1e-4)
            tv, ti = torch.topk(scaled, k=min(top_k, self.V))
            probs = torch.softmax(tv, dim=-1)
            nxt_b = int(ti[int(torch.multinomial(probs, 1).item())].item())
            out_bytes.append(nxt_b)
            gen_only.append(nxt_b)
            logits = step_one(nxt_b, len(raw_in) + s)

        # Check whether generated bytes (trimming up to 2 incomplete trailing bytes at cutoff) form 100% VALID UTF-8!
        raw_gen = bytes(gen_only)
        valid_utf8 = False
        for trim in (0, 1, 2):
            sub = raw_gen[: len(raw_gen) - trim] if trim else raw_gen
            try:
                sub.decode("utf-8")
                valid_utf8 = True
                break
            except UnicodeDecodeError:
                continue
        text = bytes(out_bytes).decode("utf-8", errors="replace")
        return text, valid_utf8

    def sample_merged_clusters(self, top_k: int = 10) -> List[Dict[str, object]]:
        items = []
        for nid, aliases in self.merged_aliases.items():
            if len(aliases) >= 2 and bool(self.alive[nid].item()):
                items.append({
                    "node_id": nid,
                    "depth": int(self.depth[nid].item()),
                    "fires": int(self.fires[nid].item()),
                    "count": len(aliases),
                    "aliases": aliases,
                })
        items.sort(key=lambda x: (x["count"], x["fires"]), reverse=True)
        return items[:top_k]

    def sample_deep_neurons(self, min_depth: int = 6, top_k: int = 12) -> List[Dict[str, object]]:
        """Show the deepest organically crystallized multi-byte UTF-8 words/phrases!"""
        mask = self.alive & (torch.arange(self.P, device=self.device) >= self.V) & (self.depth >= min_depth) & (self.fires >= 15)
        ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if ids.numel() == 0:
            return []
        score = self.depth[ids].to(torch.float32) * 10.0 + torch.log1p(self.fires[ids].to(torch.float32))
        top_i = ids[torch.topk(score, k=min(top_k, int(ids.numel()))).indices].cpu().tolist()
        return [
            {"node_id": nid, "depth_bytes": int(self.depth[nid].item()), "fires": int(self.fires[nid].item()), "text": self.describe_byte_neuron(nid)}
            for nid in top_i
        ]
