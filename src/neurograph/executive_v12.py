"""bfloat16 の大規模 DAG、多バンク長期記憶、作業記憶とモード制御（v1.2）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from neurograph.growing_graph_v8 import DynamicEdgeTable, _edge_sig
from neurograph.memory_v3b import _mix64


class MegaByteLTMBanks:

    def __init__(self, device: torch.device):
        self.num_banks = 8
        self.bank_buckets = (
            524_288, 524_288, 524_288, 524_288,
            1_048_576, 1_048_576,
            2_097_152, 2_097_152,
        )
        self.bank_masks = tuple(b - 1 for b in self.bank_buckets)
        self.ways = 4
        self.device = device

        self.sig = [torch.zeros((b, self.ways), dtype=torch.int32, device=device) for b in self.bank_buckets]
        self.dst = [torch.full((b, self.ways), -1, dtype=torch.int16, device=device) for b in self.bank_buckets]
        self.w = [torch.zeros((b, self.ways), dtype=torch.float16, device=device) for b in self.bank_buckets]

        self.bank_scales = (0.28, 0.56, 0.95, 0.45, 1.30, 0.72, 1.65, 0.95)
        self.min_infer_w = (2.0, 2.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5)
        self.prune_thresh = (2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5)

    @torch.no_grad()
    def compute_keys(
        self,
        byte_hist: torch.Tensor,
        cur_idx: int,
        H_buf: int,
        top_ctx: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        bs = [byte_hist[:, (cur_idx - d) % H_buf] for d in range(18)]
        primes = (
            17, 1315423911, 2654435761, 1000003, 1000033, 1000037,
            1000039, 1000081, 1000099, 1000117, 1000121, 1000133,
            1000151, 1000159, 1000171, 1000183, 1000187, 1000193,
        )
        h = _mix64(bs[0] + primes[0])
        for d in range(1, 6):
            h = _mix64(h ^ ((bs[d] + 31 * d) * primes[d]))
        h6, v6 = h, (bs[5] >= 0)

        for d in range(6, 9):
            h = _mix64(h ^ ((bs[d] + 31 * d) * primes[d]))
        h9, v9 = h, (bs[8] >= 0)

        for d in range(9, 12):
            h = _mix64(h ^ ((bs[d] + 31 * d) * primes[d]))
        h12, v12 = h, (bs[11] >= 0)

        for d in range(12, 15):
            h = _mix64(h ^ ((bs[d] + 31 * d) * primes[d]))
        h15, v15 = h, (bs[14] >= 0)

        for d in range(15, 18):
            h = _mix64(h ^ ((bs[d] + 31 * d) * primes[d]))
        h18, v18 = h, (bs[17] >= 0)

        h_skip6 = _mix64(h6 ^ ((top_ctx + 997) * 2246822519))
        v_skip6 = v6 & (top_ctx >= 256)

        h_skip9 = _mix64(h9 ^ ((top_ctx + 1999) * 3266489917))
        v_skip9 = v9 & (top_ctx >= 256)

        h_skip12 = _mix64(h12 ^ ((top_ctx + 3001) * 1640531513))
        v_skip12 = v12 & (top_ctx >= 256)

        return (h6, h9, h12, h_skip6, h15, h_skip9, h18, h_skip12), (v6, v9, v12, v_skip6, v15, v_skip9, v18, v_skip12)

    @torch.no_grad()
    def lookup_boost(
        self,
        keys: Tuple[torch.Tensor, ...],
        valids: Tuple[torch.Tensor, ...],
        out_boost: torch.Tensor,
        max_boost: float = 3.2,
        ltm_scale: float = 0.70,
    ) -> int:
        out_boost.zero_()
        any_b_hit = torch.zeros(out_boost.shape[0], dtype=torch.bool, device=self.device)
        for bi in range(self.num_banks):
            k, v = keys[bi], valids[bi]
            if not v.any():
                continue
            bkts = (k & self.bank_masks[bi]).to(torch.int64)
            tag = ((k >> 24) & 0x7FFFFFFF).to(torch.int32)
            tag = torch.where(tag == 0, torch.ones_like(tag), tag)

            b_sig = self.sig[bi][bkts]
            b_dst = self.dst[bi][bkts]
            b_w = self.w[bi][bkts].to(torch.float32)

            hit = v.unsqueeze(1) & (b_sig == tag.unsqueeze(1)) & (b_dst >= 0) & (b_w >= self.min_infer_w[bi])
            if not hit.any():
                continue
            any_b_hit |= hit.any(dim=1)
            w_hit = torch.where(hit, b_w, torch.zeros_like(b_w))
            w_tot = w_hit.sum(dim=1, keepdim=True)
            ev = torch.tanh(torch.log1p(w_hit) * 0.35)
            contrib = (self.bank_scales[bi] * ltm_scale) * ev * (w_hit / (w_tot + 2.0))
            hb, hs = torch.nonzero(hit, as_tuple=True)
            out_boost.index_put_((hb, b_dst[hb, hs].to(torch.int64)), contrib[hb, hs], accumulate=True)

        out_boost.clamp_(max=max_boost)
        return int(any_b_hit.sum().item())

    @torch.no_grad()
    def update(
        self,
        keys: Tuple[torch.Tensor, ...],
        valids: Tuple[torch.Tensor, ...],
        y_t: torch.Tensor,
    ) -> None:
        y_i16 = y_t.to(torch.int16)
        for bi in range(self.num_banks):
            v = valids[bi]
            if not v.any():
                continue
            v_idx = torch.nonzero(v, as_tuple=False).squeeze(1)
            k = keys[bi][v_idx]
            yt = y_i16[v_idx]

            bkts = (k & self.bank_masks[bi]).to(torch.int64)
            tag = ((k >> 24) & 0x7FFFFFFF).to(torch.int32)
            tag = torch.where(tag == 0, torch.ones_like(tag), tag)

            order = torch.argsort(bkts, stable=True)
            sb = bkts[order]
            is_f = torch.ones(sb.numel(), dtype=torch.bool, device=self.device)
            if sb.numel() > 1:
                is_f[1:] = sb[1:] != sb[:-1]
            sel = order[is_f]

            ub, ut, uy = bkts[sel], tag[sel], yt[sel]
            b_sig = self.sig[bi][ub]
            b_dst = self.dst[bi][ub]
            b_w = self.w[bi][ub].to(torch.float32)

            exact = (b_sig == ut.unsqueeze(1)) & (b_dst == uy.unsqueeze(1))
            has_ex = exact.any(dim=1)
            if has_ex.any():
                eb = ub[has_ex]
                es = exact[has_ex].to(torch.int32).argmax(dim=1)
                self.w[bi][eb, es] = (self.w[bi][eb, es].to(torch.float32) + 1.0).clamp(max=4096.0).to(torch.float16)

            need = ~has_ex
            if need.any():
                nb, nt, ny = ub[need], ut[need], uy[need]
                ndst, nw = b_dst[need], b_w[need]
                emp = ndst < 0
                has_emp = emp.any(dim=1)
                emp_slot = emp.to(torch.int32).argmax(dim=1)
                min_w, min_slot = nw.min(dim=1)
                can_evict = has_emp | (min_w <= 1.0)
                if can_evict.any():
                    cb = nb[can_evict]
                    cs = torch.where(has_emp[can_evict], emp_slot[can_evict], min_slot[can_evict])
                    self.sig[bi][cb, cs] = nt[can_evict]
                    self.dst[bi][cb, cs] = ny[can_evict]
                    self.w[bi][cb, cs] = torch.ones(cb.numel(), dtype=torch.float16, device=self.device)

    @torch.no_grad()
    def sleep_prune(self) -> Dict[str, int]:
        before = 0
        after = 0
        for bi in range(self.num_banks):
            valid = self.dst[bi] >= 0
            before += int(valid.sum().item())
            kill = valid & (self.w[bi].to(torch.float32) < self.prune_thresh[bi])
            if kill.any():
                self.sig[bi][kill] = 0
                self.dst[bi][kill] = -1
                self.w[bi][kill] = 0
            after += int((self.dst[bi] >= 0).sum().item())
        pruned = before - after
        return {
            "before": before,
            "after": after,
            "pruned": pruned,
            "prune_pct": round(100.0 * pruned / max(1, before), 1),
        }

    def count_active(self) -> int:
        return sum(int((self.dst[bi] >= 0).sum().item()) for bi in range(self.num_banks))


@dataclass
class MegaBrainV12Config:
    vocab_size: int = 256
    pool_capacity: int = 5_000_000
    edge_buckets: int = 4_194_304
    edge_ways: int = 4
    feat_dim: int = 128
    max_hops: int = 16
    k_sharp: int = 8
    enable_long_branch: bool = True
    enable_apoptosis: bool = True
    enable_merge: bool = True
    enable_divorce: bool = True
    birth_surprise_thresh: float = 0.38
    birth_pain_thresh: float = 1.8
    max_births_per_step: int = 256
    min_fires_survive: int = 4
    min_utility_survive: float = 0.015
    myelin_fires_thresh: int = 18
    merge_min_src_fires: int = 4
    merge_max_src_fires: int = 96
    merge_min_lead_fires: int = 16
    merge_lead_ratio: float = 1.5
    merge_min_norm: float = 0.25
    merge_cos_thresh: float = 0.87
    merge_w_cos_thresh: float = 0.83
    merge_max_norm_ratio: float = 2.4
    merge_lsh_bits: int = 11
    merge_rounds: int = 2
    merge_min_depth: int = 2
    divorce_surprise_thresh: float = 0.45
    divorce_align_thresh: float = -0.02
    divorce_pain_thresh: float = 2.0
    max_divorces_per_step: int = 16
    lr_feat: float = 0.22
    lr_basis: float = 0.008
    lr_sharp: float = 0.28
    metaplasticity: float = 0.015
    enable_slots: bool = True
    num_slots: int = 8
    lr_slot_read: float = 0.045
    lr_slot_mod: float = 0.025
    lr_slot_gate: float = 0.015
    ltm_max_boost: float = 3.2
    ltm_scale: float = 0.70
    enable_stm: bool = True
    stm_slots: int = 192
    stm_surp_thresh: float = 0.36
    stm_w4: float = 1.15
    stm_w7: float = 2.35
    stm_decay: float = 0.9975
    stm_max_boost: float = 4.2
    enable_executive: bool = True
    num_modes: int = 24
    mode_ema_decay: float = 0.992
    mode_temp: float = 0.28
    mode_llr_scale: float = 0.42


class MegaUTF8BrainV12:

    def __init__(self, cfg: MegaBrainV12Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size
        self.P = cfg.pool_capacity
        self.D = cfg.feat_dim
        S = cfg.num_slots

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
        self.age = torch.zeros(self.P, dtype=torch.int16, device=device)
        self.upd_cnt = torch.zeros(self.P, dtype=torch.float32, device=device)

        self.edges = DynamicEdgeTable(cfg.edge_buckets, cfg.edge_ways, device)
        self.edge_parents = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int32, device=device)
        self.edge_kinds = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int8, device=device)
        self.edge_conds = torch.full((cfg.edge_buckets, cfg.edge_ways), -1, dtype=torch.int32, device=device)
        self.edge_pain = torch.zeros((cfg.edge_buckets, cfg.edge_ways), dtype=torch.bfloat16, device=device)

        self.in_degree = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.out_degree = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.divorce_immune = torch.zeros(self.P, dtype=torch.bool, device=device)

        self.E = torch.zeros((self.V, self.D), dtype=torch.float32, device=device)
        self.bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        self.W_node = torch.zeros((self.P, self.D), dtype=torch.bfloat16, device=device)
        self.sharp_dst = torch.full((self.P, cfg.k_sharp), -1, dtype=torch.int16, device=device)
        self.sharp_w = torch.zeros((self.P, cfg.k_sharp), dtype=torch.bfloat16, device=device)

        keep_priors = torch.tensor([0.68, 0.80, 0.88, 0.93, 0.96, 0.98, 0.990, 0.995], dtype=torch.float32, device=device)[:S]
        self.keep_bias = torch.log(keep_priors / (1.0 - keep_priors))
        self.target_write_rate = torch.tensor([0.24, 0.16, 0.11, 0.07, 0.045, 0.030, 0.020, 0.012], dtype=torch.float32, device=device)[:S]
        self.write_bias = torch.log(self.target_write_rate / (1.0 - self.target_write_rate))
        self.G_write = torch.zeros((S, self.D), dtype=torch.float32, device=device)
        self.G_keep = torch.zeros((S, self.D), dtype=torch.float32, device=device)
        gen = torch.Generator(device=device)
        gen.manual_seed(1337)
        self.W_read = torch.randn((S, self.D, self.D), generator=gen, device=device) * (0.055 / math.sqrt(self.D))
        self.W_mod = torch.randn((S, self.D, self.D), generator=gen, device=device) * (0.035 / math.sqrt(self.D))
        self.rms_read = torch.full((S,), 1e-3, dtype=torch.float32, device=device)
        self.rms_mod = torch.full((S,), 1e-3, dtype=torch.float32, device=device)

        self.ltm = MegaByteLTMBanks(device)

        M = cfg.num_modes
        self.mode_centroids = torch.zeros((M, self.D), dtype=torch.float32, device=device)
        self.mode_llr = torch.zeros((M, self.V), dtype=torch.float32, device=device)
        self.mode_ready = False

        self.total_births = 0
        self.total_deaths = 0
        self.total_merges = 0
        self.total_divorces = 0

    @property
    def num_hidden_alive(self) -> int:
        return int(self.alive[self.V :].sum().item())

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
    def _lookup_with_loc(self, parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor):
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.edges.mask).to(torch.int64)
        b_sig = self.edges.sig[bkts]
        b_ch = self.edges.child[bkts]
        hit = (b_sig == sigs.unsqueeze(1)) & (b_ch >= 0)
        has = hit.any(dim=1)
        slot = hit.to(torch.int32).argmax(dim=1).to(torch.int64)
        found = b_ch[torch.arange(parent.numel(), device=self.device), slot].to(torch.int64)
        return torch.where(has, found, torch.full_like(found, -1)), bkts, slot

    @torch.no_grad()
    def _insert_edges_tracked(self, parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor, child_id: torch.Tensor) -> torch.Tensor:
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
    def transplant_from_v11_checkpoint(self, ckpt_path: str) -> Dict[str, int]:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        old_P = ck["alive"].shape[0]
        assert old_P <= self.P

        self.alive[:old_P].copy_(ck["alive"].to(self.device))
        self.parent[:old_P].copy_(ck["parent"].to(self.device))
        self.cond_kind[:old_P].copy_(ck["cond_kind"].to(self.device))
        self.cond_id[:old_P].copy_(ck["cond_id"].to(self.device))
        self.depth[:old_P].copy_(ck["depth"].to(self.device))
        self.local_span[:old_P].copy_(ck["local_span"].to(self.device))
        self.fires[:old_P].copy_(ck["fires"].to(self.device))
        self.upd_cnt[:old_P].copy_(ck["upd_cnt"].to(self.device) if "upd_cnt" in ck else ck["fires"].to(torch.float32).to(self.device))
        self.age[:old_P].copy_(torch.where(self.alive[:old_P], torch.full((old_P,), 3, dtype=torch.int16, device=self.device), torch.zeros(old_P, dtype=torch.int16, device=self.device)))
        self.utility[:old_P].copy_(torch.where(self.alive[:old_P], torch.full((old_P,), 0.25, dtype=torch.float32, device=self.device), torch.zeros(old_P, dtype=torch.float32, device=self.device)))

        self.E.copy_(ck["E"].to(self.device))
        self.bias.copy_(ck["bias"].to(self.device))
        self.W_node[:old_P].copy_(ck["W_node"].to(torch.bfloat16).to(self.device))
        self.sharp_dst[:old_P].copy_(ck["sharp_dst"].to(torch.int16).to(self.device))
        self.sharp_w[:old_P].copy_(ck["sharp_w"].to(torch.bfloat16).to(self.device))

        self.G_write.copy_(ck["G_write"].to(self.device))
        self.G_keep.copy_(ck["G_keep"].to(self.device))
        self.write_bias.copy_(ck["write_bias"].to(self.device))
        self.keep_bias.copy_(ck["keep_bias"].to(self.device))
        self.W_read.copy_(ck["W_read"].to(self.device))
        self.W_mod.copy_(ck["W_mod"].to(self.device))

        idx_all = torch.arange(old_P, device=self.device)
        alive_h = self.alive[:old_P] & (idx_all >= self.V) & (self.parent[:old_P] >= 0) & (self.cond_kind[:old_P] >= 0)
        h_ids = torch.nonzero(alive_h, as_tuple=False).squeeze(1)
        pa = self.parent[h_ids].to(torch.int64)
        kd = self.cond_kind[h_ids].to(torch.int64)
        cd = self.cond_id[h_ids].to(torch.int64)
        ok_p = self.alive[pa]
        h_ids, pa, kd, cd = h_ids[ok_p], pa[ok_p], kd[ok_p], cd[ok_p]

        rem = torch.ones(h_ids.numel(), dtype=torch.bool, device=self.device)
        for _ in range(self.edges.ways * 2):
            if not rem.any():
                break
            r_idx = torch.nonzero(rem, as_tuple=False).squeeze(1)
            ins = self._insert_edges_tracked(pa[r_idx], kd[r_idx], cd[r_idx], h_ids[r_idx])
            if not ins.any():
                break
            rem[r_idx[ins]] = False
        self._recompute_degrees()

        n_ck_banks = len(ck["ltm_sig"])
        for bi in range(min(n_ck_banks, self.ltm.num_banks)):
            self.ltm.sig[bi].copy_(ck["ltm_sig"][bi].to(self.device))
            self.ltm.dst[bi].copy_(ck["ltm_dst"][bi].to(self.device))
            self.ltm.w[bi].copy_(ck["ltm_w"][bi].to(self.device))

        del ck
        torch.cuda.empty_cache()
        return {
            "transplanted_neurons": self.num_hidden_alive,
            "transplanted_edges": self.edges.count(),
            "transplanted_ltm": self.ltm.count_active(),
            "new_pool_capacity": self.P,
            "new_edge_capacity": self.edges.num_buckets * self.edges.ways,
            "new_ltm_capacity": sum(b * self.ltm.ways for b in self.ltm.bank_buckets),
        }

    @torch.no_grad()
    def _allocate_free_ids(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if not hasattr(self, "_alloc_cursor"):
            self._alloc_cursor = self.V
        chunk = 131_072
        found_list = []
        need = count
        for _ in range((self.P // chunk) + 2):
            s = self._alloc_cursor
            e = min(self.P, s + chunk)
            if s < e:
                free_c = torch.nonzero(~self.alive[s:e], as_tuple=False).squeeze(1) + s
                if free_c.numel() > 0:
                    take = min(need, int(free_c.numel()))
                    found_list.append(free_c[:take])
                    need -= take
                    if need == 0:
                        self._alloc_cursor = int(free_c[take - 1].item()) + 1
                        if self._alloc_cursor >= self.P:
                            self._alloc_cursor = self.V
                        break
            self._alloc_cursor = e if e < self.P else self.V
        if not found_list:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.cat(found_list) if len(found_list) > 1 else found_list[0]

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
        self.W_node[born] = 0
        self.upd_cnt[born] = 0.0
        self.sharp_dst[born] = -1
        self.sharp_w[born] = 0
        self.divorce_immune[born] = False

        self.pain[bp] *= 0.35
        n_born = int(born.numel())
        self.total_births += n_born
        return n_born

    @torch.no_grad()
    def _divorce_edges(self, bkts: torch.Tensor, slots: torch.Tensor) -> int:
        if bkts.numel() == 0:
            return 0
        uniq_lin = torch.unique(bkts * self.edges.ways + slots)
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
        self.W_node[new_ids] = (self.W_node[hub].to(torch.float32) * 0.75).to(torch.bfloat16)
        self.sharp_dst[new_ids] = self.sharp_dst[hub]
        self.sharp_w[new_ids] = (self.sharp_w[hub].to(torch.float32) * 0.70).to(torch.bfloat16)
        self.upd_cnt[new_ids] = 4.0
        self.fires[new_ids] = 4
        self.recent_fires[new_ids] = 4
        self.utility[new_ids] = 0.10
        self.pain[new_ids] = 0.0
        self.age[new_ids] = 0
        self.divorce_immune[new_ids] = True

        self.edges.child[u_bkt, u_slot] = new_ids.to(torch.int32)
        self.edge_pain[u_bkt, u_slot] = 0
        ones = torch.ones(n_alloc, dtype=torch.int32, device=self.device)
        self.in_degree.scatter_add_(0, hub, -ones)
        self.in_degree.clamp_(min=0)
        self.in_degree[new_ids] = 1
        self.out_degree[new_ids] = 0
        self.total_divorces += n_alloc
        return n_alloc

    @torch.no_grad()
    def _update_sharp_batched(self, nodes, y_t, surp, top_bad, p_bad, lr_vec) -> None:
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
        yt = y_t[idx].to(torch.int16)
        sp = surp[idx]
        tb = top_bad[idx].to(torch.int16)
        pb = p_bad[idx]
        lr = lr_vec[idx]

        dst = self.sharp_dst[u]
        w = self.sharp_w[u].to(torch.float32)
        hit = dst == yt.unsqueeze(1)
        has = hit.any(dim=1)
        if has.any():
            hu = u[has]
            hs = hit[has].to(torch.int32).argmax(dim=1)
            self.sharp_w[hu, hs] = (self.sharp_w[hu, hs].to(torch.float32) + lr[has] * sp[has]).clamp(-3.5, 4.0).to(torch.bfloat16)

        need = (~has) & (sp > 0.25)
        if need.any():
            nu = u[need]
            nd = dst[need]
            nw = w[need]
            emp = nd < 0
            slot = torch.where(emp.any(dim=1), emp.to(torch.int32).argmax(dim=1), nw.abs().argmin(dim=1))
            self.sharp_dst[nu, slot] = yt[need]
            self.sharp_w[nu, slot] = (lr[need] * sp[need]).clamp(0.05, 1.5).to(torch.bfloat16)

        bad_valid = (tb != yt) & (pb > 0.08)
        if bad_valid.any():
            bu, bt, bp, blr = u[bad_valid], tb[bad_valid], pb[bad_valid], lr[bad_valid]
            bhit = self.sharp_dst[bu] == bt.unsqueeze(1)
            bhas = bhit.any(dim=1)
            if bhas.any():
                bhu = bu[bhas]
                bhs = bhit[bhas].to(torch.int32).argmax(dim=1)
                self.sharp_w[bhu, bhs] = (self.sharp_w[bhu, bhs].to(torch.float32) - (blr[bhas] * 0.8) * bp[bhas]).clamp(-3.5, 4.0).to(torch.bfloat16)

    @torch.no_grad()
    def apoptosis(self) -> Dict[str, int]:
        if not self.cfg.enable_apoptosis:
            return {"died": 0, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

        hidden_alive = self.alive.clone()
        hidden_alive[: self.V] = False
        self.age[hidden_alive] = (self.age[hidden_alive] + 1).clamp(max=30000)

        myelinated = self.fires >= self.cfg.myelin_fires_thresh
        cand_mask = hidden_alive & (~myelinated) & (self.age >= 1)
        dead = torch.zeros(self.P, dtype=torch.bool, device=self.device)
        cand_ids = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
        if cand_ids.numel() > 0:
            w_norm = self.W_node[cand_ids].to(torch.float32).norm(dim=1)
            sharp_norm = self.sharp_w[cand_ids].to(torch.float32).abs().sum(dim=1)
            weak = (self.recent_fires[cand_ids] < self.cfg.min_fires_survive) | (
                (self.utility[cand_ids] < self.cfg.min_utility_survive) & (w_norm < 0.15) & (sharp_norm < 0.15)
            )
            dead[cand_ids[weak]] = True

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
                self.edge_pain[kill_e] = 0

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
                    self.edge_pain[kill_e2] = 0

        n_died = int(dead.sum().item())
        if n_died > 0:
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
            self.W_node[dead] = 0
            self.upd_cnt[dead] = 0.0
            self.sharp_dst[dead] = -1
            self.sharp_w[dead] = 0
            self.divorce_immune[dead] = False
            self.total_deaths += n_died

        self.recent_fires.zero_()
        self.utility.mul_(0.85)
        self._recompute_degrees()
        return {"died": n_died, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

    @torch.no_grad()
    def _build_full_pred_signature(self, node_ids: torch.Tensor, chunk_size: int = 32_768) -> torch.Tensor:
        M = node_ids.numel()
        P = self.W_node[node_ids].to(torch.float32)
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            ids_c = node_ids[start:end]
            sd = self.sharp_dst[ids_c]
            sw = self.sharp_w[ids_c].to(torch.float32)
            valid = (sd >= 0) & (sw.abs() > 1e-4)
            if valid.any():
                e_sd = self.E[sd.clamp(min=0).to(torch.int64)]
                w_m = torch.where(valid, sw, torch.zeros_like(sw)).unsqueeze(-1)
                P[start:end].add_((e_sd * w_m).sum(dim=1), alpha=0.55)
        return P

    @torch.no_grad()
    def fuse_similar_neurons(self, seed: int = 100) -> Dict[str, int]:
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
            base_mask = self.alive & (idx_all >= self.V) & (self.depth >= cfg.merge_min_depth) & (redirect == idx_all)
            is_src_elig = (
                base_mask & (~self.divorce_immune)
                & (self.fires >= cfg.merge_min_src_fires) & (self.fires <= cfg.merge_max_src_fires)
                & (self.out_degree == 0)
            )
            is_lead_elig = base_mask & (self.fires >= cfg.merge_min_lead_fires)
            cand_ids = torch.nonzero(is_src_elig | is_lead_elig, as_tuple=False).squeeze(1)
            if cand_ids.numel() < 2:
                break

            w_c = self.W_node[cand_ids].to(torch.float32)
            w_c_norm = w_c.norm(dim=1)
            nz = w_c_norm >= cfg.merge_min_norm
            cand_ids, w_c, w_c_norm = cand_ids[nz], w_c[nz], w_c_norm[nz]
            M = cand_ids.numel()
            if M < 2:
                break

            P_sig = self._build_full_pred_signature(cand_ids)
            U_p = P_sig / P_sig.norm(dim=1, keepdim=True).clamp(min=1e-6)
            U_w = w_c / w_c_norm.unsqueeze(1)
            del P_sig, w_c

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
                not_leader & is_src_elig[s_ids] & is_lead_elig[lead_ids]
                & (s_fires[leader_pos].to(torch.float32) >= s_fires.to(torch.float32) * cfg.merge_lead_ratio)
            )
            cos_p = (s_Up * s_Up[leader_pos]).sum(dim=1)
            cos_w = (s_Uw * s_Uw[leader_pos]).sum(dim=1)
            norm_ratio = s_wn[leader_pos] / s_wn.clamp(min=1e-4)
            norm_ok = (norm_ratio >= 0.45) & (norm_ratio <= cfg.merge_max_norm_ratio)

            w_st = self.sharp_w[s_ids].to(torch.float32).argmax(dim=1)
            w_lt = self.sharp_w[lead_ids].to(torch.float32).argmax(dim=1)
            st_dst = self.sharp_dst[s_ids, w_st]
            lt_dst = self.sharp_dst[lead_ids, w_lt]
            st_w = self.sharp_w[s_ids, w_st].to(torch.float32)
            lt_w = self.sharp_w[lead_ids, w_lt].to(torch.float32)
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

            f_src = self.fires[src_m].to(torch.float32)
            u_dst_uniq, inv_dst = torch.unique(dst_m, return_inverse=True)
            w_sum_u = self.W_node[u_dst_uniq].to(torch.float32) * self.fires[u_dst_uniq].to(torch.float32).unsqueeze(1)
            f_sum_u = self.fires[u_dst_uniq].to(torch.float32)
            w_sum_u.index_add_(0, inv_dst, self.W_node[src_m].to(torch.float32) * f_src.unsqueeze(1))
            f_sum_u.scatter_add_(0, inv_dst, f_src)
            self.W_node[u_dst_uniq] = (w_sum_u / f_sum_u.clamp(min=1.0).unsqueeze(1)).to(torch.bfloat16)

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
            self.W_node[src_m] = 0
            self.upd_cnt[src_m] = 0.0
            self.sharp_dst[src_m] = -1
            self.sharp_w[src_m] = 0
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

        torch.cuda.empty_cache()
        return {"merged": total_merged_now, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

    @torch.no_grad()
    def _compute_stm_keys(self, byte_hist: torch.Tensor, cur_idx: int, H_buf: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b0 = byte_hist[:, cur_idx]
        b1 = byte_hist[:, (cur_idx - 1) % H_buf]
        b2 = byte_hist[:, (cur_idx - 2) % H_buf]
        b3 = byte_hist[:, (cur_idx - 3) % H_buf]
        b4 = byte_hist[:, (cur_idx - 4) % H_buf]
        b5 = byte_hist[:, (cur_idx - 5) % H_buf]
        b6 = byte_hist[:, (cur_idx - 6) % H_buf]

        h4 = _mix64(b0 + 17)
        h4 = _mix64(h4 ^ ((b1 + 31) * 1315423911))
        h4 = _mix64(h4 ^ ((b2 + 53) * 2654435761))
        h4 = _mix64(h4 ^ ((b3 + 79) * 1000003))
        h4 = torch.where((b3 >= 0) & (h4 != 0), h4, torch.zeros_like(h4))

        h7 = _mix64(h4 ^ ((b4 + 97) * 1000033))
        h7 = _mix64(h7 ^ ((b5 + 113) * 1000037))
        h7 = _mix64(h7 ^ ((b6 + 131) * 1000039))
        h7 = torch.where((b6 >= 0) & (h7 != 0), h7, torch.zeros_like(h7))
        return h4, h7

    @torch.no_grad()
    def fit_executive_modes(self, mixed_bytes: np.ndarray, passage_len: int = 320, max_passages: int = 12_000) -> Dict[str, object]:
        cfg = self.cfg
        device = self.device
        M, D, V = cfg.num_modes, self.D, self.V

        n_pass = min(max_passages, len(mixed_bytes) // passage_len)
        B = min(1024, n_pass)
        n_batches = n_pass // B
        pass_vecs = []
        pass_byte_counts = []

        H_buf = max(cfg.max_hops + 4, 22)
        for bi in range(n_batches):
            seg = mixed_bytes[bi * B * passage_len : (bi + 1) * B * passage_len].reshape(B, passage_len)
            data = torch.from_numpy(seg.astype(np.int64)).to(device)
            byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
            hist_ptr = 0
            h_sum = torch.zeros((B, D), dtype=torch.float32, device=device)
            b_cnt = torch.zeros((B, V), dtype=torch.float32, device=device)
            arange_B = torch.arange(B, device=device)
            ones_B = torch.ones(B, dtype=torch.float32, device=device)

            for t in range(passage_len - 1):
                x_t = data[:, t]
                y_t = data[:, t + 1]
                byte_hist[:, hist_ptr] = x_t
                cur_idx = hist_ptr
                hist_ptr = (hist_ptr + 1) % H_buf

                h = self.W_node[x_t].to(torch.float32)
                frontier = x_t
                for hop in range(1, 9):
                    c_loc = byte_hist[arange_B, (cur_idx - hop) % H_buf]
                    ch = self.edges.lookup(frontier, torch.zeros_like(frontier), c_loc)
                    ok = (c_loc >= 0) & (ch >= V) & self.alive[ch.clamp(min=0)]
                    if not ok.any():
                        break
                    ok_i = torch.nonzero(ok, as_tuple=False).squeeze(1)
                    h[ok_i] += self.W_node[ch[ok_i]].to(torch.float32) * (1.0 / (1.0 + 0.10 * hop))
                    frontier = torch.where(ok, ch, torch.full_like(ch, -1))
                h_sum += h
                b_cnt.scatter_add_(1, y_t.unsqueeze(1), ones_B.unsqueeze(1))

            p_v = h_sum / h_sum.norm(dim=1, keepdim=True).clamp(min=1e-6)
            pass_vecs.append(p_v)
            pass_byte_counts.append(b_cnt)

        X = torch.cat(pass_vecs, dim=0)
        C_mat = torch.cat(pass_byte_counts, 0)
        N_p = X.shape[0]

        gen = torch.Generator(device=device)
        gen.manual_seed(77)
        cents = torch.empty((M, D), dtype=torch.float32, device=device)
        cents[0] = X[0]
        md = (1.0 - (X @ cents[0])).clamp(min=0.0)
        for mi in range(1, M):
            pr = md.clamp(min=1e-4).pow(2.0)
            pr = pr / pr.sum().clamp(min=1e-8)
            idx = int(torch.multinomial(pr, 1, generator=gen).item())
            cents[mi] = X[idx]
            md = torch.minimum(md, (1.0 - (X @ cents[mi])).clamp(min=0.0))

        assign = torch.zeros(N_p, dtype=torch.int64, device=device)
        for _ in range(25):
            assign = (X @ cents.T).argmax(dim=1)
            nc = torch.zeros_like(cents)
            nc.index_add_(0, assign, X)
            nrm = nc.norm(dim=1, keepdim=True)
            ok = nrm.squeeze(1) > 1e-5
            cents[ok] = nc[ok] / nrm[ok]

        self.mode_centroids.copy_(cents)

        global_c = C_mat.sum(dim=0) + 1.0
        p_glob = global_c / global_c.sum()
        mode_c = torch.zeros((M, V), dtype=torch.float32, device=device)
        mode_c.index_add_(0, assign, C_mat)
        alpha = 60.0
        p_mode = (mode_c + alpha * p_glob.unsqueeze(0)) / (mode_c.sum(dim=1, keepdim=True) + alpha)
        llr = torch.log(p_mode / p_glob.unsqueeze(0)).clamp(-1.4, 1.4)
        self.mode_llr.copy_(llr)
        self.mode_ready = True

        sizes = torch.bincount(assign, minlength=M).cpu().tolist()
        return {"num_passages": N_p, "num_modes": M, "min_mode_sz": min(sizes), "max_mode_sz": max(sizes)}

    @torch.no_grad()
    def run_stream(
        self,
        byte_arr: np.ndarray,
        train: bool = True,
        batch_streams: int = 4096,
        lr_scale: float = 1.0,
        bytes_per_char: float = 2.88,
        use_slots: Optional[bool] = None,
        use_ltm: Optional[bool] = None,
        use_stm: Optional[bool] = None,
        use_executive: Optional[bool] = None,
        populate_ltm: bool = True,
    ) -> Dict[str, float]:
        cfg = self.cfg
        device = self.device
        V, D, S = self.V, self.D, cfg.num_slots
        self._recompute_degrees()

        slots_on = cfg.enable_slots if use_slots is None else use_slots
        ltm_infer = (not train) if use_ltm is None else use_ltm
        ltm_write = train and populate_ltm
        stm_on = (cfg.enable_stm and (not train)) if use_stm is None else use_stm
        exec_on = (cfg.enable_executive and self.mode_ready and (not train)) if use_executive is None else (use_executive and self.mode_ready)

        total_len = len(byte_arr)
        stream_len = total_len // batch_streams
        if stream_len < 64:
            batch_streams = max(1, total_len // 64)
            stream_len = total_len // batch_streams

        data = torch.from_numpy(
            byte_arr[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(device)
        B, T = data.shape

        H_buf = max(cfg.max_hops + 4, 22)
        byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0

        D_ctx = 12
        node_delay = torch.full((B, D_ctx), -1, dtype=torch.int64, device=device)
        nd_ptr = 0
        ctx_node = torch.full((B, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((B, 2), dtype=torch.float32, device=device)

        slot_state = torch.zeros((B, S, D), dtype=torch.float32, device=device)
        slot_last_x = torch.zeros((B, S, D), dtype=torch.float32, device=device)

        exec_vec = torch.zeros((B, D), dtype=torch.float32, device=device) if exec_on else None

        ltm_boost = torch.zeros((B, V), dtype=torch.float32, device=device) if ltm_infer else None

        N_stm = cfg.stm_slots
        if stm_on:
            stm_k4 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
            stm_k7 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
            stm_val = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
            stm_str = torch.zeros((B, N_stm), dtype=torch.float32, device=device)
            stm_ptr = torch.zeros(B, dtype=torch.int64, device=device)
            stm_boost = torch.zeros((B, V), dtype=torch.float32, device=device)

        arange_B = torch.arange(B, device=device)
        ones_B_f = torch.ones(B, dtype=torch.float32, device=device)
        lr_feat = cfg.lr_feat * lr_scale
        lr_basis = cfg.lr_basis * lr_scale
        lr_sharp = cfg.lr_sharp * lr_scale
        lr_sr = cfg.lr_slot_read * lr_scale
        lr_sm = cfg.lr_slot_mod * lr_scale
        lr_sg = cfg.lr_slot_gate * lr_scale

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0
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

            ctx_score.mul_(0.985)
            if t >= D_ctx:
                u_old = node_delay[:, nd_ptr]
                ok_u = (u_old >= V) & self.alive[u_old.clamp(min=0)]
                if ok_u.any():
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

            top_ctx_all = ctx_node[arange_B, ctx_score.argmax(dim=1)]

            vis_b_list: List[torch.Tensor] = [arange_B]
            vis_u_list: List[torch.Tensor] = [x_t]
            vis_g_list: List[torch.Tensor] = [ones_B_f]
            traversed_conv_edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

            frontier = x_t.unsqueeze(1)
            deepest_leaf = x_t.clone()
            deepest_span = torch.zeros(B, dtype=torch.int64, device=device)
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
                if cfg.enable_long_branch and 2 <= hop <= 5:
                    top_ctx = top_ctx_all[fb]
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
                gain_val = 1.0 / (1.0 + 0.10 * hop)

                hit_loc = ch_loc >= 0
                if hit_loc.any():
                    hl_b = fb[hit_loc]
                    hl_c = ch_loc[hit_loc]
                    hl_slot = (ff[hit_loc] * 2) % MAX_FRONT
                    next_front[hl_b, hl_slot] = hl_c
                    vis_b_list.append(hl_b)
                    vis_u_list.append(hl_c)
                    vis_g_list.append(torch.full((hl_b.numel(),), gain_val, dtype=torch.float32, device=device))

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
                    vis_b_list.append(hc_b)
                    vis_u_list.append(hc_c)
                    vis_g_list.append(torch.full((hc_b.numel(),), gain_val, dtype=torch.float32, device=device))
                    if track_divorce and ctx_bkt is not None:
                        is_conv_c = self.in_degree[hc_c] > 1
                        if is_conv_c.any():
                            traversed_conv_edges.append(
                                (hc_b[is_conv_c], ctx_bkt[hit_ctx][is_conv_c], ctx_slot[hit_ctx][is_conv_c], hc_c[is_conv_c])
                            )

                if not (hit_loc.any() or hit_ctx.any()):
                    break
                frontier = next_front

            node_delay[:, nd_ptr] = emergent_ctx_cand
            nd_ptr = (nd_ptr + 1) % D_ctx

            all_b = torch.cat(vis_b_list)
            all_u = torch.cat(vis_u_list)
            all_g = torch.cat(vis_g_list)

            h_dag = torch.zeros((B, D), dtype=torch.float32, device=device)
            h_dag.index_add_(0, all_b, self.W_node[all_u].to(torch.float32) * all_g.unsqueeze(1))

            logits_sharp.zero_()
            sd_all = self.sharp_dst[all_u]
            sw_all = self.sharp_w[all_u].to(torch.float32)
            sm_all = sd_all >= 0
            if sm_all.any():
                b_rep = all_b.unsqueeze(1).expand_as(sd_all)[sm_all]
                d_rep = sd_all[sm_all].to(torch.int64)
                w_rep = (sw_all * (all_g * 0.65).unsqueeze(1))[sm_all]
                logits_sharp.index_put_((b_rep, d_rep), w_rep, accumulate=True)

            if slots_on:
                h_norm = h_dag / h_dag.norm(dim=1, keepdim=True).clamp(min=1.0)
                e_cur = self.E[x_t]
                e_norm = e_cur / e_cur.norm(dim=1, keepdim=True).clamp(min=1.0)
                x_slot = 0.65 * h_norm + 0.35 * e_norm

                k_gate = torch.sigmoid(x_slot @ self.G_keep.T + self.keep_bias.unsqueeze(0))
                w_gate = torch.sigmoid(x_slot @ self.G_write.T + self.write_bias.unsqueeze(0))

                slot_state = slot_state * k_gate.unsqueeze(-1) + w_gate.unsqueeze(-1) * x_slot.unsqueeze(1)
                s_nrm = slot_state.norm(dim=-1, keepdim=True)
                slot_state = slot_state * (2.5 / s_nrm.clamp(min=2.5))
                slot_last_x = slot_last_x * (1.0 - w_gate).unsqueeze(-1) + w_gate.unsqueeze(-1) * x_slot.unsqueeze(1)

                r_each = torch.einsum("bsd,sdk->bsk", slot_state, self.W_read)
                m_each = torch.einsum("bsd,sdk->bsk", slot_state, self.W_mod)
                h_total = h_dag * (1.0 + 0.35 * torch.tanh(m_each.sum(dim=1))) + r_each.sum(dim=1)
            else:
                h_total = h_dag

            h_act = torch.tanh(h_total * 0.35) * 2.85
            logits = h_act @ self.E.T + self.bias.unsqueeze(0) + logits_sharp

            if ltm_infer or ltm_write:
                ltm_keys, ltm_valids = self.ltm.compute_keys(byte_hist, cur_idx, H_buf, top_ctx_all)
                if ltm_infer and ltm_boost is not None:
                    self.ltm.lookup_boost(ltm_keys, ltm_valids, ltm_boost, max_boost=cfg.ltm_max_boost, ltm_scale=cfg.ltm_scale)
                    logits = logits + ltm_boost
                if ltm_write:
                    self.ltm.update(ltm_keys, ltm_valids, y_t)

            if stm_on:
                k4, k7 = self._compute_stm_keys(byte_hist, cur_idx, H_buf)
                stm_str.mul_(cfg.stm_decay)
                if t >= 4:
                    m4 = (stm_k4 == k4.unsqueeze(1)) & (k4.unsqueeze(1) != 0) & (stm_str > 0.05)
                    m7 = (stm_k7 == k7.unsqueeze(1)) & (k7.unsqueeze(1) != 0) & (stm_str > 0.05)
                    hit_w = m4.to(torch.float32) * (stm_str * cfg.stm_w4) + m7.to(torch.float32) * (stm_str * cfg.stm_w7)
                    any_hit = hit_w > 0.0
                    if any_hit.any():
                        hb, hs = torch.nonzero(any_hit, as_tuple=True)
                        stm_boost.zero_()
                        stm_boost.index_put_((hb, stm_val[hb, hs]), hit_w[hb, hs], accumulate=True)
                        stm_boost.clamp_(max=cfg.stm_max_boost)
                        logits = logits + stm_boost

            if exec_on and exec_vec is not None:
                exec_vec.mul_(cfg.mode_ema_decay).add_(h_dag, alpha=1.0 - cfg.mode_ema_decay)
                if t >= 24:
                    ev_n = exec_vec / exec_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    mode_p = torch.softmax((ev_n @ self.mode_centroids.T) / cfg.mode_temp, dim=-1)
                    logits = logits + (mode_p @ self.mode_llr) * cfg.mode_llr_scale

            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]
            total_nll += float(nll.sum().item())

            top5 = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5[:, 0] == y_t).sum().item())
            total_top5 += int((top5 == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if stm_on and t >= 4:
                surp_stm = 1.0 - torch.exp(-nll)
                do_bind = (surp_stm > cfg.stm_surp_thresh) & (k4 != 0)
                if do_bind.any():
                    bb = torch.nonzero(do_bind, as_tuple=False).squeeze(1)
                    bk4, bk7, by = k4[bb], k7[bb], y_t[bb]
                    exists = (stm_k4[bb] == bk4.unsqueeze(1)) & (stm_val[bb] == by.unsqueeze(1)) & (stm_str[bb] > 0.05)
                    has_ex = exists.any(dim=1)
                    ex_slot = exists.to(torch.int32).argmax(dim=1).to(torch.int64)
                    w_slot = torch.where(has_ex, ex_slot, stm_ptr[bb])
                    stm_k4[bb, w_slot] = bk4
                    stm_k7[bb, w_slot] = bk7
                    stm_val[bb, w_slot] = by
                    stm_str[bb, w_slot] = torch.where(has_ex, (stm_str[bb, w_slot] + 0.65).clamp(max=2.0), torch.ones_like(surp_stm[bb]))
                    stm_ptr[bb] = torch.where(has_ex, stm_ptr[bb], (stm_ptr[bb] + 1) % N_stm)

            if not train:
                continue

            probs = torch.exp(log_p)
            surprise = 1.0 - probs[arange_B, y_t]
            e_feat = self.E[y_t] - (probs @ self.E)
            sat = (1.0 - (h_act / 2.85).pow(2)).clamp(min=0.15)
            e_local = e_feat * sat

            if slots_on:
                g_read = torch.einsum("bsd,bk->sdk", slot_state, e_local) / B
                self.rms_read.mul_(0.98).add_(g_read.pow(2).mean(dim=(1, 2)).clamp(min=1e-8) * 0.02)
                self.W_read.add_(g_read * (lr_sr / torch.sqrt(self.rms_read + 1e-6)).view(S, 1, 1))
                if t % 16 == 0:
                    self.W_read.mul_(0.9994)

                e_mod = e_local * h_dag.clamp(-2.5, 2.5) * 0.35
                g_mod = torch.einsum("bsd,bk->sdk", slot_state, e_mod) / B
                self.rms_mod.mul_(0.98).add_(g_mod.pow(2).mean(dim=(1, 2)).clamp(min=1e-8) * 0.02)
                self.W_mod.add_(g_mod * (lr_sm / torch.sqrt(self.rms_mod + 1e-6)).view(S, 1, 1))
                if t % 16 == 0:
                    self.W_mod.mul_(0.9994)

                credit = (r_each * e_local.unsqueeze(1)).sum(dim=-1) / math.sqrt(D)
                c_centered = (credit - credit.mean(dim=0, keepdim=True)).clamp(-0.5, 0.5)
                self.G_write.add_(torch.einsum("bs,bsd->sd", c_centered, slot_last_x) * (lr_sg / B))
                self.G_keep.add_(torch.einsum("bs,bd->sd", c_centered, x_slot) * (lr_sg * 0.6 / B))
                if t % 16 == 0:
                    self.G_write.mul_(0.998)
                    self.G_keep.mul_(0.998)

                cur_wr = w_gate.mean(dim=0)
                self.write_bias.add_((self.target_write_rate - cur_wr) * 0.04)
                self.write_bias.clamp_(-5.5, 1.0)

            top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
            top_bad = torch.where(top2_i[:, 0] != y_t, top2_i[:, 0], top2_i[:, 1])
            p_bad = torch.where(top2_i[:, 0] != y_t, top2_p[:, 0], top2_p[:, 1])

            if track_divorce and traversed_conv_edges:
                div_bkts_list: List[torch.Tensor] = []
                div_slots_list: List[torch.Tensor] = []
                for cb, cbkt, cslot, cnode in traversed_conv_edges:
                    c_surp = surprise[cb]
                    c_align = (self.W_node[cnode].to(torch.float32) * e_local[cb]).sum(dim=-1) / math.sqrt(D)
                    hurt = (c_surp > cfg.divorce_surprise_thresh) & (c_align < cfg.divorce_align_thresh)
                    help_ok = (c_surp < 0.30) & (c_align > 0.02)
                    if hurt.any():
                        hb, hs = cbkt[hurt], cslot[hurt]
                        self.edge_pain.index_put_((hb, hs), (c_surp[hurt] * 0.65).to(torch.bfloat16), accumulate=True)
                        over = self.edge_pain[hb, hs].to(torch.float32) >= cfg.divorce_pain_thresh
                        if over.any():
                            div_bkts_list.append(hb[over])
                            div_slots_list.append(hs[over])
                    if help_ok.any():
                        gb, gs = cbkt[help_ok], cslot[help_ok]
                        self.edge_pain[gb, gs] = (self.edge_pain[gb, gs].to(torch.float32) * 0.7).clamp(min=0.0).to(torch.bfloat16)
                if div_bkts_list:
                    divorces_this_pass += self._divorce_edges(torch.cat(div_bkts_list), torch.cat(div_slots_list))

            ones_all = torch.ones(all_u.numel(), dtype=torch.int32, device=device)
            self.fires.scatter_add_(0, all_u, ones_all)
            self.recent_fires.scatter_add_(0, all_u, ones_all)
            align_all = (self.W_node[all_u].to(torch.float32) * e_local[all_b]).sum(dim=-1) / math.sqrt(D)
            self.utility.scatter_add_(0, all_u, torch.relu(align_all + 0.05) * 0.25)
            self.upd_cnt.scatter_add_(0, all_u, torch.ones(all_u.numel(), dtype=torch.float32, device=device))
            lr_u = (lr_feat * all_g) / torch.sqrt(1.0 + cfg.metaplasticity * self.upd_cnt[all_u])
            self.W_node.index_add_(0, all_u, (e_local[all_b] * lr_u.unsqueeze(1)).to(torch.bfloat16))
            self._update_sharp_batched(all_u, y_t[all_b], surprise[all_b], top_bad[all_b], p_bad[all_b], lr_sharp * all_g)

            bias_g = torch.zeros(V, dtype=torch.float32, device=device)
            bias_g.scatter_add_(0, y_t, ones_B_f)
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
                    top_ctx = top_ctx_all[sb]
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
            "births": births_this_pass,
            "divorces": divorces_this_pass,
            "alive_hidden": self.num_hidden_alive,
            "edges": self.edges.count(),
            "ltm_active": self.ltm.count_active(),
        }

    @torch.no_grad()
    def generate_executive_utf8(
        self,
        prompt: str,
        max_new_bytes: int = 210,
        temperature: float = 0.68,
        top_k: int = 24,
        rollout_k: int = 4,
        rollout_chunk_bytes: int = 42,
        use_executive: bool = True,
    ) -> Tuple[str, bool, Dict[str, float]]:
        cfg = self.cfg
        device = self.device
        V, D, S = self.V, self.D, cfg.num_slots
        raw_in = list(prompt.encode("utf-8"))
        H_buf = max(cfg.max_hops + 4, 22)
        D_ctx = 12
        N_stm = cfg.stm_slots

        def init_state():
            return {
                "byte_hist": torch.full((1, H_buf), -1, dtype=torch.int64, device=device),
                "hist_ptr": 0,
                "node_delay": torch.full((1, D_ctx), -1, dtype=torch.int64, device=device),
                "nd_ptr": 0,
                "ctx_node": torch.full((1, 2), -1, dtype=torch.int64, device=device),
                "ctx_score": torch.zeros((1, 2), dtype=torch.float32, device=device),
                "slot_state": torch.zeros((1, S, D), dtype=torch.float32, device=device),
                "stm_k4": torch.zeros((1, N_stm), dtype=torch.int64, device=device),
                "stm_k7": torch.zeros((1, N_stm), dtype=torch.int64, device=device),
                "stm_val": torch.zeros((1, N_stm), dtype=torch.int64, device=device),
                "stm_str": torch.zeros((1, N_stm), dtype=torch.float32, device=device),
                "stm_ptr": 0,
                "exec_vec": torch.zeros((1, D), dtype=torch.float32, device=device),
                "step_i": 0,
            }

        def clone_state(st):
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in st.items()}

        ltm_b = torch.zeros((1, V), dtype=torch.float32, device=device)

        def step_state(st, b_val: int, next_true: int = -1, locked_mode_p=None):
            step_i = st["step_i"]
            st["byte_hist"][0, st["hist_ptr"]] = b_val
            cur_idx = st["hist_ptr"]
            st["hist_ptr"] = (st["hist_ptr"] + 1) % H_buf

            st["ctx_score"].mul_(0.985)
            if step_i >= D_ctx:
                u_old = int(st["node_delay"][0, st["nd_ptr"]].item())
                if u_old >= V and bool(self.alive[u_old].item()):
                    sal = 1.8 / math.sqrt(1.0 + float(self.fires[u_old].item()) * 0.02)
                    if sal > 0.55 and u_old not in st["ctx_node"][0].tolist():
                        ws = int(st["ctx_score"][0].argmin().item())
                        if sal > float(st["ctx_score"][0, ws].item()):
                            st["ctx_node"][0, ws] = u_old
                            st["ctx_score"][0, ws] = sal
            top_ctx = st["ctx_node"][:, int(st["ctx_score"][0].argmax().item())]

            h_dag = self.W_node[b_val].to(torch.float32)
            logits_s = torch.zeros(V, dtype=torch.float32, device=device)
            sd0, sw0 = self.sharp_dst[b_val], self.sharp_w[b_val].to(torch.float32)
            m0 = sd0 >= 0
            if m0.any():
                logits_s.index_add_(0, sd0[m0].to(torch.int64), sw0[m0] * 0.65)

            frontier = [b_val]
            emergent_cand = -1
            salient_topic_vec = None
            for hop in range(1, cfg.max_hops + 1):
                nxt: List[int] = []
                gain = 1.0 / (1.0 + 0.10 * hop)
                for p in frontier:
                    sp = int(self.local_span[p].item())
                    c_loc = int(st["byte_hist"][0, (cur_idx - min(sp + 1, H_buf - 1)) % H_buf].item())
                    if c_loc >= 0:
                        ch = int(self.edges.lookup(
                            torch.tensor([p], device=device), torch.zeros(1, dtype=torch.int64, device=device), torch.tensor([c_loc], device=device)
                        ).item())
                        if ch >= V and bool(self.alive[ch].item()):
                            nxt.append(ch)
                            if 2 <= hop <= 4:
                                emergent_cand = ch
                            if hop >= 3 and int(self.fires[ch].item()) < 5000:
                                salient_topic_vec = self.W_node[ch].to(torch.float32)
                    if cfg.enable_long_branch and 2 <= hop <= 5 and int(top_ctx[0].item()) >= V:
                        chc = int(self.edges.lookup(
                            torch.tensor([p], device=device), torch.ones(1, dtype=torch.int64, device=device), top_ctx
                        ).item())
                        if chc >= V and bool(self.alive[chc].item()):
                            nxt.append(chc)
                if not nxt:
                    break
                nxt = list(dict.fromkeys(nxt))[:4]
                for u in nxt:
                    h_dag.add_(self.W_node[u].to(torch.float32), alpha=gain)
                    sdu, swu = self.sharp_dst[u], self.sharp_w[u].to(torch.float32)
                    mu = sdu >= 0
                    if mu.any():
                        logits_s.index_add_(0, sdu[mu].to(torch.int64), swu[mu] * (gain * 0.65))
                frontier = nxt

            st["node_delay"][0, st["nd_ptr"]] = emergent_cand
            st["nd_ptr"] = (st["nd_ptr"] + 1) % D_ctx

            h_n = h_dag / h_dag.norm().clamp(min=1.0)
            e_n = self.E[b_val] / self.E[b_val].norm().clamp(min=1.0)
            xs = (0.65 * h_n + 0.35 * e_n).unsqueeze(0)
            kg = torch.sigmoid(xs @ self.G_keep.T + self.keep_bias.unsqueeze(0))
            wg = torch.sigmoid(xs @ self.G_write.T + self.write_bias.unsqueeze(0))
            st["slot_state"] = st["slot_state"] * kg.unsqueeze(-1) + wg.unsqueeze(-1) * xs.unsqueeze(1)
            sn = st["slot_state"].norm(dim=-1, keepdim=True)
            st["slot_state"] = st["slot_state"] * (2.5 / sn.clamp(min=2.5))
            r_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], self.W_read).sum(dim=1)[0]
            m_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], self.W_mod).sum(dim=1)[0]
            h_tot = h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum
            h_act = torch.tanh(h_tot * 0.35) * 2.85
            logits = h_act @ self.E.T + self.bias + logits_s

            lk, lv = self.ltm.compute_keys(st["byte_hist"], cur_idx, H_buf, top_ctx)
            self.ltm.lookup_boost(lk, lv, ltm_b, max_boost=cfg.ltm_max_boost, ltm_scale=cfg.ltm_scale)
            logits = logits + ltm_b[0]

            k4, k7 = self._compute_stm_keys(st["byte_hist"], cur_idx, H_buf)
            st["stm_str"].mul_(cfg.stm_decay)
            if step_i >= 4:
                m4 = (st["stm_k4"][0] == k4[0]) & (k4[0] != 0) & (st["stm_str"][0] > 0.05)
                m7 = (st["stm_k7"][0] == k7[0]) & (k7[0] != 0) & (st["stm_str"][0] > 0.05)
                hw = m4.float() * (st["stm_str"][0] * cfg.stm_w4) + m7.float() * (st["stm_str"][0] * cfg.stm_w7)
                if (hw > 0).any():
                    boost = torch.zeros(V, dtype=torch.float32, device=device)
                    idx_h = torch.nonzero(hw > 0, as_tuple=False).squeeze(1)
                    boost.index_add_(0, st["stm_val"][0, idx_h], hw[idx_h])
                    logits = logits + boost.clamp(max=cfg.stm_max_boost)

            st["exec_vec"].mul_(cfg.mode_ema_decay).add_(h_dag.unsqueeze(0), alpha=1.0 - cfg.mode_ema_decay)
            if use_executive and self.mode_ready:
                if locked_mode_p is not None:
                    ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
                    cur_mp = torch.softmax((ev_n @ self.mode_centroids.T) / cfg.mode_temp, dim=-1)
                    mp = 0.60 * locked_mode_p + 0.40 * cur_mp
                else:
                    ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
                    mp = torch.softmax((ev_n @ self.mode_centroids.T) / cfg.mode_temp, dim=-1)
                logits = logits + (mp @ self.mode_llr)[0] * (cfg.mode_llr_scale * 1.15)

            if next_true >= 0 and step_i >= 4 and int(k4[0].item()) != 0:
                p_nt = float(torch.softmax(logits, dim=-1)[next_true].item())
                if (1.0 - p_nt) > cfg.stm_surp_thresh:
                    sp_i = st["stm_ptr"]
                    st["stm_k4"][0, sp_i] = k4[0]
                    st["stm_k7"][0, sp_i] = k7[0]
                    st["stm_val"][0, sp_i] = next_true
                    st["stm_str"][0, sp_i] = 1.0
                    st["stm_ptr"] = (sp_i + 1) % N_stm

            st["step_i"] = step_i + 1
            return logits, h_dag, salient_topic_vec

        st = init_state()
        prompt_h_list = []
        topic_vecs = []
        logits = None
        for i, b in enumerate(raw_in):
            nxt_t = raw_in[i + 1] if i + 1 < len(raw_in) else -1
            logits, hd, tv = step_state(st, b, next_true=nxt_t, locked_mode_p=None)
            prompt_h_list.append(hd)
            if tv is not None:
                topic_vecs.append(tv)

        prompt_h = torch.stack(prompt_h_list).mean(dim=0)
        topic_anchor = torch.stack(topic_vecs).mean(dim=0) if topic_vecs else prompt_h
        topic_anchor = topic_anchor / topic_anchor.norm().clamp(min=1e-6)
        prompt_mode_vec = prompt_h / prompt_h.norm().clamp(min=1e-6)
        locked_mode_p = (
            torch.softmax(((prompt_mode_vec.unsqueeze(0) @ self.mode_centroids.T) / cfg.mode_temp), dim=-1)
            if (use_executive and self.mode_ready) else None
        )

        out_bytes = list(raw_in)
        gen_only: List[int] = []
        recent_ngrams: set = set()
        topic_cos_scores: List[float] = []

        rem = max_new_bytes
        while rem > 0:
            c_len = min(rollout_chunk_bytes, rem)
            k_cand = rollout_k if (use_executive and rollout_k > 1) else 1

            best_score = -1e9
            best_pack = None
            for _ in range(k_cand):
                cst = clone_state(st)
                clog = logits
                c_bytes: List[int] = []
                c_hs = []
                lp_sum = 0.0
                rep_pen = 0.0
                local_seen = set(recent_ngrams)

                for _ in range(c_len):
                    lp_all = torch.log_softmax(clog, dim=-1)
                    scaled = clog / max(temperature, 1e-4)
                    tv, ti = torch.topk(scaled, k=min(top_k, V))
                    probs = torch.softmax(tv, dim=-1)
                    nxt_b = int(ti[int(torch.multinomial(probs, 1).item())].item())
                    lp_sum += float(lp_all[nxt_b].item())
                    c_bytes.append(nxt_b)
                    if len(c_bytes) >= 9:
                        ng = tuple(c_bytes[-9:])
                        if ng in local_seen:
                            rep_pen += 0.65
                        local_seen.add(ng)
                    clog, chd, _ = step_state(cst, nxt_b, locked_mode_p=locked_mode_p)
                    c_hs.append(chd)

                raw_c = bytes(c_bytes)
                utf8_bonus = 0.0
                for trim in (0, 1, 2):
                    sub = raw_c[: len(raw_c) - trim] if trim else raw_c
                    try:
                        sub.decode("utf-8")
                        utf8_bonus = 0.4
                        break
                    except UnicodeDecodeError:
                        continue

                c_mean = torch.stack(c_hs).mean(dim=0)
                c_mean_n = c_mean / c_mean.norm().clamp(min=1e-6)
                topic_sim = float((c_mean_n @ topic_anchor).item())
                mode_sim = float((c_mean_n @ prompt_mode_vec).item())
                fluency = lp_sum / max(1, c_len)

                score = fluency + 1.15 * topic_sim + 0.85 * mode_sim + utf8_bonus - rep_pen
                if score > best_score:
                    best_score = score
                    best_pack = (cst, clog, c_bytes, topic_sim)

            st, logits, chosen_b, chosen_ts = best_pack
            out_bytes.extend(chosen_b)
            gen_only.extend(chosen_b)
            topic_cos_scores.append(chosen_ts)
            for i in range(max(0, len(gen_only) - len(chosen_b) - 8), len(gen_only) - 8):
                recent_ngrams.add(tuple(gen_only[i : i + 9]))
            rem -= c_len

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
        return text, valid_utf8, {"avg_topic_cos": round(sum(topic_cos_scores) / max(1, len(topic_cos_scores)), 3)}
