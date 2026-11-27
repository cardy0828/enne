"""ノードの新生・分岐・削除を行う成長グラフ（v0.8）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from neurograph.memory_v3b import _mix64


@dataclass
class GrowingGraphConfig:
    vocab_size: int
    pool_capacity: int = 160_000
    feat_dim: int = 96
    max_hops: int = 8
    enable_long_branch: bool = True
    enable_apoptosis: bool = True
    edge_buckets: int = 262_144
    edge_ways: int = 4
    birth_surprise_thresh: float = 0.48
    birth_pain_thresh: float = 2.2
    max_births_per_step: int = 64
    min_fires_survive: int = 3
    min_utility_survive: float = 0.02
    lr_feat: float = 0.24
    lr_basis: float = 0.04
    lr_sharp: float = 0.30
    k_sharp: int = 8
    metaplasticity: float = 0.02


def _edge_sig(parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    h = _mix64(parent.to(torch.int64) + 11)
    h = _mix64(h ^ ((kind.to(torch.int64) + 1) * 1000003))
    h = _mix64(h ^ (cond.to(torch.int64) + 2000029))
    return torch.where(h == 0, torch.ones_like(h), h)


class DynamicEdgeTable:

    def __init__(self, num_buckets: int, ways: int, device: torch.device):
        assert (num_buckets & (num_buckets - 1)) == 0
        self.num_buckets = num_buckets
        self.mask = num_buckets - 1
        self.ways = ways
        self.device = device

        self.sig = torch.zeros((num_buckets, ways), dtype=torch.int64, device=device)
        self.child = torch.full((num_buckets, ways), -1, dtype=torch.int32, device=device)

    @torch.no_grad()
    def lookup(self, parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.mask).to(torch.int64)
        b_sig = self.sig[bkts]
        b_ch = self.child[bkts]
        hit = (b_sig == sigs.unsqueeze(1)) & (b_ch >= 0)
        has = hit.any(dim=1)
        slot = hit.to(torch.int32).argmax(dim=1)
        found = b_ch[torch.arange(parent.numel(), device=self.device), slot].to(torch.int64)
        return torch.where(has, found, torch.full_like(found, -1))

    @torch.no_grad()
    def insert(
        self,
        parent: torch.Tensor,
        kind: torch.Tensor,
        cond: torch.Tensor,
        child_id: torch.Tensor,
    ) -> torch.Tensor:
        n = parent.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.mask).to(torch.int64)

        order = torch.argsort(bkts, stable=True)
        s_bkts = bkts[order]
        is_first = torch.ones(n, dtype=torch.bool, device=self.device)
        if n > 1:
            is_first[1:] = s_bkts[1:] != s_bkts[:-1]

        cand_idx = order[is_first]
        c_bkts = bkts[cand_idx]
        c_sigs = sigs[cand_idx]
        c_child = child_id[cand_idx].to(torch.int32)

        b_sig = self.sig[c_bkts]
        b_ch = self.child[c_bkts]
        already = ((b_sig == c_sigs.unsqueeze(1)) & (b_ch >= 0)).any(dim=1)
        empty = b_ch < 0
        has_empty = empty.any(dim=1)
        ok = (~already) & has_empty

        success = torch.zeros(n, dtype=torch.bool, device=self.device)
        if ok.any():
            ok_pos = cand_idx[ok]
            ok_bkts = c_bkts[ok]
            ok_slots = empty[ok].to(torch.int32).argmax(dim=1)
            self.sig[ok_bkts, ok_slots] = c_sigs[ok]
            self.child[ok_bkts, ok_slots] = c_child[ok]
            success[ok_pos] = True
        return success

    @torch.no_grad()
    def delete_children(self, dead_mask: torch.Tensor) -> int:
        valid = self.child >= 0
        if not valid.any():
            return 0
        safe_c = self.child.clamp(min=0).to(torch.int64)
        to_del = valid & dead_mask[safe_c]
        n_del = int(to_del.sum().item())
        if n_del > 0:
            self.sig[to_del] = 0
            self.child[to_del] = -1
        return n_del

    def count(self) -> int:
        return int((self.child >= 0).sum().item())


class NeurogenesisGraph:

    def __init__(self, cfg: GrowingGraphConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size
        self.P = cfg.pool_capacity
        self.D = cfg.feat_dim

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

        self.edges = DynamicEdgeTable(cfg.edge_buckets, cfg.edge_ways, device)

        self.E = torch.zeros((self.V, self.D), dtype=torch.float32, device=device)
        self.bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        self.W_node = torch.zeros((self.P, self.D), dtype=torch.float32, device=device)
        self.upd_cnt = torch.zeros(self.P, dtype=torch.float32, device=device)

        self.sharp_dst = torch.full((self.P, cfg.k_sharp), -1, dtype=torch.int32, device=device)
        self.sharp_w = torch.zeros((self.P, cfg.k_sharp), dtype=torch.float32, device=device)

        self.char_salience = torch.ones(self.V, dtype=torch.float32, device=device)

        self.total_births = 0
        self.total_deaths = 0

    @property
    def num_hidden_alive(self) -> int:
        return int(self.alive[self.V :].sum().item())

    @torch.no_grad()
    def init_sensory_basis(self, train_ids: np.ndarray, sample_len: int = 2_000_000) -> None:
        V, D = self.V, self.D
        seq = torch.from_numpy(train_ids[: min(sample_len, len(train_ids))].astype(np.int64)).to(self.device)

        cooc = torch.zeros((V, V), dtype=torch.float32, device=self.device)
        for d, w in ((1, 1.0), (2, 0.5)):
            a, b = seq[:-d], seq[d:]
            ok = (a > 0) & (b > 0)
            av, bv = a[ok], b[ok]
            vals = torch.full((av.numel(),), w, dtype=torch.float32, device=self.device)
            cooc.view(-1).scatter_add_(0, av * V + bv, vals)
            cooc.view(-1).scatter_add_(0, bv * V + av, vals)

        tot = cooc.sum().clamp(min=1.0)
        rs = cooc.sum(dim=1, keepdim=True).clamp(min=1.0)
        cs = cooc.sum(dim=0, keepdim=True).clamp(min=1.0)
        ppmi = torch.relu(torch.log((cooc * tot) / (rs * cs) + 1e-8))
        U, S, _ = torch.svd_lowrank(ppmi, q=min(D + 12, V - 1), niter=4)
        E_init = U[:, :D] * torch.sqrt(S[:D].clamp(min=1e-6)).unsqueeze(0)
        self.E.copy_((E_init / E_init.norm(dim=1, keepdim=True).clamp(min=1e-5)) * 1.3)
        self.E[0].zero_()

        counts = torch.bincount(seq, minlength=V).to(torch.float32)
        self.bias.copy_(torch.log((counts + 1.0) / (counts.sum() + V)))
        avg_c = counts.sum() / max(1, V)
        self.char_salience.copy_(((avg_c / (counts + 5.0)) ** 0.55).clamp(0.05, 3.0))
        self.char_salience[0] = 0.0

    def describe_neuron(self, nid: int, id_to_char: List[str]) -> str:
        if nid < 0 or nid >= self.P or not bool(self.alive[nid].item()):
            return "<DEAD>"
        if nid < self.V:
            return id_to_char[nid]
        local_chars: List[str] = []
        ctx_tags: List[str] = []
        cur = nid
        guard = 0
        while cur >= self.V and guard < 20:
            k = int(self.cond_kind[cur].item())
            cid = int(self.cond_id[cur].item())
            c_str = id_to_char[cid] if 0 <= cid < self.V else "?"
            if k == 0:
                local_chars.append(c_str)
            else:
                ctx_tags.append(f"@{c_str}")
            cur = int(self.parent[cur].item())
            guard += 1
        root_str = id_to_char[cur] if 0 <= cur < self.V else "?"
        seq_str = "".join(local_chars) + root_str
        if ctx_tags:
            return f"{seq_str}[{','.join(ctx_tags)}]"
        return seq_str

    @torch.no_grad()
    def _allocate_free_ids(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        free = torch.nonzero(~self.alive[self.V :], as_tuple=False).squeeze(1) + self.V
        if free.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return free[:count]

    @torch.no_grad()
    def _spawn_children(
        self,
        parents: torch.Tensor,
        kinds: torch.Tensor,
        conds: torch.Tensor,
    ) -> int:
        M = parents.numel()
        if M == 0:
            return 0
        if M > self.cfg.max_births_per_step:
            parents = parents[: self.cfg.max_births_per_step]
            kinds = kinds[: self.cfg.max_births_per_step]
            conds = conds[: self.cfg.max_births_per_step]
            M = parents.numel()

        new_ids = self._allocate_free_ids(M)
        n_alloc = new_ids.numel()
        if n_alloc == 0:
            return 0
        parents = parents[:n_alloc]
        kinds = kinds[:n_alloc]
        conds = conds[:n_alloc]

        ok = self.edges.insert(parents, kinds, conds, new_ids)
        if not ok.any():
            return 0

        born = new_ids[ok]
        bp = parents[ok]
        bk = kinds[ok]
        bc = conds[ok]

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

        self.pain[bp] *= 0.35

        n_born = int(born.numel())
        self.total_births += n_born
        return n_born

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

        need = (~has) & (sp > 0.30) & (yt > 0)
        if need.any():
            nu = u[need]
            nd = dst[need]
            nw = w[need]
            emp = nd < 0
            has_e = emp.any(dim=1)
            e_slot = emp.to(torch.int32).argmax(dim=1)
            w_slot = nw.abs().argmin(dim=1)
            slot = torch.where(has_e, e_slot, w_slot)
            self.sharp_dst[nu, slot] = yt[need]
            self.sharp_w[nu, slot] = (lr * sp[need]).clamp(0.05, 1.5)

        bad_valid = (tb != yt) & (pb > 0.08)
        if bad_valid.any():
            bu = u[bad_valid]
            bt = tb[bad_valid]
            bp = pb[bad_valid]
            bdst = self.sharp_dst[bu]
            bhit = bdst == bt.unsqueeze(1)
            bhas = bhit.any(dim=1)
            if bhas.any():
                bhu = bu[bhas]
                bhs = bhit[bhas].to(torch.int32).argmax(dim=1)
                self.sharp_w[bhu, bhs] = (self.sharp_w[bhu, bhs] - (lr * 0.8) * bp[bhas]).clamp(-3.5, 4.0)

    @torch.no_grad()
    def apoptosis(self) -> Dict[str, int]:
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

        for _ in range(self.cfg.max_hops):
            p_ids = self.parent.clamp(min=0).to(torch.int64)
            orphans = hidden_alive & (~dead) & (self.parent >= self.V) & dead[p_ids]
            if not orphans.any():
                break
            dead |= orphans

        n_died = int(dead.sum().item())
        if n_died > 0:
            self.edges.delete_children(dead)
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
            self.total_deaths += n_died

        self.recent_fires.zero_()
        self.utility.mul_(0.85)

        return {
            "died": n_died,
            "alive_hidden": self.num_hidden_alive,
            "edges": self.edges.count(),
        }

    @torch.no_grad()
    def run_stream(
        self,
        ids: np.ndarray,
        train: bool = True,
        batch_streams: int = 512,
        lr_scale: float = 1.0,
    ) -> Dict[str, float]:
        cfg = self.cfg
        device = self.device
        V, D = self.V, self.D

        total_len = len(ids)
        stream_len = total_len // batch_streams
        if stream_len < 32:
            batch_streams = max(1, total_len // 32)
            stream_len = total_len // batch_streams

        data = torch.from_numpy(
            ids[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(device)
        B, T = data.shape

        H_buf = cfg.max_hops + 4
        char_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0

        ctx_char = torch.full((B, 2), -1, dtype=torch.int64, device=device)
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

        MAX_FRONT = 4
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=device)

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]

            char_hist[:, hist_ptr] = x_t
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            ctx_score.mul_(0.965)
            if t >= 6:
                c_old = char_hist[:, (cur_idx - 6) % H_buf]
                sal_old = self.char_salience[c_old.clamp(min=0)]
                already = (ctx_char[:, 0] == c_old) | (ctx_char[:, 1] == c_old)
                weak_slot = ctx_score.argmin(dim=1)
                weak_val = ctx_score[arange_B, weak_slot]
                replace = (~already) & (c_old > 0) & (sal_old > weak_val) & (sal_old > 0.75)
                if replace.any():
                    rb = torch.nonzero(replace, as_tuple=False).squeeze(1)
                    rs = weak_slot[rb]
                    ctx_char[rb, rs] = c_old[rb]
                    ctx_score[rb, rs] = sal_old[rb]

            visited_nodes: List[torch.Tensor] = [x_t]
            visited_gains: List[float] = [1.0]

            frontier = x_t.unsqueeze(1)
            stream_max_depth = torch.zeros(B, dtype=torch.int32, device=device)
            deepest_leaf = x_t.clone()
            deepest_span = torch.zeros(B, dtype=torch.int64, device=device)

            for hop in range(1, cfg.max_hops + 1):
                F_cur = frontier.shape[1]
                valid_f = frontier >= 0
                if not valid_f.any():
                    break

                fb, ff = torch.nonzero(valid_f, as_tuple=True)
                p_ids = frontier[fb, ff]
                p_span = self.local_span[p_ids].to(torch.int64)

                next_dist = (p_span + 1).clamp(max=H_buf - 1)
                c_loc = char_hist[fb, (cur_idx - next_dist) % H_buf]
                k0 = torch.zeros_like(p_ids)
                ch_loc = self.edges.lookup(p_ids, k0, c_loc)
                ch_loc = torch.where((c_loc > 0) & (ch_loc >= 0) & self.alive[ch_loc.clamp(min=0)], ch_loc, torch.full_like(ch_loc, -1))

                ch_ctx = torch.full_like(p_ids, -1)
                if cfg.enable_long_branch and hop <= 4:
                    top_ctx = ctx_char[fb, ctx_score[fb].argmax(dim=1)]
                    k1 = torch.ones_like(p_ids)
                    raw_ctx = self.edges.lookup(p_ids, k1, top_ctx)
                    ch_ctx = torch.where(
                        (top_ctx > 0) & (raw_ctx >= 0) & self.alive[raw_ctx.clamp(min=0)],
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
                    new_sp = self.local_span[hl_c].to(torch.int64)
                    deeper = new_sp > deepest_span[hl_b]
                    if deeper.any():
                        deepest_leaf[hl_b[deeper]] = hl_c[deeper]
                        deepest_span[hl_b[deeper]] = new_sp[deeper]

                hit_ctx = ch_ctx >= 0
                if hit_ctx.any():
                    hc_b = fb[hit_ctx]
                    hc_c = ch_ctx[hit_ctx]
                    hc_slot = (ff[hit_ctx] * 2 + 1) % MAX_FRONT
                    next_front[hc_b, hc_slot] = hc_c
                    stream_max_depth[hc_b] = hop

                if not (hit_loc.any() or hit_ctx.any()):
                    break

                gain = 1.0 / (1.0 + 0.15 * hop)
                for col in range(MAX_FRONT):
                    col_v = next_front[:, col]
                    if (col_v >= 0).any():
                        visited_nodes.append(col_v)
                        visited_gains.append(gain)

                frontier = next_front

            d_mean = float(stream_max_depth.to(torch.float32).mean().item())
            total_depth_sum += d_mean * B
            cur_md = int(stream_max_depth.max().item())
            if cur_md > max_depth_seen:
                max_depth_seen = cur_md

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
            h_act = torch.tanh(h * 0.5) * 2.0
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
            sat = (1.0 - (h_act * 0.5).pow(2)).clamp(min=0.15)
            e_local = e_feat * sat

            top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
            top_bad = torch.where(top2_i[:, 0] != y_t, top2_i[:, 0], top2_i[:, 1])
            p_bad = torch.where(top2_i[:, 0] != y_t, top2_p[:, 0], top2_p[:, 1])

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
                self.E[0].zero_()

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
                    c_loc = char_hist[sb, (cur_idx - next_d) % H_buf]

                    use_long = (
                        cfg.enable_long_branch
                        & (l_span >= 2)
                        & (ctx_score[sb].max(dim=1).values > 1.1)
                        & ((t % 3) == 0)
                    )
                    top_ctx = ctx_char[sb, ctx_score[sb].argmax(dim=1)]
                    kinds = torch.where(use_long, torch.ones_like(sp), torch.zeros_like(sp))
                    conds = torch.where(use_long, top_ctx, c_loc)

                    valid_birth = conds > 0
                    if valid_birth.any():
                        births_this_pass += self._spawn_children(sp[valid_birth], kinds[valid_birth], conds[valid_birth])

        avg_nll = total_nll / max(1, total_steps)
        return {
            "bpc": round(avg_nll / math.log(2.0), 4),
            "ppl": round(math.exp(min(avg_nll, 20.0)), 2),
            "top1": round(total_top1 / max(1, total_steps), 4),
            "top5": round(total_top5 / max(1, total_steps), 4),
            "avg_depth": round(total_depth_sum / max(1, total_steps), 2),
            "avg_visited": round(total_visited_sum / max(1, total_steps), 2),
            "max_depth": max_depth_seen,
            "births": births_this_pass,
            "alive_hidden": self.num_hidden_alive,
            "edges": self.edges.count(),
        }

    @torch.no_grad()
    def trace_sentence_depths(self, ids: List[int], id_to_char: List[str]) -> List[Dict[str, object]]:
        cfg = self.cfg
        device = self.device
        H_buf = cfg.max_hops + 4
        char_hist = torch.full((1, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0
        ctx_char = torch.full((1, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((1, 2), dtype=torch.float32, device=device)

        trace = []
        for t, ch_id in enumerate(ids):
            x_t = torch.tensor([ch_id], dtype=torch.int64, device=device)
            char_hist[0, hist_ptr] = ch_id
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            ctx_score.mul_(0.965)
            if t >= 6:
                c_old = int(char_hist[0, (cur_idx - 6) % H_buf].item())
                sal_old = float(self.char_salience[max(0, c_old)].item())
                if c_old > 0 and c_old not in ctx_char[0].tolist() and sal_old > 0.75:
                    ws = int(ctx_score[0].argmin().item())
                    if sal_old > float(ctx_score[0, ws].item()):
                        ctx_char[0, ws] = c_old
                        ctx_score[0, ws] = sal_old

            frontier = [ch_id]
            fired_desc = [self.describe_neuron(ch_id, id_to_char)]
            depth_reached = 0
            for hop in range(1, cfg.max_hops + 1):
                nxt = []
                for pid in frontier:
                    p_t = torch.tensor([pid], dtype=torch.int64, device=device)
                    sp = int(self.local_span[pid].item())
                    c_loc = int(char_hist[0, (cur_idx - min(sp + 1, H_buf - 1)) % H_buf].item())
                    if c_loc > 0:
                        ch0 = int(self.edges.lookup(p_t, torch.tensor([0], device=device), torch.tensor([c_loc], device=device)).item())
                        if ch0 >= 0 and bool(self.alive[ch0].item()):
                            nxt.append(ch0)
                            fired_desc.append(self.describe_neuron(ch0, id_to_char))
                    if cfg.enable_long_branch and hop <= 4:
                        tc = int(ctx_char[0, int(ctx_score[0].argmax().item())].item())
                        if tc > 0:
                            ch1 = int(self.edges.lookup(p_t, torch.tensor([1], device=device), torch.tensor([tc], device=device)).item())
                            if ch1 >= 0 and bool(self.alive[ch1].item()):
                                nxt.append(ch1)
                                fired_desc.append(self.describe_neuron(ch1, id_to_char))
                if not nxt:
                    break
                depth_reached = hop
                frontier = nxt[:4]
            trace.append({"char": id_to_char[ch_id], "depth": depth_reached, "path": fired_desc})
        return trace

    @torch.no_grad()
    def sample_deep_branches(self, id_to_char: List[str], top_k: int = 15) -> List[str]:
        alive_h = torch.nonzero(self.alive[self.V :], as_tuple=False).squeeze(1) + self.V
        if alive_h.numel() == 0:
            return []
        score = self.depth[alive_h].to(torch.float32) * 2.0 + torch.log1p(self.fires[alive_h].to(torch.float32))
        top_i = torch.topk(score, k=min(top_k, alive_h.numel())).indices
        out = []
        for idx in top_i.tolist():
            nid = int(alive_h[idx].item())
            out.append(f"d={int(self.depth[nid])} (fires={int(self.fires[nid])}): {self.describe_neuron(nid, id_to_char)}")
        return out

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_chars: int = 100, temperature: float = 0.65, top_k: int = 15) -> List[int]:
        cfg = self.cfg
        device = self.device
        V, D = self.V, self.D
        H_buf = cfg.max_hops + 4
        char_hist = torch.full((1, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0
        ctx_char = torch.full((1, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((1, 2), dtype=torch.float32, device=device)
        step_n = 0

        def step_in(ch_id: int) -> torch.Tensor:
            nonlocal hist_ptr, step_n
            char_hist[0, hist_ptr] = ch_id
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf
            ctx_score.mul_(0.965)
            if step_n >= 6:
                c_old = int(char_hist[0, (cur_idx - 6) % H_buf].item())
                sal_old = float(self.char_salience[max(0, c_old)].item())
                if c_old > 0 and c_old not in ctx_char[0].tolist() and sal_old > 0.75:
                    ws = int(ctx_score[0].argmin().item())
                    if sal_old > float(ctx_score[0, ws].item()):
                        ctx_char[0, ws] = c_old
                        ctx_score[0, ws] = sal_old
            step_n += 1

            frontier = [ch_id]
            h = self.W_node[ch_id].clone()
            l_sharp = torch.zeros(V, dtype=torch.float32, device=device)

            def add_sharp(nid: int, gain: float) -> None:
                sd = self.sharp_dst[nid]
                sw = self.sharp_w[nid]
                sm = sd >= 0
                if sm.any():
                    l_sharp.scatter_add_(0, sd[sm].to(torch.int64), sw[sm] * (gain * 0.65))

            add_sharp(ch_id, 1.0)
            for hop in range(1, cfg.max_hops + 1):
                nxt = []
                gain = 1.0 / (1.0 + 0.15 * hop)
                for pid in frontier:
                    p_t = torch.tensor([pid], dtype=torch.int64, device=device)
                    sp = int(self.local_span[pid].item())
                    c_loc = int(char_hist[0, (cur_idx - min(sp + 1, H_buf - 1)) % H_buf].item())
                    if c_loc > 0:
                        ch0 = int(self.edges.lookup(p_t, torch.tensor([0], device=device), torch.tensor([c_loc], device=device)).item())
                        if ch0 >= 0 and bool(self.alive[ch0].item()):
                            nxt.append(ch0)
                            h.add_(self.W_node[ch0], alpha=gain)
                            add_sharp(ch0, gain)
                    if cfg.enable_long_branch and hop <= 4:
                        tc = int(ctx_char[0, int(ctx_score[0].argmax().item())].item())
                        if tc > 0:
                            ch1 = int(self.edges.lookup(p_t, torch.tensor([1], device=device), torch.tensor([tc], device=device)).item())
                            if ch1 >= 0 and bool(self.alive[ch1].item()):
                                nxt.append(ch1)
                                h.add_(self.W_node[ch1], alpha=gain)
                                add_sharp(ch1, gain)
                if not nxt:
                    break
                frontier = nxt[:4]

            h_act = torch.tanh(h * 0.5) * 2.0
            logits = h_act @ self.E.T + self.bias + l_sharp
            logits[0] = -1e9
            return logits

        out = list(prompt_ids)
        logits = None
        for ch in prompt_ids:
            logits = step_in(ch)
        for _ in range(max_new_chars):
            assert logits is not None
            vals, idx = torch.topk(logits / max(temperature, 1e-4), k=min(top_k, V))
            pick = int(idx[torch.multinomial(torch.softmax(vals, dim=-1), 1).item()].item())
            out.append(pick)
            logits = step_in(pick)
        return out
