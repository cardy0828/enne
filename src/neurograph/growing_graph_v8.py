"""NeuroGraph v0.8: Neurogenesis Branching Graph (True Dynamic Growth, Multi-Hop Branching & Apoptosis).

Starts with ZERO hidden neurons (only atomic sensory character neurons 0..V-1).
No fixed hidden layer width, no fixed depth, no Backprop, no Attention, no Tokenizer, no hardcoded rules.

Core living-graph dynamics:
1. Event-Driven Branching Traversal (starts from 1 sensory neuron x_t):
   - Hop 0: root sensory neuron x_t fires.
   - Hop h -> h+1: every firing neuron u attempts to branch along its outgoing dendritic edges:
       * Local-sequence branch (kind=0): gated by the char at physical distance (depth(u)+1)
       * Long-range context branch (kind=1): gated by a salient context neuron in window [6..40]
     Multiple branches can fire simultaneously (true parallel branching!), and each stream
     naturally stops when no further branches match -> DYNAMIC DEPTH per character!
2. Hierarchical Residual Readout:
   - Every neuron visited along the branching path contributes its local residual wave into a
     compact shared feature space (D=96 -> smooth readout over all V chars) plus sparse top-8
     sharp synapses. Easy contexts finish shallow; hard contexts traverse deep paths.
3. Surprise-Driven Neurogenesis (Birth of New Neurons):
   - Each neuron tracks accumulated prediction surprise ('pain'). When an active leaf neuron
     repeatedly experiences high surprise, it dynamically spawns a NEW child neuron from the
     free-list pool and grows a new gated branch (either deeper local distance or long-range
     context condition) -> neuron count & graph depth grow organically!
4. Metabolic Apoptosis (Death & Pruning of Useless Neurons):
   - Neurons that rarely fire or fail to reduce prediction error beyond their parent lose
     metabolic life and DIE (alive=False, edges deleted, ID returned to free-list pool).
"""

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
    pool_capacity: int = 160_000     # allocator ceiling (starts with 0 alive hidden neurons!)
    feat_dim: int = 96               # compact shared feature readout space (eliminates long-tail zero-prob)
    max_hops: int = 8                # safety cap on branching depth
    enable_long_branch: bool = True  # allow branching on long-range salient context [6..40]
    enable_apoptosis: bool = True    # allow metabolic cell death & ID recycling
    # Edge hash table [edge_buckets, edge_ways] -> unbounded fan-out per parent neuron!
    edge_buckets: int = 262_144      # power of 2
    edge_ways: int = 4
    # Neurogenesis (birth) settings
    birth_surprise_thresh: float = 0.48
    birth_pain_thresh: float = 2.2   # parent pain needed before splitting/spawning a child
    max_births_per_step: int = 64    # smooth growth rate per parallel batch step
    # Apoptosis (death) settings
    min_fires_survive: int = 3       # newborn must fire at least this many times to survive apoptosis
    min_utility_survive: float = 0.02
    # Local plasticity (zero backprop)
    lr_feat: float = 0.24
    lr_basis: float = 0.04
    lr_sharp: float = 0.30
    k_sharp: int = 8                 # tiny sharp direct synapses per neuron
    metaplasticity: float = 0.02


def _edge_sig(parent: torch.Tensor, kind: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    h = _mix64(parent.to(torch.int64) + 11)
    h = _mix64(h ^ ((kind.to(torch.int64) + 1) * 1000003))
    h = _mix64(h ^ (cond.to(torch.int64) + 2000029))
    return torch.where(h == 0, torch.ones_like(h), h)


class DynamicEdgeTable:
    """Open-addressed GPU hash table mapping (parent_id, kind, cond_id) -> child_node_id.
    Allows scale-free variable branching (0 branches to 1000+ branches per parent!)."""

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
        """Vectorized O(1) branch lookup: returns child node IDs or -1 if no branch exists."""
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.mask).to(torch.int64)
        b_sig = self.sig[bkts]      # [N, W]
        b_ch = self.child[bkts]     # [N, W]
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
        """Insert new edges into empty slots; returns bool mask of successfully inserted edges."""
        n = parent.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        sigs = _edge_sig(parent, kind, cond)
        bkts = (sigs & self.mask).to(torch.int64)

        # Deduplicate per bucket within this batch insert to prevent write collisions
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
        """Remove all edges pointing to dead child neurons (Apoptosis cleanup)."""
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
    """Living, self-growing, branching & pruning neural graph (starts at 0 hidden neurons)."""

    def __init__(self, cfg: GrowingGraphConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size
        self.P = cfg.pool_capacity
        self.D = cfg.feat_dim

        # Alive mask: ONLY sensory char neurons 0..V-1 exist initially! Hidden neurons = 0!
        self.alive = torch.zeros(self.P, dtype=torch.bool, device=device)
        self.alive[: self.V] = True

        # Neuron genealogy & structural metadata
        self.parent = torch.full((self.P,), -1, dtype=torch.int32, device=device)
        self.cond_kind = torch.full((self.P,), -1, dtype=torch.int8, device=device)  # 0=local dist, 1=long context
        self.cond_id = torch.full((self.P,), -1, dtype=torch.int32, device=device)
        self.depth = torch.zeros(self.P, dtype=torch.int16, device=device)  # root chars = depth 0
        self.local_span = torch.zeros(self.P, dtype=torch.int16, device=device)  # how many local chars back used

        # Metabolic state per neuron (drives Neurogenesis & Apoptosis!)
        self.pain = torch.zeros(self.P, dtype=torch.float32, device=device)      # accumulated surprise
        self.fires = torch.zeros(self.P, dtype=torch.int32, device=device)       # total firing count
        self.recent_fires = torch.zeros(self.P, dtype=torch.int32, device=device)
        self.utility = torch.zeros(self.P, dtype=torch.float32, device=device)   # contribution to prediction
        self.age = torch.zeros(self.P, dtype=torch.int32, device=device)         # survival cycles

        # Dynamic outgoing branch table: (parent, kind, cond) -> child
        self.edges = DynamicEdgeTable(cfg.edge_buckets, cfg.edge_ways, device)

        # Shared smooth character feature basis E [V, D] + output bias [V]
        self.E = torch.zeros((self.V, self.D), dtype=torch.float32, device=device)
        self.bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        # Per-neuron residual feature weights W_node [P, D] + local metaplastic update count
        self.W_node = torch.zeros((self.P, self.D), dtype=torch.float32, device=device)
        self.upd_cnt = torch.zeros(self.P, dtype=torch.float32, device=device)

        # Per-neuron tiny sharp direct synapses (top-K direct char connections for crisp exact patterns)
        self.sharp_dst = torch.full((self.P, cfg.k_sharp), -1, dtype=torch.int32, device=device)
        self.sharp_w = torch.zeros((self.P, cfg.k_sharp), dtype=torch.float32, device=device)

        # Character rarity salience (for picking meaningful long-range context branch conditions)
        self.char_salience = torch.ones(self.V, dtype=torch.float32, device=device)

        # Telemetry
        self.total_births = 0
        self.total_deaths = 0

    @property
    def num_hidden_alive(self) -> int:
        return int(self.alive[self.V :].sum().item())

    @torch.no_grad()
    def init_sensory_basis(self, train_ids: np.ndarray, sample_len: int = 2_000_000) -> None:
        """Self-organize sensory character feature basis E [V, D] & character rarity from text distances."""
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
        # Rarity salience: rare content characters stand out for long-range branching (zero rules!)
        avg_c = counts.sum() / max(1, V)
        self.char_salience.copy_(((avg_c / (counts + 5.0)) ** 0.55).clamp(0.05, 3.0))
        self.char_salience[0] = 0.0

    def describe_neuron(self, nid: int, id_to_char: List[str]) -> str:
        """Render the exact branching lineage of any grown neuron back to its sensory root!"""
        if nid < 0 or nid >= self.P or not bool(self.alive[nid].item()):
            return "<DEAD>"
        if nid < self.V:
            return id_to_char[nid]
        # Walk up parent chain to reconstruct local substring + long-range context tags
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
        # local_chars were added from deepest (furthest left) to shallowest (closest left)
        seq_str = "".join(local_chars) + root_str
        if ctx_tags:
            return f"{seq_str}[{','.join(ctx_tags)}]"
        return seq_str

    @torch.no_grad()
    def _allocate_free_ids(self, count: int) -> torch.Tensor:
        """Pop up to `count` dead/free neuron slots from the pool [V..P-1]."""
        if count <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        free = torch.nonzero(~self.alive[self.V :], as_tuple=False).squeeze(1) + self.V
        if free.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return free[:count]

    @torch.no_grad()
    def _spawn_children(
        self,
        parents: torch.Tensor,   # [M] int64
        kinds: torch.Tensor,     # [M] int64 (0=local, 1=long ctx)
        conds: torch.Tensor,     # [M] int64
    ) -> int:
        """Neurogenesis: allocate free neuron IDs, wire edges, and initialize newborn neurons."""
        M = parents.numel()
        if M == 0:
            return 0
        # Cap births per step for smooth organic growth
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

        # Clean fresh metabolic & synaptic state (residual neuron starts at 0 so it never hurts parent!)
        self.pain[born] = 0.0
        self.fires[born] = 1
        self.recent_fires[born] = 1
        self.utility[born] = 0.05  # grace period for newborns
        self.age[born] = 0
        self.W_node[born] = 0.0
        self.upd_cnt[born] = 0.0
        self.sharp_dst[born] = -1
        self.sharp_w[born] = 0.0

        # Parent feels relieved after spawning a specialist child
        self.pain[bp] *= 0.35

        n_born = int(born.numel())
        self.total_births += n_born
        return n_born

    @torch.no_grad()
    def _update_sharp_synapses(
        self,
        nodes: torch.Tensor,   # [M] int64 firing neurons
        y_t: torch.Tensor,     # [M] int64 true target chars
        surp: torch.Tensor,    # [M] float32 (1 - p_true)
        top_bad: torch.Tensor, # [M] int64 top-1 wrong char (for quick lateral inhibition)
        p_bad: torch.Tensor,   # [M] float32 prob of top-1 wrong char
        lr: float,
    ) -> None:
        """Update tiny top-K sharp direct synapses per active neuron (fast vectorized, zero backprop)."""
        if nodes.numel() == 0:
            return
        # Deduplicate per (node, y_t) using first occurrence for positive updates
        # (fast approximate Hogwild update: race-free on unique nodes!)
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

        # 1) Positive reinforcement / sprouting on true y_t
        dst = self.sharp_dst[u]  # [U, K]
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

        # 2) Mild inhibition on top-1 wrong target if already connected
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
        """Metabolic Cell Death: kill useless/dormant hidden neurons & prune orphaned subtrees!"""
        if not self.cfg.enable_apoptosis:
            return {"died": 0, "alive_hidden": self.num_hidden_alive, "edges": self.edges.count()}

        hidden_alive = self.alive.clone()
        hidden_alive[: self.V] = False
        self.age[hidden_alive] += 1

        # A neuron dies if it has lived >= 1 cycle AND (rarely fired OR low utility/weight norm)
        w_norm = self.W_node.norm(dim=1)
        sharp_norm = self.sharp_w.abs().sum(dim=1)
        weak = (self.recent_fires < self.cfg.min_fires_survive) | (
            (self.utility < self.cfg.min_utility_survive) & (w_norm < 0.15) & (sharp_norm < 0.15)
        )
        dead = hidden_alive & (self.age >= 1) & weak

        # Cascade death to any children whose parent just died (clean tree/graph pruning!)
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

        # Reset cycle counters & decay utility gently
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
        """Traverse the branching graph from 1 sensory neuron per stream, grow on pain, update locally."""
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

        # History ring buffer for local sequence branching (up to max_hops+2) + salient long-range context
        H_buf = cfg.max_hops + 4
        char_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0

        # Salient long-range context ring (tracks the 2 most salient rare chars in the past ~40 chars!)
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

        # Active frontier width per stream: supports parallel branching (up to 4 simultaneous branches/hop)
        MAX_FRONT = 4
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=device)

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]

            char_hist[:, hist_ptr] = x_t
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            # Update salient long-range context memory (decaying rarity tracker, zero rules!)
            ctx_score.mul_(0.965)
            # Only inject a char from 6 steps ago into long-range context so it's truly NON-LOCAL!
            if t >= 6:
                c_old = char_hist[:, (cur_idx - 6) % H_buf]
                sal_old = self.char_salience[c_old.clamp(min=0)]
                # Replace weaker slot if salient & not already held
                already = (ctx_char[:, 0] == c_old) | (ctx_char[:, 1] == c_old)
                weak_slot = ctx_score.argmin(dim=1)
                weak_val = ctx_score[arange_B, weak_slot]
                replace = (~already) & (c_old > 0) & (sal_old > weak_val) & (sal_old > 0.75)
                if replace.any():
                    rb = torch.nonzero(replace, as_tuple=False).squeeze(1)
                    rs = weak_slot[rb]
                    ctx_char[rb, rs] = c_old[rb]
                    ctx_score[rb, rs] = sal_old[rb]

            # ==================================================================
            # 1) TRUE EVENT-DRIVEN BRANCHING TRAVERSAL (starts from 1 neuron x_t!)
            # ==================================================================
            # Visited neurons accumulator per stream for readout + local plasticity
            visited_nodes: List[torch.Tensor] = [x_t]  # Hop 0 = root sensory neuron
            visited_gains: List[float] = [1.0]

            # Frontier starts as 1 root neuron per stream: [B, 1]
            frontier = x_t.unsqueeze(1)  # [B, F]
            stream_max_depth = torch.zeros(B, dtype=torch.int32, device=device)
            # Track the deepest local-spine leaf per stream (primary candidate for Neurogenesis)
            deepest_leaf = x_t.clone()
            deepest_span = torch.zeros(B, dtype=torch.int64, device=device)

            for hop in range(1, cfg.max_hops + 1):
                F_cur = frontier.shape[1]
                valid_f = frontier >= 0
                if not valid_f.any():
                    break

                # Flatten active frontier parents for O(1) GPU branch lookup
                fb, ff = torch.nonzero(valid_f, as_tuple=True)
                p_ids = frontier[fb, ff]  # [A]
                p_span = self.local_span[p_ids].to(torch.int64)  # next local dist = p_span + 1

                # (a) Local sequence branch: cond_kind=0, cond = char at dist (p_span + 1)
                next_dist = (p_span + 1).clamp(max=H_buf - 1)
                c_loc = char_hist[fb, (cur_idx - next_dist) % H_buf]
                k0 = torch.zeros_like(p_ids)
                ch_loc = self.edges.lookup(p_ids, k0, c_loc)
                ch_loc = torch.where((c_loc > 0) & (ch_loc >= 0) & self.alive[ch_loc.clamp(min=0)], ch_loc, torch.full_like(ch_loc, -1))

                # (b) Long-range context branch: cond_kind=1, cond = top salient context char
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

                # Pack discovered child branches into next frontier [B, MAX_FRONT]
                next_front = torch.full((B, MAX_FRONT), -1, dtype=torch.int64, device=device)
                # Write local children to slot (ff * 2) % MAX_FRONT and context children to (ff * 2 + 1) % MAX_FRONT
                hit_loc = ch_loc >= 0
                if hit_loc.any():
                    hl_b = fb[hit_loc]
                    hl_c = ch_loc[hit_loc]
                    hl_slot = (ff[hit_loc] * 2) % MAX_FRONT
                    next_front[hl_b, hl_slot] = hl_c
                    stream_max_depth[hl_b] = hop
                    # Update deepest local leaf
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

                # Record every active child column in this hop for readout + local learning
                gain = 1.0 / (1.0 + 0.15 * hop)
                for col in range(MAX_FRONT):
                    col_v = next_front[:, col]
                    if (col_v >= 0).any():
                        visited_nodes.append(col_v)
                        visited_gains.append(gain)

                frontier = next_front

            # Telemetry on dynamic depth & visited neurons
            d_mean = float(stream_max_depth.to(torch.float32).mean().item())
            total_depth_sum += d_mean * B
            cur_md = int(stream_max_depth.max().item())
            if cur_md > max_depth_seen:
                max_depth_seen = cur_md

            # ==================================================================
            # 2) HIERARCHICAL RESIDUAL READOUT (only visited neurons fire!)
            # ==================================================================
            h = torch.zeros((B, D), dtype=torch.float32, device=device)
            logits_sharp.zero_()
            n_vis_step = 0

            for v_ids, v_g in zip(visited_nodes, visited_gains):
                ok = v_ids >= 0
                if not ok.any():
                    continue
                n_vis_step += int(ok.sum().item())
                u_ok = v_ids[ok]
                # Smooth feature wave
                h[ok] += self.W_node[u_ok] * v_g
                # Sharp direct wave (top-K direct chars)
                sd = self.sharp_dst[u_ok]  # [M, K]
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

            # ==================================================================
            # 3) CAUSAL EVALUATION
            # ==================================================================
            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]
            total_nll += float(nll.sum().item())

            top5 = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5[:, 0] == y_t).sum().item())
            total_top5 += int((top5 == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if not train:
                continue

            # ==================================================================
            # 4) BACKPROP-FREE LOCAL PLASTICITY + SURPRISE-DRIVEN NEUROGENESIS!
            # ==================================================================
            probs = torch.exp(log_p)
            p_true = probs[arange_B, y_t]
            surprise = 1.0 - p_true  # [B]

            # Local predictive-coding feature error
            e_feat = self.E[y_t] - (probs @ self.E)
            sat = (1.0 - (h_act * 0.5).pow(2)).clamp(min=0.15)
            e_local = e_feat * sat  # [B, D]

            # Top-1 wrong candidate for sharp lateral inhibition
            top2_p, top2_i = torch.topk(probs, k=2, dim=-1)
            top_bad = torch.where(top2_i[:, 0] != y_t, top2_i[:, 0], top2_i[:, 1])
            p_bad = torch.where(top2_i[:, 0] != y_t, top2_p[:, 0], top2_p[:, 1])

            # Update ONLY the visited neurons along each stream's traversed branches!
            for v_ids, v_g in zip(visited_nodes, visited_gains):
                ok = v_ids >= 0
                if not ok.any():
                    continue
                u_ok = v_ids[ok]
                # Track firing & metabolic utility
                ones_u = torch.ones(u_ok.numel(), dtype=torch.int32, device=device)
                self.fires.scatter_add_(0, u_ok, ones_u)
                self.recent_fires.scatter_add_(0, u_ok, ones_u)

                # Local utility: positive when this neuron's feature aligns with e_local (helps prediction!)
                align = (self.W_node[u_ok] * e_local[ok]).sum(dim=-1) / math.sqrt(D)
                self.utility.scatter_add_(0, u_ok, torch.relu(align + 0.05) * 0.25)

                # Metaplastic feature update
                self.upd_cnt.scatter_add_(0, u_ok, torch.ones(u_ok.numel(), dtype=torch.float32, device=device))
                lr_u = (lr_feat * v_g) / torch.sqrt(1.0 + cfg.metaplasticity * self.upd_cnt[u_ok])
                self.W_node.index_add_(0, u_ok, e_local[ok] * lr_u.unsqueeze(1))

                # Sharp direct synapse update
                self._update_sharp_synapses(u_ok, y_t[ok], surprise[ok], top_bad[ok], p_bad[ok], lr_sharp * v_g)

            # Local output bias & basis E update
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

            # NEUROGENESIS: accumulate pain on surprised deepest leaves; split when pain > threshold!
            high_surp = surprise > cfg.birth_surprise_thresh
            if high_surp.any():
                hs_b = torch.nonzero(high_surp, as_tuple=False).squeeze(1)
                leaves = deepest_leaf[hs_b]
                self.pain.scatter_add_(0, leaves, surprise[hs_b])

                # Check which leaves crossed splitting threshold & haven't hit max depth
                can_split = (self.pain[leaves] >= cfg.birth_pain_thresh) & (self.depth[leaves] < cfg.max_hops)
                if can_split.any():
                    sb = hs_b[can_split]
                    sp = leaves[can_split]

                    # Decide whether to grow a deeper local branch (kind=0) or long-range context branch (kind=1):
                    # If local span < 4, prefer local branch; if local span >= 2 and strong salient ctx exists,
                    # also spawn long-range context branch!
                    l_span = self.local_span[sp].to(torch.int64)
                    next_d = (l_span + 1).clamp(max=H_buf - 1)
                    c_loc = char_hist[sb, (cur_idx - next_d) % H_buf]

                    # Try local branch first when c_loc > 0
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
        """Trace dynamic branching depth & fired path for every character in a sentence!"""
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
        """Inspect the deepest & most active organically grown neurons in the graph!"""
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
