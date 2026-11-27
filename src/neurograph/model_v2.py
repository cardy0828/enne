"""NeuroGraph v0.2 (fast + top-down priming): Multi-hop Neuron-to-Neuron Relay Propagation.

Core mechanisms (100% backprop-free):
1. Fine-grained neurons (sensory chars + bottom-up crystallized chunks) with continuous
   Leaky Integrate-and-Fire (LIF) membrane potential & refractory fatigue.
2. Lateral neuron-to-neuron sparse graph (lat_bank: u -> w) wired automatically by
   physical chunk distance, modulated by strong homeostatic plasticity so rare semantic
   concepts wire strongly to each other instead of collapsing onto stopwords.
3. Multi-hop relay propagation (Hop 0 -> 1 -> ... -> H) with top-k lateral inhibition
   (k-WTA) and refractory push so waves collide at conceptual intersections and spread
   forward without ping-pong oscillation.
4. Top-Down Predictive Priming + Normalized Relay Readout:
   When a multi-hop relayed chunk neuron w (e.g. '戦車', '歴史', 'ドイツ') fires,
   it sends a top-down expectation wave directly to its own onset character
   (chunk_first_char[w], e.g. '戦', '歴', 'ド') AND contributes normalized associative
   readout signals—boosting semantic content-word prediction without drowning out
   local grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
import torch

from neurograph.model import SparseSynapseBank


@dataclass
class NeuroGraphV2Config:
    vocab_size: int
    max_chunks: int = 10000
    k_readout: int = 48
    k_lateral: int = 24
    char_delays: Tuple[int, ...] = (1, 2, 3, 4, 6)
    # Multi-hop relay settings
    n_hops: int = 2
    k_wta: int = 8
    membrane_decay: float = 0.84
    hop_decay: float = 0.75
    node_fatigue_decay: float = 0.86
    node_fatigue_inc: float = 0.70
    # Lateral auto-wiring by physical chunk distance
    lateral_window: int = 5
    lateral_lr: float = 0.10
    credit_lr: float = 0.05
    # Local surprise plasticity (zero backprop)
    lr: float = 0.24
    neg_top_m: int = 8
    weight_decay: float = 0.9999
    prune_threshold: float = 0.03
    # Chunk crystallization
    chunk_min_count: int = 14
    chunk_add_per_round: int = 3500
    # Gains
    gain_char_base: float = 1.0
    gain_chunk: float = 1.30
    gain_prev_chunk: float = 0.75
    gain_relay_out: float = 0.45
    gain_topdown_prime: float = 0.35
    # Output char fatigue
    char_fatigue_decay: float = 0.85
    char_fatigue_inc: float = 0.14


class NeuroGraphV2:
    """Backprop-free, multi-hop spreading-activation + top-down priming neural graph."""

    def __init__(self, cfg: NeuroGraphV2Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size
        self.max_nodes = self.V + cfg.max_chunks
        self.num_chunks = 0

        self.out_bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        # 1) Sensory character delay readout banks
        self.char_banks: List[SparseSynapseBank] = [
            SparseSynapseBank(self.V, cfg.k_readout, device)
            for _ in cfg.char_delays
        ]

        # 2) Crystallized chunk registry
        self.pair_keys = torch.empty(0, dtype=torch.int64, device=device)
        self.pair_vals = torch.empty(0, dtype=torch.int32, device=device)
        self.chunk_pairs: List[Tuple[int, int]] = []
        self.chunk_len = torch.ones(self.max_nodes, dtype=torch.int32, device=device)
        # First character of every node (for top-down predictive priming!)
        self.chunk_first_char = torch.zeros(self.max_nodes, dtype=torch.int64, device=device)
        self.chunk_first_char[: self.V] = torch.arange(self.V, dtype=torch.int64, device=device)

        # Direct chunk & prev-chunk readout banks
        self.chunk_bank = SparseSynapseBank(self.max_nodes, cfg.k_readout, device)
        self.prev_chunk_bank = SparseSynapseBank(self.max_nodes, cfg.k_readout // 2, device)

        # 3) NEURON-TO-NEURON Lateral Synapse Graph (node -> node)
        self.lat_bank = SparseSynapseBank(self.max_nodes, cfg.k_lateral, device)
        self.relay_out_bank = SparseSynapseBank(self.max_nodes, cfg.k_readout // 2, device)
        # Adaptive per-hop top-down priming & relay readout gains (learned locally!)
        self.prime_scale = torch.full((max(1, cfg.n_hops),), cfg.gain_topdown_prime, dtype=torch.float32, device=device)

        # Homeostatic firing frequency counter per node
        self.node_freq = torch.ones(self.max_nodes, dtype=torch.float32, device=device)
        self.total_chunk_events = 1.0
        self.total_chars_seen = 0

    @property
    def num_active_nodes(self) -> int:
        return self.V + self.num_chunks

    def decode_node(self, node_id: int, id_to_char: List[str], max_depth: int = 32) -> str:
        if node_id < 0:
            return "<NONE>"
        if node_id < self.V:
            return id_to_char[node_id]
        if max_depth <= 0 or (node_id - self.V) >= len(self.chunk_pairs):
            return "?"
        l, r = self.chunk_pairs[node_id - self.V]
        return self.decode_node(l, id_to_char, max_depth - 1) + self.decode_node(r, id_to_char, max_depth - 1)

    @torch.no_grad()
    def _lookup_pairs(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.pair_keys.numel() == 0:
            return torch.full_like(left, -1, dtype=torch.int64)
        keys = (left.to(torch.int64) << 20) | right.to(torch.int64)
        pos = torch.searchsorted(self.pair_keys, keys)
        valid = pos < self.pair_keys.numel()
        out = torch.full_like(left, -1, dtype=torch.int64)
        if valid.any():
            v_pos = pos[valid]
            matched = self.pair_keys[v_pos] == keys[valid]
            if matched.any():
                idx = torch.nonzero(valid, as_tuple=False).squeeze(1)[matched]
                out[idx] = self.pair_vals[v_pos[matched]].to(torch.int64)
        return out

    @torch.no_grad()
    def crystallize_chunks(self, token_stream: np.ndarray, sample_len: int = 800_000) -> int:
        capacity_left = self.cfg.max_chunks - self.num_chunks
        if capacity_left <= 0:
            return 0

        cur = torch.from_numpy(token_stream[:sample_len].astype(np.int64)).to(self.device)
        if self.pair_keys.numel() > 0:
            for _ in range(6):
                if cur.numel() < 2:
                    break
                merged = self._lookup_pairs(cur[:-1], cur[1:])
                hit = merged >= 0
                if not hit.any():
                    break
                hit_np = hit.cpu().numpy()
                keep = np.zeros_like(hit_np)
                i = 0
                n_h = len(hit_np)
                while i < n_h:
                    if hit_np[i]:
                        keep[i] = True
                        i += 2
                    else:
                        i += 1
                keep_t = torch.from_numpy(keep).to(self.device)
                out_mask = torch.ones(cur.numel(), dtype=torch.bool, device=self.device)
                out_mask[1:] &= ~keep_t
                new_cur = cur.clone()
                new_cur[:-1] = torch.where(keep_t, merged, new_cur[:-1])
                cur = new_cur[out_mask]

        if cur.numel() < 2:
            return 0

        left, right = cur[:-1], cur[1:]
        valid = (left > 0) & (right > 0)
        keys = (left[valid] << 20) | right[valid]
        uniq_keys, counts = torch.unique(keys, return_counts=True)

        if self.pair_keys.numel() > 0:
            pos = torch.searchsorted(self.pair_keys, uniq_keys)
            in_bounds = pos < self.pair_keys.numel()
            already = torch.zeros_like(in_bounds)
            already[in_bounds] = self.pair_keys[pos[in_bounds]] == uniq_keys[in_bounds]
            keep = (~already) & (counts >= self.cfg.chunk_min_count)
        else:
            keep = counts >= self.cfg.chunk_min_count

        cand_keys = uniq_keys[keep]
        cand_counts = counts[keep]
        if cand_keys.numel() == 0:
            return 0

        n_add = min(self.cfg.chunk_add_per_round, capacity_left, int(cand_keys.numel()))
        top_idx = torch.topk(cand_counts, k=n_add).indices
        chosen_keys = cand_keys[top_idx]

        new_ids = torch.arange(
            self.V + self.num_chunks,
            self.V + self.num_chunks + n_add,
            dtype=torch.int32,
            device=self.device,
        )
        for idx_i, k in enumerate(chosen_keys.cpu().tolist()):
            l_id = int(k >> 20)
            r_id = int(k & ((1 << 20) - 1))
            self.chunk_pairs.append((l_id, r_id))
            nid = self.V + self.num_chunks + idx_i
            self.chunk_len[nid] = self.chunk_len[l_id] + self.chunk_len[r_id]
            self.chunk_first_char[nid] = self.chunk_first_char[l_id]

        self.num_chunks += n_add
        all_keys = torch.cat([self.pair_keys, chosen_keys])
        all_vals = torch.cat([self.pair_vals, new_ids])
        sort_order = torch.argsort(all_keys)
        self.pair_keys = all_keys[sort_order]
        self.pair_vals = all_vals[sort_order]
        return n_add

    @torch.no_grad()
    def _homeostatic_salience(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Strong homeostatic scaling (freq^-0.75) + length bonus so semantic words
        dominate associative waves rather than high-frequency punctuation/particles."""
        safe = node_ids.clamp(0, self.max_nodes - 1)
        freq = self.node_freq[safe]
        avg_f = max(1.0, self.total_chunk_events / max(1, self.num_active_nodes))
        # Stronger damping on ultra-frequent stopwords
        sal = ((avg_f / (freq + 4.0)) ** 0.70).clamp(0.05, 2.2)
        # Multi-char chunks carry richer semantics than raw single chars
        len_bonus = torch.where(self.chunk_len[safe] >= 2, 1.15, 0.70)
        return sal * len_bonus

    @torch.no_grad()
    def _propagate_lateral_hop(
        self,
        src_ids: torch.Tensor,      # [B, S] int64
        src_act: torch.Tensor,      # [B, S] float32
        membrane: torch.Tensor,     # [B, N] float32 (modified in-place)
        node_fatigue: torch.Tensor, # [B, N] float32 (modified in-place)
        k_wta: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = src_ids.shape
        N = membrane.shape[1]

        valid_src = (src_ids >= 0) & (src_ids < N) & (src_act > 1e-4)
        if not valid_src.any():
            return (
                torch.full((B, k_wta), -1, dtype=torch.int64, device=self.device),
                torch.zeros((B, k_wta), dtype=torch.float32, device=self.device),
            )

        safe_src = src_ids.clamp(0, N - 1)
        dst = self.lat_bank.dst[safe_src]       # [B, S, K_lat]
        w = self.lat_bank.weight[safe_src]      # [B, S, K_lat]

        mask = valid_src.unsqueeze(-1) & (dst >= 0) & (dst < N) & (w > 0.0)
        wave = (w * src_act.unsqueeze(-1)) * mask.to(w.dtype)

        dst_flat = dst.clamp(0, N - 1).reshape(B, -1).to(torch.int64)
        wave_flat = wave.reshape(B, -1)
        membrane.scatter_add_(1, dst_flat, wave_flat)

        # Lateral inhibition (k-WTA) over (membrane - fatigue)
        eff = membrane - node_fatigue
        k_eff = min(k_wta, N)
        top_v, top_i = torch.topk(eff, k=k_eff, dim=-1)

        fired_mask = top_v > 0.10
        next_ids = torch.where(fired_mask, top_i, torch.full_like(top_i, -1))
        raw_act = torch.where(fired_mask, torch.tanh(top_v * 0.6), torch.zeros_like(top_v))
        # Normalize per-stream hop energy so adding hops never blows up logit variance!
        norm_act = raw_act / (raw_act.sum(dim=1, keepdim=True) + 1.0)

        safe_next = top_i.clamp(0, N - 1)
        node_fatigue.scatter_add_(1, safe_next, fired_mask.to(torch.float32) * self.cfg.node_fatigue_inc)
        return next_ids, norm_act

    @torch.no_grad()
    def _apply_relay_readout(
        self,
        hop_winners: List[Tuple[torch.Tensor, torch.Tensor]],
        logits: torch.Tensor,
    ) -> None:
        """Apply both top-down predictive priming (chunk -> onset char) and associative readout."""
        cfg = self.cfg
        for h_idx, (w_ids, w_act) in enumerate(hop_winners):
            h_scale = 1.0 / (1.0 + 0.35 * h_idx)
            # (1) Learned associative readout wave (normalized so it stays clean)
            self.relay_out_bank.propagate(w_ids, w_act * (cfg.gain_relay_out * h_scale), logits)
            # (2) Zero-shot Top-Down Predictive Priming: relayed chunk primes its own first char!
            if cfg.gain_topdown_prime > 0:
                valid = (w_ids >= 0) & (w_ids < self.max_nodes) & (w_act > 1e-4)
                if valid.any():
                    safe_ids = w_ids.clamp(0, self.max_nodes - 1)
                    first_ch = self.chunk_first_char[safe_ids]  # [B, k_wta]
                    prime_amp = (w_act * (self.prime_scale[h_idx] * h_scale)) * valid.to(w_act.dtype)
                    logits.scatter_add_(1, first_ch, prime_amp)

    @torch.no_grad()
    def run_stream(
        self,
        ids: np.ndarray,
        batch_streams: int = 512,
        train: bool = True,
        lr_scale: float = 1.0,
    ) -> Dict[str, float]:
        cfg = self.cfg
        total_len = len(ids)
        stream_len = total_len // batch_streams
        if stream_len < 16:
            batch_streams = max(1, total_len // 16)
            stream_len = total_len // batch_streams

        data = torch.from_numpy(
            ids[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(self.device)
        B, T = data.shape
        N = self.num_active_nodes

        max_delay = max(cfg.char_delays)
        char_hist = torch.full((B, max_delay), -1, dtype=torch.int64, device=self.device)
        hist_ptr = 0

        cur_chunk = torch.full((B,), -1, dtype=torch.int64, device=self.device)
        prev_chunk = torch.full((B,), -1, dtype=torch.int64, device=self.device)

        W = cfg.lateral_window
        chunk_hist = torch.full((B, W), -1, dtype=torch.int64, device=self.device)
        chunk_ptr = torch.zeros(B, dtype=torch.int64, device=self.device)

        membrane = torch.zeros((B, N), dtype=torch.float32, device=self.device) if cfg.n_hops > 0 else None
        node_fatigue = torch.zeros((B, N), dtype=torch.float32, device=self.device) if cfg.n_hops > 0 else None
        char_fatigue = torch.zeros((B, self.V), dtype=torch.float32, device=self.device)

        delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]
        lr = cfg.lr * lr_scale

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0

        logits = torch.empty((B, self.V), dtype=torch.float32, device=self.device)
        arange_B = torch.arange(B, device=self.device)
        hop_winners: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]

            char_hist[:, hist_ptr] = x_t
            cur_hist_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % max_delay

            completed = torch.zeros(B, dtype=torch.bool, device=self.device)
            if self.num_chunks > 0:
                merged = self._lookup_pairs(cur_chunk, x_t)
                hit = (cur_chunk >= 0) & (merged >= 0)
                completed = (~hit) & (cur_chunk >= 0)
                if completed.any():
                    prev_chunk = torch.where(completed, cur_chunk, prev_chunk)
                    c_idx = torch.nonzero(completed, as_tuple=False).squeeze(1)
                    comp_nodes = cur_chunk[c_idx]
                    c_ptr = chunk_ptr[c_idx]
                    chunk_hist[c_idx, c_ptr] = comp_nodes

                    if train and cfg.n_hops > 0:
                        self.node_freq.scatter_add_(
                            0, comp_nodes, torch.ones(comp_nodes.numel(), dtype=torch.float32, device=self.device)
                        )
                        self.total_chunk_events += float(comp_nodes.numel())

                        # Batched single-call lateral auto-wiring across all physical chunk distances!
                        dst_sal = self._homeostatic_salience(comp_nodes)
                        lat_srcs, lat_dsts, lat_dels = [], [], []
                        for dist in range(1, W):
                            prev_slot = (c_ptr - dist) % W
                            u_c = chunk_hist[c_idx, prev_slot]
                            valid_u = (u_c >= 0) & (u_c != comp_nodes)
                            if valid_u.any():
                                d_gain = (cfg.lateral_lr * lr_scale) / (1.0 + 0.55 * (dist - 1))
                                u_v = u_c[valid_u]
                                w_v = comp_nodes[valid_u]
                                lat_srcs.append(u_v)
                                lat_dsts.append(w_v)
                                lat_dels.append(d_gain * dst_sal[valid_u])
                                if dist <= 3:
                                    lat_srcs.append(w_v)
                                    lat_dsts.append(u_v)
                                    lat_dels.append((d_gain * 0.35) * self._homeostatic_salience(u_v))
                        if lat_srcs:
                            self.lat_bank.local_update(
                                torch.cat(lat_srcs, dim=0),
                                torch.cat(lat_dsts, dim=0),
                                torch.cat(lat_dels, dim=0),
                                allow_sprout=True,
                            )

                    chunk_ptr[c_idx] = (c_ptr + 1) % W
                cur_chunk = torch.where(hit, merged, x_t)
            else:
                prev_chunk = cur_chunk
                cur_chunk = x_t

            char_fatigue.mul_(cfg.char_fatigue_decay)
            char_fatigue[arange_B, x_t] += cfg.char_fatigue_inc

            # 3) MULTI-HOP NEURON-TO-NEURON RELAY PROPAGATION
            if cfg.n_hops > 0 and membrane is not None and node_fatigue is not None:
                membrane.mul_(cfg.membrane_decay)
                node_fatigue.mul_(cfg.node_fatigue_decay)

                sal_cur = self._homeostatic_salience(cur_chunk)
                membrane[arange_B, cur_chunk.clamp(0, N - 1)] += 0.70 * sal_cur
                if completed.any():
                    c_idx = torch.nonzero(completed, as_tuple=False).squeeze(1)
                    pc = prev_chunk[c_idx].clamp(0, N - 1)
                    membrane[c_idx, pc] += 1.15 * self._homeostatic_salience(pc)

                # Update multi-hop spreading activation every 2 steps (smooth carry-over)
                if t % 2 == 0:
                    hop_winners = []
                    seed_ids = torch.stack([cur_chunk, prev_chunk], dim=1)
                    seed_act = torch.stack(
                        [
                            sal_cur,
                            torch.where(
                                prev_chunk >= 0,
                                self._homeostatic_salience(prev_chunk) * 0.8,
                                torch.zeros(B, device=self.device),
                            ),
                        ],
                        dim=1,
                    )
                    node_fatigue[arange_B, cur_chunk.clamp(0, N - 1)] += cfg.node_fatigue_inc * 0.5

                    cur_ids, cur_act = seed_ids, seed_act
                    for h in range(cfg.n_hops):
                        next_ids, next_act = self._propagate_lateral_hop(
                            cur_ids, cur_act, membrane, node_fatigue, k_wta=cfg.k_wta
                        )
                        hop_winners.append((next_ids, next_act))
                        cur_ids, cur_act = next_ids, next_act * cfg.hop_decay

                    membrane.clamp_(0.0, 5.0)
                    node_fatigue.clamp_(0.0, 5.0)

            # 4) READOUT WAVE CONVERGENCE
            logits.copy_(self.out_bias.unsqueeze(0))

            for d_idx, d in enumerate(cfg.char_delays):
                src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                act = torch.full((B,), delay_gains[d_idx], dtype=torch.float32, device=self.device)
                self.char_banks[d_idx].propagate(src, act, logits)

            if self.num_chunks > 0:
                is_chunk = cur_chunk >= self.V
                c_act = torch.where(
                    is_chunk,
                    torch.full((B,), cfg.gain_chunk, dtype=torch.float32, device=self.device),
                    torch.zeros(B, dtype=torch.float32, device=self.device),
                )
                self.chunk_bank.propagate(cur_chunk, c_act, logits)

                pc_act = torch.where(
                    prev_chunk >= 0,
                    torch.full((B,), cfg.gain_prev_chunk, dtype=torch.float32, device=self.device),
                    torch.zeros(B, dtype=torch.float32, device=self.device),
                )
                self.prev_chunk_bank.propagate(prev_chunk, pc_act, logits)

            if hop_winners:
                self._apply_relay_readout(hop_winners, logits)

            logits.sub_(char_fatigue)

            # 5) CAUSAL EVALUATION
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[arange_B, y_t]
            total_nll += float(nll.sum().item())

            top5_idx = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5_idx[:, 0] == y_t).sum().item())
            total_top5 += int((top5_idx == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if not train:
                continue

            # 6) BACKPROP-FREE SURPRISE-DRIVEN LOCAL PLASTICITY
            probs = torch.exp(log_probs)
            p_true = probs[arange_B, y_t]
            pos_surprise = 1.0 - p_true

            topm_p, topm_idx = torch.topk(probs, k=cfg.neg_top_m, dim=-1)
            neg_mask = topm_idx != y_t.unsqueeze(1)
            neg_err = topm_p * neg_mask.to(topm_p.dtype)

            bias_grad = torch.zeros(self.V, dtype=torch.float32, device=self.device)
            bias_grad.scatter_add_(0, y_t, pos_surprise)
            bias_grad.scatter_add_(0, topm_idx.reshape(-1), -neg_err.reshape(-1))
            self.out_bias.add_(bias_grad * (lr * 0.02 / B))
            self.out_bias.clamp_(-4.0, 4.0)

            def update_bank(bank: SparseSynapseBank, src: torch.Tensor, amp: torch.Tensor, scale: float = 1.0):
                M = cfg.neg_top_m
                pos_delta = (lr * scale) * amp * pos_surprise
                src_rep = src.unsqueeze(1).expand(B, M).reshape(-1)
                dst_rep = topm_idx.reshape(-1)
                neg_delta = -(lr * scale) * amp.unsqueeze(1).expand(B, M).reshape(-1) * neg_err.reshape(-1)
                bank.local_update(
                    torch.cat([src, src_rep], dim=0),
                    torch.cat([y_t, dst_rep], dim=0),
                    torch.cat([pos_delta, neg_delta], dim=0),
                    allow_sprout=True,
                )

            for d_idx, d in enumerate(cfg.char_delays):
                src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                amp = torch.full((B,), delay_gains[d_idx], dtype=torch.float32, device=self.device)
                update_bank(self.char_banks[d_idx], src, amp, scale=0.8)

            if self.num_chunks > 0:
                is_chunk = cur_chunk >= self.V
                c_amp = torch.where(
                    is_chunk,
                    torch.full((B,), cfg.gain_chunk, dtype=torch.float32, device=self.device),
                    torch.zeros(B, dtype=torch.float32, device=self.device),
                )
                update_bank(self.chunk_bank, cur_chunk, c_amp, scale=1.0)
                pc_amp = torch.where(
                    prev_chunk >= 0,
                    torch.full((B,), cfg.gain_prev_chunk, dtype=torch.float32, device=self.device),
                    torch.zeros(B, dtype=torch.float32, device=self.device),
                )
                update_bank(self.prev_chunk_bank, prev_chunk, pc_amp, scale=0.7)

            # Batched single-call update for ALL multi-hop relayed neurons!
            if hop_winners:
                r_srcs, r_dsts, r_dels = [], [], []
                M2 = min(3, cfg.neg_top_m)
                for h_idx, (w_ids, w_act) in enumerate(hop_winners):
                    h_scale = 1.0 / (1.0 + 0.35 * h_idx)
                    gain = cfg.gain_relay_out * h_scale
                    k_upd = min(4, w_ids.shape[1])
                    s_k = w_ids[:, :k_upd]
                    a_k = w_act[:, :k_upd] * gain

                    r_srcs.append(s_k.reshape(-1))
                    r_dsts.append(y_t.unsqueeze(1).expand(B, k_upd).reshape(-1))
                    r_dels.append((lr * 0.70) * a_k.reshape(-1) * pos_surprise.unsqueeze(1).expand(B, k_upd).reshape(-1))

                    r_srcs.append(s_k.unsqueeze(2).expand(B, k_upd, M2).reshape(-1))
                    r_dsts.append(topm_idx[:, :M2].unsqueeze(1).expand(B, k_upd, M2).reshape(-1))
                    r_dels.append(
                        -(lr * 0.70)
                        * a_k.unsqueeze(2).expand(B, k_upd, M2).reshape(-1)
                        * neg_err[:, :M2].unsqueeze(1).expand(B, k_upd, M2).reshape(-1)
                    )

                    # Locally adapt top-down priming gain per hop based on onset-char hit vs miss!
                    valid_p = (w_ids >= 0) & (w_ids < self.max_nodes) & (w_act > 1e-4)
                    if valid_p.any():
                        first_ch = self.chunk_first_char[w_ids.clamp(0, self.max_nodes - 1)]
                        hit_y = (first_ch == y_t.unsqueeze(1)) & valid_p
                        # Local gradient of priming scale: positive when onset matches y_t, negative by probs[first_ch]
                        p_fc = probs.gather(1, first_ch)
                        err_fc = hit_y.to(torch.float32) - p_fc
                        g_prime = (err_fc * w_act * valid_p.to(torch.float32)).sum() / B
                        self.prime_scale[h_idx] = (self.prime_scale[h_idx] + 0.05 * lr_scale * g_prime).clamp(0.0, 1.5)

                self.relay_out_bank.local_update(
                    torch.cat(r_srcs, dim=0),
                    torch.cat(r_dsts, dim=0),
                    torch.cat(r_dels, dim=0),
                    allow_sprout=True,
                )

                # Three-factor credit on Hop-1 lateral synapses (every 2 steps)
                if t % 2 == 0 and cfg.credit_lr > 0:
                    w0, a0 = hop_winners[0][0][:, 0], hop_winners[0][1][:, 0]
                    mod = (p_true - 0.33) * a0 * (cfg.credit_lr * lr_scale)
                    self.lat_bank.local_update(cur_chunk, w0, mod, allow_sprout=False)

        if train:
            self.total_chars_seen += total_steps

        avg_nll = total_nll / max(1, total_steps)
        bpc = avg_nll / math.log(2.0)
        return {
            "steps": total_steps,
            "nll": avg_nll,
            "bpc": bpc,
            "ppl": math.exp(min(avg_nll, 20.0)),
            "top1": total_top1 / max(1, total_steps),
            "top5": total_top5 / max(1, total_steps),
            "prime_scale": [round(float(x), 3) for x in self.prime_scale.cpu().tolist()],
        }

    @torch.no_grad()
    def metabolize_all(self) -> Dict[str, int]:
        pruned = 0
        for b in self.all_banks():
            pruned += b.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        return {
            "pruned_now": pruned,
            "active_synapses": self.total_active_synapses(),
            "lateral_synapses": self.lat_bank.active_synapse_count(),
            "total_evictions": self.total_evictions(),
            "total_pruned": self.total_pruned(),
        }

    def all_banks(self) -> List[SparseSynapseBank]:
        return list(self.char_banks) + [
            self.chunk_bank,
            self.prev_chunk_bank,
            self.lat_bank,
            self.relay_out_bank,
        ]

    def total_active_synapses(self) -> int:
        return sum(b.active_synapse_count() for b in self.all_banks())

    def total_evictions(self) -> int:
        return sum(b.total_evictions for b in self.all_banks())

    def total_pruned(self) -> int:
        return sum(b.total_pruned for b in self.all_banks())

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_chars: int = 100,
        temperature: float = 0.65,
        top_k: int = 15,
    ) -> List[int]:
        cfg = self.cfg
        N = self.num_active_nodes
        max_delay = max(cfg.char_delays)
        char_hist = torch.full((1, max_delay), -1, dtype=torch.int64, device=self.device)
        hist_ptr = 0
        cur_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        prev_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        membrane = torch.zeros((1, N), dtype=torch.float32, device=self.device) if cfg.n_hops > 0 else None
        node_fatigue = torch.zeros((1, N), dtype=torch.float32, device=self.device) if cfg.n_hops > 0 else None
        char_fatigue = torch.zeros((1, self.V), dtype=torch.float32, device=self.device)
        delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]

        def step_in(ch_id: int) -> torch.Tensor:
            nonlocal cur_chunk, prev_chunk, hist_ptr
            x_t = torch.tensor([ch_id], dtype=torch.int64, device=self.device)
            char_hist[0, hist_ptr] = ch_id
            cur_hist_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % max_delay

            completed = False
            if self.num_chunks > 0:
                merged = self._lookup_pairs(cur_chunk, x_t)
                hit = (cur_chunk >= 0) & (merged >= 0)
                if (not hit.item()) and cur_chunk.item() >= 0:
                    prev_chunk = cur_chunk.clone()
                    completed = True
                cur_chunk = torch.where(hit, merged, x_t)
            else:
                prev_chunk = cur_chunk
                cur_chunk = x_t

            char_fatigue.mul_(cfg.char_fatigue_decay)
            char_fatigue[0, ch_id] += cfg.char_fatigue_inc

            hop_winners = []
            if cfg.n_hops > 0 and membrane is not None and node_fatigue is not None:
                membrane.mul_(cfg.membrane_decay)
                node_fatigue.mul_(cfg.node_fatigue_decay)
                sal_cur = self._homeostatic_salience(cur_chunk)
                membrane[0, cur_chunk.clamp(0, N - 1)] += 0.70 * sal_cur
                if completed:
                    pc = prev_chunk.clamp(0, N - 1)
                    membrane[0, pc] += 1.15 * self._homeostatic_salience(pc)

                seed_ids = torch.stack([cur_chunk, prev_chunk], dim=1)
                seed_act = torch.stack(
                    [
                        sal_cur,
                        self._homeostatic_salience(prev_chunk) * 0.8 if prev_chunk.item() >= 0 else torch.zeros(1, device=self.device),
                    ],
                    dim=1,
                )
                node_fatigue[0, cur_chunk.clamp(0, N - 1)] += cfg.node_fatigue_inc * 0.5
                c_ids, c_act = seed_ids, seed_act
                for h in range(cfg.n_hops):
                    n_ids, n_act = self._propagate_lateral_hop(c_ids, c_act, membrane, node_fatigue, k_wta=cfg.k_wta)
                    hop_winners.append((n_ids, n_act))
                    c_ids, c_act = n_ids, n_act * cfg.hop_decay
                membrane.clamp_(0.0, 5.0)
                node_fatigue.clamp_(0.0, 5.0)

            logits = self.out_bias.unsqueeze(0).clone()
            for d_idx, d in enumerate(cfg.char_delays):
                src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                act = torch.tensor([delay_gains[d_idx]], dtype=torch.float32, device=self.device)
                self.char_banks[d_idx].propagate(src, act, logits)

            if self.num_chunks > 0:
                if cur_chunk.item() >= self.V:
                    self.chunk_bank.propagate(cur_chunk, torch.tensor([cfg.gain_chunk], device=self.device), logits)
                if prev_chunk.item() >= 0:
                    self.prev_chunk_bank.propagate(prev_chunk, torch.tensor([cfg.gain_prev_chunk], device=self.device), logits)

            if hop_winners:
                self._apply_relay_readout(hop_winners, logits)

            logits.sub_(char_fatigue)
            logits[0, 0] = -1e9
            return logits[0]

        out = list(prompt_ids)
        logits = None
        for ch in prompt_ids:
            logits = step_in(ch)

        for _ in range(max_new_chars):
            assert logits is not None
            scaled = logits / max(temperature, 1e-4)
            if top_k > 0 and top_k < self.V:
                vals, idx = torch.topk(scaled, k=top_k)
                probs = torch.softmax(vals, dim=-1)
                pick = int(idx[torch.multinomial(probs, 1).item()].item())
            else:
                probs = torch.softmax(scaled, dim=-1)
                pick = int(torch.multinomial(probs, 1).item())
            out.append(pick)
            logits = step_in(pick)
        return out

    @torch.no_grad()
    def trace_associations(
        self,
        prompt_ids: List[int],
        id_to_char: List[str],
        hops: int = 3,
        top_n: int = 6,
    ) -> Dict[str, List[str]]:
        N = self.num_active_nodes
        cur_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        prev_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        membrane = torch.zeros((1, N), dtype=torch.float32, device=self.device)
        node_fatigue = torch.zeros((1, N), dtype=torch.float32, device=self.device)

        fired_chunks = []
        for ch_id in prompt_ids:
            x_t = torch.tensor([ch_id], dtype=torch.int64, device=self.device)
            if self.num_chunks > 0:
                merged = self._lookup_pairs(cur_chunk, x_t)
                hit = (cur_chunk >= 0) & (merged >= 0)
                if (not hit.item()) and cur_chunk.item() >= 0:
                    prev_chunk = cur_chunk.clone()
                    fired_chunks.append(int(prev_chunk.item()))
                cur_chunk = torch.where(hit, merged, x_t)
            else:
                prev_chunk = cur_chunk
                cur_chunk = x_t
        if cur_chunk.item() >= 0:
            fired_chunks.append(int(cur_chunk.item()))

        seeds = list(dict.fromkeys(fired_chunks[-4:]))
        seed_t = torch.tensor([seeds], dtype=torch.int64, device=self.device)
        seed_a = self._homeostatic_salience(seed_t[0]).unsqueeze(0)
        for s_id, s_val in zip(seeds, seed_a[0].tolist()):
            if 0 <= s_id < N:
                membrane[0, s_id] += s_val
                node_fatigue[0, s_id] += self.cfg.node_fatigue_inc

        trace: Dict[str, List[str]] = {
            "Hop0_Stimulus": [self.decode_node(i, id_to_char) for i in seeds]
        }
        c_ids, c_act = seed_t, seed_a
        for h in range(1, hops + 1):
            n_ids, n_act = self._propagate_lateral_hop(c_ids, c_act, membrane, node_fatigue, k_wta=top_n)
            labels = []
            for nid, act in zip(n_ids[0].tolist(), n_act[0].tolist()):
                if nid >= 0 and act > 0.02:
                    labels.append(f"{self.decode_node(nid, id_to_char)}({act:.2f})")
            trace[f"Hop{h}_Relay"] = labels
            c_ids, c_act = n_ids, n_act * self.cfg.hop_decay
        return trace
