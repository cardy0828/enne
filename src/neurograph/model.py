"""NeuroGraph v0.1: Backprop-free, event-driven, self-wiring neural graph.

Core principles:
1. Fine-grained neurons (sensory chars, crystallized chunks, emergent abstract hubs)
   with individual activation, fatigue/adaptation, and firing events. No layers, no backprop.
2. Physical text distance -> automatic delayed synapse wiring (near, mid, far resonance).
3. Sparse fixed-capacity outgoing synapse slots per neuron: new connections automatically
   evict the weakest/lowest-energy synapse (built-in pruning / metabolism).
4. Surprise-driven local plasticity: only active neurons at the moment of prediction error
   strengthen synapses to the true target and locally inhibit top false-firing targets.
5. Wiring-compression hub emergence: neurons with similar outgoing synapse profiles are
   clustered into shared abstract hub neurons, enabling generalization & wiring diet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch


@dataclass
class NeuroGraphConfig:
    vocab_size: int
    # Capacity limits
    max_chunks: int = 32768
    num_hubs: int = 512
    # Outgoing synapse capacity per neuron (enforces sparsity + auto-eviction)
    k_synapses: int = 48
    # Physical distance delays (exact character delays + chunk/hub/resonance channels)
    # Channels:
    #   0..num_char_delays-1 : sensory char at distance d+1 (d=0..num_char_delays-1)
    #   chunk_ch             : crystallized chunk neurons ending at t (multi-char scale)
    #   prev_chunk_ch        : previous chunk neuron (word-to-word distance!)
    #   hub_ch               : emergent abstract hub neurons
    #   trace_ch             : long-range decaying resonance (context / topic wave)
    char_delays: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    # Local learning / plasticity
    lr: float = 0.25
    neg_top_m: int = 8            # inhibit top-M false-firing output neurons
    weight_decay: float = 0.99992 # slow metabolic decay per metabolism cycle
    prune_threshold: float = 0.03
    # Feature toggles for clean ablations
    use_surprise_learning: bool = True   # False = pure distance co-occurrence baseline
    use_chunks: bool = True              # bottom-up char -> word/phrase crystallization
    use_hubs: bool = True                # wiring-compression abstract hub emergence
    use_resonance: bool = True           # long-range decaying context wave
    use_fatigue: bool = True             # neuron firing fatigue / adaptation
    # Crystallization & metabolism settings
    chunk_min_count: int = 12
    chunk_add_per_round: int = 4096
    resonance_decay: float = 0.92
    resonance_slots: int = 12            # number of lingering active trace neurons per stream
    fatigue_decay: float = 0.85
    fatigue_strength: float = 0.15
    # Channel gain factors
    gain_char_base: float = 1.0
    gain_chunk: float = 1.35
    gain_prev_chunk: float = 0.85
    gain_hub: float = 0.75
    gain_resonance: float = 0.35


class SparseSynapseBank:
    """GPU sparse outgoing synapse table with fixed capacity K and automatic eviction.

    Each source neuron u in [0, num_sources) owns K synapse slots:
      dst[u, k]    : target output neuron ID (-1 = empty)
      weight[u, k] : signed synapse strength (excitatory > 0, inhibitory < 0)
      energy[u, k] : metabolic utility / usage score (lowest gets evicted when full)
    """

    def __init__(self, num_sources: int, k_slots: int, device: torch.device):
        self.num_sources = num_sources
        self.k_slots = k_slots
        self.device = device
        self.dst = torch.full((num_sources, k_slots), -1, dtype=torch.int32, device=device)
        self.weight = torch.zeros((num_sources, k_slots), dtype=torch.float32, device=device)
        self.energy = torch.zeros((num_sources, k_slots), dtype=torch.float32, device=device)
        self.total_evictions = 0
        self.total_pruned = 0

    @torch.no_grad()
    def propagate(
        self,
        src_ids: torch.Tensor,   # [B] or [B, A] int64, -1 for inactive
        src_act: torch.Tensor,   # [B] or [B, A] float32 firing amplitude
        logits: torch.Tensor,    # [B, V] float32 membrane potential buffer (modified in-place)
    ) -> None:
        """Event-driven wave propagation: only firing src_ids transmit to their connected targets."""
        if src_ids.dim() == 1:
            src_ids = src_ids.unsqueeze(1)
            src_act = src_act.unsqueeze(1)

        B, A = src_ids.shape
        valid_src = (src_ids >= 0) & (src_ids < self.num_sources) & (src_act > 1e-6)
        if not valid_src.any():
            return

        safe_src = src_ids.clamp(0, self.num_sources - 1)
        dst = self.dst[safe_src]       # [B, A, K]
        w = self.weight[safe_src]      # [B, A, K]

        mask = valid_src.unsqueeze(-1) & (dst >= 0)
        contrib = (w * src_act.unsqueeze(-1)) * mask.to(w.dtype)  # [B, A, K]

        # Scatter-add into [B, V] membrane potentials
        dst_flat = dst.clamp(min=0).reshape(B, -1).to(torch.int64)
        contrib_flat = contrib.reshape(B, -1)
        logits.scatter_add_(1, dst_flat, contrib_flat)

    @torch.no_grad()
    def local_update(
        self,
        src_ids: torch.Tensor,     # [M] int64 firing source neurons
        dst_ids: torch.Tensor,     # [M] int64 target output neurons
        deltas: torch.Tensor,      # [M] float32 signed local weight updates
        allow_sprout: bool = True, # allow creating new synapses on surprise
    ) -> None:
        """Apply local synaptic plasticity updates; sprout new synapses & evict weakest if full."""
        if src_ids.numel() == 0:
            return

        valid = (src_ids >= 0) & (src_ids < self.num_sources) & (dst_ids >= 0) & (deltas.abs() > 1e-7)
        if not valid.any():
            return

        src_ids = src_ids[valid]
        dst_ids = dst_ids[valid]
        deltas = deltas[valid]

        # Aggregate duplicate (src, dst) pairs in this batch step to avoid write races
        keys = (src_ids.to(torch.int64) << 20) | dst_ids.to(torch.int64)
        uniq_keys, inv = torch.unique(keys, return_inverse=True)
        agg_deltas = torch.zeros(uniq_keys.numel(), dtype=torch.float32, device=self.device)
        agg_deltas.scatter_add_(0, inv, deltas)

        u_src = (uniq_keys >> 20).to(torch.int64)
        u_dst = (uniq_keys & ((1 << 20) - 1)).to(torch.int32)

        # Look up existing slots for each unique (src, dst)
        cur_dst = self.dst[u_src]                      # [U, K]
        match = cur_dst == u_dst.unsqueeze(1)          # [U, K]
        has_match = match.any(dim=1)                   # [U]

        # 1) Update existing synapses
        if has_match.any():
            m_src = u_src[has_match]
            m_slot = match[has_match].to(torch.int32).argmax(dim=1)
            m_delta = agg_deltas[has_match]
            self.weight[m_src, m_slot] = (self.weight[m_src, m_slot] + m_delta).clamp(-8.0, 8.0)
            self.energy[m_src, m_slot] = (self.energy[m_src, m_slot] * 0.98 + m_delta.abs()).clamp(0.0, 50.0)

        # 2) Sprout new synapses for unmatched positive (excitatory) surprise events
        if allow_sprout:
            need_sprout = (~has_match) & (agg_deltas > 0.0)
            if need_sprout.any():
                s_src = u_src[need_sprout]
                s_dst = u_dst[need_sprout]
                s_delta = agg_deltas[need_sprout]

                # Deduplicate per source neuron so we insert at most 1 new synapse per neuron per step
                # (pick the highest-surprise candidate for each source neuron)
                order = torch.argsort(s_delta, descending=True)
                s_src = s_src[order]
                s_dst = s_dst[order]
                s_delta = s_delta[order]

                # Keep first occurrence per s_src
                uniq_src, first_idx = _first_occurrence(s_src)
                s_src = uniq_src
                s_dst = s_dst[first_idx]
                s_delta = s_delta[first_idx]

                slots_dst = self.dst[s_src]            # [S, K]
                slots_en = self.energy[s_src]          # [S, K]
                empty_mask = slots_dst < 0             # [S, K]
                has_empty = empty_mask.any(dim=1)

                # Choose empty slot if available, otherwise evict lowest-energy slot
                empty_slot = empty_mask.to(torch.int32).argmax(dim=1)
                weakest_slot = slots_en.argmin(dim=1)
                target_slot = torch.where(has_empty, empty_slot, weakest_slot)

                # Only evict if new surprise outweighs weakest existing energy * 0.25 (or slot is empty)
                weakest_en = slots_en[torch.arange(s_src.numel(), device=self.device), target_slot]
                can_write = has_empty | (s_delta > weakest_en * 0.25)

                if can_write.any():
                    w_src = s_src[can_write]
                    w_slot = target_slot[can_write]
                    w_dst = s_dst[can_write]
                    w_delta = s_delta[can_write]
                    self.total_evictions += int((~has_empty[can_write]).sum().item())

                    self.dst[w_src, w_slot] = w_dst
                    self.weight[w_src, w_slot] = w_delta.clamp(-8.0, 8.0)
                    self.energy[w_src, w_slot] = w_delta.abs().clamp(0.05, 50.0)

    @torch.no_grad()
    def metabolize(self, decay: float, prune_threshold: float) -> int:
        """Apply metabolic decay and prune weak/useless synapses."""
        active = self.dst >= 0
        self.weight[active] *= decay
        self.energy[active] *= decay
        dead = active & (self.weight.abs() < prune_threshold) & (self.energy < prune_threshold * 1.5)
        n_pruned = int(dead.sum().item())
        if n_pruned > 0:
            self.dst[dead] = -1
            self.weight[dead] = 0.0
            self.energy[dead] = 0.0
            self.total_pruned += n_pruned
        return n_pruned

    def active_synapse_count(self) -> int:
        return int((self.dst >= 0).sum().item())


def _first_occurrence(sorted_or_unsorted_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return unique values and their first index in the input 1D tensor."""
    n = sorted_or_unsorted_ids.numel()
    idx = torch.arange(n, device=sorted_or_unsorted_ids.device)
    # Stable sort by id so lowest original index comes first
    sort_order = torch.argsort(sorted_or_unsorted_ids, stable=True)
    sorted_vals = sorted_or_unsorted_ids[sort_order]
    sorted_idx = idx[sort_order]
    is_first = torch.ones(n, dtype=torch.bool, device=sorted_or_unsorted_ids.device)
    if n > 1:
        is_first[1:] = sorted_vals[1:] != sorted_vals[:-1]
    return sorted_vals[is_first], sorted_idx[is_first]


class NeuroGraphModel:
    """Event-driven, backprop-free, self-wiring language neural graph."""

    def __init__(self, cfg: NeuroGraphConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.V = cfg.vocab_size

        # Output resting bias (unigram firing baseline, updated locally)
        self.out_bias = torch.zeros(self.V, dtype=torch.float32, device=device)

        # 1) Sensory character delay banks: one sparse bank per physical distance delta
        self.char_banks: List[SparseSynapseBank] = [
            SparseSynapseBank(self.V, cfg.k_synapses, device)
            for _ in cfg.char_delays
        ]

        # 2) Crystallized chunk neurons (IDs 0..V-1 are atomic chars; V..V+num_chunks-1 are chunks)
        self.max_nodes = self.V + cfg.max_chunks
        self.num_chunks = 0
        # Chunk pair lookup table: sorted keys ((left << 20) | right) -> chunk_id
        self.pair_keys = torch.empty(0, dtype=torch.int64, device=device)
        self.pair_vals = torch.empty(0, dtype=torch.int32, device=device)
        # Human-readable chunk text registry (for inspection)
        self.chunk_pairs: List[Tuple[int, int]] = []  # index i -> (left_node, right_node) for node V+i
        self.chunk_len = torch.ones(self.max_nodes, dtype=torch.int32, device=device)

        self.chunk_bank = SparseSynapseBank(self.max_nodes, cfg.k_synapses, device)
        self.prev_chunk_bank = SparseSynapseBank(self.max_nodes, cfg.k_synapses // 2, device)

        # 3) Emergent Abstract Hub neurons (wiring compression)
        self.hub_bank = SparseSynapseBank(cfg.num_hubs, cfg.k_synapses, device)
        # Mapping from node (char or chunk) -> top-2 hub IDs and connection weights
        self.node_hub_ids = torch.full((self.max_nodes, 2), -1, dtype=torch.int32, device=device)
        self.node_hub_w = torch.zeros((self.max_nodes, 2), dtype=torch.float32, device=device)
        self.hubs_initialized = False

        # 4) Long-range decaying context resonance bank
        self.resonance_bank = SparseSynapseBank(self.max_nodes, cfg.k_synapses // 2, device)

        # Stats
        self.total_chars_seen = 0
        self.cooc_counts = torch.zeros(self.V, dtype=torch.float32, device=device)

    @property
    def num_active_nodes(self) -> int:
        return self.V + self.num_chunks

    def decode_node(self, node_id: int, id_to_char: List[str], max_depth: int = 32) -> str:
        """Recursively render a crystallized chunk neuron back to its character string."""
        if node_id < self.V:
            return id_to_char[node_id]
        if max_depth <= 0 or (node_id - self.V) >= len(self.chunk_pairs):
            return "?"
        l, r = self.chunk_pairs[node_id - self.V]
        return self.decode_node(l, id_to_char, max_depth - 1) + self.decode_node(r, id_to_char, max_depth - 1)

    @torch.no_grad()
    def _lookup_pairs(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Vectorized GPU lookup of (left, right) -> chunk_id, or -1 if not crystallized."""
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
        """Bottom-up crystallization: find frequent distance-1 pairs in current chunked stream
        and mint new fine-grained chunk neurons (no BPE tokenizer needed)."""
        if not self.cfg.use_chunks:
            return 0
        capacity_left = self.cfg.max_chunks - self.num_chunks
        if capacity_left <= 0:
            return 0

        # Segment a contiguous slice using existing chunk neurons (greedy left-to-right firing)
        seq = torch.from_numpy(token_stream[:sample_len].astype(np.int64)).to(self.device)
        # Apply existing merges iteratively in vectorized passes
        cur = seq
        if self.pair_keys.numel() > 0:
            # Up to 6 hierarchical passes to build current chunk stream
            for _ in range(6):
                if cur.numel() < 2:
                    break
                merged = self._lookup_pairs(cur[:-1], cur[1:])
                hit = merged >= 0
                if not hit.any():
                    break
                # Resolve overlapping adjacent hits greedily (even/odd non-overlapping pass)
                # Keep hits where previous position was NOT kept (simple non-overlap via 2-step mask)
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
                # Build new compressed sequence
                out_mask = torch.ones(cur.numel(), dtype=torch.bool, device=self.device)
                out_mask[1:] &= ~keep_t
                new_cur = cur.clone()
                new_cur[:-1] = torch.where(keep_t, merged, new_cur[:-1])
                cur = new_cur[out_mask]

        if cur.numel() < 2:
            return 0

        # Count adjacent (distance-1) pairs on GPU
        left = cur[:-1]
        right = cur[1:]
        # Avoid pairing <UNK>=0
        valid = (left > 0) & (right > 0)
        keys = (left[valid] << 20) | right[valid]
        uniq_keys, counts = torch.unique(keys, return_counts=True)

        # Exclude pairs that are already crystallized
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

        # Select top frequent pairs to crystallize into new chunk neurons
        n_add = min(self.cfg.chunk_add_per_round, capacity_left, int(cand_keys.numel()))
        top_idx = torch.topk(cand_counts, k=n_add).indices
        chosen_keys = cand_keys[top_idx]

        new_ids = torch.arange(
            self.V + self.num_chunks,
            self.V + self.num_chunks + n_add,
            dtype=torch.int32,
            device=self.device,
        )

        # Register lengths and human-readable pairs
        chosen_cpu = chosen_keys.cpu().tolist()
        for idx_i, k in enumerate(chosen_cpu):
            l_id = int(k >> 20)
            r_id = int(k & ((1 << 20) - 1))
            self.chunk_pairs.append((l_id, r_id))
            nid = self.V + self.num_chunks + idx_i
            self.chunk_len[nid] = self.chunk_len[l_id] + self.chunk_len[r_id]

        self.num_chunks += n_add

        # Rebuild sorted pair lookup table on GPU
        all_keys = torch.cat([self.pair_keys, chosen_keys])
        all_vals = torch.cat([self.pair_vals, new_ids])
        sort_order = torch.argsort(all_keys)
        self.pair_keys = all_keys[sort_order]
        self.pair_vals = all_vals[sort_order]

        return n_add

    @torch.no_grad()
    def emerge_abstract_hubs(self, n_iters: int = 8) -> Dict[str, float]:
        """Wiring-compression hub emergence:
        Cluster active neurons that share similar outgoing synapse profiles into shared
        abstract hub neurons. This achieves both wiring compression and generalization.
        """
        if not self.cfg.use_hubs:
            return {"hubs_active": 0, "nodes_mapped": 0}

        n_nodes = self.num_active_nodes
        H = self.cfg.num_hubs

        # Build compact projected wiring signature for every active node using a fixed
        # Johnson-Lindenstrauss random projection of its sparse outgoing synapses (out-profile)
        proj_dim = 64
        gen = torch.Generator(device=self.device)
        gen.manual_seed(42)
        rand_proj = torch.randn((self.V, proj_dim), generator=gen, device=self.device) / math.sqrt(proj_dim)

        # Gather outgoing synapse profiles from char_banks[0] (for chars) and chunk_bank (for all nodes)
        dst = self.chunk_bank.dst[:n_nodes]          # [N, K]
        w = self.chunk_bank.weight[:n_nodes]         # [N, K]
        mask = (dst >= 0) & (w > 0.05)

        # For sensory chars (0..V-1), also incorporate direct distance-1 outgoing profile
        c_dst = self.char_banks[0].dst[: self.V]
        c_w = self.char_banks[0].weight[: self.V]
        c_mask = (c_dst >= 0) & (c_w > 0.05)

        sig = torch.zeros((n_nodes, proj_dim), dtype=torch.float32, device=self.device)

        # Vectorized sparse-to-projected signature in chunks to stay tiny in VRAM
        bs = 4096
        for start in range(0, n_nodes, bs):
            end = min(n_nodes, start + bs)
            d_b = dst[start:end].clamp(min=0).to(torch.int64)
            w_b = (w[start:end] * mask[start:end].to(w.dtype)).unsqueeze(-1)
            emb = rand_proj[d_b]  # [bs, K, proj_dim]
            sig[start:end] = (emb * w_b).sum(dim=1)

        # Add distance-1 char outgoing signature for sensory neurons
        for start in range(0, self.V, bs):
            end = min(self.V, start + bs)
            d_b = c_dst[start:end].clamp(min=0).to(torch.int64)
            w_b = (c_w[start:end] * c_mask[start:end].to(c_w.dtype)).unsqueeze(-1)
            sig[start:end] += (rand_proj[d_b] * w_b).sum(dim=1)

        norms = sig.norm(dim=1, keepdim=True)
        valid_nodes = (norms.squeeze(1) > 1e-3)
        n_valid = int(valid_nodes.sum().item())
        if n_valid < H:
            return {"hubs_active": 0, "nodes_mapped": 0}

        unit_sig = torch.zeros_like(sig)
        unit_sig[valid_nodes] = sig[valid_nodes] / norms[valid_nodes]

        # Spherical k-means (local competitive prototype learning) on GPU
        valid_idx = torch.nonzero(valid_nodes, as_tuple=False).squeeze(1)
        perm = torch.randperm(n_valid, generator=gen, device=self.device)[:H]
        centroids = unit_sig[valid_idx[perm]].clone()  # [H, proj_dim]

        v_sig = unit_sig[valid_idx]  # [n_valid, proj_dim]
        for _ in range(n_iters):
            # Assign each valid node to closest hub centroid
            assign = torch.empty(n_valid, dtype=torch.int64, device=self.device)
            for s in range(0, n_valid, 4096):
                e = min(n_valid, s + 4096)
                sims = v_sig[s:e] @ centroids.T
                assign[s:e] = sims.argmax(dim=1)

            new_c = torch.zeros_like(centroids)
            new_c.index_add_(0, assign, v_sig)
            c_norms = new_c.norm(dim=1, keepdim=True)
            alive = c_norms.squeeze(1) > 1e-5
            centroids[alive] = new_c[alive] / c_norms[alive]

        # Connect each valid node to its top-2 abstract hubs
        self.node_hub_ids.fill_(-1)
        self.node_hub_w.zero_()
        for s in range(0, n_valid, 4096):
            e = min(n_valid, s + 4096)
            sims = v_sig[s:e] @ centroids.T  # [chunk, H]
            top_v, top_i = torch.topk(sims, k=2, dim=1)
            # Soft positive weights
            top_v = torch.relu(top_v)
            denom = top_v.sum(dim=1, keepdim=True).clamp(min=1e-5)
            w_norm = top_v / denom
            node_slice = valid_idx[s:e]
            self.node_hub_ids[node_slice] = top_i.to(torch.int32)
            self.node_hub_w[node_slice] = w_norm

        self.hubs_initialized = True
        used_hubs = int(torch.unique(self.node_hub_ids[valid_idx, 0]).numel())
        return {"hubs_active": used_hubs, "nodes_mapped": n_valid}

    @torch.no_grad()
    def run_stream(
        self,
        ids: np.ndarray,
        batch_streams: int = 256,
        train: bool = True,
        lr_scale: float = 1.0,
    ) -> Dict[str, float]:
        """Run event-driven propagation (and optionally local surprise learning) over a corpus slice.

        Splits the sequence into `batch_streams` parallel contiguous streams so RTX 3060
        processes hundreds of sparse neuron events simultaneously with zero backprop.
        """
        cfg = self.cfg
        total_len = len(ids)
        stream_len = total_len // batch_streams
        if stream_len < 16:
            batch_streams = max(1, total_len // 16)
            stream_len = total_len // batch_streams

        # Reshape into [B, T] on GPU
        data = torch.from_numpy(
            ids[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(self.device)
        B, T = data.shape

        max_delay = max(cfg.char_delays)
        # Ring buffer of sensory chars: [B, max_delay] with write pointer (zero copy)
        char_hist = torch.full((B, max_delay), -1, dtype=torch.int64, device=self.device)
        hist_ptr = 0

        # Active chunk tracking per stream:
        #   cur_chunk: current crystallized chunk neuron (starts as char, grows via distance-1 matches)
        #   prev_chunk: previously completed chunk neuron (word-to-word distance!)
        cur_chunk = torch.full((B,), -1, dtype=torch.int64, device=self.device)
        prev_chunk = torch.full((B,), -1, dtype=torch.int64, device=self.device)

        # Long-range decaying resonance buffer (ring of R recent salient nodes + decay amplitudes)
        R = cfg.resonance_slots
        res_nodes = torch.full((B, R), -1, dtype=torch.int64, device=self.device)
        res_amps = torch.zeros((B, R), dtype=torch.float32, device=self.device)
        res_ptr = 0

        # Neuron firing fatigue (short-term adaptation on recent output chars)
        fatigue = torch.zeros((B, self.V), dtype=torch.float32, device=self.device)

        # Precompute delay gains (closer physical distance -> stronger initial gain)
        delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]
        lr = cfg.lr * lr_scale

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0

        # Allocate reusable buffers
        logits = torch.empty((B, self.V), dtype=torch.float32, device=self.device)
        arange_B = torch.arange(B, device=self.device)

        for t in range(T - 1):
            x_t = data[:, t]        # current input sensory neuron [B]
            y_t = data[:, t + 1]    # next true sensory neuron [B]

            # Update character ring history (O(1) pointer advance, no slice overlap)
            char_hist[:, hist_ptr] = x_t
            cur_hist_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % max_delay

            # Update crystallized chunk firing (event-driven merge)
            if cfg.use_chunks and self.num_chunks > 0:
                merged = self._lookup_pairs(cur_chunk, x_t)
                hit = (cur_chunk >= 0) & (merged >= 0)
                # When merge fails, previous cur_chunk completes and moves into prev_chunk + resonance
                completed = (~hit) & (cur_chunk >= 0)
                if completed.any():
                    prev_chunk = torch.where(completed, cur_chunk, prev_chunk)
                    if cfg.use_resonance:
                        # Inject completed chunk into long-range resonance wave
                        res_nodes[:, res_ptr] = torch.where(completed, cur_chunk, res_nodes[:, res_ptr])
                        res_amps[:, res_ptr] = torch.where(
                            completed,
                            torch.ones(B, device=self.device),
                            res_amps[:, res_ptr],
                        )
                        res_ptr = (res_ptr + 1) % R
                cur_chunk = torch.where(hit, merged, x_t)
            else:
                prev_chunk = cur_chunk
                cur_chunk = x_t
                if cfg.use_resonance and t % 2 == 0:
                    res_nodes[:, res_ptr] = x_t
                    res_amps[:, res_ptr] = 1.0
                    res_ptr = (res_ptr + 1) % R

            # Decay long-range resonance & sensory fatigue
            if cfg.use_resonance:
                res_amps.mul_(cfg.resonance_decay)
            if cfg.use_fatigue:
                fatigue.mul_(cfg.fatigue_decay)
                fatigue[arange_B, x_t] += cfg.fatigue_strength

            # ------------------------------------------------------------------
            # 1) EVENT-DRIVEN WAVE PROPAGATION -> membrane potential logits[B, V]
            # ------------------------------------------------------------------
            logits.copy_(self.out_bias.unsqueeze(0))

            # (a) Sensory char waves at each physical distance delta
            for d_idx, d in enumerate(cfg.char_delays):
                src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                act = torch.full((B,), delay_gains[d_idx], dtype=torch.float32, device=self.device)
                self.char_banks[d_idx].propagate(src, act, logits)

            # (b) Crystallized chunk wave + previous chunk (word-distance) wave
            if cfg.use_chunks:
                # Chunks of length >= 2 fire through chunk_bank (sensory chars already fired above)
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

            # (c) Emergent Abstract Hub wave (generalization across similar nodes)
            hub_ids = None
            hub_w = None
            if cfg.use_hubs and self.hubs_initialized:
                safe_cur = cur_chunk.clamp(0, self.max_nodes - 1)
                hub_ids = self.node_hub_ids[safe_cur].to(torch.int64)  # [B, 2]
                hub_w = self.node_hub_w[safe_cur] * cfg.gain_hub       # [B, 2]
                self.hub_bank.propagate(hub_ids, hub_w, logits)

            # (d) Long-range decaying context resonance wave
            if cfg.use_resonance:
                self.resonance_bank.propagate(res_nodes, res_amps * cfg.gain_resonance, logits)

            # (e) Neuron fatigue / short-term adaptation
            if cfg.use_fatigue:
                logits.sub_(fatigue)

            # ------------------------------------------------------------------
            # 2) EVALUATE PREDICTION (strictly pre-update, online causal eval)
            # ------------------------------------------------------------------
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[arange_B, y_t]
            total_nll += float(nll.sum().item())

            top5_idx = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5_idx[:, 0] == y_t).sum().item())
            total_top5 += int((top5_idx == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if not train:
                continue

            # ------------------------------------------------------------------
            # 3) BACKPROP-FREE LOCAL PLASTICITY (Surprise-driven or Pure Co-occurrence)
            # ------------------------------------------------------------------
            probs = torch.exp(log_probs)
            p_true = probs[arange_B, y_t]  # [B]

            if cfg.use_surprise_learning:
                # Surprise at true target neuron: (1 - p_true)
                pos_surprise = (1.0 - p_true)  # [B]

                # Update resting output bias locally
                bias_grad = torch.zeros(self.V, dtype=torch.float32, device=self.device)
                bias_grad.scatter_add_(0, y_t, pos_surprise)
                # Top-M false-firing inhibition
                topm_p, topm_idx = torch.topk(probs, k=cfg.neg_top_m, dim=-1)  # [B, M]
                neg_mask = topm_idx != y_t.unsqueeze(1)
                neg_err = topm_p * neg_mask.to(topm_p.dtype)  # [B, M]
                bias_grad.scatter_add_(0, topm_idx.reshape(-1), -neg_err.reshape(-1))
                self.out_bias.add_(bias_grad * (lr * 0.02 / B))
                self.out_bias.clamp_(-4.0, 4.0)

                def update_bank(bank: SparseSynapseBank, src: torch.Tensor, amp: torch.Tensor, scale: float = 1.0):
                    # Combine positive surprise reinforcement (+ sprouting) & negative lateral inhibition
                    # into a single vectorized local_update call (only positive net deltas sprout).
                    M = cfg.neg_top_m
                    pos_delta = (lr * scale) * amp * pos_surprise
                    src_rep = src.unsqueeze(1).expand(B, M).reshape(-1)
                    dst_rep = topm_idx.reshape(-1)
                    neg_delta = -(lr * scale) * amp.unsqueeze(1).expand(B, M).reshape(-1) * neg_err.reshape(-1)
                    all_src = torch.cat([src, src_rep], dim=0)
                    all_dst = torch.cat([y_t, dst_rep], dim=0)
                    all_delta = torch.cat([pos_delta, neg_delta], dim=0)
                    bank.local_update(all_src, all_dst, all_delta, allow_sprout=True)

                # Update sensory char delay banks
                for d_idx, d in enumerate(cfg.char_delays):
                    src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                    amp = torch.full((B,), delay_gains[d_idx], dtype=torch.float32, device=self.device)
                    update_bank(self.char_banks[d_idx], src, amp, scale=0.8)

                # Update chunk & prev_chunk banks
                if cfg.use_chunks:
                    is_chunk = cur_chunk >= self.V
                    c_amp = torch.where(
                        is_chunk,
                        torch.full((B,), cfg.gain_chunk, dtype=torch.float32, device=self.device),
                        torch.zeros(B, dtype=torch.float32, device=self.device),
                    )
                    update_bank(self.chunk_bank, cur_chunk, c_amp, scale=1.0)
                    # Also let atomic chars populate chunk_bank lightly for hub wiring profile
                    pc_amp = torch.where(
                        prev_chunk >= 0,
                        torch.full((B,), cfg.gain_prev_chunk, dtype=torch.float32, device=self.device),
                        torch.zeros(B, dtype=torch.float32, device=self.device),
                    )
                    update_bank(self.prev_chunk_bank, prev_chunk, pc_amp, scale=0.7)

                # Update abstract hub bank
                if cfg.use_hubs and self.hubs_initialized and hub_ids is not None:
                    for h_slot in range(2):
                        update_bank(self.hub_bank, hub_ids[:, h_slot], hub_w[:, h_slot], scale=0.8)

                # Update long-range resonance bank (subsample 2 slots per step for speed)
                if cfg.use_resonance:
                    for r_off in (0, R // 2):
                        r_idx = (res_ptr - 1 - r_off) % R
                        update_bank(
                            self.resonance_bank,
                            res_nodes[:, r_idx],
                            res_amps[:, r_idx] * cfg.gain_resonance,
                            scale=0.5,
                        )
            else:
                # Baseline: Pure physical distance co-occurrence (Hebbian without surprise inhibition)
                # Simply strengthens synapses whenever (src at distance d, y_t) co-occur.
                self.cooc_counts.scatter_add_(0, y_t, torch.ones(B, dtype=torch.float32, device=self.device))
                self.out_bias.copy_(torch.log((self.cooc_counts + 1.0) / (self.cooc_counts.sum() + self.V)))
                for d_idx, d in enumerate(cfg.char_delays):
                    src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                    # Pure Hebbian increment (saturates softly via clamp)
                    delta = torch.full((B,), 0.05 * delay_gains[d_idx], dtype=torch.float32, device=self.device)
                    self.char_banks[d_idx].local_update(src, y_t, delta, allow_sprout=True)

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
        }

    @torch.no_grad()
    def metabolize_all(self) -> Dict[str, int]:
        """Run metabolic decay and pruning across all synapse banks."""
        pruned = 0
        for b in self.char_banks:
            pruned += b.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        pruned += self.chunk_bank.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        pruned += self.prev_chunk_bank.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        pruned += self.hub_bank.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        pruned += self.resonance_bank.metabolize(self.cfg.weight_decay, self.cfg.prune_threshold)
        return {
            "pruned_now": pruned,
            "active_synapses": self.total_active_synapses(),
            "total_evictions": self.total_evictions(),
            "total_pruned": self.total_pruned(),
        }

    def all_banks(self) -> List[SparseSynapseBank]:
        return list(self.char_banks) + [
            self.chunk_bank,
            self.prev_chunk_bank,
            self.hub_bank,
            self.resonance_bank,
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
        max_new_chars: int = 120,
        temperature: float = 0.7,
        top_k: int = 20,
    ) -> List[int]:
        """Generate text via pure event-driven wave propagation."""
        cfg = self.cfg
        max_delay = max(cfg.char_delays)
        char_hist = torch.full((1, max_delay), -1, dtype=torch.int64, device=self.device)
        hist_ptr = 0
        cur_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        prev_chunk = torch.full((1,), -1, dtype=torch.int64, device=self.device)
        R = cfg.resonance_slots
        res_nodes = torch.full((1, R), -1, dtype=torch.int64, device=self.device)
        res_amps = torch.zeros((1, R), dtype=torch.float32, device=self.device)
        res_ptr = 0
        fatigue = torch.zeros((1, self.V), dtype=torch.float32, device=self.device)
        delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]

        def step_in(ch_id: int) -> torch.Tensor:
            nonlocal cur_chunk, prev_chunk, res_ptr, hist_ptr
            x_t = torch.tensor([ch_id], dtype=torch.int64, device=self.device)
            char_hist[0, hist_ptr] = ch_id
            cur_hist_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % max_delay

            if cfg.use_chunks and self.num_chunks > 0:
                merged = self._lookup_pairs(cur_chunk, x_t)
                hit = (cur_chunk >= 0) & (merged >= 0)
                if (not hit.item()) and cur_chunk.item() >= 0:
                    prev_chunk = cur_chunk.clone()
                    if cfg.use_resonance:
                        res_nodes[0, res_ptr] = cur_chunk[0]
                        res_amps[0, res_ptr] = 1.0
                        res_ptr = (res_ptr + 1) % R
                cur_chunk = torch.where(hit, merged, x_t)
            else:
                prev_chunk = cur_chunk
                cur_chunk = x_t

            if cfg.use_resonance:
                res_amps.mul_(cfg.resonance_decay)
            if cfg.use_fatigue:
                fatigue.mul_(cfg.fatigue_decay)
                fatigue[0, ch_id] += cfg.fatigue_strength

            logits = self.out_bias.unsqueeze(0).clone()
            for d_idx, d in enumerate(cfg.char_delays):
                src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
                act = torch.tensor([delay_gains[d_idx]], dtype=torch.float32, device=self.device)
                self.char_banks[d_idx].propagate(src, act, logits)

            if cfg.use_chunks:
                if cur_chunk.item() >= self.V:
                    self.chunk_bank.propagate(
                        cur_chunk, torch.tensor([cfg.gain_chunk], device=self.device), logits
                    )
                if prev_chunk.item() >= 0:
                    self.prev_chunk_bank.propagate(
                        prev_chunk, torch.tensor([cfg.gain_prev_chunk], device=self.device), logits
                    )

            if cfg.use_hubs and self.hubs_initialized:
                cid = int(cur_chunk.item())
                if 0 <= cid < self.max_nodes:
                    h_ids = self.node_hub_ids[cid : cid + 1].to(torch.int64)
                    h_w = self.node_hub_w[cid : cid + 1] * cfg.gain_hub
                    self.hub_bank.propagate(h_ids, h_w, logits)

            if cfg.use_resonance:
                self.resonance_bank.propagate(res_nodes, res_amps * cfg.gain_resonance, logits)

            if cfg.use_fatigue:
                logits.sub_(fatigue)

            # Avoid emitting <UNK>=0
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

    def inspect_emergent_structures(self, id_to_char: List[str]) -> Dict[str, object]:
        """Inspect crystallized chunk neurons and emergent abstract hub clusters."""
        # Sample longest & newest crystallized chunks
        chunk_samples = []
        if self.num_chunks > 0:
            lens = self.chunk_len[self.V : self.V + self.num_chunks].cpu().numpy()
            long_idx = np.argsort(-lens)[:25]
            for i in long_idx:
                s = self.decode_node(self.V + int(i), id_to_char)
                chunk_samples.append(s)

        # Sample abstract hub clusters (show member neurons grouped by wiring similarity)
        hub_clusters = []
        if self.hubs_initialized:
            h0 = self.node_hub_ids[: self.num_active_nodes, 0].cpu().numpy()
            valid = np.where(h0 >= 0)[0]
            if len(valid) > 0:
                hub_to_members: Dict[int, List[str]] = {}
                for nid in valid:
                    hid = int(h0[nid])
                    lst = hub_to_members.setdefault(hid, [])
                    if len(lst) < 12:
                        lst.append(self.decode_node(int(nid), id_to_char))
                # Pick diverse medium-sized clusters to display
                ranked = sorted(hub_to_members.items(), key=lambda kv: len(kv[1]), reverse=True)
                for hid, members in ranked[:12]:
                    hub_clusters.append({"hub_id": hid, "members": members})

        return {
            "num_chunks": self.num_chunks,
            "longest_chunks": chunk_samples,
            "hub_clusters": hub_clusters,
        }
