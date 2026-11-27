"""NeuroGraph v1.3: Unified 6-System Raw UTF-8 Living Brain (D=256, S=16 Slots, 3M bf16 DAG, 33.6M LTM, Executive, Lateral Semantic Wave).

Zero Backprop | Zero Attention | Zero Tokenizer | Zero <UNK> | Zero Hardcoded Rules.
Starts from 256 atomic raw byte sensory neurons and 0 hidden neurons.

Six cooperating biological systems:
1. Cortex: 3M-neuron bfloat16 Living Convergent DAG (D=256, 16-hop, selective merge + auto-divorce + myelination).
2. Working Memory: S=16 Homeostatic Autonomous Gated Slots (D=256, geometric timescales ~3B to ~400B).
3. Long-Term Knowledge Memory: 8-Bank MegaByteLTM (33.55M knowledge neurons in 268 MB + Sleep Pruning).
4. Short-Term Episodic Memory: 1-Shot ByteSTM (192 slots/stream, instant online binding of unseen UTF-8 strings).
5. Executive Discourse Critic: 24-Mode Unsupervised Discourse Critic (analytical smoothed byte LLR).
6. Lateral Semantic Thought Wave (NEW!):
   - Extracts K=24,000 crystallized multi-byte content neurons directly from the DAG (zero tokenizer!).
   - Wires a sparse Hebbian PPMI Lateral Association Graph across a 180-byte passage window (24 neighbors/concept).
   - 2-hop spreading activation lights up semantically associated multi-byte concepts BEFORE they appear,
     proactively priming their exact byte transitions (boosting val BPB!) AND guiding Prefrontal Rollouts
     so multi-sentence generation stays truly ON-TOPIC!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from neurograph.executive_v12 import MegaBrainV12Config, MegaUTF8BrainV12
from neurograph.memory_v3b import _mix64


@dataclass
class UnifiedBrainV13Config(MegaBrainV12Config):
    # Wider shared representation + 2x more working-memory slots!
    pool_capacity: int = 3_000_000      # 3M bfloat16 neurons x D=256 = 1.50 GB VRAM!
    edge_buckets: int = 4_194_304       # 16.78M dynamic edge slots
    feat_dim: int = 256                 # 2x wider feature space (eliminates v1.2 D=128 bottleneck!)
    num_slots: int = 16                 # 2x more autonomous working-memory slots (~3B to ~400B!)
    max_hops: int = 16
    max_births_per_step: int = 256
    # Full fresh balanced training LRs (no domain-drift penalty when trained on shuffled 3-domain mix!)
    lr_feat: float = 0.22
    lr_basis: float = 0.035
    lr_sharp: float = 0.28
    lr_slot_read: float = 0.12
    lr_slot_mod: float = 0.07
    lr_slot_gate: float = 0.045
    # System 6: Lateral Multi-Byte Semantic Association Graph (Zero Tokenizer, Zero Rules!)
    enable_lateral: bool = True
    sem_vocab_max: int = 24_000         # top emergent crystallized multi-byte concept neurons
    sem_min_depth: int = 5              # >= 5 bytes (~2+ Japanese chars = real content morphemes/words!)
    sem_max_depth: int = 14
    sem_min_fires: int = 12
    sem_max_fires: int = 6000           # excludes ultra-common stop-phrases like 'いました。'
    lat_k_neighbors: int = 24           # top-24 PPMI semantic neighbors per concept
    lat_window_concepts: int = 10       # ~180-byte semantic co-occurrence window
    lat_prime_scale: float = 0.65       # gentle proactive byte-transition priming boost
    lat_max_boost: float = 2.4


class UnifiedBrainV13(MegaUTF8BrainV12):
    """Complete 6-System Raw UTF-8 Living Brain with D=256, S=16 Slots, and 2-Hop Lateral Semantic Thought Wave."""

    def __init__(self, cfg: UnifiedBrainV13Config, device: torch.device):
        # Override S=16 geometric priors cleanly before calling super? Super assumes [:S] from len-8 tensor!
        # Let's initialize cleanly by calling super with num_slots=8 then expanding S=16 slots right here.
        base_cfg = type(cfg)(**{**cfg.__dict__, "num_slots": 8})
        super().__init__(base_cfg, device)
        self.cfg = cfg
        S, D = cfg.num_slots, cfg.feat_dim

        # Re-initialize S=16 geometric multi-timescale Autonomous Working-Memory Slots (~2.5B to ~400B half-life!)
        # Geometric half-lives from 2.2 bytes to 380 bytes (~130 Japanese chars!)
        half_lives = torch.exp(torch.linspace(math.log(2.2), math.log(380.0), S, device=device))
        keep_priors = torch.exp(-math.log(2.0) / half_lives).clamp(0.65, 0.9982)
        self.keep_bias = torch.log(keep_priors / (1.0 - keep_priors))
        # Target write rates geometrically spaced from 0.25 down to 0.008 (ALL 16 slots awake!)
        self.target_write_rate = torch.exp(torch.linspace(math.log(0.25), math.log(0.008), S, device=device))
        self.write_bias = torch.log(self.target_write_rate / (1.0 - self.target_write_rate))

        self.G_write = torch.zeros((S, D), dtype=torch.float32, device=device)
        self.G_keep = torch.zeros((S, D), dtype=torch.float32, device=device)
        gen = torch.Generator(device=device)
        gen.manual_seed(2026)
        self.W_read = torch.randn((S, D, D), generator=gen, device=device) * (0.045 / math.sqrt(D))
        self.W_mod = torch.randn((S, D, D), generator=gen, device=device) * (0.028 / math.sqrt(D))
        self.rms_read = torch.full((S,), 1e-3, dtype=torch.float32, device=device)
        self.rms_mod = torch.full((S,), 1e-3, dtype=torch.float32, device=device)

        # System 6: Lateral Semantic Association Graph (populated after Cortex crystallizes concepts!)
        self.node_to_sem = torch.full((self.P,), -1, dtype=torch.int32, device=device)
        self.sem_to_node: List[int] = []
        self.sem_texts: List[str] = []
        self.sem_raw_bytes: List[bytes] = []
        # Sparse lateral neighbors: [K_sem, lat_k]
        self.lat_dst = torch.empty((0, cfg.lat_k_neighbors), dtype=torch.int32, device=device)
        self.lat_w = torch.empty((0, cfg.lat_k_neighbors), dtype=torch.float32, device=device)
        # Compact byte-transition priming signatures per concept:
        # For each concept s = b0 b1 b2 b3 b4..., store up to 4 internal (k3_hash -> next_byte) transitions
        # so firing concept s in the lateral wave proactively primes ALL of its distinctive byte transitions!
        self.sem_prime_k3 = torch.empty((0, 4), dtype=torch.int64, device=device)
        self.sem_prime_y = torch.empty((0, 4), dtype=torch.int16, device=device)
        self.sem_ready = False

    @torch.no_grad()
    def init_sensory_basis(self, train_bytes: np.ndarray, sample_len: int = 4_000_000) -> None:
        """Self-organize byte feature basis E [256, D=256] from distance-1..3 raw byte co-occurrence."""
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
        ppmi.add_(torch.eye(V, device=self.device) * 0.30)
        # Full rank 256x256 SVD when D == 256!
        U, S_val, _ = torch.linalg.svd(ppmi)
        E_init = U[:, :D] * torch.sqrt(S_val[:D].clamp(min=1e-5)).unsqueeze(0)
        self.E.copy_((E_init / E_init.norm(dim=1, keepdim=True).clamp(min=1e-5)) * 1.35)

        counts = torch.bincount(seq, minlength=V).to(torch.float32)
        self.bias.copy_(torch.log((counts + 1.0) / (counts.sum() + V)))

    def _extract_neuron_bytes(self, nid: int) -> bytes:
        """Walk parent pointers back to root byte to reconstruct the exact raw UTF-8 byte sequence of node nid."""
        out: List[int] = []
        cur = nid
        guard = 0
        while cur >= self.V and guard < 24:
            k = int(self.cond_kind[cur].item())
            cid = int(self.cond_id[cur].item())
            if k == 0 and 0 <= cid < 256:
                out.append(cid)
            cur = int(self.parent[cur].item())
            guard += 1
        if 0 <= cur < 256:
            out.append(cur)
        return bytes(out)

    @staticmethod
    def _hash3_tensor(b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """Bit-exact 3-byte hash shared by both build_lateral_semantic_graph and _compute_k3_gpu."""
        h = _mix64(b0 + 23)
        h = _mix64(h ^ ((b1 + 41) * 1315423911))
        h = _mix64(h ^ ((b2 + 67) * 2654435761))
        h = torch.where(h == 0, torch.ones_like(h), h)
        return torch.where(b2 >= 0, h, torch.zeros_like(h))

    @torch.no_grad()
    def _compute_k3_gpu(self, byte_hist: torch.Tensor, cur_idx: int, H_buf: int) -> torch.Tensor:
        b0 = byte_hist[:, cur_idx]
        b1 = byte_hist[:, (cur_idx - 1) % H_buf]
        b2 = byte_hist[:, (cur_idx - 2) % H_buf]
        return self._hash3_tensor(b0, b1, b2)

    @torch.no_grad()
    def build_lateral_semantic_graph(
        self,
        mixed_bytes: np.ndarray,
        scan_bytes: int = 16_000_000,
        batch_streams: int = 2048,
    ) -> Dict[str, object]:
        """System 6: Discover emergent multi-byte concepts from our OWN Living DAG and wire a
        sparse 2-hop Hebbian PPMI Lateral Semantic Association Graph (ZERO tokenizer, ZERO rules!)."""
        cfg = self.cfg
        device = self.device
        V = self.V

        # 1) Select top K_sem crystallized content-like multi-byte neurons from the Living DAG
        idx_all = torch.arange(self.P, device=device)
        cand_mask = (
            self.alive
            & (idx_all >= V)
            & (self.depth >= cfg.sem_min_depth)
            & (self.depth <= cfg.sem_max_depth)
            & (self.fires >= cfg.sem_min_fires)
            & (self.fires <= cfg.sem_max_fires)
        )
        cand_ids = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
        if cand_ids.numel() < 100:
            self.sem_ready = False
            return {"num_concepts": 0, "num_edges": 0}

        f_c = self.fires[cand_ids].to(torch.float32)
        d_c = self.depth[cand_ids].to(torch.float32)
        # Prefer rich multi-byte content words (mid-frequency, deeper bytes) over generic stop-phrases
        score = d_c * torch.log1p(f_c) / (1.0 + f_c / 350.0)
        k_take = min(cfg.sem_vocab_max, int(cand_ids.numel()))
        top_ids = cand_ids[torch.topk(score, k=k_take).indices].cpu().tolist()

        # Reconstruct exact UTF-8 byte strings & deduplicate by clean decoded text
        self.node_to_sem.fill_(-1)
        self.sem_to_node = []
        self.sem_texts = []
        self.sem_raw_bytes = []
        prime_b0_list: List[List[int]] = []
        prime_b1_list: List[List[int]] = []
        prime_b2_list: List[List[int]] = []
        prime_y_list: List[List[int]] = []

        # Pull parent/cond arrays to CPU once for blazing-fast lineage extraction!
        pa_cpu = self.parent.cpu().numpy()
        kd_cpu = self.cond_kind.cpu().numpy()
        cd_cpu = self.cond_id.cpu().numpy()

        seen_raw = set()
        for nid in top_ids:
            # Fast CPU walk
            bs_l: List[int] = []
            cur = nid
            guard = 0
            while cur >= V and guard < 24:
                if kd_cpu[cur] == 0 and 0 <= cd_cpu[cur] < 256:
                    bs_l.append(int(cd_cpu[cur]))
                cur = int(pa_cpu[cur])
                guard += 1
            if 0 <= cur < 256:
                bs_l.append(cur)
            raw = bytes(bs_l)
            if len(raw) < 5 or raw in seen_raw:
                continue
            # Require at least one clean multi-byte or meaningful substring (skip pure whitespace/punct)
            dec = raw.decode("utf-8", errors="ignore").strip()
            if len(dec) < 2:
                continue
            seen_raw.add(raw)
            sid = len(self.sem_to_node)
            self.sem_to_node.append(nid)
            self.sem_texts.append(dec)
            self.sem_raw_bytes.append(raw)

            # Build up to 4 internal 3-byte -> next-byte priming pairs for this concept!
            pb0, pb1, pb2 = [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]
            py = [-1, -1, -1, -1]
            slot_j = 0
            for pos in range(3, len(raw)):
                if slot_j >= 4:
                    break
                # At step t we look back (b0=raw[pos-1], b1=raw[pos-2], b2=raw[pos-3]) -> predict raw[pos]!
                pb0[slot_j] = raw[pos - 1]
                pb1[slot_j] = raw[pos - 2]
                pb2[slot_j] = raw[pos - 3]
                py[slot_j] = raw[pos]
                slot_j += 1
            prime_b0_list.append(pb0)
            prime_b1_list.append(pb1)
            prime_b2_list.append(pb2)
            prime_y_list.append(py)

        K_sem = len(self.sem_to_node)
        if K_sem < 64:
            self.sem_ready = False
            return {"num_concepts": K_sem, "num_edges": 0}

        n_t = torch.tensor(self.sem_to_node, dtype=torch.int64, device=device)
        s_t = torch.arange(K_sem, dtype=torch.int32, device=device)
        self.node_to_sem[n_t] = s_t
        tb0 = torch.tensor(prime_b0_list, dtype=torch.int64, device=device)
        tb1 = torch.tensor(prime_b1_list, dtype=torch.int64, device=device)
        tb2 = torch.tensor(prime_b2_list, dtype=torch.int64, device=device)
        self.sem_prime_k3 = self._hash3_tensor(tb0, tb1, tb2)
        self.sem_prime_y = torch.tensor(prime_y_list, dtype=torch.int16, device=device)

        # 2) Fast GPU streaming co-occurrence scan over mixed_bytes (180-byte passage window!)
        use_len = min(scan_bytes, len(mixed_bytes))
        stream_len = use_len // batch_streams
        data = torch.from_numpy(
            mixed_bytes[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(device)
        B, T = data.shape

        H_buf = max(cfg.max_hops + 4, 22)
        byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0
        arange_B = torch.arange(B, device=device)

        # Per-stream ring of last W_sem semantic concepts fired (with byte timestamp to skip self-overlap < 12B!)
        W_sem = cfg.lat_window_concepts
        sem_ring = torch.full((B, W_sem), -1, dtype=torch.int32, device=device)
        sem_time = torch.full((B, W_sem), -999, dtype=torch.int32, device=device)
        sem_ptr = torch.zeros(B, dtype=torch.int64, device=device)

        # Collect co-occurrence pairs in compact GPU hash/list chunks -> aggregate on CPU/GPU cleanly
        sem_counts = torch.zeros(K_sem, dtype=torch.int64, device=device)
        pair_a_chunks: List[torch.Tensor] = []
        pair_b_chunks: List[torch.Tensor] = []
        total_pairs_collected = 0
        MAX_PAIRS_COLLECT = 6_000_000

        for t in range(T):
            x_t = data[:, t]
            byte_hist[:, hist_ptr] = x_t
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            # Walk exact local spine up to max_depth and record deepest matching semantic concept!
            frontier = x_t
            hit_sem = torch.full((B,), -1, dtype=torch.int32, device=device)
            for hop in range(1, cfg.sem_max_depth + 1):
                c_loc = byte_hist[arange_B, (cur_idx - hop) % H_buf]
                ch = self.edges.lookup(frontier, torch.zeros_like(frontier), c_loc)
                ok = (frontier >= 0) & (c_loc >= 0) & (ch >= V) & self.alive[ch.clamp(min=0)]
                if not ok.any():
                    break
                if hop >= cfg.sem_min_depth:
                    s_id = self.node_to_sem[ch.clamp(min=0)]
                    got = ok & (s_id >= 0)
                    if got.any():
                        hit_sem[got] = s_id[got]
                frontier = torch.where(ok, ch, torch.full_like(ch, -1))

            fired = hit_sem >= 0
            if not fired.any():
                continue
            fb = torch.nonzero(fired, as_tuple=False).squeeze(1)
            fs = hit_sem[fb].to(torch.int64)
            sem_counts.scatter_add_(0, fs, torch.ones(fs.numel(), dtype=torch.int64, device=device))

            # Pair fs with past concepts in sem_ring[fb] that occurred 12..180 bytes ago!
            if total_pairs_collected < MAX_PAIRS_COLLECT:
                past_s = sem_ring[fb].to(torch.int64)     # [M, W_sem]
                dt_ring = t - sem_time[fb]                 # [M, W_sem]
                valid_p = (past_s >= 0) & (past_s != fs.unsqueeze(1)) & (dt_ring >= 12) & (dt_ring <= 180)
                if valid_p.any():
                    pm, pw = torch.nonzero(valid_p, as_tuple=True)
                    pa_v = past_s[pm, pw].to(torch.int32)
                    pb_v = fs[pm].to(torch.int32)
                    pair_a_chunks.append(pa_v.cpu())
                    pair_b_chunks.append(pb_v.cpu())
                    total_pairs_collected += int(pa_v.numel())

            # Write fired concept into per-stream ring
            wp = sem_ptr[fb]
            sem_ring[fb, wp] = hit_sem[fb]
            sem_time[fb, wp] = t
            sem_ptr[fb] = (wp + 1) % W_sem

        if not pair_a_chunks:
            self.sem_ready = False
            return {"num_concepts": K_sem, "num_edges": 0}

        # 3) Compute Symmetric PPMI & keep top-K_lat neighbors per concept (fast vectorized sort on CPU/GPU!)
        pa_all = torch.cat(pair_a_chunks).to(torch.int64)
        pb_all = torch.cat(pair_b_chunks).to(torch.int64)
        # Symmetrize so concept A <-> concept B associate bidirectionally
        src_all = torch.cat([pa_all, pb_all])
        dst_all = torch.cat([pb_all, pa_all])
        keys = src_all * K_sem + dst_all
        uniq_k, cnt_k = torch.unique(keys, return_counts=True)
        # Filter co-occurrences >= 2 to remove noise
        keep = cnt_k >= 2
        uniq_k, cnt_k = uniq_k[keep], cnt_k[keep].to(torch.float32)

        u_src = uniq_k // K_sem
        u_dst = uniq_k % K_sem
        sc_cpu = sem_counts.cpu().to(torch.float32).clamp(min=1.0)
        tot_c = float(cnt_k.sum().item()) + 1.0
        # Smoothed PPMI weighted gently by log1p(count) so reliable associations win over 2-hit flukes!
        pmi = torch.relu(torch.log((cnt_k * tot_c) / (sc_cpu[u_src] * sc_cpu[u_dst] + 25.0) + 1e-8))
        score_e = pmi * torch.log1p(cnt_k)

        # Group by u_src and keep top lat_k_neighbors
        K_lat = cfg.lat_k_neighbors
        lat_dst_np = np.full((K_sem, K_lat), -1, dtype=np.int32)
        lat_w_np = np.zeros((K_sem, K_lat), dtype=np.float32)

        # Sort by (u_src asc, score_e desc)
        ord_sc = torch.argsort(score_e, descending=True)
        u_src = u_src[ord_sc]
        u_dst = u_dst[ord_sc]
        score_e = score_e[ord_sc]
        ord_src = torch.argsort(u_src, stable=True)
        u_src_n = u_src[ord_src].numpy()
        u_dst_n = u_dst[ord_src].numpy()
        sc_n = score_e[ord_src].numpy()

        # Vectorized boundary slicing per unique source
        if len(u_src_n) > 0:
            starts = np.r_[0, np.flatnonzero(u_src_n[1:] != u_src_n[:-1]) + 1]
            ends = np.r_[starts[1:], len(u_src_n)]
            for s_i, e_i in zip(starts, ends):
                src_id = int(u_src_n[s_i])
                take = min(K_lat, e_i - s_i)
                lat_dst_np[src_id, :take] = u_dst_n[s_i : s_i + take]
                w_sl = sc_n[s_i : s_i + take]
                # Normalize outgoing lateral weights cleanly
                lat_w_np[src_id, :take] = w_sl / (float(w_sl[0]) + 1e-5)

        self.lat_dst = torch.from_numpy(lat_dst_np).to(device)
        self.lat_w = torch.from_numpy(lat_w_np).to(device)
        self.sem_ready = True
        n_edges = int((lat_dst_np >= 0).sum())
        return {"num_concepts": K_sem, "num_edges": n_edges, "pairs_scanned": total_pairs_collected}

    @torch.no_grad()
    def trace_semantic_wave(self, query_substr: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Show 2-hop Lateral Semantic Thought Wave radiating from any crystallized UTF-8 word!"""
        if not self.sem_ready:
            return []
        # Find concept matching query_substr
        matches = [i for i, txt in enumerate(self.sem_texts) if query_substr in txt]
        if not matches:
            return []
        # Seed wave on matching concept(s)
        K_sem = len(self.sem_texts)
        wave = torch.zeros(K_sem, dtype=torch.float32, device=self.device)
        for m in matches[:3]:
            wave[m] = 1.0
        # 2-hop lateral spreading activation
        w1 = torch.zeros_like(wave)
        for hop_scale in (1.0, 0.45):
            active = torch.nonzero(wave > 0.05, as_tuple=False).squeeze(1)
            if active.numel() == 0:
                break
            dst = self.lat_dst[active]   # [A, K]
            wt = self.lat_w[active] * wave[active].unsqueeze(1) * hop_scale
            ok = dst >= 0
            if ok.any():
                w1.scatter_add_(0, dst[ok].to(torch.int64), wt[ok])
            wave = w1.clone()
        # Exclude seed matches themselves
        for m in matches:
            wave[m] = 0.0
        tv, ti = torch.topk(wave, k=min(top_k, K_sem))
        return [(self.sem_texts[int(i)], round(float(v), 2)) for v, i in zip(tv.cpu().tolist(), ti.cpu().tolist()) if v > 0.05]

    @torch.no_grad()
    def eval_with_lateral_wave(
        self,
        byte_arr: np.ndarray,
        batch_streams: int = 128,
        bytes_per_char: float = 2.88,
        use_executive: bool = True,
        use_lateral: bool = True,
    ) -> Dict[str, float]:
        """Full 6-System Eval (Cortex DAG + 16 Slots + 33.6M LTM + ByteSTM + 24-Mode Executive + Lateral Semantic Priming!)."""
        cfg = self.cfg
        device = self.device
        V, D, S = self.V, self.D, cfg.num_slots
        lat_on = use_lateral and self.sem_ready
        exec_on = use_executive and self.mode_ready

        total_len = len(byte_arr)
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
        exec_vec = torch.zeros((B, D), dtype=torch.float32, device=device) if exec_on else None
        ltm_boost = torch.zeros((B, V), dtype=torch.float32, device=device)

        # STM
        N_stm = cfg.stm_slots
        stm_k4 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_k7 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_val = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_str = torch.zeros((B, N_stm), dtype=torch.float32, device=device)
        stm_ptr = torch.zeros(B, dtype=torch.int64, device=device)
        stm_boost = torch.zeros((B, V), dtype=torch.float32, device=device)

        # System 6 Proactive Lateral Priming Cache per stream: holds up to 96 primed (k3 -> y) byte transitions
        # populated proactively by the 1-hop/2-hop lateral semantic wave BEFORE those words appear!
        N_lat_c = 96
        if lat_on:
            lp_k3 = torch.zeros((B, N_lat_c), dtype=torch.int64, device=device)
            lp_y = torch.full((B, N_lat_c), -1, dtype=torch.int64, device=device)
            lp_w = torch.zeros((B, N_lat_c), dtype=torch.float32, device=device)
            lp_ptr = torch.zeros(B, dtype=torch.int64, device=device)
            lat_boost = torch.zeros((B, V), dtype=torch.float32, device=device)

        arange_B = torch.arange(B, device=device)
        ones_B_f = torch.ones(B, dtype=torch.float32, device=device)
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=device)

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0
        total_lat_hits = 0

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
                    ws = ctx_score.argmin(dim=1)
                    rep = ok_u & (~already) & (sal_old > ctx_score[arange_B, ws]) & (sal_old > 0.55)
                    if rep.any():
                        rb = torch.nonzero(rep, as_tuple=False).squeeze(1)
                        ctx_node[rb, ws[rb]] = u_old[rb]
                        ctx_score[rb, ws[rb]] = sal_old[rb]
            top_ctx_all = ctx_node[arange_B, ctx_score.argmax(dim=1)]

            # System 1: Cortex DAG Traversal
            vis_b_list = [arange_B]
            vis_u_list = [x_t]
            vis_g_list = [ones_B_f]
            frontier = x_t.unsqueeze(1)
            emergent_ctx_cand = torch.full((B,), -1, dtype=torch.int64, device=device)
            fired_sem = torch.full((B,), -1, dtype=torch.int32, device=device)

            for hop in range(1, cfg.max_hops + 1):
                valid_f = frontier >= 0
                if not valid_f.any():
                    break
                fb, ff = torch.nonzero(valid_f, as_tuple=True)
                p_ids = frontier[fb, ff]
                p_span = self.local_span[p_ids].to(torch.int64)
                c_loc = byte_hist[fb, (cur_idx - (p_span + 1).clamp(max=H_buf - 1)) % H_buf]
                ch_loc = self.edges.lookup(p_ids, torch.zeros_like(p_ids), c_loc)
                ch_loc = torch.where((c_loc >= 0) & (ch_loc >= V) & self.alive[ch_loc.clamp(min=0)], ch_loc, torch.full_like(ch_loc, -1))

                ch_ctx = torch.full_like(p_ids, -1)
                if cfg.enable_long_branch and 2 <= hop <= 5:
                    tc = top_ctx_all[fb]
                    rc = self.edges.lookup(p_ids, torch.ones_like(p_ids), tc)
                    ch_ctx = torch.where((tc >= V) & (rc >= V) & self.alive[rc.clamp(min=0)], rc, torch.full_like(rc, -1))

                next_front = torch.full((B, 4), -1, dtype=torch.int64, device=device)
                gv = 1.0 / (1.0 + 0.10 * hop)
                hl = ch_loc >= 0
                if hl.any():
                    hl_b, hl_c = fb[hl], ch_loc[hl]
                    next_front[hl_b, (ff[hl] * 2) % 4] = hl_c
                    vis_b_list.append(hl_b)
                    vis_u_list.append(hl_c)
                    vis_g_list.append(torch.full((hl_b.numel(),), gv, dtype=torch.float32, device=device))
                    if 2 <= hop <= 4:
                        emergent_ctx_cand[hl_b] = hl_c
                    if lat_on and cfg.sem_min_depth <= hop <= cfg.sem_max_depth:
                        sid = self.node_to_sem[hl_c]
                        gs = sid >= 0
                        if gs.any():
                            fired_sem[hl_b[gs]] = sid[gs]
                hc = ch_ctx >= 0
                if hc.any():
                    hc_b, hc_c = fb[hc], ch_ctx[hc]
                    next_front[hc_b, (ff[hc] * 2 + 1) % 4] = hc_c
                    vis_b_list.append(hc_b)
                    vis_u_list.append(hc_c)
                    vis_g_list.append(torch.full((hc_b.numel(),), gv, dtype=torch.float32, device=device))
                if not (hl.any() or hc.any()):
                    break
                frontier = next_front

            node_delay[:, nd_ptr] = emergent_ctx_cand
            nd_ptr = (nd_ptr + 1) % D_ctx

            all_b, all_u, all_g = torch.cat(vis_b_list), torch.cat(vis_u_list), torch.cat(vis_g_list)
            h_dag = torch.zeros((B, D), dtype=torch.float32, device=device)
            h_dag.index_add_(0, all_b, self.W_node[all_u].to(torch.float32) * all_g.unsqueeze(1))
            logits_sharp.zero_()
            sd_a, sw_a = self.sharp_dst[all_u], self.sharp_w[all_u].to(torch.float32)
            sm_a = sd_a >= 0
            if sm_a.any():
                logits_sharp.index_put_(
                    (all_b.unsqueeze(1).expand_as(sd_a)[sm_a], sd_a[sm_a].to(torch.int64)),
                    (sw_a * (all_g * 0.65).unsqueeze(1))[sm_a],
                    accumulate=True,
                )

            # System 2: S=16 Autonomous Slots
            h_norm = h_dag / h_dag.norm(dim=1, keepdim=True).clamp(min=1.0)
            e_norm = self.E[x_t] / self.E[x_t].norm(dim=1, keepdim=True).clamp(min=1.0)
            x_slot = 0.65 * h_norm + 0.35 * e_norm
            kg = torch.sigmoid(x_slot @ self.G_keep.T + self.keep_bias.unsqueeze(0))
            wg = torch.sigmoid(x_slot @ self.G_write.T + self.write_bias.unsqueeze(0))
            slot_state = slot_state * kg.unsqueeze(-1) + wg.unsqueeze(-1) * x_slot.unsqueeze(1)
            sn = slot_state.norm(dim=-1, keepdim=True)
            slot_state = slot_state * (2.5 / sn.clamp(min=2.5))
            r_sum = torch.einsum("bsd,sdk->bsk", slot_state, self.W_read).sum(dim=1)
            m_sum = torch.einsum("bsd,sdk->bsk", slot_state, self.W_mod).sum(dim=1)
            h_act = torch.tanh((h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum) * 0.35) * 2.85
            logits = h_act @ self.E.T + self.bias.unsqueeze(0) + logits_sharp

            # System 3: 8-Bank MegaByteLTM
            lk, lv = self.ltm.compute_keys(byte_hist, cur_idx, H_buf, top_ctx_all)
            self.ltm.lookup_boost(lk, lv, ltm_boost, max_boost=cfg.ltm_max_boost, ltm_scale=cfg.ltm_scale)
            logits = logits + ltm_boost

            # System 4: 1-Shot ByteSTM
            k4, k7 = self._compute_stm_keys(byte_hist, cur_idx, H_buf)
            stm_str.mul_(cfg.stm_decay)
            if t >= 4:
                m4 = (stm_k4 == k4.unsqueeze(1)) & (k4.unsqueeze(1) != 0) & (stm_str > 0.05)
                m7 = (stm_k7 == k7.unsqueeze(1)) & (k7.unsqueeze(1) != 0) & (stm_str > 0.05)
                hw = m4.float() * (stm_str * cfg.stm_w4) + m7.float() * (stm_str * cfg.stm_w7)
                if (hw > 0).any():
                    hb, hs = torch.nonzero(hw > 0, as_tuple=True)
                    stm_boost.zero_()
                    stm_boost.index_put_((hb, stm_val[hb, hs]), hw[hb, hs], accumulate=True)
                    logits = logits + stm_boost.clamp(max=cfg.stm_max_boost)

            # System 5: 24-Mode Executive Discourse Critic
            if exec_on and exec_vec is not None:
                exec_vec.mul_(cfg.mode_ema_decay).add_(h_dag, alpha=1.0 - cfg.mode_ema_decay)
                if t >= 24:
                    ev_n = exec_vec / exec_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    mp = torch.softmax((ev_n @ self.mode_centroids.T) / cfg.mode_temp, dim=-1)
                    logits = logits + (mp @ self.mode_llr) * cfg.mode_llr_scale

            # System 6: Proactive Lateral Semantic Wave Byte-Transition Priming!
            if lat_on:
                lp_w.mul_(0.9965)  # half-life ~200 bytes (~70 Japanese chars)
                # (a) First check if current 3-byte context matches any proactively primed associated concept!
                if t >= 3:
                    k3_now = self._compute_k3_gpu(byte_hist, cur_idx, H_buf)
                    ml = (lp_k3 == k3_now.unsqueeze(1)) & (k3_now.unsqueeze(1) != 0) & (lp_y >= 0) & (lp_w > 0.08)
                    if ml.any():
                        lb, ls = torch.nonzero(ml, as_tuple=True)
                        lat_boost.zero_()
                        lat_boost.index_put_((lb, lp_y[lb, ls]), lp_w[lb, ls] * cfg.lat_prime_scale, accumulate=True)
                        logits = logits + lat_boost.clamp(max=cfg.lat_max_boost)
                        total_lat_hits += int(ml.any(dim=1).sum().item())

                # (b) When a semantic concept fires, proactively prime its top-6 lateral neighbors' byte transitions!
                has_fs = fired_sem >= 0
                if has_fs.any():
                    fs_b = torch.nonzero(has_fs, as_tuple=False).squeeze(1)
                    fs_s = fired_sem[fs_b].to(torch.int64)
                    # Top-4 strongest lateral neighbors x 2 distinctive internal byte transitions each = 8 primes!
                    nb_ids = self.lat_dst[fs_s, :4]  # [M, 4]
                    nb_wt = self.lat_w[fs_s, :4]     # [M, 4]
                    for j in range(4):
                        nj = nb_ids[:, j]
                        ok_j = nj >= 0
                        if not ok_j.any():
                            continue
                        jb = fs_b[ok_j]
                        js = nj[ok_j].to(torch.int64)
                        jw = nb_wt[ok_j, j]
                        # Prime first 2 byte transitions of neighbor concept js
                        pk0 = self.sem_prime_k3[js, :2]  # [M2, 2]
                        py0 = self.sem_prime_y[js, :2].to(torch.int64)
                        for pj in range(2):
                            ok_p = (pk0[:, pj] != 0) & (py0[:, pj] >= 0)
                            if ok_p.any():
                                pb = jb[ok_p]
                                ptr_p = lp_ptr[pb]
                                lp_k3[pb, ptr_p] = pk0[ok_p, pj]
                                lp_y[pb, ptr_p] = py0[ok_p, pj]
                                lp_w[pb, ptr_p] = jw[ok_p]
                                lp_ptr[pb] = (ptr_p + 1) % N_lat_c

            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]
            total_nll += float(nll.sum().item())
            top5 = torch.topk(logits, k=5, dim=-1).indices
            total_top1 += int((top5[:, 0] == y_t).sum().item())
            total_top5 += int((top5 == y_t.unsqueeze(1)).any(dim=1).sum().item())
            total_steps += B

            if t >= 4:
                surp_stm = 1.0 - torch.exp(-nll)
                do_bind = (surp_stm > cfg.stm_surp_thresh) & (k4 != 0)
                if do_bind.any():
                    bb = torch.nonzero(do_bind, as_tuple=False).squeeze(1)
                    bk4, bk7, by = k4[bb], k7[bb], y_t[bb]
                    exists = (stm_k4[bb] == bk4.unsqueeze(1)) & (stm_val[bb] == by.unsqueeze(1)) & (stm_str[bb] > 0.05)
                    has_ex = exists.any(dim=1)
                    w_slot = torch.where(has_ex, exists.to(torch.int32).argmax(dim=1).to(torch.int64), stm_ptr[bb])
                    stm_k4[bb, w_slot] = bk4
                    stm_k7[bb, w_slot] = bk7
                    stm_val[bb, w_slot] = by
                    stm_str[bb, w_slot] = torch.where(has_ex, (stm_str[bb, w_slot] + 0.65).clamp(max=2.0), torch.ones_like(surp_stm[bb]))
                    stm_ptr[bb] = torch.where(has_ex, stm_ptr[bb], (stm_ptr[bb] + 1) % N_stm)

        avg_nll = total_nll / max(1, total_steps)
        bpb = avg_nll / math.log(2.0)
        bpc = bpb * bytes_per_char
        return {
            "bpb": round(bpb, 4),
            "bpc_lossless": round(bpc, 4),
            "ppl": round(2.0 ** min(bpc, 20.0), 2),
            "byte_top1": round(total_top1 / max(1, total_steps), 4),
            "byte_top5": round(total_top5 / max(1, total_steps), 4),
            "lat_hit_rate": round(total_lat_hits / max(1, total_steps), 4),
        }

    @torch.no_grad()
    def generate_semantic_utf8(
        self,
        prompt: str,
        max_new_bytes: int = 210,
        temperature: float = 0.68,
        top_k: int = 24,
        rollout_k: int = 4,
        rollout_chunk_bytes: int = 42,
        use_semantic_exec: bool = True,
    ) -> Tuple[str, bool, Dict[str, object]]:
        """6-System Raw UTF-8 Generation guided by 2-Hop Lateral Semantic Wave + 24-Mode Executive Rollout!"""
        cfg = self.cfg
        device = self.device
        V, D, S = self.V, self.D, cfg.num_slots
        raw_in = list(prompt.encode("utf-8"))
        H_buf = max(cfg.max_hops + 4, 22)
        D_ctx = 12
        N_stm = cfg.stm_slots
        K_sem = len(self.sem_texts)

        # Build the prompt's 2-hop Lateral Semantic Topic Wave BEFORE generating!
        prompt_wave = torch.zeros(max(1, K_sem), dtype=torch.float32, device=device)
        prompt_seeds: List[str] = []

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
        # Proactive lateral byte-transition primes from the prompt's 2-hop semantic wave!
        lat_prime_map: Dict[int, List[Tuple[int, float]]] = {}

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
            fired_sem_id = -1
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
                            if self.sem_ready and cfg.sem_min_depth <= hop <= cfg.sem_max_depth:
                                sid = int(self.node_to_sem[ch].item())
                                if sid >= 0:
                                    fired_sem_id = sid
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
            h_act = torch.tanh((h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum) * 0.35) * 2.85
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

            # System 5: Discourse Mode Modulation
            st["exec_vec"].mul_(cfg.mode_ema_decay).add_(h_dag.unsqueeze(0), alpha=1.0 - cfg.mode_ema_decay)
            if use_semantic_exec and self.mode_ready:
                ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
                cur_mp = torch.softmax((ev_n @ self.mode_centroids.T) / cfg.mode_temp, dim=-1)
                mp = (0.60 * locked_mode_p + 0.40 * cur_mp) if locked_mode_p is not None else cur_mp
                logits = logits + (mp @ self.mode_llr)[0] * (cfg.mode_llr_scale * 1.15)

            # System 6: Proactive Lateral Semantic Wave Byte-Transition Priming!
            if use_semantic_exec and lat_prime_map and step_i >= 3:
                k3_val = int(self._compute_k3_gpu(st["byte_hist"], cur_idx, H_buf)[0].item())
                if k3_val in lat_prime_map:
                    for py_b, pw_v in lat_prime_map[k3_val]:
                        logits[py_b] = logits[py_b] + min(pw_v, cfg.lat_max_boost)

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
            return logits, h_dag, fired_sem_id

        # 1) Prime on prompt, collect emergent semantic concepts, and launch 2-Hop Lateral Thought Wave!
        st = init_state()
        prompt_hs = []
        logits = None
        for i, b in enumerate(raw_in):
            nxt_t = raw_in[i + 1] if i + 1 < len(raw_in) else -1
            logits, hd, f_sid = step_state(st, b, next_true=nxt_t, locked_mode_p=None)
            prompt_hs.append(hd)
            if f_sid >= 0 and self.sem_ready:
                prompt_wave[f_sid] = 1.0
                if self.sem_texts[f_sid] not in prompt_seeds:
                    prompt_seeds.append(self.sem_texts[f_sid])

        # Propagate 2-hop Lateral Semantic Wave from prompt's fired concepts!
        top_wave_words: List[str] = []
        if use_semantic_exec and self.sem_ready and (prompt_wave > 0).any():
            w_acc = prompt_wave.clone()
            cur_w = prompt_wave.clone()
            for hop_s in (1.0, 0.45):
                act = torch.nonzero(cur_w > 0.05, as_tuple=False).squeeze(1)
                if act.numel() == 0:
                    break
                dst = self.lat_dst[act]
                wt = self.lat_w[act] * cur_w[act].unsqueeze(1) * hop_s
                ok = dst >= 0
                nxt_w = torch.zeros_like(cur_w)
                if ok.any():
                    nxt_w.scatter_add_(0, dst[ok].to(torch.int64), wt[ok])
                w_acc += nxt_w
                cur_w = nxt_w
            prompt_wave = w_acc / w_acc.max().clamp(min=1e-5)

            # Preload top-40 lit-up semantic concepts' byte transitions into lat_prime_map!
            tv, ti = torch.topk(prompt_wave, k=min(40, K_sem))
            for val, idx_s in zip(tv.cpu().tolist(), ti.cpu().tolist()):
                if val < 0.12:
                    break
                if len(top_wave_words) < 8 and self.sem_texts[idx_s] not in prompt_seeds:
                    top_wave_words.append(self.sem_texts[idx_s])
                pk_row = self.sem_prime_k3[idx_s].cpu().tolist()
                py_row = self.sem_prime_y[idx_s].cpu().tolist()
                for pk_v, py_v in zip(pk_row, py_row):
                    if pk_v != 0 and py_v >= 0:
                        lat_prime_map.setdefault(pk_v, []).append((py_v, val * cfg.lat_prime_scale * 1.35))

        prompt_mode_vec = torch.stack(prompt_hs).mean(dim=0)
        prompt_mode_vec = prompt_mode_vec / prompt_mode_vec.norm().clamp(min=1e-6)
        locked_mode_p = (
            torch.softmax(((prompt_mode_vec.unsqueeze(0) @ self.mode_centroids.T) / cfg.mode_temp), dim=-1)
            if (use_semantic_exec and self.mode_ready) else None
        )

        # 2) Prefrontal Rollout Generation guided by TRUE 2-Hop Lateral Semantic Resonance!
        out_bytes = list(raw_in)
        gen_only: List[int] = []
        recent_ngrams: set = set()
        hit_wave_concepts: List[str] = []

        rem = max_new_bytes
        while rem > 0:
            c_len = min(rollout_chunk_bytes, rem)
            k_cand = rollout_k if (use_semantic_exec and rollout_k > 1) else 1

            best_score = -1e9
            best_pack = None
            for _ in range(k_cand):
                cst = clone_state(st)
                clog = logits
                c_bytes: List[int] = []
                c_hs = []
                lp_sum = 0.0
                rep_pen = 0.0
                sem_res = 0.0
                c_hits: List[str] = []
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
                    clog, chd, f_sid = step_state(cst, nxt_b, locked_mode_p=locked_mode_p)
                    c_hs.append(chd)
                    if use_semantic_exec and f_sid >= 0 and self.sem_ready:
                        wv = float(prompt_wave[f_sid].item())
                        if wv > 0.08:
                            sem_res += wv * 0.9
                            if self.sem_texts[f_sid] not in c_hits:
                                c_hits.append(self.sem_texts[f_sid])

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
                mode_sim = float((c_mean_n @ prompt_mode_vec).item())
                fluency = lp_sum / max(1, c_len)

                score = fluency + 1.2 * min(sem_res, 2.5) + 0.80 * mode_sim + utf8_bonus - rep_pen
                if score > best_score:
                    best_score = score
                    best_pack = (cst, clog, c_bytes, c_hits)

            st, logits, chosen_b, chosen_hits = best_pack
            out_bytes.extend(chosen_b)
            gen_only.extend(chosen_b)
            for h_w in chosen_hits:
                if h_w not in hit_wave_concepts:
                    hit_wave_concepts.append(h_w)
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
        return text, valid_utf8, {
            "prompt_seeds": prompt_seeds[:6],
            "wave_lit": top_wave_words[:6],
            "wave_hits_in_gen": hit_wave_concepts[:6],
        }
