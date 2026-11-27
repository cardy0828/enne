"""短期記憶ノード（v0.3a）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from neurograph.model_v2 import NeuroGraphV2


@dataclass
class EpisodicMemoryConfig:
    enabled: bool = True
    num_memory_neurons: int = 128
    surprise_threshold: float = 0.55
    memory_decay: float = 0.9965
    recall_gain: float = 2.6
    reconsolidate_boost: float = 0.45
    use_uncertainty_gate: bool = True
    min_resonance: float = 1.5


class ShortTermMemoryBank:

    def __init__(self, batch_size: int, cfg: EpisodicMemoryConfig, device: torch.device):
        self.B = batch_size
        self.M = cfg.num_memory_neurons
        self.cfg = cfg
        self.device = device

        self.k1 = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)
        self.k2 = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)
        self.k3 = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)
        self.kc = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)
        self.kp = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)

        self.target = torch.full((batch_size, self.M), -1, dtype=torch.int64, device=device)
        self.strength = torch.zeros((batch_size, self.M), dtype=torch.float32, device=device)
        self.recall_hits = torch.zeros((batch_size, self.M), dtype=torch.int32, device=device)

        self.total_bindings = 0
        self.total_overwrites = 0
        self.total_recalls = 0
        self.total_reconsolidations = 0

    @torch.no_grad()
    def recall(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        cur_chunk: torch.Tensor,
        prev_chunk: torch.Tensor,
        vocab_size: int,
        base_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M = self.B, self.M
        cfg = self.cfg
        recall_wave = torch.zeros((B, vocab_size), dtype=torch.float32, device=self.device)
        if not cfg.enabled:
            return recall_wave, torch.zeros((B, M), dtype=torch.float32, device=self.device)

        alive = (self.target >= 0) & (self.strength > 0.02)
        if not alive.any():
            return recall_wave, torch.zeros((B, M), dtype=torch.float32, device=self.device)

        m1 = (self.k1 == c1.unsqueeze(1)) & (c1.unsqueeze(1) >= 0)
        m2 = (self.k2 == c2.unsqueeze(1)) & (c2.unsqueeze(1) >= 0)
        m3 = (self.k3 == c3.unsqueeze(1)) & (c3.unsqueeze(1) >= 0)
        mc = (self.kc == cur_chunk.unsqueeze(1)) & (cur_chunk.unsqueeze(1) >= vocab_size)
        mp = (self.kp == prev_chunk.unsqueeze(1)) & (prev_chunk.unsqueeze(1) >= vocab_size)

        anchor = m1 | mc
        score = (
            m1.to(torch.float32) * 0.6
            + (m1 & m2).to(torch.float32) * 1.2
            + (m1 & m2 & m3).to(torch.float32) * 2.4
            + (mc & m1).to(torch.float32) * 1.6
            + (mp & m1).to(torch.float32) * 1.0
        ) * (alive & anchor).to(torch.float32)

        fired = score >= cfg.min_resonance
        if not fired.any():
            return recall_wave, torch.zeros((B, M), dtype=torch.float32, device=self.device)

        self.total_recalls += int(fired.any(dim=1).sum().item())

        res = torch.where(fired, score * self.strength.clamp(0.0, 2.5), torch.zeros_like(score))

        safe_tgt = self.target.clamp(0, vocab_size - 1)
        recall_wave.scatter_add_(1, safe_tgt, res)
        recall_wave = torch.log1p(recall_wave)

        if cfg.use_uncertainty_gate:
            probs = torch.softmax(base_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1, keepdim=True)
            gate = (0.35 + 0.30 * entropy).clamp(0.35, 1.45)
            recall_wave = recall_wave * (cfg.recall_gain * gate)
        else:
            recall_wave = recall_wave * cfg.recall_gain

        return recall_wave, res

    @torch.no_grad()
    def bind_on_surprise(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        cur_chunk: torch.Tensor,
        prev_chunk: torch.Tensor,
        y_t: torch.Tensor,
        surprise: torch.Tensor,
        resonance: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        if not cfg.enabled:
            return torch.zeros(self.B, dtype=torch.bool, device=self.device)

        self.strength.mul_(cfg.memory_decay)

        hit_mem = (resonance > 0.0) & (self.target == y_t.unsqueeze(1))
        if hit_mem.any():
            self.strength = torch.where(
                hit_mem,
                (self.strength + cfg.reconsolidate_boost).clamp(0.0, 2.5),
                self.strength,
            )
            self.recall_hits.add_(hit_mem.to(torch.int32))
            self.total_reconsolidations += int(hit_mem.sum().item())

        wrong_mem = (resonance > 0.5) & (self.target != y_t.unsqueeze(1))
        if wrong_mem.any():
            self.strength = torch.where(wrong_mem, self.strength * 0.85, self.strength)

        need_bind = (surprise >= cfg.surprise_threshold) & (y_t > 0) & (c1 >= 0) & (c2 >= 0)
        if not need_bind.any():
            return need_bind

        b_idx = torch.nonzero(need_bind, as_tuple=False).squeeze(1)

        dup = (
            (self.k1[b_idx] == c1[b_idx].unsqueeze(1))
            & (self.k2[b_idx] == c2[b_idx].unsqueeze(1))
            & (self.k3[b_idx] == c3[b_idx].unsqueeze(1))
            & (self.target[b_idx] == y_t[b_idx].unsqueeze(1))
        )
        has_dup = dup.any(dim=1)

        if has_dup.any():
            d_b = b_idx[has_dup]
            d_slot = dup[has_dup].to(torch.int32).argmax(dim=1)
            self.strength[d_b, d_slot] = (self.strength[d_b, d_slot] + 0.65).clamp(0.0, 2.5)
            self.kc[d_b, d_slot] = cur_chunk[d_b]
            self.kp[d_b, d_slot] = prev_chunk[d_b]

        is_new = ~has_dup
        if is_new.any():
            n_b = b_idx[is_new]
            slot = self.strength[n_b].argmin(dim=1)
            was_occupied = self.target[n_b, slot] >= 0
            self.total_overwrites += int(was_occupied.sum().item())
            self.total_bindings += int(n_b.numel())

            self.k1[n_b, slot] = c1[n_b]
            self.k2[n_b, slot] = c2[n_b]
            self.k3[n_b, slot] = c3[n_b]
            self.kc[n_b, slot] = cur_chunk[n_b]
            self.kp[n_b, slot] = prev_chunk[n_b]
            self.target[n_b, slot] = y_t[n_b]
            self.strength[n_b, slot] = (0.75 + 0.75 * surprise[n_b]).clamp(0.5, 1.8)
            self.recall_hits[n_b, slot] = 0

        return need_bind


@torch.no_grad()
def evaluate_with_short_term_memory(
    cortex: NeuroGraphV2,
    ids: np.ndarray,
    mem_cfg: EpisodicMemoryConfig,
    batch_streams: int = 32,
) -> Dict[str, float]:
    cfg = cortex.cfg
    device = cortex.device
    V = cortex.V
    N = cortex.num_active_nodes

    total_len = len(ids)
    stream_len = total_len // batch_streams
    if stream_len < 16:
        batch_streams = max(1, total_len // 16)
        stream_len = total_len // batch_streams

    data = torch.from_numpy(
        ids[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
    ).to(device)
    B, T = data.shape

    max_delay = max(max(cfg.char_delays), 3)
    char_hist = torch.full((B, max_delay), -1, dtype=torch.int64, device=device)
    hist_ptr = 0

    cur_chunk = torch.full((B,), -1, dtype=torch.int64, device=device)
    prev_chunk = torch.full((B,), -1, dtype=torch.int64, device=device)

    membrane = torch.zeros((B, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    node_fatigue = torch.zeros((B, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    char_fatigue = torch.zeros((B, V), dtype=torch.float32, device=device)
    delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]

    memory = ShortTermMemoryBank(B, mem_cfg, device)

    logits_cortex = torch.empty((B, V), dtype=torch.float32, device=device)
    arange_B = torch.arange(B, device=device)
    hop_winners: List[Tuple[torch.Tensor, torch.Tensor]] = []

    total_nll = 0.0
    total_top1 = 0
    total_top5 = 0
    hard_nll = 0.0
    hard_top1 = 0
    hard_steps = 0
    total_steps = 0

    for t in range(T - 1):
        x_t = data[:, t]
        y_t = data[:, t + 1]

        char_hist[:, hist_ptr] = x_t
        cur_hist_idx = hist_ptr
        hist_ptr = (hist_ptr + 1) % max_delay

        c1 = x_t
        c2 = char_hist[:, (cur_hist_idx - 1) % max_delay]
        c3 = char_hist[:, (cur_hist_idx - 2) % max_delay]

        completed = torch.zeros(B, dtype=torch.bool, device=device)
        if cortex.num_chunks > 0:
            merged = cortex._lookup_pairs(cur_chunk, x_t)
            hit = (cur_chunk >= 0) & (merged >= 0)
            completed = (~hit) & (cur_chunk >= 0)
            if completed.any():
                prev_chunk = torch.where(completed, cur_chunk, prev_chunk)
            cur_chunk = torch.where(hit, merged, x_t)
        else:
            prev_chunk = cur_chunk
            cur_chunk = x_t

        char_fatigue.mul_(cfg.char_fatigue_decay)
        char_fatigue[arange_B, x_t] += cfg.char_fatigue_inc

        if cfg.n_hops > 0 and membrane is not None and node_fatigue is not None:
            membrane.mul_(cfg.membrane_decay)
            node_fatigue.mul_(cfg.node_fatigue_decay)
            sal_cur = cortex._homeostatic_salience(cur_chunk)
            membrane[arange_B, cur_chunk.clamp(0, N - 1)] += 0.70 * sal_cur
            if completed.any():
                c_idx = torch.nonzero(completed, as_tuple=False).squeeze(1)
                pc = prev_chunk[c_idx].clamp(0, N - 1)
                membrane[c_idx, pc] += 1.15 * cortex._homeostatic_salience(pc)
            if t % 2 == 0:
                hop_winners = []
                seed_ids = torch.stack([cur_chunk, prev_chunk], dim=1)
                seed_act = torch.stack(
                    [
                        sal_cur,
                        torch.where(
                            prev_chunk >= 0,
                            cortex._homeostatic_salience(prev_chunk) * 0.8,
                            torch.zeros(B, device=device),
                        ),
                    ],
                    dim=1,
                )
                node_fatigue[arange_B, cur_chunk.clamp(0, N - 1)] += cfg.node_fatigue_inc * 0.5
                c_ids, c_act = seed_ids, seed_act
                for _ in range(cfg.n_hops):
                    n_ids, n_act = cortex._propagate_lateral_hop(c_ids, c_act, membrane, node_fatigue, k_wta=cfg.k_wta)
                    hop_winners.append((n_ids, n_act))
                    c_ids, c_act = n_ids, n_act * cfg.hop_decay
                membrane.clamp_(0.0, 5.0)
                node_fatigue.clamp_(0.0, 5.0)

        logits_cortex.copy_(cortex.out_bias.unsqueeze(0))
        for d_idx, d in enumerate(cfg.char_delays):
            src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
            act = torch.full((B,), delay_gains[d_idx], dtype=torch.float32, device=device)
            cortex.char_banks[d_idx].propagate(src, act, logits_cortex)
        if cortex.num_chunks > 0:
            is_chunk = cur_chunk >= V
            c_act = torch.where(is_chunk, torch.full((B,), cfg.gain_chunk, device=device), torch.zeros(B, device=device))
            cortex.chunk_bank.propagate(cur_chunk, c_act, logits_cortex)
            pc_act = torch.where(prev_chunk >= 0, torch.full((B,), cfg.gain_prev_chunk, device=device), torch.zeros(B, device=device))
            cortex.prev_chunk_bank.propagate(prev_chunk, pc_act, logits_cortex)
        if hop_winners:
            cortex._apply_relay_readout(hop_winners, logits_cortex)
        logits_cortex.sub_(char_fatigue)

        cortex_log_p = torch.log_softmax(logits_cortex, dim=-1)
        cortex_p = torch.exp(cortex_log_p)
        cortex_ent = -(cortex_p * cortex_log_p).sum(dim=-1)
        is_hard = cortex_ent > 3.5

        recall_wave, resonance = memory.recall(c1, c2, c3, cur_chunk, prev_chunk, V, logits_cortex)
        logits_total = logits_cortex + recall_wave

        log_probs = torch.log_softmax(logits_total, dim=-1)
        nll = -log_probs[arange_B, y_t]
        total_nll += float(nll.sum().item())

        top5_idx = torch.topk(logits_total, k=5, dim=-1).indices
        hit1 = top5_idx[:, 0] == y_t
        total_top1 += int(hit1.sum().item())
        total_top5 += int((top5_idx == y_t.unsqueeze(1)).any(dim=1).sum().item())
        total_steps += B

        if is_hard.any():
            hard_nll += float(nll[is_hard].sum().item())
            hard_top1 += int(hit1[is_hard].sum().item())
            hard_steps += int(is_hard.sum().item())

        p_true = torch.exp(log_probs[arange_B, y_t])
        surprise = 1.0 - p_true
        memory.bind_on_surprise(c1, c2, c3, cur_chunk, prev_chunk, y_t, surprise, resonance)

    avg_nll = total_nll / max(1, total_steps)
    hard_avg_nll = hard_nll / max(1, hard_steps)
    return {
        "steps": total_steps,
        "bpc": avg_nll / math.log(2.0),
        "ppl": math.exp(min(avg_nll, 20.0)),
        "top1": total_top1 / max(1, total_steps),
        "top5": total_top5 / max(1, total_steps),
        "hard_bpc": hard_avg_nll / math.log(2.0),
        "hard_ppl": math.exp(min(hard_avg_nll, 20.0)),
        "hard_top1": hard_top1 / max(1, hard_steps),
        "hard_frac": hard_steps / max(1, total_steps),
        "bindings": memory.total_bindings,
        "overwrites": memory.total_overwrites,
        "recalls": memory.total_recalls,
        "reconsolidations": memory.total_reconsolidations,
    }


@torch.no_grad()
def demo_oneshot_article(
    cortex: NeuroGraphV2,
    article_ids: np.ndarray,
    id_to_char: List[str],
    mem_cfg: EpisodicMemoryConfig,
    prompt_after: List[str],
    encode_fn,
    decode_fn,
) -> Dict[str, object]:
    cfg = cortex.cfg
    device = cortex.device
    V = cortex.V
    N = cortex.num_active_nodes

    max_delay = max(max(cfg.char_delays), 3)
    char_hist = torch.full((1, max_delay), -1, dtype=torch.int64, device=device)
    hist_ptr = 0
    cur_chunk = torch.full((1,), -1, dtype=torch.int64, device=device)
    prev_chunk = torch.full((1,), -1, dtype=torch.int64, device=device)
    membrane = torch.zeros((1, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    node_fatigue = torch.zeros((1, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    char_fatigue = torch.zeros((1, V), dtype=torch.float32, device=device)
    delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]

    memory = ShortTermMemoryBank(1, mem_cfg, device)

    def forward_step(ch_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        nonlocal cur_chunk, prev_chunk, hist_ptr
        x_t = torch.tensor([ch_id], dtype=torch.int64, device=device)
        char_hist[0, hist_ptr] = ch_id
        cur_hist_idx = hist_ptr
        hist_ptr = (hist_ptr + 1) % max_delay

        c1 = x_t
        c2 = char_hist[:, (cur_hist_idx - 1) % max_delay]
        c3 = char_hist[:, (cur_hist_idx - 2) % max_delay]

        completed = False
        if cortex.num_chunks > 0:
            merged = cortex._lookup_pairs(cur_chunk, x_t)
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
            sal_cur = cortex._homeostatic_salience(cur_chunk)
            membrane[0, cur_chunk.clamp(0, N - 1)] += 0.70 * sal_cur
            if completed:
                pc = prev_chunk.clamp(0, N - 1)
                membrane[0, pc] += 1.15 * cortex._homeostatic_salience(pc)
            seed_ids = torch.stack([cur_chunk, prev_chunk], dim=1)
            seed_act = torch.stack(
                [
                    sal_cur,
                    cortex._homeostatic_salience(prev_chunk) * 0.8 if prev_chunk.item() >= 0 else torch.zeros(1, device=device),
                ],
                dim=1,
            )
            node_fatigue[0, cur_chunk.clamp(0, N - 1)] += cfg.node_fatigue_inc * 0.5
            c_ids, c_act = seed_ids, seed_act
            for _ in range(cfg.n_hops):
                n_ids, n_act = cortex._propagate_lateral_hop(c_ids, c_act, membrane, node_fatigue, k_wta=cfg.k_wta)
                hop_winners.append((n_ids, n_act))
                c_ids, c_act = n_ids, n_act * cfg.hop_decay
            membrane.clamp_(0.0, 5.0)
            node_fatigue.clamp_(0.0, 5.0)

        logits_c = cortex.out_bias.unsqueeze(0).clone()
        for d_idx, d in enumerate(cfg.char_delays):
            src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
            act = torch.tensor([delay_gains[d_idx]], dtype=torch.float32, device=device)
            cortex.char_banks[d_idx].propagate(src, act, logits_c)
        if cortex.num_chunks > 0:
            if cur_chunk.item() >= V:
                cortex.chunk_bank.propagate(cur_chunk, torch.tensor([cfg.gain_chunk], device=device), logits_c)
            if prev_chunk.item() >= 0:
                cortex.prev_chunk_bank.propagate(prev_chunk, torch.tensor([cfg.gain_prev_chunk], device=device), logits_c)
        if hop_winners:
            cortex._apply_relay_readout(hop_winners, logits_c)
        logits_c.sub_(char_fatigue)

        recall_w, res = memory.recall(c1, c2, c3, cur_chunk, prev_chunk, V, logits_c)
        return logits_c, logits_c + recall_w, res, (c1, c2, c3, cur_chunk.clone(), prev_chunk.clone())

    cortex_nll_sum = 0.0
    mem_nll_sum = 0.0
    big_wins = []

    T = len(article_ids)
    for t in range(T - 1):
        ch_id = int(article_ids[t])
        y_id = int(article_ids[t + 1])
        logits_c, logits_m, res, keys = forward_step(ch_id)

        lp_c = torch.log_softmax(logits_c, dim=-1)[0, y_id].item()
        lp_m = torch.log_softmax(logits_m, dim=-1)[0, y_id].item()
        cortex_nll_sum += -lp_c
        mem_nll_sum += -lp_m

        gain_bits = (lp_m - lp_c) / math.log(2.0)
        if gain_bits > 2.5 and len(big_wins) < 12:
            ctx = decode_fn(article_ids[max(0, t - 6) : t + 1])
            big_wins.append({
                "pos": t,
                "context": ctx,
                "next_char": id_to_char[y_id],
                "p_cortex": round(math.exp(lp_c), 4),
                "p_with_mem": round(math.exp(lp_m), 4),
                "gain_bits": round(gain_bits, 2),
            })

        y_t = torch.tensor([y_id], dtype=torch.int64, device=device)
        surprise = torch.tensor([1.0 - math.exp(lp_m)], dtype=torch.float32, device=device)
        memory.bind_on_surprise(*keys, y_t, surprise, res)

    mem_gens = {}
    cortex_gens = {}
    for p in prompt_after:
        p_ids = encode_fn(p)
        cortex_gens[p] = decode_fn(cortex.generate(p_ids, max_new_chars=60, temperature=0.5, top_k=12))

        saved_str = memory.strength.clone()
        char_hist.fill_(-1)
        cur_chunk.fill_(-1)
        prev_chunk.fill_(-1)
        char_fatigue.zero_()
        if membrane is not None:
            membrane.zero_()
            node_fatigue.zero_()

        out = list(p_ids)
        logits_m = None
        for ch in p_ids:
            _, logits_m, _, _ = forward_step(ch)
        for _ in range(60):
            assert logits_m is not None
            sc = logits_m[0].clone()
            sc[0] = -1e9
            sc = sc / 0.5
            vals, idx = torch.topk(sc, k=12)
            probs = torch.softmax(vals, dim=-1)
            pick = int(idx[torch.multinomial(probs, 1).item()].item())
            out.append(pick)
            _, logits_m, _, _ = forward_step(pick)
        memory.strength.copy_(saved_str)
        mem_gens[p] = decode_fn(out)

    top_mems = []
    s_np = memory.strength[0].cpu().numpy()
    order = np.argsort(-s_np)[:12]
    for idx_m in order:
        tgt = int(memory.target[0, idx_m].item())
        if tgt < 0:
            continue
        k3 = int(memory.k3[0, idx_m].item())
        k2 = int(memory.k2[0, idx_m].item())
        k1 = int(memory.k1[0, idx_m].item())
        tri = "".join(id_to_char[x] if 0 <= x < V else "" for x in (k3, k2, k1))
        top_mems.append(
            f"[{tri}]->{id_to_char[tgt]} (str={s_np[idx_m]:.2f}, hits={int(memory.recall_hits[0, idx_m].item())})"
        )

    return {
        "article_len": T,
        "cortex_bpc": round((cortex_nll_sum / max(1, T - 1)) / math.log(2.0), 4),
        "with_mem_bpc": round((mem_nll_sum / max(1, T - 1)) / math.log(2.0), 4),
        "bindings": memory.total_bindings,
        "reconsolidations": memory.total_reconsolidations,
        "big_wins": big_wins,
        "top_memories": top_mems,
        "cortex_gens": cortex_gens,
        "mem_gens": mem_gens,
    }
