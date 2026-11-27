"""集合連想型の疎な長期記憶ノード（v0.3b）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from neurograph.memory_v3a import EpisodicMemoryConfig, ShortTermMemoryBank
from neurograph.model_v2 import NeuroGraphV2


@dataclass
class LongTermMemoryConfig:
    enabled: bool = True
    use_topic_context: bool = True
    num_buckets: int = 32768
    ways: int = 8
    surprise_threshold: float = 0.40
    lr: float = 0.45
    min_hits_eval: int = 2
    gain_char4: float = 0.95
    gain_chunk_pair: float = 1.15
    gain_topic_ctx: float = 0.75


def _mix64(x: torch.Tensor) -> torch.Tensor:
    h = x
    h = h ^ ((h >> 30) & 0x3FFFFFFFF)
    h = h * (-4658895280553007687)
    h = h ^ ((h >> 27) & 0x1FFFFFFFFF)
    h = h * (-7723592293110705685)
    h = h ^ ((h >> 31) & 0x1FFFFFFFF)
    return h


def make_sig_4(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    h = _mix64(a.to(torch.int64) + 1)
    h = _mix64(h ^ (b.to(torch.int64) + 1000003))
    h = _mix64(h ^ (c.to(torch.int64) + 2000003))
    h = _mix64(h ^ (d.to(torch.int64) + 3000017))
    return torch.where(h == 0, torch.ones_like(h), h)


def make_sig_3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    h = _mix64(a.to(torch.int64) + 7)
    h = _mix64(h ^ (b.to(torch.int64) + 1000033))
    h = _mix64(h ^ (c.to(torch.int64) + 2000029))
    return torch.where(h == 0, torch.ones_like(h), h)


class SetAssociativeLTMBank:

    def __init__(self, num_buckets: int, ways: int, device: torch.device):
        assert (num_buckets & (num_buckets - 1)) == 0, "num_buckets must be power of 2"
        self.num_buckets = num_buckets
        self.mask = num_buckets - 1
        self.ways = ways
        self.device = device

        self.sig = torch.zeros((num_buckets, ways), dtype=torch.int64, device=device)
        self.target = torch.full((num_buckets, ways), -1, dtype=torch.int32, device=device)
        self.weight = torch.zeros((num_buckets, ways), dtype=torch.float32, device=device)
        self.energy = torch.zeros((num_buckets, ways), dtype=torch.float32, device=device)
        self.hits = torch.zeros((num_buckets, ways), dtype=torch.int32, device=device)

        self.total_writes = 0
        self.total_evictions = 0
        self.total_pruned = 0

    @torch.no_grad()
    def recall(
        self,
        keys: torch.Tensor,
        valid: torch.Tensor,
        gain: float,
        min_hits: int,
        logits: torch.Tensor,
    ) -> None:
        if not valid.any():
            return
        buckets = (keys & self.mask).to(torch.int64)
        b_sig = self.sig[buckets]
        b_tgt = self.target[buckets]
        b_w = self.weight[buckets]
        b_hits = self.hits[buckets]

        hit = (
            valid.unsqueeze(1)
            & (b_sig == keys.unsqueeze(1))
            & (b_tgt >= 0)
            & (b_hits >= min_hits)
            & (b_w > 0.01)
        )
        if not hit.any():
            return

        contrib = (b_w * gain) * hit.to(torch.float32)
        safe_tgt = b_tgt.clamp(min=0).to(torch.int64)
        logits.scatter_add_(1, safe_tgt, contrib)

    @torch.no_grad()
    def local_update(
        self,
        keys: torch.Tensor,
        valid: torch.Tensor,
        y_t: torch.Tensor,
        probs: torch.Tensor,
        lr: float,
        surprise_threshold: float,
    ) -> None:
        if not valid.any():
            return

        B = keys.shape[0]
        arange_B = torch.arange(B, device=self.device)
        p_true = probs[arange_B, y_t]
        surprise = 1.0 - p_true

        buckets = (keys & self.mask).to(torch.int64)
        b_sig = self.sig[buckets]
        b_tgt = self.target[buckets]

        sig_match = valid.unsqueeze(1) & (b_sig == keys.unsqueeze(1)) & (b_tgt >= 0)
        exact_match = sig_match & (b_tgt == y_t.unsqueeze(1).to(torch.int32))
        wrong_match = sig_match & (b_tgt != y_t.unsqueeze(1).to(torch.int32))

        if wrong_match.any():
            wb, ww = torch.nonzero(wrong_match, as_tuple=True)
            w_bkt = buckets[wb]
            w_tgt = b_tgt[wb, ww].to(torch.int64)
            p_wrong = probs[wb, w_tgt]
            flat_idx = w_bkt * self.ways + ww
            self.weight.view(-1).scatter_add_(0, flat_idx, -(lr * 0.85) * p_wrong)
            self.weight.clamp_(0.0, 5.5)

        has_exact = exact_match.any(dim=1)
        if has_exact.any():
            eb = torch.nonzero(has_exact, as_tuple=False).squeeze(1)
            ew = exact_match[eb].to(torch.int32).argmax(dim=1)
            e_bkt = buckets[eb]
            flat_idx = e_bkt * self.ways + ew
            self.weight.view(-1).scatter_add_(0, flat_idx, lr * surprise[eb])
            self.energy.view(-1).scatter_add_(0, flat_idx, 0.5 + surprise[eb])
            self.hits.view(-1).scatter_add_(0, flat_idx, torch.ones_like(eb, dtype=torch.int32))
            self.weight.clamp_(0.0, 5.5)
            self.energy.clamp_(0.0, 100.0)

        need_new = valid & (~has_exact) & (surprise >= surprise_threshold) & (y_t > 0)
        if need_new.any():
            nb = torch.nonzero(need_new, as_tuple=False).squeeze(1)
            n_bkt = buckets[nb]
            order = torch.argsort(surprise[nb], descending=True)
            nb = nb[order]
            n_bkt = n_bkt[order]
            sort_ord = torch.argsort(n_bkt, stable=True)
            s_bkt = n_bkt[sort_ord]
            s_nb = nb[sort_ord]
            is_first = torch.ones(s_bkt.numel(), dtype=torch.bool, device=self.device)
            if s_bkt.numel() > 1:
                is_first[1:] = s_bkt[1:] != s_bkt[:-1]
            u_bkt = s_bkt[is_first]
            u_nb = s_nb[is_first]

            slot_tgt = self.target[u_bkt]
            slot_en = self.energy[u_bkt]
            empty = slot_tgt < 0
            has_empty = empty.any(dim=1)
            empty_slot = empty.to(torch.int32).argmax(dim=1)
            weak_slot = slot_en.argmin(dim=1)
            chosen_slot = torch.where(has_empty, empty_slot, weak_slot)

            weak_en = slot_en[torch.arange(u_bkt.numel(), device=self.device), chosen_slot]
            can_write = has_empty | (weak_en < 1.8)
            if can_write.any():
                wb = u_bkt[can_write]
                ws = chosen_slot[can_write]
                wi = u_nb[can_write]
                self.total_evictions += int((~has_empty[can_write]).sum().item())
                self.total_writes += int(wb.numel())

                self.sig[wb, ws] = keys[wi]
                self.target[wb, ws] = y_t[wi].to(torch.int32)
                self.weight[wb, ws] = (lr * surprise[wi]).clamp(0.1, 2.0)
                self.energy[wb, ws] = (0.4 + 0.6 * surprise[wi]).clamp(0.4, 1.5)
                self.hits[wb, ws] = 1

    @torch.no_grad()
    def sleep_consolidate(self, min_hits: int = 2, min_weight: float = 0.08) -> Dict[str, int]:
        occupied = self.target >= 0
        before = int(occupied.sum().item())
        dead = occupied & ((self.hits < min_hits) | (self.weight < min_weight))
        n_pruned = int(dead.sum().item())
        if n_pruned > 0:
            self.sig[dead] = 0
            self.target[dead] = -1
            self.weight[dead] = 0.0
            self.energy[dead] = 0.0
            self.hits[dead] = 0
            self.total_pruned += n_pruned
        after = int((self.target >= 0).sum().item())
        return {"before": before, "after": after, "pruned": n_pruned}

    def active_count(self) -> int:
        return int((self.target >= 0).sum().item())


class LongTermMemorySystem:

    def __init__(self, cfg: LongTermMemoryConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.bank_char4 = SetAssociativeLTMBank(cfg.num_buckets, cfg.ways, device)
        self.bank_chunk = SetAssociativeLTMBank(cfg.num_buckets, cfg.ways, device)
        self.bank_topic = SetAssociativeLTMBank(cfg.num_buckets, cfg.ways, device)
        self.sample_registry: Dict[int, str] = {}

    def active_counts(self) -> Dict[str, int]:
        return {
            "char4": self.bank_char4.active_count(),
            "chunk_pair": self.bank_chunk.active_count(),
            "topic_ctx": self.bank_topic.active_count(),
            "total": (
                self.bank_char4.active_count()
                + self.bank_chunk.active_count()
                + self.bank_topic.active_count()
            ),
        }

    def sleep_consolidate_all(self, min_hits: int = 2) -> Dict[str, Dict[str, int]]:
        return {
            "char4": self.bank_char4.sleep_consolidate(min_hits=min_hits),
            "chunk_pair": self.bank_chunk.sleep_consolidate(min_hits=min_hits),
            "topic_ctx": self.bank_topic.sleep_consolidate(min_hits=min_hits),
        }


@torch.no_grad()
def run_three_system_pass(
    cortex: NeuroGraphV2,
    ids: np.ndarray,
    ltm: Optional[LongTermMemorySystem] = None,
    ltm_train: bool = False,
    ltm_use_topic: bool = True,
    stm_cfg: Optional[EpisodicMemoryConfig] = None,
    batch_streams: int = 256,
    lr_scale: float = 1.0,
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

    max_delay = max(max(cfg.char_delays), 4)
    char_hist = torch.full((B, max_delay), -1, dtype=torch.int64, device=device)
    hist_ptr = 0

    cur_chunk = torch.full((B,), -1, dtype=torch.int64, device=device)
    prev_chunk = torch.full((B,), -1, dtype=torch.int64, device=device)

    topic_node = torch.full((B,), -1, dtype=torch.int64, device=device)
    topic_score = torch.zeros((B,), dtype=torch.float32, device=device)

    membrane = torch.zeros((B, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    node_fatigue = torch.zeros((B, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    char_fatigue = torch.zeros((B, V), dtype=torch.float32, device=device)
    delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]

    stm = ShortTermMemoryBank(B, stm_cfg, device) if (stm_cfg is not None and stm_cfg.enabled) else None

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

    min_hits = 1 if ltm_train else (ltm.cfg.min_hits_eval if ltm is not None else 2)
    ltm_lr = (ltm.cfg.lr * lr_scale) if ltm is not None else 0.0

    for t in range(T - 1):
        x_t = data[:, t]
        y_t = data[:, t + 1]

        char_hist[:, hist_ptr] = x_t
        cur_hist_idx = hist_ptr
        hist_ptr = (hist_ptr + 1) % max_delay

        c1 = x_t
        c2 = char_hist[:, (cur_hist_idx - 1) % max_delay]
        c3 = char_hist[:, (cur_hist_idx - 2) % max_delay]
        c4 = char_hist[:, (cur_hist_idx - 3) % max_delay]

        completed = torch.zeros(B, dtype=torch.bool, device=device)
        if cortex.num_chunks > 0:
            merged = cortex._lookup_pairs(cur_chunk, x_t)
            hit = (cur_chunk >= 0) & (merged >= 0)
            completed = (~hit) & (cur_chunk >= 0)
            if completed.any():
                prev_chunk = torch.where(completed, cur_chunk, prev_chunk)
                c_idx = torch.nonzero(completed, as_tuple=False).squeeze(1)
                pc = prev_chunk[c_idx]
                sal = cortex._homeostatic_salience(pc)
                is_multi = cortex.chunk_len[pc.clamp(0, cortex.max_nodes - 1)] >= 2
                cand_score = sal * is_multi.to(torch.float32)
                better = cand_score > topic_score[c_idx]
                if better.any():
                    b_sub = c_idx[better]
                    topic_node[b_sub] = pc[better]
                    topic_score[b_sub] = cand_score[better]
            cur_chunk = torch.where(hit, merged, x_t)
        else:
            prev_chunk = cur_chunk
            cur_chunk = x_t

        topic_score.mul_(0.975)
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

        cortex_lp = torch.log_softmax(logits_cortex, dim=-1)
        cortex_ent = -(torch.exp(cortex_lp) * cortex_lp).sum(dim=-1)
        is_hard = cortex_ent > 3.5

        logits_total = logits_cortex.clone()

        sig_c4 = sig_ch = sig_tp = None
        v_c4 = v_ch = v_tp = None
        if ltm is not None and ltm.cfg.enabled:
            v_c4 = (c1 > 0) & (c2 > 0) & (c3 > 0) & (c4 > 0)
            sig_c4 = make_sig_4(c1, c2, c3, c4)
            ltm.bank_char4.recall(sig_c4, v_c4, ltm.cfg.gain_char4, min_hits, logits_total)

            v_ch = (prev_chunk >= V) & (cur_chunk >= 0) & (c1 > 0)
            sig_ch = make_sig_3(prev_chunk, cur_chunk, c1)
            ltm.bank_chunk.recall(sig_ch, v_ch, ltm.cfg.gain_chunk_pair, min_hits, logits_total)

            if ltm_use_topic and ltm.cfg.use_topic_context:
                v_tp = (topic_node >= V) & (topic_score > 0.25) & (c1 > 0) & (c2 > 0)
                sig_tp = make_sig_3(topic_node, c1, c2)
                ltm.bank_topic.recall(sig_tp, v_tp, ltm.cfg.gain_topic_ctx, min_hits, logits_total)

        stm_res = None
        if stm is not None:
            recall_wave, stm_res = stm.recall(c1, c2, c3, cur_chunk, prev_chunk, V, logits_total)
            logits_total = logits_total + recall_wave

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

        if ltm_train and ltm is not None and ltm.cfg.enabled:
            probs = torch.exp(log_probs)
            assert sig_c4 is not None and v_c4 is not None
            ltm.bank_char4.local_update(sig_c4, v_c4, y_t, probs, ltm_lr, ltm.cfg.surprise_threshold)
            assert sig_ch is not None and v_ch is not None
            ltm.bank_chunk.local_update(sig_ch, v_ch, y_t, probs, ltm_lr, ltm.cfg.surprise_threshold)
            if ltm_use_topic and ltm.cfg.use_topic_context and sig_tp is not None and v_tp is not None:
                ltm.bank_topic.local_update(sig_tp, v_tp, y_t, probs, ltm_lr * 0.8, ltm.cfg.surprise_threshold)

        if stm is not None and stm_res is not None:
            p_true = torch.exp(log_probs[arange_B, y_t])
            stm.bind_on_surprise(c1, c2, c3, cur_chunk, prev_chunk, y_t, 1.0 - p_true, stm_res)

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
        "stm_bindings": stm.total_bindings if stm is not None else 0,
        "stm_recalls": stm.total_recalls if stm is not None else 0,
    }


@torch.no_grad()
def generate_three_system(
    cortex: NeuroGraphV2,
    prompt_ids: List[int],
    ltm: Optional[LongTermMemorySystem] = None,
    stm_cfg: Optional[EpisodicMemoryConfig] = None,
    max_new_chars: int = 90,
    temperature: float = 0.6,
    top_k: int = 15,
) -> List[int]:
    cfg = cortex.cfg
    device = cortex.device
    V = cortex.V
    N = cortex.num_active_nodes

    max_delay = max(max(cfg.char_delays), 4)
    char_hist = torch.full((1, max_delay), -1, dtype=torch.int64, device=device)
    hist_ptr = 0
    cur_chunk = torch.full((1,), -1, dtype=torch.int64, device=device)
    prev_chunk = torch.full((1,), -1, dtype=torch.int64, device=device)
    topic_node = torch.full((1,), -1, dtype=torch.int64, device=device)
    topic_score = torch.zeros((1,), dtype=torch.float32, device=device)

    membrane = torch.zeros((1, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    node_fatigue = torch.zeros((1, N), dtype=torch.float32, device=device) if cfg.n_hops > 0 else None
    char_fatigue = torch.zeros((1, V), dtype=torch.float32, device=device)
    delay_gains = [cfg.gain_char_base / (1.0 + 0.35 * (d - 1)) for d in cfg.char_delays]
    stm = ShortTermMemoryBank(1, stm_cfg, device) if (stm_cfg is not None and stm_cfg.enabled) else None

    def step_in(ch_id: int) -> torch.Tensor:
        nonlocal cur_chunk, prev_chunk, hist_ptr
        x_t = torch.tensor([ch_id], dtype=torch.int64, device=device)
        char_hist[0, hist_ptr] = ch_id
        cur_hist_idx = hist_ptr
        hist_ptr = (hist_ptr + 1) % max_delay

        c1 = x_t
        c2 = char_hist[:, (cur_hist_idx - 1) % max_delay]
        c3 = char_hist[:, (cur_hist_idx - 2) % max_delay]
        c4 = char_hist[:, (cur_hist_idx - 3) % max_delay]

        completed = False
        if cortex.num_chunks > 0:
            merged = cortex._lookup_pairs(cur_chunk, x_t)
            hit = (cur_chunk >= 0) & (merged >= 0)
            if (not hit.item()) and cur_chunk.item() >= 0:
                prev_chunk = cur_chunk.clone()
                completed = True
                sal = cortex._homeostatic_salience(prev_chunk)
                if cortex.chunk_len[prev_chunk.clamp(0, cortex.max_nodes - 1)].item() >= 2 and sal.item() > topic_score.item():
                    topic_node[0] = prev_chunk[0]
                    topic_score[0] = sal[0]
            cur_chunk = torch.where(hit, merged, x_t)
        else:
            prev_chunk = cur_chunk
            cur_chunk = x_t

        topic_score.mul_(0.975)
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

        logits = cortex.out_bias.unsqueeze(0).clone()
        for d_idx, d in enumerate(cfg.char_delays):
            src = char_hist[:, (cur_hist_idx - (d - 1)) % max_delay]
            act = torch.tensor([delay_gains[d_idx]], dtype=torch.float32, device=device)
            cortex.char_banks[d_idx].propagate(src, act, logits)
        if cortex.num_chunks > 0:
            if cur_chunk.item() >= V:
                cortex.chunk_bank.propagate(cur_chunk, torch.tensor([cfg.gain_chunk], device=device), logits)
            if prev_chunk.item() >= 0:
                cortex.prev_chunk_bank.propagate(prev_chunk, torch.tensor([cfg.gain_prev_chunk], device=device), logits)
        if hop_winners:
            cortex._apply_relay_readout(hop_winners, logits)
        logits.sub_(char_fatigue)

        if ltm is not None and ltm.cfg.enabled:
            min_h = ltm.cfg.min_hits_eval
            v_c4 = (c1 > 0) & (c2 > 0) & (c3 > 0) & (c4 > 0)
            ltm.bank_char4.recall(make_sig_4(c1, c2, c3, c4), v_c4, ltm.cfg.gain_char4, min_h, logits)
            v_ch = (prev_chunk >= V) & (cur_chunk >= 0) & (c1 > 0)
            ltm.bank_chunk.recall(make_sig_3(prev_chunk, cur_chunk, c1), v_ch, ltm.cfg.gain_chunk_pair, min_h, logits)
            if ltm.cfg.use_topic_context:
                v_tp = (topic_node >= V) & (topic_score > 0.25) & (c1 > 0) & (c2 > 0)
                ltm.bank_topic.recall(make_sig_3(topic_node, c1, c2), v_tp, ltm.cfg.gain_topic_ctx, min_h, logits)

        if stm is not None:
            rw, _ = stm.recall(c1, c2, c3, cur_chunk, prev_chunk, V, logits)
            logits = logits + rw

        logits[0, 0] = -1e9
        return logits[0]

    out = list(prompt_ids)
    logits = None
    for ch in prompt_ids:
        logits = step_in(ch)
    for _ in range(max_new_chars):
        assert logits is not None
        sc = logits / max(temperature, 1e-4)
        vals, idx = torch.topk(sc, k=min(top_k, V))
        probs = torch.softmax(vals, dim=-1)
        pick = int(idx[torch.multinomial(probs, 1).item()].item())
        out.append(pick)
        logits = step_in(pick)
    return out
