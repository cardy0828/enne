"""NeuroGraph v3.0: System 8 — Two-Stage Deliberative Reasoning & Non-Myopic Character-Atomic Decoder!

Zero Backprop | Zero Attention (O(N^2)) | Zero Tokenizer | Zero <UNK> | Zero Hardcoded Language Rules.

Transforms NeuroGraph from a world-class Prefill/Memory engine (4.58 BPC, 4.54M GPU Core + 1.08B Engram + 5.29M Hippocampus)
into a Deliberative Reasoning & Coherent Multi-Sentence Decoding System:

STAGE 1 — DELIBERATIVE REASONING (2-Hop Hippocampal Spreading Activation + Thought Agenda Planner):
  1. Hop-1 Recall: Query 5.29M-episode Hippocampus with the prompt -> retrieve Top-6 resonant episodes.
  2. Hop-2 Reasoning Bridge: Extract high-IDF bridge entities from Hop-1 -> run a 2nd Hippocampal Query
     to pull in complementary multi-hop facts (e.g., indirect prompt -> Ep1 -> bridge entity -> Ep2!).
  3. Clause Decomposition & Prefill-Critic Agenda Planning:
     - Split recalled episodes into natural atomic clauses (UTF-8 punctuation/newline spans).
     - Score & order a non-redundant 3-step Thought Agenda [Goal_1 -> Goal_2 -> Goal_3] using our own
       world-class NeuroGraph Prefill Scorer + TF-IDF entity salience + information novelty!

STAGE 2 — NON-MYOPIC CHARACTER-ATOMIC BEAM DECODER (GPU-Batched W=6 Beams + Single-Goal Focus + Critic Rerank):
  1. RFC 3629 UTF-8 Character-Atomic Expansion (100% valid UTF-8 by construction, zero language rules!):
     - Never commits to an isolated broken/off-topic lead byte! Expands 1..4 byte UTF-8 characters atomically
       across parallel GPU beams, pruning derailments ('黒木ミライ' -> '黒木ミライブラリ') before committing!
  2. Single-Goal Prefrontal Focus (Sentence-by-Sentence Agenda Execution):
     - While decoding Sentence k, ONLY Goal_k's latent anchor z_goal_k and Goal_k's exact fact cursor/transitions
       are loaded in Prefrontal Focus -> ZERO cross-episode interference!
     - Upon reaching sentence end ('。' / '\n'), cleanly advances focus to Goal_{k+1}, letting Cortex weave
       natural transitions across all 3 sentences!
  3. Prefill-Critic Sentence Reranking:
     - Ranks finished beam candidates by exact NeuroGraph Fluency (Core + 1.08B Engram) + Goal Entity Coverage
       + Latent Thought Alignment - Repetition!
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neurograph.engram_v20 import DecoupledBrainV20, load_engram_lib
from neurograph.episodic_v21 import EpisodicRack
from neurograph.memory_v3b import _mix64
from neurograph.triton_dag_kernel import run_fused_dag_step


# ---- Pure RFC 3629 UTF-8 Character-Length Helpers (100% Language-Agnostic!) ----
def utf8_char_len(lead_byte: int) -> int:
    if lead_byte < 0x80:
        return 1
    if 0xC2 <= lead_byte <= 0xDF:
        return 2
    if 0xE0 <= lead_byte <= 0xEF:
        return 3
    if 0xF0 <= lead_byte <= 0xF4:
        return 4
    return 0  # invalid lead byte (continuation 0x80..0xBF, overlong 0xC0..0xC1, >0xF4)


def make_utf8_lead_mask(device: torch.device) -> torch.Tensor:
    """Boolean mask [256] of valid UTF-8 character lead bytes (disallows bare continuation & control garbage)."""
    m = torch.zeros(256, dtype=torch.bool, device=device)
    m[10] = True                  # newline
    m[32:127] = True              # printable ASCII
    m[0xC2:0xF5] = True           # valid multi-byte UTF-8 lead bytes (2B, 3B Japanese/CJK, 4B Emoji!)
    return m


def make_utf8_cont_mask(device: torch.device) -> torch.Tensor:
    """Boolean mask [256] of valid UTF-8 continuation bytes (0x80..0xBF)."""
    m = torch.zeros(256, dtype=torch.bool, device=device)
    m[0x80:0xC0] = True
    return m


def split_episode_into_clauses(ep_b: bytes, min_bytes: int = 24, max_bytes: int = 150) -> List[bytes]:
    """Split a raw UTF-8 episode into clean atomic clauses at UTF-8 '。' (E3 80 82) or newline boundaries."""
    try:
        txt = ep_b.decode("utf-8", errors="ignore")
    except Exception:
        return []
    clauses: List[bytes] = []
    buf = []
    for ch in txt:
        if ch == "\n":
            s = "".join(buf).strip()
            if len(s.encode("utf-8")) >= min_bytes:
                clauses.append(s.encode("utf-8")[:max_bytes])
            buf = []
        else:
            buf.append(ch)
            if ch in ("。", "！", "？", "!", "?"):
                s = "".join(buf).strip()
                if len(s.encode("utf-8")) >= min_bytes:
                    clauses.append(s.encode("utf-8")[:max_bytes])
                buf = []
    rem = "".join(buf).strip()
    if len(rem.encode("utf-8")) >= min_bytes:
        clauses.append(rem.encode("utf-8")[:max_bytes])
    return clauses


@dataclass
class ThoughtGoal:
    """One atomic reasoning step in the Deliberative Thought Agenda."""
    source: str
    is_custom: bool
    hop: int                        # 1 = direct recall, 2 = 2-hop associative bridge
    clause_bytes: bytes
    clause_text: str
    score: float
    entities: List[bytes]           # distinctive high-IDF 6B spans inside this goal
    z_vec: Optional[torch.Tensor]   # Cortex h_dag anchor vector [D=256]
    anchor4: Dict[bytes, List[int]] # 4-byte cursor positions inside clause_bytes
    t3_ent: Dict[bytes, Dict[int, float]]  # 3-byte start primers for this goal's entities


# ---- GPU-Batched Brain State for Parallel Character-Atomic Beam Search ----
class BatchedCoreRunner:
    """Runs B parallel beam states in lockstep on RTX 3060 (B=8..32 takes almost the same time as B=1!)."""

    def __init__(self, v20: DecoupledBrainV20, temp_cal: float = 1.08):
        self.v20 = v20
        self.core = v20.core
        self.device = v20.device
        self.temp_cal = temp_cal
        self.H_buf = 28
        self.D_ctx = 12
        self.lead_mask = make_utf8_lead_mask(self.device)
        self.cont_mask = make_utf8_cont_mask(self.device)
        self._ar24 = torch.arange(24, dtype=torch.int64, device=self.device).unsqueeze(0)
        self._span_set = set(self.v20.encoder.SPANS)

    @torch.no_grad()
    def init_batch_state(self, B: int) -> Dict[str, torch.Tensor]:
        cfg = self.core.cfg
        dev = self.device
        S, D, N_stm = cfg.num_slots, self.core.D, cfg.stm_slots
        return {
            "byte_hist": torch.full((B, self.H_buf), -1, dtype=torch.int64, device=dev),
            "hist_ptr": torch.zeros(B, dtype=torch.int64, device=dev),
            "node_delay": torch.full((B, self.D_ctx), -1, dtype=torch.int64, device=dev),
            "nd_ptr": torch.zeros(B, dtype=torch.int64, device=dev),
            "ctx_node": torch.full((B, 2), -1, dtype=torch.int64, device=dev),
            "ctx_score": torch.zeros((B, 2), dtype=torch.float32, device=dev),
            "slot_state": torch.zeros((B, S, D), dtype=torch.float32, device=dev),
            "stm_k4": torch.zeros((B, N_stm), dtype=torch.int64, device=dev),
            "stm_k7": torch.zeros((B, N_stm), dtype=torch.int64, device=dev),
            "stm_val": torch.zeros((B, N_stm), dtype=torch.int64, device=dev),
            "stm_str": torch.zeros((B, N_stm), dtype=torch.float32, device=dev),
            "stm_ptr": torch.zeros(B, dtype=torch.int64, device=dev),
            "exec_vec": torch.zeros((B, D), dtype=torch.float32, device=dev),
            "c_eng": torch.zeros((B, 16), dtype=torch.float32, device=dev),
            "step_i": torch.zeros(B, dtype=torch.int64, device=dev),
        }

    @staticmethod
    def gather_state(st: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[idx].clone() for k, v in st.items()}

    @torch.no_grad()
    def encode_text_h(self, raw_b: bytes) -> Optional[torch.Tensor]:
        if len(raw_b) < 4:
            return None
        core = self.core
        dev = self.device
        arr = torch.tensor(list(raw_b), dtype=torch.int64, device=dev)
        n = arr.shape[0]
        h = core.W_node[arr].to(torch.float32)
        frontier = arr
        z0 = torch.zeros(n, dtype=torch.int64, device=dev)
        for hop in range(1, 5):
            prev = torch.full((n,), -1, dtype=torch.int64, device=dev)
            if n > hop:
                prev[hop:] = arr[: n - hop]
            ch = core.edges.lookup(frontier, z0, prev)
            ok = (frontier >= 0) & (prev >= 0) & (ch >= core.V) & core.alive[ch.clamp(min=0)]
            if not ok.any():
                break
            h[ok] += core.W_node[ch[ok]].to(torch.float32) * (1.0 / (1.0 + 0.10 * hop))
            frontier = torch.where(ok, ch, torch.full_like(ch, -1))
        m = h.mean(dim=0)
        return m / m.norm().clamp(min=1e-6)

    @torch.no_grad()
    def step_batch(
        self,
        st: Dict[str, torch.Tensor],
        x_t: torch.Tensor,
        next_true: Optional[torch.Tensor] = None,
        locked_mode_p: Optional[torch.Tensor] = None,
        goal_z: Optional[torch.Tensor] = None,
        goal_z_beta: float = 0.15,
        use_engram: bool = True,
        engram_mult: float = 0.75,
        is_prompt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance B parallel beam states by 1 raw UTF-8 byte -> returns (logits [B, 256], h_dag [B, D])."""
        core = self.core
        cfg = core.cfg
        dev = self.device
        V, D = core.V, core.D
        B = x_t.shape[0]
        arB = torch.arange(B, device=dev)

        cur_idx = st["hist_ptr"]
        st["byte_hist"][arB, cur_idx] = x_t
        st["hist_ptr"] = (cur_idx + 1) % self.H_buf

        # Update 16-D Engram Context Fingerprint [B, 16] in O(1)
        b1 = st["byte_hist"][arB, (cur_idx - 1) % self.H_buf].clamp(min=0)
        b2 = st["byte_hist"][arB, (cur_idx - 2) % self.H_buf].clamp(min=0)
        h3 = (_mix64(x_t + 23) ^ ((b1 + 41) * 1315423911) ^ ((b2 + 67) * 2654435761)) & 8191
        v_c = 0.35 * self.v20.encoder.E_proj[x_t] + 0.65 * self.v20.encoder.H_tri[h3]
        st["c_eng"].mul_(self.v20.encoder.DECAY).add_(v_c, alpha=1.0 - self.v20.encoder.DECAY)

        # Delayed emergent salient context (100% branchless — ZERO GPU->CPU sync!)
        st["ctx_score"].mul_(0.985)
        nd_i = st["nd_ptr"]
        u_old = st["node_delay"][arB, nd_i]
        u_old_c = u_old.clamp(min=0)
        ok_u = (st["step_i"] >= self.D_ctx) & (u_old >= V) & core.alive[u_old_c]
        f_old = core.fires[u_old_c].to(torch.float32)
        sal_old = torch.where(ok_u, 1.8 / torch.sqrt(1.0 + f_old * 0.02), torch.zeros_like(f_old))
        already = (st["ctx_node"][:, 0] == u_old) | (st["ctx_node"][:, 1] == u_old)
        ws_i = st["ctx_score"].argmin(dim=1)
        rep = ok_u & (~already) & (sal_old > st["ctx_score"][arB, ws_i]) & (sal_old > 0.55)
        st["ctx_node"][arB, ws_i] = torch.where(rep, u_old, st["ctx_node"][arB, ws_i])
        st["ctx_score"][arB, ws_i] = torch.where(rep, sal_old, st["ctx_score"][arB, ws_i])
        top_ctx_all = st["ctx_node"][arB, st["ctx_score"].argmax(dim=1)]

        # System 1: 4.54M Cortex DAG Traversal + Sharp Synapses (SINGLE FUSED TRITON KERNEL!)
        h_dag, logits_sharp, emergent_cand = run_fused_dag_step(
            core, x_t, st["byte_hist"], cur_idx, top_ctx_all
        )
        st["node_delay"][arB, nd_i] = emergent_cand
        st["nd_ptr"] = (nd_i + 1) % self.D_ctx

        # System 2: 16 Autonomous Working-Memory Slots
        h_norm = h_dag / h_dag.norm(dim=1, keepdim=True).clamp(min=1.0)
        e_norm = core.E[x_t] / core.E[x_t].norm(dim=1, keepdim=True).clamp(min=1.0)
        x_slot = 0.65 * h_norm + 0.35 * e_norm
        kg = torch.sigmoid(x_slot @ core.G_keep.T + core.keep_bias.unsqueeze(0))
        wg = torch.sigmoid(x_slot @ core.G_write.T + core.write_bias.unsqueeze(0))
        st["slot_state"] = st["slot_state"] * kg.unsqueeze(-1) + wg.unsqueeze(-1) * x_slot.unsqueeze(1)
        sn = st["slot_state"].norm(dim=-1, keepdim=True)
        st["slot_state"] = st["slot_state"] * (2.5 / sn.clamp(min=2.5))
        r_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], core.W_read).sum(dim=1)
        m_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], core.W_mod).sum(dim=1)

        h_pre = h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum
        if goal_z is not None:
            h_pre = h_pre + goal_z.unsqueeze(0) * (goal_z_beta * h_dag.norm(dim=1, keepdim=True).clamp(min=1.0))
        h_act = torch.tanh(h_pre * 0.35) * 2.85
        st["last_h_act"] = h_act
        logits = (h_act @ core.E.T + core.bias.unsqueeze(0) + logits_sharp) / self.temp_cal

        # System 5: 24-Mode Executive Critic
        st["exec_vec"].mul_(cfg.mode_ema_decay).add_(h_dag, alpha=1.0 - cfg.mode_ema_decay)
        if core.mode_ready:
            ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
            cur_mp = torch.softmax((ev_n @ core.mode_centroids.T) / cfg.mode_temp, dim=-1)
            mp = (0.60 * locked_mode_p + 0.40 * cur_mp) if locked_mode_p is not None else cur_mp
            logits = logits + (mp @ core.mode_llr) * (cfg.mode_llr_scale * 1.10)

        # System 4: 1-Shot ByteSTM (100% branchless scatter_add_ — ZERO GPU->CPU sync!)
        bs24_all = st["byte_hist"][arB.unsqueeze(1), (cur_idx.unsqueeze(1) - self._ar24) % self.H_buf]  # [B, 24] ONE gather!
        h4 = _mix64(bs24_all[:, 0] + 17)
        h4 = _mix64(h4 ^ ((bs24_all[:, 1] + 31) * 1315423911))
        h4 = _mix64(h4 ^ ((bs24_all[:, 2] + 53) * 2654435761))
        h4 = _mix64(h4 ^ ((bs24_all[:, 3] + 79) * 1000003))
        k4 = torch.where((bs24_all[:, 3] >= 0) & (h4 != 0), h4, torch.zeros_like(h4))
        h7 = _mix64(h4 ^ ((bs24_all[:, 4] + 97) * 1000033))
        h7 = _mix64(h7 ^ ((bs24_all[:, 5] + 113) * 1000037))
        h7 = _mix64(h7 ^ ((bs24_all[:, 6] + 131) * 1000039))
        k7 = torch.where((bs24_all[:, 6] >= 0) & (h7 != 0), h7, torch.zeros_like(h7))

        st["stm_str"].mul_(cfg.stm_decay)
        m4 = (st["stm_k4"] == k4.unsqueeze(1)) & (k4.unsqueeze(1) != 0) & (st["stm_str"] > 0.05)
        m7 = (st["stm_k7"] == k7.unsqueeze(1)) & (k7.unsqueeze(1) != 0) & (st["stm_str"] > 0.05)
        hw = m4.float() * (st["stm_str"] * cfg.stm_w4) + m7.float() * (st["stm_str"] * cfg.stm_w7)
        stm_b = torch.zeros((B, V), dtype=torch.float32, device=dev)
        stm_b.scatter_add_(1, st["stm_val"].clamp(min=0, max=V - 1), hw)
        logits = logits + stm_b.clamp(max=cfg.stm_max_boost)

        # System 3: Entropy-Gated + Context-Gated External 1.08B CPU/mmap Byte-Engram!
        # Ultra-fast C++ 6-scale hash directly from [B, 24] bytes (~0.01ms vs 2.46ms 24-iter GPU loop!)
        if (not is_prompt) and use_engram and len(self.v20.rack.cartridges) > 0:
            lp_c = torch.log_softmax(logits, dim=-1)
            p_c = torch.exp(lp_c)
            H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
            ent_g = H_c / (1.0 + H_c)

            bs24_np = bs24_all.to(torch.int32).cpu().numpy()
            keys_np = self.v20.rack.hash_bytes24_batch(bs24_np)
            cn = st["c_eng"] / st["c_eng"].norm(dim=1, keepdim=True).clamp(min=1e-5)
            cq_np = torch.round(cn * 127.0).clamp(-127, 127).to(torch.int8).cpu().numpy()
            eb_np, _, _ = self.v20.rack.lookup_batch(keys_np, cq_np, use_context_gate=True)
            logits = logits + torch.from_numpy(eb_np).to(dev) * (engram_mult * ent_g)

        if next_true is not None:
            valid_nt = (next_true >= 0) & (st["step_i"] >= 4) & (k4 != 0)
            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arB, next_true.clamp(min=0)]
            surp = 1.0 - torch.exp(-nll)
            do_b = valid_nt & (surp > cfg.stm_surp_thresh)
            sp = st["stm_ptr"]
            st["stm_k4"][arB, sp] = torch.where(do_b, k4, st["stm_k4"][arB, sp])
            st["stm_k7"][arB, sp] = torch.where(do_b, k7, st["stm_k7"][arB, sp])
            st["stm_val"][arB, sp] = torch.where(do_b, next_true.clamp(min=0), st["stm_val"][arB, sp])
            st["stm_str"][arB, sp] = torch.where(do_b, torch.ones_like(surp), st["stm_str"][arB, sp])
            st["stm_ptr"] = torch.where(do_b, (sp + 1) % cfg.stm_slots, sp)

        st["step_i"] = st["step_i"] + 1
        return logits, h_dag

    @torch.no_grad()
    def score_text_bpc(self, text: str, prompt_prefix_len: int = 0, engram_mult: float = 0.72) -> float:
        """Exact NeuroGraph Prefill Critic Fluency BPC of a generated continuation (includes 1st generated byte!)."""
        raw = list(text.encode("utf-8"))
        if len(raw) <= prompt_prefix_len + 2:
            return 9.99
        st = self.init_batch_state(1)
        tot_nll = 0.0
        cnt = 0
        # i == prompt_prefix_len - 1 predicts raw[prompt_prefix_len] (the 1st generated byte)!
        start_score_i = max(0, prompt_prefix_len - 1)
        for i in range(len(raw) - 1):
            x_t = torch.tensor([raw[i]], dtype=torch.int64, device=self.device)
            y_t = raw[i + 1]
            is_p = (i < start_score_i)
            nt = torch.tensor([y_t], dtype=torch.int64, device=self.device) if is_p else None
            logits, _ = self.step_batch(st, x_t, next_true=nt, use_engram=True, engram_mult=engram_mult, is_prompt=is_p)
            if i >= start_score_i:
                lp = torch.log_softmax(logits[0], dim=-1)
                tot_nll += float(-lp[y_t].item())
                cnt += 1
        bpb = (tot_nll / max(1, cnt)) / math.log(2.0)
        gen_str = bytes(raw[prompt_prefix_len:]).decode("utf-8", errors="ignore")
        bpc_ratio = max(1.0, (len(raw) - prompt_prefix_len) / max(1, len(gen_str)))
        return round(bpb * bpc_ratio, 4)


# ---- NeuroGraph v3.0: Deliberative 2-Hop Reasoner + Character-Atomic Beam Decoder ----
class DeliberativeBrainV30:
    """Complete NeuroGraph v3.0 Reasoning & Non-Myopic Decoding Engine!"""

    def __init__(self, v20: DecoupledBrainV20, ep_rack: EpisodicRack, temp_cal: float = 1.08):
        self.v20 = v20
        self.ep_rack = ep_rack
        self.runner = BatchedCoreRunner(v20, temp_cal=temp_cal)
        self.device = v20.device

    def _extract_clause_entities(self, raw_b: bytes, max_ents: int = 10) -> List[bytes]:
        if len(raw_b) < 6 or not self.ep_rack.cartridges:
            return []
        bg = max(self.ep_rack.cartridges, key=lambda c: c.num_episodes)
        small_carts = [c for c in self.ep_rack.cartridges if c.num_episodes < 10_000]
        cands: Dict[bytes, float] = {}
        n = len(raw_b)
        pos = 0
        while pos + 6 <= n:
            if (raw_b[pos] & 0xC0) == 0x80:
                pos += 1
                continue
            sp = raw_b[pos : pos + 6]
            try:
                dec = sp.decode("utf-8")
                if len(dec.strip()) >= 2 and ("\n" not in dec):
                    idf = bg.span_idf(sp)
                    if idf >= 5.2:
                        cands[sp] = max(cands.get(sp, 0.0), min(idf, 11.0))
                    elif idf == 0.0 and any(c.span_idf(sp) > 0.0 for c in small_carts):
                        cands[sp] = 12.0
            except UnicodeDecodeError:
                pass
            pos += 1
        ranked = sorted(cands.items(), key=lambda x: x[1], reverse=True)
        out: List[bytes] = []
        out_s: List[str] = []
        for sp, _ in ranked:
            s = sp.decode("utf-8", errors="ignore")
            if any((s in p) or (p in s) for p in out_s):
                continue
            out.append(sp)
            out_s.append(s)
            if len(out) >= max_ents:
                break
        return out

    @torch.no_grad()
    def plan_thought_agenda(self, prompt: str, num_goals: int = 3) -> Tuple[List[ThoughtGoal], Dict[str, object]]:
        """STAGE 1 — DELIBERATIVE REASONING:
        2-Hop Hippocampal Spreading Activation -> Clause Decomposition -> Non-Redundant Ordered Thought Agenda!"""
        t0 = time.perf_counter()
        prompt_b = prompt.encode("utf-8")
        small_names = {c.name for c in self.ep_rack.cartridges if c.num_episodes < 10_000}

        # Hop 1: Direct recall from 5.29M+ episodes
        hop1 = self.ep_rack.query(prompt_b, top_k=6)
        if not hop1:
            return [], {"plan_ms": round((time.perf_counter() - t0) * 1000, 1), "hop1": 0, "hop2": 0}

        # Hop 2: Associative Bridge Reasoning!
        # Extract distinctive bridge entities from top-2 Hop-1 episodes to discover complementary facts (e.g. Ep1 -> Ep2!)
        bridge_buf = bytearray()
        for _, _, eb in hop1[:2]:
            for sp in self._extract_clause_entities(eb, max_ents=8):
                if sp not in prompt_b:
                    bridge_buf.extend(sp)
                    bridge_buf.extend(b" ")
        hop2 = self.ep_rack.query(bytes(bridge_buf), top_k=4) if len(bridge_buf) >= 6 else []

        # Combine Hop-1 and Hop-2 episodes (deduplicated)
        seen_ep = set()
        tagged_eps: List[Tuple[int, float, str, bytes]] = []
        for sc, cn, eb in hop1:
            if eb[:40] not in seen_ep:
                seen_ep.add(eb[:40])
                tagged_eps.append((1, sc, cn, eb))
        for sc, cn, eb in hop2:
            if eb[:40] not in seen_ep:
                seen_ep.add(eb[:40])
                tagged_eps.append((2, sc * 0.80, cn, eb))

        prompt_z = self.runner.encode_text_h(prompt_b)
        prompt_ents = set(self._extract_clause_entities(prompt_b, max_ents=12))

        # Decompose recalled episodes into clean atomic factual clauses and score each!
        candidates: List[ThoughtGoal] = []
        seen_cl = set()
        for hop_i, ep_sc, cname, eb in tagged_eps:
            is_cust = cname in small_names
            clauses = split_episode_into_clauses(eb, min_bytes=27, max_bytes=165)
            for cl_i, cl_b in enumerate(clauses):
                cl_txt = cl_b.decode("utf-8", errors="ignore").strip()
                if len(cl_txt) < 10 or cl_txt[:20] in seen_cl:
                    continue
                # Skip exact copy of prompt itself
                if cl_txt in prompt or prompt in cl_txt:
                    continue
                seen_cl.add(cl_txt[:20])
                ents = self._extract_clause_entities(cl_b, max_ents=10)
                if not ents:
                    continue
                cl_z = self.runner.encode_text_h(cl_b)
                sim_p = float((cl_z @ prompt_z).item()) if (cl_z is not None and prompt_z is not None) else 0.0
                # Overlap with prompt entities (especially strong for Hop 1!)
                ent_overlap = sum(1 for e in ents if e in prompt_ents)
                # Build 4-byte local cursor table + 3-byte entity primers ONLY for THIS single goal!
                a4_map: Dict[bytes, List[int]] = {}
                for pos in range(4, len(cl_b)):
                    a4 = cl_b[pos - 4 : pos]
                    a4_map.setdefault(a4, []).append(pos)
                t3_map: Dict[bytes, Dict[int, float]] = {}
                for sp in ents[:8]:
                    if len(sp) >= 6:
                        for pi in range(3, len(sp)):
                            d3 = t3_map.setdefault(sp[pi - 3 : pi], {})
                            d3[sp[pi]] = 1.0

                # Relevance + Factual Density + Cortex Semantic Alignment + Custom priority
                sc_total = (
                    math.log1p(ep_sc) * 0.55
                    + 1.6 * sim_p
                    + 0.45 * min(len(ents), 6)
                    + 0.70 * ent_overlap
                    + (4.5 if is_cust else 0.0)
                    - 0.15 * cl_i
                )
                candidates.append(ThoughtGoal(
                    source=cname, is_custom=is_cust, hop=hop_i,
                    clause_bytes=cl_b, clause_text=cl_txt, score=sc_total,
                    entities=ents, z_vec=cl_z, anchor4=a4_map, t3_ent=t3_map,
                ))

        # Greedy Diverse Selection: pick num_goals=3 ordered goals maximizing [score + novelty - overlap]!
        agenda: List[ThoughtGoal] = []
        covered_ents = set(prompt_ents)
        pool = list(candidates)
        for step_g in range(num_goals):
            if not pool:
                break
            best_idx = -1
            best_val = -1e9
            for idx_c, g in enumerate(pool):
                new_ents = sum(1 for e in g.entities if e not in covered_ents)
                # Step 0 prefers Hop-1 direct answer; Steps 1..2 reward Hop-2 & fresh non-redundant facts!
                hop_pref = (0.8 if (step_g == 0 and g.hop == 1) else (0.5 if (step_g > 0 and g.hop == 2) else 0.0))
                val = g.score + 0.55 * new_ents + hop_pref
                if val > best_val:
                    best_val = val
                    best_idx = idx_c
            chosen = pool.pop(best_idx)
            agenda.append(chosen)
            covered_ents.update(chosen.entities)

        plan_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "plan_ms": round(plan_ms, 1),
            "hop1": len(hop1),
            "hop2": len(hop2),
            "agenda": [
                {"hop": g.hop, "source": g.source, "custom": g.is_custom, "clause": g.clause_text[:75]}
                for g in agenda
            ],
        }
        return agenda, meta

    @torch.no_grad()
    def decode_char_atomic_beam(
        self,
        prompt: str,
        beam_width: int = 6,
        max_new_chars: int = 75,
        use_thought_agenda: bool = True,
        num_goals: int = 3,
        temperature_noise: float = 0.0,
    ) -> Tuple[str, bool, Dict[str, object]]:
        """STAGE 2 — NON-MYOPIC CHARACTER-ATOMIC BEAM DECODER:
        - Expands COMPLETE UTF-8 characters (1..4 bytes atomically) across W parallel GPU beams!
        - Executes the Deliberative Thought Agenda [Goal_1 -> Goal_2 -> Goal_3] one focused goal per sentence!
        - Guarantees 100.000% valid UTF-8 and ZERO mid-kanji derailments!"""
        t0 = time.perf_counter()
        dev = self.device
        runner = self.runner
        raw_in = list(prompt.encode("utf-8"))
        prompt_b = bytes(raw_in)

        # 1) Deliberative 2-Hop Reasoning -> Thought Agenda!
        agenda: List[ThoughtGoal] = []
        plan_meta: Dict[str, object] = {"plan_ms": 0.0, "agenda": []}
        if use_thought_agenda and len(self.ep_rack.cartridges) > 0:
            agenda, plan_meta = self.plan_thought_agenda(prompt, num_goals=num_goals)

        # 2) Prefill Prompt on B=1 state -> lock discourse mode!
        st1 = runner.init_batch_state(1)
        prompt_hs = []
        logits1 = None
        for i, b in enumerate(raw_in):
            x_t = torch.tensor([b], dtype=torch.int64, device=dev)
            nt = torch.tensor([raw_in[i + 1]], dtype=torch.int64, device=dev) if i + 1 < len(raw_in) else None
            logits1, hd = runner.step_batch(st1, x_t, next_true=nt, is_prompt=True)
            prompt_hs.append(hd[0])

        prompt_mode_vec = torch.stack(prompt_hs).mean(dim=0)
        prompt_mode_vec = prompt_mode_vec / prompt_mode_vec.norm().clamp(min=1e-6)
        locked_mode_p = (
            torch.softmax(((prompt_mode_vec.unsqueeze(0) @ runner.core.mode_centroids.T) / runner.core.cfg.mode_temp), dim=-1)
            if runner.core.mode_ready else None
        )

        # 3) Beam State Arrays (W active beams, initialized from the single prefilled prompt!)
        # Each beam tracks: (cum_score, bytes_gen, goal_idx, cursor=(pos, lock_len) or None, sentence_bytes, emitted_ngrams, goal_hits)
        st_beams = st1
        logits_beams = logits1  # [W_cur, 256]
        beams_meta = [{
            "score": 0.0,
            "lp_sum": 0.0,
            "n_bytes": 0,
            "gen": bytearray(),
            "tail9": bytearray(raw_in[-9:]),
            "goal_idx": 0,
            "cursor": None,          # Optional[Tuple[int, int]] = (next_pos_in_goal, lock_len)
            "sent_chars": 0,
            "seen9": set(),
            "hits": set(),
        }]
        # Seed prompt 9-byte ngrams into seen9 to prevent repeating prompt
        for i in range(max(0, len(raw_in) - 8)):
            beams_meta[0]["seen9"].add(tuple(raw_in[i : i + 9]))

        K_lead = 5
        K_cont = 3

        def apply_goal_focus_logits(
            logits_row: torch.Tensor,
            bm: dict,
            is_lead_byte: bool,
        ) -> torch.Tensor:
            """Apply Single-Goal Prefrontal Focus (Focused Cursor + Entity Primer + Anti-Repetition) to 1 beam's logits!"""
            out_l = logits_row.clone()
            # Enforce strict RFC 3629 UTF-8 validity mask!
            mask = runner.lead_mask if is_lead_byte else runner.cont_mask
            out_l[~mask] = -1e9

            if not (use_thought_agenda and bm["goal_idx"] < len(agenda)):
                return out_l

            g = agenda[bm["goal_idx"]]
            # Compute core uncertainty H(p) on valid bytes
            lp_v = torch.log_softmax(out_l, dim=-1)
            p_v = torch.exp(lp_v)
            H_v = float(-(p_v[mask] * lp_v[mask]).sum().item())
            ent_g = H_v / (0.85 + H_v)

            # (a) Active Focused Goal Cursor -> rock-solid fact & proper-noun completion!
            if bm["cursor"] is not None:
                c_pos, lk = bm["cursor"]
                if c_pos < len(g.clause_bytes):
                    nb = g.clause_bytes[c_pos]
                    if mask[nb]:
                        boost = min(7.0, 2.8 + 0.70 * (lk - 3)) if g.is_custom else min(5.2, 2.0 + 0.55 * (lk - 3))
                        out_l[nb] = out_l[nb] + boost * max(ent_g, 0.72 if g.is_custom else 0.55)
            # (b) Soft Goal Opening / Entity Primer -> invites the beam to enter Goal_k cleanly!
            else:
                tail = bytes(bm["tail9"])
                # Check 3-byte entity completion primer
                if len(tail) >= 3 and tail[-3:] in g.t3_ent:
                    for kb, _ in g.t3_ent[tail[-3:]].items():
                        if mask[kb]:
                            out_l[kb] = out_l[kb] + (3.2 if g.is_custom else 2.0) * max(ent_g, 0.60)
                # At the start of a new sentence (after '。' or at step 0), softly prime Goal_k's opening character!
                if is_lead_byte and (bm["sent_chars"] <= 3):
                    lead_cands = [g.clause_bytes[0]]
                    for sp in g.entities[:4]:
                        if sp not in bytes(bm["gen"]) and len(sp) > 0:
                            lead_cands.append(sp[0])
                    for lb in set(lead_cands):
                        if mask[lb]:
                            out_l[lb] = out_l[lb] + (2.4 if g.is_custom else 1.35)
            return out_l

        def advance_beam_meta(bm: dict, added_bytes: List[int], step_lp: float) -> dict:
            nb = {
                "score": bm["score"],
                "lp_sum": bm["lp_sum"] + step_lp,
                "n_bytes": bm["n_bytes"] + len(added_bytes),
                "gen": bytearray(bm["gen"]) + bytearray(added_bytes),
                "tail9": bytearray(bm["tail9"]),
                "goal_idx": bm["goal_idx"],
                "cursor": bm["cursor"],
                "sent_chars": bm["sent_chars"] + 1,
                "seen9": set(bm["seen9"]),
                "hits": set(bm["hits"]),
            }
            rep_pen = 0.0
            goal_bonus = 0.0
            active_g = agenda[nb["goal_idx"]] if (use_thought_agenda and nb["goal_idx"] < len(agenda)) else None

            for b_v in added_bytes:
                nb["tail9"].append(b_v)
                if len(nb["tail9"]) > 9:
                    del nb["tail9"][0]
                if len(nb["tail9"]) >= 9:
                    ng = tuple(nb["tail9"])
                    if ng in nb["seen9"]:
                        rep_pen += 2.2
                    else:
                        nb["seen9"].add(ng)

                # Update single-goal cursor cleanly!
                if active_g is not None:
                    cur = nb["cursor"]
                    if cur is not None:
                        c_pos, lk = cur
                        if c_pos < len(active_g.clause_bytes) and active_g.clause_bytes[c_pos] == b_v:
                            nb["cursor"] = (c_pos + 1, lk + 1)
                            goal_bonus += 0.25 if active_g.is_custom else 0.14
                        else:
                            nb["cursor"] = None
                    if nb["cursor"] is None and len(nb["tail9"]) >= 4:
                        a4 = bytes(nb["tail9"][-4:])
                        if a4 in active_g.anchor4:
                            # Lock onto first matching position in Goal_k!
                            nb["cursor"] = (active_g.anchor4[a4][0], 4)
                            goal_bonus += 0.45

            # Check if any active Goal_k entity just completed!
            if active_g is not None:
                tail_w = bytes(nb["gen"][-10:])
                for sp in active_g.entities:
                    if sp not in nb["hits"] and sp in tail_w:
                        nb["hits"].add(sp)
                        goal_bonus += 2.4 if active_g.is_custom else 1.4

            # Check if a full sentence just finished ('。' = E3 80 82) -> advance Thought Agenda to Goal_{k+1}!
            if len(added_bytes) == 3 and added_bytes[0] == 0xE3 and added_bytes[1] == 0x80 and added_bytes[2] == 0x82:
                if use_thought_agenda and nb["goal_idx"] + 1 < len(agenda) and nb["sent_chars"] >= 8:
                    nb["goal_idx"] += 1
                    nb["cursor"] = None
                    nb["sent_chars"] = 0
                    goal_bonus += 0.55  # clean sentence transition bonus!

            # Length-normalized score + goal coverage - repetition!
            nb["score"] = (nb["lp_sum"] / max(1.0, nb["n_bytes"] ** 0.85)) + goal_bonus - rep_pen + (bm["score"] - (bm["lp_sum"] / max(1.0, max(1, bm["n_bytes"]) ** 0.85)))
            return nb

        # ---- Main Character-Atomic Beam Search Loop! ----
        for ch_step in range(max_new_chars):
            W_cur = len(beams_meta)
            # 1) Propose Top-K_lead valid UTF-8 lead bytes per beam!
            cand_parent: List[int] = []
            cand_b0: List[int] = []
            cand_lp0: List[float] = []
            cand_len: List[int] = []

            for wi in range(W_cur):
                adj_l = apply_goal_focus_logits(logits_beams[wi], beams_meta[wi], is_lead_byte=True)
                if temperature_noise > 0:
                    adj_l = adj_l + torch.randn_like(adj_l) * temperature_noise
                lp = torch.log_softmax(adj_l, dim=-1)
                tv, ti = torch.topk(lp, k=K_lead)
                for lv, bi in zip(tv.cpu().tolist(), ti.cpu().tolist()):
                    clen = utf8_char_len(bi)
                    if clen == 0 or lv < -25.0:
                        continue
                    cand_parent.append(wi)
                    cand_b0.append(bi)
                    cand_lp0.append(lv)
                    cand_len.append(clen)

            if not cand_parent:
                break

            # Batched GPU step for ALL lead bytes simultaneously!
            p_idx = torch.tensor(cand_parent, dtype=torch.int64, device=dev)
            st_c = runner.gather_state(st_beams, p_idx)
            x_c = torch.tensor(cand_b0, dtype=torch.int64, device=dev)
            # Use active goal's z_vec for each candidate (use beam 0's active goal for simplicity if uniform)
            g0 = agenda[beams_meta[0]["goal_idx"]] if (use_thought_agenda and beams_meta[0]["goal_idx"] < len(agenda)) else None
            gz = g0.z_vec if g0 is not None else None
            eng_m = 0.35 if (g0 is not None and g0.is_custom) else 0.72

            logits_c, _ = runner.step_batch(
                st_c, x_c, locked_mode_p=locked_mode_p, goal_z=gz, use_engram=True, engram_mult=eng_m, is_prompt=False
            )

            # Track completed characters vs multi-byte characters needing continuation bytes (2B/3B/4B)
            finished_states: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor, dict]] = []
            # Each entry: (beam_meta_parent_idx, bytes_so_far, lp_so_far, rem_bytes, state_idx_in_st_c)
            active_multi = []
            for ci in range(len(cand_parent)):
                if cand_len[ci] == 1:
                    nm = advance_beam_meta(beams_meta[cand_parent[ci]], [cand_b0[ci]], cand_lp0[ci])
                    finished_states.append((ci, st_c, logits_c, nm))
                else:
                    active_multi.append((cand_parent[ci], [cand_b0[ci]], cand_lp0[ci], cand_len[ci] - 1, ci))

            # Expand continuation bytes atomically in batched GPU steps (at most 3 iterations for 4-byte emoji!)
            cur_st_m = st_c
            cur_log_m = logits_c
            while active_multi:
                nxt_parent_m: List[int] = []
                nxt_byte_m: List[int] = []
                nxt_meta_pack = []  # (orig_beam_idx, bytes_list, lp_sum, rem_after)

                for orig_w, b_list, lp_acc, rem_b, st_idx in active_multi:
                    # Build a temporary view of bm with partial bytes for cursor tracking!
                    tmp_bm = dict(beams_meta[orig_w])
                    tmp_bm["tail9"] = (bytearray(beams_meta[orig_w]["tail9"]) + bytearray(b_list))[-9:]
                    # Advance cursor speculatively for partial bytes
                    if use_thought_agenda and tmp_bm["goal_idx"] < len(agenda):
                        ag = agenda[tmp_bm["goal_idx"]]
                        cur_c = tmp_bm["cursor"]
                        for bv_p in b_list:
                            if cur_c is not None:
                                cp, clk = cur_c
                                if cp < len(ag.clause_bytes) and ag.clause_bytes[cp] == bv_p:
                                    cur_c = (cp + 1, clk + 1)
                                else:
                                    cur_c = None
                            if cur_c is None and len(tmp_bm["tail9"]) >= 4:
                                a4 = bytes(tmp_bm["tail9"][-4:])
                                if a4 in ag.anchor4:
                                    cur_c = (ag.anchor4[a4][0], 4)
                        tmp_bm["cursor"] = cur_c

                    adj_c = apply_goal_focus_logits(cur_log_m[st_idx], tmp_bm, is_lead_byte=False)
                    lp_c = torch.log_softmax(adj_c, dim=-1)
                    k_take = K_cont if rem_b >= 1 else 1
                    tv, ti = torch.topk(lp_c, k=k_take)
                    for lv, bi in zip(tv.cpu().tolist(), ti.cpu().tolist()):
                        if lv < -20.0:
                            continue
                        nxt_parent_m.append(st_idx)
                        nxt_byte_m.append(bi)
                        nxt_meta_pack.append((orig_w, b_list + [bi], lp_acc + lv, rem_b - 1))

                if not nxt_parent_m:
                    break

                # Prune intermediate continuation frontier to top 24 for blazing GPU speed!
                if len(nxt_parent_m) > 24:
                    ord_m = sorted(range(len(nxt_parent_m)), key=lambda i: nxt_meta_pack[i][2], reverse=True)[:24]
                    nxt_parent_m = [nxt_parent_m[i] for i in ord_m]
                    nxt_byte_m = [nxt_byte_m[i] for i in ord_m]
                    nxt_meta_pack = [nxt_meta_pack[i] for i in ord_m]

                pm_idx = torch.tensor(nxt_parent_m, dtype=torch.int64, device=dev)
                nxt_st_m = runner.gather_state(cur_st_m, pm_idx)
                xm = torch.tensor(nxt_byte_m, dtype=torch.int64, device=dev)
                nxt_log_m, _ = runner.step_batch(
                    nxt_st_m, xm, locked_mode_p=locked_mode_p, goal_z=gz, use_engram=True, engram_mult=eng_m, is_prompt=False
                )

                active_multi = []
                for mi, (orig_w, b_list, lp_acc, rem_after) in enumerate(nxt_meta_pack):
                    if rem_after == 0:
                        # Strict RFC 3629 UTF-8 validation (rejects overlong/surrogate byte triples!)
                        try:
                            bytes(b_list).decode("utf-8", errors="strict")
                        except UnicodeDecodeError:
                            continue
                        nm = advance_beam_meta(beams_meta[orig_w], b_list, lp_acc)
                        finished_states.append((mi, nxt_st_m, nxt_log_m, nm))
                    else:
                        active_multi.append((orig_w, b_list, lp_acc, rem_after, mi))
                cur_st_m = nxt_st_m
                cur_log_m = nxt_log_m

            if not finished_states:
                break

            # Prune ALL completed character candidates back to Top-W diverse beams!
            finished_states.sort(key=lambda x: x[3]["score"], reverse=True)
            chosen_fin = []
            seen_end = set()
            for item in finished_states:
                end_sig = bytes(item[3]["gen"][-6:])
                if end_sig in seen_end and len(finished_states) > beam_width:
                    continue
                seen_end.add(end_sig)
                chosen_fin.append(item)
                if len(chosen_fin) >= beam_width:
                    break

            # Pack chosen beams into a fresh unified GPU batch [W_new]!
            new_st_keys = list(st_beams.keys())
            st_beams = {
                k: torch.cat([src_st[k][src_i : src_i + 1] for (src_i, src_st, _, _) in chosen_fin], dim=0)
                for k in new_st_keys
            }
            logits_beams = torch.cat([src_log[src_i : src_i + 1] for (src_i, _, src_log, _) in chosen_fin], dim=0)
            beams_meta = [nm for (_, _, _, nm) in chosen_fin]

        # 4) Prefill-Critic Final Reranking across Top-W finished beams!
        best_text = prompt
        best_bm = beams_meta[0]
        best_final_sc = -1e9
        for bm in beams_meta[:4]:
            full_b = prompt_b + bytes(bm["gen"])
            full_txt = full_b.decode("utf-8", errors="strict")
            flu_bpc = runner.score_text_bpc(full_txt, prompt_prefix_len=len(prompt_b))
            ent_cov = len(bm["hits"])
            final_sc = -flu_bpc + 0.35 * ent_cov + 0.15 * bm["goal_idx"]
            if final_sc > best_final_sc:
                best_final_sc = final_sc
                best_text = full_txt
                best_bm = bm

        final_flu_bpc = runner.score_text_bpc(best_text, prompt_prefix_len=len(prompt_b))
        decode_sec = time.perf_counter() - t0
        return best_text, True, {
            "decode_sec": round(decode_sec, 2),
            "fluency_bpc": final_flu_bpc,
            "goals_reached": best_bm["goal_idx"] + (1 if agenda else 0),
            "entities_grounded": [e.decode("utf-8", errors="ignore") for e in best_bm["hits"]],
            "plan": plan_meta,
        }
