"""文の構文的な前置きの解析と、対話専用 DAG の扱い（v10.0）。"""

from __future__ import annotations

import gc
import math
import pickle
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neurograph.engram_v20 import DecoupledBrainV20, EngramCartridge
from neurograph.instruct_cortex_v80 import (
    InstructAdapterV80,
    InstructBrainV80,
    _build_isolated_batches,
    _mem_snapshot,
)
from neurograph.lateral_v13b import LateralV13bConfig, UnifiedBrainV13b
from neurograph.memory_v3b import _mix64
from neurograph.reason_decode_v30 import BatchedCoreRunner
from neurograph.reasoning_graph_v90 import (
    DeliberativeReasoningBrainV90,
    PrefrontalReasoningGraphV90,
    trim_utf8_safe,
)
from neurograph.role_cortex_v60 import RoleBoundThoughtCortexV60
from neurograph.sft_data_v80 import (
    TURN_A_PREFIX,
    TURN_EOT_BYTE,
    TURN_U_PREFIX,
    serialize_dialogue_with_mask,
)
from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50
from neurograph.triton_dag_kernel import run_fused_dag_step


def _shannon_entropy(counts: Dict[str, int]) -> float:
    tot = sum(counts.values())
    if tot <= 1:
        return 0.0
    h = 0.0
    inv = 1.0 / tot
    for v in counts.values():
        p = v * inv
        if p > 0.0:
            h -= p * math.log2(p)
    return h


@dataclass
class ParsedSpan:
    text: str
    kind: str
    left_b: str = ""
    right_b: str = ""
    u_idf: float = 0.0
    c_idf: float = 0.0
    spec_score: float = 0.0


class BrocaSyntacticParserV100:

    def __init__(self, seed: int = 100):
        self.seed = seed
        self.boundaries_1: set[str] = set()
        self.boundaries_2: set[str] = set()
        self.boilerplate_frames: set[str] = set()
        self.content_vocab: Dict[str, int] = {}
        self.chunk_u_idf: Dict[str, float] = {}
        self.chunk_wiki_df: Dict[str, int] = {}
        self.n_user_docs: int = 0
        self.n_wiki_docs: int = 0
        self.tau_u_idf: float = 5.0
        self.discovery_summary: Dict[str, object] = {}

    def grow_from_corpora(
        self,
        train_dialogues: List[Dict],
        prose_texts: List[str],
        mc: Optional[ContinuousThoughtCortexV50] = None,
        rc: Optional[RoleBoundThoughtCortexV60] = None,
        max_prompts: int = 55_000,
        max_vocab: int = 24_000,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        prompts = [u.strip() for d in train_dialogues[:max_prompts] for u, _ in d["turns"] if len(u.strip()) >= 2]
        n_u = len(prompts)
        self.n_user_docs = n_u
        prose_sample = prose_texts[:50_000]
        n_ps = max(1, len(prose_sample))

        b1: set[str] = set()
        if mc is not None and rc is not None:
            for mid in rc.role_motor_ids.tolist():
                s = mc.motor_bytes[int(mid)].decode("utf-8", errors="ignore")
                if len(s) == 1 and not s.isspace():
                    b1.add(s)
                    sn = unicodedata.normalize("NFKC", s)
                    if len(sn) == 1:
                        b1.add(sn)
        p_cnt: Counter[str] = Counter()
        for ln in (prompts[:30_000] + prose_sample[:20_000]):
            for ch in ln:
                if unicodedata.category(ch) in ("Po", "Ps", "Pe", "Pi", "Pf"):
                    p_cnt[ch] += 1
        for ch, c in p_cnt.most_common(40):
            if c >= 20:
                b1.add(ch)
                cn = unicodedata.normalize("NFKC", ch)
                if len(cn) == 1:
                    b1.add(cn)

        df_u_sub: Counter[str] = Counter()
        for p in prompts:
            pn = unicodedata.normalize("NFKC", p)
            L = len(pn)
            seen_sub = set()
            for k in (2, 3, 4, 5, 6):
                for i in range(0, L - k + 1):
                    sub = pn[i : i + k]
                    if any(c.isspace() for c in sub):
                        continue
                    seen_sub.add(sub)
            for s in seen_sub:
                df_u_sub[s] += 1

        b2: set[str] = set()
        for sub, cnt in df_u_sub.items():
            if len(sub) == 2 and (cnt / max(1, n_u)) >= 0.015 and (sub[0] in b1 and sub[1] in b1):
                b2.add(sub)

        df_p_sub: Counter[str] = Counter()
        for ln in prose_sample:
            ln_n = unicodedata.normalize("NFKC", ln)
            L = len(ln_n)
            seen_sub = set()
            for k in (2, 3, 4, 5, 6):
                for i in range(0, L - k + 1):
                    seen_sub.add(ln_n[i : i + k])
            for s in seen_sub:
                df_p_sub[s] += 1

        boiler: set[str] = set()
        for sub, u_cnt in df_u_sub.items():
            u_rate = u_cnt / max(1, n_u)
            p_rate = (df_p_sub.get(sub, 0) + 0.5) / n_ps
            if u_rate >= 0.0025 and (u_rate / max(p_rate, 1e-6)) >= 2.5:
                boiler.add(sub)
            elif u_rate >= 0.025:
                boiler.add(sub)

        self.boundaries_1 = b1
        self.boundaries_2 = b2
        self.boilerplate_frames = boiler

        span_u_df: Counter[str] = Counter()
        span_p_df: Counter[str] = Counter()
        for p in prompts:
            seen = set(self._raw_split_content_spans(p))
            for sp in seen:
                span_u_df[sp] += 1
        for ln in prose_sample:
            seen = set(self._raw_split_content_spans(ln))
            for sp in seen:
                span_p_df[sp] += 1

        self.n_wiki_docs = n_ps
        comb = Counter(span_u_df)
        comb.update(span_p_df)
        self.content_vocab = {}
        self.chunk_u_idf = {}
        self.chunk_wiki_df = {}
        for idx, (sp, _) in enumerate(comb.most_common(max_vocab)):
            self.content_vocab[sp] = idx
            self.chunk_u_idf[sp] = float(math.log((n_u + 10.0) / (span_u_df.get(sp, 0) + 1.0)))
            self.chunk_wiki_df[sp] = int(span_p_df.get(sp, 0))

        obs_idfs = [self.chunk_u_idf[sp] for sp, c in span_u_df.items() if c >= 2 and sp in self.chunk_u_idf]
        self.tau_u_idf = float(np.percentile(obs_idfs, 25)) if obs_idfs else 5.0

        self.discovery_summary = {
            "n_prompts": n_u,
            "n_prose_lines": n_ps,
            "n_boundaries_1": len(b1),
            "n_boundaries_2": len(b2),
            "n_boilerplate_frames": len(boiler),
            "content_vocab_size": len(self.content_vocab),
            "tau_u_idf": round(self.tau_u_idf, 3),
            "discovered_boundaries_1": sorted(b1)[:30],
            "sample_boilerplate": [s for s, _ in sorted(((s, df_u_sub[s]) for s in boiler), key=lambda x: x[1], reverse=True)[:20]],
            "sec": round(time.perf_counter() - t0, 2),
        }
        return self.discovery_summary

    def _is_boundary_1(self, ch: str) -> bool:
        if ch in self.boundaries_1:
            return True
        cn = unicodedata.normalize("NFKC", ch)
        return len(cn) == 1 and cn in self.boundaries_1

    def _raw_split_content_spans(self, text: str) -> List[str]:
        out: List[str] = []
        buf: List[str] = []
        i = 0
        L = len(text)
        b2 = self.boundaries_2
        while i < L:
            ch = text[i]
            if ch.isspace():
                if buf:
                    out.append("".join(buf)); buf = []
                i += 1
                continue
            if i + 1 < L and unicodedata.normalize("NFKC", text[i : i + 2]) in b2:
                if buf:
                    out.append("".join(buf)); buf = []
                i += 2
                continue
            if self._is_boundary_1(ch):
                if buf:
                    out.append("".join(buf)); buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        if buf:
            out.append("".join(buf))
        return out

    def _strip_boilerplate_edges(self, span: str) -> str:
        s = span.strip()
        if not s:
            return ""
        sn = unicodedata.normalize("NFKC", s)
        if sn in self.boilerplate_frames:
            return ""
        changed = True
        while changed and len(s) >= 2:
            changed = False
            sn = unicodedata.normalize("NFKC", s)
            for k in (6, 5, 4, 3, 2):
                if len(sn) > k and sn[-k:] in self.boilerplate_frames:
                    s = s[:-k].strip()
                    changed = True
                    break
                if len(sn) > k and sn[:k] in self.boilerplate_frames:
                    s = s[k:].strip()
                    changed = True
                    break
        sn = unicodedata.normalize("NFKC", s)
        if not s or sn in self.boilerplate_frames or self._is_boundary_1(s) or sn in self.boundaries_2:
            return ""
        return s

    def parse_prompt(
        self,
        user_prompt: str,
        mc: Optional[ContinuousThoughtCortexV50] = None,
        rc: Optional[RoleBoundThoughtCortexV60] = None,
    ) -> Dict[str, object]:
        text = user_prompt.strip()
        L = len(text)
        b2 = self.boundaries_2

        raw_tokens: List[Tuple[str, str]] = []
        buf: List[str] = []
        i = 0
        while i < L:
            ch = text[i]
            if ch.isspace():
                if buf:
                    raw_tokens.append(("".join(buf), "RAW")); buf = []
                i += 1
                continue
            if i + 1 < L and unicodedata.normalize("NFKC", text[i : i + 2]) in b2:
                if buf:
                    raw_tokens.append(("".join(buf), "RAW")); buf = []
                raw_tokens.append((text[i : i + 2], "BOUNDARY"))
                i += 2
                continue
            if self._is_boundary_1(ch):
                if buf:
                    raw_tokens.append(("".join(buf), "RAW")); buf = []
                raw_tokens.append((ch, "BOUNDARY"))
                i += 1
                continue
            buf.append(ch)
            i += 1
        if buf:
            raw_tokens.append(("".join(buf), "RAW"))

        parsed: List[ParsedSpan] = []
        entities_scored: List[Tuple[float, bytes]] = []
        max_u_idf = math.log(self.n_user_docs + 10.0) if self.n_user_docs > 0 else 10.9

        for idx_t, (tok, kind) in enumerate(raw_tokens):
            lb = raw_tokens[idx_t - 1][0] if (idx_t > 0 and raw_tokens[idx_t - 1][1] == "BOUNDARY") else ""
            rb = raw_tokens[idx_t + 1][0] if (idx_t + 1 < len(raw_tokens) and raw_tokens[idx_t + 1][1] == "BOUNDARY") else ""
            if kind == "BOUNDARY":
                parsed.append(ParsedSpan(text=tok, kind="BOUNDARY"))
                continue

            cleaned = self._strip_boilerplate_edges(tok)
            if not cleaned:
                parsed.append(ParsedSpan(text=tok, kind="BOILERPLATE", left_b=lb, right_b=rb))
                continue

            cb = cleaned.encode("utf-8")
            if len(cleaned) == 1 and unicodedata.category(cleaned)[0] in ("P", "S"):
                parsed.append(ParsedSpan(text=cleaned, kind="BOUNDARY"))
                continue
            while cleaned and unicodedata.category(cleaned[0])[0] in ("P", "S"):
                cleaned = cleaned[1:]
            while cleaned and unicodedata.category(cleaned[-1])[0] in ("P", "S"):
                cleaned = cleaned[:-1]
            cleaned = self._strip_boilerplate_edges(cleaned)
            if not cleaned:
                parsed.append(ParsedSpan(text=tok, kind="BOILERPLATE", left_b=lb, right_b=rb))
                continue

            cb = cleaned.encode("utf-8")
            u_idf = self.chunk_u_idf.get(cleaned, max_u_idf)
            w_df = self.chunk_wiki_df.get(cleaned, 0)

            c_idf = 10.0
            max_cg = 0.7
            if mc is not None and rc is not None:
                ms = mc.encode(np.frombuffer(cb, dtype=np.uint8)).tolist()
                if ms:
                    c_idf = float(np.mean([float(mc.motor_idfs[m]) for m in ms]))
                    max_cg = max((float(rc.cg_vec[m]) for m in ms if m >= 256), default=(0.8 if any(m < 256 for m in ms) else 0.0))

            is_oov_proper = (u_idf >= max_u_idf - 1.0 and len(cb) >= 6)
            is_enc_entity = (
                len(cb) >= 6
                and max_cg >= 0.45
                and (
                    (u_idf >= 6.8 and w_df >= 1)
                    or (u_idf >= 3.2 and w_df >= 12 and c_idf >= 8.0)
                )
            )
            is_short_enc = (len(cb) == 3 and max_cg >= 0.55 and u_idf >= 6.5 and w_df >= 8 and (lb != "" or rb != ""))

            if is_oov_proper or is_enc_entity or is_short_enc:
                spec = (c_idf * (0.5 + 0.5 * max_cg) + 0.6 * u_idf) * math.sqrt(len(cb))
                if is_oov_proper:
                    spec *= 1.35
                parsed.append(ParsedSpan(text=cleaned, kind="ENTITY", left_b=lb, right_b=rb, u_idf=round(u_idf, 2), c_idf=round(c_idf, 2), spec_score=round(spec, 2)))
                entities_scored.append((spec, cb))
            else:
                parsed.append(ParsedSpan(text=cleaned, kind="CONTENT", left_b=lb, right_b=rb, u_idf=round(u_idf, 2), c_idf=round(c_idf, 2)))

        entities_scored.sort(key=lambda x: x[0], reverse=True)
        ent_bytes: List[bytes] = []
        for _, eb in entities_scored:
            if eb not in ent_bytes:
                ent_bytes.append(eb)

        ent_mids: List[int] = []
        if mc is not None:
            for eb in ent_bytes:
                for m in mc.encode(np.frombuffer(eb, dtype=np.uint8)).tolist():
                    if m >= 256 and m not in ent_mids:
                        ent_mids.append(m)

        if len(ent_bytes) == 0:
            intent_mode = "SOCIAL"
        elif len(ent_bytes) == 1:
            intent_mode = "FACT_SINGLE"
        else:
            intent_mode = "FACT_MULTI"

        return {
            "spans": parsed,
            "entities": ent_bytes,
            "entity_strs": [e.decode("utf-8", errors="ignore") for e in ent_bytes],
            "entity_mids": ent_mids,
            "intent_mode": intent_mode,
            "want_hip": len(ent_bytes) > 0,
        }

    def save(self, folder: str) -> float:
        t0 = time.perf_counter()
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "broca_parser_v100.pkl", "wb") as f:
            pickle.dump(
                {
                    "version": "v100_parser",
                    "seed": self.seed,
                    "boundaries_1": self.boundaries_1,
                    "boundaries_2": self.boundaries_2,
                    "boilerplate_frames": self.boilerplate_frames,
                    "content_vocab": self.content_vocab,
                    "chunk_u_idf": self.chunk_u_idf,
                    "chunk_wiki_df": self.chunk_wiki_df,
                    "n_user_docs": self.n_user_docs,
                    "n_wiki_docs": self.n_wiki_docs,
                    "tau_u_idf": self.tau_u_idf,
                    "discovery_summary": self.discovery_summary,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return round(time.perf_counter() - t0, 3)

    @classmethod
    def load(cls, folder: str) -> "BrocaSyntacticParserV100":
        with open(Path(folder) / "broca_parser_v100.pkl", "rb") as f:
            d = pickle.load(f)
        obj = cls(seed=d["seed"])
        obj.boundaries_1 = d["boundaries_1"]
        obj.boundaries_2 = d["boundaries_2"]
        obj.boilerplate_frames = d["boilerplate_frames"]
        obj.content_vocab = d["content_vocab"]
        obj.chunk_u_idf = d["chunk_u_idf"]
        obj.chunk_wiki_df = d["chunk_wiki_df"]
        obj.n_user_docs = d["n_user_docs"]
        obj.n_wiki_docs = d["n_wiki_docs"]
        obj.tau_u_idf = d["tau_u_idf"]
        obj.discovery_summary = d.get("discovery_summary", {})
        return obj


def build_sft_growth_stream(train_dialogues: List[Dict], seed: int = 100) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_dialogues))
    sep = bytes(64)
    parts: List[bytes] = []
    tot_chars = 0
    for k, i in enumerate(idx):
        raw, _ = serialize_dialogue_with_mask(train_dialogues[int(i)]["turns"])
        if k > 0:
            parts.append(sep)
        parts.append(raw)
        tot_chars += sum(len(u) + len(a) for u, a in train_dialogues[int(i)]["turns"])
    blob = b"".join(parts)
    arr = np.frombuffer(blob, dtype=np.uint8).copy()
    bpc_ratio = len(arr) / max(1, tot_chars)
    return arr, bpc_ratio


@torch.no_grad()
def grow_broca_layer1_dag(
    train_dialogues: List[Dict],
    device: torch.device,
    save_ckpt: str = "broca_v100/broca_l1_core.pt",
    pool_capacity: int = 600_000,
    edge_buckets: int = 1_048_576,
    rounds: int = 3,
    batch_streams: int = 1024,
) -> Tuple[UnifiedBrainV13b, Dict[str, object]]:
    t0 = time.perf_counter()
    stream, bpc_ratio = build_sft_growth_stream(train_dialogues, seed=100)
    torch.manual_seed(100)

    cfg = LateralV13bConfig(
        pool_capacity=pool_capacity,
        edge_buckets=edge_buckets,
        feat_dim=256,
        num_slots=16,
        max_hops=16,
        max_births_per_step=256,
        lr_feat=0.22,
        lr_basis=0.035,
        lr_sharp=0.28,
        lr_slot_read=0.12,
        lr_slot_mod=0.07,
        lr_slot_gate=0.045,
        enable_lateral=False,
    )
    broca = UnifiedBrainV13b(cfg, device)
    for bi in range(broca.ltm.num_banks):
        broca.ltm.sig[bi] = torch.empty((1, 4), dtype=torch.int32, device=device)
        broca.ltm.dst[bi] = torch.full((1, 4), -1, dtype=torch.int16, device=device)
        broca.ltm.w[bi] = torch.zeros((1, 4), dtype=torch.float16, device=device)
    torch.cuda.empty_cache()

    broca.init_sensory_basis(stream, sample_len=min(6_000_000, len(stream)))

    seg_len = len(stream) // rounds
    round_logs = []
    for r in range(rounds):
        rt = time.perf_counter()
        seg = stream[r * seg_len : (r + 1) * seg_len]
        lr_s = 1.0 / (1.0 + 0.20 * r)
        tr = broca.run_stream(
            seg, train=True, batch_streams=batch_streams, lr_scale=lr_s,
            bytes_per_char=bpc_ratio, use_slots=True, use_ltm=False, use_stm=False,
            use_executive=False, populate_ltm=False,
        )
        ap = broca.apoptosis()
        fu = broca.fuse_similar_neurons(seed=900 + r * 29, chunk_size=16_384)
        gc.collect()
        torch.cuda.empty_cache()
        round_logs.append({
            "round": r + 1,
            "sec": round(time.perf_counter() - rt, 1),
            "births": int(tr["births"]),
            "deaths": int(ap["died"]),
            "merged": int(fu["merged"]),
            "alive_hidden": int(fu["alive_hidden"]),
            "edges": int(fu["edges"]),
        })

    exec_fit = broca.fit_executive_modes(stream[: min(6_000_000, len(stream))], passage_len=320, max_passages=10_000)
    Path(save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    broca.save_full_checkpoint(save_ckpt)
    del stream
    gc.collect()
    torch.cuda.empty_cache()

    return broca, {
        "pool_capacity": pool_capacity,
        "alive_hidden": int(broca.num_hidden_alive),
        "edges": int(broca.edges.count()),
        "exec_modes": int(exec_fit["num_modes"]),
        "rounds": round_logs,
        "total_sec": round(time.perf_counter() - t0, 1),
        "mem": _mem_snapshot(device),
    }


class DualCortexAdapterV100:

    def __init__(self, core_dim: int = 256):
        self.D = core_dim
        self.W_joint: Optional[np.ndarray] = None
        self.b_joint: Optional[np.ndarray] = None
        self.alpha_broca: float = 0.25
        self.readout_scale: float = 1.6
        self._W_gpu: Optional[torch.Tensor] = None
        self._b_gpu: Optional[torch.Tensor] = None

    def _ensure_gpu(self, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.W_joint is None or self.b_joint is None:
            return None, None
        if self._W_gpu is None or self._W_gpu.device != device:
            self._W_gpu = torch.from_numpy(self.W_joint).to(device=device, dtype=torch.float32)
            self._b_gpu = torch.from_numpy(self.b_joint).to(device=device, dtype=torch.float32)
        return self._W_gpu, self._b_gpu

    @torch.no_grad()
    def train_joint_readout(
        self,
        runner_base: BatchedCoreRunner,
        runner_broca: BatchedCoreRunner,
        train_dialogues: List[Dict],
        alpha_broca: float = 0.25,
        max_dialogues: int = 12_000,
        batch_size: int = 64,
        ridge_alpha: float = 0.25,
        eot_weight: float = 3.0,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        dev = runner_base.device
        V, D = 256, self.D
        D2 = 2 * D
        self.alpha_broca = float(alpha_broca)

        use_diags = train_dialogues[:max_dialogues]
        batches = _build_isolated_batches(use_diags, batch_size=batch_size)

        C_hh = torch.zeros((D2 + 1, D2 + 1), dtype=torch.float64, device=dev)
        C_hr = torch.zeros((D2 + 1, V), dtype=torch.float64, device=dev)
        sum_p_comb = torch.zeros((V,), dtype=torch.float64, device=dev)
        cnt_y_asst = torch.zeros((V,), dtype=torch.float64, device=dev)
        n_asst = 0
        n_eot = 0

        for data_np, mask_np, vlen_np, _ in batches:
            data = torch.from_numpy(data_np).to(dev)
            mask_2d = torch.from_numpy(mask_np).to(dev)
            vlen_t = torch.from_numpy(vlen_np).to(dev)
            B, T = data.shape
            st_b = runner_base.init_batch_state(B)
            st_r = runner_broca.init_batch_state(B)
            ar = torch.arange(B, device=dev)
            ones_col = torch.ones((B, 1), dtype=torch.float32, device=dev)

            for t in range(T - 1):
                x_t = data[:, t]
                y_t = data[:, t + 1]
                valid_step = (t + 1) < vlen_t
                m_t = mask_2d[:, t] & valid_step
                any_asst = bool(m_t.any().item())
                lg_b, _ = runner_base.step_batch(st_b, x_t, next_true=None, use_engram=False, is_prompt=(not any_asst))
                lg_r, _ = runner_broca.step_batch(st_r, x_t, next_true=None, use_engram=False, is_prompt=(not any_asst))
                if not any_asst:
                    continue
                h_b = st_b["last_h_act"]
                h_r = st_r["last_h_act"]
                logits = lg_b + self.alpha_broca * lg_r

                mf = m_t.to(torch.float32).unsqueeze(1)
                is_eot_y = (y_t == TURN_EOT_BYTE).unsqueeze(1)
                w_eot = torch.where(is_eot_y, torch.full((B, 1), eot_weight, device=dev), torch.ones((B, 1), device=dev))
                wt = mf * w_eot

                probs = torch.softmax(logits, dim=-1)
                sum_p_comb += (probs * mf).sum(dim=0).to(torch.float64)
                cnt_y_asst.scatter_add_(0, y_t, m_t.to(torch.float64))
                resid = -probs
                resid[ar, y_t] += 1.0

                h_aug = torch.cat([h_b, h_r, ones_col], dim=1)
                h_w = (h_aug * wt).to(torch.float64)
                C_hh += h_w.T @ h_aug.to(torch.float64)
                C_hr += h_w.T @ resid.to(torch.float64)
                n_asst += int(m_t.sum().item())
                n_eot += int((m_t & (y_t == TURN_EOT_BYTE)).sum().item())

        b_prior = torch.log((cnt_y_asst + 0.5) / (sum_p_comb + 1e-6)).clamp(-4.0, 18.0)
        b_prior = torch.where(cnt_y_asst > 0.5, b_prior, torch.zeros_like(b_prior))

        p_emp = (cnt_y_asst / max(1.0, float(n_asst))).clamp(min=0.0)
        inv_std = (1.0 / torch.sqrt(p_emp + 0.002)).clamp(max=18.0)
        lam = ridge_alpha * float((torch.trace(C_hh[:D2, :D2]) / D2).clamp(min=1.0).item())
        reg = torch.eye(D2 + 1, dtype=torch.float64, device=dev) * lam
        reg[D2, D2] = lam * 0.25
        W_aug = torch.linalg.solve(C_hh + reg, C_hr)
        W_scaled = (W_aug[:D2, :] * inv_std.unsqueeze(0)).to(torch.float32).clamp(-3.5, 3.5)

        self.W_joint = np.ascontiguousarray(W_scaled.cpu().numpy(), dtype=np.float32)
        self.b_joint = np.ascontiguousarray(b_prior.to(torch.float32).cpu().numpy(), dtype=np.float32)
        self._W_gpu = None
        self._b_gpu = None

        w_base_norm = float(np.linalg.norm(self.W_joint[:D]))
        w_broca_norm = float(np.linalg.norm(self.W_joint[D:]))
        del C_hh, C_hr, W_aug, W_scaled, b_prior, sum_p_comb, cnt_y_asst
        torch.cuda.empty_cache()

        return {
            "dialogues_fit": len(use_diags),
            "assistant_bytes_fit": n_asst,
            "eot_events_fit": n_eot,
            "alpha_broca": self.alpha_broca,
            "W_base_half_norm": round(w_base_norm, 3),
            "W_broca_half_norm": round(w_broca_norm, 3),
            "eot_bias": round(float(self.b_joint[TURN_EOT_BYTE]), 4),
            "sec": round(time.perf_counter() - t0, 2),
            "mem": _mem_snapshot(dev),
        }

    def save(self, folder: str) -> float:
        t0 = time.perf_counter()
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "dual_cortex_adapter_v100.pkl", "wb") as f:
            pickle.dump(
                {
                    "D": self.D,
                    "W_joint": self.W_joint,
                    "b_joint": self.b_joint,
                    "alpha_broca": self.alpha_broca,
                    "readout_scale": self.readout_scale,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return round(time.perf_counter() - t0, 3)

    @classmethod
    def load(cls, folder: str) -> "DualCortexAdapterV100":
        with open(Path(folder) / "dual_cortex_adapter_v100.pkl", "rb") as f:
            d = pickle.load(f)
        obj = cls(core_dim=d["D"])
        obj.W_joint = d["W_joint"]
        obj.b_joint = d["b_joint"]
        obj.alpha_broca = float(d["alpha_broca"])
        obj.readout_scale = float(d["readout_scale"])
        return obj


class DualBrocaReasoningBrainV100:

    def __init__(
        self,
        rbrain_v90: DeliberativeReasoningBrainV90,
        parser_v100: BrocaSyntacticParserV100,
        broca_v20: DecoupledBrainV20,
        dual_adapter: DualCortexAdapterV100,
        temp_cal_broca: float = 1.08,
    ):
        self.rbrain = rbrain_v90
        self.chat = rbrain_v90.chat
        self.v20 = rbrain_v90.v20
        self.mc = rbrain_v90.mc
        self.rc = rbrain_v90.rc
        self.hip = rbrain_v90.hip
        self.runner_base = rbrain_v90.runner
        self.device = rbrain_v90.device
        self.adapter_v80 = rbrain_v90.adapter
        self.rgraph = rbrain_v90.rgraph

        self.parser = parser_v100
        self.broca_v20 = broca_v20
        self.runner_broca = BatchedCoreRunner(broca_v20, temp_cal=temp_cal_broca)
        self.dual_adapter = dual_adapter

    @staticmethod
    def _write_stm_post(runner: BatchedCoreRunner, st: Dict[str, torch.Tensor], x_t: torch.Tensor, next_true: torch.Tensor, logits: torch.Tensor) -> None:
        dev = runner.device
        cfg = runner.core.cfg
        B_cur = x_t.shape[0]
        ar = torch.arange(B_cur, device=dev)
        prev_idx = (st["hist_ptr"] - 1) % runner.H_buf
        bs7 = st["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - runner._ar24[:, :7]) % runner.H_buf]
        h4 = _mix64(bs7[:, 0] + 17)
        h4 = _mix64(h4 ^ ((bs7[:, 1] + 31) * 1315423911))
        h4 = _mix64(h4 ^ ((bs7[:, 2] + 53) * 2654435761))
        h4 = _mix64(h4 ^ ((bs7[:, 3] + 79) * 1000003))
        k4 = torch.where((bs7[:, 3] >= 0) & (h4 != 0), h4, torch.zeros_like(h4))
        h7 = _mix64(h4 ^ ((bs7[:, 4] + 97) * 1000033))
        h7 = _mix64(h7 ^ ((bs7[:, 5] + 113) * 1000037))
        h7 = _mix64(h7 ^ ((bs7[:, 6] + 131) * 1000039))
        k7 = torch.where((bs7[:, 6] >= 0) & (h7 != 0), h7, torch.zeros_like(h7))

        valid_nt = (next_true >= 0) & (st["step_i"] >= 5) & (k4 != 0)
        log_p = torch.log_softmax(logits, dim=-1)
        surp = 1.0 - torch.exp(log_p[ar, next_true.clamp(min=0)])
        do_b = valid_nt & (surp > cfg.stm_surp_thresh)
        sp = st["stm_ptr"]
        st["stm_k4"][ar, sp] = torch.where(do_b, k4, st["stm_k4"][ar, sp])
        st["stm_k7"][ar, sp] = torch.where(do_b, k7, st["stm_k7"][ar, sp])
        st["stm_val"][ar, sp] = torch.where(do_b, next_true.clamp(min=0), st["stm_val"][ar, sp])
        st["stm_str"][ar, sp] = torch.where(do_b, torch.ones_like(surp), st["stm_str"][ar, sp])
        st["stm_ptr"] = torch.where(do_b, (sp + 1) % cfg.stm_slots, sp)

    @torch.no_grad()
    def _step_dual(
        self,
        st_b: Dict[str, torch.Tensor],
        st_r: Optional[Dict[str, torch.Tensor]],
        x_t: torch.Tensor,
        next_true: Optional[torch.Tensor],
        mode: str = "dual",
        use_base_engram: bool = True,
        use_sft_engram: bool = True,
        base_engram_mult: float = 0.72,
        sft_engram_scale: float = 0.85,
        readout_scale_override: Optional[float] = None,
        alpha_broca_override: Optional[float] = None,
        goal_z: Optional[torch.Tensor] = None,
        goal_z_beta: float = 0.0,
        wm_table: Optional[Dict[bytes, Dict[int, float]]] = None,
        wm_scale: float = 0.0,
        wm_cap: float = 2.6,
        is_prompt: bool = False,
    ) -> torch.Tensor:
        dev = self.device
        if mode == "broca_only":
            assert st_r is not None
            lg_r, _ = self.runner_broca.step_batch(st_r, x_t, next_true=None, use_engram=False, is_prompt=is_prompt)
            if next_true is not None:
                self._write_stm_post(self.runner_broca, st_r, x_t, next_true, lg_r)
            return lg_r

        lg_b, _ = self.runner_base.step_batch(
            st_b, x_t, next_true=None, locked_mode_p=None,
            goal_z=(goal_z if (not is_prompt and goal_z_beta > 1e-4) else None),
            goal_z_beta=(goal_z_beta if not is_prompt else 0.0),
            use_engram=False, is_prompt=is_prompt,
        )
        h_b = st_b["last_h_act"]

        if mode == "dual":
            assert st_r is not None
            lg_r, _ = self.runner_broca.step_batch(st_r, x_t, next_true=None, use_engram=False, is_prompt=is_prompt)
            h_r = st_r["last_h_act"]
            alpha_b = self.dual_adapter.alpha_broca if alpha_broca_override is None else alpha_broca_override
            w_b = max(0.0, min(1.0, float(alpha_b)))
            logits = (1.0 - w_b) * lg_b + w_b * lg_r
            if not is_prompt:
                W_g, b_g = self.dual_adapter._ensure_gpu(dev)
                if W_g is not None:
                    rs = self.dual_adapter.readout_scale if readout_scale_override is None else readout_scale_override
                    h_cat = torch.cat([h_b, h_r], dim=1)
                    logits = logits + (h_cat @ W_g) * rs + b_g.unsqueeze(0) * (1.0 - w_b)
        else:
            logits = lg_b
            if not is_prompt:
                W_g, b_g = self.adapter_v80._ensure_gpu_adapter(dev)
                if W_g is not None:
                    rs = self.adapter_v80.readout_scale if readout_scale_override is None else readout_scale_override
                    logits = logits + (h_b @ W_g) * rs + b_g.unsqueeze(0)

        want_base = (not is_prompt) and use_base_engram and (len(self.v20.rack.cartridges) > 0)
        want_sft = (not is_prompt) and use_sft_engram and (self.adapter_v80.sft_engram is not None)
        want_wm = (not is_prompt) and bool(wm_table) and wm_scale > 0.0
        if want_base or want_sft or want_wm:
            B_cur = x_t.shape[0]
            ar = torch.arange(B_cur, device=dev)
            prev_idx = (st_b["hist_ptr"] - 1) % self.runner_base.H_buf
            bs24 = st_b["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - self.runner_base._ar24) % self.runner_base.H_buf]
            bs24_np = bs24.to(torch.int32).cpu().numpy()
            keys_np = self.v20.rack.hash_bytes24_batch(bs24_np)
            cn = st_b["c_eng"] / st_b["c_eng"].norm(dim=1, keepdim=True).clamp(min=1e-5)
            cq_np = torch.round(cn * 127.0).clamp(-127, 127).to(torch.int8).cpu().numpy()

            lp_c = torch.log_softmax(logits, dim=-1)
            p_c = torch.exp(lp_c)
            H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
            ent_g = H_c / (1.0 + H_c)

            if want_base:
                eb_np, _, _ = self.v20.rack.lookup_batch(keys_np, cq_np, use_context_gate=True)
                logits = logits + torch.from_numpy(eb_np).to(dev) * (base_engram_mult * ent_g)
            if want_sft:
                s_np = np.zeros((B_cur, 256), dtype=np.float32)
                self.adapter_v80.sft_engram.lookup_into(keys_np, cq_np, s_np, use_context_gate=True)
                np.minimum(s_np, 2.8, out=s_np)
                logits = logits + torch.from_numpy(s_np).to(dev) * (sft_engram_scale * ent_g)
            if want_wm and wm_table is not None:
                wm_np = np.zeros((B_cur, 256), dtype=np.float32)
                has_hit = False
                for ri_b in range(B_cur):
                    row_raw = bs24_np[ri_b, :10]
                    vn = int(np.sum(row_raw >= 0))
                    if vn >= 3:
                        chr_b = bytes(int(row_raw[k]) for k in range(vn - 1, -1, -1))
                        for wl in (3, 4, 5, 6, 8, 10):
                            if len(chr_b) >= wl:
                                d_hit = wm_table.get(chr_b[-wl:])
                                if d_hit:
                                    has_hit = True
                                    for nb_k, w_k in d_hit.items():
                                        wm_np[ri_b, nb_k] += w_k
                if has_hit:
                    np.clip(wm_np, 0.0, wm_cap, out=wm_np)
                    logits = logits + torch.from_numpy(wm_np).to(dev) * (wm_scale * ent_g)

        if next_true is not None:
            self._write_stm_post(self.runner_base, st_b, x_t, next_true, logits)
            if mode == "dual" and st_r is not None:
                self._write_stm_post(self.runner_broca, st_r, x_t, next_true, logits)
        return logits

    @torch.no_grad()
    def verify_eval_gen_parity(self, user_prompt: str = "日本の首都はどこですか？") -> float:
        dev = self.device
        prefix = list(TURN_U_PREFIX + user_prompt.encode("utf-8") + TURN_A_PREFIX)
        st_b1 = self.runner_base.init_batch_state(1)
        st_r1 = self.runner_broca.init_batch_state(1)
        st_b2 = self.runner_base.init_batch_state(1)
        st_r2 = self.runner_broca.init_batch_state(1)
        lg1 = lg2 = None
        for i, b in enumerate(prefix):
            x_t = torch.tensor([b], dtype=torch.int64, device=dev)
            is_last = (i == len(prefix) - 1)
            nt = torch.tensor([prefix[i + 1]], dtype=torch.int64, device=dev) if not is_last else None
            lg1 = self._step_dual(st_b1, st_r1, x_t, nt, mode="dual", is_prompt=(not is_last))
            lg2 = self._step_dual(st_b2, st_r2, x_t, nt, mode="dual", is_prompt=(not is_last))
        diff = float((lg1 - lg2).abs().max().item())
        assert diff < 1e-4, f"Dual-cortex eval/gen parity failed: {diff}"
        return diff

    @torch.no_grad()
    def eval_sft_causal_bpc(
        self,
        val_dialogues: List[Dict],
        mode: str = "dual",
        use_base_engram: bool = True,
        use_sft_engram: bool = True,
        base_engram_mult: float = 0.72,
        sft_engram_scale: float = 0.85,
        readout_scale_override: Optional[float] = None,
        alpha_broca_override: Optional[float] = None,
        batch_size: int = 48,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        dev = self.device
        batches = _build_isolated_batches(val_dialogues, batch_size=batch_size)

        body_nll_t = torch.zeros((), dtype=torch.float64, device=dev)
        eot_nll_t = torch.zeros((), dtype=torch.float64, device=dev)
        body_top1_t = torch.zeros((), dtype=torch.int64, device=dev)
        body_cnt_t = torch.zeros((), dtype=torch.int64, device=dev)
        eot_cnt_t = torch.zeros((), dtype=torch.int64, device=dev)
        total_body_chars = 0

        for data_np, mask_np, vlen_np, bchars in batches:
            total_body_chars += bchars
            data = torch.from_numpy(data_np).to(dev)
            mask_2d = torch.from_numpy(mask_np).to(dev)
            vlen_t = torch.from_numpy(vlen_np).to(dev)
            B, T = data.shape
            st_b = self.runner_base.init_batch_state(B)
            st_r = self.runner_broca.init_batch_state(B) if mode in ("dual", "broca_only") else None
            ar = torch.arange(B, device=dev)

            for t in range(T - 1):
                x_t = data[:, t]
                y_t = data[:, t + 1]
                valid_step = (t + 1) < vlen_t
                m_t = mask_2d[:, t] & valid_step
                nt = torch.where((~m_t) & valid_step, y_t, torch.full_like(y_t, -1))
                any_asst = bool(m_t.any().item())
                logits = self._step_dual(
                    st_b, st_r, x_t, next_true=nt, mode=mode,
                    use_base_engram=use_base_engram, use_sft_engram=use_sft_engram,
                    base_engram_mult=base_engram_mult, sft_engram_scale=sft_engram_scale,
                    readout_scale_override=readout_scale_override,
                    alpha_broca_override=alpha_broca_override,
                    is_prompt=(not any_asst),
                )
                if any_asst:
                    log_p = torch.log_softmax(logits, dim=-1)
                    nll = -log_p[ar, y_t]
                    is_eot = m_t & (y_t == TURN_EOT_BYTE)
                    is_body = m_t & (y_t != TURN_EOT_BYTE)
                    body_nll_t += torch.where(is_body, nll, torch.zeros_like(nll)).sum().to(torch.float64)
                    eot_nll_t += torch.where(is_eot, nll, torch.zeros_like(nll)).sum().to(torch.float64)
                    body_top1_t += ((logits.argmax(dim=-1) == y_t) & is_body).sum()
                    body_cnt_t += is_body.sum()
                    eot_cnt_t += is_eot.sum()

        body_bytes = max(1, int(body_cnt_t.item()))
        eot_cnt = max(1, int(eot_cnt_t.item()))
        body_bits = float(body_nll_t.item()) / math.log(2.0)
        eot_bits_tot = float(eot_nll_t.item()) / math.log(2.0)
        return {
            "body_bpc": round(body_bits / max(1, total_body_chars), 4),
            "body_bpb": round(body_bits / body_bytes, 4),
            "body_top1": round(int(body_top1_t.item()) / body_bytes, 4),
            "eot_bits": round(eot_bits_tot / eot_cnt, 3),
            "body_bytes": body_bytes,
            "eot_count": eot_cnt,
            "sec": round(time.perf_counter() - t0, 2),
        }

    @torch.no_grad()
    def respond_v100(
        self,
        user_prompt: str,
        use_broca_l2_parser: bool = True,
        use_broca_l1_dag: bool = True,
        max_motor_steps: int = 36,
        beam_width: int = 3,
        top_k_motor: int = 12,
        frontier_cap: int = 24,
        escape_k: int = 3,
        goal_z_beta: float = 0.14,
        base_engram_mult: float = 0.48,
        sft_engram_scale: float = 1.15,
        score_critic: bool = True,
    ) -> Tuple[str, bool, Dict]:
        t0 = time.perf_counter()
        dev = self.device
        mc = self.mc
        rc = self.rc
        rb = self.runner_base
        rr = self.runner_broca

        u_bytes = user_prompt.strip().encode("utf-8")
        frame_prefix = TURN_U_PREFIX + u_bytes + TURN_A_PREFIX
        u_motors = mc.encode(np.frombuffer(u_bytes, dtype=np.uint8))
        ctx_motor_list = u_motors.tolist()
        ctx_motors_arr = np.array(ctx_motor_list, dtype=np.int32)

        h_qa = self.adapter_v80.predict_qa_intent(rc, ctx_motors_arr)
        node, node_sim = self.rgraph.route(mc, rc, self.adapter_v80, u_motors, ctx_motors_arr)

        if use_broca_l2_parser:
            p_out = self.parser.parse_prompt(user_prompt, mc=mc, rc=rc)
            entity_spans: List[bytes] = p_out["entities"]
            entity_mids: List[int] = p_out["entity_mids"]
            intent_mode = str(p_out["intent_mode"])
            want_hip = bool(p_out["want_hip"])
            is_social = (intent_mode == "SOCIAL")
            top_k_ag = 2 if (len(entity_spans) >= 2 or node.multi_entity_rate >= 0.50) else 1
        else:
            entity_spans, entity_mids = self.rgraph.extract_entity_spans(mc, rc, self.adapter_v80, user_prompt)
            has_custom_mounted = any(c.name.startswith("custom") for c in self.hip.cartridges) if self.hip else False
            want_hip = bool(len(entity_spans) > 0 and (node.hip_need_rate >= 0.35 or has_custom_mounted))
            intent_mode = "LEGACY_SOCIAL" if not want_hip else "LEGACY_FACT"
            is_social = not want_hip
            top_k_ag = 2 if (node.multi_entity_rate >= 0.30 or len(entity_spans) >= 2) else 1

        delib = self.rbrain._conjunctive_hippocampus_deliberate(
            entity_spans, u_bytes, node, h_qa, top_k_agenda=top_k_ag, conjunctive=True
        ) if want_hip else {
            "agenda_bytes": [], "agenda_texts": [], "agenda_sources": [], "agenda_scores": [],
            "exact_entities": [], "partial_entities": [], "uncovered_entities": [e.decode("utf-8", errors="ignore") for e in entity_spans[:5]],
            "ep_concepts": {}, "ep_trans1": {}, "ep_trans2": {}, "ep_4grams": set(),
            "ep_top_motors": [], "h_ev_l2": np.zeros(mc.thought_dim, dtype=np.float32),
            "epi_conf": 0.0, "has_custom": False, "hop23_ms": 0.0,
        }
        epi_conf = delib["epi_conf"]
        has_custom = delib["has_custom"]
        has_evidence = len(delib["agenda_bytes"]) > 0
        ep_concepts = delib["ep_concepts"]
        ep_trans1 = delib["ep_trans1"]
        ep_trans2 = delib["ep_trans2"]
        ep_4grams = delib["ep_4grams"]
        if not has_evidence and intent_mode == "SOCIAL":
            is_social = True

        wm_table: Dict[bytes, Dict[int, float]] = {}
        if has_evidence:
            wm_w = {3: 0.45, 4: 0.75, 5: 1.05, 6: 1.35, 8: 1.75, 10: 2.10}
            for ag_i, ag_b in enumerate(delib["agenda_bytes"]):
                src_mult = 1.45 if delib["agenda_sources"][ag_i].startswith("custom") else 1.0
                for p_i in range(3, len(ag_b)):
                    nb = ag_b[p_i]
                    for wl, wv in wm_w.items():
                        if p_i >= wl:
                            d_s = wm_table.setdefault(ag_b[p_i - wl : p_i], {})
                            d_s[nb] = d_s.get(nb, 0.0) + wv * src_mult

        h_root = mc.compute_root_thought_vec(np.array(entity_mids, dtype=np.int32)) if entity_mids else mc.compute_root_thought_vec(u_motors)
        h_plan_l2 = (0.45 * h_qa + 0.55 * delib["h_ev_l2"]) if has_evidence else (0.75 * h_qa + 0.25 * node.ans_concept_proto)
        npl = float(np.linalg.norm(h_plan_l2))
        if npl > 1e-6:
            h_plan_l2 = (h_plan_l2 / npl).astype(np.float32)

        goal_z = None
        if goal_z_beta > 0.0:
            gz_l2 = self.rgraph.project_to_goal_z(h_plan_l2, dev)
            gz_ev = rb.encode_text_h(b"\n".join(delib["agenda_bytes"])) if has_evidence else None
            if gz_ev is not None and self.rgraph.y_l1_mean is not None:
                gz_ev = gz_ev - torch.from_numpy(self.rgraph.y_l1_mean).to(dev)
                gz_ev = gz_ev / gz_ev.norm().clamp(min=1e-6)
            if gz_l2 is not None and gz_ev is not None:
                goal_z = 0.45 * gz_l2 + 0.55 * gz_ev
                goal_z = goal_z / goal_z.norm().clamp(min=1e-6)
            else:
                goal_z = gz_ev if gz_ev is not None else gz_l2

        if has_evidence and float(np.linalg.norm(delib["h_ev_l2"])) > 0.1:
            alpha_e = 0.80 if has_custom else min(0.70, 0.55 * epi_conf)
            h_anchor = (1.0 - alpha_e) * h_root + alpha_e * delib["h_ev_l2"]
            na = float(np.linalg.norm(h_anchor))
            if na > 1e-6:
                h_anchor = (h_anchor / na).astype(np.float32)
        else:
            h_anchor = h_root.copy()
        has_anchor = bool(float(np.linalg.norm(h_anchor)) > 0.1)
        root_t_ids, root_t_w = mc.extract_active_topics(
            np.array(entity_mids, dtype=np.int32) if entity_mids else u_motors,
            max_topics=8, min_idf=max(5.5, mc.tau_idf - 0.3),
        )

        l1_mode = "dual" if use_broca_l1_dag else "base_v8"
        st_b1 = rb.init_batch_state(1)
        st_r1 = rr.init_batch_state(1) if use_broca_l1_dag else None
        if has_evidence:
            replay_safe = trim_utf8_safe(b"\n".join(delib["agenda_bytes"]), 420)
            replay_b = list(replay_safe)
            for ri in range(len(replay_b) - 1):
                rx = torch.tensor([replay_b[ri]], dtype=torch.int64, device=dev)
                ry = torch.tensor([replay_b[ri + 1]], dtype=torch.int64, device=dev)
                rb.step_batch(st_b1, rx, next_true=ry, use_engram=False, is_prompt=True)
                if st_r1 is not None:
                    rr.step_batch(st_r1, rx, next_true=ry, use_engram=False, is_prompt=True)
            for st_tmp in ([st_b1, st_r1] if st_r1 is not None else [st_b1]):
                st_tmp["byte_hist"].fill_(-1); st_tmp["hist_ptr"].zero_()
                st_tmp["node_delay"].fill_(-1); st_tmp["nd_ptr"].zero_()
                st_tmp["c_eng"].zero_()

        wm_scale = (1.95 if has_custom else 1.25) * epi_conf
        wm_cap = 4.6 if has_custom else 2.6

        def _step_gen(stb, str_, x_in, nt_in, is_p: bool, ans_pos: int = 0) -> torch.Tensor:
            beta_eff = (goal_z_beta * math.exp(-ans_pos / 42.0)) if (not is_p and goal_z is not None and goal_z_beta > 0.0) else 0.0
            return self._step_dual(
                stb, str_, x_in, nt_in, mode=l1_mode,
                use_base_engram=True, use_sft_engram=True,
                base_engram_mult=base_engram_mult, sft_engram_scale=sft_engram_scale,
                goal_z=goal_z, goal_z_beta=beta_eff,
                wm_table=wm_table, wm_scale=wm_scale, wm_cap=wm_cap,
                is_prompt=is_p,
            )

        raw_in = list(frame_prefix)
        logits1 = None
        for i, b in enumerate(raw_in):
            x_t = torch.tensor([b], dtype=torch.int64, device=dev)
            is_last = (i == len(raw_in) - 1)
            nt_u = torch.tensor([raw_in[i + 1]], dtype=torch.int64, device=dev) if (not is_last and is_social) else None
            logits1 = _step_gen(st_b1, st_r1, x_t, nt_u, is_p=(not is_last), ans_pos=0)

        st_b_beams = st_b1
        st_r_beams = st_r1
        logits_beams = logits1
        bytes_to_mid: Dict[bytes, int] = {b: i for i, b in enumerate(mc.motor_bytes)}
        beams = [{
            "score": 0.0, "raw_l1_lp": 0.0, "raw_l2_sc": 0.0, "tot_inh": 0.0, "tot_rep": 0.0,
            "n_bytes": 0, "motor_seq": list(ctx_motor_list), "gen_motors": [], "gen_bytes": bytearray(),
            "seen_m3": set(), "seen_m2": set(), "used_content": set(), "done": False,
        }]
        finished_beams: List[Dict] = []

        if is_social:
            min_eot_b = 6
        else:
            min_eot_b = int(max(30, min(60, node.p25_ans_bytes * 0.60)))

        for step_i in range(max_motor_steps):
            cand_parent, cand_mid, cand_l2_base, cand_l2_gated, cand_inh_b, cand_bytes, cand_cg, cand_keep_prio, cand_is_l1_esc = [], [], [], [], [], [], [], [], []
            mb_escape_jobs = []
            eot_candidates: List[Tuple[float, Dict]] = []

            for wi, bm in enumerate(beams):
                if bm["done"]:
                    continue
                if bm["n_bytes"] >= min_eot_b:
                    lp_now = torch.log_softmax(logits_beams[wi], dim=-1)
                    eot_lp = float(lp_now[TURN_EOT_BYTE].item())
                    last_gm = bm["gen_motors"][-1] if bm["gen_motors"] else -1
                    eot_prior = self.adapter_v80.eot_after_motor.get(last_gm, 0.0)
                    nb_e = bm["n_bytes"] + 1
                    nm_e = len(bm["gen_motors"]) + 1
                    eot_l2 = 1.8 * math.sqrt(eot_prior) if eot_prior > 0.0 else 0.0
                    len_fin = (0.30 * math.sqrt(eot_prior)) if (bm["n_bytes"] >= min_eot_b and eot_prior >= 0.04) else 0.0
                    eot_sc = (bm["raw_l1_lp"] + eot_lp - bm["tot_inh"]) / (nb_e ** 0.90) + 0.30 * ((bm["raw_l2_sc"] + eot_l2) / (nm_e ** 0.85)) - 0.18 * bm["tot_rep"] + len_fin
                    if eot_lp > -2.8:
                        done_bm = dict(bm)
                        done_bm["done"] = True
                        done_bm["score"] = eot_sc
                        eot_candidates.append((eot_sc, done_bm))

                gm = bm["gen_motors"]
                last1 = gm[-1] if gm else -1
                last2 = (gm[-2], gm[-1]) if len(gm) >= 2 else (-1, -1)
                recent_14 = set(gm[-14:])
                added_bytes_wi = set()
                last_b = bm["gen_bytes"][-1] if bm["gen_bytes"] else -1

                def _score_cand(
                    mid_v: int, mb: bytes, base_l2: float, extra_gated: float = 0.0,
                    _bm=bm, _rec14=recent_14, _last_b=last_b,
                ) -> Tuple[float, float, float, float]:
                    cg = float(rc.cg_vec[mid_v]) if 0 <= mid_v < mc.V_m else 0.0
                    l2_b = float(base_l2)
                    l2_g = float(extra_gated)
                    inh_b = 0.0
                    if mid_v >= 256 and mid_v in _rec14:
                        l2_b -= (0.95 + 1.20 * cg)
                    if b"\n" in mb:
                        l2_b -= (3.2 if (_last_b == 10 or _bm["n_bytes"] < 54) else 0.9)
                    ep_w = ep_concepts.get(mid_v, 0.0) if has_evidence else 0.0
                    if ep_w > 0.0:
                        l2_g += epi_conf * (2.10 if has_custom else 1.55) * math.tanh(ep_w)
                    if mid_v >= 256 and cg > 0.15:
                        em = mc.emb_np[mid_v]
                        if float(np.dot(em, em)) > 0.1:
                            if has_anchor:
                                cos_anc = float(np.dot(em, h_anchor))
                                dir_aff = mc.directed_affinity(mid_v, root_t_ids, root_t_w)
                                align = cos_anc + 0.18 * math.tanh(dir_aff) + 0.45 * epi_conf * math.tanh(ep_w)
                                if align < 0.18:
                                    inh_b = cg * 2.15 * min(1.3, (0.18 - align) / 0.18)
                                elif has_evidence and ep_w == 0.0 and cg > 0.65 and cos_anc < 0.44:
                                    inh_b = cg * (1.60 if has_custom else 0.85) * (0.44 - cos_anc)
                                l2_b += cg * 0.95 * align
                            l2_g += cg * 1.45 * float(np.dot(em, h_plan_l2))
                    return l2_b, l2_g, inh_b, cg

                if step_i == 0:
                    if not is_social and entity_spans:
                        for rk, es in enumerate(entity_spans[:3]):
                            if es not in added_bytes_wi and b"\n" not in es and b"\r" not in es:
                                mid_e = bytes_to_mid.get(es, -1)
                                l2_b, l2_g, inh_b, cg = _score_cand(mid_e, es, 0.80, extra_gated=1.95 - 0.20 * rk)
                                cand_parent.append(wi); cand_mid.append(mid_e); cand_l2_base.append(l2_b)
                                cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(es)
                                cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                                added_bytes_wi.add(es)
                    else:
                        openers = self.adapter_v80.sft_openers[:8]
                        for om, ow in openers:
                            mb = mc.motor_bytes[om]
                            if om < 256 or b"\r" in mb or b"\n" in mb or mb in added_bytes_wi:
                                continue
                            l2_b, l2_g, inh_b, cg = _score_cand(om, mb, 0.45, extra_gated=1.35 * ow)
                            cand_parent.append(wi); cand_mid.append(om); cand_l2_base.append(l2_b)
                            cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                            cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                            added_bytes_wi.add(mb)
                else:
                    sft_cands: Dict[int, float] = {}
                    if last2 in self.adapter_v80.sft_m2:
                        for nm, nw in self.adapter_v80.sft_m2[last2].items():
                            sft_cands[nm] = max(sft_cands.get(nm, 0.0), 1.55 * nw)
                    if last1 in self.adapter_v80.sft_m1:
                        for nm, nw in self.adapter_v80.sft_m1[last1].items():
                            sft_cands[nm] = max(sft_cands.get(nm, 0.0), 1.05 * nw)
                    for mid_v, sw in sorted(sft_cands.items(), key=lambda x: x[1], reverse=True)[:8]:
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 or b"\r" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.40, extra_gated=1.30 * sw)
                        cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                        cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                        cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                        added_bytes_wi.add(mb)

                if len(gm) > 0:
                    p_ids, p_sc = mc.predict_topk(
                        np.array(gm[-6:], dtype=np.int32), root_t_ids, root_t_w, topic_ppmi_gain=1.35, top_k=top_k_motor * 2,
                    )
                    taken = 0
                    for mid_v, sc_v in zip(p_ids.tolist(), p_sc.tolist()):
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 and (mid_v != 10 and (mid_v < 32 or mid_v >= 127)):
                            continue
                        if b"\r" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, sc_v * 0.72)
                        cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                        cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                        cand_cg.append(cg); cand_keep_prio.append(False); cand_is_l1_esc.append(False)
                        added_bytes_wi.add(mb)
                        taken += 1
                        if taken >= top_k_motor:
                            break

                if has_evidence:
                    ev_cands: Dict[int, float] = {}
                    if last2 in ep_trans2:
                        for nm, w in ep_trans2[last2].items():
                            ev_cands[nm] = max(ev_cands.get(nm, 0.0), 2.1 * w)
                    if last1 in ep_trans1:
                        for nm, w in ep_trans1[last1].items():
                            ev_cands[nm] = max(ev_cands.get(nm, 0.0), 1.4 * w)
                    for emid, ew in delib["ep_top_motors"][:10]:
                        if emid not in bm["used_content"] and emid not in recent_14:
                            ev_cands[emid] = max(ev_cands.get(emid, 0.0), 0.90 * ew)
                    for mid_v, ew in sorted(ev_cands.items(), key=lambda x: x[1], reverse=True)[:8]:
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 or b"\r" in mb or b"\n" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.35, extra_gated=(2.20 if has_custom else 1.40) * epi_conf * math.tanh(ew))
                        cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                        cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                        cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                        added_bytes_wi.add(mb)

                if escape_k > 0:
                    lrow = logits_beams[wi].clone()
                    lrow[~rb.lead_mask] = -1e9
                    lrow[0] = -1e9; lrow[TURN_EOT_BYTE] = -1e9; lrow[13] = -1e9
                    if step_i == 0 or last_b == 10:
                        lrow[10] = -1e9
                    for b0 in torch.topk(lrow, k=escape_k).indices.cpu().tolist():
                        if b0 < 0x80:
                            mb = bytes([b0])
                            if mb not in added_bytes_wi:
                                mid_v = bytes_to_mid.get(mb, -1)
                                l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.25)
                                cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                                cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                                cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(True)
                                added_bytes_wi.add(mb)
                        else:
                            clen = 2 if (b0 & 0xE0) == 0xC0 else (3 if (b0 & 0xF0) == 0xE0 else 4)
                            mb_escape_jobs.append((wi, b0, clen, added_bytes_wi, _score_cand))

            for _, dbm in eot_candidates:
                finished_beams.append(dbm)

            if mb_escape_jobs:
                J = len(mb_escape_jobs)
                j_parents = torch.tensor([x[0] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                st_b_esc = rb.gather_state(st_b_beams, j_parents)
                st_r_esc = rr.gather_state(st_r_beams, j_parents) if st_r_beams is not None else None
                cur_bs = torch.tensor([x[1] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                esc_bufs = [bytearray([x[1]]) for x in mb_escape_jobs]
                esc_clens = [x[2] for x in mb_escape_jobs]
                esc_pos = int(beams[0]["n_bytes"]) if beams else 0
                for ci_u in range(1, max(esc_clens)):
                    act_e = [ji for ji in range(J) if ci_u < esc_clens[ji]]
                    if not act_e:
                        break
                    if len(act_e) == J:
                        lg_e = _step_gen(st_b_esc, st_r_esc, cur_bs, None, is_p=False, ans_pos=esc_pos + ci_u)
                    else:
                        act_et = torch.tensor(act_e, dtype=torch.int64, device=dev)
                        stb_sub = rb.gather_state(st_b_esc, act_et)
                        str_sub = rr.gather_state(st_r_esc, act_et) if st_r_esc is not None else None
                        lg_e = _step_gen(stb_sub, str_sub, cur_bs[act_et], None, is_p=False, ans_pos=esc_pos + ci_u)
                        for k_st in st_b_esc.keys():
                            st_b_esc[k_st][act_et] = stb_sub[k_st]
                        if st_r_esc is not None and str_sub is not None:
                            for k_st in st_r_esc.keys():
                                st_r_esc[k_st][act_et] = str_sub[k_st]
                    mask_e = torch.zeros((len(act_e), 256), dtype=torch.bool, device=dev)
                    for row_i, ji in enumerate(act_e):
                        b0 = mb_escape_jobs[ji][1]
                        lo, hi = 0x80, 0xBF
                        if ci_u == 1:
                            if b0 == 0xE0: lo = 0xA0
                            elif b0 == 0xED: hi = 0x9F
                            elif b0 == 0xF0: lo = 0x90
                            elif b0 == 0xF4: hi = 0x8F
                        mask_e[row_i, lo : hi + 1] = True
                    nxt_list = lg_e.masked_fill(~mask_e, -1e9).argmax(dim=-1).cpu().tolist()
                    for row_i, ji in enumerate(act_e):
                        esc_bufs[ji].append(nxt_list[row_i])
                        cur_bs[ji] = nxt_list[row_i]
                for ji, (wi, _, _, added_set, score_fn) in enumerate(mb_escape_jobs):
                    mb = bytes(esc_bufs[ji])
                    if mb in added_set:
                        continue
                    try:
                        mb.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        continue
                    mid_v = bytes_to_mid.get(mb, -1)
                    l2_b, l2_g, inh_b, cg = score_fn(mid_v, mb, 0.25)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                    cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                    cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(True)
                    added_set.add(mb)

            if not cand_parent:
                break

            if len(cand_parent) > frontier_cap:
                reserved, seen_esc = [], set()
                for i in range(len(cand_parent)):
                    if cand_is_l1_esc[i] and cand_parent[i] not in seen_esc:
                        reserved.append(i); seen_esc.add(cand_parent[i])
                res_set = set(reserved)
                rem_cap = max(0, frontier_cap - len(reserved))
                half_cap = max(4, rem_cap // 2)
                pr_idx = [i for i, f in enumerate(cand_keep_prio) if f and i not in res_set]
                pr_idx.sort(key=lambda i: beams[cand_parent[i]]["score"] + 0.55 * (cand_l2_base[i] + cand_l2_gated[i]), reverse=True)
                pr_idx = pr_idx[:half_cap]
                ot_idx = [i for i, f in enumerate(cand_keep_prio) if (not f) and i not in res_set]
                ot_idx.sort(key=lambda i: beams[cand_parent[i]]["score"] + 0.55 * (cand_l2_base[i] + cand_l2_gated[i]), reverse=True)
                ord_c = reserved + pr_idx + ot_idx[: max(0, rem_cap - len(pr_idx))]
                cand_parent = [cand_parent[i] for i in ord_c]
                cand_mid = [cand_mid[i] for i in ord_c]
                cand_l2_base = [cand_l2_base[i] for i in ord_c]
                cand_l2_gated = [cand_l2_gated[i] for i in ord_c]
                cand_inh_b = [cand_inh_b[i] for i in ord_c]
                cand_bytes = [cand_bytes[i] for i in ord_c]
                cand_cg = [cand_cg[i] for i in ord_c]

            M = len(cand_parent)
            p_idx = torch.tensor(cand_parent, dtype=torch.int64, device=dev)
            st_b_c = rb.gather_state(st_b_beams, p_idx)
            st_r_c = rr.gather_state(st_r_beams, p_idx) if st_r_beams is not None else None
            log_c = logits_beams[p_idx].clone()
            max_blen = max(len(mb) for mb in cand_bytes)
            cand_l1_lp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_cur_clp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_min_clp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_has_chr = torch.zeros(M, dtype=torch.bool, device=dev)

            for bi in range(max_blen):
                act = [i for i in range(M) if bi < len(cand_bytes[i])]
                if not act:
                    break
                cur_pos = (int(beams[0]["n_bytes"]) if beams else 0) + bi
                if len(act) == M:
                    lp_act = torch.log_softmax(log_c, dim=-1)
                    yb_t = torch.tensor([cand_bytes[i][bi] for i in range(M)], dtype=torch.int64, device=dev)
                    step_lp = lp_act[torch.arange(M, device=dev), yb_t]
                    cand_l1_lp += step_lp
                    is_lead = (yb_t & 0xC0) != 0x80
                    cand_min_clp = torch.where(is_lead & cand_has_chr, torch.minimum(cand_min_clp, cand_cur_clp), cand_min_clp)
                    cand_cur_clp = torch.where(is_lead, step_lp, cand_cur_clp + step_lp)
                    cand_has_chr = cand_has_chr | is_lead
                    log_c = _step_gen(st_b_c, st_r_c, yb_t, None, is_p=False, ans_pos=cur_pos)
                else:
                    act_t = torch.tensor(act, dtype=torch.int64, device=dev)
                    lp_act = torch.log_softmax(log_c[act_t], dim=-1)
                    yb_t = torch.tensor([cand_bytes[i][bi] for i in act], dtype=torch.int64, device=dev)
                    step_lp = lp_act[torch.arange(len(act), device=dev), yb_t]
                    cand_l1_lp[act_t] += step_lp
                    is_lead = (yb_t & 0xC0) != 0x80
                    prev_has = cand_has_chr[act_t]
                    cand_min_clp[act_t] = torch.where(is_lead & prev_has, torch.minimum(cand_min_clp[act_t], cand_cur_clp[act_t]), cand_min_clp[act_t])
                    cand_cur_clp[act_t] = torch.where(is_lead, step_lp, cand_cur_clp[act_t] + step_lp)
                    cand_has_chr[act_t] = prev_has | is_lead
                    stb_sub = rb.gather_state(st_b_c, act_t)
                    str_sub = rr.gather_state(st_r_c, act_t) if st_r_c is not None else None
                    log_sub = _step_gen(stb_sub, str_sub, yb_t, None, is_p=False, ans_pos=cur_pos)
                    for k_st in st_b_c.keys():
                        st_b_c[k_st][act_t] = stb_sub[k_st]
                    if st_r_c is not None and str_sub is not None:
                        for k_st in st_r_c.keys():
                            st_r_c[k_st][act_t] = str_sub[k_st]
                    log_c[act_t] = log_sub

            cand_min_clp = torch.where(cand_has_chr, torch.minimum(cand_min_clp, cand_cur_clp), cand_l1_lp)
            l1_lp_cpu = cand_l1_lp.cpu().tolist()
            min_clp_cpu = cand_min_clp.cpu().tolist()
            tau_syn = -2.40 if (has_custom and step_i < 12) else -1.85
            tau_wch = -5.20 if (has_custom and step_i < 12) else -3.95

            next_pool = []
            for ci in range(M):
                bm = beams[cand_parent[ci]]
                mid = cand_mid[ci]
                mb = cand_bytes[ci]
                cg = cand_cg[ci]
                l1_step = l1_lp_cpu[ci]
                worst_c = min_clp_cpu[ci]

                l1_pb = l1_step / max(1, len(mb))
                g_avg = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, (l1_pb - tau_syn) * 2.2))))
                g_wch = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, (worst_c - tau_wch) * 1.4))))
                g_syn = g_avg * g_wch
                wch_pen = 0.28 * max(0.0, (tau_wch + 0.15) - worst_c)

                l2_step = cand_l2_base[ci] + g_syn * cand_l2_gated[ci] - wch_pen
                inh_step = cand_inh_b[ci] * len(mb)

                m_add = [mid] if mid >= 0 else list(mb)
                new_mseq = bm["motor_seq"] + m_add
                new_seen3 = set(bm["seen_m3"])
                new_seen2 = set(bm["seen_m2"])
                rep_pen = 0.0
                if len(new_mseq) >= 3:
                    t3 = (new_mseq[-3], new_mseq[-2], new_mseq[-1])
                    if t3 in new_seen3:
                        rep_pen += 3.5
                    new_seen3.add(t3)
                if len(new_mseq) >= 2:
                    t2 = (new_mseq[-2], new_mseq[-1])
                    if t2 in new_seen2:
                        rep_pen += 1.4
                    new_seen2.add(t2)
                if has_evidence and (not has_custom) and len(new_mseq) >= 4:
                    t4 = (new_mseq[-4], new_mseq[-3], new_mseq[-2], new_mseq[-1])
                    if t4 in ep_4grams:
                        rep_pen += 0.8

                new_used = set(bm["used_content"])
                if cg >= 0.45 and mid >= 256 and mid not in new_used:
                    new_used.add(mid)

                raw_l1 = bm["raw_l1_lp"] + l1_step
                raw_l2 = bm["raw_l2_sc"] + l2_step
                tot_inh = bm["tot_inh"] + inh_step
                tot_rep = bm["tot_rep"] + rep_pen
                nb = bm["n_bytes"] + len(mb)
                nm = len(bm["gen_motors"]) + 1
                total_sc = (raw_l1 - tot_inh) / (nb ** 0.90) + 0.30 * (raw_l2 / (nm ** 0.85)) - 0.18 * tot_rep
                next_pool.append((total_sc, ci, {
                    "score": total_sc, "raw_l1_lp": raw_l1, "raw_l2_sc": raw_l2, "tot_inh": tot_inh,
                    "tot_rep": tot_rep, "n_bytes": nb, "motor_seq": new_mseq,
                    "gen_motors": bm["gen_motors"] + m_add, "gen_bytes": bm["gen_bytes"] + mb,
                    "seen_m3": new_seen3, "seen_m2": new_seen2, "used_content": new_used, "done": False,
                }))

            next_pool.sort(key=lambda x: x[0], reverse=True)
            kept, kept_ci, seen_txt = [], [], set()
            for sc, ci, nbm in next_pool:
                tb = bytes(nbm["gen_bytes"])
                if tb in seen_txt:
                    continue
                seen_txt.add(tb)
                kept.append(nbm); kept_ci.append(ci)
                if len(kept) >= beam_width:
                    break
            if not kept:
                break
            k_t = torch.tensor(kept_ci, dtype=torch.int64, device=dev)
            st_b_beams = rb.gather_state(st_b_c, k_t)
            st_r_beams = rr.gather_state(st_r_c, k_t) if st_r_c is not None else None
            logits_beams = log_c[k_t].clone()
            beams = kept

            if finished_beams:
                best_fin = max(finished_beams, key=lambda x: x["score"])
                if best_fin["score"] > beams[0]["score"] + 0.12 and len(finished_beams) >= beam_width:
                    break

        all_cands = finished_beams + beams
        all_cands.sort(key=lambda x: x["score"], reverse=True)
        best = all_cands[0]
        decode_sec = time.perf_counter() - t0
        ans_b = bytes(best["gen_bytes"])
        try:
            ans_str = ans_b.decode("utf-8", errors="strict")
            valid_utf8 = True
        except UnicodeDecodeError:
            ans_str = ans_b.decode("utf-8", errors="ignore")
            valid_utf8 = False

        bpc = rb.score_text_bpc(frame_prefix.decode("utf-8", errors="ignore") + ans_str, prompt_prefix_len=len(frame_prefix)) if score_critic else -1.0
        return ans_str, valid_utf8, {
            "intent_mode": intent_mode,
            "want_hip": want_hip,
            "entities": [e.decode("utf-8", errors="ignore") for e in entity_spans],
            "exact_entities": delib["exact_entities"],
            "agenda_preview": [t[:75].replace("\n", " ") for t in delib["agenda_texts"]],
            "stopped_eot": bool(best.get("done", False)),
            "decode_sec": round(decode_sec, 2),
            "fluency_bpc": bpc,
        }
