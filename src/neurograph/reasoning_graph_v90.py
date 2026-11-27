"""推論用の成長ノードと、条件の積で検索する文章記憶（v9.0）。"""

from __future__ import annotations

import hashlib
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neurograph.instruct_cortex_v80 import InstructAdapterV80, InstructBrainV80, _mem_snapshot
from neurograph.role_cortex_v60 import RoleBoundThoughtCortexV60
from neurograph.sft_data_v80 import TURN_A_PREFIX, TURN_EOT_BYTE, TURN_U_PREFIX
from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50


def trim_utf8_safe(raw: bytes, max_bytes: int) -> bytes:
    if len(raw) <= max_bytes:
        return raw
    end = max_bytes
    while end > 0 and (raw[end] & 0xC0) == 0x80:
        end -= 1
    cand = raw[:end]
    try:
        cand.decode("utf-8", errors="strict")
        return cand
    except UnicodeDecodeError:
        while end > 0:
            end -= 1
            if (raw[end] & 0xC0) != 0x80:
                try:
                    raw[:end].decode("utf-8", errors="strict")
                    return raw[:end]
                except UnicodeDecodeError:
                    continue
        return b""


def compute_vocab_sha256(motor_bytes: List[bytes]) -> str:
    h = hashlib.sha256()
    h.update(str(len(motor_bytes)).encode("ascii"))
    for mb in motor_bytes:
        h.update(len(mb).to_bytes(2, "little"))
        h.update(mb)
    return h.hexdigest()[:32]


@dataclass
class ReasoningNode:
    node_id: int
    centroid: np.ndarray
    ans_proto_sum: np.ndarray
    ans_concept_proto: np.ndarray
    fires: int = 0
    hip_need_sum: float = 0.0
    multi_ent_sum: float = 0.0
    ans_bytes_sum: float = 0.0
    ans_bytes_samples: List[int] = field(default_factory=list)
    opener_counts: Dict[int, int] = field(default_factory=dict)
    sample_prompts: List[str] = field(default_factory=list)

    hip_need_rate: float = 0.5
    multi_entity_rate: float = 0.0
    mean_ans_bytes: float = 90.0
    p25_ans_bytes: float = 42.0
    top_openers: List[Tuple[int, float]] = field(default_factory=list)


class PrefrontalReasoningGraphV90:

    def __init__(self, V_m: int, thought_dim: int = 256, core_dim: int = 256, seed: int = 90, vocab_hash: str = ""):
        self.V_m = int(V_m)
        self.D_m = int(thought_dim)
        self.D_c = int(core_dim)
        self.seed = int(seed)
        self.vocab_hash = vocab_hash

        rng = np.random.default_rng(seed)
        self.P_op = rng.standard_normal((self.V_m, self.D_m)).astype(np.float32)
        norms = np.linalg.norm(self.P_op, axis=1, keepdims=True)
        self.P_op /= np.maximum(norms, 1e-6)

        self.nodes: List[ReasoningNode] = []
        self.centroid_mat: Optional[np.ndarray] = None
        self.tau_user_idf: float = 6.2
        self.rare_fact_idf: float = 12.4
        self.W_l2_to_l1: Optional[np.ndarray] = None
        self.y_l1_mean: Optional[np.ndarray] = None
        self._W_bridge_gpu: Optional[torch.Tensor] = None

    def extract_entity_spans(
        self,
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        adapter: InstructAdapterV80,
        user_prompt: str,
    ) -> Tuple[List[bytes], List[int]]:
        u_bytes = user_prompt.strip().encode("utf-8")
        u_motors = mc.encode(np.frombuffer(u_bytes, dtype=np.uint8)).tolist()
        u_idf_arr = adapter.user_prompt_idf

        role_bytes_set = [mc.motor_bytes[int(m)] for m in rc.role_motor_ids if len(mc.motor_bytes[int(m)]) >= 2]
        boiler_bytes: List[bytes] = []
        if u_idf_arr is not None:
            cand_b = np.where((np.arange(len(u_idf_arr)) >= 256) & (u_idf_arr < 4.3))[0]
            boiler_bytes = sorted(
                [mc.motor_bytes[int(m)] for m in cand_b if len(mc.motor_bytes[int(m)]) >= 3],
                key=len, reverse=True,
            )[:48]

        def _strip_learned_boundaries(sb: bytes) -> List[bytes]:
            pieces = [sb]
            for rb in role_bytes_set:
                nxt: List[bytes] = []
                for pc in pieces:
                    if rb in pc:
                        for sub in pc.split(rb):
                            sub = sub.strip()
                            if len(sub) >= 4:
                                nxt.append(sub)
                    else:
                        nxt.append(pc)
                pieces = nxt
            cleaned: List[bytes] = []
            for pc in pieces:
                changed = True
                while changed and len(pc) >= 4:
                    changed = False
                    for bb in boiler_bytes:
                        if pc.endswith(bb) and len(pc) - len(bb) >= 3:
                            pc = pc[: -len(bb)].strip()
                            changed = True
                            break
                        if pc.startswith(bb) and len(pc) - len(bb) >= 3:
                            pc = pc[len(bb) :].strip()
                            changed = True
                            break
                if len(pc) >= 6:
                    for edge in ("end", "start"):
                        p_ms = mc.encode(np.frombuffer(pc, dtype=np.uint8)).tolist()
                        if len(p_ms) >= 2:
                            em = p_ms[-1] if edge == "end" else p_ms[0]
                            emb = mc.motor_bytes[em]
                            if len(emb) <= 3 and em >= 256 and float(rc.cg_vec[em]) < 0.35:
                                pc = pc[: -len(emb)] if edge == "end" else pc[len(emb) :]
                if len(pc) >= 4:
                    try:
                        pc.decode("utf-8", errors="strict")
                        pm = mc.encode(np.frombuffer(pc, dtype=np.uint8)).tolist()
                        max_cg = max((float(rc.cg_vec[m]) for m in pm if m >= 256), default=0.0)
                        has_oov_alnum = any(m < 256 and m > 32 for m in pm)
                        mean_c_idf = float(np.mean([float(mc.motor_idfs[m]) for m in pm])) if pm else 0.0
                        if (max_cg >= 0.40 or has_oov_alnum) and mean_c_idf >= (mc.tau_idf + 1.0):
                            cleaned.append(pc)
                    except UnicodeDecodeError:
                        pass
            return cleaned

        raw_spans: List[bytes] = []
        entity_mids: List[int] = []
        cur_span = bytearray()

        def _flush():
            nonlocal cur_span
            if len(cur_span) >= 4:
                sb = bytes(cur_span).strip()
                if len(sb) >= 4:
                    try:
                        sb.decode("utf-8", errors="strict")
                        raw_spans.append(sb)
                    except UnicodeDecodeError:
                        pass
            cur_span = bytearray()

        idf_cut = max(5.8, self.tau_user_idf)
        for idx_m, m in enumerate(u_motors):
            if m < 256:
                if (m & 0xC0) == 0x80:
                    if len(cur_span) > 0:
                        cur_span.append(m)
                elif m > 32:
                    u_idf_b = float(u_idf_arr[m]) if (u_idf_arr is not None and m < len(u_idf_arr)) else float(mc.motor_idfs[m])
                    if u_idf_b >= idf_cut or (len(cur_span) > 0 and u_idf_b >= (idf_cut - 1.5)):
                        cur_span.append(m)
                    else:
                        _flush()
                else:
                    _flush()
                continue
            if rc.mid_to_role[m] >= 0:
                _flush()
                continue
            c_idf = float(mc.motor_idfs[m])
            u_idf = float(u_idf_arr[m]) if (u_idf_arr is not None and m < len(u_idf_arr)) else c_idf
            cg_m = float(rc.cg_vec[m])
            eff_idf = min(c_idf, u_idf)
            is_ent_mod = False
            if eff_idf < idf_cut and cg_m >= 0.45 and c_idf >= (mc.tau_idf + 1.2) and u_idf >= 4.4:
                if idx_m + 1 < len(u_motors):
                    nm = u_motors[idx_m + 1]
                    if nm >= 256 and rc.mid_to_role[nm] < 0:
                        nu = float(u_idf_arr[nm]) if (u_idf_arr is not None and nm < len(u_idf_arr)) else float(mc.motor_idfs[nm])
                        if min(float(mc.motor_idfs[nm]), nu) >= idf_cut:
                            is_ent_mod = True
            if (eff_idf >= idf_cut and cg_m >= 0.30) or is_ent_mod:
                cur_span.extend(mc.motor_bytes[m])
                if m not in entity_mids:
                    entity_mids.append(m)
            else:
                _flush()
        _flush()

        spans: List[bytes] = []
        for rs in raw_spans:
            for sp in _strip_learned_boundaries(rs):
                sm = mc.encode(np.frombuffer(sp, dtype=np.uint8)).tolist()
                if not sm:
                    continue
                max_u = max((float(u_idf_arr[m]) if (u_idf_arr is not None and m < len(u_idf_arr)) else float(mc.motor_idfs[m])) for m in sm)
                if len(sp) <= 3 and max_u < 11.0:
                    continue
                if sp not in spans:
                    spans.append(sp)

        def _span_prio(sp: bytes) -> float:
            sm = mc.encode(np.frombuffer(sp, dtype=np.uint8)).tolist()
            if not sm:
                return 0.0
            c_sc = sum(float(mc.motor_idfs[m]) * (0.5 + 0.5 * float(rc.cg_vec[m])) for m in sm)
            return c_sc * math.sqrt(len(sp))

        spans.sort(key=_span_prio, reverse=True)
        return spans, entity_mids

    def compute_question_signature(
        self,
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        adapter: InstructAdapterV80,
        u_motors: np.ndarray,
        ctx_motors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        D = self.D_m
        u_idf_arr = adapter.user_prompt_idf
        q_op = np.zeros(D, dtype=np.float32)

        for pos_i, m in enumerate(u_motors.tolist()):
            if m < 0 or m >= self.V_m:
                continue
            u_idf = float(u_idf_arr[m]) if (u_idf_arr is not None and m < len(u_idf_arr)) else float(mc.motor_idfs[m])
            if m < 256 or rc.mid_to_role[m] >= 0 or u_idf < (self.tau_user_idf + 0.8):
                w_op = max(0.15, (self.tau_user_idf + 1.0) - u_idf)
                if pos_i == 0 or pos_i >= max(0, len(u_motors) - 4):
                    w_op *= 1.35
                q_op += w_op * self.P_op[m]

        nq = float(np.linalg.norm(q_op))
        if nq > 1e-6:
            q_op /= nq

        src_ctx = ctx_motors if (ctx_motors is not None and len(ctx_motors) > 0) else u_motors
        q_sem = adapter.predict_qa_intent(rc, src_ctx)
        ns = float(np.linalg.norm(q_sem))
        if ns > 1e-6:
            q_sem = q_sem / ns

        sig = np.concatenate([0.65 * q_op, 0.35 * q_sem]).astype(np.float32)
        nt = float(np.linalg.norm(sig))
        if nt > 1e-6:
            sig /= nt
        return sig

    def grow_from_sft_dialogues(
        self,
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        adapter: InstructAdapterV80,
        train_dialogues: List[Dict],
        max_dialogues: int = 35_000,
        vigilance_rho: float = 0.48,
        max_nodes: int = 96,
        min_fires_survive: int = 40,
    ) -> Dict[str, object]:
        if not train_dialogues:
            raise ValueError("train_dialogues is empty; cannot grow Reasoning Graph!")
        if mc.emb_np is None or mc.emb_np.shape != (self.V_m, self.D_m):
            raise ValueError(f"mc.emb_np missing or shape mismatch: expected ({self.V_m}, {self.D_m})")

        t0 = time.perf_counter()
        self.nodes = []
        self.vocab_hash = compute_vocab_sha256(mc.motor_bytes)

        if adapter.user_prompt_idf is None:
            adapter.compute_user_prompt_idf(mc, train_dialogues)

        u_idfs = adapter.user_prompt_idf[256:].astype(np.float64)
        max_u = float(u_idfs.max()) if len(u_idfs) > 0 else 10.0
        obs = u_idfs[u_idfs < (max_u - 0.05)]
        if len(obs) > 0:
            w = np.exp(-obs)
            h1_u = float((w * obs).sum() / (w.sum() + 1e-12))
            self.tau_user_idf = max(5.2, 0.78 * h1_u)
        self.rare_fact_idf = float(np.percentile(mc.motor_idfs[256:], 65)) if mc.V_m > 256 else 12.4

        turns_seen = 0
        births = 0
        centroid_list: List[np.ndarray] = []

        for d in train_dialogues[:max_dialogues]:
            hist_m: List[int] = []
            for u, a in d["turns"]:
                u_m = mc.encode(np.frombuffer(u.encode("utf-8"), dtype=np.uint8))
                a_m = mc.encode(np.frombuffer(a.encode("utf-8"), dtype=np.uint8))
                hist_m.extend(u_m.tolist())
                if len(u_m) == 0 or len(a_m) == 0:
                    continue
                turns_seen += 1

                ctx_arr = np.array(hist_m[-48:], dtype=np.int32)
                sig = self.compute_question_signature(mc, rc, adapter, u_m, ctx_arr)

                best_idx = -1
                best_sim = -1.0
                if centroid_list:
                    c_mat = np.stack(centroid_list, axis=0)
                    sims = c_mat @ sig
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])

                if best_idx < 0 or (best_sim < vigilance_rho and len(self.nodes) < max_nodes):
                    nid = len(self.nodes)
                    nd = ReasoningNode(
                        node_id=nid,
                        centroid=sig.copy(),
                        ans_proto_sum=np.zeros(self.D_m, dtype=np.float64),
                        ans_concept_proto=np.zeros(self.D_m, dtype=np.float32),
                    )
                    self.nodes.append(nd)
                    centroid_list.append(nd.centroid)
                    best_idx = nid
                    births += 1

                nd = self.nodes[best_idx]
                nd.fires += 1
                lr_c = 1.0 / min(nd.fires + 2, 120)
                nc = (1.0 - lr_c) * nd.centroid + lr_c * sig
                nn = float(np.linalg.norm(nc))
                if nn > 1e-6:
                    nd.centroid = (nc / nn).astype(np.float32)
                    centroid_list[best_idx] = nd.centroid

                ent_spans, ent_mids = self.extract_entity_spans(mc, rc, adapter, u)
                a_list = a_m.tolist()
                u_set = set(u_m.tolist())
                novel_rare_facts = sum(
                    1 for m in set(a_list)
                    if m >= 256 and m not in u_set and rc.mid_to_role[m] < 0
                    and float(mc.motor_idfs[m]) >= self.rare_fact_idf
                    and float(rc.cg_vec[m]) >= 0.45
                )
                need_fact = 1.0 if (len(ent_spans) >= 1 and novel_rare_facts >= 3) else 0.0
                nd.hip_need_sum += need_fact
                nd.multi_ent_sum += (1.0 if len(ent_spans) >= 2 else 0.0)

                ab_len = len(a.encode("utf-8"))
                nd.ans_bytes_sum += ab_len
                if len(nd.ans_bytes_samples) < 512:
                    nd.ans_bytes_samples.append(ab_len)

                if a_list[0] >= 256:
                    nd.opener_counts[a_list[0]] = nd.opener_counts.get(a_list[0], 0) + 1

                a_vecs = [
                    mc.emb_np[m] * float(rc.cg_vec[m])
                    for m in a_list[:32]
                    if m >= 256 and rc.mid_to_role[m] < 0 and rc.cg_vec[m] >= rc.content_cg_thresh
                    and float(np.dot(mc.emb_np[m], mc.emb_np[m])) > 0.1
                ]
                if a_vecs:
                    av = np.sum(a_vecs, axis=0)
                    na = float(np.linalg.norm(av))
                    if na > 1e-6:
                        nd.ans_proto_sum += (av / na)

                if len(nd.sample_prompts) < 3:
                    nd.sample_prompts.append(u[:60])

                hist_m.extend(a_list)

        if not self.nodes:
            raise RuntimeError("No Reasoning Nodes were grown from train_dialogues!")

        surviving = [nd for nd in self.nodes if nd.fires >= min_fires_survive]
        if not surviving:
            surviving = sorted(self.nodes, key=lambda x: x.fires, reverse=True)[:16]
        surviving.sort(key=lambda x: x.fires, reverse=True)

        self.nodes = []
        for new_id, nd in enumerate(surviving):
            nd.node_id = new_id
            f = max(1, nd.fires)
            nd.hip_need_rate = round(float(nd.hip_need_sum / f), 3)
            nd.multi_entity_rate = round(float(nd.multi_ent_sum / f), 3)
            nd.mean_ans_bytes = round(float(nd.ans_bytes_sum / f), 1)
            nd.p25_ans_bytes = round(float(np.percentile(nd.ans_bytes_samples, 25)), 1) if nd.ans_bytes_samples else 36.0
            pn = float(np.linalg.norm(nd.ans_proto_sum))
            nd.ans_concept_proto = (nd.ans_proto_sum / pn).astype(np.float32) if pn > 1e-6 else np.zeros(self.D_m, dtype=np.float32)
            top_o = sorted(nd.opener_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            omx = max(top_o[0][1], 1) if top_o else 1
            nd.top_openers = [(m, round(math.log1p(c) / math.log1p(omx), 4)) for m, c in top_o]
            nd.ans_bytes_samples = []
            nd.opener_counts = {}
            self.nodes.append(nd)

        self.centroid_mat = np.ascontiguousarray(np.stack([nd.centroid for nd in self.nodes], axis=0), dtype=np.float32)
        return {
            "turns_processed": turns_seen,
            "nodes_born": births,
            "nodes_survived_apoptosis": len(self.nodes),
            "tau_user_idf": round(self.tau_user_idf, 3),
            "hip_need_range": [min(n.hip_need_rate for n in self.nodes), max(n.hip_need_rate for n in self.nodes)],
            "sec": round(time.perf_counter() - t0, 2),
        }

    @torch.no_grad()
    def train_l2_to_l1_goal_bridge(
        self,
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        runner,
        train_dialogues: List[Dict],
        val_dialogues: Optional[List[Dict]] = None,
        max_answers: int = 3_000,
        ridge_alpha: float = 0.25,
    ) -> Dict[str, object]:
        if mc.emb_np is None:
            raise ValueError("mc.emb_np is None; cannot train L2->L1 bridge!")
        t0 = time.perf_counter()
        dev = runner.device
        C_xx = torch.zeros((self.D_m, self.D_m), dtype=torch.float64, device=dev)
        C_xy = torch.zeros((self.D_m, self.D_c), dtype=torch.float64, device=dev)
        x_list: List[np.ndarray] = []
        y_list: List[torch.Tensor] = []
        used = 0

        def _extract_pair(ans_str: str) -> Optional[Tuple[np.ndarray, torch.Tensor]]:
            ab = trim_utf8_safe(ans_str.encode("utf-8"), 240)
            if len(ab) < 12:
                return None
            a_m = mc.encode(np.frombuffer(ab, dtype=np.uint8)).tolist()
            a_vecs = [
                mc.emb_np[m] * float(rc.cg_vec[m])
                for m in a_m[:32]
                if m >= 256 and rc.mid_to_role[m] < 0 and rc.cg_vec[m] >= rc.content_cg_thresh
                and float(np.dot(mc.emb_np[m], mc.emb_np[m])) > 0.1
            ]
            if not a_vecs:
                return None
            xl2 = np.sum(a_vecs, axis=0).astype(np.float32)
            nx = float(np.linalg.norm(xl2))
            if nx < 1e-6:
                return None
            xl2 /= nx
            yl1 = runner.encode_text_h(ab)
            if yl1 is None:
                return None
            return xl2, yl1

        for d in train_dialogues:
            for _, a in d["turns"]:
                if used >= max_answers:
                    break
                pr = _extract_pair(a)
                if pr is None:
                    continue
                x_list.append(pr[0])
                y_list.append(pr[1])
                used += 1
            if used >= max_answers:
                break

        if used < 32:
            raise RuntimeError(f"Insufficient valid answers ({used}) to train L2->L1 bridge!")

        xt = torch.from_numpy(np.stack(x_list)).to(device=dev, dtype=torch.float64)
        yt = torch.stack(y_list).to(device=dev, dtype=torch.float64)
        x_list.clear(); y_list.clear()

        y_mean_t = yt.mean(dim=0)
        yt_c = yt - y_mean_t.unsqueeze(0)
        C_xx = xt.T @ xt
        C_xy = xt.T @ yt_c

        lam = ridge_alpha * float((torch.trace(C_xx) / self.D_m).clamp(min=1.0).item())
        W = torch.linalg.solve(C_xx + lam * torch.eye(self.D_m, dtype=torch.float64, device=dev), C_xy)
        self.W_l2_to_l1 = np.ascontiguousarray(W.T.to(torch.float32).cpu().numpy(), dtype=np.float32)
        self.y_l1_mean = np.ascontiguousarray(y_mean_t.to(torch.float32).cpu().numpy(), dtype=np.float32)
        self._W_bridge_gpu = None
        del xt, yt, yt_c, C_xx, C_xy, W

        val_cos_c, val_cos_raw_c = [], []
        if val_dialogues:
            ym_dev = torch.from_numpy(self.y_l1_mean).to(dev)
            for d in val_dialogues[:250]:
                for _, a in d["turns"]:
                    pr = _extract_pair(a)
                    if pr is None:
                        continue
                    xl2, yl1 = pr
                    yl1_c = yl1 - ym_dev
                    yl1_cn = yl1_c / yl1_c.norm().clamp(min=1e-6)
                    gz = self.project_to_goal_z(xl2, dev)
                    if gz is not None:
                        val_cos_c.append(float(torch.dot(gz, yl1_cn).item()))
                        val_cos_raw_c.append(float(torch.dot(torch.from_numpy(xl2).to(dev), yl1_cn).item()))

        return {
            "answers_aligned": used,
            "W_bridge_norm": round(float(np.linalg.norm(self.W_l2_to_l1)), 3),
            "val_centered_bridge_cos": round(float(np.mean(val_cos_c)), 4) if val_cos_c else None,
            "val_centered_unbridged_cos": round(float(np.mean(val_cos_raw_c)), 4) if val_cos_raw_c else None,
            "sec": round(time.perf_counter() - t0, 2),
            "mem": _mem_snapshot(dev),
        }

    def route(
        self,
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        adapter: InstructAdapterV80,
        u_motors: np.ndarray,
        ctx_motors: Optional[np.ndarray] = None,
    ) -> Tuple[ReasoningNode, float]:
        assert self.centroid_mat is not None and len(self.nodes) > 0, "Reasoning graph has 0 grown nodes!"
        sig = self.compute_question_signature(mc, rc, adapter, u_motors, ctx_motors)
        sims = self.centroid_mat @ sig
        idx = int(np.argmax(sims))
        return self.nodes[idx], float(sims[idx])

    def project_to_goal_z(self, h_l2: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
        if self.W_l2_to_l1 is None or float(np.linalg.norm(h_l2)) < 1e-6:
            return None
        if self._W_bridge_gpu is None or self._W_bridge_gpu.device != device:
            self._W_bridge_gpu = torch.from_numpy(self.W_l2_to_l1).to(device=device, dtype=torch.float32)
        h_t = torch.from_numpy(h_l2.astype(np.float32)).to(device)
        gz = self._W_bridge_gpu @ h_t
        return gz / gz.norm().clamp(min=1e-6)

    def save(self, folder: str) -> float:
        if not self.nodes or self.centroid_mat is None or self.W_l2_to_l1 is None:
            raise RuntimeError("Cannot save incomplete PrefrontalReasoningGraphV90!")
        t0 = time.perf_counter()
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "v90_v3",
            "V_m": self.V_m,
            "D_m": self.D_m,
            "D_c": self.D_c,
            "seed": self.seed,
            "vocab_hash": self.vocab_hash,
            "tau_user_idf": self.tau_user_idf,
            "rare_fact_idf": self.rare_fact_idf,
            "nodes": self.nodes,
            "centroid_mat": self.centroid_mat,
            "W_l2_to_l1": self.W_l2_to_l1,
            "y_l1_mean": self.y_l1_mean,
        }
        with open(p / "reasoning_graph_v90.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return round(time.perf_counter() - t0, 3)

    @classmethod
    def load(cls, folder: str, mc: Optional[ContinuousThoughtCortexV50] = None) -> "PrefrontalReasoningGraphV90":
        p = Path(folder)
        with open(p / "reasoning_graph_v90.pkl", "rb") as f:
            d = pickle.load(f)
        obj = cls(V_m=d["V_m"], thought_dim=d["D_m"], core_dim=d["D_c"], seed=d["seed"], vocab_hash=d.get("vocab_hash", ""))
        if mc is not None:
            if mc.V_m != obj.V_m or mc.thought_dim != obj.D_m:
                raise ValueError(f"ReasoningGraph shape mismatch vs mc: ({obj.V_m},{obj.D_m}) vs ({mc.V_m},{mc.thought_dim})")
            cur_hash = compute_vocab_sha256(mc.motor_bytes)
            if obj.vocab_hash and cur_hash != obj.vocab_hash:
                raise ValueError(f"ReasoningGraph vocab hash mismatch: saved={obj.vocab_hash} vs mc={cur_hash}")
        obj.tau_user_idf = float(d["tau_user_idf"])
        obj.rare_fact_idf = float(d.get("rare_fact_idf", 12.4))
        obj.nodes = d["nodes"]
        obj.centroid_mat = d["centroid_mat"]
        obj.W_l2_to_l1 = d["W_l2_to_l1"]
        obj.y_l1_mean = d.get("y_l1_mean", None)
        return obj


class DeliberativeReasoningBrainV90:

    def __init__(self, chat_v80: InstructBrainV80, rgraph: PrefrontalReasoningGraphV90):
        self.chat = chat_v80
        self.unified = chat_v80.unified
        self.v20 = chat_v80.v20
        self.mc = chat_v80.mc
        self.rc = chat_v80.rc
        self.hip = chat_v80.hip
        self.runner = chat_v80.runner
        self.device = chat_v80.device
        self.adapter = chat_v80.adapter
        self.rgraph = rgraph

    def _split_evidence_clauses(self, ep_bytes: bytes, min_bytes: int = 24, max_bytes: int = 240) -> List[bytes]:
        raw_lines = [ln.strip() for ln in ep_bytes.split(b"\n") if len(ln.strip()) >= min_bytes]
        out: List[bytes] = []
        top_term = b""
        if self.adapter.eot_after_motor:
            best_tm = max(
                ((m, p) for m, p in self.adapter.eot_after_motor.items() if m >= 256 and len(self.mc.motor_bytes[m]) <= 4),
                key=lambda x: x[1],
                default=(-1, 0.0),
            )[0]
            if best_tm >= 256:
                top_term = self.mc.motor_bytes[best_tm]

        def _window_utf8_safe(seg: bytes) -> List[bytes]:
            res: List[bytes] = []
            pos = 0
            while pos < len(seg):
                chunk = trim_utf8_safe(seg[pos:], max_bytes).strip()
                if len(chunk) >= min_bytes:
                    res.append(chunk)
                step = max(len(chunk), min_bytes)
                pos += step
                while pos < len(seg) and (seg[pos] & 0xC0) == 0x80:
                    pos += 1
            return res

        for ln in raw_lines:
            if len(ln) <= max_bytes:
                safe = trim_utf8_safe(ln, max_bytes)
                if len(safe) >= min_bytes:
                    out.append(safe)
            elif top_term and top_term in ln:
                parts = ln.split(top_term)
                for idx, pt in enumerate(parts):
                    seg = (pt + top_term).strip() if idx < len(parts) - 1 else pt.strip()
                    if len(seg) < min_bytes:
                        continue
                    if len(seg) <= max_bytes:
                        out.append(seg)
                    else:
                        out.extend(_window_utf8_safe(seg))
            else:
                out.extend(_window_utf8_safe(ln))
        return out

    def _conjunctive_hippocampus_deliberate(
        self,
        entity_spans: List[bytes],
        u_bytes: bytes,
        node: Optional[ReasoningNode],
        h_qa: np.ndarray,
        top_k_agenda: int = 2,
        conjunctive: bool = True,
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        empty = {
            "agenda_bytes": [], "agenda_texts": [], "agenda_sources": [], "agenda_scores": [],
            "exact_entities": [], "partial_entities": [], "uncovered_entities": [e.decode("utf-8", errors="ignore") for e in entity_spans[:5]],
            "ep_concepts": {}, "ep_trans1": {}, "ep_trans2": {}, "ep_4grams": set(),
            "ep_top_motors": [], "h_ev_l2": np.zeros(self.mc.thought_dim, dtype=np.float32),
            "epi_conf": 0.0, "has_custom": False, "hop23_ms": 0.0,
        }
        if self.hip is None or self.hip.total_episodes == 0 or not entity_spans:
            return empty

        target_ents = entity_spans[:5]
        n_ents = len(target_ents)
        joint_q = b" ".join(target_ents)
        queries: List[bytes] = []
        if len(joint_q) >= 6:
            queries.append(joint_q)
        if conjunctive:
            for es in target_ents[:3]:
                if len(es) >= 6 and es not in queries:
                    queries.append(es)
        if not queries and len(u_bytes) >= 6:
            queries.append(u_bytes)

        raw_pool: Dict[bytes, Tuple[float, str]] = {}
        for q_b in queries:
            k_q = (40 if q_b == joint_q else 24) if conjunctive else 6
            for cart in self.hip.cartridges:
                for sc, src, ep_b in cart.query(q_b, top_k=k_q):
                    prev = raw_pool.get(ep_b)
                    if prev is None or sc > prev[0]:
                        raw_pool[ep_b] = (sc, src)

        if not raw_pool:
            empty["hop23_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return empty

        if not conjunctive:
            ranked_raw = sorted(raw_pool.items(), key=lambda x: x[1][0], reverse=True)[:top_k_agenda]
            agenda = []
            for ep_b, (sc, src) in ranked_raw:
                cl = trim_utf8_safe(ep_b.strip(), 240)
                if len(cl) < 12:
                    continue
                cl_m = self.mc.encode(np.frombuffer(cl, dtype=np.uint8)).tolist()
                mm = 0
                for ei, es in enumerate(target_ents):
                    if es in cl:
                        mm |= (1 << ei)
                agenda.append((sc, mm, 0, src, cl, cl_m))
        else:
            node_proto = node.ans_concept_proto if node is not None else np.zeros(self.mc.thought_dim, dtype=np.float32)
            intent_v = 0.65 * h_qa + 0.35 * node_proto
            ni = float(np.linalg.norm(intent_v))
            if ni > 1e-6:
                intent_v = (intent_v / ni).astype(np.float32)

            scored_clauses: List[Tuple[float, int, int, str, bytes, List[int]]] = []
            seen_cl = set()
            for ep_b, (raw_sc, src) in raw_pool.items():
                is_cust = src.startswith("custom")
                clauses = self._split_evidence_clauses(ep_b)
                if not clauses:
                    fb = trim_utf8_safe(ep_b.strip(), 240)
                    clauses = [fb] if len(fb) >= 12 else []
                for cl in clauses:
                    if cl in seen_cl:
                        continue
                    seen_cl.add(cl)
                    exact_mask = 0
                    part_mask = 0
                    matched_cnt = 0.0
                    min_ent_pos = 9999
                    for ei, es in enumerate(target_ents):
                        pos_e = cl.find(es)
                        if pos_e >= 0:
                            exact_mask |= (1 << ei)
                            matched_cnt += 1.0
                            if pos_e < min_ent_pos:
                                min_ent_pos = pos_e
                        elif len(es) >= 9:
                            p1 = cl.find(trim_utf8_safe(es, 6))
                            p2 = cl.find(es[-6:])
                            if p1 >= 0 or p2 >= 0:
                                part_mask |= (1 << ei)
                                matched_cnt += 0.60
                                p_best = p1 if (p1 >= 0 and (p2 < 0 or p1 < p2)) else p2
                                if p_best < min_ent_pos:
                                    min_ent_pos = p_best
                    if matched_cnt < 0.60 and not (is_cust and raw_sc > 0.5):
                        continue

                    cl_m = self.mc.encode(np.frombuffer(cl, dtype=np.uint8)).tolist()
                    c_vecs = [
                        self.mc.emb_np[m] * float(self.rc.cg_vec[m])
                        for m in cl_m
                        if m >= 256 and self.rc.mid_to_role[m] < 0 and self.rc.cg_vec[m] >= self.rc.content_cg_thresh
                        and float(np.dot(self.mc.emb_np[m], self.mc.emb_np[m])) > 0.1
                    ]
                    cos_int = 0.0
                    if c_vecs and ni > 1e-6:
                        cv = np.sum(c_vecs, axis=0)
                        nc = float(np.linalg.norm(cv))
                        if nc > 1e-6:
                            cos_int = float(np.dot(cv / nc, intent_v))

                    cov_frac = matched_cnt / max(1, n_ents)
                    and_mult = 1.0 + 2.8 * cov_frac + (2.5 if matched_cnt >= 1.9 else 0.0)
                    subj_bonus = 2.0 if min_ent_pos <= 15 else (0.85 if min_ent_pos <= 45 else 0.0)
                    cust_mult = 2.4 if is_cust else 1.0
                    len_bonus = 1.0 if 36 <= len(cl) <= 220 else 0.78
                    tot_sc = (math.log1p(max(0.0, raw_sc)) * and_mult + 2.4 * max(0.0, cos_int) + subj_bonus) * cust_mult * len_bonus
                    scored_clauses.append((tot_sc, exact_mask, part_mask, src, cl, cl_m))

            if not scored_clauses:
                empty["hop23_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                return empty

            scored_clauses.sort(key=lambda x: x[0], reverse=True)
            agenda = [scored_clauses[0]]
            covered_mask = scored_clauses[0][1] | scored_clauses[0][2]

            if top_k_agenda > 1 and len(scored_clauses) > 1:
                uncovered_cands = [it for it in scored_clauses[1:] if ((it[1] | it[2]) & (~covered_mask)) != 0]
                if uncovered_cands:
                    agenda.append(uncovered_cands[0])
                    covered_mask |= (uncovered_cands[0][1] | uncovered_cands[0][2])
                else:
                    for it in scored_clauses[1:]:
                        if it[3].startswith("custom") and it[0] >= agenda[0][0] * 0.50:
                            agenda.append(it)
                            break

        if not agenda:
            empty["hop23_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return empty

        exact_all = 0
        part_all = 0
        for it in agenda:
            exact_all |= it[1]
            part_all |= it[2]
        exact_ents = [target_ents[i].decode("utf-8", errors="ignore") for i in range(n_ents) if exact_all & (1 << i)]
        part_ents = [target_ents[i].decode("utf-8", errors="ignore") for i in range(n_ents) if (part_all & (1 << i)) and not (exact_all & (1 << i))]
        uncov_ents = [target_ents[i].decode("utf-8", errors="ignore") for i in range(n_ents) if not ((exact_all | part_all) & (1 << i))]

        ep_concepts: Dict[int, float] = {}
        ep_trans1: Dict[int, Dict[int, float]] = {}
        ep_trans2: Dict[Tuple[int, int], Dict[int, float]] = {}
        ep_4grams = set()
        h_ev_l2 = np.zeros(self.mc.thought_dim, dtype=np.float32)
        has_custom = any(x[3].startswith("custom") for x in agenda)
        top_sc = max(agenda[0][0], 1e-5)

        for rk, (sc, _, _, src, cl, cl_m) in enumerate(agenda):
            ew = (sc / top_sc) * (1.0 / (1.0 + 0.20 * rk)) * (1.5 if src.startswith("custom") else 1.0)
            for i, mid in enumerate(cl_m):
                if mid >= 256:
                    cg = float(self.rc.cg_vec[mid])
                    if cg >= 0.25:
                        ep_concepts[mid] = max(ep_concepts.get(mid, 0.0), ew * (0.4 + 0.6 * cg))
                        em = self.mc.emb_np[mid]
                        if float(np.dot(em, em)) > 0.1:
                            h_ev_l2 += (ew * cg * float(self.mc.motor_idfs[mid])) * em
                if i >= 1:
                    d1 = ep_trans1.setdefault(cl_m[i - 1], {})
                    d1[mid] = d1.get(mid, 0.0) + ew
                if i >= 2:
                    d2 = ep_trans2.setdefault((cl_m[i - 2], cl_m[i - 1]), {})
                    d2[mid] = d2.get(mid, 0.0) + ew * 1.4
                if i >= 3:
                    ep_4grams.add((cl_m[i - 3], cl_m[i - 2], cl_m[i - 1], mid))

        ne = float(np.linalg.norm(h_ev_l2))
        if ne > 1e-6:
            h_ev_l2 /= ne
        ep_top_motors = sorted(ep_concepts.items(), key=lambda x: x[1], reverse=True)[:48]
        epi_conf = min(1.0, math.tanh(top_sc / 8.0))

        return {
            "agenda_bytes": [x[4] for x in agenda],
            "agenda_texts": [x[4].decode("utf-8", errors="ignore") for x in agenda],
            "agenda_sources": [x[3] for x in agenda],
            "agenda_scores": [round(x[0], 2) for x in agenda],
            "exact_entities": exact_ents,
            "partial_entities": part_ents,
            "uncovered_entities": uncov_ents,
            "ep_concepts": ep_concepts,
            "ep_trans1": ep_trans1,
            "ep_trans2": ep_trans2,
            "ep_4grams": ep_4grams,
            "ep_top_motors": ep_top_motors,
            "h_ev_l2": h_ev_l2,
            "epi_conf": round(epi_conf, 3),
            "has_custom": has_custom,
            "hop23_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    @torch.no_grad()
    def respond_with_reasoning(
        self,
        user_prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
        max_motor_steps: int = 36,
        beam_width: int = 3,
        top_k_motor: int = 12,
        frontier_cap: int = 24,
        escape_k: int = 3,
        use_reasoning_node: bool = True,
        use_conjunctive_hip: bool = True,
        goal_z_beta: float = 0.14,
        base_engram_mult: float = 0.48,
        sft_engram_scale: float = 1.15,
        score_critic: bool = True,
    ) -> Tuple[str, bool, Dict]:
        t0 = time.perf_counter()
        dev = self.device
        mc = self.mc
        rc = self.rc
        runner = self.runner

        pref_buf = bytearray()
        ctx_motor_list: List[int] = []
        if history:
            for hu, ha in history[-2:]:
                hub, hab = hu.strip().encode("utf-8"), ha.strip().encode("utf-8")
                pref_buf.extend(TURN_U_PREFIX + hub + TURN_A_PREFIX + hab + bytes([TURN_EOT_BYTE]))
                ctx_motor_list.extend(mc.encode(np.frombuffer(hub, dtype=np.uint8)).tolist())
                ctx_motor_list.extend(mc.encode(np.frombuffer(hab, dtype=np.uint8)).tolist())
        u_bytes = user_prompt.strip().encode("utf-8")
        pref_buf.extend(TURN_U_PREFIX + u_bytes + TURN_A_PREFIX)
        frame_prefix = bytes(pref_buf)
        u_motors = mc.encode(np.frombuffer(u_bytes, dtype=np.uint8))
        ctx_motor_list.extend(u_motors.tolist())
        ctx_motors_arr = np.array(ctx_motor_list, dtype=np.int32)

        entity_spans, entity_mids = self.rgraph.extract_entity_spans(mc, rc, self.adapter, user_prompt)
        h_qa = self.adapter.predict_qa_intent(rc, ctx_motors_arr)

        if use_reasoning_node:
            node, node_sim = self.rgraph.route(mc, rc, self.adapter, u_motors, ctx_motors_arr)
            has_custom_mounted = any(c.name.startswith("custom") for c in self.hip.cartridges) if self.hip else False
            want_hip = bool(len(entity_spans) > 0 and (node.hip_need_rate >= 0.35 or has_custom_mounted))
            top_k_ag = 2 if (node.multi_entity_rate >= 0.30 or len(entity_spans) >= 2) else 1
        else:
            node, node_sim = None, 0.0
            want_hip = len(entity_spans) > 0
            top_k_ag = 1

        delib = self._conjunctive_hippocampus_deliberate(
            entity_spans, u_bytes, node, h_qa, top_k_agenda=top_k_ag, conjunctive=use_conjunctive_hip
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

        wm_table: Dict[bytes, Dict[int, float]] = {}
        if has_evidence:
            wm_lens = (3, 4, 5, 6, 8, 10)
            wm_w = {3: 0.45, 4: 0.75, 5: 1.05, 6: 1.35, 8: 1.75, 10: 2.10}
            for ag_i, ag_b in enumerate(delib["agenda_bytes"]):
                src_mult = 1.45 if delib["agenda_sources"][ag_i].startswith("custom") else 1.0
                n_ag = len(ag_b)
                for p_i in range(3, n_ag):
                    nxt_b = ag_b[p_i]
                    for wl in wm_lens:
                        if p_i >= wl:
                            suf = ag_b[p_i - wl : p_i]
                            d_s = wm_table.setdefault(suf, {})
                            d_s[nxt_b] = d_s.get(nxt_b, 0.0) + wm_w[wl] * src_mult

        h_root = mc.compute_root_thought_vec(np.array(entity_mids, dtype=np.int32)) if entity_mids else mc.compute_root_thought_vec(u_motors)
        node_proto = node.ans_concept_proto if node is not None else np.zeros(mc.thought_dim, dtype=np.float32)
        h_plan_l2 = (0.45 * h_qa + 0.55 * delib["h_ev_l2"]) if has_evidence else (0.75 * h_qa + 0.25 * node_proto)
        np_l2 = float(np.linalg.norm(h_plan_l2))
        if np_l2 > 1e-6:
            h_plan_l2 = (h_plan_l2 / np_l2).astype(np.float32)

        goal_z = None
        if goal_z_beta > 0.0:
            gz_l2 = self.rgraph.project_to_goal_z(h_plan_l2, dev)
            gz_ev = runner.encode_text_h(b"\n".join(delib["agenda_bytes"])) if has_evidence else None
            if gz_ev is not None and self.rgraph.y_l1_mean is not None:
                ym_t = torch.from_numpy(self.rgraph.y_l1_mean).to(dev)
                gz_ev = gz_ev - ym_t
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

        st1 = runner.init_batch_state(1)
        if has_evidence:
            replay_safe = trim_utf8_safe(b"\n".join(delib["agenda_bytes"]), 420)
            replay_b = list(replay_safe)
            for ri in range(len(replay_b) - 1):
                rx = torch.tensor([replay_b[ri]], dtype=torch.int64, device=dev)
                ry = torch.tensor([replay_b[ri + 1]], dtype=torch.int64, device=dev)
                runner.step_batch(st1, rx, next_true=ry, goal_z=None, goal_z_beta=0.0, use_engram=False, is_prompt=True)
            st1["byte_hist"].fill_(-1)
            st1["hist_ptr"].zero_()
            st1["node_delay"].fill_(-1)
            st1["nd_ptr"].zero_()
            st1["c_eng"].zero_()

        r_scale = self.adapter.readout_scale
        wm_scale = (1.95 if has_custom else 1.25) * epi_conf
        wm_cap = 4.6 if has_custom else 2.6

        def _step_v90(st_in, x_in, nt_in, is_p: bool, ans_byte_pos: int = 0) -> torch.Tensor:
            beta_eff = (goal_z_beta * math.exp(-ans_byte_pos / 42.0)) if (not is_p and goal_z is not None and goal_z_beta > 0.0) else 0.0
            lg, _ = runner.step_batch(
                st_in, x_in, next_true=None, locked_mode_p=None,
                goal_z=(goal_z if beta_eff > 1e-4 else None), goal_z_beta=beta_eff,
                use_engram=False, is_prompt=is_p,
            )
            h_act = st_in["last_h_act"]
            if not is_p:
                W_g, b_g = self.adapter._ensure_gpu_adapter(dev)
                if W_g is not None:
                    lg = lg + (h_act @ W_g) * r_scale + b_g.unsqueeze(0)
                B_cur = x_in.shape[0]
                ar = torch.arange(B_cur, device=dev)
                prev_idx = (st_in["hist_ptr"] - 1) % runner.H_buf
                bs24 = st_in["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - runner._ar24) % runner.H_buf]
                bs24_np = bs24.to(torch.int32).cpu().numpy()
                keys_np = self.v20.rack.hash_bytes24_batch(bs24_np)
                cn = st_in["c_eng"] / st_in["c_eng"].norm(dim=1, keepdim=True).clamp(min=1e-5)
                cq_np = torch.round(cn * 127.0).clamp(-127, 127).to(torch.int8).cpu().numpy()
                lp_c = torch.log_softmax(lg, dim=-1)
                p_c = torch.exp(lp_c)
                H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
                ent_g = H_c / (1.0 + H_c)
                if len(self.v20.rack.cartridges) > 0:
                    eb_np, _, _ = self.v20.rack.lookup_batch(keys_np, cq_np, use_context_gate=True)
                    lg = lg + torch.from_numpy(eb_np).to(dev) * (base_engram_mult * ent_g)
                if self.adapter.sft_engram is not None:
                    s_np = np.zeros((B_cur, 256), dtype=np.float32)
                    self.adapter.sft_engram.lookup_into(keys_np, cq_np, s_np, use_context_gate=True)
                    np.minimum(s_np, 2.8, out=s_np)
                    lg = lg + torch.from_numpy(s_np).to(dev) * (sft_engram_scale * ent_g)
                if wm_table:
                    wm_np = np.zeros((B_cur, 256), dtype=np.float32)
                    has_wm_hit = False
                    for ri_b in range(B_cur):
                        row_raw = bs24_np[ri_b, :10]
                        valid_n = int(np.sum(row_raw >= 0))
                        if valid_n >= 3:
                            chr_b = bytes(int(row_raw[k]) for k in range(valid_n - 1, -1, -1))
                            for wl in (3, 4, 5, 6, 8, 10):
                                if len(chr_b) >= wl:
                                    d_hit = wm_table.get(chr_b[-wl:])
                                    if d_hit:
                                        has_wm_hit = True
                                        for nb_k, w_k in d_hit.items():
                                            wm_np[ri_b, nb_k] += w_k
                    if has_wm_hit:
                        np.clip(wm_np, 0.0, wm_cap, out=wm_np)
                        lg = lg + torch.from_numpy(wm_np).to(dev) * (wm_scale * ent_g)
            elif nt_in is not None:
                from neurograph.memory_v3b import _mix64
                core = self.v20.core
                cfg = core.cfg
                B_cur = x_in.shape[0]
                ar = torch.arange(B_cur, device=dev)
                prev_idx = (st_in["hist_ptr"] - 1) % runner.H_buf
                bs7 = st_in["byte_hist"][ar.unsqueeze(1), (prev_idx.unsqueeze(1) - runner._ar24[:, :7]) % runner.H_buf]
                h4 = _mix64(bs7[:, 0] + 17)
                h4 = _mix64(h4 ^ ((bs7[:, 1] + 31) * 1315423911))
                h4 = _mix64(h4 ^ ((bs7[:, 2] + 53) * 2654435761))
                h4 = _mix64(h4 ^ ((bs7[:, 3] + 79) * 1000003))
                k4 = torch.where((bs7[:, 3] >= 0) & (h4 != 0), h4, torch.zeros_like(h4))
                h7 = _mix64(h4 ^ ((bs7[:, 4] + 97) * 1000033))
                h7 = _mix64(h7 ^ ((bs7[:, 5] + 113) * 1000037))
                h7 = _mix64(h7 ^ ((bs7[:, 6] + 131) * 1000039))
                k7 = torch.where((bs7[:, 6] >= 0) & (h7 != 0), h7, torch.zeros_like(h7))
                valid_nt = (nt_in >= 0) & (st_in["step_i"] >= 5) & (k4 != 0)
                lp_s = torch.log_softmax(lg, dim=-1)
                surp = 1.0 - torch.exp(lp_s[ar, nt_in.clamp(min=0)])
                do_b = valid_nt & (surp > cfg.stm_surp_thresh)
                sp = st_in["stm_ptr"]
                st_in["stm_k4"][ar, sp] = torch.where(do_b, k4, st_in["stm_k4"][ar, sp])
                st_in["stm_k7"][ar, sp] = torch.where(do_b, k7, st_in["stm_k7"][ar, sp])
                st_in["stm_val"][ar, sp] = torch.where(do_b, nt_in.clamp(min=0), st_in["stm_val"][ar, sp])
                st_in["stm_str"][ar, sp] = torch.where(do_b, torch.ones_like(surp), st_in["stm_str"][ar, sp])
                st_in["stm_ptr"] = torch.where(do_b, (sp + 1) % cfg.stm_slots, sp)
            return lg

        raw_in = list(frame_prefix)
        logits1 = None
        is_social = (not has_evidence) and (len(entity_spans) == 0 or (node is not None and node.hip_need_rate < 0.35))
        for i, b in enumerate(raw_in):
            x_t = torch.tensor([b], dtype=torch.int64, device=dev)
            is_last = (i == len(raw_in) - 1)
            nt_u = torch.tensor([raw_in[i + 1]], dtype=torch.int64, device=dev) if (not is_last and is_social) else None
            logits1 = _step_v90(st1, x_t, nt_u, is_p=(not is_last))

        st_beams = st1
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
        elif node is not None:
            min_eot_b = int(max(36, min(66, node.p25_ans_bytes * 0.65)))
        else:
            min_eot_b = 42

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
                    eot_prior = self.adapter.eot_after_motor.get(last_gm, 0.0)
                    nb_e = bm["n_bytes"] + 1
                    nm_e = len(bm["gen_motors"]) + 1
                    eot_l2 = 1.8 * math.sqrt(eot_prior) if eot_prior > 0.0 else 0.0
                    len_fin = (0.30 * math.sqrt(eot_prior)) if (bm["n_bytes"] >= min_eot_b and eot_prior >= 0.04) else 0.0
                    eot_sc = (bm["raw_l1_lp"] + eot_lp - bm["tot_inh"]) / (nb_e ** 0.90) + 0.30 * ((bm["raw_l2_sc"] + eot_l2) / (nm_e ** 0.85)) - 0.18 * bm["tot_rep"] + len_fin
                    if eot_lp > -2.8:
                        done_bm = dict(bm)
                        done_bm["done"] = True
                        done_bm["score"] = eot_sc
                        done_bm["eot_lp"] = round(eot_lp, 3)
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
                        if _last_b == 10 or _bm["n_bytes"] < 54:
                            l2_b -= 3.2
                        else:
                            l2_b -= 0.9
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
                            cos_plan = float(np.dot(em, h_plan_l2))
                            l2_g += cg * 1.45 * cos_plan
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
                        openers = (node.top_openers if (node is not None and node.top_openers) else self.adapter.sft_openers[:8])
                        for om, ow in openers[:8]:
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
                    if last2 in self.adapter.sft_m2:
                        for nm, nw in self.adapter.sft_m2[last2].items():
                            sft_cands[nm] = max(sft_cands.get(nm, 0.0), 1.55 * nw)
                    if last1 in self.adapter.sft_m1:
                        for nm, nw in self.adapter.sft_m1[last1].items():
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
                    lrow[~runner.lead_mask] = -1e9
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
                st_esc = runner.gather_state(st_beams, j_parents)
                cur_bs = torch.tensor([x[1] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                esc_bufs = [bytearray([x[1]]) for x in mb_escape_jobs]
                esc_clens = [x[2] for x in mb_escape_jobs]
                beta_esc = (goal_z_beta * math.exp(-int(beams[0]["n_bytes"]) / 42.0)) if (goal_z is not None and goal_z_beta > 0.0 and beams) else 0.0
                for ci_u in range(1, max(esc_clens)):
                    act_e = [ji for ji in range(J) if ci_u < esc_clens[ji]]
                    if not act_e:
                        break
                    if len(act_e) == J:
                        lg_e, _ = runner.step_batch(st_esc, cur_bs, goal_z=(goal_z if beta_esc > 1e-4 else None), goal_z_beta=beta_esc, use_engram=False, is_prompt=False)
                    else:
                        act_et = torch.tensor(act_e, dtype=torch.int64, device=dev)
                        st_esub = runner.gather_state(st_esc, act_et)
                        lg_e, _ = runner.step_batch(st_esub, cur_bs[act_et], goal_z=(goal_z if beta_esc > 1e-4 else None), goal_z_beta=beta_esc, use_engram=False, is_prompt=False)
                        for k_st in st_esc.keys():
                            st_esc[k_st][act_et] = st_esub[k_st]
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
            st_c = runner.gather_state(st_beams, p_idx)
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
                if len(act) == M:
                    lp_act = torch.log_softmax(log_c, dim=-1)
                    yb_t = torch.tensor([cand_bytes[i][bi] for i in range(M)], dtype=torch.int64, device=dev)
                    step_lp = lp_act[torch.arange(M, device=dev), yb_t]
                    cand_l1_lp += step_lp
                    is_lead = (yb_t & 0xC0) != 0x80
                    cand_min_clp = torch.where(is_lead & cand_has_chr, torch.minimum(cand_min_clp, cand_cur_clp), cand_min_clp)
                    cand_cur_clp = torch.where(is_lead, step_lp, cand_cur_clp + step_lp)
                    cand_has_chr = cand_has_chr | is_lead
                    cur_ans_pos = int(beams[0]["n_bytes"]) + bi if beams else bi
                    log_c = _step_v90(st_c, yb_t, None, is_p=False, ans_byte_pos=cur_ans_pos)
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
                    st_sub = runner.gather_state(st_c, act_t)
                    cur_ans_pos = int(beams[0]["n_bytes"]) + bi if beams else bi
                    log_sub = _step_v90(st_sub, yb_t, None, is_p=False, ans_byte_pos=cur_ans_pos)
                    for k_st in st_c.keys():
                        st_c[k_st][act_t] = st_sub[k_st]
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
                kept.append(nbm)
                kept_ci.append(ci)
                if len(kept) >= beam_width:
                    break
            if not kept:
                break
            k_t = torch.tensor(kept_ci, dtype=torch.int64, device=dev)
            st_beams = runner.gather_state(st_c, k_t)
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

        bpc = runner.score_text_bpc(frame_prefix.decode("utf-8", errors="ignore") + ans_str, prompt_prefix_len=len(frame_prefix)) if score_critic else -1.0
        return ans_str, valid_utf8, {
            "reasoning_node_id": node.node_id if node is not None else -1,
            "reasoning_node_sim": round(node_sim, 3),
            "node_hip_need": node.hip_need_rate if node is not None else -1.0,
            "node_multi_ent": node.multi_entity_rate if node is not None else -1.0,
            "want_hip": want_hip,
            "node_min_eot_b": min_eot_b,
            "entities": [e.decode("utf-8", errors="ignore") for e in entity_spans],
            "exact_entities": delib["exact_entities"],
            "partial_entities": delib["partial_entities"],
            "uncovered_entities": delib["uncovered_entities"],
            "agenda_sources": delib["agenda_sources"],
            "agenda_scores": delib["agenda_scores"],
            "agenda_preview": [t[:75].replace("\n", " ") for t in delib["agenda_texts"]],
            "hop23_ms": delib["hop23_ms"],
            "stopped_eot": bool(best.get("done", False)),
            "gen_chars": len(ans_str),
            "decode_sec": round(decode_sec, 2),
            "chars_per_sec": round(len(ans_str) / max(decode_sec, 1e-3), 2),
            "fluency_bpc": bpc,
        }
