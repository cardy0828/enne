"""本体 DAG と外部記憶をまとめた統合モデル（v6.2）。"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from neurograph.episodic_v21 import EpisodicRack
from neurograph.role_cortex_v60 import RoleBoundThoughtCortexV60
from neurograph.thought_cortex_v50 import HierarchicalBrainV50


def _longest_common_substring_chars(gen_text: str, episodes: List[str]) -> int:
    if not gen_text or not episodes:
        return 0
    best = 0
    g_len = len(gen_text)
    for ep in episodes:
        if not ep:
            continue
        for i in range(g_len):
            if g_len - i <= best:
                break
            hi = best + 1
            while i + hi <= g_len and gen_text[i : i + hi] in ep:
                best = hi
                hi += 1
    return best


class UnifiedCorticoHippocampalBrainV62:

    def __init__(
        self,
        brain_l12: HierarchicalBrainV50,
        role_cortex: RoleBoundThoughtCortexV60,
        hippocampus: Optional[EpisodicRack] = None,
    ):
        self.brain = brain_l12
        self.mc = brain_l12.mc
        self.rc = role_cortex
        self.hip = hippocampus

    def _extract_hippocampal_grounding(
        self,
        prompt_bytes: bytes,
        top_k_ep: int = 5,
    ) -> Dict[str, object]:
        empty = {
            "episodes_txt": [],
            "ep_scores": [],
            "ep_sources": [],
            "ep_concepts": {},
            "ep_trans1": {},
            "ep_trans2": {},
            "ep_4grams": set(),
            "ep_top_motors": [],
            "h_epi": np.zeros(self.mc.thought_dim, dtype=np.float32),
            "epi_conf": 0.0,
            "has_custom": False,
            "query_ms": 0.0,
        }
        if self.hip is None or self.hip.total_episodes == 0 or len(prompt_bytes) < 6:
            return empty

        t0 = time.perf_counter()
        hits: List[Tuple[float, str, bytes]] = []
        for cart in self.hip.cartridges:
            hits.extend(cart.query(prompt_bytes, top_k=top_k_ep))
        hits.sort(key=lambda x: x[0], reverse=True)
        hits = hits[:top_k_ep]
        q_ms = (time.perf_counter() - t0) * 1000.0
        if not hits:
            empty["query_ms"] = round(q_ms, 2)
            return empty

        mc = self.mc
        cg_v = self.rc.cg_vec
        top_sc = max(hits[0][0], 1e-5)

        ep_concepts: Dict[int, float] = {}
        ep_trans1: Dict[int, Dict[int, float]] = {}
        ep_trans2: Dict[Tuple[int, int], Dict[int, float]] = {}
        ep_4grams: Set[Tuple[int, int, int, int]] = set()
        episodes_txt: List[str] = []
        ep_scores: List[float] = []
        ep_sources: List[str] = []
        has_custom = False

        h_epi = np.zeros(mc.thought_dim, dtype=np.float32)

        for rank_i, (sc, src, ep_b) in enumerate(hits):
            if sc < max(2.5, top_sc * 0.35):
                continue
            ep_txt = ep_b.decode("utf-8", errors="ignore")
            episodes_txt.append(ep_txt)
            ep_scores.append(round(sc, 2))
            ep_sources.append(src)
            if src.startswith("custom"):
                has_custom = True

            ew = (sc / top_sc) * (1.0 / (1.0 + 0.25 * rank_i))
            if src.startswith("custom"):
                ew *= 1.6

            m_ids = mc.encode(np.frombuffer(ep_b, dtype=np.uint8)).tolist()
            L = len(m_ids)
            for i, mid in enumerate(m_ids):
                if mid >= 256:
                    cg = float(cg_v[mid])
                    if cg >= 0.25:
                        ep_concepts[mid] = max(ep_concepts.get(mid, 0.0), ew * (0.4 + 0.6 * cg))
                        em = mc.emb_np[mid]
                        if float(np.dot(em, em)) > 0.1:
                            h_epi += (ew * cg * float(mc.motor_idfs[mid])) * em
                if i >= 1:
                    p1 = m_ids[i - 1]
                    d1 = ep_trans1.setdefault(p1, {})
                    d1[mid] = d1.get(mid, 0.0) + ew
                if i >= 2:
                    p2 = (m_ids[i - 2], m_ids[i - 1])
                    d2 = ep_trans2.setdefault(p2, {})
                    d2[mid] = d2.get(mid, 0.0) + ew * 1.4
                if i >= 3:
                    ep_4grams.add((m_ids[i - 3], m_ids[i - 2], m_ids[i - 1], mid))

        nrm = float(np.linalg.norm(h_epi))
        if nrm > 1e-6:
            h_epi /= nrm

        ep_top_motors = sorted(ep_concepts.items(), key=lambda x: x[1], reverse=True)[:48]
        epi_conf = min(1.0, math.tanh(top_sc / 28.0)) if episodes_txt else 0.0

        return {
            "episodes_txt": episodes_txt,
            "ep_scores": ep_scores,
            "ep_sources": ep_sources,
            "ep_concepts": ep_concepts,
            "ep_trans1": ep_trans1,
            "ep_trans2": ep_trans2,
            "ep_4grams": ep_4grams,
            "ep_top_motors": ep_top_motors,
            "h_epi": h_epi,
            "epi_conf": round(epi_conf, 3),
            "has_custom": has_custom,
            "query_ms": round(q_ms, 2),
        }

    @torch.no_grad()
    def generate_unified_beam(
        self,
        prompt: str,
        max_motor_steps: int = 28,
        beam_width: int = 6,
        top_k_motor: int = 16,
        frontier_cap: int = 48,
        escape_k: int = 4,
        score_critic: bool = True,
        use_hippocampus: bool = True,
        use_syntax_gate: bool = True,
        anti_copy_damper: float = 2.2,
        topic_ppmi_gain: float = 1.65,
        topic_align_gain: float = 1.00,
        role_rank_gain: float = 1.35,
        epi_concept_gain: float = 1.85,
        inhib_gain: float = 1.35,
        inhib_thresh: float = 0.24,
        rhythm_penalty: float = 0.55,
    ) -> Tuple[str, bool, Dict]:
        t0 = time.perf_counter()
        dev = self.brain.device
        runner = self.brain.runner
        mc = self.mc
        rc = self.rc
        prompt_b = prompt.encode("utf-8")
        if not prompt_b:
            return "", True, {}

        p_motors = mc.encode(np.frombuffer(prompt_b, dtype=np.uint8))
        prompt_m = p_motors.tolist()

        epi = self._extract_hippocampal_grounding(prompt_b, top_k_ep=5) if use_hippocampus else {
            "episodes_txt": [], "ep_scores": [], "ep_sources": [],
            "ep_concepts": {}, "ep_trans1": {}, "ep_trans2": {}, "ep_4grams": set(),
            "ep_top_motors": [], "h_epi": np.zeros(mc.thought_dim, dtype=np.float32),
            "epi_conf": 0.0, "has_custom": False, "query_ms": 0.0,
        }
        epi_conf = float(epi["epi_conf"])
        ep_concepts: Dict[int, float] = epi["ep_concepts"]
        ep_trans1: Dict[int, Dict[int, float]] = epi["ep_trans1"]
        ep_trans2: Dict[Tuple[int, int], Dict[int, float]] = epi["ep_trans2"]
        ep_4grams: Set[Tuple[int, int, int, int]] = epi["ep_4grams"]
        has_custom = bool(epi["has_custom"])

        root_t_ids, root_t_w = mc.extract_active_topics(p_motors, max_topics=6, min_idf=5.8, use_resonance=True)
        h_root = mc.compute_root_thought_vec(p_motors)
        h_epi = epi["h_epi"]
        if epi_conf > 0.25 and float(np.linalg.norm(h_epi)) > 0.1:
            if float(np.linalg.norm(h_root)) > 0.1:
                h_anchor = (1.0 - 0.42 * epi_conf) * h_root + (0.42 * epi_conf) * h_epi
            else:
                h_anchor = h_epi.copy()
            an = float(np.linalg.norm(h_anchor))
            if an > 1e-6:
                h_anchor /= an
        else:
            h_anchor = h_root

        root_mass = sum(
            float(rc.cg_vec[int(m)])
            for m in prompt_m
            if int(m) >= 256 and float(np.dot(mc.emb_np[int(m)], mc.emb_np[int(m)])) > 0.1
        )
        root_conf = min(1.0, max(root_mass / 1.6, epi_conf))
        has_anchor = bool(float(np.linalg.norm(h_anchor)) > 0.1 and root_conf > 0.25)

        p_role_pred = rc.predict_next_concept_vec(p_motors, mode="joint")
        p_role_events = rc.extract_role_events(p_motors)

        st1 = runner.init_batch_state(1)
        prompt_hs = []
        raw_in = list(prompt_b)
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

        st_beams = st1
        logits_beams = logits1
        bytes_to_mid: Dict[bytes, int] = {b: i for i, b in enumerate(mc.motor_bytes)}
        beams = [{
            "score": 0.0,
            "raw_l1_lp": 0.0,
            "raw_l2_sc": 0.0,
            "tot_inh": 0.0,
            "tot_rep": 0.0,
            "n_bytes": 0,
            "motor_seq": list(prompt_m),
            "gen_motors": [],
            "gen_bytes": bytearray(),
            "seen_m3": set(tuple(prompt_m[i : i + 3]) for i in range(max(0, len(prompt_m) - 2))),
            "seen_m2": set(),
            "used_content": set(int(m) for m in root_t_ids.tolist()),
            "consec_hi": 0.0,
        }]

        for _ in range(max_motor_steps):
            cand_parent, cand_mid, cand_l2_base, cand_l2_gated, cand_inh_b, cand_bytes, cand_cg, cand_keep_prio, cand_is_l1_esc = [], [], [], [], [], [], [], [], []
            mb_escape_jobs = []

            for wi, bm in enumerate(beams):
                m_arr = np.array(bm["motor_seq"][-32:], dtype=np.int32)
                live_role_pred = rc.predict_next_concept_vec(m_arr, mode="joint")
                role_pred_vec = 0.60 * p_role_pred + 0.40 * live_role_pred
                rn = float(np.linalg.norm(role_pred_vec))
                if rn > 1e-6:
                    role_pred_vec /= rn

                rec_ids, rec_w = mc.extract_active_topics(m_arr, max_topics=4, min_idf=5.2)
                t_map: Dict[int, float] = {int(k): float(v) for k, v in zip(root_t_ids.tolist(), root_t_w.tolist())}
                for dk, dv in zip(rec_ids.tolist(), rec_w.tolist()):
                    if has_anchor and dk >= 256:
                        if float(np.dot(mc.emb_np[dk], h_anchor)) < 0.22 and dk not in ep_concepts:
                            continue
                    t_map[int(dk)] = max(t_map.get(int(dk), 0.0), float(dv) * 0.35)
                if epi_conf > 0.25:
                    for emid, ew in epi["ep_top_motors"][:3]:
                        t_map[emid] = max(t_map.get(emid, 0.0), 0.70 * epi_conf * min(1.0, ew))

                ranked_t = sorted(t_map.items(), key=lambda x: x[1], reverse=True)[:8]
                t_arr = np.array([m for m, _ in ranked_t], dtype=np.int32) if ranked_t else np.zeros(0, dtype=np.int32)
                w_arr = np.array([w for _, w in ranked_t], dtype=np.float32) if ranked_t else np.zeros(0, dtype=np.float32)

                p_ids, p_sc = mc.predict_topk(
                    np.array(bm["motor_seq"][-6:], dtype=np.int32),
                    t_arr, w_arr, topic_ppmi_gain=topic_ppmi_gain, top_k=top_k_motor * 3,
                )

                recent_16 = set(bm["gen_motors"][-16:])
                c_hi = bm["consec_hi"]
                added_bytes_wi = set()
                last1 = bm["motor_seq"][-1] if bm["motor_seq"] else -1
                last2 = (bm["motor_seq"][-2], bm["motor_seq"][-1]) if len(bm["motor_seq"]) >= 2 else (-1, -1)

                def _score_cand(mid_v: int, mb: bytes, base_l2: float, ep_trans_bonus: float = 0.0) -> Tuple[float, float, float, float]:
                    cg = float(rc.cg_vec[mid_v]) if 0 <= mid_v < mc.V_m else 0.0
                    l2_base = float(base_l2)
                    l2_gated = float(ep_trans_bonus)
                    inh_b = 0.0

                    if mid_v >= 256 and mid_v in recent_16:
                        l2_base -= (0.95 + 1.25 * cg)
                    if cg > 0.55 and c_hi >= 1.8:
                        l2_base -= rhythm_penalty * (c_hi - 1.5) * cg
                    elif cg < 0.25 and c_hi >= 2.1:
                        l2_base += 0.45

                    ep_w = ep_concepts.get(mid_v, 0.0) if epi_conf > 0.20 else 0.0
                    if ep_w > 0.0:
                        l2_gated += epi_conf * epi_concept_gain * math.tanh(ep_w)

                    if has_anchor and mid_v >= 256 and cg > 0.15:
                        em = mc.emb_np[mid_v]
                        if float(np.dot(em, em)) > 0.1:
                            cos_anc = float(np.dot(em, h_anchor))
                            dir_aff = mc.directed_affinity(mid_v, root_t_ids, root_t_w)
                            gate_align = cos_anc + 0.18 * math.tanh(dir_aff) + (0.35 * epi_conf * math.tanh(ep_w))
                            if gate_align < inhib_thresh:
                                mismatch = min(1.25, (inhib_thresh - gate_align) / max(0.05, inhib_thresh))
                                inh_b = root_conf * cg * inhib_gain * mismatch
                            elif epi_conf > 0.55 and ep_w == 0.0 and cg > 0.70 and cos_anc < 0.42:
                                inh_b = 0.45 * epi_conf * cg * (0.42 - cos_anc)

                            cos_role = float(np.dot(em, role_pred_vec))
                            disc_r = 0.0
                            for _, smid, op, scg in p_role_events[-6:]:
                                mp = rc.role_pred_table.get((smid, op))
                                if mp and mid_v in mp:
                                    disc_r += scg * mp[mid_v]
                            l2_base += cg * topic_align_gain * gate_align
                            l2_gated += cg * role_rank_gain * (cos_role + 0.25 * math.tanh(disc_r))
                    return l2_base, l2_gated, inh_b, cg

                taken = 0
                for mid_v, sc_v in zip(p_ids.tolist(), p_sc.tolist()):
                    mb = mc.motor_bytes[mid_v]
                    if mid_v < 256 and (mid_v < 32 or mid_v >= 127):
                        continue
                    if b"\n" in mb or b"\r" in mb or mb in added_bytes_wi:
                        continue
                    if last1 == mid_v:
                        continue
                    tr_b = 0.0
                    if epi_conf > 0.20:
                        if last2 in ep_trans2 and mid_v in ep_trans2[last2]:
                            tr_b += 1.35 * epi_conf * math.tanh(ep_trans2[last2][mid_v])
                        elif last1 in ep_trans1 and mid_v in ep_trans1[last1]:
                            tr_b += 0.85 * epi_conf * math.tanh(ep_trans1[last1][mid_v])
                    l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, sc_v, ep_trans_bonus=tr_b)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                    cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                    cand_cg.append(cg); cand_keep_prio.append(False); cand_is_l1_esc.append(False)
                    added_bytes_wi.add(mb)
                    taken += 1
                    if taken >= top_k_motor:
                        break

                if epi_conf > 0.20:
                    epi_cands: Dict[int, float] = {}
                    if last2 in ep_trans2:
                        for nm, w in ep_trans2[last2].items():
                            epi_cands[nm] = max(epi_cands.get(nm, 0.0), 1.8 * w)
                    if last1 in ep_trans1:
                        for nm, w in ep_trans1[last1].items():
                            epi_cands[nm] = max(epi_cands.get(nm, 0.0), 1.1 * w)
                    for emid, ew in epi["ep_top_motors"][:10]:
                        if emid not in bm["used_content"] and emid not in recent_16:
                            epi_cands[emid] = max(epi_cands.get(emid, 0.0), 0.75 * ew)

                    for mid_v, ew in sorted(epi_cands.items(), key=lambda x: x[1], reverse=True)[:8]:
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 and (mid_v < 32 or mid_v >= 127):
                            continue
                        if b"\n" in mb or b"\r" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        ep_inj = (1.6 if has_custom else 1.0) * epi_conf * math.tanh(ew)
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.15, ep_trans_bonus=ep_inj)
                        cand_parent.append(wi); cand_mid.append(mid_v); cand_l2_base.append(l2_b)
                        cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(mb)
                        cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                        added_bytes_wi.add(mb)

                if escape_k > 0:
                    lrow = logits_beams[wi].clone()
                    lrow[~runner.lead_mask] = -1e9
                    lrow[0] = -1e9; lrow[10] = -1e9; lrow[13] = -1e9
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

            if mb_escape_jobs:
                J = len(mb_escape_jobs)
                j_parents = torch.tensor([x[0] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                st_esc = runner.gather_state(st_beams, j_parents)
                cur_bs = torch.tensor([x[1] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                esc_bufs = [bytearray([x[1]]) for x in mb_escape_jobs]
                esc_clens = [x[2] for x in mb_escape_jobs]
                max_cl = max(esc_clens)
                for ci_u in range(1, max_cl):
                    act_e = [ji for ji in range(J) if ci_u < esc_clens[ji]]
                    if not act_e:
                        break
                    if len(act_e) == J:
                        lg_e, _ = runner.step_batch(
                            st_esc, cur_bs, locked_mode_p=locked_mode_p, use_engram=False, is_prompt=False
                        )
                    else:
                        act_et = torch.tensor(act_e, dtype=torch.int64, device=dev)
                        st_esub = runner.gather_state(st_esc, act_et)
                        lg_e, _ = runner.step_batch(
                            st_esub, cur_bs[act_et], locked_mode_p=locked_mode_p, use_engram=False, is_prompt=False
                        )
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
                    lg_e = lg_e.masked_fill(~mask_e, -1e9)
                    nxt_bs = lg_e.argmax(dim=-1)
                    nxt_list = nxt_bs.cpu().tolist()
                    for row_i, ji in enumerate(act_e):
                        nb_v = nxt_list[row_i]
                        esc_bufs[ji].append(nb_v)
                        cur_bs[ji] = nb_v

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
                reserved: List[int] = []
                seen_esc_beam = set()
                for i in range(len(cand_parent)):
                    if cand_is_l1_esc[i] and cand_parent[i] not in seen_esc_beam:
                        reserved.append(i)
                        seen_esc_beam.add(cand_parent[i])
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
                    close_prev = is_lead & cand_has_chr
                    cand_min_clp = torch.where(close_prev, torch.minimum(cand_min_clp, cand_cur_clp), cand_min_clp)
                    cand_cur_clp = torch.where(is_lead, step_lp, cand_cur_clp + step_lp)
                    cand_has_chr = cand_has_chr | is_lead
                    log_c, _ = runner.step_batch(
                        st_c, yb_t, locked_mode_p=locked_mode_p, use_engram=True, engram_mult=0.72, is_prompt=False
                    )
                else:
                    act_t = torch.tensor(act, dtype=torch.int64, device=dev)
                    lp_act = torch.log_softmax(log_c[act_t], dim=-1)
                    yb_t = torch.tensor([cand_bytes[i][bi] for i in act], dtype=torch.int64, device=dev)
                    step_lp = lp_act[torch.arange(len(act), device=dev), yb_t]
                    cand_l1_lp[act_t] += step_lp
                    is_lead = (yb_t & 0xC0) != 0x80
                    prev_has = cand_has_chr[act_t]
                    prev_cur = cand_cur_clp[act_t]
                    prev_min = cand_min_clp[act_t]
                    cand_min_clp[act_t] = torch.where(is_lead & prev_has, torch.minimum(prev_min, prev_cur), prev_min)
                    cand_cur_clp[act_t] = torch.where(is_lead, step_lp, prev_cur + step_lp)
                    cand_has_chr[act_t] = prev_has | is_lead
                    st_sub = runner.gather_state(st_c, act_t)
                    log_sub, _ = runner.step_batch(
                        st_sub, yb_t, locked_mode_p=locked_mode_p, use_engram=True, engram_mult=0.72, is_prompt=False
                    )
                    for k_st in st_c.keys():
                        st_c[k_st][act_t] = st_sub[k_st]
                    log_c[act_t] = log_sub

            cand_min_clp = torch.where(cand_has_chr, torch.minimum(cand_min_clp, cand_cur_clp), cand_l1_lp)
            l1_lp_cpu = cand_l1_lp.cpu().tolist()
            min_clp_cpu = cand_min_clp.cpu().tolist()
            tau_syn = -2.75 if has_custom else -1.85
            tau_wch = -5.80 if has_custom else -3.95
            next_pool = []
            for ci in range(M):
                bm = beams[cand_parent[ci]]
                mid = cand_mid[ci]
                mb = cand_bytes[ci]
                cg = cand_cg[ci]
                l1_step = l1_lp_cpu[ci]
                worst_c = min_clp_cpu[ci]

                if use_syntax_gate:
                    l1_pb = l1_step / max(1, len(mb))
                    z_syn = max(-12.0, min(12.0, (l1_pb - tau_syn) * 2.2))
                    g_avg = 1.0 / (1.0 + math.exp(-z_syn))
                    z_wch = max(-12.0, min(12.0, (worst_c - tau_wch) * 1.4))
                    g_wch = 1.0 / (1.0 + math.exp(-z_wch))
                    g_syn = g_avg * g_wch
                    wch_pen = 0.28 * max(0.0, (tau_wch + 0.15) - worst_c) if (not has_custom) else 0.0
                else:
                    g_syn = 1.0
                    wch_pen = 0.0

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

                if use_hippocampus and anti_copy_damper > 0.0 and (not has_custom) and len(new_mseq) >= 4:
                    t4 = (new_mseq[-4], new_mseq[-3], new_mseq[-2], new_mseq[-1])
                    if t4 in ep_4grams:
                        rep_pen += anti_copy_damper

                new_used = set(bm["used_content"])
                prog_bonus = 0.0
                if cg >= 0.45 and mid >= 256 and mid not in new_used:
                    is_grounded = (mid in ep_concepts) or (not has_anchor) or (float(np.dot(mc.emb_np[mid], h_anchor)) >= inhib_thresh)
                    if is_grounded:
                        prog_bonus = (0.45 * g_syn) if (mid in ep_concepts) else (0.25 * g_syn)
                        new_used.add(mid)

                raw_l1 = bm["raw_l1_lp"] + l1_step
                raw_l2 = bm["raw_l2_sc"] + l2_step + prog_bonus
                tot_inh = bm["tot_inh"] + inh_step
                tot_rep = bm["tot_rep"] + rep_pen
                tot_nb = bm["n_bytes"] + len(mb)
                n_steps = len(bm["gen_motors"]) + 1

                joint_sc = (
                    (raw_l1 / max(1.0, tot_nb ** 0.90))
                    + 0.60 * (raw_l2 / max(1.0, n_steps ** 0.85))
                    - (tot_inh / max(1.0, tot_nb ** 0.90))
                    - 0.60 * (tot_rep / max(1.0, n_steps ** 0.85))
                )
                c_hi_n = (bm["consec_hi"] + cg) if cg > 0.45 else (bm["consec_hi"] * 0.25)

                next_pool.append((joint_sc, ci, {
                    "score": joint_sc,
                    "raw_l1_lp": raw_l1,
                    "raw_l2_sc": raw_l2,
                    "tot_inh": tot_inh,
                    "tot_rep": tot_rep,
                    "n_bytes": tot_nb,
                    "motor_seq": new_mseq,
                    "gen_motors": bm["gen_motors"] + [mid],
                    "gen_bytes": bytearray(bm["gen_bytes"]) + bytearray(mb),
                    "seen_m3": new_seen3,
                    "seen_m2": new_seen2,
                    "used_content": new_used,
                    "consec_hi": c_hi_n,
                }))

            next_pool.sort(key=lambda x: x[0], reverse=True)
            chosen = []
            seen_last = set()
            for item in next_pool:
                sig = bytes(item[2]["gen_bytes"][-6:])
                if sig in seen_last and len(next_pool) > beam_width:
                    continue
                seen_last.add(sig)
                chosen.append(item)
                if len(chosen) >= beam_width:
                    break

            ch_idx = torch.tensor([ci for (_, ci, _) in chosen], dtype=torch.int64, device=dev)
            st_beams = runner.gather_state(st_c, ch_idx)
            logits_beams = log_c[ch_idx].clone()
            beams = [nb for (_, _, nb) in chosen]
            if beams[0]["n_bytes"] >= 190:
                break

        decode_elapsed = time.perf_counter() - t0
        best_bm = beams[0]
        gen_bytes_raw = bytes(best_bm["gen_bytes"])
        full_b = prompt_b + gen_bytes_raw
        full_txt = full_b.decode("utf-8", errors="replace")
        gen_only_txt = gen_bytes_raw.decode("utf-8", errors="replace")
        valid_utf8 = True
        try:
            gen_bytes_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            valid_utf8 = False
        t_cr = time.perf_counter()
        flu_bpc = runner.score_text_bpc(full_txt, prompt_prefix_len=len(prompt_b), engram_mult=0.72) if score_critic else -1.0
        critic_elapsed = time.perf_counter() - t_cr

        gen_m = best_bm["gen_motors"]
        c_sims_1, c_sims_2 = [], []
        n_content_gen = 0
        n_ep_hit = 0
        if float(np.linalg.norm(h_root)) > 0.1 and len(gen_m) > 0:
            half = max(1, len(gen_m) // 2)
            for idx_m, mid in enumerate(gen_m):
                if mid >= 256 and rc.cg_vec[mid] > 0.45:
                    n_content_gen += 1
                    if mid in ep_concepts:
                        n_ep_hit += 1
                    em = mc.emb_np[mid]
                    if float(np.dot(em, em)) > 0.1:
                        cs = float(np.dot(em, h_root))
                        (c_sims_1 if idx_m < half else c_sims_2).append(cs)

        max_verb = _longest_common_substring_chars(gen_only_txt, epi["episodes_txt"]) if use_hippocampus else 0
        gen_chars = max(1, len(gen_only_txt))
        info = {
            "fluency_bpc": round(flu_bpc, 3),
            "gen_bytes": int(best_bm["n_bytes"]),
            "gen_chars": gen_chars,
            "motor_steps": int(len(gen_m)),
            "bytes_per_motor": round(best_bm["n_bytes"] / max(1, len(gen_m)), 2),
            "decode_sec": round(decode_elapsed, 2),
            "critic_sec": round(critic_elapsed, 2),
            "chars_per_sec": round(gen_chars / max(1e-4, decode_elapsed), 2),
            "hip_query_ms": epi["query_ms"],
            "epi_conf": epi["epi_conf"],
            "ep_sources": epi["ep_sources"][:3],
            "thought_lock_2nd": round(float(np.mean(c_sims_2)), 3) if c_sims_2 else 0.0,
            "grounded_concept_rate": round(n_ep_hit / max(1, n_content_gen), 3),
            "max_verbatim_chars": int(max_verb),
        }
        return full_txt, valid_utf8, info
