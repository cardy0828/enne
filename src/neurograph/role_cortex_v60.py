"""HRR 束縛による文中の役割表現（v6.0）。"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50, HierarchicalBrainV50


class RoleBoundThoughtCortexV60:

    def __init__(
        self,
        mc: ContinuousThoughtCortexV50,
        num_roles: int = 64,
        seed: int = 60,
    ):
        self.mc = mc
        self.V_m = mc.V_m
        self.D_m = mc.thought_dim
        self.num_roles = num_roles
        self.seed = seed

        grown_ids = np.arange(256, self.V_m, dtype=np.int32)
        grown_idfs = mc.motor_idfs[256:]
        low_mask = (grown_idfs > 0.1) & (grown_idfs < (mc.tau_idf - 0.50))
        cand_ids = grown_ids[low_mask]
        cand_idfs = grown_idfs[low_mask]
        order = np.argsort(cand_idfs)[:num_roles]
        self.role_motor_ids: np.ndarray = cand_ids[order].astype(np.int32)
        self.R = int(len(self.role_motor_ids))

        self.mid_to_role: np.ndarray = np.full(self.V_m, -1, dtype=np.int32)
        for ri, mid in enumerate(self.role_motor_ids.tolist()):
            self.mid_to_role[mid] = ri
        self.content_cg_thresh: float = 0.50

        self.total_ops = 2 * self.R + 1
        rng = np.random.default_rng(seed)
        self.shifts = np.array([(i * 19 + 7) % self.D_m for i in range(self.total_ops)], dtype=np.int32)
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(self.total_ops, self.D_m))
        self.signs = np.ascontiguousarray(self.signs, dtype=np.float32)

        self.W_joint: Optional[np.ndarray] = None
        self.W_role_only: Optional[np.ndarray] = None
        self.W_topic_only: Optional[np.ndarray] = None

        self.role_pred_table: Dict[Tuple[int, int], Dict[int, float]] = {}

        idfs = mc.motor_idfs.astype(np.float32)
        z = np.clip((idfs - mc.tau_idf) * 2.2, -12.0, 12.0)
        self.cg_vec = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        self.cg_vec[:256] = 0.0

    def bind_vec(self, e: np.ndarray, op_idx: int) -> np.ndarray:
        sh = int(self.shifts[op_idx])
        if sh == 0:
            return e * self.signs[op_idx]
        return np.roll(e, sh) * self.signs[op_idx]

    def extract_role_events(self, motor_seq: np.ndarray) -> List[Tuple[int, int, int, float]]:
        L = len(motor_seq)
        events: List[Tuple[int, int, int, float]] = []
        R = self.R
        m2r = self.mid_to_role
        cg_v = self.cg_vec
        for i in range(L):
            mid = int(motor_seq[i])
            if mid < 256 or mid >= self.V_m:
                continue
            if m2r[mid] >= 0:
                continue
            cg = float(cg_v[mid])
            if cg < self.content_cg_thresh:
                continue
            op = 2 * R
            if i + 1 < L:
                m_r = int(motor_seq[i + 1])
                if 0 <= m_r < self.V_m and m2r[m_r] >= 0:
                    op = int(m2r[m_r])
            if op == 2 * R and i - 1 >= 0:
                m_l = int(motor_seq[i - 1])
                if 0 <= m_l < self.V_m and m2r[m_l] >= 0:
                    op = R + int(m2r[m_l])
            events.append((i, mid, op, cg))
        return events

    def compute_states_causal(
        self,
        motor_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = self.D_m
        h_topic = np.zeros(D, dtype=np.float32)
        h_rf = np.zeros(D, dtype=np.float32)
        h_rs = np.zeros(D, dtype=np.float32)
        if self.mc.emb_np is None or len(motor_seq) == 0:
            return h_topic, h_rf, h_rs

        events = self.extract_role_events(motor_seq)
        emb = self.mc.emb_np
        for _, mid, op, cg in events:
            e = emb[mid]
            if float(np.dot(e, e)) < 0.1:
                continue
            b = self.bind_vec(e, op)
            at = 0.12 * cg
            af = 0.45 * cg
            as_ = 0.20 * cg
            h_topic = (1.0 - at) * h_topic + at * e
            h_rf = (1.0 - af) * h_rf + af * b
            h_rs = (1.0 - as_) * h_rs + as_ * b

        for v in (h_topic, h_rf, h_rs):
            nrm = float(np.linalg.norm(v))
            if nrm > 1e-6:
                v /= nrm
        return h_topic, h_rf, h_rs

    @torch.no_grad()
    def train_hebbian_role_predictors(
        self,
        motor_streams: List[np.ndarray],
        device: torch.device,
        max_motors_per_stream: int = 8_000_000,
        chunk_sz: int = 32_768,
        build_discrete_table: bool = True,
    ) -> Dict[str, float]:
        t0 = time.time()
        D = self.D_m
        R = self.R
        E_gpu = torch.from_numpy(self.mc.emb_np).to(device)
        signs_gpu = torch.from_numpy(self.signs).to(device)
        shifts_np = self.shifts
        m2r = self.mid_to_role
        cg_v = self.cg_vec

        C_xx = torch.zeros((3 * D, 3 * D), device=device, dtype=torch.float32)
        C_xy = torch.zeros((3 * D, D), device=device, dtype=torch.float32)

        K = 24
        d_idx = torch.arange(K, device=device, dtype=torch.float32)
        def _mk_ker(alpha: float) -> torch.Tensor:
            k = alpha * ((1.0 - alpha) ** d_idx)
            return (k / k.sum()).flip(0).view(1, 1, K)

        ker_t = _mk_ker(0.12)
        ker_rf = _mk_ker(0.45)
        ker_rs = _mk_ker(0.20)

        col_base = torch.arange(D, device=device, dtype=torch.int64).unsqueeze(0)
        sh_t = torch.from_numpy(shifts_np.astype(np.int64)).to(device).unsqueeze(1)
        gather_idx = (col_base - sh_t) % D

        total_events = 0
        disc_counts: Dict[Tuple[int, int], Dict[int, float]] = {}

        for st in motor_streams:
            sub = st[:max_motors_per_stream]
            L = len(sub)
            if L < 64:
                continue

            valid_c = (sub >= 256) & (sub < self.V_m)
            sub_cl = np.clip(sub, 0, self.V_m - 1)
            cgs = np.where(valid_c & (m2r[sub_cl] < 0), cg_v[sub_cl], 0.0)
            pos = np.nonzero(cgs >= self.content_cg_thresh)[0]
            if len(pos) <= K + 4:
                continue

            c_mids = sub[pos].astype(np.int64)
            ops = np.full(len(pos), 2 * R, dtype=np.int64)

            has_r = (pos + 1) < L
            r_ids = np.where(has_r, sub[np.minimum(pos + 1, L - 1)], 0)
            r_role = np.where(has_r & (r_ids >= 0) & (r_ids < self.V_m), m2r[np.clip(r_ids, 0, self.V_m - 1)], -1)
            ops = np.where(r_role >= 0, r_role, ops)

            need_l = (ops == 2 * R) & (pos > 0)
            l_ids = np.where(need_l, sub[np.maximum(pos - 1, 0)], 0)
            l_role = np.where(need_l & (l_ids >= 0) & (l_ids < self.V_m), m2r[np.clip(l_ids, 0, self.V_m - 1)], -1)
            ops = np.where(l_role >= 0, R + l_role, ops)

            if build_discrete_table:
                M_ev = len(c_mids)
                for ei in range(0, M_ev - 1, 2):
                    op_i = int(ops[ei])
                    if op_i >= 2 * R:
                        continue
                    s_mid = int(c_mids[ei])
                    key = (s_mid, op_i)
                    mp = disc_counts.get(key)
                    if mp is None:
                        if len(disc_counts) > 120_000:
                            continue
                        mp = {}
                        disc_counts[key] = mp
                    lim = min(M_ev, ei + 7)
                    for ej in range(ei + 1, lim):
                        p_mid = int(c_mids[ej])
                        if p_mid == s_mid:
                            continue
                        mp[p_mid] = mp.get(p_mid, 0.0) + (1.0 / (1.0 + 0.35 * (ej - ei)))
                    if len(mp) > 96:
                        top_k = sorted(mp.items(), key=lambda x: x[1], reverse=True)[:48]
                        disc_counts[key] = dict(top_k)

            M_ev = len(c_mids)
            for c0 in range(0, M_ev, chunk_sz):
                c1 = min(M_ev, c0 + chunk_sz)
                if c1 - c0 <= K + 2:
                    continue
                cm_t = torch.from_numpy(c_mids[c0:c1]).to(device)
                op_t = torch.from_numpy(ops[c0:c1]).to(device)

                e_seq = E_gpu[cm_t]
                idx_g = gather_idx[op_t]
                b_seq = torch.gather(e_seq, 1, idx_g) * signs_gpu[op_t]

                e_in = e_seq[:-1].T.unsqueeze(1)
                b_in = b_seq[:-1].T.unsqueeze(1)
                ht = torch.nn.functional.conv1d(e_in, ker_t).squeeze(1).T
                hrf = torch.nn.functional.conv1d(b_in, ker_rf).squeeze(1).T
                hrs = torch.nn.functional.conv1d(b_in, ker_rs).squeeze(1).T

                ht = ht / ht.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                hrf = hrf / hrf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                hrs = hrs / hrs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

                X = torch.cat([ht, hrf, hrs], dim=-1)
                Y = e_seq[K:]

                C_xx += X.T @ X
                C_xy += X.T @ Y
                total_events += int(X.shape[0])

                del cm_t, op_t, e_seq, idx_g, b_seq, e_in, b_in, ht, hrf, hrs, X, Y

        eye3 = torch.eye(3 * D, device=device)
        lam3 = 0.18 * (torch.trace(C_xx) / (3 * D)).clamp(min=1.0)
        W_j = torch.linalg.solve(C_xx + lam3 * eye3, C_xy)
        self.W_joint = np.ascontiguousarray(W_j.T.cpu().numpy(), dtype=np.float32)

        C_tt = C_xx[:D, :D]
        C_ty = C_xy[:D, :]
        lam1 = 0.18 * (torch.trace(C_tt) / D).clamp(min=1.0)
        W_t = torch.linalg.solve(C_tt + lam1 * torch.eye(D, device=device), C_ty)
        self.W_topic_only = np.ascontiguousarray(W_t.T.cpu().numpy(), dtype=np.float32)

        C_rr = C_xx[D:, D:]
        C_ry = C_xy[D:, :]
        lam2 = 0.18 * (torch.trace(C_rr) / (2 * D)).clamp(min=1.0)
        W_r = torch.linalg.solve(C_rr + lam2 * torch.eye(2 * D, device=device), C_ry)
        self.W_role_only = np.ascontiguousarray(W_r.T.cpu().numpy(), dtype=np.float32)

        self.role_pred_table = {}
        for k, mp in disc_counts.items():
            if not mp:
                continue
            top_items = sorted(mp.items(), key=lambda x: x[1], reverse=True)[:32]
            mx = max(top_items[0][1], 1e-5)
            filt = {m: round(w / mx, 4) for m, w in top_items if w >= 1.5}
            if filt:
                self.role_pred_table[k] = filt

        del E_gpu, signs_gpu, gather_idx, C_xx, C_xy, W_j, W_t, W_r
        torch.cuda.empty_cache()

        return {
            "num_emergent_roles": self.R,
            "events_trained": total_events,
            "discrete_role_keys": len(self.role_pred_table),
            "sec": round(time.time() - t0, 2),
        }

    def save(self, path: str) -> float:
        import pickle
        t0 = time.perf_counter()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "V_m": self.V_m,
            "D_m": self.D_m,
            "role_motor_ids": self.role_motor_ids,
            "shifts": self.shifts,
            "signs": self.signs,
            "W_joint": self.W_joint,
            "W_role_only": self.W_role_only,
            "W_topic_only": self.W_topic_only,
            "role_pred_table": self.role_pred_table,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return round(time.perf_counter() - t0, 3)

    def load_weights(self, path: str) -> float:
        import pickle
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["V_m"] != self.V_m or d["D_m"] != self.D_m:
            raise ValueError(f"RoleCortex shape mismatch: file=({d['V_m']},{d['D_m']}) vs mc=({self.V_m},{self.D_m})")
        self.role_motor_ids = d["role_motor_ids"]
        self.R = int(len(self.role_motor_ids))
        self.mid_to_role = np.full(self.V_m, -1, dtype=np.int32)
        for ri, mid in enumerate(self.role_motor_ids.tolist()):
            self.mid_to_role[mid] = ri
        self.shifts = d["shifts"]
        self.signs = d["signs"]
        self.W_joint = d["W_joint"]
        self.W_role_only = d["W_role_only"]
        self.W_topic_only = d["W_topic_only"]
        self.role_pred_table = d["role_pred_table"]
        return round(time.perf_counter() - t0, 3)

    def predict_next_concept_vec(
        self,
        motor_seq: np.ndarray,
        mode: str = "joint",
    ) -> np.ndarray:
        ht, hrf, hrs = self.compute_states_causal(motor_seq)
        if mode == "unbound_topic":
            out = ht
        elif mode == "topic_ridge":
            out = self.W_topic_only @ ht
        elif mode == "role_only":
            out = self.W_role_only @ np.concatenate([hrf, hrs])
        else:
            out = self.W_joint @ np.concatenate([ht, hrf, hrs])
        nrm = float(np.linalg.norm(out))
        if nrm > 1e-6:
            out = out / nrm
        return out.astype(np.float32)

    def score_continuation_phrase(
        self,
        prompt: str,
        candidate_phrase: str,
        mode: str = "joint",
    ) -> float:
        p_ids = self.mc.encode(np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8))
        c_ids = self.mc.encode(np.frombuffer(candidate_phrase.encode("utf-8"), dtype=np.uint8))
        pred_v = self.predict_next_concept_vec(p_ids, mode=mode)

        disc_bonus = 0.0
        if mode in ("role_only", "joint") and self.role_pred_table:
            p_ev = self.extract_role_events(p_ids)
            for idx_e, (_, smid, op, cg) in enumerate(p_ev[-6:]):
                mp = self.role_pred_table.get((smid, op))
                if not mp:
                    continue
                for cm in c_ids.tolist():
                    if cm in mp:
                        disc_bonus += 0.25 * cg * mp[cm]

        sims = []
        for cm in c_ids.tolist():
            if cm >= 256 and self.cg_vec[cm] > 0.25:
                e = self.mc.emb_np[cm]
                if float(np.dot(e, e)) > 0.1:
                    sims.append(float(np.dot(e, pred_v)))
        base = float(np.mean(sims)) if sims else 0.0
        return base + disc_bonus

    @torch.no_grad()
    def generate_role_bound_beam(
        self,
        brain: HierarchicalBrainV50,
        prompt: str,
        max_motor_steps: int = 28,
        beam_width: int = 6,
        top_k_motor: int = 16,
        use_role_binding: bool = True,
        topic_ppmi_gain: float = 1.65,
        role_align_gain: float = 1.60,
        inhib_gain: float = 1.30,
        inhib_thresh: float = 0.22,
        rhythm_penalty: float = 0.55,
    ) -> Tuple[str, bool, Dict]:
        if not use_role_binding:
            return brain.generate_motor_beam_v50(
                prompt,
                max_motor_steps=max_motor_steps,
                beam_width=beam_width,
                top_k_motor=top_k_motor,
                use_thought_field=True,
                use_topdown_inhibition=True,
                topic_ppmi_gain=topic_ppmi_gain,
                cont_align_gain=role_align_gain,
                inhib_gain=inhib_gain,
                inhib_thresh=inhib_thresh,
                rhythm_penalty=rhythm_penalty,
            )

        t0 = time.perf_counter()
        dev = brain.device
        runner = brain.runner
        mc = self.mc
        prompt_b = prompt.encode("utf-8")
        if not prompt_b:
            return "", True, {}

        p_motors = mc.encode(np.frombuffer(prompt_b, dtype=np.uint8))
        prompt_m = p_motors.tolist()

        root_t_ids, root_t_w = mc.extract_active_topics(p_motors, max_topics=6, min_idf=5.8, use_resonance=True)
        h_root = mc.compute_root_thought_vec(p_motors)
        p_ht, p_hrf, p_hrs = self.compute_states_causal(p_motors)
        p_pred_role = self.predict_next_concept_vec(p_motors, mode="joint")
        p_role_events = self.extract_role_events(p_motors)

        root_mass = sum(
            float(self.cg_vec[int(m)])
            for m in prompt_m
            if int(m) >= 256 and float(np.dot(mc.emb_np[int(m)], mc.emb_np[int(m)])) > 0.1
        )
        root_conf = min(1.0, root_mass / 1.6)
        has_role_vec = bool(float(np.linalg.norm(p_pred_role)) > 0.1 and root_conf > 0.25)

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
            cand_parent, cand_mid, cand_l2, cand_inh_b, cand_bytes, cand_cg, cand_is_esc = [], [], [], [], [], [], []

            for wi, bm in enumerate(beams):
                m_arr = np.array(bm["motor_seq"][-32:], dtype=np.int32)
                cur_pred = self.predict_next_concept_vec(m_arr, mode="joint")
                target_role_vec = 0.55 * p_pred_role + 0.45 * cur_pred
                tn = float(np.linalg.norm(target_role_vec))
                if tn > 1e-6:
                    target_role_vec = target_role_vec / tn

                rec_ids, rec_w = mc.extract_active_topics(m_arr, max_topics=4, min_idf=5.2)
                t_map: Dict[int, float] = {int(k): float(v) for k, v in zip(root_t_ids.tolist(), root_t_w.tolist())}
                for dk, dv in zip(rec_ids.tolist(), rec_w.tolist()):
                    if has_role_vec and dk >= 256:
                        if float(np.dot(mc.emb_np[dk], target_role_vec)) < 0.20:
                            continue
                    t_map[int(dk)] = max(t_map.get(int(dk), 0.0), float(dv) * 0.35)
                ranked_t = sorted(t_map.items(), key=lambda x: x[1], reverse=True)[:6]
                t_arr = np.array([m for m, _ in ranked_t], dtype=np.int32) if ranked_t else np.zeros(0, dtype=np.int32)
                w_arr = np.array([w for _, w in ranked_t], dtype=np.float32) if ranked_t else np.zeros(0, dtype=np.float32)

                p_ids, p_sc = mc.predict_topk(
                    np.array(bm["motor_seq"][-6:], dtype=np.int32),
                    t_arr, w_arr, topic_ppmi_gain=topic_ppmi_gain, top_k=top_k_motor * 3,
                )

                recent_16 = set(bm["gen_motors"][-16:])
                c_hi = bm["consec_hi"]
                added_bytes_wi = set()

                def _score_cand(mid_v: int, mb: bytes, base_l2: float) -> Tuple[float, float, float]:
                    cg = float(self.cg_vec[mid_v]) if 0 <= mid_v < self.V_m else 0.0
                    l2_s = float(base_l2)
                    inh_b = 0.0

                    if mid_v >= 256 and mid_v in recent_16:
                        l2_s -= (0.95 + 1.25 * cg)
                    if cg > 0.55 and c_hi >= 1.8:
                        l2_s -= rhythm_penalty * (c_hi - 1.5) * cg
                    elif cg < 0.25 and c_hi >= 2.1:
                        l2_s += 0.45

                    if has_role_vec and mid_v >= 256 and cg > 0.15:
                        em = mc.emb_np[mid_v]
                        if float(np.dot(em, em)) > 0.1:
                            cos_role = float(np.dot(em, target_role_vec))
                            cos_root = float(np.dot(em, h_root))
                            disc_r = 0.0
                            for _, smid, op, scg in p_role_events[-6:]:
                                mp = self.role_pred_table.get((smid, op))
                                if mp and mid_v in mp:
                                    disc_r += scg * mp[mid_v]
                            eff_align = 0.65 * cos_role + 0.35 * cos_root + 0.22 * math.tanh(disc_r)

                            l2_s += cg * role_align_gain * eff_align
                            if eff_align < inhib_thresh:
                                mismatch = min(1.25, (inhib_thresh - eff_align) / max(0.05, inhib_thresh))
                                inh_b = root_conf * cg * inhib_gain * mismatch
                    return l2_s, inh_b, cg

                taken = 0
                for mid_v, sc_v in zip(p_ids.tolist(), p_sc.tolist()):
                    mb = mc.motor_bytes[mid_v]
                    if mid_v < 256 and (mid_v < 32 or mid_v >= 127):
                        continue
                    if b"\n" in mb or b"\r" in mb or mb in added_bytes_wi:
                        continue
                    if bm["motor_seq"] and bm["motor_seq"][-1] == mid_v:
                        continue
                    l2_s, inh_b, cg = _score_cand(mid_v, mb, sc_v)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2.append(l2_s)
                    cand_inh_b.append(inh_b); cand_bytes.append(mb); cand_cg.append(cg)
                    cand_is_esc.append(False)
                    added_bytes_wi.add(mb)
                    taken += 1
                    if taken >= top_k_motor:
                        break

                lrow = logits_beams[wi].clone()
                lrow[~runner.lead_mask] = -1e9
                lrow[0] = -1e9; lrow[10] = -1e9; lrow[13] = -1e9
                for b0 in torch.topk(lrow, k=4).indices.cpu().tolist():
                    if b0 < 0x80:
                        mb = bytes([b0])
                    else:
                        clen = 2 if (b0 & 0xE0) == 0xC0 else (3 if (b0 & 0xF0) == 0xE0 else 4)
                        ch_b = bytearray([b0])
                        st_tmp = runner.gather_state(st_beams, torch.tensor([wi], dtype=torch.int64, device=dev))
                        cur_b = b0
                        for ci_u in range(1, clen):
                            lg_u, _ = runner.step_batch(
                                st_tmp, torch.tensor([cur_b], dtype=torch.int64, device=dev),
                                locked_mode_p=locked_mode_p, use_engram=True, engram_mult=0.72, is_prompt=False,
                            )
                            row_u = lg_u[0].clone()
                            mask_u = torch.zeros(256, dtype=torch.bool, device=dev)
                            lo, hi = 0x80, 0xBF
                            if ci_u == 1:
                                if b0 == 0xE0: lo = 0xA0
                                elif b0 == 0xED: hi = 0x9F
                                elif b0 == 0xF0: lo = 0x90
                                elif b0 == 0xF4: hi = 0x8F
                            mask_u[lo : hi + 1] = True
                            row_u[~mask_u] = -1e9
                            cur_b = int(row_u.argmax().item())
                            ch_b.append(cur_b)
                        mb = bytes(ch_b)
                        try:
                            mb.decode("utf-8", errors="strict")
                        except UnicodeDecodeError:
                            continue
                    if mb in added_bytes_wi:
                        continue
                    mid_v = bytes_to_mid.get(mb, -1)
                    l2_s, inh_b, cg = _score_cand(mid_v, mb, 0.25)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2.append(l2_s)
                    cand_inh_b.append(inh_b); cand_bytes.append(mb); cand_cg.append(cg)
                    cand_is_esc.append(True)
                    added_bytes_wi.add(mb)

            if not cand_parent:
                break

            if len(cand_parent) > 42:
                esc_idx = [i for i, f in enumerate(cand_is_esc) if f][:16]
                l2_idx = [i for i, f in enumerate(cand_is_esc) if not f]
                l2_idx.sort(key=lambda i: beams[cand_parent[i]]["score"] + 0.55 * cand_l2[i], reverse=True)
                ord_c = esc_idx + l2_idx[: max(0, 42 - len(esc_idx))]
                cand_parent = [cand_parent[i] for i in ord_c]
                cand_mid = [cand_mid[i] for i in ord_c]
                cand_l2 = [cand_l2[i] for i in ord_c]
                cand_inh_b = [cand_inh_b[i] for i in ord_c]
                cand_bytes = [cand_bytes[i] for i in ord_c]
                cand_cg = [cand_cg[i] for i in ord_c]

            M = len(cand_parent)
            p_idx = torch.tensor(cand_parent, dtype=torch.int64, device=dev)
            st_c = runner.gather_state(st_beams, p_idx)
            log_c = logits_beams[p_idx].clone()
            max_blen = max(len(mb) for mb in cand_bytes)
            cand_l1_lp = torch.zeros(M, dtype=torch.float32, device=dev)

            for bi in range(max_blen):
                act = [i for i in range(M) if bi < len(cand_bytes[i])]
                if not act:
                    break
                act_t = torch.tensor(act, dtype=torch.int64, device=dev)
                lp_act = torch.log_softmax(log_c[act_t], dim=-1)
                yb_t = torch.tensor([cand_bytes[i][bi] for i in act], dtype=torch.int64, device=dev)
                cand_l1_lp[act_t] += lp_act[torch.arange(len(act), device=dev), yb_t]
                st_sub = runner.gather_state(st_c, act_t)
                log_sub, _ = runner.step_batch(
                    st_sub, yb_t, locked_mode_p=locked_mode_p, use_engram=True, engram_mult=0.72, is_prompt=False
                )
                for k_st in st_c.keys():
                    st_c[k_st][act_t] = st_sub[k_st]
                log_c[act_t] = log_sub

            l1_lp_cpu = cand_l1_lp.cpu().tolist()
            next_pool = []
            for ci in range(M):
                bm = beams[cand_parent[ci]]
                mid = cand_mid[ci]
                mb = cand_bytes[ci]
                cg = cand_cg[ci]
                l1_step = l1_lp_cpu[ci]
                l2_step = cand_l2[ci]
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

                new_used = set(bm["used_content"])
                prog_bonus = 0.0
                if cg >= 0.45 and mid >= 256 and mid not in new_used:
                    if (not has_role_vec) or float(np.dot(mc.emb_np[mid], p_pred_role)) >= inhib_thresh:
                        prog_bonus = 0.30
                        new_used.add(mid)

                raw_l1 = bm["raw_l1_lp"] + l1_step
                raw_l2 = bm["raw_l2_sc"] + l2_step + prog_bonus
                tot_inh = bm["tot_inh"] + inh_step
                tot_rep = bm["tot_rep"] + rep_pen
                tot_nb = bm["n_bytes"] + len(mb)
                n_steps = len(bm["gen_motors"]) + 1

                joint_sc = (
                    (raw_l1 / max(1.0, tot_nb ** 0.90))
                    + 0.58 * (raw_l2 / max(1.0, n_steps ** 0.85))
                    - (tot_inh / max(1.0, tot_nb ** 0.90))
                    - 0.58 * (tot_rep / max(1.0, n_steps ** 0.85))
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

        best_bm = beams[0]
        full_b = prompt_b + bytes(best_bm["gen_bytes"])
        full_txt = full_b.decode("utf-8", errors="replace")
        valid_utf8 = True
        try:
            bytes(best_bm["gen_bytes"]).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            valid_utf8 = False
        flu_bpc = runner.score_text_bpc(full_txt, prompt_prefix_len=len(prompt_b), engram_mult=0.72)

        gen_m = best_bm["gen_motors"]
        c_sims_1, c_sims_2 = [], []
        if float(np.linalg.norm(h_root)) > 0.1 and len(gen_m) > 0:
            half = max(1, len(gen_m) // 2)
            for idx_m, mid in enumerate(gen_m):
                if mid >= 256 and self.cg_vec[mid] > 0.45:
                    em = mc.emb_np[mid]
                    if float(np.dot(em, em)) > 0.1:
                        cs = float(np.dot(em, h_root))
                        (c_sims_1 if idx_m < half else c_sims_2).append(cs)

        all_sims = c_sims_1 + c_sims_2
        info = {
            "fluency_bpc": round(flu_bpc, 3),
            "gen_bytes": int(best_bm["n_bytes"]),
            "motor_steps": int(len(gen_m)),
            "bytes_per_motor": round(best_bm["n_bytes"] / max(1, len(gen_m)), 2),
            "decode_sec": round(time.perf_counter() - t0, 2),
            "thought_lock_mean": round(float(np.mean(all_sims)), 3) if all_sims else 0.0,
            "thought_lock_2nd": round(float(np.mean(c_sims_2)), 3) if c_sims_2 else 0.0,
        }
        return full_txt, valid_utf8, info
