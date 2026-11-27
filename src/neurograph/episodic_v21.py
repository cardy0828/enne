"""文章断片の転置索引による文章記憶と、検索結果を置く作業領域（v2.1）。"""

from __future__ import annotations

import ctypes
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neurograph.engram_v20 import DecoupledBrainV20, EngramCartridge
from neurograph.memory_v3b import _mix64


_EP_LIB_CACHE: Optional[ctypes.CDLL] = None


def load_episodic_lib() -> ctypes.CDLL:
    global _EP_LIB_CACHE
    if _EP_LIB_CACHE is not None:
        return _EP_LIB_CACHE
    here = Path(__file__).resolve().parent
    cpp_path = here / "episodic_kernel.cpp"
    so_path = here / "libepisodic.so"
    if (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime):
        cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(cpp_path), "-o", str(so_path)]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))

    c_u8_p = ctypes.POINTER(ctypes.c_uint8)
    c_u64_p = ctypes.POINTER(ctypes.c_uint64)
    c_u32_p = ctypes.POINTER(ctypes.c_uint32)
    c_f32_p = ctypes.POINTER(ctypes.c_float)

    lib.episodic_build.argtypes = [
        c_u8_p, ctypes.c_uint64, c_u64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
    ]
    lib.episodic_build.restype = ctypes.c_void_p

    lib.episodic_free.argtypes = [ctypes.c_void_p]
    lib.episodic_free.restype = None

    lib.episodic_num_postings.argtypes = [ctypes.c_void_p]
    lib.episodic_num_postings.restype = ctypes.c_uint64

    lib.episodic_query.argtypes = [
        ctypes.c_void_p, c_u8_p, ctypes.c_int, ctypes.c_int, c_u32_p, c_f32_p,
    ]
    lib.episodic_query.restype = ctypes.c_int

    lib.episodic_span_idf.argtypes = [ctypes.c_void_p, c_u8_p]
    lib.episodic_span_idf.restype = ctypes.c_float

    _EP_LIB_CACHE = lib
    return lib


def slice_raw_bytes_into_episodes(
    raw_bytes: np.ndarray,
    target_len: int = 240,
    min_len: int = 80,
) -> np.ndarray:
    N = len(raw_bytes)
    if N <= target_len:
        return np.array([0, N], dtype=np.uint64)

    offs = [0]
    pos = 0
    nl_pos = np.flatnonzero(raw_bytes == 10)
    nl_idx = 0
    n_nl = len(nl_pos)

    while pos + min_len < N:
        want = min(pos + target_len, N)
        if want >= N:
            break
        while nl_idx < n_nl and nl_pos[nl_idx] < pos + min_len:
            nl_idx += 1
        if nl_idx < n_nl and nl_pos[nl_idx] <= pos + target_len + 60:
            nxt = int(nl_pos[nl_idx]) + 1
            nl_idx += 1
        else:
            nxt = want
            while nxt < N and (raw_bytes[nxt] & 0xC0) == 0x80:
                nxt += 1
        if nxt <= pos:
            nxt = min(pos + target_len, N)
        offs.append(nxt)
        pos = nxt

    if offs[-1] < N:
        offs.append(N)
    return np.array(offs, dtype=np.uint64)


class EpisodicCartridge:

    def __init__(
        self,
        name: str,
        raw_bytes: np.ndarray,
        target_ep_len: int = 240,
        num_buckets: int = 1_048_576,
        max_df_ratio: float = 0.004,
        stride: int = 3,
        priority_boost: float = 1.0,
    ):
        self.name = name
        self.lib = load_episodic_lib()
        self.raw_bytes = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        self.ep_off = slice_raw_bytes_into_episodes(self.raw_bytes, target_len=target_ep_len)
        self.num_episodes = int(len(self.ep_off) - 1)
        self.priority_boost = float(priority_boost)

        max_df_keep = max(8, int(self.num_episodes * max_df_ratio))
        t0 = time.time()
        self._handle = self.lib.episodic_build(
            self.raw_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint64(len(self.raw_bytes)),
            self.ep_off.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint32(self.num_episodes),
            ctypes.c_uint32(num_buckets),
            ctypes.c_int(max_df_keep),
            ctypes.c_int(stride),
        )
        self.build_sec = round(time.time() - t0, 2)
        self.num_postings = int(self.lib.episodic_num_postings(self._handle))

    def __del__(self):
        if getattr(self, "_handle", None):
            self.lib.episodic_free(self._handle)
            self._handle = None

    def get_episode_bytes(self, ep_id: int) -> bytes:
        s0 = int(self.ep_off[ep_id])
        s1 = int(self.ep_off[ep_id + 1])
        return self.raw_bytes[s0:s1].tobytes()

    def query(self, query_bytes: bytes, top_k: int = 4) -> List[Tuple[float, str, bytes]]:
        if not self._handle or len(query_bytes) < 6:
            return []
        q_arr = np.frombuffer(query_bytes, dtype=np.uint8)
        out_ids = np.zeros(top_k, dtype=np.uint32)
        out_sc = np.zeros(top_k, dtype=np.float32)
        n = int(self.lib.episodic_query(
            self._handle,
            q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(len(q_arr)),
            ctypes.c_int(top_k),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ))
        res = []
        for i in range(n):
            eid = int(out_ids[i])
            sc = float(out_sc[i]) * self.priority_boost
            res.append((sc, self.name, self.get_episode_bytes(eid)))
        return res

    def span_idf(self, span6: bytes) -> float:
        if not self._handle or len(span6) < 6:
            return 0.0
        arr = np.frombuffer(span6[:6], dtype=np.uint8)
        return float(self.lib.episodic_span_idf(self._handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))))


class EpisodicRack:

    def __init__(self):
        self.cartridges: List[EpisodicCartridge] = []

    def mount(self, cart: EpisodicCartridge) -> None:
        self.cartridges.append(cart)

    def unmount(self, name: str) -> None:
        self.cartridges = [c for c in self.cartridges if c.name != name]

    def clear(self) -> None:
        self.cartridges.clear()

    @property
    def total_episodes(self) -> int:
        return sum(c.num_episodes for c in self.cartridges)

    def query(self, query_bytes: bytes, top_k: int = 3) -> List[Tuple[float, str, bytes]]:
        all_hits: List[Tuple[float, str, bytes]] = []
        for c in self.cartridges:
            all_hits.extend(c.query(query_bytes, top_k=top_k))
        all_hits.sort(key=lambda x: x[0], reverse=True)
        out: List[Tuple[float, str, bytes]] = []
        seen_pref = set()
        for sc, cname, ep_b in all_hits:
            pref = ep_b[:36]
            if pref in seen_pref:
                continue
            seen_pref.add(pref)
            out.append((sc, cname, ep_b))
            if len(out) >= top_k:
                break
        return out

    def extract_high_idf_spans(self, text_bytes: bytes, max_spans: int = 28) -> List[Tuple[bytes, float]]:
        if len(text_bytes) < 6 or not self.cartridges:
            return []
        bg = max(self.cartridges, key=lambda c: c.num_episodes)
        cands: Dict[bytes, float] = {}
        n = len(text_bytes)
        pos = 0
        while pos + 6 <= n:
            if (text_bytes[pos] & 0xC0) == 0x80:
                pos += 1
                continue
            sp = text_bytes[pos : pos + 6]
            try:
                dec = sp.decode("utf-8")
                if len(dec.strip()) >= 2 and ("\n" not in dec):
                    idf = bg.span_idf(sp)
                    if idf > 6.5:
                        cands[sp] = max(cands.get(sp, 0.0), idf)
                    elif idf == 0.0:
                        in_small = any((c.num_episodes < 10_000 and c.span_idf(sp) > 0.0) for c in self.cartridges)
                        if in_small:
                            other_big_low = any((c.num_episodes >= 10_000 and 0.0 < c.span_idf(sp) < 6.0) for c in self.cartridges)
                            if not other_big_low:
                                cands[sp] = max(cands.get(sp, 0.0), 14.0)
            except UnicodeDecodeError:
                pass
            pos += 1
        ranked = sorted(cands.items(), key=lambda x: x[1], reverse=True)
        chosen: List[Tuple[bytes, float]] = []
        chosen_strs: List[str] = []
        for sp, sc in ranked:
            s_dec = sp.decode("utf-8", errors="ignore")
            if any((s_dec in prev) or (prev in s_dec) for prev in chosen_strs):
                continue
            chosen.append((sp, sc))
            chosen_strs.append(s_dec)
            if len(chosen) >= max_spans:
                break
        return chosen


class ActiveThoughtWorkspace:

    def __init__(self, device: torch.device):
        self.device = device
        self.recalled: List[Tuple[float, str, bytes]] = []
        self.t3_ent: Dict[bytes, Dict[int, float]] = {}
        self.t4: Dict[bytes, Dict[int, float]] = {}
        self.t6: Dict[bytes, Dict[int, float]] = {}
        self.t9: Dict[bytes, Dict[int, float]] = {}
        self.topic_spans: List[Tuple[bytes, float]] = []
        self.episode_ngrams18: set = set()
        self.z_thought: Optional[torch.Tensor] = None

    def load_episodes(
        self,
        recalled: List[Tuple[float, str, bytes]],
        prompt_bytes: bytes,
        ep_rack: EpisodicRack,
        core_brain,
        max_score_norm: float = 120.0,
    ) -> None:
        self.recalled = recalled
        self.t3_ent.clear(); self.t4.clear(); self.t6.clear(); self.t9.clear()
        self.episode_ngrams18.clear()

        if not recalled:
            self.topic_spans = ep_rack.extract_high_idf_spans(prompt_bytes, max_spans=16)
            self.z_thought = None
            return

        top_sc = max(recalled[0][0], 1e-3)
        all_topic_bytes = bytearray(prompt_bytes)
        ep_vecs = []

        for rank, (sc, cname, ep_b) in enumerate(recalled):
            rel = min(1.0, sc / max_score_norm) * (0.82 ** rank)
            rel = max(0.25, rel)
            all_topic_bytes.extend(b"\n")
            all_topic_bytes.extend(ep_b)

            n = len(ep_b)
            for pos in range(4, n):
                yb = ep_b[pos]
                k4 = ep_b[pos - 4 : pos]
                d4 = self.t4.setdefault(k4, {})
                d4[yb] = d4.get(yb, 0.0) + rel * 0.45
                if pos >= 6:
                    k6 = ep_b[pos - 6 : pos]
                    d6 = self.t6.setdefault(k6, {})
                    d6[yb] = d6.get(yb, 0.0) + rel * 1.10
                if pos >= 9:
                    k9 = ep_b[pos - 9 : pos]
                    d9 = self.t9.setdefault(k9, {})
                    d9[yb] = d9.get(yb, 0.0) + rel * 1.80
                if pos >= 18:
                    self.episode_ngrams18.add(ep_b[pos - 18 : pos])

            ev = self._encode_episode_h(core_brain, ep_b)
            if ev is not None:
                ep_vecs.append(ev * rel)

        self.topic_spans = ep_rack.extract_high_idf_spans(bytes(all_topic_bytes), max_spans=28)

        for sp_b, idf_v in self.topic_spans[:20]:
            if len(sp_b) >= 6:
                ew = min(1.0, idf_v / 10.0) * 0.75
                for p_i in range(3, len(sp_b)):
                    k3 = sp_b[p_i - 3 : p_i]
                    d3 = self.t3_ent.setdefault(k3, {})
                    d3[sp_b[p_i]] = max(d3.get(sp_b[p_i], 0.0), ew)

        p_vec = self._encode_episode_h(core_brain, prompt_bytes)
        if ep_vecs:
            e_mean = torch.stack(ep_vecs).mean(dim=0)
            e_mean = e_mean / e_mean.norm().clamp(min=1e-6)
            if p_vec is not None:
                z = 0.55 * p_vec + 0.45 * e_mean
            else:
                z = e_mean
            self.z_thought = z / z.norm().clamp(min=1e-6)
        else:
            self.z_thought = p_vec

    @torch.no_grad()
    def _encode_episode_h(self, core_brain, raw_b: bytes) -> Optional[torch.Tensor]:
        if len(raw_b) < 4:
            return None
        dev = self.device
        arr = torch.tensor(list(raw_b), dtype=torch.int64, device=dev)
        n = arr.shape[0]
        h = core_brain.W_node[arr].to(torch.float32)
        frontier = arr
        z0 = torch.zeros(n, dtype=torch.int64, device=dev)
        for hop in range(1, 5):
            prev = torch.full((n,), -1, dtype=torch.int64, device=dev)
            if n > hop:
                prev[hop:] = arr[: n - hop]
            ch = core_brain.edges.lookup(frontier, z0, prev)
            ok = (frontier >= 0) & (prev >= 0) & (ch >= core_brain.V) & core_brain.alive[ch.clamp(min=0)]
            if not ok.any():
                break
            h[ok] += core_brain.W_node[ch[ok]].to(torch.float32) * (1.0 / (1.0 + 0.10 * hop))
            frontier = torch.where(ok, ch, torch.full_like(ch, -1))
        m = h.mean(dim=0)
        return m / m.norm().clamp(min=1e-6)

    def lookup_boost_vec(
        self, tail_bytes: bytes, scale: float = 1.0, max_boost: float = 2.6, include_entity_k3: bool = False
    ) -> Optional[Dict[int, float]]:
        if not self.recalled or len(tail_bytes) < 3:
            return None
        acc: Dict[int, float] = {}
        if include_entity_k3 and len(tail_bytes) >= 3:
            k3 = tail_bytes[-3:]
            if k3 in self.t3_ent:
                for b, w in self.t3_ent[k3].items():
                    acc[b] = acc.get(b, 0.0) + w
        if len(tail_bytes) >= 4:
            k4 = tail_bytes[-4:]
            if k4 in self.t4:
                for b, w in self.t4[k4].items():
                    acc[b] = acc.get(b, 0.0) + w
        if len(tail_bytes) >= 6:
            k6 = tail_bytes[-6:]
            if k6 in self.t6:
                for b, w in self.t6[k6].items():
                    acc[b] = acc.get(b, 0.0) + w
        if len(tail_bytes) >= 9:
            k9 = tail_bytes[-9:]
            if k9 in self.t9:
                for b, w in self.t9[k9].items():
                    acc[b] = acc.get(b, 0.0) + w
        if not acc:
            return None
        return {b: math.tanh(w * 0.55) * max_boost * scale for b, w in acc.items()}


class ThinkingBrainV21:

    def __init__(self, brain_v20: DecoupledBrainV20, ep_rack: EpisodicRack):
        self.v20 = brain_v20
        self.core = brain_v20.core
        self.device = brain_v20.device
        self.ep_rack = ep_rack

    @torch.no_grad()
    def eval_passages_with_episodic_engram(
        self,
        val_bytes: np.ndarray,
        bytes_per_char: float,
        num_passages: int = 240,
        prompt_bytes_len: int = 96,
        cont_bytes_len: int = 240,
        use_byte_engram: bool = True,
        use_episodic_engram: bool = True,
        ep_scale: float = 0.65,
        entropy_h0: float = 1.0,
    ) -> Dict[str, float]:
        core = self.core
        cfg = core.cfg
        device = self.device
        V, D, S = core.V, core.D, cfg.num_slots

        p_len = prompt_bytes_len + cont_bytes_len
        n_avail = min(num_passages, len(val_bytes) // p_len)
        data_np = val_bytes[: n_avail * p_len].reshape(n_avail, p_len)
        data = torch.from_numpy(data_np.astype(np.int64)).to(device)
        B, T = data.shape

        eng_boost_all = None
        if use_byte_engram and len(self.v20.rack.cartridges) > 0:
            old_mb = self.v20.rack.max_boost
            self.v20.rack.max_boost = 2.4
            keys_2d = self.v20.encoder.compute_keys_2d(data)
            cq_2d = self.v20.encoder.compute_context_q_2d(data)
            boost_np, _, _ = self.v20.rack.lookup_batch(
                [k.reshape(-1).cpu().numpy() for k in keys_2d],
                cq_2d.reshape(-1, 16).cpu().numpy(),
                use_context_gate=True,
            )
            self.v20.rack.max_boost = old_mb
            eng_boost_all = torch.from_numpy(boost_np.reshape(B, T - 1, 256)).permute(1, 0, 2).to(torch.float16).contiguous()

        ep_boost_all = None
        ep_recall_hits = 0
        ep_step_hits = 0
        if use_episodic_engram and len(self.ep_rack.cartridges) > 0:
            ep_buf = np.zeros((T - 1, B, 256), dtype=np.float16)
            for bi in range(B):
                p_raw = data_np[bi, :prompt_bytes_len].tobytes()
                rec = self.ep_rack.query(p_raw, top_k=3)
                if not rec or rec[0][0] < 18.0:
                    continue
                ep_recall_hits += 1
                ws = ActiveThoughtWorkspace(device)
                ws.load_episodes(rec, p_raw, self.ep_rack, core)
                seq_b = data_np[bi].tobytes()
                for t in range(prompt_bytes_len, T - 1):
                    bv = ws.lookup_boost_vec(seq_b[max(0, t - 8) : t + 1], scale=1.0, max_boost=2.2)
                    if bv:
                        ep_step_hits += 1
                        for k_b, v_w in bv.items():
                            ep_buf[t, bi, k_b] = v_w
            ep_boost_all = torch.from_numpy(ep_buf)

        H_buf = max(cfg.max_hops + 4, 22)
        byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=device)
        hist_ptr = 0
        D_ctx = 12
        node_delay = torch.full((B, D_ctx), -1, dtype=torch.int64, device=device)
        nd_ptr = 0
        ctx_node = torch.full((B, 2), -1, dtype=torch.int64, device=device)
        ctx_score = torch.zeros((B, 2), dtype=torch.float32, device=device)
        slot_state = torch.zeros((B, S, D), dtype=torch.float32, device=device)
        exec_vec = torch.zeros((B, D), dtype=torch.float32, device=device)

        N_stm = cfg.stm_slots
        stm_k4 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_k7 = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_val = torch.zeros((B, N_stm), dtype=torch.int64, device=device)
        stm_str = torch.zeros((B, N_stm), dtype=torch.float32, device=device)
        stm_ptr = torch.zeros(B, dtype=torch.int64, device=device)
        stm_boost = torch.zeros((B, V), dtype=torch.float32, device=device)

        arange_B = torch.arange(B, device=device)
        ones_B_f = torch.ones(B, dtype=torch.float32, device=device)
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=device)

        total_nll = 0.0
        total_top1 = 0
        total_top5 = 0
        total_steps = 0

        for t in range(T - 1):
            x_t = data[:, t]
            y_t = data[:, t + 1]
            byte_hist[:, hist_ptr] = x_t
            cur_idx = hist_ptr
            hist_ptr = (hist_ptr + 1) % H_buf

            ctx_score.mul_(0.985)
            if t >= D_ctx:
                u_old = node_delay[:, nd_ptr]
                ok_u = (u_old >= V) & core.alive[u_old.clamp(min=0)]
                if ok_u.any():
                    f_old = core.fires[u_old.clamp(min=0)].to(torch.float32)
                    sal_old = torch.where(ok_u, 1.8 / torch.sqrt(1.0 + f_old * 0.02), torch.zeros_like(f_old))
                    already = (ctx_node[:, 0] == u_old) | (ctx_node[:, 1] == u_old)
                    ws_i = ctx_score.argmin(dim=1)
                    rep = ok_u & (~already) & (sal_old > ctx_score[arange_B, ws_i]) & (sal_old > 0.55)
                    if rep.any():
                        rb = torch.nonzero(rep, as_tuple=False).squeeze(1)
                        ctx_node[rb, ws_i[rb]] = u_old[rb]
                        ctx_score[rb, ws_i[rb]] = sal_old[rb]
            top_ctx_all = ctx_node[arange_B, ctx_score.argmax(dim=1)]

            vis_b_list, vis_u_list, vis_g_list = [arange_B], [x_t], [ones_B_f]
            frontier = x_t.unsqueeze(1)
            emergent_ctx_cand = torch.full((B,), -1, dtype=torch.int64, device=device)
            for hop in range(1, cfg.max_hops + 1):
                valid_f = frontier >= 0
                if not valid_f.any():
                    break
                fb, ff = torch.nonzero(valid_f, as_tuple=True)
                p_ids = frontier[fb, ff]
                p_span = core.local_span[p_ids].to(torch.int64)
                c_loc = byte_hist[fb, (cur_idx - (p_span + 1).clamp(max=H_buf - 1)) % H_buf]
                ch_loc = core.edges.lookup(p_ids, torch.zeros_like(p_ids), c_loc)
                ch_loc = torch.where((c_loc >= 0) & (ch_loc >= V) & core.alive[ch_loc.clamp(min=0)], ch_loc, torch.full_like(ch_loc, -1))
                ch_ctx = torch.full_like(p_ids, -1)
                if cfg.enable_long_branch and 2 <= hop <= 5:
                    tc = top_ctx_all[fb]
                    rc = core.edges.lookup(p_ids, torch.ones_like(p_ids), tc)
                    ch_ctx = torch.where((tc >= V) & (rc >= V) & core.alive[rc.clamp(min=0)], rc, torch.full_like(rc, -1))
                next_front = torch.full((B, 4), -1, dtype=torch.int64, device=device)
                gv = 1.0 / (1.0 + 0.10 * hop)
                hl = ch_loc >= 0
                if hl.any():
                    hl_b, hl_c = fb[hl], ch_loc[hl]
                    next_front[hl_b, (ff[hl] * 2) % 4] = hl_c
                    vis_b_list.append(hl_b); vis_u_list.append(hl_c)
                    vis_g_list.append(torch.full((hl_b.numel(),), gv, dtype=torch.float32, device=device))
                    if 2 <= hop <= 4:
                        emergent_ctx_cand[hl_b] = hl_c
                hc = ch_ctx >= 0
                if hc.any():
                    hc_b, hc_c = fb[hc], ch_ctx[hc]
                    next_front[hc_b, (ff[hc] * 2 + 1) % 4] = hc_c
                    vis_b_list.append(hc_b); vis_u_list.append(hc_c)
                    vis_g_list.append(torch.full((hc_b.numel(),), gv, dtype=torch.float32, device=device))
                if not (hl.any() or hc.any()):
                    break
                frontier = next_front

            node_delay[:, nd_ptr] = emergent_ctx_cand
            nd_ptr = (nd_ptr + 1) % D_ctx

            all_b, all_u, all_g = torch.cat(vis_b_list), torch.cat(vis_u_list), torch.cat(vis_g_list)
            h_dag = torch.zeros((B, D), dtype=torch.float32, device=device)
            h_dag.index_add_(0, all_b, core.W_node[all_u].to(torch.float32) * all_g.unsqueeze(1))
            logits_sharp.zero_()
            sd_a, sw_a = core.sharp_dst[all_u], core.sharp_w[all_u].to(torch.float32)
            sm_a = sd_a >= 0
            if sm_a.any():
                logits_sharp.index_put_(
                    (all_b.unsqueeze(1).expand_as(sd_a)[sm_a], sd_a[sm_a].to(torch.int64)),
                    (sw_a * (all_g * 0.65).unsqueeze(1))[sm_a],
                    accumulate=True,
                )

            h_norm = h_dag / h_dag.norm(dim=1, keepdim=True).clamp(min=1.0)
            e_norm = core.E[x_t] / core.E[x_t].norm(dim=1, keepdim=True).clamp(min=1.0)
            x_slot = 0.65 * h_norm + 0.35 * e_norm
            kg = torch.sigmoid(x_slot @ core.G_keep.T + core.keep_bias.unsqueeze(0))
            wg = torch.sigmoid(x_slot @ core.G_write.T + core.write_bias.unsqueeze(0))
            slot_state = slot_state * kg.unsqueeze(-1) + wg.unsqueeze(-1) * x_slot.unsqueeze(1)
            sn = slot_state.norm(dim=-1, keepdim=True)
            slot_state = slot_state * (2.5 / sn.clamp(min=2.5))
            r_sum = torch.einsum("bsd,sdk->bsk", slot_state, core.W_read).sum(dim=1)
            m_sum = torch.einsum("bsd,sdk->bsk", slot_state, core.W_mod).sum(dim=1)
            h_act = torch.tanh((h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum) * 0.35) * 2.85
            logits = h_act @ core.E.T + core.bias.unsqueeze(0) + logits_sharp

            exec_vec.mul_(cfg.mode_ema_decay).add_(h_dag, alpha=1.0 - cfg.mode_ema_decay)
            if core.mode_ready and t >= 24:
                ev_n = exec_vec / exec_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                mp = torch.softmax((ev_n @ core.mode_centroids.T) / cfg.mode_temp, dim=-1)
                logits = logits + (mp @ core.mode_llr) * cfg.mode_llr_scale

            k4, k7 = core._compute_stm_keys(byte_hist, cur_idx, H_buf)
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

            if (eng_boost_all is not None) or (ep_boost_all is not None and t >= prompt_bytes_len):
                lp_c = torch.log_softmax(logits, dim=-1)
                p_c = torch.exp(lp_c)
                H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
                ent_gate = H_c / (entropy_h0 + H_c)
                if eng_boost_all is not None:
                    eb = eng_boost_all[t].to(device=device, dtype=torch.float32, non_blocking=True)
                    logits = logits + eb * (0.78 * ent_gate)
                if ep_boost_all is not None and t >= prompt_bytes_len:
                    epb = ep_boost_all[t].to(device=device, dtype=torch.float32, non_blocking=True)
                    logits = logits + epb * (ep_scale * ent_gate)

            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]

            if t >= prompt_bytes_len:
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
                    stm_k4[bb, w_slot] = bk4; stm_k7[bb, w_slot] = bk7; stm_val[bb, w_slot] = by
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
            "ep_recall_rate": round(ep_recall_hits / max(1, B), 3),
            "ep_step_hit_rate": round(ep_step_hits / max(1, total_steps), 4),
        }

    @torch.no_grad()
    def generate_with_thought_workspace(
        self,
        prompt: str,
        max_new_bytes: int = 210,
        temperature: float = 0.62,
        top_k: int = 24,
        rollout_k: int = 6,
        rollout_chunk_bytes: int = 42,
        use_episodic_thought: bool = True,
        ep_boost_scale: float = 1.15,
        thought_anchor_beta: float = 0.14,
        allow_high_fidelity_custom: bool = False,
    ) -> Tuple[str, bool, Dict[str, object]]:
        core = self.core
        cfg = core.cfg
        device = self.device
        V, D, S = core.V, core.D, cfg.num_slots
        raw_in = list(prompt.encode("utf-8"))
        prompt_b = bytes(raw_in)
        H_buf = 28
        D_ctx = 12
        N_stm = cfg.stm_slots

        ws = ActiveThoughtWorkspace(device)
        t_q0 = time.perf_counter()
        if use_episodic_thought and len(self.ep_rack.cartridges) > 0:
            recalled = self.ep_rack.query(prompt_b, top_k=3)
            ws.load_episodes(recalled, prompt_b, self.ep_rack, core)
        recall_us = (time.perf_counter() - t_q0) * 1e6

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
                "c_eng": torch.zeros(16, dtype=torch.float32, device=device),
                "tail9": bytearray(),
                "step_i": 0,
            }

        def clone_state(st):
            out = {}
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.clone()
                elif isinstance(v, bytearray):
                    out[k] = bytearray(v)
                else:
                    out[k] = v
            return out

        def step_state(st, b_val: int, next_true: int = -1, locked_mode_p=None, is_prompt: bool = False):
            step_i = st["step_i"]
            st["byte_hist"][0, st["hist_ptr"]] = b_val
            cur_idx = st["hist_ptr"]
            st["hist_ptr"] = (st["hist_ptr"] + 1) % H_buf
            st["tail9"].append(b_val)
            if len(st["tail9"]) > 9:
                del st["tail9"][0]

            b1 = int(st["byte_hist"][0, (cur_idx - 1) % H_buf].item())
            b2 = int(st["byte_hist"][0, (cur_idx - 2) % H_buf].item())
            st["c_eng"], cq_np = self.v20.encoder.step_context_single(st["c_eng"], b_val, b1, b2)

            st["ctx_score"].mul_(0.985)
            if step_i >= D_ctx:
                u_old = int(st["node_delay"][0, st["nd_ptr"]].item())
                if u_old >= V and bool(core.alive[u_old].item()):
                    sal = 1.8 / math.sqrt(1.0 + float(core.fires[u_old].item()) * 0.02)
                    if sal > 0.55 and u_old not in st["ctx_node"][0].tolist():
                        ws_i = int(st["ctx_score"][0].argmin().item())
                        if sal > float(st["ctx_score"][0, ws_i].item()):
                            st["ctx_node"][0, ws_i] = u_old
                            st["ctx_score"][0, ws_i] = sal
            top_ctx = st["ctx_node"][:, int(st["ctx_score"][0].argmax().item())]

            h_dag = core.W_node[b_val].to(torch.float32)
            logits_s = torch.zeros(V, dtype=torch.float32, device=device)
            sd0, sw0 = core.sharp_dst[b_val], core.sharp_w[b_val].to(torch.float32)
            m0 = sd0 >= 0
            if m0.any():
                logits_s.index_add_(0, sd0[m0].to(torch.int64), sw0[m0] * 0.65)

            frontier = [b_val]
            emergent_cand = -1
            for hop in range(1, cfg.max_hops + 1):
                nxt: List[int] = []
                gain = 1.0 / (1.0 + 0.10 * hop)
                for p in frontier:
                    sp = int(core.local_span[p].item())
                    c_loc = int(st["byte_hist"][0, (cur_idx - min(sp + 1, H_buf - 1)) % H_buf].item())
                    if c_loc >= 0:
                        ch = int(core.edges.lookup(
                            torch.tensor([p], device=device), torch.zeros(1, dtype=torch.int64, device=device), torch.tensor([c_loc], device=device)
                        ).item())
                        if ch >= V and bool(core.alive[ch].item()):
                            nxt.append(ch)
                            if 2 <= hop <= 4:
                                emergent_cand = ch
                    if cfg.enable_long_branch and 2 <= hop <= 5 and int(top_ctx[0].item()) >= V:
                        chc = int(core.edges.lookup(
                            torch.tensor([p], device=device), torch.ones(1, dtype=torch.int64, device=device), top_ctx
                        ).item())
                        if chc >= V and bool(core.alive[chc].item()):
                            nxt.append(chc)
                if not nxt:
                    break
                nxt = list(dict.fromkeys(nxt))[:4]
                for u in nxt:
                    h_dag.add_(core.W_node[u].to(torch.float32), alpha=gain)
                    sdu, swu = core.sharp_dst[u], core.sharp_w[u].to(torch.float32)
                    mu = sdu >= 0
                    if mu.any():
                        logits_s.index_add_(0, sdu[mu].to(torch.int64), swu[mu] * (gain * 0.65))
                frontier = nxt

            st["node_delay"][0, st["nd_ptr"]] = emergent_cand
            st["nd_ptr"] = (st["nd_ptr"] + 1) % D_ctx

            h_n = h_dag / h_dag.norm().clamp(min=1.0)
            e_n = core.E[b_val] / core.E[b_val].norm().clamp(min=1.0)
            xs = (0.65 * h_n + 0.35 * e_n).unsqueeze(0)
            kg = torch.sigmoid(xs @ core.G_keep.T + core.keep_bias.unsqueeze(0))
            wg = torch.sigmoid(xs @ core.G_write.T + core.write_bias.unsqueeze(0))
            st["slot_state"] = st["slot_state"] * kg.unsqueeze(-1) + wg.unsqueeze(-1) * xs.unsqueeze(1)
            sn = st["slot_state"].norm(dim=-1, keepdim=True)
            st["slot_state"] = st["slot_state"] * (2.5 / sn.clamp(min=2.5))
            r_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], core.W_read).sum(dim=1)[0]
            m_sum = torch.einsum("bsd,sdk->bsk", st["slot_state"], core.W_mod).sum(dim=1)[0]

            h_core_pre = h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum
            if use_episodic_thought and ws.z_thought is not None:
                h_core_pre = h_core_pre + ws.z_thought * (thought_anchor_beta * h_dag.norm().clamp(min=1.0))
            h_act = torch.tanh(h_core_pre * 0.35) * 2.85
            logits = h_act @ core.E.T + core.bias + logits_s

            st["exec_vec"].mul_(cfg.mode_ema_decay).add_(h_dag.unsqueeze(0), alpha=1.0 - cfg.mode_ema_decay)
            if core.mode_ready:
                ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
                cur_mp = torch.softmax((ev_n @ core.mode_centroids.T) / cfg.mode_temp, dim=-1)
                mp = (0.60 * locked_mode_p + 0.40 * cur_mp) if locked_mode_p is not None else cur_mp
                logits = logits + (mp @ core.mode_llr)[0] * (cfg.mode_llr_scale * 1.15)

            k4, k7 = core._compute_stm_keys(st["byte_hist"], cur_idx, H_buf)
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

            if (not is_prompt) and (len(self.v20.rack.cartridges) > 0 or (use_episodic_thought and ws.recalled)):
                lp_c = torch.log_softmax(logits, dim=-1)
                p_c = torch.exp(lp_c)
                H_c = float(-(p_c * lp_c).sum().item())
                ent_g = H_c / (1.0 + H_c)

                if len(self.v20.rack.cartridges) > 0:
                    ks = self.v20.encoder.step_keys_single(st["byte_hist"], cur_idx, H_buf)
                    eb_np, _, _ = self.v20.rack.lookup_batch(
                        [ks[bi : bi + 1] for bi in range(6)], cq_np.reshape(1, 16), use_context_gate=True
                    )
                    logits = logits + torch.from_numpy(eb_np[0]).to(device) * (0.72 * ent_g)

                if use_episodic_thought and ws.recalled:
                    ep_bv = ws.lookup_boost_vec(
                        bytes(st["tail9"]), scale=ep_boost_scale,
                        max_boost=3.2 if allow_high_fidelity_custom else 2.4,
                        include_entity_k3=True,
                    )
                    if ep_bv:
                        ep_g = max(ent_g, 0.70 if allow_high_fidelity_custom else 0.45)
                        for kb, vw in ep_bv.items():
                            logits[kb] = logits[kb] + vw * ep_g

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
            return logits, h_dag

        st = init_state()
        if use_episodic_thought and ws.recalled:
            top_ep_b = ws.recalled[0][2][:180]
            tmp_hist = torch.full((1, H_buf), -1, dtype=torch.int64, device=device)
            tmp_ptr = 0
            for idx_e, eb_v in enumerate(top_ep_b):
                tmp_hist[0, tmp_ptr] = eb_v
                c_i = tmp_ptr
                tmp_ptr = (tmp_ptr + 1) % H_buf
                if idx_e >= 4 and idx_e + 1 < len(top_ep_b):
                    ek4, ek7 = core._compute_stm_keys(tmp_hist, c_i, H_buf)
                    if int(ek4[0].item()) != 0:
                        sp_i = st["stm_ptr"]
                        st["stm_k4"][0, sp_i] = ek4[0]
                        st["stm_k7"][0, sp_i] = ek7[0]
                        st["stm_val"][0, sp_i] = top_ep_b[idx_e + 1]
                        st["stm_str"][0, sp_i] = 0.85 if allow_high_fidelity_custom else 0.45
                        st["stm_ptr"] = (sp_i + 1) % N_stm

        prompt_hs = []
        logits = None
        for i, b in enumerate(raw_in):
            nxt_t = raw_in[i + 1] if i + 1 < len(raw_in) else -1
            logits, hd = step_state(st, b, next_true=nxt_t, locked_mode_p=None, is_prompt=True)
            prompt_hs.append(hd)

        prompt_mode_vec = torch.stack(prompt_hs).mean(dim=0)
        prompt_mode_vec = prompt_mode_vec / prompt_mode_vec.norm().clamp(min=1e-6)
        locked_mode_p = (
            torch.softmax(((prompt_mode_vec.unsqueeze(0) @ core.mode_centroids.T) / cfg.mode_temp), dim=-1)
            if core.mode_ready else None
        )

        out_bytes = list(raw_in)
        gen_only: List[int] = []
        recent_ngrams: set = set()
        used_topic_spans: set = set()
        for sp_b, _ in ws.topic_spans:
            if sp_b in prompt_b:
                used_topic_spans.add(sp_b)

        hit_entities_log: List[str] = []
        rem = max_new_bytes
        while rem > 0:
            c_len = min(rollout_chunk_bytes, rem)
            k_cand = rollout_k if use_episodic_thought else max(1, min(4, rollout_k))

            best_score = -1e9
            best_pack = None
            for _ in range(k_cand):
                cst = clone_state(st)
                clog = logits
                c_bytes: List[int] = []
                c_hs = []
                lp_sum = 0.0
                rep_pen = 0.0
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
                            rep_pen += 0.70
                        local_seen.add(ng)
                    clog, chd = step_state(cst, nxt_b, locked_mode_p=locked_mode_p, is_prompt=False)
                    c_hs.append(chd)

                raw_c = bytes(c_bytes)
                utf8_bonus = 0.0
                for trim in (0, 1, 2):
                    sub = raw_c[: len(raw_c) - trim] if trim else raw_c
                    try:
                        sub.decode("utf-8")
                        utf8_bonus = 0.40
                        break
                    except UnicodeDecodeError:
                        continue

                c_mean = torch.stack(c_hs).mean(dim=0)
                c_mean_n = c_mean / c_mean.norm().clamp(min=1e-6)
                anchor_vec = ws.z_thought if (use_episodic_thought and ws.z_thought is not None) else prompt_mode_vec
                thought_sim = float((c_mean_n @ anchor_vec).item())

                ent_bonus = 0.0
                c_new_ents: List[bytes] = []
                if use_episodic_thought and ws.topic_spans:
                    window_c = bytes(out_bytes[-5:]) + raw_c
                    for sp_b, idf_v in ws.topic_spans:
                        if sp_b in window_c:
                            if sp_b not in used_topic_spans:
                                ent_bonus += 0.55 * min(idf_v / 8.0, 1.5)
                                c_new_ents.append(sp_b)
                            else:
                                ent_bonus += 0.12

                copy_pen = 0.0
                if (not allow_high_fidelity_custom) and ws.episode_ngrams18 and len(raw_c) >= 18:
                    for pi in range(0, len(raw_c) - 17, 6):
                        if raw_c[pi : pi + 18] in ws.episode_ngrams18:
                            copy_pen += 0.35

                fluency = lp_sum / max(1, c_len)
                score = fluency + 1.10 * thought_sim + min(ent_bonus, 2.2) + utf8_bonus - rep_pen - copy_pen
                if score > best_score:
                    best_score = score
                    best_pack = (cst, clog, c_bytes, c_new_ents)

            st, logits, chosen_b, chosen_ents = best_pack
            out_bytes.extend(chosen_b)
            gen_only.extend(chosen_b)
            for sp_b in chosen_ents:
                used_topic_spans.add(sp_b)
                dec_s = sp_b.decode("utf-8", errors="ignore")
                if dec_s and dec_s not in hit_entities_log:
                    hit_entities_log.append(dec_s)
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

        gen_topic_hits = [
            sp_b.decode("utf-8", errors="ignore")
            for sp_b, _ in ws.topic_spans
            if sp_b in raw_gen
        ]
        recalled_preview = [
            {"source": cn, "score": round(sc, 1), "snippet": eb[:72].decode("utf-8", errors="ignore").replace("\n", " ")}
            for sc, cn, eb in ws.recalled
        ]
        return bytes(out_bytes).decode("utf-8", errors="replace"), valid_utf8, {
            "recall_us": round(recall_us, 1),
            "recalled": recalled_preview,
            "active_entities": [sp.decode("utf-8", errors="ignore") for sp, _ in ws.topic_spans[:10]],
            "gen_entity_hits": list(dict.fromkeys(gen_topic_hits))[:10],
            "num_gen_entity_hits": len(set(gen_topic_hits)),
        }
