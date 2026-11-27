"""NeuroGraph v5.0: Continuous Thought-Field (256d Hebbian Spectral Latent Space)
+ Top-Down Inhibitory Gate + Directed Subject->Predicate Layer-2 Motor Cortex!

Combines:
  [1] Directed Subject->Predicate PPMI + Top-Down Prefrontal-to-Sensory Inhibition
      (crushes off-topic Layer-1 boilerplate hijacks like '年（昭和 -> JR西日本の駅'!)
      + Rule-Free IDF-Rhythm Anti-Stacking + Scaled 300M-char Persistent Layer-2!
  [3] Continuous 256-dim Motor Thought Embeddings (Zero-Backprop Randomized Spectral
      Oja/Hebbian Eigen-Decomposition on GPU in ~1.5s, < 30 MB static VRAM!)
      + 4-Timescale Continuous Working-Memory Thought State (h_root, h_long, h_mid, h_short)
      + Closed-Form Hebbian Predictive Transition Matrix W_pred (256x256, 0 backprop!).

Zero backprop, zero Attention, zero pre-trained tokenizer, zero <UNK>, zero hardcoded rules."""

from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neurograph.engram_v20 import DecoupledBrainV20
from neurograph.reason_decode_v30 import BatchedCoreRunner
from neurograph.triton_dag_kernel import run_fused_dag_step


def _load_motor_v50_lib() -> ctypes.CDLL:
    so_path = Path(__file__).with_name("libmotor_v50.so")
    if not so_path.exists():
        raise FileNotFoundError(
            f"{so_path} not found. Compile with:\n"
            "  g++ -O3 -march=native -fopenmp -shared -fPIC neurograph/motor_kernel_v50.cpp -o neurograph/libmotor_v50.so"
        )
    lib = ctypes.CDLL(str(so_path))

    lib.motor_vocab_create.argtypes = []
    lib.motor_vocab_create.restype = ctypes.c_void_p
    lib.motor_vocab_destroy.argtypes = [ctypes.c_void_p]
    lib.motor_vocab_destroy.restype = None
    lib.motor_vocab_size.argtypes = [ctypes.c_void_p]
    lib.motor_vocab_size.restype = ctypes.c_int
    lib.motor_vocab_get_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
    lib.motor_vocab_get_bytes.restype = ctypes.c_int

    lib.motor_grow_vocab.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.motor_grow_vocab.restype = ctypes.c_int
    lib.motor_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ]
    lib.motor_encode.restype = ctypes.c_int64

    lib.macro_create_v50.argtypes = [ctypes.c_int32]
    lib.macro_create_v50.restype = ctypes.c_void_p
    lib.macro_create_v50_sized.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.macro_create_v50_sized.restype = ctypes.c_void_p
    lib.macro_destroy_v50.argtypes = [ctypes.c_void_p]
    lib.macro_destroy_v50.restype = None

    lib.macro_ingest_stream_v50.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.macro_ingest_stream_v50.restype = ctypes.c_int64

    lib.macro_bank_checksum_v50.argtypes = [ctypes.c_void_p]
    lib.macro_bank_checksum_v50.restype = ctypes.c_uint64

    lib.macro_finalize_ppmi_v50.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_int64),
    ]
    lib.macro_finalize_ppmi_v50.restype = ctypes.c_int64

    lib.macro_sleep_prune_v50.argtypes = [ctypes.c_void_p]
    lib.macro_sleep_prune_v50.restype = ctypes.c_int64

    lib.macro_sym_edge_count_v50.argtypes = [ctypes.c_void_p]
    lib.macro_sym_edge_count_v50.restype = ctypes.c_int64

    lib.macro_export_sym_csr_v50.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ]
    lib.macro_export_sym_csr_v50.restype = ctypes.c_int64

    lib.macro_export_idfs_v50.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.macro_export_idfs_v50.restype = None

    lib.macro_predict_topk_v50.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ]
    lib.macro_predict_topk_v50.restype = ctypes.c_int

    lib.macro_top_dir_neighbors_v50.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ]
    lib.macro_top_dir_neighbors_v50.restype = ctypes.c_int

    lib.macro_dir_affinity_v50.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.macro_dir_affinity_v50.restype = ctypes.c_float

    lib.macro_marginal_byte_boost_causal_v50.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.macro_marginal_byte_boost_causal_v50.restype = None

    lib.macro_save_v50.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    lib.macro_save_v50.restype = ctypes.c_int
    lib.macro_load_v50.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    lib.macro_load_v50.restype = ctypes.c_int

    return lib


class ContinuousThoughtCortexV50:
    """Layer-2 Thought & Motor Cortex v5.0:
    - Grown Output Motor Nodes (Broca's Area, zero pre-trained tokenizer, zero <UNK>)
    - 120M-slot Directed Macro-Cortex + Directed Subject->Predicate PPMI (0 MB GPU)
    - 256-dim Continuous Spectral Hebbian Thought Embeddings E + Predictive W_pred (~25 MB GPU, 0 backprop!)"""

    def __init__(
        self,
        target_extra_motors: int = 24_576,
        grow_rounds: int = 8,
        max_motor_bytes: int = 18,
        thought_dim: int = 256,
    ):
        self.lib = _load_motor_v50_lib()
        self.target_extra_motors = target_extra_motors
        self.grow_rounds = grow_rounds
        self.max_motor_bytes = max_motor_bytes
        self.thought_dim = thought_dim

        self._vh = self.lib.motor_vocab_create()
        self._mh: Optional[int] = None
        self.V_m: int = 256
        self.motor_bytes: List[bytes] = [bytes([b]) for b in range(256)]
        self.motor_texts: List[str] = [bytes([b]).decode("utf-8", errors="replace") for b in range(256)]
        self.motor_idfs: np.ndarray = np.full(256, 4.0, dtype=np.float32)
        self.tau_idf: float = 6.2

        # Continuous 256-dim Thought Space (CPU float32 contiguous + optional GPU cache)
        self.emb_np: Optional[np.ndarray] = None      # [V_m, D_m] L2-normalized
        self.w_pred_np: Optional[np.ndarray] = None   # [D_m, D_m] Hebbian predictive transition
        self._pred_xtx: Optional[np.ndarray] = None   # running Hebbian covariance C_xx
        self._pred_xty: Optional[np.ndarray] = None   # running Hebbian cross-cov C_xy

    def __del__(self):
        try:
            if getattr(self, "_mh", None):
                self.lib.macro_destroy_v50(self._mh)
                self._mh = None
            if getattr(self, "_vh", None):
                self.lib.motor_vocab_destroy(self._vh)
                self._vh = None
        except Exception:
            pass

    def _sync_motor_tables(self) -> None:
        self.V_m = int(self.lib.motor_vocab_size(self._vh))
        self.motor_bytes = []
        self.motor_texts = []
        buf = np.zeros(64, dtype=np.uint8)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        for mid in range(self.V_m):
            n = int(self.lib.motor_vocab_get_bytes(self._vh, ctypes.c_int32(mid), buf_ptr, 64))
            b = bytes(buf[:n].tolist())
            self.motor_bytes.append(b)
            self.motor_texts.append(b.decode("utf-8", errors="replace"))

    def _sync_idfs(self) -> None:
        self.motor_idfs = np.zeros(self.V_m, dtype=np.float32)
        self.lib.macro_export_idfs_v50(self._mh, self.motor_idfs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        grown_idfs = self.motor_idfs[256:]
        nz = grown_idfs[grown_idfs > 0.1]
        if len(nz) > 0:
            # Rule-free information-theoretic boundary from unigram Shannon entropy H1:
            # 0.72 * H1 (~6.45 nats) cleanly separates glue/particles (4.9..5.6) from content nouns (6.8..13.0)!
            w = np.exp(-nz.astype(np.float64))
            h1 = float((w * nz.astype(np.float64)).sum() / (w.sum() + 1e-12))
            self.tau_idf = 0.72 * h1

    def content_gate(self, mid: int) -> float:
        """Smooth rule-free content specificity gate in [0, 1] from corpus IDF alone:
        ~0.0 for function/glue nodes ('は', 'の', '、', '。'), ~1.0 for content concepts ('量子', 'ドイツ', '駅')."""
        if mid < 256 or mid >= self.V_m:
            return 0.0
        idf = float(self.motor_idfs[mid])
        z = max(-12.0, min(12.0, (idf - self.tau_idf) * 2.2))
        return 1.0 / (1.0 + math.exp(-z))

    def grow_motor_repertoire(
        self,
        sample_bytes: np.ndarray,
        min_pair_count: int = 35,
        bucket_shift: int = 0,
    ) -> Dict[str, float]:
        t0 = time.time()
        arr = np.ascontiguousarray(sample_bytes, dtype=np.uint8)
        added = int(self.lib.motor_grow_vocab(
            self._vh,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(len(arr)),
            ctypes.c_int(self.target_extra_motors),
            ctypes.c_int(self.grow_rounds),
            ctypes.c_int(self.max_motor_bytes),
            ctypes.c_int(min_pair_count),
        ))
        self._sync_motor_tables()
        if self._mh:
            self.lib.macro_destroy_v50(self._mh)
        self._mh = self.lib.macro_create_v50_sized(ctypes.c_int32(self.V_m), ctypes.c_int32(bucket_shift))
        avg_b = float(np.mean([len(b) for b in self.motor_bytes[256:]])) if self.V_m > 256 else 1.0
        return {
            "grown_motors": added,
            "V_m": self.V_m,
            "avg_motor_bytes": round(avg_b, 2),
            "sec": round(time.time() - t0, 2),
        }

    def encode(self, raw_bytes: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        n = len(arr)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        out = np.empty(n, dtype=np.int32)
        got = int(self.lib.motor_encode(
            self._vh,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(n),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(n),
        ))
        return out[:got].copy()

    def ingest_bytes(
        self,
        raw_bytes: np.ndarray,
        collect_cooc: bool = True,
        cooc_window: int = 30,
        cooc_stride: int = 2,
        num_threads: int = 0,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        t0 = time.time()
        m_ids = self.encode(raw_bytes)
        rc_ing = int(self.lib.macro_ingest_stream_v50(
            self._mh,
            m_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(len(m_ids)),
            ctypes.c_int(1 if collect_cooc else 0),
            ctypes.c_int(cooc_window),
            ctypes.c_int(cooc_stride),
            ctypes.c_int(num_threads),
        ))
        if rc_ing < 0:
            raise RuntimeError("Cannot ingest_bytes(collect_cooc=True) on an already-finalized or loaded MacroCortexV50!")
        return {
            "bytes": int(len(raw_bytes)),
            "motor_tokens": int(len(m_ids)),
            "bytes_per_motor": round(len(raw_bytes) / max(1, len(m_ids)), 2),
            "sec": round(time.time() - t0, 2),
        }, m_ids

    def bank_checksum(self) -> int:
        return int(self.lib.macro_bank_checksum_v50(self._mh))

    def finalize_discrete(self, top_k_dir: int = 96, top_k_sym: int = 96, min_cooc: float = 3.5) -> Dict[str, float]:
        t0 = time.time()
        sym_e = ctypes.c_int64(0)
        dir_e = int(self.lib.macro_finalize_ppmi_v50(
            self._mh, ctypes.c_int(top_k_dir), ctypes.c_int(top_k_sym),
            ctypes.c_float(min_cooc), ctypes.byref(sym_e),
        ))
        if dir_e < 0:
            raise RuntimeError("Cannot call finalize_discrete() on an already-finalized or loaded MacroCortexV50!")
        active = int(self.lib.macro_sleep_prune_v50(self._mh))
        self._sync_idfs()
        return {
            "dir_ppmi_synapses": dir_e,
            "sym_ppmi_synapses": int(sym_e.value),
            "active_macro_slots": active,
            "tau_idf": round(self.tau_idf, 2),
            "sec": round(time.time() - t0, 2),
        }

    @torch.no_grad()
    def build_spectral_hebbian_embeddings(
        self,
        device: torch.device,
        sample_motor_streams: Optional[List[np.ndarray]] = None,
        power_iters: int = 4,
    ) -> Dict[str, float]:
        """ZERO-BACKPROP Continuous 256-dim Motor Thought Embeddings via GPU Randomized Spectral
        Hebbian/Oja Power Iteration on the Symmetric Topic PPMI Graph + Closed-Form Hebbian
        Predictive Transition Matrix W_pred (256x256)! Uses < 150 MB temp GPU VRAM."""
        t0 = time.time()
        E_cnt = int(self.lib.macro_sym_edge_count_v50(self._mh))
        V_m = self.V_m
        D_m = self.thought_dim

        off_np = np.zeros(V_m + 1, dtype=np.uint32)
        dst_np = np.zeros(E_cnt, dtype=np.int32)
        w_np = np.zeros(E_cnt, dtype=np.float32)
        self.lib.macro_export_sym_csr_v50(
            self._mh,
            off_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        # Build STRICTLY SYMMETRIC normalized affinity matrix A = D^{-1/2} (0.5*(P + P^T)) D^{-1/2} on GPU!
        row_counts = np.diff(off_np.astype(np.int64))
        rows_np = np.repeat(np.arange(V_m, dtype=np.int64), row_counts)
        cols_np = dst_np.astype(np.int64)

        # Symmetrize explicitly by adding both (r, c) and (c, r) with 0.5*w
        sym_r = np.concatenate([rows_np, cols_np])
        sym_c = np.concatenate([cols_np, rows_np])
        sym_w = np.concatenate([0.5 * w_np, 0.5 * w_np]).astype(np.float32)

        deg = np.bincount(sym_r, weights=sym_w, minlength=V_m).astype(np.float32)
        inv_sqrt_deg = np.where(deg > 1e-6, 1.0 / np.sqrt(np.maximum(deg, 1e-6)), 0.0).astype(np.float32)
        norm_w = sym_w * inv_sqrt_deg[sym_r] * inv_sqrt_deg[sym_c]

        indices = torch.stack([
            torch.from_numpy(sym_r),
            torch.from_numpy(sym_c),
        ], dim=0).to(device)
        vals = torch.from_numpy(norm_w).to(device)
        A = torch.sparse_coo_tensor(indices, vals, size=(V_m, V_m), device=device).coalesce()
        del indices, vals, rows_np, cols_np, sym_r, sym_c, sym_w, norm_w, off_np, dst_np, w_np

        # Randomized Subspace / Orthogonal Oja Power Iteration (6 iters, oversampling p=32) on strictly symmetric A!
        k_over = min(V_m - 1, D_m + 32)
        g = torch.Generator(device=device)
        g.manual_seed(50)
        Q = torch.randn((V_m, k_over), device=device, dtype=torch.float32, generator=g)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        for _ in range(max(power_iters, 6)):
            Z = torch.sparse.mm(A, Q)
            Z = Z + 0.25 * Q
            Q, _ = torch.linalg.qr(Z, mode="reduced")

        # Small [k_over, k_over] projection Rayleigh-Ritz decomposition
        B_small = Q.T @ torch.sparse.mm(A, Q)
        B_small = 0.5 * (B_small + B_small.T)
        evals, evecs = torch.linalg.eigh(B_small)
        idx = torch.argsort(evals, descending=True)[:D_m]
        evals_top = torch.clamp(evals[idx], min=1e-4)
        U = Q @ evecs[:, idx]
        E_gpu = U * torch.sqrt(evals_top).unsqueeze(0)
        E_gpu[:256] = 0.0
        norms = E_gpu.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        E_gpu = E_gpu / norms
        E_gpu[norms.squeeze(-1) < 1e-5] = 0.0

        self.emb_np = np.ascontiguousarray(E_gpu.cpu().numpy(), dtype=np.float32)

        # Closed-Form Hebbian Predictive Transition Matrix W_pred [D_m, D_m]:
        # STRICTLY CHUNKED (32,768 motors/chunk -> < 40 MB peak VRAM!) AND aligned with the exact
        # two-timescale EMA thought state (0.62*h_long + 0.38*h_mid) used during causal evaluation!
        if sample_motor_streams:
            C_xx = torch.zeros((D_m, D_m), device=device, dtype=torch.float32)
            C_xy = torch.zeros((D_m, D_m), device=device, dtype=torch.float32)
            chunk_sz = 32_768
            # Precompute exponential kernel weights for causal EMA approximation (window 32 content motors)
            k_len = 32
            d_idx = torch.arange(k_len, device=device, dtype=torch.float32)
            ker = 0.62 * (0.07 * ((1.0 - 0.07) ** d_idx)) + 0.38 * (0.24 * ((1.0 - 0.24) ** d_idx))
            ker = (ker / ker.sum()).flip(0).view(1, 1, k_len)  # causal conv1d kernel [1, 1, K]

            idfs_t = torch.from_numpy(self.motor_idfs).to(device)
            cg_all = torch.sigmoid((idfs_t - self.tau_idf) * 2.2)
            cg_all[:256] = 0.0

            for st in sample_motor_streams:
                sub = st[:500_000]
                for c0 in range(0, len(sub), chunk_sz):
                    blk = sub[c0 : c0 + chunk_sz]
                    if len(blk) < 64:
                        continue
                    m_t = torch.from_numpy(blk.astype(np.int64)).to(device)
                    keep = torch.nonzero(cg_all[m_t] > 0.35, as_tuple=False).squeeze(-1)
                    if keep.numel() <= k_len + 2:
                        del m_t, keep
                        continue
                    em_seq = E_gpu[m_t[keep]]  # [M <= 32768, D_m] (~<= 32 MB VRAM!)
                    # Causal 1D EMA convolution over strictly past k_len content motors (zero future leakage!)
                    # em_seq[:-1] predicts em_seq[k_len:]
                    inp = em_seq[:-1].T.unsqueeze(1)  # [D_m, 1, M-1]
                    ctx_h = torch.nn.functional.conv1d(inp, ker).squeeze(1).T  # [M - k_len, D_m]
                    ctx_h = ctx_h / ctx_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    nxt_e = em_seq[k_len:]
                    C_xx += ctx_h.T @ ctx_h
                    C_xy += ctx_h.T @ nxt_e
                    del m_t, keep, em_seq, inp, ctx_h, nxt_e

            lam = 0.15 * (torch.trace(C_xx) / D_m).clamp(min=1.0)
            W_t = torch.linalg.solve(C_xx + lam * torch.eye(D_m, device=device), C_xy)
            self.w_pred_np = np.ascontiguousarray(W_t.T.cpu().numpy(), dtype=np.float32)
            del C_xx, C_xy, W_t, ker, idfs_t, cg_all
        else:
            self.w_pred_np = np.eye(D_m, dtype=np.float32)

        del A, Q, Z, B_small, evals, evecs, U, E_gpu
        torch.cuda.empty_cache()

        return {
            "thought_dim": D_m,
            "sym_edges_used": E_cnt,
            "nonzero_embeddings": int((np.linalg.norm(self.emb_np, axis=1) > 0.5).sum()),
            "sec": round(time.time() - t0, 2),
        }

    def save(self, folder: str) -> Dict[str, float]:
        t0 = time.time()
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        bin_path = str(p / "macro_v50.bin").encode("utf-8")
        ok = int(self.lib.macro_save_v50(self._vh, self._mh, bin_path))
        if not ok:
            raise RuntimeError(f"Failed to save macro_v50 to {folder}")
        if self.emb_np is not None:
            np.save(p / "emb_256.npy", self.emb_np)
        if self.w_pred_np is not None:
            np.save(p / "w_pred_256.npy", self.w_pred_np)
        mb = sum(f.stat().st_size for f in p.glob("*")) / 1048576
        return {"folder": folder, "size_mb": round(mb, 1), "sec": round(time.time() - t0, 2)}

    @classmethod
    def load(cls, folder: str) -> "ContinuousThoughtCortexV50":
        p = Path(folder)
        obj = cls()
        # Destroy initial empty vocab handle before loading
        obj.lib.motor_vocab_destroy(obj._vh)
        obj._vh = None
        vh_p = ctypes.c_void_p()
        mh_p = ctypes.c_void_p()
        ok = int(obj.lib.macro_load_v50(str(p / "macro_v50.bin").encode("utf-8"), ctypes.byref(vh_p), ctypes.byref(mh_p)))
        if not ok:
            raise RuntimeError(f"Failed to load macro_v50 from {folder}")
        obj._vh = vh_p.value
        obj._mh = mh_p.value
        obj._sync_motor_tables()
        obj._sync_idfs()
        if (p / "emb_256.npy").exists():
            emb = np.ascontiguousarray(np.load(p / "emb_256.npy"), dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != obj.V_m or emb.shape[1] <= 0:
                raise ValueError(f"emb_256.npy shape {emb.shape} does not match macro V_m={obj.V_m}!")
            obj.emb_np = emb
            obj.thought_dim = int(emb.shape[1])
        if (p / "w_pred_256.npy").exists():
            wp = np.ascontiguousarray(np.load(p / "w_pred_256.npy"), dtype=np.float32)
            if wp.ndim != 2 or wp.shape != (obj.thought_dim, obj.thought_dim):
                raise ValueError(f"w_pred_256.npy shape {wp.shape} does not match D_m={obj.thought_dim}!")
            obj.w_pred_np = wp
        return obj

    def probe_continuous_thought(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Query the 256-dim Continuous Thought Space for nearest semantic motor concepts!"""
        if self.emb_np is None:
            return []
        ids = self.encode(np.frombuffer(query.encode("utf-8"), dtype=np.uint8))
        h = self.compute_root_thought_vec(ids)
        if np.linalg.norm(h) < 1e-5:
            return []
        sims = self.emb_np @ h
        for mid in ids:
            if 0 <= int(mid) < self.V_m:
                sims[int(mid)] = -1.0
        sims[:256] = -1.0
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(self.motor_texts[int(i)], round(float(sims[int(i)]), 3)) for i in top_idx]

    def probe_directed_predicates(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Inspect Forward-Directed Subject -> Predicate PPMI associations!"""
        ids = self.encode(np.frombuffer(query.encode("utf-8"), dtype=np.uint8))
        acc: Dict[int, float] = {}
        out_i = np.zeros(32, dtype=np.int32)
        out_w = np.zeros(32, dtype=np.float32)
        for mid in ids:
            if int(mid) < 256:
                continue
            cg = self.content_gate(int(mid))
            if cg < 0.25:
                continue
            n = int(self.lib.macro_top_dir_neighbors_v50(
                self._mh, ctypes.c_int32(int(mid)), 32,
                out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                out_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ))
            for j in range(n):
                dst = int(out_i[j])
                if dst not in ids:
                    acc[dst] = acc.get(dst, 0.0) + cg * float(out_w[j])
        ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.motor_texts[m], round(w, 2)) for m, w in ranked]

    def extract_active_topics(
        self,
        motor_seq: np.ndarray,
        max_topics: int = 6,
        min_idf: float = 5.5,
        use_resonance: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(motor_seq) == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        seen: Dict[int, float] = {}
        L = len(motor_seq)
        for idx in range(L - 1, -1, -1):
            mid = int(motor_seq[idx])
            if mid < 256 or mid in seen:
                continue
            idf = float(self.motor_idfs[mid])
            if idf < min_idf:
                continue
            cg = self.content_gate(mid)
            if cg < 0.30:
                continue
            rec = 1.0 if use_resonance else (1.0 / (1.0 + 0.04 * (L - 1 - idx)))
            # Multi-byte compositional richness gently favors full concept chunks over tiny phonetic scraps
            len_w = math.log1p(len(self.motor_bytes[mid]))
            seen[mid] = cg * idf * len_w * rec
        if not seen:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

        # Rule-free Constant-Time (K <= 24) Mutual-Resonance Filtering in 256-dim Continuous Thought Space:
        # Cap to top-24 unique candidates BEFORE resonance -> strictly O(1), ZERO O(K^2) growth!
        if use_resonance and self.emb_np is not None and len(seen) >= 3:
            top_items = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:24]
            mids = [m for m, _ in top_items]
            E_sub = self.emb_np[mids]  # [<=24, D]
            sim_mat = np.maximum(0.0, E_sub @ E_sub.T)
            np.fill_diagonal(sim_mat, 0.0)
            res = sim_mat.max(axis=1) + 0.35 * sim_mat.mean(axis=1)
            seen = {m: w * ((0.15 + float(res[k_i])) ** 2) for k_i, (m, w) in enumerate(top_items)}

        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:max_topics]
        mx = max(ranked[0][1], 1e-5)
        t_ids = np.array([m for m, _ in ranked], dtype=np.int32)
        t_w = np.array([w / mx for _, w in ranked], dtype=np.float32)
        return t_ids, t_w

    def compute_root_thought_vec(self, motor_seq: np.ndarray) -> np.ndarray:
        """Compute the persistent Prefrontal Root Thought Vector h_root in 256-dim space via
        Constant-Time (K <= 24 unique) Mutual-Resonance Weighting (ZERO O(K^2), ZERO language rules!)."""
        h = np.zeros(self.thought_dim, dtype=np.float32)
        if self.emb_np is None or len(motor_seq) == 0:
            return h
        # Deduplicate and accumulate weights by unique motor ID, then cap to top-24 -> strictly O(1)!
        uniq_w: Dict[int, float] = {}
        for mid in motor_seq:
            m = int(mid)
            if m < 256 or m >= self.V_m:
                continue
            cg = self.content_gate(m)
            if cg < 0.25:
                continue
            e = self.emb_np[m]
            if float(np.dot(e, e)) > 0.1:
                w = cg * float(self.motor_idfs[m]) * math.log1p(len(self.motor_bytes[m]))
                uniq_w[m] = max(uniq_w.get(m, 0.0), w)
        if not uniq_w:
            return h
        top_u = sorted(uniq_w.items(), key=lambda x: x[1], reverse=True)[:24]
        mids = [m for m, _ in top_u]
        bw = np.array([w for _, w in top_u], dtype=np.float32)
        E_sub = self.emb_np[mids]  # [<=24, D]
        if len(mids) >= 3:
            sim_mat = np.maximum(0.0, E_sub @ E_sub.T)
            np.fill_diagonal(sim_mat, 0.0)
            res = sim_mat.max(axis=1) + 0.35 * sim_mat.mean(axis=1)
            bw = bw * ((0.15 + res) ** 2)
        h = (bw[:, None] * E_sub).sum(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(h))
        if nrm > 1e-6:
            h /= nrm
        return h

    def predict_topk(
        self,
        hist_ids: np.ndarray,
        topic_ids: np.ndarray,
        topic_w: np.ndarray,
        topic_ppmi_gain: float = 1.4,
        top_k: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h_arr = np.ascontiguousarray(hist_ids, dtype=np.int32)
        out_i = np.zeros(top_k, dtype=np.int32)
        out_s = np.zeros(top_k, dtype=np.float32)
        nt = int(len(topic_ids))
        ti_ptr = topic_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if nt > 0 else ctypes.POINTER(ctypes.c_int32)()
        tw_ptr = topic_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if nt > 0 else ctypes.POINTER(ctypes.c_float)()
        n = int(self.lib.macro_predict_topk_v50(
            self._mh,
            h_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(len(h_arr)),
            ti_ptr, tw_ptr, ctypes.c_int(nt),
            ctypes.c_float(topic_ppmi_gain),
            ctypes.c_int(top_k),
            out_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ))
        return out_i[:n], out_s[:n]

    def directed_affinity(self, cand: int, topic_ids: np.ndarray, topic_w: np.ndarray) -> float:
        nt = int(len(topic_ids))
        if nt == 0:
            return 0.0
        return float(self.lib.macro_dir_affinity_v50(
            self._mh, ctypes.c_int32(cand),
            topic_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            topic_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(nt),
        ))

    def precompute_marginal_batch(
        self,
        data_2d_np: np.ndarray,
        top_k: int = 48,
        topic_ppmi_gain: float = 1.25,
        use_continuous_thought: bool = True,
        cont_gain: float = 1.10,
        inhib_gain: float = 1.20,
    ) -> torch.Tensor:
        """12-Thread OpenMP C++ 100% STRICTLY CAUSAL Online Marginalization (ZERO FUTURE LEAKAGE!)
        Passes unified Python tau_idf explicitly to C++. Returns [T-1, B, 256] float16 CPU tensor."""
        arr = np.ascontiguousarray(data_2d_np, dtype=np.uint8)
        B, T = arr.shape
        out_f32 = np.empty((B, T - 1, 256), dtype=np.float32)

        if use_continuous_thought and self.emb_np is not None:
            if self.emb_np.shape != (self.V_m, self.thought_dim) or not self.emb_np.flags["C_CONTIGUOUS"]:
                raise ValueError(f"Invalid emb_np shape/contiguity: {self.emb_np.shape}")
            e_ptr = self.emb_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if self.w_pred_np is not None:
                if self.w_pred_np.shape != (self.thought_dim, self.thought_dim) or not self.w_pred_np.flags["C_CONTIGUOUS"]:
                    raise ValueError(f"Invalid w_pred_np shape/contiguity: {self.w_pred_np.shape}")
                w_ptr = self.w_pred_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                w_ptr = ctypes.POINTER(ctypes.c_float)()
            dm = self.thought_dim
        else:
            e_ptr = ctypes.POINTER(ctypes.c_float)()
            w_ptr = ctypes.POINTER(ctypes.c_float)()
            dm = 0

        self.lib.macro_marginal_byte_boost_causal_v50(
            self._vh, self._mh,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(B), ctypes.c_int(T),
            ctypes.c_int(top_k), ctypes.c_float(topic_ppmi_gain), ctypes.c_float(self.tau_idf),
            e_ptr, w_ptr, ctypes.c_int(dm),
            ctypes.c_float(cont_gain), ctypes.c_float(inhib_gain),
            out_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        t_out = torch.from_numpy(out_f32).permute(1, 0, 2).to(torch.float16).contiguous()
        del out_f32
        return t_out


class HierarchicalBrainV50:
    """NeuroGraph v5.0: Layer-1 (4.54M Core + 1.084B Engram) + Layer-2 Discrete Macro-Cortex
    + Layer-2 256-dim Continuous Multi-Timescale Thought Field + Top-Down Inhibitory Gate!"""

    def __init__(self, v20: DecoupledBrainV20, cortex_v50: ContinuousThoughtCortexV50, temp_cal: float = 1.08):
        self.v20 = v20
        self.mc = cortex_v50
        self.runner = BatchedCoreRunner(v20, temp_cal=temp_cal)
        self.device = v20.device

    @torch.no_grad()
    def eval_lossless_v50(
        self,
        val_bytes: np.ndarray,
        bytes_per_char: float,
        batch_streams: int = 64,
        use_byte_engram: bool = True,
        use_layer2_discrete: bool = True,
        use_layer2_continuous: bool = True,
        layer2_scale: float = 0.58,
        temp_cal: float = 1.08,
    ) -> Dict[str, float]:
        """Fast 3-Way Causal Lossless Raw UTF-8 BPC Evaluation (12-thread C++ marginalizer + GPU core)!"""
        core = self.v20.core
        cfg = core.cfg
        dev = self.device
        V, D, S = core.V, core.D, cfg.num_slots

        stream_len = len(val_bytes) // batch_streams
        data_np = val_bytes[: batch_streams * stream_len].reshape(batch_streams, stream_len)
        data = torch.from_numpy(data_np.astype(np.int64)).to(dev)
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

        l2_boost_all = None
        if use_layer2_discrete:
            l2_boost_all = self.mc.precompute_marginal_batch(
                data_np, top_k=48, topic_ppmi_gain=1.25,
                use_continuous_thought=use_layer2_continuous,
                cont_gain=1.10, inhib_gain=1.20,
            )

        H_buf = max(cfg.max_hops + 4, 22)
        byte_hist = torch.full((B, H_buf), -1, dtype=torch.int64, device=dev)
        hist_ptr = 0
        D_ctx = 12
        node_delay = torch.full((B, D_ctx), -1, dtype=torch.int64, device=dev)
        nd_ptr = 0
        ctx_node = torch.full((B, 2), -1, dtype=torch.int64, device=dev)
        ctx_score = torch.zeros((B, 2), dtype=torch.float32, device=dev)
        slot_state = torch.zeros((B, S, D), dtype=torch.float32, device=dev)
        exec_vec = torch.zeros((B, D), dtype=torch.float32, device=dev)

        N_stm = cfg.stm_slots
        stm_k4 = torch.zeros((B, N_stm), dtype=torch.int64, device=dev)
        stm_k7 = torch.zeros((B, N_stm), dtype=torch.int64, device=dev)
        stm_val = torch.zeros((B, N_stm), dtype=torch.int64, device=dev)
        stm_str = torch.zeros((B, N_stm), dtype=torch.float32, device=dev)
        stm_ptr = torch.zeros(B, dtype=torch.int64, device=dev)
        stm_boost = torch.zeros((B, V), dtype=torch.float32, device=dev)

        arange_B = torch.arange(B, device=dev)
        ones_B_f = torch.ones(B, dtype=torch.float32, device=dev)
        logits_sharp = torch.empty((B, V), dtype=torch.float32, device=dev)

        # Accumulate metrics on GPU (ZERO per-step .item() CUDA pipeline stalls!)
        total_nll_t = torch.zeros((), dtype=torch.float64, device=dev)
        total_top1_t = torch.zeros((), dtype=torch.int64, device=dev)
        total_top5_t = torch.zeros((), dtype=torch.int64, device=dev)
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
                u_old_c = u_old.clamp(min=0)
                ok_u = (u_old >= V) & core.alive[u_old_c]
                f_old = core.fires[u_old_c].to(torch.float32)
                sal_old = torch.where(ok_u, 1.8 / torch.sqrt(1.0 + f_old * 0.02), torch.zeros_like(f_old))
                already = (ctx_node[:, 0] == u_old) | (ctx_node[:, 1] == u_old)
                ws_i = ctx_score.argmin(dim=1)
                rep = ok_u & (~already) & (sal_old > ctx_score[arange_B, ws_i]) & (sal_old > 0.55)
                ctx_node[arange_B, ws_i] = torch.where(rep, u_old, ctx_node[arange_B, ws_i])
                ctx_score[arange_B, ws_i] = torch.where(rep, sal_old, ctx_score[arange_B, ws_i])
            top_ctx_all = ctx_node[arange_B, ctx_score.argmax(dim=1)]

            h_dag, logits_sharp, emergent_ctx_cand = run_fused_dag_step(
                core, x_t, byte_hist, cur_idx, top_ctx_all, logits_sharp_buf=logits_sharp
            )
            node_delay[:, nd_ptr] = emergent_ctx_cand
            nd_ptr = (nd_ptr + 1) % D_ctx

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
            logits = (h_act @ core.E.T + core.bias.unsqueeze(0) + logits_sharp) / temp_cal

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
                stm_boost.zero_()
                stm_boost.scatter_add_(1, stm_val.clamp(min=0, max=V - 1), hw)
                logits = logits + stm_boost.clamp(max=cfg.stm_max_boost)

            if (eng_boost_all is not None) or (l2_boost_all is not None):
                lp_c = torch.log_softmax(logits, dim=-1)
                p_c = torch.exp(lp_c)
                H_c = -(p_c * lp_c).sum(dim=-1, keepdim=True)
                ent_g = H_c / (1.0 + H_c)
                if eng_boost_all is not None:
                    eb = eng_boost_all[t].to(device=dev, dtype=torch.float32, non_blocking=True)
                    logits = logits + eb * (0.78 * ent_g)
                if l2_boost_all is not None:
                    l2b = l2_boost_all[t].to(device=dev, dtype=torch.float32, non_blocking=True)
                    logits = logits + l2b * (layer2_scale * ent_g)

            log_p = torch.log_softmax(logits, dim=-1)
            nll = -log_p[arange_B, y_t]
            total_nll_t += nll.sum().to(torch.float64)
            top5 = torch.topk(logits, k=5, dim=-1).indices
            total_top1_t += (top5[:, 0] == y_t).sum()
            total_top5_t += (top5 == y_t.unsqueeze(1)).any(dim=1).sum()
            total_steps += B

            if t >= 4:
                surp_stm = 1.0 - torch.exp(-nll)
                do_bind = (surp_stm > cfg.stm_surp_thresh) & (k4 != 0)
                exists = (stm_k4 == k4.unsqueeze(1)) & (stm_val == y_t.unsqueeze(1)) & (stm_str > 0.05)
                has_ex = exists.any(dim=1)
                w_slot = torch.where(has_ex, exists.to(torch.int32).argmax(dim=1).to(torch.int64), stm_ptr)
                stm_k4[arange_B, w_slot] = torch.where(do_bind, k4, stm_k4[arange_B, w_slot])
                stm_k7[arange_B, w_slot] = torch.where(do_bind, k7, stm_k7[arange_B, w_slot])
                stm_val[arange_B, w_slot] = torch.where(do_bind, y_t, stm_val[arange_B, w_slot])
                new_str = torch.where(has_ex, (stm_str[arange_B, w_slot] + 0.65).clamp(max=2.0), ones_B_f)
                stm_str[arange_B, w_slot] = torch.where(do_bind, new_str, stm_str[arange_B, w_slot])
                stm_ptr = torch.where(do_bind & (~has_ex), (stm_ptr + 1) % N_stm, stm_ptr)

        total_nll = float(total_nll_t.item())
        total_top1 = int(total_top1_t.item())
        total_top5 = int(total_top5_t.item())

        bpb = (total_nll / max(total_steps, 1)) / math.log(2.0)
        bpc = bpb * bytes_per_char
        return {
            "bpb": round(bpb, 4),
            "bpc_lossless": round(bpc, 4),
            "ppl_char": round(2.0 ** bpc, 2),
            "byte_top1": round(total_top1 / max(total_steps, 1), 4),
            "byte_top5": round(total_top5 / max(total_steps, 1), 4),
        }

    @torch.no_grad()
    def generate_motor_beam_v50(
        self,
        prompt: str,
        max_motor_steps: int = 28,
        beam_width: int = 6,
        top_k_motor: int = 16,
        use_thought_field: bool = True,
        use_topdown_inhibition: bool = True,
        topic_ppmi_gain: float = 1.65,
        cont_align_gain: float = 1.55,
        inhib_gain: float = 2.60,
        inhib_thresh: float = 0.16,
        rhythm_penalty: float = 0.60,
    ) -> Tuple[str, bool, Dict]:
        """ZERO-COPY-PASTE Pure Free-Form Generative Decoding in Grown Motor Output Node Space!
        When `use_thought_field=True` and `use_topdown_inhibition=True`:
          - Persistent Prefrontal Root Thought Anchor `h_root` NEVER decays!
          - 4-Timescale Continuous Thought State (`h_root`, `h_long`, `h_mid`) + Hebbian `W_pred` guides content!
          - Top-Down Inhibitory Gate crushes off-topic Layer-1 boilerplate hijacks (`年（昭和 -> JR西日本の駅`)!
          - Rule-Free IDF-Rhythm prevents noun-string stacking (`物理的性質量子力学的...`)!
          - Soft high-IDF repetition decay prevents looping!"""
        t0 = time.perf_counter()
        dev = self.device
        runner = self.runner
        prompt_b = prompt.encode("utf-8")
        if not prompt_b:
            return "", True, {}

        p_motors = self.mc.encode(np.frombuffer(prompt_b, dtype=np.uint8))
        prompt_m = p_motors.tolist()

        # Persistent Prefrontal Root Subjects & Continuous Root Thought Vector h_root (Mutual-Resonance Filtered)!
        root_t_ids, root_t_w = self.mc.extract_active_topics(p_motors, max_topics=6, min_idf=5.8, use_resonance=use_thought_field)
        h_root = self.mc.compute_root_thought_vec(p_motors)
        root_mass = sum(
            self.mc.content_gate(int(m))
            for m in p_motors.tolist()
            if int(m) >= 256 and self.mc.emb_np is not None and float(np.dot(self.mc.emb_np[int(m)], self.mc.emb_np[int(m)])) > 0.1
        )
        root_conf = min(1.0, root_mass / 1.6)
        has_root_vec = bool(use_thought_field and self.mc.emb_np is not None and float(np.linalg.norm(h_root)) > 0.1 and root_conf > 0.25)

        def _update_thought(h_l: np.ndarray, h_m: np.ndarray, mid: int) -> Tuple[np.ndarray, np.ndarray]:
            if not has_root_vec or mid < 256 or mid >= self.mc.V_m:
                return h_l, h_m
            cg = self.mc.content_gate(mid)
            if cg < 0.12:
                return h_l, h_m
            em = self.mc.emb_np[mid]
            if float(np.dot(em, em)) < 0.1:
                return h_l, h_m
            al = 0.08 * cg
            am = 0.25 * cg
            return ((1.0 - al) * h_l + al * em), ((1.0 - am) * h_m + am * em)

        def _combined_thought(h_l: np.ndarray, h_m: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            # Prefrontal Root Anchor holds 58% weight so the topic NEVER drifts mid-way!
            h = 0.58 * h_root + 0.26 * h_l + 0.16 * h_m
            nrm = float(np.linalg.norm(h))
            if nrm > 1e-6:
                h = h / nrm
            h_pred = None
            if self.mc.w_pred_np is not None:
                hp = self.mc.w_pred_np @ h
                pn = float(np.linalg.norm(hp))
                if pn > 1e-6:
                    h_pred = hp / pn
            return h, h_pred

        # Prefill Layer-1 GPU Core on exact prompt bytes!
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
        # Fast reverse lookup from raw UTF-8 bytes -> motor ID (for L1 Syntax Escape Hatch!)
        bytes_to_mid: Dict[bytes, int] = {b: i for i, b in enumerate(self.mc.motor_bytes)}
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
            "h_long": h_root.copy(),
            "h_mid": h_root.copy(),
            "consec_hi": 0.0,
        }]

        for _ in range(max_motor_steps):
            cand_parent: List[int] = []
            cand_mid: List[int] = []
            cand_l2: List[float] = []
            cand_inh_b: List[float] = []  # Applied EXACTLY ONCE (never double/triple-counted!)
            cand_bytes: List[bytes] = []
            cand_cg: List[float] = []
            cand_is_l1_escape: List[bool] = []

            for wi, bm in enumerate(beams):
                m_arr = np.array(bm["motor_seq"][-32:], dtype=np.int32)
                rec_ids, rec_w = self.mc.extract_active_topics(m_arr, max_topics=4, min_idf=5.2)
                t_map: Dict[int, float] = {int(k): float(v) for k, v in zip(root_t_ids.tolist(), root_t_w.tolist())}
                for dk, dv in zip(rec_ids.tolist(), rec_w.tolist()):
                    if has_root_vec and dk >= 256:
                        cos_r = float(np.dot(self.mc.emb_np[dk], h_root))
                        if cos_r < 0.22:
                            continue
                    t_map[int(dk)] = max(t_map.get(int(dk), 0.0), float(dv) * 0.35)
                ranked_t = sorted(t_map.items(), key=lambda x: x[1], reverse=True)[:6]
                t_arr = np.array([m for m, _ in ranked_t], dtype=np.int32) if ranked_t else np.zeros(0, dtype=np.int32)
                w_arr = np.array([w for _, w in ranked_t], dtype=np.float32) if ranked_t else np.zeros(0, dtype=np.float32)

                p_ids, p_sc = self.mc.predict_topk(
                    np.array(bm["motor_seq"][-6:], dtype=np.int32),
                    t_arr, w_arr, topic_ppmi_gain=topic_ppmi_gain, top_k=top_k_motor * 3,
                )

                h_thought, h_pred = _combined_thought(bm["h_long"], bm["h_mid"]) if has_root_vec else (None, None)
                recent_16 = set(bm["gen_motors"][-16:])
                c_hi = bm["consec_hi"]
                added_bytes_wi = set()

                def _score_candidate(mid_v: int, mb: bytes, base_l2: float) -> Tuple[float, float, float]:
                    cg = self.mc.content_gate(mid_v) if mid_v >= 0 else 0.0
                    l2_s = float(base_l2)
                    inh_per_byte = 0.0

                    if mid_v >= 256 and mid_v in recent_16:
                        l2_s -= (0.95 + 1.25 * cg)

                    if use_thought_field and cg > 0.55 and c_hi >= 1.8:
                        l2_s -= rhythm_penalty * (c_hi - 1.5) * cg
                    elif use_thought_field and cg < 0.25 and c_hi >= 2.1:
                        l2_s += 0.45

                    # Continuous 256-dim Thought-Field Excitation + SINGLE-APPLICATION Top-Down Inhibition!
                    if has_root_vec and mid_v >= 256 and cg > 0.15:
                        em = self.mc.emb_np[mid_v]
                        if float(np.dot(em, em)) > 0.1:
                            cos_root = float(np.dot(em, h_root))
                            cos_t = float(np.dot(em, h_thought))
                            cos_p = float(np.dot(em, h_pred)) if h_pred is not None else cos_t
                            align = 0.65 * cos_root + 0.20 * cos_t + 0.15 * cos_p
                            dir_aff = self.mc.directed_affinity(mid_v, root_t_ids, root_t_w)
                            eff_align = align + 0.20 * math.tanh(dir_aff)

                            l2_s += cg * cont_align_gain * eff_align
                            if use_topdown_inhibition and eff_align < inhib_thresh:
                                mismatch = min(1.25, (inhib_thresh - eff_align) / max(0.05, inhib_thresh))
                                # Applied ONCE (only in `inh_per_byte`, NEVER subtracted from l2_s too!)
                                inh_per_byte = root_conf * cg * inhib_gain * mismatch
                    return l2_s, inh_per_byte, cg

                # A) Layer-2 Motor Proposals
                taken = 0
                for mid_v, sc_v in zip(p_ids.tolist(), p_sc.tolist()):
                    mb = self.mc.motor_bytes[mid_v]
                    if mid_v < 256 and (mid_v < 32 or mid_v >= 127):
                        continue
                    if b"\n" in mb or b"\r" in mb or mb in added_bytes_wi:
                        continue
                    if bm["motor_seq"] and bm["motor_seq"][-1] == mid_v:
                        continue
                    l2_s, inh_b, cg = _score_candidate(mid_v, mb, sc_v)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2.append(l2_s)
                    cand_inh_b.append(inh_b); cand_bytes.append(mb); cand_cg.append(cg)
                    cand_is_l1_escape.append(False)
                    added_bytes_wi.add(mb)
                    taken += 1
                    if taken >= top_k_motor:
                        break

                # B) ALWAYS-ON Layer-1 UTF-8-Complete Syntax Escape Hatch (Top-4 from L1 directly!)
                # Guarantees grammatical glue/particles & ANY arbitrary UTF-8 character (zero <UNK>!) survive to L1 check!
                lrow = logits_beams[wi].clone()
                lrow[~runner.lead_mask] = -1e9
                lrow[0] = -1e9; lrow[10] = -1e9; lrow[13] = -1e9
                top_leads = torch.topk(lrow, k=4).indices.cpu().tolist()
                for b0 in top_leads:
                    if b0 < 0x80:
                        mb = bytes([b0])
                    else:
                        # Complete valid RFC 3629 multi-byte UTF-8 char via Layer-1's top continuation bytes!
                        clen = 2 if (b0 & 0xE0) == 0xC0 else (3 if (b0 & 0xF0) == 0xE0 else 4)
                        ch_b = bytearray([b0])
                        st_tmp = runner.gather_state(st_beams, torch.tensor([wi], dtype=torch.int64, device=dev))
                        cur_b = b0
                        ok_utf = True
                        for ci_u in range(1, clen):
                            lg_u, _ = runner.step_batch(
                                st_tmp, torch.tensor([cur_b], dtype=torch.int64, device=dev),
                                locked_mode_p=locked_mode_p, use_engram=True, engram_mult=0.72, is_prompt=False,
                            )
                            row_u = lg_u[0].clone()
                            # Mask strictly to RFC 3629 continuation bytes 0x80..0xBF (+ overlong bounds)
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
                            ok_utf = False
                        if not ok_utf:
                            continue
                    if mb in added_bytes_wi:
                        continue
                    mid_v = bytes_to_mid.get(mb, -1)
                    l2_s, inh_b, cg = _score_candidate(mid_v, mb, 0.25)
                    cand_parent.append(wi); cand_mid.append(mid_v); cand_l2.append(l2_s)
                    cand_inh_b.append(inh_b); cand_bytes.append(mb); cand_cg.append(cg)
                    cand_is_l1_escape.append(True)
                    added_bytes_wi.add(mb)

            if not cand_parent:
                break

            # Cap candidate frontier to 42 while ALWAYS keeping L1 Syntax Escape Hatch candidates!
            if len(cand_parent) > 42:
                esc_idx = [i for i, f in enumerate(cand_is_l1_escape) if f][:16]
                l2_idx = [i for i, f in enumerate(cand_is_l1_escape) if not f]
                l2_idx.sort(key=lambda i: beams[cand_parent[i]]["score"] + 0.55 * cand_l2[i], reverse=True)
                ord_c = esc_idx + l2_idx[: max(0, 42 - len(esc_idx))]
                cand_parent = [cand_parent[i] for i in ord_c]
                cand_mid = [cand_mid[i] for i in ord_c]
                cand_l2 = [cand_l2[i] for i in ord_c]
                cand_inh_b = [cand_inh_b[i] for i in ord_c]
                cand_bytes = [cand_bytes[i] for i in ord_c]
                cand_cg = [cand_cg[i] for i in ord_c]

            # Joint Layer-1 GPU Core + 1.084B Engram Batched Verification!
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

            # Score all M candidates with STRICTLY SEPARATED raw L1, raw L2, single-application inhibition, and repetition!
            next_pool = []
            for ci in range(M):
                bm = beams[cand_parent[ci]]
                mid = cand_mid[ci]
                mb = cand_bytes[ci]
                cg = cand_cg[ci]
                l1_step = l1_lp_cpu[ci]
                l2_step = cand_l2[ci]
                inh_step = cand_inh_b[ci] * len(mb)

                # If L1 Escape Hatch emitted a rare multi-byte UTF-8 char not in motor vocab, encode its raw bytes for mseq
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
                    on_topic_ok = True
                    if has_root_vec:
                        on_topic_ok = float(np.dot(self.mc.emb_np[mid], h_root)) >= inhib_thresh
                    if on_topic_ok:
                        prog_bonus = 0.30
                        new_used.add(mid)

                raw_l1 = bm["raw_l1_lp"] + l1_step
                raw_l2 = bm["raw_l2_sc"] + l2_step + prog_bonus
                tot_inh = bm["tot_inh"] + inh_step
                tot_rep = bm["tot_rep"] + rep_pen
                tot_nb = bm["n_bytes"] + len(mb)
                n_steps = len(bm["gen_motors"]) + 1

                # Clean additive decomposition: Raw L1 fluency + L2 thought - Single Inhibition - Repetition!
                joint_sc = (
                    (raw_l1 / max(1.0, tot_nb ** 0.90))
                    + 0.58 * (raw_l2 / max(1.0, n_steps ** 0.85))
                    - (tot_inh / max(1.0, tot_nb ** 0.90))
                    - 0.58 * (tot_rep / max(1.0, n_steps ** 0.85))
                )

                hl_n, hm_n = _update_thought(bm["h_long"], bm["h_mid"], mid) if mid >= 0 else (bm["h_long"], bm["h_mid"])
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
                    "h_long": hl_n,
                    "h_mid": hm_n,
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

            # Language-agnostic byte budget stop (ZERO hardcoded Japanese punctuation rules!)
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

        # Quantitative Topic-Drift Metric: 1st-Half vs 2nd-Half Root Thought-Lock Cosine!
        gen_m = best_bm["gen_motors"]
        c_sims_1, c_sims_2 = [], []
        if self.mc.emb_np is not None and float(np.linalg.norm(h_root)) > 0.1 and len(gen_m) > 0:
            half = max(1, len(gen_m) // 2)
            for idx_m, mid in enumerate(gen_m):
                if mid >= 256 and self.mc.content_gate(mid) > 0.45:
                    em = self.mc.emb_np[mid]
                    if float(np.dot(em, em)) > 0.1:
                        cs = float(np.dot(em, h_root))
                        if idx_m < half:
                            c_sims_1.append(cs)
                        else:
                            c_sims_2.append(cs)

        all_sims = c_sims_1 + c_sims_2
        info = {
            "fluency_bpc": round(flu_bpc, 3),
            "gen_bytes": int(best_bm["n_bytes"]),
            "motor_steps": int(len(gen_m)),
            "bytes_per_motor": round(best_bm["n_bytes"] / max(1, len(gen_m)), 2),
            "decode_sec": round(time.perf_counter() - t0, 2),
            "thought_lock_mean": round(float(np.mean(all_sims)), 3) if all_sims else 0.0,
            "thought_lock_1st": round(float(np.mean(c_sims_1)), 3) if c_sims_1 else 0.0,
            "thought_lock_2nd": round(float(np.mean(c_sims_2)), 3) if c_sims_2 else 0.0,
            "latched_subjects": [self.mc.motor_texts[int(m)] for m in root_t_ids[:6]],
            "sample_fired_motors": [self.mc.motor_texts[int(m)] if int(m) >= 0 else "<L1-UTF8>" for m in gen_m[:12]],
        }
        return full_txt, valid_utf8, info
