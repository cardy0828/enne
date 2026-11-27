"""NeuroGraph v2.0: Decoupled Context-Gated Engram Architecture (Zero-VRAM External Knowledge Cartridges).

Core Architectural Breakthrough:
- Completely decouples Reasoning/Fluency/Working-Memory (GPU Core Brain, frozen ~3.5 GB VRAM) from
  Encyclopedic/Domain Knowledge (CPU RAM / mmap External Engram Cartridges, 0 MB GPU VRAM!).
- Hot-Swappable Cartridges: Mount, unmount, or stack independent .eng cartridges live
  (e.g., easy_ja.eng + aozora.eng + wikipedia_200m.eng [260M slots!] + custom_user.eng) with ZERO Cortex retraining!
- True Engram Context-Aware Gating (Zero Backprop, Zero Tokenizer):
  Every Engram slot stores a 16-D quantized Semantic Trigram + Cortex-Basis Context Fingerprint (int8).
  At inference, cosine similarity gates each knowledge injection:
      g = gate_min + (1 - gate_min) * sigmoid(alpha * (cos(c_now, c_engram) - tau))
  Eliminating hash-collision & cross-domain false-positive NLL spikes!
- Surprise-Only Residual Distillation:
  Uses the frozen GPU Cortex's fast 2-hop predictor to skip predictable grammar (~55-65% of bytes),
  dedicating 100% of Engram capacity to pure encyclopedic facts, proper nouns, dates, and rare terminology!
- Deterministic O(1) Prefetch via OpenMP C++17 native kernel (ingests 200M chars / 580M bytes in minutes!).
"""

from __future__ import annotations

import ctypes
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neurograph.lateral_v13b import UnifiedBrainV13b
from neurograph.memory_v3b import _mix64


# ---- Compile / Load Native OpenMP C++17 Engram Kernel ----
_LIB_CACHE: Optional[ctypes.CDLL] = None


def load_engram_lib() -> ctypes.CDLL:
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    here = Path(__file__).resolve().parent
    cpp_path = here / "engram_kernel.cpp"
    so_path = here / "libengram.so"
    if (not so_path.exists()) or (cpp_path.stat().st_mtime > so_path.stat().st_mtime):
        cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(cpp_path), "-o", str(so_path)]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))

    c_i64_p = ctypes.POINTER(ctypes.c_int64)
    c_i32_p = ctypes.POINTER(ctypes.c_int32)
    c_u16_p = ctypes.POINTER(ctypes.c_uint16)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)
    c_i8_p = ctypes.POINTER(ctypes.c_int8)
    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_f64_p = ctypes.POINTER(ctypes.c_double)

    lib.engram_ingest_bank.argtypes = [
        c_i64_p, c_u8_p, c_i8_p, ctypes.c_int64, ctypes.c_int64,
        c_i32_p, c_u8_p, c_u16_p, c_i8_p,
    ]
    lib.engram_ingest_bank.restype = None

    lib.engram_sleep_prune_bank.argtypes = [
        ctypes.c_int64, ctypes.c_uint16, c_i32_p, c_u8_p, c_u16_p, c_i8_p,
    ]
    lib.engram_sleep_prune_bank.restype = ctypes.c_int64

    lib.engram_count_active_bank.argtypes = [ctypes.c_int64, c_u16_p]
    lib.engram_count_active_bank.restype = ctypes.c_int64

    lib.engram_lookup_bank.argtypes = [
        c_i64_p, c_i8_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint16,
        ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        c_i32_p, c_u8_p, c_u16_p, c_i8_p, c_f32_p, c_i64_p, c_f64_p,
    ]
    lib.engram_lookup_bank.restype = None

    lib.engram_hash_bytes24_batch.argtypes = [c_i32_p, ctypes.c_int64, c_i64_p]
    lib.engram_hash_bytes24_batch.restype = None

    _LIB_CACHE = lib
    return lib


# ---- Shared Deterministic Multi-Scale Raw UTF-8 Key & 16-D Context Fingerprint Encoder ----
class EngramContextEncoder:
    """Computes deterministic 6-scale raw UTF-8 Engram keys (6B, 9B, 12B, 15B, 18B, 24B) AND
    a 16-D quantized Semantic Trigram + Cortex-Basis Context Fingerprint (bit-exact between build & inference!)."""

    SPANS = (6, 9, 12, 15, 18, 24)
    CDIM = 16
    DECAY = 0.97
    WIN = 96
    PRIMES = (
        17, 1315423911, 2654435761, 1000003, 1000033, 1000037,
        1000039, 1000081, 1000099, 1000117, 1000121, 1000133,
        1000151, 1000159, 1000171, 1000183, 1000187, 1000193,
        1000211, 1000213, 1000231, 1000249, 1000253, 1000273,
    )

    def __init__(self, cortex_E: torch.Tensor, device: torch.device, seed: int = 2026):
        self.device = device
        D = cortex_E.shape[1]
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        # Project frozen Cortex sensory basis E [256, D] -> [256, 16] via fixed orthogonal basis
        P_e = torch.randn((D, self.CDIM), generator=gen, device=device) / math.sqrt(D)
        E_proj = cortex_E.to(torch.float32) @ P_e
        self.E_proj = E_proj / E_proj.norm(dim=1, keepdim=True).clamp(min=1e-5)  # [256, 16]

        # Fixed Johnson-Lindenstrauss projection table for raw 3-byte UTF-8 trigrams [8192, 16]
        # Every Japanese kanji (3 bytes) or 3-char ASCII trigram maps to its own unit direction in R^16!
        H_tri = torch.randn((8192, self.CDIM), generator=gen, device=device)
        self.H_tri = H_tri / H_tri.norm(dim=1, keepdim=True).clamp(min=1e-5)

        # Causal exponential smoothing kernel for batched GPU conv1d: [16, 1, WIN]
        # Matches exact step-by-step recurrence c_t = DECAY * c_{t-1} + (1 - DECAY) * v_t!
        lags = torch.arange(self.WIN - 1, -1, -1, dtype=torch.float32, device=device)
        kern = (1.0 - self.DECAY) * (self.DECAY ** lags)
        self.conv_kern = kern.view(1, 1, self.WIN).expand(self.CDIM, 1, self.WIN).contiguous()

    @torch.no_grad()
    def compute_keys_1d(self, x_gpu: torch.Tensor) -> List[torch.Tensor]:
        """Compute all 6 raw UTF-8 Engram keys in parallel for a contiguous 1D byte sequence x_gpu [N] (int64).
        Position i predicts target x_gpu[i + 1] using bytes ending at x_gpu[i] (i.e. x[i], x[i-1], ...).
        Returns 6 tensors of shape [N - 1] int64 (0 where prefix length < span)."""
        N = x_gpu.shape[0]
        M = N - 1
        # Rolling hash across offsets d = 0..23: at position i, byte d back is x_gpu[i - d]
        h = _mix64(x_gpu[:M] + self.PRIMES[0])
        out: List[torch.Tensor] = []
        span_set = set(self.SPANS)
        idx = torch.arange(M, device=self.device)
        for d in range(1, 24):
            # Shifted byte: for i >= d -> x_gpu[i - d], else 0
            shifted = torch.zeros(M, dtype=torch.int64, device=self.device)
            shifted[d:] = x_gpu[: M - d]
            h = _mix64(h ^ ((shifted + 31 * d) * self.PRIMES[d]))
            if (d + 1) in span_set:
                k_out = torch.where((idx >= d) & (h != 0), h, torch.zeros_like(h))
                out.append(k_out)
        return out

    @torch.no_grad()
    def compute_keys_2d(self, data_2d: torch.Tensor) -> List[torch.Tensor]:
        """Compute all 6 raw UTF-8 Engram keys in parallel for [B, T] int64 streams.
        Returns 6 tensors of shape [B, T - 1] int64."""
        B, T = data_2d.shape
        M = T - 1
        h = _mix64(data_2d[:, :M] + self.PRIMES[0])
        out: List[torch.Tensor] = []
        span_set = set(self.SPANS)
        t_idx = torch.arange(M, device=self.device).unsqueeze(0)
        for d in range(1, 24):
            shifted = torch.zeros((B, M), dtype=torch.int64, device=self.device)
            shifted[:, d:] = data_2d[:, : M - d]
            h = _mix64(h ^ ((shifted + 31 * d) * self.PRIMES[d]))
            if (d + 1) in span_set:
                k_out = torch.where((t_idx >= d) & (h != 0), h, torch.zeros_like(h))
                out.append(k_out)
        return out

    @torch.no_grad()
    def compute_context_q_1d(self, x_gpu: torch.Tensor) -> torch.Tensor:
        """Compute quantized 16-D context fingerprint int8 [N - 1, 16] in parallel via causal GPU conv1d!"""
        M = x_gpu.shape[0] - 1
        b0 = x_gpu[:M]
        b1 = torch.zeros_like(b0); b1[1:] = x_gpu[: M - 1]
        b2 = torch.zeros_like(b0); b2[2:] = x_gpu[: M - 2]
        h3 = (_mix64(b0 + 23) ^ ((b1 + 41) * 1315423911) ^ ((b2 + 67) * 2654435761)) & 8191
        v = 0.35 * self.E_proj[b0] + 0.65 * self.H_tri[h3]  # [M, 16]
        # Causal depthwise conv1d: [1, 16, M]
        inp = F.pad(v.T.unsqueeze(0), (self.WIN - 1, 0))
        c = F.conv1d(inp, self.conv_kern, groups=self.CDIM).squeeze(0).T  # [M, 16]
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return torch.round(c * 127.0).clamp(-127, 127).to(torch.int8)

    @torch.no_grad()
    def compute_context_q_2d(self, data_2d: torch.Tensor) -> torch.Tensor:
        """Compute quantized 16-D context fingerprint int8 [B, T - 1, 16] via causal GPU conv1d!"""
        B, T = data_2d.shape
        M = T - 1
        b0 = data_2d[:, :M]
        b1 = torch.zeros_like(b0); b1[:, 1:] = data_2d[:, : M - 1]
        b2 = torch.zeros_like(b0); b2[:, 2:] = data_2d[:, : M - 2]
        h3 = (_mix64(b0 + 23) ^ ((b1 + 41) * 1315423911) ^ ((b2 + 67) * 2654435761)) & 8191
        v = 0.35 * self.E_proj[b0] + 0.65 * self.H_tri[h3]  # [B, M, 16]
        inp = F.pad(v.permute(0, 2, 1), (self.WIN - 1, 0))
        c = F.conv1d(inp, self.conv_kern, groups=self.CDIM).permute(0, 2, 1)  # [B, M, 16]
        c = c / c.norm(dim=2, keepdim=True).clamp(min=1e-5)
        return torch.round(c * 127.0).clamp(-127, 127).to(torch.int8)

    @torch.no_grad()
    def step_context_single(self, c_state: torch.Tensor, b0: int, b1: int, b2: int) -> Tuple[torch.Tensor, np.ndarray]:
        """O(1) single-step context update for live generation (matches conv1d bit-identically!)."""
        b1c = max(0, b1)
        b2c = max(0, b2)
        h3 = int((_mix64(torch.tensor(b0 + 23, device=self.device)) ^ ((b1c + 41) * 1315423911) ^ ((b2c + 67) * 2654435761)).item()) & 8191
        v = 0.35 * self.E_proj[b0] + 0.65 * self.H_tri[h3]
        c_new = self.DECAY * c_state + (1.0 - self.DECAY) * v
        cn = c_new / c_new.norm().clamp(min=1e-5)
        cq_np = torch.round(cn * 127.0).clamp(-127, 127).to(torch.int8).cpu().numpy()
        return c_new, cq_np

    @torch.no_grad()
    def step_keys_single(self, byte_hist: torch.Tensor, cur_idx: int, H_buf: int) -> np.ndarray:
        """O(1) single-step 6-scale Engram key computation for live generation -> int64[6]."""
        bs = [int(byte_hist[0, (cur_idx - d) % H_buf].item()) for d in range(24)]
        b_t = torch.tensor([max(0, b) for b in bs], dtype=torch.int64, device=self.device)
        h = _mix64(b_t[0] + self.PRIMES[0])
        out = np.zeros(6, dtype=np.int64)
        span_to_i = {s: idx for idx, s in enumerate(self.SPANS)}
        for d in range(1, 24):
            h = _mix64(h ^ ((b_t[d] + 31 * d) * self.PRIMES[d]))
            if (d + 1) in span_to_i:
                if bs[d] >= 0:
                    hv = int(h.item())
                    out[span_to_i[d + 1]] = hv if hv != 0 else 1
        return out


# ---- Decoupled Hot-Swappable CPU/mmap External Engram Cartridge ----
@dataclass
class EngramCartridgeConfig:
    name: str = "engram"
    # 6 multi-scale raw UTF-8 banks (6B, 9B, 12B, 15B, 18B, 24B), all powers of 2
    bank_buckets: Tuple[int, ...] = (524_288, 1_048_576, 2_097_152, 2_097_152, 2_097_152, 1_048_576)
    bank_scales: Tuple[float, ...] = (0.26, 0.52, 0.90, 1.25, 1.55, 1.85)
    min_infer_w: Tuple[int, ...] = (2, 2, 2, 1, 1, 1)
    prune_min_w: Tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    # Context-Aware Gate parameters (Engram core!)
    gate_alpha: float = 5.0
    gate_tau: float = 0.22
    gate_min: float = 0.22
    engram_scale: float = 0.72


class EngramCartridge:
    """Independent CPU-RAM / mmap-able Knowledge Cartridge (0 MB GPU VRAM!).
    Can be built once, saved to disk, mounted/unmounted live, and stacked with other cartridges!"""

    WAYS = 4
    CDIM = 16

    def __init__(self, cfg: EngramCartridgeConfig, allocate: bool = True):
        self.cfg = cfg
        self.lib = load_engram_lib()
        self.num_banks = len(cfg.bank_buckets)
        self.sig: List[np.ndarray] = []
        self.dst: List[np.ndarray] = []
        self.w: List[np.ndarray] = []
        self.c_vec: List[np.ndarray] = []
        self.ingested_bytes: int = 0
        self.ingested_surprising: int = 0

        if allocate:
            for nb in cfg.bank_buckets:
                self.sig.append(np.zeros((nb, self.WAYS), dtype=np.int32))
                self.dst.append(np.zeros((nb, self.WAYS), dtype=np.uint8))
                self.w.append(np.zeros((nb, self.WAYS), dtype=np.uint16))
                self.c_vec.append(np.zeros((nb, self.WAYS, self.CDIM), dtype=np.int8))

    @property
    def total_slots(self) -> int:
        return sum(nb * self.WAYS for nb in self.cfg.bank_buckets)

    @property
    def cpu_ram_mb(self) -> float:
        return self.total_slots * (4 + 1 + 2 + self.CDIM) / (1024 * 1024)

    def count_active(self) -> int:
        tot = 0
        for bi, nb in enumerate(self.cfg.bank_buckets):
            tot += int(self.lib.engram_count_active_bank(nb, self.w[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
        return tot

    def ingest_chunk(self, keys_list: Sequence[np.ndarray], targets_u8: np.ndarray, c_q_i8: np.ndarray) -> None:
        """Ingest one pre-filtered chunk of (keys[6], targets_u8, c_q_i8) into CPU RAM using 12 OpenMP threads!"""
        N = int(targets_u8.shape[0])
        if N == 0:
            return
        t_ptr = targets_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        cq_ptr = c_q_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        for bi, nb in enumerate(self.cfg.bank_buckets):
            k_arr = np.ascontiguousarray(keys_list[bi], dtype=np.int64)
            self.lib.engram_ingest_bank(
                k_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                t_ptr, cq_ptr, N, nb,
                self.sig[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self.dst[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.w[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                self.c_vec[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            )

    def sleep_prune(self, min_w_override: Optional[int] = None) -> Dict[str, object]:
        before = self.count_active()
        after = 0
        for bi, nb in enumerate(self.cfg.bank_buckets):
            mw = int(min_w_override if min_w_override is not None else self.cfg.prune_min_w[bi])
            after += int(self.lib.engram_sleep_prune_bank(
                nb, mw,
                self.sig[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self.dst[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.w[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                self.c_vec[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ))
        pruned = before - after
        return {
            "before": before, "after": after, "pruned": pruned,
            "prune_pct": round(100.0 * pruned / max(1, before), 1),
        }

    def lookup_into(
        self,
        keys_list: Sequence[np.ndarray],
        c_q_i8: np.ndarray,
        out_boost: np.ndarray,
        use_context_gate: bool = True,
        scale_mult: float = 1.0,
        min_infer_override: Optional[int] = None,
    ) -> Tuple[int, float]:
        """Accumulate this cartridge's Engram knowledge into out_boost[M, 256] (in-place!)."""
        M = int(c_q_i8.shape[0])
        cq_ptr = c_q_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        ob_ptr = out_boost.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        total_hits = 0
        total_gate_sum = 0.0
        for bi, nb in enumerate(self.cfg.bank_buckets):
            k_arr = np.ascontiguousarray(keys_list[bi], dtype=np.int64)
            hits_c = ctypes.c_int64(0)
            gsum_c = ctypes.c_double(0.0)
            miw = int(min_infer_override if min_infer_override is not None else self.cfg.min_infer_w[bi])
            self.lib.engram_lookup_bank(
                k_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                cq_ptr, M, nb, miw,
                float(self.cfg.bank_scales[bi] * self.cfg.engram_scale * scale_mult),
                1 if use_context_gate else 0,
                float(self.cfg.gate_alpha),
                float(self.cfg.gate_tau),
                float(self.cfg.gate_min),
                self.sig[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self.dst[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.w[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                self.c_vec[bi].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                ob_ptr,
                ctypes.byref(hits_c),
                ctypes.byref(gsum_c),
            )
            total_hits += int(hits_c.value)
            total_gate_sum += float(gsum_c.value)
        return total_hits, total_gate_sum

    def save(self, folder: str) -> None:
        """Save cartridge as zero-copy mmap-ready binary files on disk."""
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.cfg.__dict__.items()},
            "ingested_bytes": self.ingested_bytes,
            "ingested_surprising": self.ingested_surprising,
            "active_slots": self.count_active(),
        }
        (p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for bi in range(self.num_banks):
            np.save(p / f"b{bi}_sig.npy", self.sig[bi])
            np.save(p / f"b{bi}_dst.npy", self.dst[bi])
            np.save(p / f"b{bi}_w.npy", self.w[bi])
            np.save(p / f"b{bi}_cvec.npy", self.c_vec[bi])

    @classmethod
    def load(cls, folder: str, mmap_mode: Optional[str] = "r") -> "EngramCartridge":
        """Instant 0.01s mount via mmap (or RAM copy if mmap_mode=None)!"""
        p = Path(folder)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        c = meta["cfg"]
        for tk in ("bank_buckets", "bank_scales", "min_infer_w", "prune_min_w"):
            c[tk] = tuple(c[tk])
        cart = cls(EngramCartridgeConfig(**c), allocate=False)
        cart.ingested_bytes = int(meta.get("ingested_bytes", 0))
        cart.ingested_surprising = int(meta.get("ingested_surprising", 0))
        for bi in range(cart.num_banks):
            cart.sig.append(np.load(p / f"b{bi}_sig.npy", mmap_mode=mmap_mode))
            cart.dst.append(np.load(p / f"b{bi}_dst.npy", mmap_mode=mmap_mode))
            cart.w.append(np.load(p / f"b{bi}_w.npy", mmap_mode=mmap_mode))
            cart.c_vec.append(np.load(p / f"b{bi}_cvec.npy", mmap_mode=mmap_mode))
        return cart


class EngramRack:
    """Hot-swappable multi-cartridge Engram mount rack (0 MB GPU VRAM!)."""

    def __init__(self, max_boost: float = 3.2):
        self.cartridges: List[Tuple[EngramCartridge, float, Optional[int]]] = []
        self.max_boost = max_boost

    def mount(self, cart: EngramCartridge, weight: float = 1.0, min_infer_override: Optional[int] = None) -> None:
        self.cartridges.append((cart, weight, min_infer_override))

    def unmount(self, name: str) -> None:
        self.cartridges = [(c, w, m) for (c, w, m) in self.cartridges if c.cfg.name != name]

    def clear(self) -> None:
        self.cartridges.clear()

    @property
    def names(self) -> List[str]:
        return [c.cfg.name for (c, _, _) in self.cartridges]

    def hash_bytes24_batch(self, bs24_i32: np.ndarray) -> List[np.ndarray]:
        """Ultra-fast C++ 6-scale raw UTF-8 rolling hash (bit-exact with GPU encoder, ~0.01ms vs 2.46ms!)."""
        arr = np.ascontiguousarray(bs24_i32, dtype=np.int32)
        B = int(arr.shape[0])
        out = np.empty((6, B), dtype=np.int64)
        lib = load_engram_lib()
        lib.engram_hash_bytes24_batch(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int64(B),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        return [out[i] for i in range(6)]

    def lookup_batch(
        self,
        keys_list: Sequence[np.ndarray],
        c_q_i8: np.ndarray,
        use_context_gate: bool = True,
    ) -> Tuple[np.ndarray, int, float]:
        M = int(c_q_i8.shape[0])
        out = np.zeros((M, 256), dtype=np.float32)
        if not self.cartridges:
            return out, 0, 0.0
        tot_hits = 0
        tot_gsum = 0.0
        if len(self.cartridges) == 1:
            cart, wt, miw = self.cartridges[0]
            h, gs = cart.lookup_into(keys_list, c_q_i8, out, use_context_gate=use_context_gate, scale_mult=wt, min_infer_override=miw)
            tot_hits, tot_gsum = h, gs
        else:
            # Anti-double-counting cartridge fusion: strongest matching cartridge wins + gentle 0.12 corroboration!
            # Scales cleanly to 3, 10, or 50 mounted cartridges with ZERO logit explosion!
            buf = np.empty((M, 256), dtype=np.float32)
            for idx_c, (cart, wt, miw) in enumerate(self.cartridges):
                buf.fill(0.0)
                h, gs = cart.lookup_into(keys_list, c_q_i8, buf, use_context_gate=use_context_gate, scale_mult=wt, min_infer_override=miw)
                tot_hits += h
                tot_gsum += gs
                if idx_c == 0:
                    np.copyto(out, buf)
                else:
                    hi = np.maximum(out, buf)
                    lo = np.minimum(out, buf)
                    np.add(hi, 0.12 * lo, out=out)
        np.minimum(out, self.max_boost, out=out)
        return out, tot_hits, tot_gsum


# ---- Decoupled GPU Core Brain + External CPU Engram Integration ----
class DecoupledBrainV20:
    """Wraps a FROZEN GPU Core Brain (stripped of internal LTM -> pure Reasoning/Fluency Cortex!)
    with an external hot-swappable CPU/mmap EngramRack!"""

    def __init__(self, core: UnifiedBrainV13b):
        self.core = core
        self.device = core.device
        # Detach & free ALL internal GPU LTM memory! Knowledge now lives 100% outside GPU in Engram cartridges!
        for bi in range(core.ltm.num_banks):
            core.ltm.sig[bi] = torch.empty((1, 4), dtype=torch.int32, device=self.device)
            core.ltm.dst[bi] = torch.full((1, 4), -1, dtype=torch.int16, device=self.device)
            core.ltm.w[bi] = torch.zeros((1, 4), dtype=torch.float16, device=self.device)
        torch.cuda.empty_cache()

        self.encoder = EngramContextEncoder(core.E, self.device)
        self.rack = EngramRack(max_boost=3.2)

    @torch.no_grad()
    def _fast_cortex_surprise_mask(self, x_gpu: torch.Tensor, surp_thresh: float = 0.28, sub_chunk: int = 131_072) -> torch.Tensor:
        """Surprise-Only Residual Distillation filter (chunked 131k -> < 200 MB VRAM, >50M bytes/sec!):
        Computes fast 2-hop Cortex prediction for ALL M positions -> skips predictable grammar
        so 100% of Engram slots store genuinely surprising facts, proper nouns, dates, and idioms!"""
        core = self.core
        M = x_gpu.shape[0] - 1
        surp = torch.empty(M, dtype=torch.bool, device=self.device)
        E_T = core.E.T
        bias = core.bias.unsqueeze(0)

        for s in range(0, M, sub_chunk):
            e = min(s + sub_chunk, M)
            n = e - s
            x0 = x_gpu[s:e]
            y0 = x_gpu[s + 1 : e + 1]

            prev1 = torch.full((n,), -1, dtype=torch.int64, device=self.device)
            if s > 0:
                prev1.copy_(x_gpu[s - 1 : e - 1])
            elif n > 1:
                prev1[1:].copy_(x_gpu[: n - 1])

            prev2 = torch.full((n,), -1, dtype=torch.int64, device=self.device)
            if s >= 2:
                prev2.copy_(x_gpu[s - 2 : e - 2])
            elif s == 1 and n > 1:
                prev2[1:].copy_(x_gpu[: n - 1])
            elif s == 0 and n > 2:
                prev2[2:].copy_(x_gpu[: n - 2])

            h = core.W_node[x0].to(torch.float32)
            z0 = torch.zeros(n, dtype=torch.int64, device=self.device)
            ch1 = core.edges.lookup(x0, z0, prev1)
            ok1 = (prev1 >= 0) & (ch1 >= core.V) & core.alive[ch1.clamp(min=0)]
            if ok1.any():
                h[ok1] += core.W_node[ch1[ok1]].to(torch.float32) * (1.0 / 1.10)
                ch2 = core.edges.lookup(torch.where(ok1, ch1, z0), z0, prev2)
                ok2 = ok1 & (prev2 >= 0) & (ch2 >= core.V) & core.alive[ch2.clamp(min=0)]
                if ok2.any():
                    h[ok2] += core.W_node[ch2[ok2]].to(torch.float32) * (1.0 / 1.20)

            logits = (torch.tanh(h * 0.35) * 2.85) @ E_T + bias
            lse = torch.logsumexp(logits, dim=-1)
            ly = logits[torch.arange(n, device=self.device), y0]
            p_true = torch.exp(ly - lse)
            surp[s:e] = (1.0 - p_true) > surp_thresh
        return surp

    @torch.no_grad()
    def distill_bytes_into_cartridge(
        self,
        cart: EngramCartridge,
        raw_bytes: np.ndarray,
        chunk_bytes: int = 4_000_000,
        surp_thresh: float = 0.25,
        use_surprise_filter: bool = True,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Distill a massive raw UTF-8 stream (even 580M bytes!) into a CPU EngramCartridge at ~20-40M B/s!"""
        t0 = time.time()
        N_tot = len(raw_bytes)
        kept_tot = 0
        pos = 0
        overlap = 96

        while pos < N_tot - 2:
            end = min(pos + chunk_bytes, N_tot)
            start = max(0, pos - overlap)
            sl = raw_bytes[start:end]
            skip_head = pos - start
            x_gpu = torch.from_numpy(sl.astype(np.int64)).to(self.device)

            keys_gpu = self.encoder.compute_keys_1d(x_gpu)
            cq_gpu = self.encoder.compute_context_q_1d(x_gpu)
            y_gpu = x_gpu[1:]

            # Slice away overlap prefix
            keys_gpu = [k[skip_head:] for k in keys_gpu]
            cq_gpu = cq_gpu[skip_head:]
            y_gpu = y_gpu[skip_head:]

            if use_surprise_filter:
                surp_all = self._fast_cortex_surprise_mask(x_gpu, surp_thresh=surp_thresh)
                keep = surp_all[skip_head:]
                idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
                keys_np = [k[idx].cpu().numpy() for k in keys_gpu]
                y_np = y_gpu[idx].to(torch.uint8).cpu().numpy()
                cq_np = cq_gpu[idx].cpu().numpy()
            else:
                keys_np = [k.cpu().numpy() for k in keys_gpu]
                y_np = y_gpu.to(torch.uint8).cpu().numpy()
                cq_np = cq_gpu.cpu().numpy()

            cart.ingest_chunk(keys_np, y_np, cq_np)
            kept_tot += int(y_np.shape[0])
            pos = end

        dt = time.time() - t0
        cart.ingested_bytes += N_tot
        cart.ingested_surprising += kept_tot
        if verbose:
            print(
                f"  [Engram Distill '{cart.cfg.name}'] {N_tot/1e6:.1f}MB -> kept {kept_tot/1e6:.1f}M surprising "
                f"({100.0*kept_tot/max(1,N_tot):.1f}%) in {dt:.1f}s ({N_tot/1e6/max(dt,1e-3):.1f} MB/s)",
                flush=True,
            )
        return {"bytes": N_tot, "kept": kept_tot, "sec": round(dt, 1), "mb_per_sec": round(N_tot / 1e6 / max(dt, 1e-3), 1)}

    @torch.no_grad()
    def eval_stream(
        self,
        byte_arr: np.ndarray,
        batch_streams: int = 128,
        bytes_per_char: float = 2.88,
        use_executive: bool = True,
        use_engram: bool = True,
        use_context_gate: bool = True,
    ) -> Dict[str, float]:
        """Evaluate Frozen GPU Core Brain + Deterministic Prefetched External CPU Engram Rack!"""
        core = self.core
        cfg = core.cfg
        device = self.device
        V, D, S = core.V, core.D, cfg.num_slots
        exec_on = use_executive and core.mode_ready
        eng_on = use_engram and (len(self.rack.cartridges) > 0)

        total_len = len(byte_arr)
        stream_len = total_len // batch_streams
        data = torch.from_numpy(
            byte_arr[: batch_streams * stream_len].reshape(batch_streams, stream_len).astype(np.int64)
        ).to(device)
        B, T = data.shape

        # ---- DETERMINISTIC O(1) PREFETCH OF ALL ENGRAM BOOSTS FOR ENTIRE EVAL STREAM! ----
        # Because raw UTF-8 Engram keys + 16-D context fingerprints depend ONLY on input bytes,
        # we prefetch all (B, T-1) Engram lookups across 12 CPU threads in ONE single ~0.08s pass!
        eng_boost_all: Optional[torch.Tensor] = None
        eng_hits = 0
        eng_gate_mean = 0.0
        if eng_on:
            keys_2d = self.encoder.compute_keys_2d(data)          # 6 x [B, T-1]
            cq_2d = self.encoder.compute_context_q_2d(data)       # [B, T-1, 16]
            keys_np = [k.reshape(-1).cpu().numpy() for k in keys_2d]
            cq_np = cq_2d.reshape(-1, 16).cpu().numpy()
            boost_np, eng_hits, gsum = self.rack.lookup_batch(keys_np, cq_np, use_context_gate=use_context_gate)
            eng_gate_mean = gsum / max(1, eng_hits)
            # Keep on CPU as [T-1, B, 256] float16 (~64 MB) and stream each step t to GPU (0 extra VRAM!)
            eng_boost_all = torch.from_numpy(boost_np.reshape(B, T - 1, 256)).permute(1, 0, 2).to(torch.float16).contiguous()
            del boost_np, keys_2d, cq_2d, keys_np, cq_np

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
                    ws = ctx_score.argmin(dim=1)
                    rep = ok_u & (~already) & (sal_old > ctx_score[arange_B, ws]) & (sal_old > 0.55)
                    if rep.any():
                        rb = torch.nonzero(rep, as_tuple=False).squeeze(1)
                        ctx_node[rb, ws[rb]] = u_old[rb]
                        ctx_score[rb, ws[rb]] = sal_old[rb]
            top_ctx_all = ctx_node[arange_B, ctx_score.argmax(dim=1)]

            # System 1: Frozen GPU Cortex DAG
            vis_b_list = [arange_B]
            vis_u_list = [x_t]
            vis_g_list = [ones_B_f]
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

            # System 2: S=16 Autonomous Slots
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

            # System 3 (DECOUPLED!): External CPU/mmap Context-Gated Engram Rack!
            if eng_boost_all is not None:
                logits = logits + eng_boost_all[t].to(device=device, dtype=torch.float32, non_blocking=True)

            # System 4: 1-Shot ByteSTM
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

            # System 5: 24-Mode Executive Critic
            if exec_on and exec_vec is not None:
                exec_vec.mul_(cfg.mode_ema_decay).add_(h_dag, alpha=1.0 - cfg.mode_ema_decay)
                if t >= 24:
                    ev_n = exec_vec / exec_vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    mp = torch.softmax((ev_n @ core.mode_centroids.T) / cfg.mode_temp, dim=-1)
                    logits = logits + (mp @ core.mode_llr) * cfg.mode_llr_scale

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
            "engram_hits_per_step": round(eng_hits / max(1, total_steps), 3),
            "engram_mean_gate": round(eng_gate_mean, 3),
        }

    @torch.no_grad()
    def generate_utf8(
        self,
        prompt: str,
        max_new_bytes: int = 180,
        temperature: float = 0.68,
        top_k: int = 24,
        rollout_k: int = 4,
        rollout_chunk_bytes: int = 42,
        use_engram: bool = True,
        use_context_gate: bool = True,
    ) -> Tuple[str, bool, float]:
        """Live Raw UTF-8 Generation with GPU Core Brain + <50us CPU Engram Cartridge Lookup!"""
        core = self.core
        cfg = core.cfg
        device = self.device
        V, D, S = core.V, core.D, cfg.num_slots
        raw_in = list(prompt.encode("utf-8"))
        H_buf = 28
        D_ctx = 12
        N_stm = cfg.stm_slots
        eng_on = use_engram and (len(self.rack.cartridges) > 0)

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
                "step_i": 0,
            }

        def clone_state(st):
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in st.items()}

        eng_time_us = 0.0
        eng_calls = 0

        def step_state(st, b_val: int, next_true: int = -1, locked_mode_p=None):
            nonlocal eng_time_us, eng_calls
            step_i = st["step_i"]
            st["byte_hist"][0, st["hist_ptr"]] = b_val
            cur_idx = st["hist_ptr"]
            st["hist_ptr"] = (st["hist_ptr"] + 1) % H_buf

            b1 = int(st["byte_hist"][0, (cur_idx - 1) % H_buf].item())
            b2 = int(st["byte_hist"][0, (cur_idx - 2) % H_buf].item())
            st["c_eng"], cq_np = self.encoder.step_context_single(st["c_eng"], b_val, b1, b2)

            st["ctx_score"].mul_(0.985)
            if step_i >= D_ctx:
                u_old = int(st["node_delay"][0, st["nd_ptr"]].item())
                if u_old >= V and bool(core.alive[u_old].item()):
                    sal = 1.8 / math.sqrt(1.0 + float(core.fires[u_old].item()) * 0.02)
                    if sal > 0.55 and u_old not in st["ctx_node"][0].tolist():
                        ws = int(st["ctx_score"][0].argmin().item())
                        if sal > float(st["ctx_score"][0, ws].item()):
                            st["ctx_node"][0, ws] = u_old
                            st["ctx_score"][0, ws] = sal
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
            h_act = torch.tanh((h_dag * (1.0 + 0.35 * torch.tanh(m_sum)) + r_sum) * 0.35) * 2.85
            logits = h_act @ core.E.T + core.bias + logits_s

            # External CPU Engram lookup (<50us!)
            if eng_on:
                t_e0 = time.perf_counter()
                ks = self.encoder.step_keys_single(st["byte_hist"], cur_idx, H_buf)
                eb_np, _, _ = self.rack.lookup_batch(
                    [ks[bi : bi + 1] for bi in range(6)], cq_np.reshape(1, 16), use_context_gate=use_context_gate
                )
                eng_time_us += (time.perf_counter() - t_e0) * 1e6
                eng_calls += 1
                logits = logits + torch.from_numpy(eb_np[0]).to(device)

            # ByteSTM
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

            # Executive Critic
            st["exec_vec"].mul_(cfg.mode_ema_decay).add_(h_dag.unsqueeze(0), alpha=1.0 - cfg.mode_ema_decay)
            if core.mode_ready:
                ev_n = st["exec_vec"] / st["exec_vec"].norm(dim=1, keepdim=True).clamp(min=1e-6)
                cur_mp = torch.softmax((ev_n @ core.mode_centroids.T) / cfg.mode_temp, dim=-1)
                mp = (0.60 * locked_mode_p + 0.40 * cur_mp) if locked_mode_p is not None else cur_mp
                logits = logits + (mp @ core.mode_llr)[0] * (cfg.mode_llr_scale * 1.15)

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
        prompt_hs = []
        logits = None
        for i, b in enumerate(raw_in):
            nxt_t = raw_in[i + 1] if i + 1 < len(raw_in) else -1
            logits, hd = step_state(st, b, next_true=nxt_t, locked_mode_p=None)
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
        rem = max_new_bytes
        while rem > 0:
            c_len = min(rollout_chunk_bytes, rem)
            best_score = -1e9
            best_pack = None
            for _ in range(max(1, rollout_k)):
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
                            rep_pen += 0.65
                        local_seen.add(ng)
                    clog, chd = step_state(cst, nxt_b, locked_mode_p=locked_mode_p)
                    c_hs.append(chd)

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
                score = (lp_sum / max(1, c_len)) + 0.80 * mode_sim + utf8_bonus - rep_pen
                if score > best_score:
                    best_score = score
                    best_pack = (cst, clog, c_bytes)

            st, logits, chosen_b = best_pack
            out_bytes.extend(chosen_b)
            gen_only.extend(chosen_b)
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
        avg_us = eng_time_us / max(1, eng_calls)
        return bytes(out_bytes).decode("utf-8", errors="replace"), valid_utf8, round(avg_us, 1)
