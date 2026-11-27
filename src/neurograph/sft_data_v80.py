"""対話データの整形・重複除去・学習と検証の分割（v8.0）。"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


from neurograph.settings import DATA_DIR, SFT_DIR, SFT_RAW_DIR

RAW_DIR = SFT_RAW_DIR
OUT_DIR = SFT_DIR
BASE_DATA_DIR = DATA_DIR

TURN_U_PREFIX = b"\n> "
TURN_A_PREFIX = b"\n\n"
TURN_EOT_BYTE = 0x04

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_NORM_RE = re.compile(r"\s+")

_META_BOILERPLATE = (
    "責任を負いたくない",
    "AI言語モデルとして",
    "AIアシスタントとして",
    "大規模言語モデルとして",
    "OpenAI",
    "ChatGPT",
    "私はAI",
)
_ORPHAN_TURN0_PREFIXES = ("彼女", "彼は", "彼の", "彼を", "彼に", "その作品", "この作品", "あらすじを", "どのバージョン")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_RE.sub("", s)
    return s.strip()


def jp_char_ratio(s: str) -> float:
    non_ws = [c for c in s if not c.isspace()]
    if not non_ws:
        return 0.0
    jp = 0
    for c in non_ws:
        cp = ord(c)
        if (
            0x3040 <= cp <= 0x309F
            or 0x30A0 <= cp <= 0x30FF
            or 0x4E00 <= cp <= 0x9FFF
            or 0x3000 <= cp <= 0x303F
            or 0xFF00 <= cp <= 0xFFEF
        ):
            jp += 1
    return jp / len(non_ws)


def norm_hash(s: str) -> str:
    norm = _WS_NORM_RE.sub(" ", s.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def _extract_dialogues() -> Tuple[List[Dict], Dict[str, Counter]]:
    dialogues: List[Dict] = []
    drop_stats: Dict[str, Counter] = {}

    def _valid_pair(u: str, a: str, turn_idx: int, st: Counter) -> Optional[Tuple[str, str]]:
        u = clean_text(u)
        a = clean_text(a)
        if len(u) < 4 or len(a) < 8:
            st["too_short"] += 1
            return None
        if len(u) > 360 or len(a) > 480:
            st["too_long"] += 1
            return None
        if jp_char_ratio(u) < 0.25 or jp_char_ratio(a) < 0.35:
            st["low_jp_ratio"] += 1
            return None
        if "```" in u or "```" in a:
            st["code_fence"] += 1
            return None
        if a.count("**") >= 4 or a.count("\n- ") >= 4 or a.count("\n1.") >= 1 and a.count("\n3.") >= 1:
            st["markdown_wall"] += 1
            return None
        if any(pat in a for pat in _META_BOILERPLATE):
            st["meta_disclaimer"] += 1
            return None
        if turn_idx == 0 and any(u.startswith(p) for p in _ORPHAN_TURN0_PREFIXES):
            st["orphan_turn0"] += 1
            return None
        if u == a:
            st["echo_identical"] += 1
            return None
        return u, a

    def _extract_turn_list(conv: List[Dict], u_key: str, u_val: str, a_val: str, txt_key: str, st: Counter) -> List[Tuple[str, str]]:
        turns: List[Tuple[str, str]] = []
        i = 0
        while i + 1 < len(conv) and len(turns) < 3:
            m0, m1 = conv[i], conv[i + 1]
            if m0.get(u_key) != u_val or m1.get(u_key) != a_val:
                break
            vp = _valid_pair(m0.get(txt_key, ""), m1.get(txt_key, ""), len(turns), st)
            if vp is None:
                break
            turns.append(vp)
            i += 2
        return turns

    src = "oasst1_ja"
    st = drop_stats[src] = Counter()
    p = RAW_DIR / "oasst1-21k-ja.jsonl"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for idx, ln in enumerate(f):
                st["raw_convs"] += 1
                obj = json.loads(ln)
                turns = _extract_turn_list(obj.get("conversations", []), "from", "human", "gpt", "value", st)
                if turns:
                    dialogues.append({"source": src, "conv_id": f"{src}_{idx}", "turns": turns})
                    st["kept_convs"] += 1

    src = "dolly_ja"
    st = drop_stats[src] = Counter()
    p = RAW_DIR / "databricks-dolly-15k-ja.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        for idx, obj in enumerate(data):
            st["raw_convs"] += 1
            ins = clean_text(obj.get("instruction", ""))
            inp = clean_text(obj.get("input", ""))
            out = clean_text(obj.get("output", ""))
            u = f"{ins}\n{inp}" if inp else ins
            vp = _valid_pair(u, out, 0, st)
            if vp is not None:
                dialogues.append({"source": src, "conv_id": f"{src}_{idx}", "turns": [vp]})
                st["kept_convs"] += 1

    src = "wiki_conv_ja"
    st = drop_stats[src] = Counter()
    p = RAW_DIR / "ja_conv_gpt3.5_100K_fix.jsonl"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for idx, ln in enumerate(f):
                st["raw_convs"] += 1
                obj = json.loads(ln)
                turns = _extract_turn_list(obj.get("conversations", []), "from", "human", "gpt", "value", st)
                if turns:
                    dialogues.append({"source": src, "conv_id": f"{src}_{idx}", "turns": turns})
                    st["kept_convs"] += 1

    src = "magpie_ja"
    st = drop_stats[src] = Counter()
    p = RAW_DIR / "magpie-sft-v1.0.jsonl"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for idx, ln in enumerate(f):
                st["raw_convs"] += 1
                obj = json.loads(ln)
                turns = _extract_turn_list(obj.get("conversations", []), "role", "user", "assistant", "content", st)
                if turns:
                    dialogues.append({"source": src, "conv_id": f"{src}_{idx}", "turns": turns})
                    st["kept_convs"] += 1

    return dialogues, drop_stats


def serialize_dialogue_with_mask(turns: List[Tuple[str, str]]) -> Tuple[bytes, np.ndarray]:
    buf = bytearray()
    is_assist_byte: List[bool] = []
    for u, a in turns:
        ub = u.encode("utf-8")
        ab = a.encode("utf-8")
        assert TURN_EOT_BYTE not in ub and TURN_EOT_BYTE not in ab, "Raw text contains EOT byte 0x04!"
        buf.extend(TURN_U_PREFIX)
        is_assist_byte.extend([False] * len(TURN_U_PREFIX))
        buf.extend(ub)
        is_assist_byte.extend([False] * len(ub))
        buf.extend(TURN_A_PREFIX)
        is_assist_byte.extend([False] * len(TURN_A_PREFIX))
        buf.extend(ab)
        is_assist_byte.extend([True] * len(ab))
        buf.append(TURN_EOT_BYTE)
        is_assist_byte.append(True)
    raw = bytes(buf)
    mask = np.array(is_assist_byte[1:], dtype=np.bool_)
    return raw, mask


def _check_base_val_overlap_exact(train_dialogues: List[Dict], win: int = 36) -> Dict[str, int]:
    if not BASE_DATA_DIR.exists():
        return {}
    val_sets: Dict[str, set] = {}
    for dname in ("easy_ja", "aozora", "wikipedia"):
        vp = BASE_DATA_DIR / f"{dname}.val"
        if not vp.exists():
            continue
        u32 = np.memmap(vp, dtype=np.uint32, mode="r")[:120_000]
        vstr = clean_text(np.array(u32, dtype="<u4").tobytes().decode("utf-32-le", errors="ignore"))
        val_sets[dname] = {hash(vstr[i : i + win]) for i in range(max(0, len(vstr) - win + 1))}

    hits = {k: 0 for k in val_sets}
    for d in train_dialogues:
        txt = "\n".join(u + "\n" + a for u, a in d["turns"])
        if len(txt) < win:
            continue
        for k, hset in val_sets.items():
            matched = False
            for i in range(len(txt) - win + 1):
                if hash(txt[i : i + win]) in hset:
                    matched = True
                    break
            if matched:
                hits[k] += 1
    return hits


def build_sft_dataset(
    max_train_per_source: Dict[str, int] | None = None,
    val_per_source: int = 250,
    seed: int = 80,
) -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if max_train_per_source is None:
        max_train_per_source = {
            "oasst1_ja": 15_000,
            "dolly_ja": 12_000,
            "wiki_conv_ja": 22_000,
            "magpie_ja": 22_000,
        }

    raw_diags, drop_stats = _extract_dialogues()

    seen_any = set()
    dedup_by_src: Dict[str, List[Dict]] = {}
    dup_dropped = 0
    for d in raw_diags:
        turn_hashes = [norm_hash(u) for u, _ in d["turns"]] + [norm_hash(a) for _, a in d["turns"]]
        if any(h in seen_any for h in turn_hashes):
            dup_dropped += 1
            continue
        for h in turn_hashes:
            seen_any.add(h)
        dedup_by_src.setdefault(d["source"], []).append(d)

    rng = random.Random(seed)
    train_set: List[Dict] = []
    val_set: List[Dict] = []
    avail_by_src: Dict[str, int] = {}

    for src, items in sorted(dedup_by_src.items()):
        avail_by_src[src] = len(items)
        rng.shuffle(items)
        n_val = min(val_per_source, max(50, len(items) // 10))
        v_sub = items[:n_val]
        t_cap = max_train_per_source.get(src, 20_000)
        t_sub = items[n_val : n_val + t_cap]
        val_set.extend(v_sub)
        train_set.extend(t_sub)

    tr_all = {norm_hash(s) for d in train_set for pair in d["turns"] for s in pair}
    va_all = {norm_hash(s) for d in val_set for pair in d["turns"] for s in pair}
    assert len(tr_all & va_all) == 0, f"Train/Val overlap detected: {len(tr_all & va_all)}"

    rng.shuffle(train_set)
    rng.shuffle(val_set)

    per_src_counts: Dict[str, Dict[str, float]] = {}
    sample_preview: Dict[str, Dict[str, List[Dict]]] = {}
    for src in sorted(avail_by_src.keys()):
        t_s = [d for d in train_set if d["source"] == src]
        v_s = [d for d in val_set if d["source"] == src]
        t_ab = sum(len(a.encode("utf-8")) for d in t_s for _, a in d["turns"])
        v_ab = sum(len(a.encode("utf-8")) for d in v_s for _, a in d["turns"])
        per_src_counts[src] = {
            "dedup_avail": avail_by_src[src],
            "train_convs": len(t_s),
            "val_convs": len(v_s),
            "train_turns": sum(len(d["turns"]) for d in t_s),
            "val_turns": sum(len(d["turns"]) for d in v_s),
            "train_asst_mb": round(t_ab / 1e6, 2),
            "val_asst_mb": round(v_ab / 1e6, 2),
        }
        sample_preview[src] = {
            "train_samples": t_s[:10],
            "val_samples": v_s[:5],
        }

    tr_u_lens, tr_a_lens, tr_tot_bytes, tr_asst_bytes = [], [], 0, 0
    for d in train_set:
        raw, mask = serialize_dialogue_with_mask(d["turns"])
        tr_tot_bytes += len(raw)
        tr_asst_bytes += int(mask.sum())
        for u, a in d["turns"]:
            tr_u_lens.append(len(u))
            tr_a_lens.append(len(a))

    va_tot_bytes, va_asst_bytes = 0, 0
    for d in val_set:
        raw, mask = serialize_dialogue_with_mask(d["turns"])
        va_tot_bytes += len(raw)
        va_asst_bytes += int(mask.sum())

    base_val_hits = _check_base_val_overlap_exact(train_set, win=36)

    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for d in train_set:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for d in val_set:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    report = {
        "total_train_convs": len(train_set),
        "total_train_turns": sum(len(d["turns"]) for d in train_set),
        "total_val_convs": len(val_set),
        "total_val_turns": sum(len(d["turns"]) for d in val_set),
        "dup_dropped": dup_dropped,
        "train_val_any_overlap": len(tr_all & va_all),
        "train_total_mb": round(tr_tot_bytes / 1e6, 2),
        "train_assistant_mb": round(tr_asst_bytes / 1e6, 2),
        "val_total_mb": round(va_tot_bytes / 1e6, 2),
        "val_assistant_mb": round(va_asst_bytes / 1e6, 2),
        "user_char_len_p50_p95": [int(np.percentile(tr_u_lens, 50)), int(np.percentile(tr_u_lens, 95))],
        "assistant_char_len_p50_p95": [int(np.percentile(tr_a_lens, 50)), int(np.percentile(tr_a_lens, 95))],
        "base_val_exact_36char_overlap_convs": base_val_hits,
        "per_source": per_src_counts,
        "drop_stats": {k: dict(v) for k, v in drop_stats.items()},
    }
    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sample_preview.json").write_text(json.dumps(sample_preview, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_sft_jsonl(path: Path) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))
    return out


if __name__ == "__main__":
    rep = build_sft_dataset()
    print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
