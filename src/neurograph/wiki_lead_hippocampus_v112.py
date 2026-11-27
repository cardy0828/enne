"""NeuroGraph v11.2: Persistent Article-Lead-Prioritized Wikipedia Hippocampus + Single Merged Brain v11.1 Engine!

Zero Backprop | Zero Attention | Zero Hardcoded Language Rules | Ultra-Low RAM Persistent SQLite Index!

Fixes the exact factual-retrieval bottleneck isolated in v9–v10:
  - Flat 6-gram inverted indexes over raw text streams prefer incidental mentions (`東海道新幹線墨子事件`,
    `1943年クルスクの戦い`, `豚肉と吾輩は猫である`) over encyclopedic article definitions.
  - WikiLeadHippocampusV112 streams ALL 1.67 BILLION chars of `wikipedia.train` in 16M-char chunks
    (< 1.2 GB peak RAM!), detects Article Lead Paragraphs strictly structurally (paragraphs after `\n\n`
    whose line 0 is a full definitional sentence containing the learned terminator, NOT a short heading line),
    extracts each article's Subject Head (prefix before first learned bracket/topic role marker), and stores
    them in a persistent SQLite index (`wiki_lead_v112/wiki_leads.sqlite`) with O(1) exact/prefix + 3-gram lookup!
  - Uses our OWN persistent Hippocampus to verify encyclopedic entity heads cleanly (prevents greetings like
    `こんにちは！調子はどうですか？` from triggering Hippocampus while recovering `日本`, `首都`, `吾輩は猫である`!).
  - Runs on ONE SINGLE v11.1 Merged Brain (`merged_v111_broca_core.pt` + `merged_v111_adapter.pkl`)!
"""

from __future__ import annotations

import gc
import math
import os
import itertools
import random
import pickle
import sqlite3
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neurograph.sampling_v123 import choose, social_options
from neurograph.broca_cortex_v100 import BrocaSyntacticParserV100
from neurograph.engram_v20 import DecoupledBrainV20, EngramCartridge, EngramContextEncoder
from neurograph.instruct_cortex_v80 import InstructAdapterV80, _mem_snapshot
from neurograph.lateral_v13b import UnifiedBrainV13b
from neurograph.memory_v3b import _mix64
from neurograph.reason_decode_v30 import BatchedCoreRunner
from neurograph.reasoning_graph_v90 import PrefrontalReasoningGraphV90, trim_utf8_safe
from neurograph.role_cortex_v60 import RoleBoundThoughtCortexV60
from neurograph.sft_data_v80 import TURN_A_PREFIX, TURN_EOT_BYTE, TURN_U_PREFIX
from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50


from neurograph.memory_gate_v120 import ThalamicMemoryGateV120, ThalamicConsensusGateV124, subject


def _norm_key(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower()


def _next_sentence_end_bytes(path_b: bytes, terminators, start_byte: int = 0) -> int:
    """括弧の開閉深さ (Ps/Pi .. Pe/Pf) が 0 の位置でのみ最初の文末バイト位置を返す。"""
    text = path_b[start_byte:].decode("utf-8", errors="ignore")
    depth = 0
    fallback = None
    for idx, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat in ("Ps", "Pi"):
            depth += 1
        elif cat in ("Pe", "Pf"):
            depth = max(0, depth - 1)
        elif ch in terminators:
            end_b = start_byte + len(text[: idx + 1].encode("utf-8"))
            if depth == 0:
                return end_b
            if fallback is None:
                fallback = end_b
    return fallback if fallback is not None else len(path_b)


def _extract_subject_head(
    line0: str,
    head_delims: Sequence[str],
    term_str: str,
) -> Optional[Tuple[str, int]]:
    """冒頭から見出しと定義の強さを返す。括弧や休止直前の役割語を優先する。"""
    text = line0.strip()
    if len(text) < 28 or not term_str or term_str not in text:
        return None
    prefix = text.split(term_str, 1)[0][:38]
    if not prefix:
        return None
    if unicodedata.category(prefix[0]) in ("Ps", "Pi"):
        for end, ch in enumerate(prefix[1:], 1):
            if unicodedata.category(ch) in ("Pe", "Pf"):
                title = prefix[1:end].strip()
                if 2 <= len(title) <= 30:
                    tail = text[end + 1:].lstrip()
                    if tail and unicodedata.category(tail[0]) == "Ps":
                        close = next((i for i, c in enumerate(tail) if unicodedata.category(c) == "Pe"), None)
                        if close is not None:
                            tail = tail[close + 1:].lstrip()
                    explicit = any(tail.startswith(m) and tail[len(m):] and
                                   unicodedata.category(tail[len(m)]) == "Po" for m in head_delims)
                    return title, 5 if explicit else 2
                break

    weak = None
    for pos, ch in enumerate(prefix):
        cat = unicodedata.category(ch)
        head = prefix[:pos].strip()
        if cat == "Ps":
            if 1 <= len(head) <= 30:
                # 括弧の後に学習済み役割語と休止が続けば、定義冒頭として優先する。
                depth = 1
                for end in range(pos + 1, len(text)):
                    kind = unicodedata.category(text[end])
                    if kind == "Ps":
                        depth += 1
                    elif kind == "Pe":
                        depth -= 1
                    if depth == 0:
                        tail = text[end+1:].lstrip()
                        for marker in head_delims:
                            if tail.startswith(marker) and tail[len(marker):] and unicodedata.category(tail[len(marker)]) == "Po":
                                return head, 5
                        break
                return head, 4
            break
        if cat.startswith("P"):
            break
        if pos < 2 or len(head) > 30 or ch not in head_delims:
            continue
        tail = prefix[pos + 1:]
        if tail and unicodedata.category(tail[0]) == "Po":
            return head, 3
        if weak is None and unicodedata.name(ch, "").split()[:1] != unicodedata.name(prefix[pos-1], "").split()[:1]:
            weak = (head, 1)
    return weak


class WikiLeadHippocampusV112:
    """Persistent Disk-Backed Article-Lead Hippocampus (< 50 MB RAM at query time, < 2ms latency!)."""

    def __init__(self, db_path: str = "wiki_lead_v112/wiki_leads.sqlite"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self.custom_facts: List[Tuple[str, str, bytes]] = []  # (name, text, bytes)
        self.total_leads: int = 0
        if self.db_path.exists():
            self._open()

    def _open(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.execute("PRAGMA cache_size = -16000")  # 16 MB page cache
        row = self.conn.execute("SELECT COUNT(*) FROM leads").fetchone()
        self.total_leads = int(row[0]) if row else 0
        self.has_senses = bool(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='senses'").fetchone())
        self.has_aliases = bool(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='aliases'").fetchone())

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def mount_custom(self, name: str, text: str) -> None:
        self.unmount_custom(name)
        for ln in text.split("\n"):
            s = ln.strip()
            if len(s) >= 8:
                self.custom_facts.append((name, s, s.encode("utf-8")))

    def unmount_custom(self, name: str) -> None:
        self.custom_facts = [x for x in self.custom_facts if x[0] != name]

    @classmethod
    def build_from_utf32_corpus(
        cls,
        corpus_paths: Sequence[Tuple[str, str, float]],  # [(name, path, priority)]
        db_path: str,
        head_delims: Sequence[str],
        term_str: str,
        chunk_chars: int = 16_000_000,
        max_lead_bytes: int = 1024,
        max_senses: int = 1,
    ) -> Tuple["WikiLeadHippocampusV112", Dict[str, object]]:
        """Stream full multi-billion-char UTF-32-LE corpora in 16M-char chunks (< 1.2 GB Peak RAM!)
        and persist an exact + 3-gram indexed Article-Lead SQLite Hippocampus!"""
        t0 = time.perf_counter()
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, staging = tempfile.mkstemp(dir=p.parent, suffix=".sqlite")
        os.close(fd)

        conn = sqlite3.connect(staging)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA temp_store = FILE")
        conn.execute("CREATE TABLE leads (id INTEGER PRIMARY KEY, head_norm TEXT, head TEXT, src TEXT, prio REAL, lead TEXT, def_quality INTEGER)")
        conn.execute("CREATE TABLE head_grams (gram TEXT, lead_id INTEGER)")

        if max_senses > 1:
            conn.execute("CREATE TABLE senses (head_norm TEXT, src TEXT, lead TEXT, quality INTEGER)")
            conn.execute("CREATE INDEX idx_senses ON senses(head_norm)")

        def retain_sense(key, source, passage, quality):
            """同じ見出し（例 K=3）の定義を品質順に、同点では語彙の多様性順に残す。"""
            existing = conn.execute("SELECT rowid, lead, quality FROM senses WHERE head_norm=?", (key,)).fetchall()
            if any(text == passage for _, text, _ in existing):
                return
            if len(existing) >= max_senses:
                candidates = [(None, passage, quality), *existing]
                grams = [set(zip(text, text[1:])) for _, text, _ in candidates]
                def priority(i):
                    """他の候補との最大 Jaccard が高い同品質の候補から外す。"""
                    overlap = max(len(grams[i] & g) / max(1, len(grams[i] | g))
                                  for j, g in enumerate(grams) if i != j)
                    return candidates[i][2], -overlap
                drop = min(range(len(candidates)), key=priority)
                if drop == 0:
                    return
                conn.execute("DELETE FROM senses WHERE rowid=?", (candidates[drop][0],))
            conn.execute("INSERT INTO senses VALUES(?,?,?,?)", (key, source, passage, quality))

        seen_heads: Dict[str, Tuple[int, int]] = {}  # 正規化見出しから行番号と品質を引く
        lead_rows: List[Tuple[int, str, str, str, float, str, int]] = []
        gram_rows: List[Tuple[str, int]] = []
        next_id = 1
        total_chars_scanned = 0
        total_paras = 0

        def _flush():
            nonlocal lead_rows, gram_rows
            if lead_rows:
                conn.executemany("INSERT OR REPLACE INTO leads VALUES (?,?,?,?,?,?,?)", lead_rows)
                lead_rows = []
            if gram_rows:
                conn.executemany("INSERT INTO head_grams VALUES (?,?)", gram_rows)
                gram_rows = []

        for src_name, fpath, prio in corpus_paths:
            total_u32 = os.path.getsize(fpath) // 4
            with open(fpath, "rb") as f:
                carry = ""
                pos = 0
                while pos < total_u32:
                    n_read = min(chunk_chars, total_u32 - pos)
                    arr = np.fromfile(f, dtype=np.uint32, count=n_read)
                    pos += n_read
                    total_chars_scanned += len(arr)
                    txt = carry + arr.tobytes().decode("utf-32-le", errors="ignore")
                    del arr
                    parts = txt.split("\n\n")
                    carry = parts[-1] if pos < total_u32 else ""
                    use_parts = parts[:-1] if pos < total_u32 else parts

                    for para in use_parts:
                        total_paras += 1
                        if len(para) < 35:
                            continue
                        nl_idx = para.find("\n")
                        line0 = para if nl_idx < 0 else para[:nl_idx]
                        title = ""
                        if nl_idx >= 0 and 1 <= len(line0.strip()) <= 30 and term_str not in line0:
                            title = line0.strip()
                            para = para[nl_idx + 1:]
                            line0 = para.partition("\n")[0]
                        candidate = _extract_subject_head(line0, head_delims, term_str)
                        annotated = ""
                        if title and max_senses > 1 and unicodedata.category(title[-1]) == "Pe":
                            boundary = next((i for i,c in enumerate(title) if unicodedata.category(c) == "Ps"), 0)
                            base = title[:boundary].strip()
                            if base and line0.startswith(base) and term_str in line0:
                                annotated = base
                        if not candidate and not annotated:
                            continue
                        head, quality = candidate or ("", 0)
                        legacy_ok = bool(candidate) and (not title or
                            (quality >= 3 and "".join(head.split()) == "".join(title.split())))
                        if not legacy_ok and not annotated:
                            continue
                        if title and legacy_ok:
                            quality = max(4, quality)
                        h_norm = _norm_key(title or head)
                        previous = seen_heads.get(h_norm)
                        legacy_write = legacy_ok and h_norm and (not previous or previous[1] < quality)
                        if not legacy_write and max_senses == 1:
                            continue
                        # Keep opening 1-2 sentences cleanly within max_lead_bytes on UTF-8 boundary!
                        lead_raw = para.replace("\n", " ").strip()
                        first_end = lead_raw.find(term_str) + len(term_str)
                        first_size = len(lead_raw[:first_end].encode())
                        byte_limit = max(max_lead_bytes, min(2 * max_lead_bytes, first_size))
                        lead_b = trim_utf8_safe(lead_raw.encode("utf-8"), byte_limit)
                        lead_s = lead_b.decode("utf-8", errors="ignore").strip()
                        # Trim cleanly to last sentence terminator if present past 40 chars
                        if term_str:
                            last_t = lead_s.rfind(term_str)
                            if last_t >= 0:
                                lead_s = lead_s[: last_t + len(term_str)]
                        if len(lead_s) < 28:
                            continue

                        if max_senses > 1 and (annotated or quality >= 3):
                            retain_sense(_norm_key(annotated) if annotated else h_norm, src_name, lead_s,
                                         6 if annotated else quality)
                        if not legacy_write:
                            continue
                        lid = previous[0] if previous else next_id
                        seen_heads[h_norm] = (lid, quality)
                        lead_rows.append((lid, h_norm, head, src_name, prio, lead_s, quality))
                        if previous:
                            continue
                        next_id += 1

                        # Index compact 3-char grams of `h_norm` (max 10 grams/head -> fast fuzzy/multi-word lookup!)
                        if len(h_norm) >= 3:
                            grams = set()
                            step = 1 if len(h_norm) <= 12 else 2
                            for gi in range(0, len(h_norm) - 2, step):
                                grams.add(h_norm[gi : gi + 3])
                                if len(grams) >= 10:
                                    break
                            for g in grams:
                                gram_rows.append((g, lid))
                        else:
                            gram_rows.append((h_norm, lid))

                        if len(lead_rows) >= 25_000:
                            _flush()

        _flush()
        conn.execute("CREATE INDEX idx_leads_head_norm ON leads(head_norm)")
        conn.execute("CREATE INDEX idx_grams_gram ON head_grams(gram)")
        # 最初の文の短い述部を索引化し、見出しに現れない関係も逆引きする。
        pause = next((d for d in head_delims if len(d) == 1 and d != term_str and unicodedata.category(d) == "Po"), "")
        conn.execute("CREATE TABLE predicates (phrase TEXT, lead_id INTEGER)")
        if pause:
            rows = conn.execute("SELECT id, lead FROM leads WHERE def_quality >= 3")
            while batch := rows.fetchmany(10000):
                predicates = []
                for lid, lead in batch:
                    first, end, _ = lead.partition(term_str)
                    _, boundary, predicate = first.rpartition(pause)
                    if end and boundary and 3 <= len(predicate.strip()) <= 24:
                        predicates.append((_norm_key(predicate), lid))
                conn.executemany("INSERT INTO predicates VALUES (?, ?)", predicates)
        conn.execute("CREATE INDEX idx_predicate ON predicates(phrase)")
        conn.execute("CREATE TABLE aliases (alias TEXT, lead_id INTEGER)")
        rows = conn.execute("SELECT id,head,lead FROM leads WHERE def_quality >= 4")
        while batch := rows.fetchmany(10000):
            aliases = []
            for lid, head, lead in batch:
                tail = lead[len(head):] if lead.startswith(head) else ""
                if tail and unicodedata.category(tail[0]) == "Ps":
                    end = next((j for j,c in enumerate(tail) if unicodedata.category(c) == "Pe"), 0)
                    annotation = tail[1:end]
                    if any(unicodedata.category(c) in ("Ps", "Pd") for c in annotation):
                        continue
                    fields = itertools.groupby(annotation, key=lambda c: unicodedata.category(c) == "Po")
                    names = ["".join(chars).strip() for is_pause, chars in fields if not is_pause]
                    acronyms = [name for name in names if name.isalpha() and name.isupper()]
                    if len(acronyms) == 1 and 2 <= len(acronyms[0]) <= 12:
                        aliases.append((_norm_key(acronyms[0]), lid))
            conn.executemany("INSERT INTO aliases VALUES (?,?)", aliases)
        conn.execute("CREATE INDEX idx_alias ON aliases(alias)")
        conn.commit()
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("索引の整合性検査に失敗しました")
        conn.close()
        os.replace(staging, p)
        del seen_heads
        gc.collect()

        obj = cls(db_path=db_path)
        db_mb = round(p.stat().st_size / (1024 * 1024), 1)
        return obj, {
            "total_chars_scanned": total_chars_scanned,
            "total_paragraphs": total_paras,
            "unique_article_leads": obj.total_leads,
            "db_size_mb": db_mb,
            "sec": round(time.perf_counter() - t0, 2),
        }

    def has_exact_lead(self, entity: str, min_quality: int = 1, allow_ideograph: bool = False) -> bool:
        """O(1) < 0.05ms check whether `entity` is an exact encyclopedic article head (or custom fact match)!"""
        en = _norm_key(entity)
        if len(en) < 2:
            if not (allow_ideograph and len(en)==1 and unicodedata.name(en,"").startswith("CJK UNIFIED IDEOGRAPH")):
                return False
            return bool(self.conn and self.conn.execute("SELECT 1 FROM leads WHERE head_norm=? AND def_quality>=? LIMIT 1",
                                                        (en,max(3,min_quality))).fetchone())
        for _, txt, _ in self.custom_facts:
            if en in _norm_key(txt):
                return True
        if self.conn is None:
            return False
        if min_quality > 1:
            row = self.conn.execute("SELECT 1 FROM leads WHERE head_norm = ? AND def_quality >= ? LIMIT 1", (en, min_quality)).fetchone()
        else:
            row = self.conn.execute("SELECT 1 FROM leads WHERE head_norm = ? LIMIT 1", (en,)).fetchone()
        return row is not None

    def query_leads(
        self,
        entities: Sequence[str],
        full_prompt: str = "",
        top_k: int = 2,
        prefer_subject: bool = False,
        include_senses: bool = False,
        retain_primary: bool = False,
        exclude_entities: Sequence[str] = (),
        exclude_senses: Sequence[str] = (),
    ) -> List[Tuple[float, str, str, bytes]]:
        """Retrieve prioritized Article Leads matching `entities` (and hot-mounted custom facts!).
        Returns `[(score, source, head, lead_bytes)]` with Uncovered-Entity-First multi-slot coverage!"""
        cands: Dict[str, Tuple[float, str, str, bytes, int]] = {}  # lead_str -> (score, src, head, bytes, ent_mask)
        if exclude_entities:
            entities = [e for e in entities if e not in exclude_entities]
        ent_norms = [_norm_key(e) for e in entities if len(_norm_key(e)) >= 2 or (retain_primary and self.has_exact_lead(e,3,True))]
        n_ents = len(ent_norms)

        # 1) Hot-mounted custom facts (top priority!)
        for cname, ctxt, cb in self.custom_facts:
            cn = _norm_key(ctxt)
            mask = 0
            hits = 0
            for ei, en in enumerate(ent_norms):
                if en in cn:
                    mask |= (1 << ei); hits += 1
            if hits > 0 or (full_prompt and any(len(w) >= 3 and w in cn for w in ent_norms)):
                sc = 40.0 + 18.0 * hits
                cands[ctxt] = (sc, f"custom:{cname}", cname, cb, mask)

        predicate_hits = []
        if self.conn is not None and ent_norms:
            # 2) Exact & Combined-Pair Article-Head Match (O(1) B-tree index!)
            exact_queries: List[Tuple[str, float, int]] = []
            for ei, en in enumerate(ent_norms[:5]):
                exact_queries.append((en, 24.0, 1 << ei))
            # Also try adjacent pair concatenation (e.g. `日本` + `の` + `首都` -> `日本の首都`!)
            if n_ents >= 2:
                for i in range(min(3, n_ents)):
                    for j in range(i + 1, min(4, n_ents)):
                        e1, e2 = ent_norms[i], ent_norms[j]
                        m_pair = (1 << i) | (1 << j)
                        exact_queries.append((e1 + e2, 32.0, m_pair))
                        # Check exact substring in normalized prompt spanning both e1 and e2!
                        pn = _norm_key(full_prompt)
                        p1, p2 = pn.find(e1), pn.find(e2)
                        if p1 >= 0 and p2 >= 0 and abs(p2 - p1) <= 16:
                            lo = min(p1, p2)
                            hi = max(p1 + len(e1), p2 + len(e2))
                            span_sub = pn[lo:hi]
                            if 3 <= len(span_sub) <= 24:
                                exact_queries.append((span_sub, 34.0, m_pair))

            for q_norm, base_sc, q_mask in exact_queries:
                for head, src, prio, lead in self.conn.execute(
                    "SELECT head, src, prio, lead FROM leads WHERE head_norm = ? " +
                    ("ORDER BY def_quality DESC, id " if prefer_subject else "") + "LIMIT 3", (q_norm,)
                ).fetchall():
                    ln = _norm_key(lead)
                    mask = q_mask
                    extra = 0
                    for ei, en in enumerate(ent_norms):
                        if en in ln:
                            mask |= (1 << ei); extra += 1
                    sc = base_sc * float(prio) + 5.0 * extra
                    if prefer_subject:
                        sc += 12 * subject(head, lead)
                        if q_norm == _norm_key(head) and full_prompt.startswith(head):
                            sc += 1
                    prev = cands.get(lead)
                    if prev is None or sc > prev[0]:
                        cands[lead] = (sc, f"lead:{src}", head, lead.encode("utf-8"), mask)

            # 比較する二見出しが一文字の接続で隣接する場合、共通する定義語で略称を解決する。
            named = sorted({e for e in ent_norms if e in _norm_key(full_prompt)}, key=_norm_key(full_prompt).find)
            for left, right in zip(named, named[1:]):
                prompt = _norm_key(full_prompt)
                gap = prompt[prompt.index(left)+len(left):prompt.index(right)]
                if len(gap) != 1 or not gap.isalpha() or not prefer_subject:
                    continue
                options = []
                for key in (left, right):
                    rows = self.conn.execute("SELECT head,src,lead FROM leads WHERE head_norm=?", (key,)).fetchall()
                    if getattr(self,"has_aliases",False):
                        rows += self.conn.execute("SELECT head,src,lead FROM aliases JOIN leads ON leads.id=lead_id WHERE alias=? ORDER BY def_quality DESC, leads.id LIMIT 12", (key,)).fetchall()
                    options.append(rows)
                def defining_terms(row):
                    """例として略称候補の先頭80字から、文字二連の集合を得る。"""
                    text = row[2][:80]
                    return {a+b for a,b in zip(text,text[1:]) if a.isalpha() and b.isalpha()}
                pairs = [(a,b) for a in options[0] for b in options[1] if a[0]!=b[0]]
                if pairs and not self.custom_facts:
                    def similarity(pair):
                        """二つの定義の文字二連に対する Jaccard 係数を返す。"""
                        a,b = map(defining_terms,pair)
                        return len(a & b)/max(1,len(a | b))
                    for key,row in zip((left,right),max(pairs,key=similarity)):
                        head,src,lead = row
                        cands[lead] = (80.,f"lead:{src}",head,lead.encode(),1 << ent_norms.index(key))

            for phrase, _, mask in exact_queries:
                if mask.bit_count() < 2:
                    continue
                for head, src, lead in self.conn.execute(
                    "SELECT head, src, lead FROM predicates JOIN leads ON leads.id = lead_id "
                    "WHERE phrase = ? ORDER BY def_quality DESC, length(head), leads.id LIMIT 2", (phrase,)
                ):
                    if phrase in _norm_key(full_prompt):
                        predicate_hits.append(lead)
                    cands[lead] = (66.0, f"lead:{src}", head, lead.encode(), mask)

            # 3) Prefix & 3-gram Head Fallback ONLY for entities not yet covered by an exact head hit!
            covered_now = 0
            for v in cands.values():
                covered_now |= v[4]
            for ei, en in enumerate(ent_norms[:4]):
                if covered_now & (1 << ei):
                    continue
                # Prefix match on head_norm
                rows = self.conn.execute(
                    "SELECT head, src, prio, lead FROM leads WHERE head_norm >= ? AND head_norm < ? LIMIT 6",
                    (en, en + "\uffff"),
                ).fetchall()
                for head, src, prio, lead in rows:
                    ln = _norm_key(lead)
                    mask = 1 << ei
                    extra = 0
                    for ej, e2 in enumerate(ent_norms):
                        if ej != ei and e2 in ln:
                            mask |= (1 << ej); extra += 1
                    sc = 14.0 * float(prio) + 6.0 * extra - 0.3 * max(0, len(head) - len(en))
                    prev = cands.get(lead)
                    if prev is None or sc > prev[0]:
                        cands[lead] = (sc, f"lead_pref:{src}", head, lead.encode("utf-8"), mask)

        if not cands:
            return []

        if prefer_subject and predicate_hits and not any(v[1].startswith("custom") for v in cands.values()):
            # 質問中の複合述部と完全一致した定義一件で関係が充足する。
            candidates = [cands[predicate_hits[0]]]
        else:
            candidates = cands.values()
        if prefer_subject and not self.custom_facts:
            quoted = []
            for item in candidates:
                head = item[2]
                pos = full_prompt.find(head)
                end = pos + len(head)
                if (0 < pos and end < len(full_prompt)
                        and unicodedata.category(full_prompt[pos-1]) in ("Ps", "Pi")
                        and unicodedata.category(full_prompt[end]) in ("Pe", "Pf")):
                    quoted.append(item)
            if quoted:
                candidates = quoted
        ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
        chosen = [ranked[0]]
        cov = ranked[0][4]
        if top_k > 1 and len(ranked) > 1:
            # Uncovered-Entity-First selection for slot 2!
            uncov = [it for it in ranked[1:] if (it[4] & (~cov)) != 0]
            if uncov:
                chosen.append(uncov[0])
            else:
                # Secondary corroboration if custom fact
                for it in ranked[1:]:
                    if it[1].startswith("custom") and it[0] >= chosen[0][0] * 0.50:
                        chosen.append(it)
                        break

        if retain_primary and len(ent_norms)==2 and all(e not in _norm_key(full_prompt) for e in ent_norms):
            # 表記別名で照合した二主題は、互いの本文に名前があっても独立した定義を読む。
            exact = [next((it for it in ranked if _norm_key(it[2])==e),None) for e in ent_norms]
            if all(exact):
                chosen = exact
        hits = [(round(it[0], 2), it[1], it[2], it[3]) for it in chosen]
        if include_senses and getattr(self,"has_senses",False):
            expanded = []
            for hit in hits:
                _,source,head,_ = hit
                senses = []
                if source.startswith("lead:") and _norm_key(head) in ent_norms:
                    senses = self.conn.execute("SELECT src,lead FROM senses WHERE head_norm=? ORDER BY quality DESC,rowid",
                                               (_norm_key(head),)).fetchall()
                options = [(hit[0],"lead:"+src,head,text.encode()) for src,text in senses]
                if hit in options:
                    options.remove(hit)
                    options.insert(0,hit)
                if retain_primary and hit not in options:
                    options.insert(0,hit)
                expanded.extend(options or [hit])
            hits = expanded
        if exclude_senses:
            hits = [row for row in hits if row[3].decode() not in exclude_senses]
        return hits


# ==============================================================================
# SINGLE MERGED BRAIN v11.1 + WIKI LEAD HIPPOCAMPUS DELIBERATIVE ENGINE (v11.2)
# ==============================================================================

class EntityLookupContext:
    """推論と CPU 学習で共有する見出し・実体照合。"""

    def extract_verified_entities(self, user_prompt: str, gated: bool = False) -> Tuple[List[bytes], List[int], str]:
        """Hippocampus-Verified Boundary-Respecting Entity Extractor (ZERO Hardcoded Language Rules!):
        1) Extracts any quoted title spans inside paired Unicode open/close quotes (`Ps/Pi ... Pe/Pf`, e.g. `『吾輩は猫である』`, `「アストラX-99」`).
        2) Runs BrocaSyntacticParserV100 boundary split without breaking short stems (`len >= 2` chars after trim).
        3) Verifies each candidate span against our OWN WikiLeadHippocampusV112 (`has_exact_lead`) + Dual-IDF:
           - Recovers high-frequency encyclopedic nouns like `日本`, `首都`, `一般相対性理論`, `量子力学`!
           - Eliminates chitchat/greeting words (`こんにちは`, `調子`, `簡単`, `わかり`) cleanly -> `SOCIAL`!"""
        text = user_prompt.strip()
        cand_strs: List[str] = []

        # 1) Universal paired quote/bracket span extraction (e.g. 『吾輩は猫である』, 「アストラX-99」)
        i = 0
        L = len(text)
        while i < L:
            if unicodedata.category(text[i]) in ("Ps", "Pi"):
                for j in range(i + 2, min(L, i + 36)):
                    if unicodedata.category(text[j]) in ("Pe", "Pf"):
                        inner = text[i + 1 : j].strip()
                        if 2 <= len(inner) <= 30 and inner not in cand_strs:
                            cand_strs.append(inner)
                        i = j
                        break
            i += 1

        quoted = set(cand_strs)
        # 2) Boundary-respecting raw spans from Broca L2 Parser (with safe boilerplate check!)
        def encyclopedic(span: str) -> bool:
            """見出し一致と散文・質問の文書頻度を比較して、頻出名詞を救済する。"""
            df = self.parser.chunk_wiki_df.get(span, 0)
            user_rate = math.exp(-self.parser.chunk_u_idf.get(span, 0.0))
            prose_rate = df / max(1, self.parser.n_wiki_docs)
            min_q = 3 if (gated and len(span) <= 2) else 1
            return df >= 8 and prose_rate >= user_rate and self.lead_hip.has_exact_lead(span, min_quality=min_q)

        frames = self.parser.boilerplate_frames
        spans = self.parser._raw_split_content_spans(text)
        if gated:
            # 記号を含む英数字の見出しも辞書照合する。
            import itertools
            def token_class(ch):
                """Unicode の文字体系と数学記号から技術名の連続範囲を識別する。"""
                if ch.isdecimal() or unicodedata.category(ch) in ("Sm","Pd","Po"):
                    return "symbol"
                return unicodedata.name(ch,"").partition(" ")[0]
            classes = [token_class(ch) for ch in text]
            for i in range(1,len(classes)):
                if classes[i] == "symbol":
                    classes[i] = classes[i-1]
            for _, run in itertools.groupby(zip(classes,text),key=lambda pair:pair[0]):
                token = "".join(ch for _,ch in run)
                technical = any(ch.lower() != ch.upper() or unicodedata.category(ch) == "Sm" for ch in token)
                if technical and self.lead_hip.has_exact_lead(token):
                    cand_strs.append(token)
        relation_members = set()
        pos = 0
        previous = None
        for span in spans:
            begin = text.index(span, pos)
            pos = begin + len(span)
            if previous and len(span) >= 2 and pos - previous[0] <= 24 and self.lead_hip.conn:
                phrase = _norm_key(text[previous[0]:pos])
                if self.lead_hip.conn.execute("SELECT 1 FROM predicates WHERE phrase=? LIMIT 1", (phrase,)).fetchone():
                    relation_members.update((previous[1], span))
            previous = (begin, span)
        for span in spans:
            if span in relation_members:
                cand_strs.append(span)
                continue
            clean = unicodedata.normalize("NFKC", span)
            if encyclopedic(clean):
                cand_strs.append(clean)
                continue
            if clean in frames:
                continue
            for size in range(min(6, len(clean)-2), 1, -1):
                edges = [(clean[:size], clean[size:]), (clean[-size:], clean[:-size])]
                matches = [(edge, rest) for edge, rest in edges if edge in frames]
                if matches:
                    edge, rest = matches[0]
                    if (not gated and len(rest) >= 3 and 2 * size < len(clean)) or (rest not in frames and self.lead_hip.has_exact_lead(rest)):
                        clean = rest
                    else:
                        clean = ""
                    break
            if len(clean) >= 2 and clean not in frames:
                cand_strs.append(clean)

        # 3) Verify each candidate strictly via Hippocampus Head Knowledge + Statistical Specificity!
        max_u_idf = math.log(self.parser.n_user_docs + 10.0) if self.parser.n_user_docs > 0 else 10.9
        verified: List[Tuple[float, bytes]] = []
        for cs in cand_strs:
            cb = cs.encode("utf-8")
            min_q = 3 if (gated and len(cs) <= 2) else 1
            if len(cb) < 4 and not (gated and self.lead_hip.has_exact_lead(cs, min_quality=min_q)):
                continue
            has_lead = self.lead_hip.has_exact_lead(cs, min_quality=min_q)
            u_idf = self.parser.chunk_u_idf.get(cs, max_u_idf)
            # Check motor-level content gate
            ms = self.mc.encode(np.frombuffer(cb, dtype=np.uint8)).tolist()
            max_cg = max((float(self.rc.cg_vec[m]) for m in ms if m >= 256), default=(0.8 if any(m < 256 for m in ms) else 0.0))
            c_idf = float(np.mean([float(self.mc.motor_idfs[m]) for m in ms])) if ms else 10.0

            # Exclude pure boilerplate fragments even if a rare Wikipedia stub shares the word (e.g. `こんにちは` / `簡単` / `調子`)
            # Statistical check: conversational/SFT-heavy words have low u_idf (< 6.2) AND low content-gate OR high SFT prompt frequency
            # EXCEPT high-priority proper encyclopedic nouns (verified by exact Lead AND high corpus specificity / multi-kanji)!
            u_in_boiler = unicodedata.normalize("NFKC", cs) in self.parser.boilerplate_frames
            if u_in_boiler and not encyclopedic(cs):
                continue
            # Drop ultra-short hiragana-only / conversational predicates when not an exact quoted proper noun:
            # Purely statistical: check SFT user-prompt IDF vs prose DF (`w_df`)!
            w_df = self.parser.chunk_wiki_df.get(cs, 0)
            # A genuine encyclopedic entity is EITHER:
            #   (a) Exact Article Lead match in WikiLeadHippocampus AND (`max_cg >= 0.45` AND (`len(cb) >= 6` AND (`w_df >= 5` or `u_idf >= 7.2`)))
            #   (b) OOV / custom proper noun (`u_idf >= max_u_idf - 1.0` and `len(cb) >= 6` and `has_lead`)
            is_enc = has_lead and max_cg >= 0.45 and len(cb) >= 6 and (w_df >= 8 or u_idf >= 7.2)
            is_custom_or_oov = (u_idf >= max_u_idf - 1.0) and len(cb) >= 6 and (has_lead or max_cg >= 0.55)
            if gated and not has_lead and cs not in quoted and cs not in relation_members and not w_df:
                continue
            if (gated and has_lead and not u_in_boiler) or cs in relation_members or is_enc or is_custom_or_oov or encyclopedic(cs) or (has_lead and w_df >= 8 and u_idf >= self.parser.tau_u_idf):
                sc = (c_idf * (0.5 + 0.5 * max_cg) + 0.8 * u_idf) * math.sqrt(len(cb)) * (1.4 if has_lead else 1.0)
                verified.append((sc, cb))

        # Deduplicate subsumed sub-spans (e.g. drop `吾輩` if `吾輩は猫である` is already verified!)
        verified.sort(key=lambda x: len(x[1]), reverse=True)
        kept: List[Tuple[float, bytes]] = []
        for sc, eb in verified:
            if any(eb in existing_b for _, existing_b in kept):
                continue
            kept.append((sc, eb))
        kept.sort(key=lambda x: x[0], reverse=True)
        ent_bytes = [eb for _, eb in kept]

        ent_mids: List[int] = []
        for eb in ent_bytes:
            for m in self.mc.encode(np.frombuffer(eb, dtype=np.uint8)).tolist():
                if m >= 256 and m not in ent_mids:
                    ent_mids.append(m)

        intent = "SOCIAL" if len(ent_bytes) == 0 else ("FACT_SINGLE" if len(ent_bytes) == 1 else "FACT_MULTI")
        return ent_bytes, ent_mids, intent

class SingleMergedBrainV112(EntityLookupContext):
    """Fast Single-Core Merged NeuroGraph v11.1 + Persistent Wiki Lead Hippocampus v11.2!"""

    def __init__(
        self,
        v20_merged: DecoupledBrainV20,
        W_res: Optional[np.ndarray],
        b_res: Optional[np.ndarray],
        readout_scale: float,
        sft_engram: Optional[EngramCartridge],
        mc: ContinuousThoughtCortexV50,
        rc: RoleBoundThoughtCortexV60,
        adapter_v80: InstructAdapterV80,
        rgraph: PrefrontalReasoningGraphV90,
        parser: BrocaSyntacticParserV100,
        lead_hip: WikiLeadHippocampusV112,
        flat_hip=None,
        temp_cal: float = 1.08,
    ):
        self.v20 = v20_merged
        self.device = v20_merged.device
        self.runner = BatchedCoreRunner(v20_merged, temp_cal=temp_cal)
        self.W_res_gpu = torch.from_numpy(W_res).to(device=self.device, dtype=torch.float32) if W_res is not None else None
        self.b_res_gpu = torch.from_numpy(b_res).to(device=self.device, dtype=torch.float32) if b_res is not None else None
        self.readout_scale = float(readout_scale)
        self.sft_engram = sft_engram
        self.mc = mc
        self.rc = rc
        self.adapter = adapter_v80
        self.rgraph = rgraph
        self.parser = parser
        self.lead_hip = lead_hip
        self.flat_hip = flat_hip
        from neurograph.evolution_v140 import SleepConsolidatorV140
        self.evolution = SleepConsolidatorV140()
        gen = self.evolution.current()
        self.evolution.verify(gen)
        self.evolution.mount(self, self.evolution.gens/gen)
        self.memory_gate = None
        self._bytes_to_mid: Dict[bytes, int] = {b: i for i, b in enumerate(mc.motor_bytes)}

    def teach_fact(self, topic, fact_text):
        """主題と本文を次回睡眠に向けて記録する。"""
        from neurograph.evolution_v140 import clean
        self.evolution.record("teach", topic, clean(fact_text))

    def record_correction(self, prompt, kind, target=""):
        """例 tool:clock の訂正経験を追記する。"""
        self.evolution.correction(prompt,kind,target)

    def sleep_evolve(self, run_curiosity=True, guard_fn=None):
        """固定検査に合格した世代だけを公開する。"""
        return self.evolution.sleep(self,run_curiosity,guard_fn)

    @torch.no_grad()
    def respond(
        self,
        user_prompt: str,
        use_lead_hip: bool = True,
        max_motor_steps: int = 38,
        beam_width: int = 3,
        top_k_motor: int = 12,
        frontier_cap: int = 24,
        escape_k: int = 3,
        score_critic: bool = True,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        hip_enabled: bool = True,
        use_memory_gate: bool = True,
        temperature: float = 0.0,
        top_p: float = 0.9,
        min_p: float = 0.05,
        seed: Optional[int] = None,
        use_consensus: Optional[bool] = None,
        arbiter_mode: str = "arbiter",
        use_tools: bool = True,
        use_utterance_nodes: bool = True,
        tool_axis_mode: str = "orthogonal",
        record_episode: bool = False,
        use_metacog: bool = True,
    ) -> Tuple[str, bool, Dict]:
        """認識と根拠採否を確定し、発話を生成する。temperature=0 は従来探索、0.7 等は発話候補を抽選する。"""
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or not 0 <= min_p <= 1:
            raise ValueError("temperature >= 0 (finite), 0 < top_p <= 1, 0 <= min_p <= 1")
        rng = random.Random(seed) if temperature else None
        social_path = False

        def sample(options, scores):
            """発話計画の候補と対数重みから一つ選ぶ。認識・検索には使用しない。"""
            return choose(options, scores, rng, temperature, top_p, min_p)

        t0 = time.perf_counter()
        dev = self.device
        mc = self.mc
        rc = self.rc
        runner = self.runner

        if use_memory_gate and self.memory_gate is None:
            self.memory_gate = ThalamicMemoryGateV120.load()
        gate = self.memory_gate if use_memory_gate else None
        panel = None
        if use_consensus is not None:
            arbiter_mode = "consensus" if use_consensus else "single"
        if arbiter_mode not in ("single","consensus","arbiter"):
            raise ValueError("arbiter_mode")
        arbiter = None
        utterance_trace = None
        tool_trace = None
        arbiter_trace = None
        if gate and arbiter_mode == "arbiter":
            from neurograph.consensus_arbiter_v125 import ConsensusArbiterGraphV125, attribute_gap
            if not hasattr(self,"arbiter_graph"):
                self.arbiter_graph = ConsensusArbiterGraphV125.load()
            arbiter = self.arbiter_graph
        if gate and arbiter_mode == "consensus":
            if not hasattr(self,"consensus_gate"):
                self.consensus_gate = ThalamicConsensusGateV124.load()
            panel = self.consensus_gate
        u_bytes = user_prompt.strip().encode("utf-8")
        prior = bytearray()
        ctx_motor_list = []
        for question, answer in (history or [])[-2:]:
            q, a = question.encode(), answer.encode()
            prior.extend(TURN_U_PREFIX + q + TURN_A_PREFIX + a + bytes([TURN_EOT_BYTE]))
            for turn in (q, a):
                ctx_motor_list.extend(mc.encode(np.frombuffer(turn, dtype=np.uint8)).tolist())
        frame_prefix = bytes(prior) + TURN_U_PREFIX + u_bytes + TURN_A_PREFIX
        u_motors = mc.encode(np.frombuffer(u_bytes, dtype=np.uint8))
        ctx_motor_list.extend(u_motors.tolist())
        ctx_motors_arr = np.array(ctx_motor_list, dtype=np.int32)

        # ---- HOP 1: Verified Entity Extraction & Intent Routing ----
        h_qa = self.adapter.predict_qa_intent(rc, ctx_motors_arr)
        node, _ = self.rgraph.route(mc, rc, self.adapter, u_motors, ctx_motors_arr)
        entity_spans, entity_mids, intent_mode = self.extract_verified_entities(user_prompt, use_memory_gate)
        if tool_axis_mode not in ("orthogonal","mixed","off"):
            raise ValueError("tool_axis_mode")
        if not use_tools:
            tool_axis_mode = "off"
        axis1_action = axis2_memory = result = None
        learned = next((row for row in self.lead_hip.learned_facts if row[0] in user_prompt),None) if hip_enabled and use_lead_hip and arbiter else None
        if arbiter and hip_enabled and tool_axis_mode=="orthogonal":
            from neurograph.action_axis_v132 import OrthogonalRouter
            from neurograph.tool_cortex_v130 import norm
            import re
            if not hasattr(self,"orthogonal_router"):
                self.orthogonal_router = OrthogonalRouter()
            spans = [s.decode() for s in entity_spans]
            # マウント本文の実体と直前の主題を、行動提案より先に照合する。
            subjects = spans + self.parser._raw_split_content_spans(user_prompt)
            if history and self.lead_hip.custom_facts:
                subjects += self.parser._raw_split_content_spans(history[-1][0])
            custom = any(name in user_prompt or any(len(s)>=2 and s in fact for s in subjects)
                         for name,fact,_ in self.lead_hip.custom_facts)
            historical = bool(re.search(r"\d{4,}",norm(user_prompt)) and
                              not re.fullmatch(r"[0-9.()+*/\s-]+",norm(user_prompt)))
            guard = "custom" if custom or learned else "numeric_context" if historical else None
            axis1_action,result = self.orthogonal_router.select("action",self,user_prompt,spans,[],
                                            "memory",None,[],guard)
            tool_trace = axis1_action
        if arbiter and result is None:
            entity_spans = arbiter.entities(self,user_prompt,entity_spans)
            entity_mids = list(dict.fromkeys(m for b in entity_spans for m in mc.encode(np.frombuffer(b,dtype=np.uint8)).tolist() if m>=256))
            intent_mode = 'SOCIAL' if not entity_spans else 'FACT_SINGLE' if len(entity_spans)==1 else 'FACT_MULTI'
        pre_carry_ents = [e.decode("utf-8", errors="ignore") for e in entity_spans]
        carried = []
        current_heads = []
        if result is None and gate and history:
            current_heads = self.lead_hip.query_leads([e.decode() for e in entity_spans], user_prompt, prefer_subject=True)
        if result is None and gate and history and not any(src.startswith("custom") or src=="lead:wikipedia" for _,src,_,_ in current_heads):
            old = [e.decode() for e in self.extract_verified_entities(history[-1][0], True)[0]]
            for _, _, head, _ in self.lead_hip.query_leads(old, history[-1][0], prefer_subject=True):
                if self.lead_hip.has_exact_lead(head):
                    carried.append(head)
            if carried:
                entity_spans = [e.encode() for e in carried]
                entity_mids = [m for e in entity_spans for m in mc.encode(np.frombuffer(e,dtype=np.uint8)).tolist() if m >= 256]
                intent_mode = "FACT_SINGLE" if len(carried)==1 else "FACT_MULTI"
        ent_strs = [e.decode("utf-8", errors="ignore") for e in entity_spans]
        is_social = (intent_mode == "SOCIAL")
        want_hip = result is None and hip_enabled and not is_social
        top_k_ag = 2 if len(entity_spans) >= 2 else 1

        # ---- HOP 2 & 3: Evidence Retrieval (1-Factor Switch: NEW WikiLead vs Legacy Flat 6-gram!) ----
        t_h = time.perf_counter()
        agenda_bytes: List[bytes] = []
        agenda_sources: List[str] = []
        agenda_heads: List[str] = []
        has_custom = False

        if want_hip:
            if use_lead_hip:
                hits = self.lead_hip.query_leads(ent_strs, full_prompt=user_prompt, top_k=top_k_ag, prefer_subject=use_memory_gate, include_senses=bool(panel or arbiter),retain_primary=bool(arbiter))
                for _, src, head, lb in hits:
                    agenda_bytes.append(lb)
                    agenda_sources.append(src)
                    agenda_heads.append(head)
                    if src.startswith("custom"):
                        has_custom = True
            elif self.flat_hip is not None:
                # Legacy flat 6-gram baseline on the EXACT SAME entities & single merged brain!
                q_b = b" ".join(entity_spans[:3])
                pool: List[Tuple[float, str, bytes]] = []
                for cart in self.flat_hip.cartridges:
                    for sc, src, ep_b in cart.query(q_b, top_k=6):
                        pool.append((sc, src, ep_b))
                pool.sort(key=lambda x: x[0], reverse=True)
                for _, src, ep_b in pool[:top_k_ag]:
                    cl = trim_utf8_safe(ep_b.strip(), 240)
                    if len(cl) >= 16:
                        agenda_bytes.append(cl)
                        agenda_sources.append(src)
                        agenda_heads.append(src)
                        if src.startswith("custom"):
                            has_custom = True

        decisions = []
        confidences = []
        social_reply = None
        arbiter_gap = None
        if result is not None:
            arbiter_trace = dict(chosen_decision="TOOL_CALL")
        elif arbiter or (panel and agenda_bytes):
            rows = []
            for h,b,s in zip(agenda_heads,agenda_bytes,agenda_sources):
                quality = None
                if getattr(self.lead_hip,"has_senses",False) and s.startswith("lead:"):
                    quality = self.lead_hip.conn.execute("SELECT quality FROM senses WHERE head_norm=? AND lead=?",
                                                        (_norm_key(h),b.decode())).fetchone()
                if arbiter and quality is None:
                    quality=self.lead_hip.conn.execute('select def_quality from leads where head_norm=? and lead=?',(_norm_key(h),b.decode())).fetchone()
                rows.append((h,b,s,float(quality[0]) if quality else 0.))
            if arbiter:
                arbiter_gap = attribute_gap(gate,self,user_prompt,ent_strs,agenda_bytes,has_custom)
                if carried and not arbiter_gap:
                    arbiter_gap = gate.detect_attribute_gap(self,user_prompt,ent_strs,carried,pre_carry_ents,agenda_bytes,has_custom)
                decisions,social_reply,arbiter_trace = arbiter.decide(gate,self,user_prompt,ent_strs,rows,h_qa,node.ans_concept_proto,arbiter_gap)
            else:
                decisions,social_reply = panel.decide(gate,self,user_prompt,ent_strs,rows,h_qa,node.ans_concept_proto)
            accepted = [row for row,d in zip(rows,decisions) if d["active"]]
            # 進化世代では、複数の定義を併記する前に全ての直接主題の裏付けを要求する。
            if (self.generation_id != "gen_000" and arbiter and not has_custom and
                    len({r[0] for r in accepted}) > 1 and
                    any(e in user_prompt and not any(e.encode() in r[1] for r in accepted) for e in ent_strs)):
                accepted = []
                for item in decisions:
                    item.update(active=False,chosen_sense=False)
                arbiter_trace.update(chosen_decision="ABSTAIN",reason="uncovered_entity")
            agenda_heads = [row[0] for row in accepted]
            agenda_bytes = [row[1] for row in accepted]
            agenda_sources = [row[2] for row in accepted]
            confidences = [d["score"] for d in decisions if d["active"]]
            has_custom = any(s.startswith("custom") for s in agenda_sources)
            if social_reply:
                is_social = True
                intent_mode = "SOCIAL"
        elif gate:
            accepted = []
            slot_features = [gate.features(self,user_prompt,ent_strs,h,b,s,h_qa,node.ans_concept_proto)
                             for h,b,s in zip(agenda_heads,agenda_bytes,agenda_sources)]
            primary_supported = any(x[1] for x in slot_features)
            for head, lead, src, features in zip(agenda_heads, agenda_bytes, agenda_sources, slot_features):
                score = gate.score_turn_gate(features) if primary_supported else 0.
                active = score >= gate.tau_turn
                decisions.append(dict(head=head, score=score, active=active))
                if active:
                    accepted.append((head, lead, src))
                    confidences.append(score)
            # 同一資料の複数行を別主題として重複注入しない。
            distinct = {}
            for row, confidence in zip(accepted,confidences):
                key = (row[0],row[2])
                if key not in distinct or confidence > distinct[key][1]:
                    distinct[key] = (row,confidence)
            accepted = [row for row,_ in distinct.values()]
            confidences = [confidence for _,confidence in distinct.values()]
            agenda_heads = [row[0] for row in accepted]
            agenda_bytes = [row[1] for row in accepted]
            agenda_sources = [row[2] for row in accepted]
            has_custom = any(src.startswith("custom") for src in agenda_sources)
        tool_evidence = False
        if arbiter and tool_axis_mode=='mixed' and hip_enabled:
            from neurograph.tool_cortex_v130 import ToolUseArbiterV130
            if not hasattr(self,"tool_cortex"):
                self.tool_cortex = ToolUseArbiterV130()
            tool_trace, result = self.tool_cortex.decide(self,user_prompt,ent_strs,rows,
                arbiter_trace["chosen_decision"],arbiter_gap,carried,any(r[2].startswith("custom") for r in rows))
        elif arbiter and tool_axis_mode=="orthogonal" and result is None and hip_enabled:
            mode = arbiter_trace["chosen_decision"]
            guard = "custom" if has_custom or axis1_action["guard"]=="custom" else None
            tool_trace,result = self.orthogonal_router.select("memory_hop2",self,user_prompt,ent_strs,
                                                   rows,mode,arbiter_gap,carried,guard)
            axis2_memory = dict(decision=mode,judges=decisions,arbiter=dict(arbiter_trace),hop2=tool_trace)
        if result and result.ok:
            # 数字・ISO日時にはコア自身の学習済み文末を接続する。
            passage = result.text
            if passage and passage[-1] not in gate.terminators:
                ending = max(sorted(gate.terminators), key=lambda ch:
                    self.adapter.eot_after_motor.get(int(mc.encode(np.frombuffer(ch.encode(),dtype=np.uint8))[-1]),0.))
                passage += ending
            agenda_heads, agenda_bytes = [""], [passage.encode()]
            agenda_sources, confidences = ["tool:"+tool_trace["called_tool"]], [1.]
            has_custom = tool_evidence = True
            is_social, social_reply, arbiter_gap = False,None,None
            arbiter_trace["chosen_decision"] = "TOOL_CALL"
            intent_mode = "TOOL_CALL"
        hop23_ms = round((time.perf_counter() - t_h) * 1000.0, 2)
        if learned and result is None:
            topic, fact, source = learned
            agenda_heads, agenda_bytes, agenda_sources = [topic], [fact.encode()], [source]
            confidences, has_custom, is_social = [1.], True, False
            arbiter_gap = social_reply = None
            arbiter_trace["chosen_decision"] = "GROUNDED"
            intent_mode = "FACT_SINGLE"
            if axis2_memory:
                axis2_memory.update(decision="GROUNDED", learned_source=source, arbiter=dict(arbiter_trace))
        mode=(arbiter_trace or {}).get("chosen_decision",intent_mode)
        metacog_trace=dict(enabled=False,initial_mode=mode,final_mode=mode,initial_confidence=None,
                           final_confidence=None,confidence_band=None,audit_flags=[],actions_taken=[],revisions=0,
                           self_prior=None,candidate_history=[])
        if use_metacog and arbiter:
            from neurograph.metacog_v152 import Plan,deliberate
            plan=Plan(mode,ent_strs,agenda_heads,[b.decode() for b in agenda_bytes],agenda_sources,decisions,tool_trace)
            final,metacog_trace=deliberate(self,user_prompt,plan,h_qa,node.ans_concept_proto,carried,arbiter_gap,result,
                hip_enabled and use_lead_hip,use_tools and tool_axis_mode!="off")
            if final is not plan:
                ent_strs,agenda_heads,agenda_sources=final.entities,final.heads,final.sources
                agenda_bytes=[s.encode() for s in final.evidence]
                confidences=[metacog_trace["final_confidence"]]*len(agenda_bytes)
                decisions=final.judges
                tool_trace=final.tool or {}
                tool_evidence=bool(tool_trace.get("tool_ok"))
                has_custom=tool_evidence
                from neurograph.tool_cortex_v130 import ToolResult
                result=ToolResult(True,tool_trace["tool_result"]) if tool_evidence else None
                is_social=False
                social_reply=arbiter_gap=None
                arbiter_trace=dict(arbiter_trace,chosen_decision=final.mode,candidates=decisions)
                if axis2_memory:
                    axis2_memory=dict(axis2_memory,decision=final.mode,judges=decisions,arbiter=arbiter_trace)
                if axis1_action and tool_trace.get("category")=="action":
                    axis1_action=tool_trace
                intent_mode=final.mode
        focused_passage = tool_evidence or bool(learned)
        if gate and len(agenda_bytes) == 1 and agenda_sources[0].startswith("lead:"):
            head = agenda_heads[0]
            if head in user_prompt:
                full = agenda_bytes[0].decode()
                terms = [term for term in set(ent_strs + self.parser._raw_split_content_spans(user_prompt))
                         if len(term) >= 2 and term not in head and (term in ent_strs or term not in self.parser.boilerplate_frames)]
                sentences = []
                start = 0
                depth = 0
                for pos, ch in enumerate(full):
                    cat = unicodedata.category(ch)
                    if cat in ("Ps", "Pi"):
                        depth += 1
                    elif cat in ("Pe", "Pf"):
                        depth = max(0, depth - 1)
                    elif ch in gate.terminators and depth == 0:
                        sentences.append(full[start:pos+1].strip())
                        start = pos + 1
                # 原文中で質問の内容語が再登場する箇所は、その直前の一文も文脈として残す。
                if len(sentences) >= 4:
                    # 役割境界の直後に修飾語が残る場合、定義文と共有する辞書語も照合する。
                    for term in list(terms):
                        for width in range(min(len(term), 12), 1, -1):
                            shared = [term[j:j+width] for j in range(len(term)-width+1)
                                      if term[j:j+width] in sentences[0] and term[j:j+width] not in head
                                      and self.lead_hip.has_exact_lead(term[j:j+width], min_quality=3)]
                            if shared:
                                terms.extend(shared)
                                break
                    terms = set(terms)
                    scores = [sum(len(t) for t in terms if t in sent and t not in agenda_heads)
                              for sent in sentences]
                    best = max(range(2, len(scores)), key=scores.__getitem__)
                    if scores[best] and scores[best] > scores[1]:
                        focused_passage = True
                        agenda_bytes[0] = "".join([sentences[0], *sentences[best-1:best+1]]).encode()
        has_evidence = len(agenda_bytes) > 0
        epistemic_abstain = False
        abstain_reason = None
        if gate:
            missing_attr = None
            if social_reply:
                pass
            elif arbiter_trace and arbiter_trace["chosen_decision"]=="ABSTAIN":
                epistemic_abstain = True
                missing_attr = arbiter_gap
                abstain_reason = "attribute_missing:"+arbiter_gap if arbiter_gap else "arbiter"
            elif len(decisions) > 0 and not has_evidence:
                epistemic_abstain = True
                abstain_reason = "gate_rejected"
            elif want_hip and len(ent_strs) > 0 and not has_evidence:
                epistemic_abstain = True
                abstain_reason = "no_lead"
            elif has_evidence:
                missing_attr = arbiter_gap if arbiter else gate.detect_attribute_gap(self, user_prompt, ent_strs, carried, pre_carry_ents, agenda_bytes, has_custom)
                if missing_attr:
                    epistemic_abstain = True
                    abstain_reason = f"attribute_missing:{missing_attr}"
            elif gate.is_ungrounded_knowledge_query(self, user_prompt):
                epistemic_abstain = True
                abstain_reason = "ungrounded_query"
            if epistemic_abstain:
                abst_paths = gate.build_abstention_paths(self, user_prompt, ent_strs, carried, missing_attr=missing_attr)
                if abst_paths:
                    agenda_heads = ["epistemic_abstain"]
                    agenda_bytes = [sample(abst_paths, [-0.3*i for i in range(len(abst_paths))])
                                    if rng else abst_paths[0]]
                    agenda_sources = ["epistemic:abstention"]
                    confidences = [0.95]
                    has_custom = False
                    has_evidence = True
                else:
                    epistemic_abstain = False
                    abstain_reason = None
        if use_utterance_nodes and arbiter_trace and (tool_evidence or (
                arbiter_trace["chosen_decision"]=="SOCIAL" and is_social and not has_evidence and not epistemic_abstain)):
            from neurograph.utterance_cortex_v131 import UtteranceCortex
            if not hasattr(self,"utterance_cortex"):
                self.utterance_cortex = UtteranceCortex()
            value,utterance_trace = self.utterance_cortex.frame(mc,user_prompt,arbiter_trace["chosen_decision"],
                result.text if tool_evidence else social_reply, tool_trace["called_tool"] if tool_evidence else None,
                sample if rng else None)
            if tool_evidence:
                agenda_bytes = [value.encode()]
            else:
                social_reply = value
        if (rng or social_reply) and gate and is_social and not has_evidence:
            options = [(social_reply,1.)] if social_reply else social_options(user_prompt)
            if options:
                replies, scores = zip(*options)
                reply = sample(replies, scores) if rng else replies[0]
                social_path = True
                agenda_heads = [""]
                agenda_bytes = [reply.encode()]
                agenda_sources = ["sft:utterance"]
                confidences = [1.]
                has_evidence = True
        epi_conf = max(confidences, default=0.) if gate else (0.92 if has_evidence else 0.)

        # Build L2 evidence transitions + Fast-Hebbian Working-Memory Suffix Table!
        ep_concepts: Dict[int, float] = {}
        ep_trans1: Dict[int, Dict[int, float]] = {}
        ep_trans2: Dict[Tuple[int, int], Dict[int, float]] = {}
        h_ev_l2 = np.zeros(mc.thought_dim, dtype=np.float32)
        wm_table: Dict[bytes, Dict[int, float]] = {}

        slot_tables = []
        if has_evidence:
            wm_w = {3: 0.45, 4: 0.85, 5: 1.35, 6: 1.85, 8: 2.35, 10: 2.80}
            for rk, ag_b in enumerate(agenda_bytes):
                if gate:
                    ep_concepts, ep_trans1, ep_trans2, wm_table = {}, {}, {}, {}
                    slot_tables.append((ep_concepts, ep_trans1, ep_trans2, wm_table))
                ew = (1.0 / (1.0 + 0.15 * rk)) * (1.45 if agenda_sources[rk].startswith("custom") else 1.25)
                cl_m = mc.encode(np.frombuffer(ag_b, dtype=np.uint8)).tolist()
                for i, mid in enumerate(cl_m):
                    if mid >= 256:
                        cg = float(rc.cg_vec[mid])
                        if cg >= 0.25:
                            ep_concepts[mid] = max(ep_concepts.get(mid, 0.0), ew * (0.4 + 0.6 * cg))
                            em = mc.emb_np[mid]
                            if float(np.dot(em, em)) > 0.1:
                                h_ev_l2 += (ew * cg * float(mc.motor_idfs[mid])) * em
                    if i >= 1:
                        d1 = ep_trans1.setdefault(cl_m[i - 1], {})
                        d1[mid] = d1.get(mid, 0.0) + ew
                    if i >= 2:
                        d2 = ep_trans2.setdefault((cl_m[i - 2], cl_m[i - 1]), {})
                        d2[mid] = d2.get(mid, 0.0) + ew * 1.5
                for p_i in range(3, len(ag_b)):
                    nb = ag_b[p_i]
                    for wl, wv in wm_w.items():
                        if p_i >= wl:
                            d_s = wm_table.setdefault(ag_b[p_i - wl : p_i], {})
                            d_s[nb] = d_s.get(nb, 0.0) + wv * ew
            ne = float(np.linalg.norm(h_ev_l2))
            if ne > 1e-6:
                h_ev_l2 /= ne
        ep_top_motors = sorted(ep_concepts.items(), key=lambda x: x[1], reverse=True)[:48]

        # 根拠中の主題から順方向に読める短い区間をビーム候補に加える。
        evidence_paths = []
        fact_weights = {}
        for head, evidence, src in zip(agenda_heads, agenda_bytes, agenda_sources):
            if src.startswith("learned:") or tool_evidence or social_path or (epistemic_abstain and src == "epistemic:abstention"):
                evidence_paths.append(evidence)
                for motor in mc.encode(np.frombuffer(evidence, dtype=np.uint8)).tolist():
                    if motor >= 256:
                        fact_weights[motor] = 1.0
                continue
            passage = evidence.decode("utf-8", errors="ignore")
            mentions = [passage.find(e) for e in ent_strs if e in passage]
            if passage.startswith(head):
                mentions.append(0)
            if not mentions:
                if gate:
                    evidence_paths.append(b"")
                continue
            start = min(mentions)
            if start and unicodedata.category(passage[start-1]) in ("Ps", "Pi"):
                start -= 1
            passage = passage[start:]
            if gate and passage.startswith(head):
                # 定義見出し直後の括弧注釈を省いた短い提示経路。WM は原文を保持する。
                begin = len(head)
                if begin < len(passage) and unicodedata.category(passage[begin]) == "Ps":
                    end = next((i for i in range(begin+1,len(passage)) if unicodedata.category(passage[i])=="Pe"),None)
                    if end is not None and not any(e in passage[begin:end] for e in ent_strs):
                        passage = passage[:begin] + passage[end+1:]
            evidence_paths.append(passage.encode())
            for motor in mc.encode(np.frombuffer(passage.encode(), dtype=np.uint8)).tolist():
                token = mc.motor_bytes[motor].decode(errors="ignore")
                if token.isnumeric() or (motor >= 256 and mc.motor_idfs[motor] >= mc.tau_idf):
                    fact_weights[motor] = 1.0

        slot_targets = []
        slot_subjects = []
        if gate:
            for head, evidence, src in zip(agenda_heads,agenda_bytes,agenda_sources):
                if src.startswith("learned:") or tool_evidence or social_path or src == "epistemic:abstention":
                    names = [b""]
                else:
                    names = [head.encode()] if not src.startswith("custom") else [e for e in entity_spans if e in evidence]
                slot_subjects.append(names)
                path = evidence_paths[len(slot_targets)]
                first_end = _next_sentence_end_bytes(path, gate.terminators, 0)
                target = first_end
                # 見出しを直接尋ねた短い定義だけ、次文の新規内容語を評価する。
                # 述部逆引きや外部知識・棄権は最初の完結文で十分。
                if (src.startswith("lead:") and len(agenda_heads) == 1 and head in user_prompt
                        and (first_end < node.p25_ans_bytes or sum(e in path[:first_end].decode() for e in ent_strs) >= 2)
                        and first_end < len(path)):
                    second_end = _next_sentence_end_bytes(path, gate.terminators, first_end)
                    if second_end > first_end:
                        second = path[first_end:second_end]
                        first_ids = set(mc.encode(np.frombuffer(path[:first_end], dtype=np.uint8)))
                        second_ids = set(mc.encode(np.frombuffer(second, dtype=np.uint8)))
                        novel = second_ids.difference(first_ids)
                        if sum(m >= 256 and mc.motor_idfs[m] >= mc.tau_idf for m in novel) >= 2:
                            target = second_end
                if rng and target > first_end and not focused_passage:
                    target = sample([target, first_end], [0., -0.35])
                slot_targets.append(len(path) if focused_passage or social_path else target)
        slot_top_motors = [sorted(st[0].items(), key=lambda item: -item[1])[:48] for st in slot_tables] if gate else []
        h_root = mc.compute_root_thought_vec(np.array(entity_mids, dtype=np.int32)) if entity_mids else mc.compute_root_thought_vec(u_motors)
        h_plan_l2 = (0.35 * h_qa + 0.65 * h_ev_l2) if has_evidence else (0.75 * h_qa + 0.25 * node.ans_concept_proto)
        npl = float(np.linalg.norm(h_plan_l2))
        if npl > 1e-6:
            h_plan_l2 = (h_plan_l2 / npl).astype(np.float32)

        if has_evidence and float(np.linalg.norm(h_ev_l2)) > 0.1:
            h_anchor = 0.25 * h_root + 0.75 * h_ev_l2
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

        # ---- HOP 4: Single-Merged-Core Prefill + Working-Memory-Grounded Articulation ----
        st1 = runner.init_batch_state(1)
        if has_evidence:
            replay_safe = trim_utf8_safe(b"\n".join(agenda_bytes), 480)
            rb_list = list(replay_safe)
            for ri in range(len(rb_list) - 1):
                rx = torch.tensor([rb_list[ri]], dtype=torch.int64, device=dev)
                ry = torch.tensor([rb_list[ri + 1]], dtype=torch.int64, device=dev)
                runner.step_batch(st1, rx, next_true=ry, use_engram=False, is_prompt=True)
            st1["byte_hist"].fill_(-1); st1["hist_ptr"].zero_()
            st1["node_delay"].fill_(-1); st1["nd_ptr"].zero_()
            st1["c_eng"].zero_()

        # When grounded by verified evidence, keep Base Engram gentle so it never overwrites grounded numbers/dates!
        base_eng_m = 0.32 if has_evidence else 0.45
        sft_eng_s = 0.85 if has_evidence else 1.15
        wm_scale = (2.15 if has_custom else 1.75) * (1. if gate else epi_conf)
        wm_cap = 5.4 if has_custom else 4.2

        core_entropy = []
        def _step_m(st_in, x_in, nt_in, is_p: bool, slots=None) -> torch.Tensor:
            lg, _ = runner.step_batch(st_in, x_in, next_true=None, use_engram=False, is_prompt=is_p)
            if not is_p:
                if self.W_res_gpu is not None and self.b_res_gpu is not None:
                    lg = lg + (st_in["last_h_act"] @ self.W_res_gpu) * self.readout_scale + self.b_res_gpu.unsqueeze(0)
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
                if gate:
                    core_entropy[:] = ent_g.flatten().tolist()
                    core_lp = lp_c.cpu().numpy()
                    syntax = np.exp(core_lp-core_lp.max(axis=1,keepdims=True))
                if len(self.v20.rack.cartridges) > 0:
                    eb_np, _, _ = self.v20.rack.lookup_batch(keys_np, cq_np, use_context_gate=True)
                    if gate:
                        eb_np = eb_np * syntax
                    lg = lg + torch.from_numpy(eb_np).to(dev) * (base_eng_m * ent_g)
                if self.sft_engram is not None:
                    s_np = np.zeros((B_cur, 256), dtype=np.float32)
                    self.sft_engram.lookup_into(keys_np, cq_np, s_np, use_context_gate=True)
                    np.minimum(s_np, 2.8, out=s_np)
                    if gate:
                        s_np *= syntax
                    lg = lg + torch.from_numpy(s_np).to(dev) * (sft_eng_s * ent_g)
                if wm_table or (gate and slot_tables):
                    wm_np = np.zeros((B_cur, 256), dtype=np.float32)
                    has_hit = False
                    for ri_b in range(B_cur):
                        table = wm_table
                        strength = 1.
                        if gate:
                            slot = slots[ri_b] if slots is not None else 0
                            if slot >= len(slot_tables):
                                continue
                            table = slot_tables[slot][3]
                            strength = confidences[slot]
                        row_raw = bs24_np[ri_b, :10]
                        vn = int(np.sum(row_raw >= 0))
                        if vn >= 3:
                            chr_b = bytes(int(row_raw[k]) for k in range(vn - 1, -1, -1))
                            for wl in (3, 4, 5, 6, 8, 10):
                                if len(chr_b) >= wl:
                                    d_hit = table.get(chr_b[-wl:])
                                    if d_hit:
                                        has_hit = True
                                        for nb_k, w_k in d_hit.items():
                                            wm_np[ri_b, nb_k] += strength * w_k
                    if has_hit:
                        np.clip(wm_np, 0.0, wm_cap, out=wm_np)
                        if gate:
                            # コアが受理する次バイトだけに WM を加える。構文分布は外部記憶加算前。
                            wm_np *= syntax
                        lg = lg + torch.from_numpy(wm_np).to(dev) * (wm_scale * ent_g)
            elif nt_in is not None:
                cfg = self.v20.core.cfg
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
        for i, b in enumerate(raw_in):
            x_t = torch.tensor([b], dtype=torch.int64, device=dev)
            is_last = (i == len(raw_in) - 1)
            nt_u = torch.tensor([raw_in[i + 1]], dtype=torch.int64, device=dev) if (not is_last and is_social) else None
            logits1 = _step_m(st1, x_t, nt_u, is_p=(not is_last))

        st_beams = st1
        logits_beams = logits1
        bytes_to_mid = self._bytes_to_mid
        enc_cache: Dict[bytes, List[int]] = {}
        def _enc_mb(mb_v: bytes) -> List[int]:
            res = enc_cache.get(mb_v)
            if res is None:
                res = mc.encode(np.frombuffer(mb_v, dtype=np.uint8)).tolist()
                enc_cache[mb_v] = res
            return res
        beams = [{
            "score": 0.0, "raw_l1_lp": 0.0, "raw_l2_sc": 0.0, "tot_inh": 0.0, "tot_rep": 0.0,
            "n_bytes": 0, "motor_seq": list(ctx_motor_list), "gen_motors": [], "gen_bytes": bytearray(),
            "seen_m3": set(), "seen_m2": set(), "used_content": set(), "done": False,
        }]
        # 受理済みスロット間で最初に発話するものだけ選ぶ。検索順位と判定は保存する。
        first_slot = (sample(list(range(len(evidence_paths))), [-0.2*i for i in range(len(evidence_paths))])
                      if rng and gate and len(evidence_paths) > 1 else 0)
        constrained = bool((rng or tool_evidence) and gate and evidence_paths and all(evidence_paths))
        beams[0]["span_chain"] = True
        beams[0]["grounding"] = 0.
        beams[0]["uncertainty"] = core_entropy[0] if gate else 0.
        beams[0]["progress"] = tuple((0, frozenset(), False) for _ in evidence_paths)
        finished_beams: List[Dict] = []
        fin_lead_streak = 0
        min_eot_b = 6 if (is_social or epistemic_abstain) else int(max(36, min(68, node.p25_ans_bytes * 0.65)))

        for step_i in range(max_motor_steps):
            cand_parent, cand_mid, cand_l2_base, cand_l2_gated, cand_inh_b, cand_bytes, cand_cg, cand_keep_prio, cand_is_l1_esc = [], [], [], [], [], [], [], [], []
            channel = {}
            span_slots = {}
            mb_escape_jobs = []
            eot_candidates: List[Tuple[float, Dict]] = []

            for wi, bm in enumerate(beams):
                if bm["done"]:
                    continue
                active = next((j for j,(_,_,done) in enumerate(bm["progress"]) if not done),len(evidence_paths))
                if gate and bm.get("active_slot",active)<len(evidence_paths):
                    active = bm.get("active_slot",active)
                if gate:
                    ep_concepts, ep_trans1, ep_trans2, _ = slot_tables[active] if active<len(slot_tables) else ({},{},{},{})
                    ep_top_motors = slot_top_motors[active] if active<len(slot_top_motors) else []
                    epi_conf = confidences[active] if active<len(confidences) else 0.
                if bm["n_bytes"] >= min_eot_b or (gate and evidence_paths and active==len(evidence_paths)):
                    lp_now = torch.log_softmax(logits_beams[wi], dim=-1)
                    eot_lp = float(lp_now[TURN_EOT_BYTE].item())
                    last_gm = bm["gen_motors"][-1] if bm["gen_motors"] else -1
                    eot_prior = self.adapter.eot_after_motor.get(last_gm, 0.0)
                    nb_e = bm["n_bytes"] + 1
                    nm_e = len(bm["gen_motors"]) + 1
                    eot_l2 = 1.8 * math.sqrt(eot_prior) if eot_prior > 0.0 else 0.0
                    len_fin = (0.30 * math.sqrt(eot_prior)) if (bm["n_bytes"] >= min_eot_b and eot_prior >= 0.04) else 0.0
                    eot_sc = (bm["raw_l1_lp"] + eot_lp - bm["tot_inh"]) / (nb_e ** 0.90) + 0.30 * ((bm["raw_l2_sc"] + eot_l2) / (nm_e ** 0.85)) - 0.18 * bm["tot_rep"] + len_fin
                    if gate and evidence_paths:
                        # 得点への寄与は構文・不確実性を通過した根拠量に限定する。
                        # 無関係な続きを足しても、応答全体の被覆密度は上がらない。
                        density = min(1., sum(slot_targets)/max(1,bm["n_bytes"]))
                        eot_sc += mc.tau_idf * density * bm["grounding"] / len(evidence_paths)
                    if gate and len(evidence_paths)>1:
                        pending = sum(not done for _,_,done in bm["progress"])
                        eot_sc -= pending / len(evidence_paths)
                    completed = gate and evidence_paths and active==len(evidence_paths) and bm["span_chain"]
                    if (not constrained or completed) and (eot_lp > -2.8 or (has_evidence and eot_prior >= 0.1) or (completed and eot_prior >= 0.04)):
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
                    _bm=bm, _rec14=recent_14, _last_b=last_b, ep_concepts=ep_concepts, epi_conf=epi_conf,
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
                        l2_g += epi_conf * (2.30 if has_custom else 1.85) * math.tanh(ep_w)
                    if mid_v >= 256 and cg > 0.15:
                        em = mc.emb_np[mid_v]
                        if float(np.dot(em, em)) > 0.1:
                            if has_anchor:
                                cos_anc = float(np.dot(em, h_anchor))
                                dir_aff = mc.directed_affinity(mid_v, root_t_ids, root_t_w)
                                align = cos_anc + 0.18 * math.tanh(dir_aff) + 0.50 * epi_conf * math.tanh(ep_w)
                                if align < 0.18:
                                    inh_b = cg * 2.15 * min(1.3, (0.18 - align) / 0.18)
                                elif has_evidence and ep_w == 0.0 and cg > 0.65 and cos_anc < 0.44:
                                    inh_b = cg * (1.60 if has_custom else 0.95) * (0.44 - cos_anc)
                                l2_b += cg * 0.95 * align
                            l2_g += cg * 1.45 * float(np.dot(em, h_plan_l2))
                    return l2_b, l2_g, inh_b, cg

                generated = bytes(bm["gen_bytes"])
                paths = evidence_paths if generated else evidence_paths[:1]
                if gate:
                    paths = evidence_paths if step_i==0 else evidence_paths[active:active+1]
                if constrained:
                    slot = first_slot if step_i == 0 else active
                    paths = evidence_paths[slot:slot+1]
                for path in paths:
                    slot_index = evidence_paths.index(path) if gate else 0
                    if gate:
                        generated = path[:bm["progress"][slot_index][0]]
                    if not path.startswith(generated) or path == generated:
                        continue
                    suffix = path[len(generated):].decode()
                    # モデルが学習した終端の直前で区切り、EOT と続行を競合させる。
                    segment = ""
                    for ch in suffix[:12]:
                        segment += ch
                        tail_motor = bytes_to_mid.get(ch.encode(), -1)
                        if (ch in gate.terminators if gate else self.adapter.eot_after_motor.get(tail_motor, 0) >= 0.04):
                            break
                    mb = segment.encode()
                    if mb in added_bytes_wi:
                        continue
                    motors = _enc_mb(mb)
                    novel = set(motors).intersection(fact_weights).difference(bm["used_content"])
                    bonus = 4.0 + min(3, len(novel))
                    cand_parent.append(wi)
                    cand_mid.append(bytes_to_mid.get(mb, -1))
                    if gate:
                        channel[wi,mb] = ("span", confidences[slot_index])
                        span_slots[wi,mb] = slot_index
                    cand_l2_base.append(0. if gate else bonus * len(motors))
                    cand_l2_gated.append(bonus * len(motors) if gate else 0.)
                    cand_inh_b.append(0.0)
                    cand_bytes.append(mb)
                    cand_cg.append(0.0)
                    cand_keep_prio.append(True)
                    cand_is_l1_esc.append(False)
                    added_bytes_wi.add(mb)
                if constrained:
                    continue
                if not gate and step_i == 0 and evidence_paths:
                    continue

                if step_i == 0:
                    if not is_social and entity_spans:
                        for rk, es in enumerate(entity_spans[:3]):
                            if es not in added_bytes_wi and b"\n" not in es and b"\r" not in es:
                                mid_e = bytes_to_mid.get(es, -1)
                                l2_b, l2_g, inh_b, cg = _score_cand(mid_e, es, 0.85, extra_gated=2.05 - 0.20 * rk)
                                cand_parent.append(wi); cand_mid.append(mid_e); cand_l2_base.append(l2_b)
                                cand_l2_gated.append(l2_g); cand_inh_b.append(inh_b); cand_bytes.append(es)
                                cand_cg.append(cg); cand_keep_prio.append(True); cand_is_l1_esc.append(False)
                                added_bytes_wi.add(es)
                    else:
                        for om, ow in self.adapter.sft_openers[:8]:
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
                            sft_cands[nm] = max(sft_cands.get(nm, 0.0), 1.45 * nw)
                    if last1 in self.adapter.sft_m1:
                        for nm, nw in self.adapter.sft_m1[last1].items():
                            sft_cands[nm] = max(sft_cands.get(nm, 0.0), 1.00 * nw)
                    for mid_v, sw in sorted(sft_cands.items(), key=lambda x: x[1], reverse=True)[:8]:
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 or b"\r" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.35, extra_gated=1.25 * sw)
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
                        if gate:
                            continuity = gate.continuity(last1,mid_v)
                            if (last1,mid_v) not in gate.pairs:
                                continue
                            channel[wi,mb] = ("base",continuity)
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
                            ev_cands[nm] = max(ev_cands.get(nm, 0.0), 2.3 * w)
                    if last1 in ep_trans1:
                        for nm, w in ep_trans1[last1].items():
                            ev_cands[nm] = max(ev_cands.get(nm, 0.0), 1.6 * w)
                    for emid, ew in ep_top_motors[:12]:
                        if emid not in bm["used_content"] and emid not in recent_14:
                            ev_cands[emid] = max(ev_cands.get(emid, 0.0), 1.05 * ew)
                    for mid_v, ew in sorted(ev_cands.items(), key=lambda x: x[1], reverse=True)[:10]:
                        mb = mc.motor_bytes[mid_v]
                        if mid_v < 256 or b"\r" in mb or b"\n" in mb or mb in added_bytes_wi or last1 == mid_v:
                            continue
                        if gate:
                            channel[wi,mb] = ("transition",epi_conf)
                        l2_b, l2_g, inh_b, cg = _score_cand(mid_v, mb, 0.45, extra_gated=(2.35 if has_custom else 1.85) * epi_conf * math.tanh(ew))
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
                            mb_escape_jobs.append((wi, b0, clen, added_bytes_wi, _score_cand, active))

            for _, dbm in eot_candidates:
                finished_beams.append(dbm)

            if gate and any(b["span_chain"] and b["progress"] and all(d for _, _, d in b["progress"])
                            for _, b in eot_candidates):
                break
            if gate and finished_beams and evidence_paths:
                best_fin_now = max(finished_beams, key=lambda x: x["score"])
                reading_now = any(b["span_chain"] and b["active_slot"] < len(evidence_paths) for b in beams)
                if (not reading_now) and all(d for _, _, d in best_fin_now["progress"]) and best_fin_now["score"] >= beams[0]["score"]:
                    break

            if mb_escape_jobs:
                J = len(mb_escape_jobs)
                j_parents = torch.tensor([x[0] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                st_esc = runner.gather_state(st_beams, j_parents)
                cur_bs = torch.tensor([x[1] for x in mb_escape_jobs], dtype=torch.int64, device=dev)
                esc_bufs = [bytearray([x[1]]) for x in mb_escape_jobs]
                esc_clens = [x[2] for x in mb_escape_jobs]
                for ci_u in range(1, max(esc_clens)):
                    act_e = [ji for ji in range(J) if ci_u < esc_clens[ji]]
                    if not act_e:
                        break
                    if len(act_e) == J:
                        lg_e = _step_m(st_esc, cur_bs, None, is_p=False, slots=[job[5] for job in mb_escape_jobs] if gate else None)
                    else:
                        act_et = torch.tensor(act_e, dtype=torch.int64, device=dev)
                        st_esub = runner.gather_state(st_esc, act_et)
                        lg_e = _step_m(st_esub, cur_bs[act_et], None, is_p=False, slots=[mb_escape_jobs[j][5] for j in act_e] if gate else None)
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
                for ji, (wi, _, _, added_set, score_fn, _) in enumerate(mb_escape_jobs):
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
                if gate:
                    coherent = [i for i,(parent,mb) in enumerate(zip(cand_parent,cand_bytes))
                                if beams[parent]["span_chain"] and (parent,mb) in span_slots]
                    if coherent and len(reserved)<frontier_cap:
                        reserved.append(max(coherent,key=lambda i:beams[cand_parent[i]]["score"]+cand_l2_gated[i]))
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
            # バイト長降順に並べ替え、各ステップの活性候補を先頭連続スライス 0:n_act に統一する。
            len_order = sorted(range(M), key=lambda i: len(cand_bytes[i]), reverse=True)
            cand_parent = [cand_parent[i] for i in len_order]
            cand_mid = [cand_mid[i] for i in len_order]
            cand_l2_base = [cand_l2_base[i] for i in len_order]
            cand_l2_gated = [cand_l2_gated[i] for i in len_order]
            cand_inh_b = [cand_inh_b[i] for i in len_order]
            cand_bytes = [cand_bytes[i] for i in len_order]
            cand_cg = [cand_cg[i] for i in len_order]

            p_idx = torch.tensor(cand_parent, dtype=torch.int64, device=dev)
            st_c = runner.gather_state(st_beams, p_idx)
            log_c = logits_beams[p_idx].clone()
            cand_lens = [len(mb) for mb in cand_bytes]
            max_blen = cand_lens[0]
            cand_l1_lp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_cur_clp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_min_clp = torch.zeros(M, dtype=torch.float32, device=dev)
            cand_has_chr = torch.zeros(M, dtype=torch.bool, device=dev)

            progress = []
            next_slots = []
            current_slots = []
            uncertainties = [0.] * M
            for parent, mb in zip(cand_parent,cand_bytes):
                state = list(beams[parent]["progress"])
                slot = next((j for j,(_,_,done) in enumerate(state) if not done),len(state))
                slot = span_slots.get((parent,mb),beams[parent].get("active_slot",slot))
                current_slots.append(slot)
                if gate and slot < len(state):
                    path = evidence_paths[slot]
                    consumed, covered, _ = state[slot]
                    segment = _enc_mb(mb)
                    covered = covered.union(set(segment).intersection(slot_tables[slot][0]))
                    position = path.find(mb, consumed)
                    if position == consumed or (position >= 0 and set(segment).intersection(slot_tables[slot][0])):
                        consumed = position + len(mb)
                    utterance = bytes(beams[parent]["gen_bytes"]) + mb
                    named = any(name in utterance for name in slot_subjects[slot])
                    terminated = consumed >= slot_targets[slot] and any(mb.endswith(t.encode()) for t in gate.terminators)
                    done = named and (bool(covered) or epistemic_abstain or social_path or tool_evidence) and (consumed >= len(path) or terminated)
                    state[slot] = (consumed,covered,done)
                progress.append(tuple(state))
                if slot<len(state) and not state[slot][2]:
                    next_slots.append(slot)
                else:
                    next_slots.append(next((j for j,(_,_,done) in enumerate(state) if not done),len(state)))
            n_act = M
            for bi in range(max_blen):
                while n_act > 0 and bi >= cand_lens[n_act - 1]:
                    n_act -= 1
                if n_act == 0:
                    break
                slots = [next_slots[i] if bi+1==cand_lens[i] else current_slots[i] for i in range(n_act)] if gate else None
                lp_act = torch.log_softmax(log_c[:n_act], dim=-1)
                yb_t = torch.tensor([cand_bytes[i][bi] for i in range(n_act)], dtype=torch.int64, device=dev)
                step_lp = lp_act[torch.arange(n_act, device=dev), yb_t]
                cand_l1_lp[:n_act] += step_lp
                is_lead = (yb_t & 0xC0) != 0x80
                prev_has = cand_has_chr[:n_act]
                cand_min_clp[:n_act] = torch.where(is_lead & prev_has, torch.minimum(cand_min_clp[:n_act], cand_cur_clp[:n_act]), cand_min_clp[:n_act])
                cand_cur_clp[:n_act] = torch.where(is_lead, step_lp, cand_cur_clp[:n_act] + step_lp)
                cand_has_chr[:n_act] = prev_has | is_lead
                if n_act == M:
                    log_c = _step_m(st_c, yb_t, None, is_p=False, slots=slots)
                else:
                    st_sub = {k_st: v_st[:n_act] for k_st, v_st in st_c.items()}
                    log_sub = _step_m(st_sub, yb_t, None, is_p=False, slots=slots)
                    for k_st in st_c.keys():
                        st_c[k_st][:n_act] = st_sub[k_st]
                    log_c[:n_act] = log_sub
                if gate:
                    for row in range(n_act):
                        uncertainties[row] = core_entropy[row]

            cand_min_clp = torch.where(cand_has_chr, torch.minimum(cand_min_clp, cand_cur_clp), cand_l1_lp)
            l1_lp_cpu = cand_l1_lp.cpu().tolist()
            min_clp_cpu = cand_min_clp.cpu().tolist()
            tau_syn = -2.40 if (has_evidence and step_i < 16) else -1.85
            tau_wch = -5.20 if (has_evidence and step_i < 16) else -3.95

            next_pool = []
            for ci in range(M):
                bm = beams[cand_parent[ci]]
                mid = cand_mid[ci]
                mb = cand_bytes[ci]
                cg = cand_cg[ci]
                l1_step = l1_lp_cpu[ci]
                worst_c = min_clp_cpu[ci]

                if rng:
                    try:
                        mb.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                l1_pb = l1_step / max(1, len(mb))
                g_avg = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, (l1_pb - tau_syn) * 2.2))))
                g_wch = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, (worst_c - tau_wch) * 1.4))))
                g_syn = g_avg * g_wch
                wch_pen = 0.28 * max(0.0, (tau_wch + 0.15) - worst_c)

                l2_step = cand_l2_base[ci] + g_syn * cand_l2_gated[ci] - wch_pen
                grounding = bm["grounding"]
                span_chain = bm["span_chain"] and (cand_parent[ci],mb) in span_slots
                if gate and (cand_parent[ci],mb) in channel:
                    kind, confidence = channel[cand_parent[ci],mb]
                    gain = gate.score_step_gate(g_syn,bm["uncertainty"],confidence)
                    reward = cand_l2_base[ci]+cand_l2_gated[ci]
                    if kind == "span":
                        slot = span_slots[cand_parent[ci],mb]
                        if span_chain:
                            # 連続して読めた最初の一文の被覆率を、実際の注入ゲートで重み付けする。
                            target_bytes = slot_targets[slot]
                            grounding += gain * len(mb) / max(1,target_bytes)
                        prefix = bytes(bm["gen_bytes"])
                        if any(name and name not in prefix and name in prefix+mb for name in slot_subjects[slot]):
                            reward += sum(mc.motor_idfs[m] for m in _enc_mb(mb) if m>=256)
                    l2_step = gain * reward - wch_pen
                inh_step = cand_inh_b[ci] * len(mb)

                m_add = [mid] if mid >= 0 else _enc_mb(mb)
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
                new_used.update(fact_weights.keys() & set(m_add))
                if cg >= 0.45 and mid >= 256 and mid not in new_used:
                    new_used.add(mid)

                raw_l1 = bm["raw_l1_lp"] + l1_step
                raw_l2 = bm["raw_l2_sc"] + l2_step
                tot_inh = bm["tot_inh"] + inh_step
                tot_rep = bm["tot_rep"] + rep_pen
                nb = bm["n_bytes"] + len(mb)
                nm = len(new_mseq) - len(ctx_motor_list)
                total_sc = (raw_l1 - tot_inh) / (nb ** 0.90) + 0.30 * (raw_l2 / (nm ** 0.85)) - 0.18 * tot_rep
                next_pool.append((total_sc, ci, {
                    "score": total_sc, "raw_l1_lp": raw_l1, "raw_l2_sc": raw_l2, "tot_inh": tot_inh,
                    "tot_rep": tot_rep, "n_bytes": nb, "motor_seq": new_mseq,
                    "gen_motors": bm["gen_motors"] + m_add, "gen_bytes": bm["gen_bytes"] + mb,
                    "grounding": grounding, "span_chain": span_chain,
                    "active_slot": next_slots[ci], "progress": progress[ci], "uncertainty": uncertainties[ci],
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
            if gate and beam_width > 1 and len(kept)==beam_width:
                # 一貫した根拠を読む候補を一枠だけ保つ。他の枠は通常の得点順。
                coherent = next((item for item in next_pool if item[2]["span_chain"] and item[2]["grounding"]>0),None)
                if coherent and coherent[1] not in kept_ci:
                    kept[-1],kept_ci[-1] = coherent[2],coherent[1]
            if not kept:
                break
            k_t = torch.tensor(kept_ci, dtype=torch.int64, device=dev)
            st_beams = runner.gather_state(st_c, k_t)
            logits_beams = log_c[k_t].clone()
            beams = kept

            if finished_beams:
                best_fin = max(finished_beams, key=lambda x: x["score"])
                reading = gate and any(b["span_chain"] for b in beams)
                if gate:
                    all_done_fin = bool(evidence_paths) and all(d for _, _, d in best_fin["progress"])
                    if not reading and best_fin["score"] >= beams[0]["score"] - (0.01 if all_done_fin else -0.04):
                        fin_lead_streak += 1
                    else:
                        fin_lead_streak = 0
                    if not reading and ((all_done_fin and fin_lead_streak >= 1) or (best_fin["score"] > beams[0]["score"] + 0.06 and fin_lead_streak >= 2)):
                        break
                else:
                    if not reading and best_fin["score"] > beams[0]["score"] + 0.12 and len(finished_beams) >= beam_width:
                        break

        all_cands = finished_beams or beams
        # 完結候補のうち、受理済みの根拠を読み切ったものを優先する。
        if gate and evidence_paths:
            complete = [b for b in all_cands if b["span_chain"] and all(done for _, _, done in b["progress"])]
            if complete:
                all_cands = complete
        all_cands.sort(key=lambda x: x["score"], reverse=True)
        best = all_cands[0]
        if rng and not evidence_paths and finished_beams:
            # 自由生成では EOT に到達した、反復・抑制の少ない上位候補間だけで抽選する。
            safe = [b for b in all_cands if b["tot_rep"] <= best["tot_rep"]
                    and b["tot_inh"]/b["n_bytes"] <= best["tot_inh"]/best["n_bytes"]
                    and b["score"] >= best["score"] - 0.25 and len(b["gen_bytes"].decode()) >= 2]
            if safe:
                best = sample(safe, [b["score"] for b in safe])
        decode_sec = time.perf_counter() - t0
        ans_b = bytes(best["gen_bytes"])
        try:
            ans_str = ans_b.decode("utf-8", errors="strict")
            valid_utf8 = True
        except UnicodeDecodeError:
            ans_str = ans_b.decode("utf-8", errors="ignore")
            valid_utf8 = False

        bpc = runner.score_text_bpc(frame_prefix.decode("utf-8", errors="ignore") + ans_str, prompt_prefix_len=len(frame_prefix)) if score_critic else -1.0
        if record_episode and valid_utf8 and best.get("done"):
            if metacog_trace["revisions"] and (metacog_trace["final_confidence"]>metacog_trace["initial_confidence"]
                    or epistemic_abstain and metacog_trace["audit_flags"] and "DEMOTE_ABSTAIN" in metacog_trace["actions_taken"]):
                self.evolution.record("metacog_recovery",user_prompt,metacog_trace=metacog_trace)
            tool = tool_trace or {}
            probs = sorted((arbiter_trace or {}).get("arbiter_mode_probs",{}).values(),reverse=True)
            if epistemic_abstain:
                self.evolution.record("abstain",user_prompt,entities=ent_strs,attribute=arbiter_gap)
            spread = max((float(d.get("disagreement",0)) for d in decisions),default=0.)
            if spread>=.75 or (len(probs)>1 and probs[0]-probs[1]<.2):
                self.evolution.record("disagreement",user_prompt,judge_spread=spread)
            if tool.get("tool_ok"):
                self.evolution.record("success",user_prompt,tool["tool_result"],tool=tool["called_tool"])
        return ans_str, valid_utf8, {
            "generation_id": self.generation_id, "metacog_trace":metacog_trace,
            "temperature": temperature, "top_p": top_p, "min_p": min_p, "seed": seed,
            "sft_utterance": social_path,
            "tau_turn": gate.tau_turn if gate else None,
            "axis1_action": axis1_action, "axis2_memory": axis2_memory, "utterance_trace": utterance_trace, "tool_trace": tool_trace, "arbiter_trace": arbiter_trace, "use_memory_gate": use_memory_gate, "gate_decisions": decisions, "consensus_trace": decisions if panel else [], "carried": carried,
            "epistemic_abstain": epistemic_abstain, "abstain_reason": abstain_reason,
            "slot_targets": slot_targets,
            "slot_progress": [(n,len(f),done) for n,f,done in best["progress"]] if gate else [],
            "intent_mode": intent_mode,
            "want_hip": want_hip,
            "entities": ent_strs,
            "agenda_heads": agenda_heads,
            "agenda_sources": agenda_sources,
            "agenda_passages": [b.decode() for b in agenda_bytes],
            "agenda_preview": [b.decode("utf-8", errors="ignore")[:120] for b in agenda_bytes],
            "hop23_ms": hop23_ms,
            "stopped_eot": bool(best.get("done", False)),
            "decode_sec": round(decode_sec, 2),
            "chars_per_sec": round(len(ans_str) / max(decode_sec, 1e-3), 2),
            "fluency_bpc": bpc,
        }
