"""語義別の冒頭文を原文に照合し、同じ段落内の最大3200字を索引にする（v13.1）。"""
import hashlib
import sqlite3
import unicodedata
from neurograph.memory_gate_v120 import normalize
from pathlib import Path
from neurograph.build_v130_body_index import CORPUS, BLOCK_CHARS

INDEX = Path("wiki_lead_v112/bodies_v131.sqlite")
WINDOW = 3200


def signature(text):
    """冒頭32文字を語義照合用の短い固定長キーへ変換する。"""
    return hashlib.blake2b(text[:32].encode(), digest_size=16).digest()


def build():
    """既存の各語義の完全なリードを照合し、段落境界を越えない位置を公開する。"""
    prefixes = {}
    known = set()
    with sqlite3.connect("file:wiki_lead_v112/v124.sqlite?mode=ro", uri=True) as leads:
        for key, lead in leads.execute("select head_norm,lead from leads union all select head_norm,lead from senses"):
            known.add(key)
            if len(lead) >= 32:
                prefixes.setdefault(lead[:32], []).append((key, lead))
    stage = INDEX.with_suffix(".tmp")
    with sqlite3.connect(stage) as db, CORPUS.open("rb") as stream:
        db.execute("create table bodies(head text, sense blob, pos integer, chars integer, preview text, primary key(head,sense)) without rowid")
        carry, offset = "", 0
        pending = []
        while True:
            block = stream.read(BLOCK_CHARS * 4)
            parts = (carry + block.decode("utf-32-le")).split("\n\n")
            carry = parts.pop() if block else ""
            batch = []
            for para in parts:
                first, sep, _ = para.partition("\n")
                skip = len(first)+1 if sep and len(first.strip()) <= 30 else 0
                passage = para[skip:].replace("\n", " ")
                senses = [(key,lead) for key,lead in prefixes.get(passage[:32],()) if passage.startswith(lead)]
                # 既知の同名見出しの未収録語義を、定義冒頭の括弧境界から補う。
                if not senses:
                    for j,ch in enumerate(passage[:31]):
                        if unicodedata.category(ch)=="Ps":
                            key = normalize(passage[:j])
                            if key in known:
                                senses = [(key,passage)]
                            break
                if pending:
                    old, preview = pending[0][0],pending[0][4]
                    shared = {preview[i:i+3] for i in range(len(preview)-2)
                              if preview[i:i+3] in passage[:500]}
                    contiguous = 4*offset==pending[0][2]+4*(pending[0][3]+2)
                    if not senses and skip and contiguous and (old in normalize(passage[:500]) or len(shared)>=4):
                        for row in pending:
                            row[3] = min(WINDOW,row[3]+len(para)+2)
                    else:
                        batch.extend(pending)
                        pending = []
                if senses:
                    pending = [[key,signature(lead),4*(offset+skip),min(len(passage),WINDOW),lead[:180]]
                               for key,lead in senses]
                offset += len(para)+2
            db.executemany("insert or ignore into bodies values(?,?,?,?,?)", batch)
            if not block:
                db.executemany("insert or ignore into bodies values(?,?,?,?,?)",pending)
                break
        if db.execute("pragma quick_check").fetchone() != ("ok",):
            raise ValueError("quick_check")
        print(db.execute("select count(*) from bodies").fetchone(), flush=True)
    stage.replace(INDEX)


if __name__ == "__main__":
    build()
