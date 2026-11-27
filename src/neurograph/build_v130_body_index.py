"""既存見出しと原文冒頭を照合し、記事本文の UTF-32 ファイル位置を索引にする（v13.0）。"""
import json
import os
from pathlib import Path
import resource
import sqlite3
import tempfile

from neurograph.settings import DATA_DIR

CORPUS = DATA_DIR / "wikipedia.train"
INDEX = Path("wiki_lead_v112/bodies_v130.sqlite")
BLOCK_CHARS = 16_000_000


def build():
    """16M文字ずつ読み、同じ段落の本文長と冒頭位置を一時DBから公開する。"""
    with sqlite3.connect("file:wiki_lead_v112/v124.sqlite?mode=ro", uri=True) as leads:
        prefixes = {lead[:32]: (head, lead) for head, lead in leads.execute(
            "select head, lead from leads where src='wikipedia'") if len(lead) >= 32}
    fd, tmp = tempfile.mkstemp(dir=INDEX.parent)
    os.close(fd)
    try:
        with sqlite3.connect(tmp) as db, CORPUS.open("rb") as stream:
            db.execute("create table bodies(head text primary key, pos integer, chars integer) without rowid")
            offset, carry = 0, ""
            pending = None
            while True:
                block = stream.read(BLOCK_CHARS * 4)
                parts = (carry + block.decode("utf-32-le")).split("\n\n")
                carry = parts.pop() if block else ""
                batch = []
                for para in parts:
                    first, sep, tail = para.partition("\n")
                    skip = len(first)+1 if sep and len(first.strip()) <= 30 else 0
                    passage = para[skip:].replace("\n", " ")
                    match = prefixes.get(passage[:32])
                    if pending:
                        head, pos, length = pending
                        # 独立した本文冒頭を越えず、見出し付き節を最大1200字まで含める。
                        if not match and skip and length < 1200 and para:
                            length = min(1200, length+2+len(para))
                            pending = (head,pos,length)
                        else:
                            batch.append(pending)
                            pending = None
                    if match and passage.startswith(match[1]):
                        pending = (match[0], 4*(offset+skip), min(1200,len(passage)))
                    offset += len(para)+2
                db.executemany("insert or ignore into bodies values(?,?,?)", batch)
                if not block:
                    break
            if pending:
                db.execute("insert or ignore into bodies values(?,?,?)",pending)
            check = db.execute("pragma quick_check").fetchone()[0]
            if check != "ok":
                raise ValueError(check)
            count = db.execute("select count(*) from bodies").fetchone()[0]
        os.replace(tmp, INDEX)
        print(json.dumps(dict(rows=count, quick_check=check, rss_kib=resource.getrusage(0).ru_maxrss)))
    finally:
        Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    build()
