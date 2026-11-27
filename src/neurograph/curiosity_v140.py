"""棄権の出典照合と、訓練区間の明示的な等式を使う自習。"""
from itertools import islice
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from types import SimpleNamespace

from neurograph.settings import SFT_DIR
from neurograph.tool_cortex_v130 import INDEX, CORPUS, normalize, DeepWiki, norm, thought, arithmetic

TRAIN = (SFT_DIR / "train.jsonl")


def oracle():
    """訓練区間のみを二巡し、質問の二数と一致する完全な整数等式を収集する。"""
    split = json.loads(Path("memory_gate_v120/training_v132.json").read_bytes())["train_records"]
    examples = {}
    with TRAIN.open(encoding="utf8") as f:
        for line in islice(f,split):
            for q,a in json.loads(line)["turns"]:
                nums = re.findall(r"\d+",norm(q))
                if len(nums)!=2 or len(q)>240 or not q.isprintable() or not any(ch.isalpha() for ch in q):
                    continue
                if re.search(r"\d\s*[+*/-]\s*\d", norm(q)):
                    continue
                for x,op,y,z in re.findall(r"(?<![.\d])(\d+)\s*([+*/-])\s*(\d+)\s*=\s*(\d+)(?![.\d])",norm(a)):
                    if nums!=[x,y] or y=="0" or arithmetic(x+op+y)!=z:
                        continue
                    key = re.sub(r"\d+","#",norm(q))
                    examples.setdefault(key,[]).append(dict(prompt=q,kind="tool:calc:"+op,target=z,origin="oracle"))
    return [group[0] for _,group in sorted(examples.items()) if len({r["kind"] for r in group})==1]


def body_fact(brain, q):
    """主題・属格・属性だけの短い照会を全語義の原文と照合し、単一の数量文を返す。"""
    linker = brain.memory_gate.noun_linker
    key = q.strip()
    while key and unicodedata.category(key[-1]).startswith("P"):
        key = key[:-1]
    if not linker or key.count(linker)!=1:
        return None
    head,attr = key.split(linker)
    if len(attr)<2 or not attr.isalpha() or not brain.lead_hip.has_exact_lead(head,min_quality=3):
        return None
    leads = {r[0] for r in brain.lead_hip.conn.execute(
        "select substr(lead,1,180) from leads where head_norm=? union select substr(lead,1,180) from senses where head_norm=?",
        (normalize(head),normalize(head)))}
    evidence = {}
    with sqlite3.connect(f"file:{INDEX}?mode=ro",uri=True) as db,CORPUS.open("rb") as stream:
        for preview,pos,size in db.execute("select preview,pos,chars from bodies where head=?",(normalize(head),)):
            if preview not in leads:
                continue
            stream.seek(pos)
            # 既存索引の窓を超える場合も、同段落の終端を越えない。
            raw = stream.read(4*max(size,8192)).decode("utf-32-le")
            body = raw[:size]
            if size>=3200 and raw[size-1:size+1]!="\n\n":
                body += raw[size:].partition("\n\n")[0]
            endings = re.escape("".join(sorted(brain.memory_gate.terminators)))
            pattern = rf"(?:^|[{endings}\n])\s*("+re.escape(key)+rf"[^{endings}\n]{{1,110}}[{endings}])"
            for match in re.finditer(pattern,body):
                sentence=match[1]
                if any(ch.isdigit() for ch in sentence) and float(thought(brain.mc,sentence)@thought(brain.mc,key))>=.50:
                    evidence[sentence]=dict(head=head,attribute=attr,byte_offset=pos+4*match.start(1),text=sentence)
    return next(iter(evidence.values())) if len(evidence)==1 else None


def explore(brain, rows):
    """一意な見出しと属性が実際の本文に現れる発見だけを候補にする。"""
    mined = oracle()
    held = mined[::4]
    train = [row for i,row in enumerate(mined) if i%4]
    audit = []
    for row in rows:
        if row["kind"]!="abstain":
            continue
        q = row["prompt"]
        heads = [h for h in row.get("entities",[]) if h in q and brain.lead_hip.has_exact_lead(h,min_quality=3)]
        attr = row.get("attribute")
        entry = dict(prompt=q,promoted=False)
        audit.append(entry)
        body = body_fact(brain,q)
        if body:
            train.append(dict(kind="discovery",prompt=q,target=body["text"]))
            entry.update(promoted=True,evidence=body)
            continue
        if len(heads)!=1 or not attr or len(attr)<2 or attr not in q:
            continue
        ctx = SimpleNamespace(brain=brain,q=q,vec=thought(brain.mc,q),carried=[],rows=[],rows_text=b"",gap=attr,mode="ABSTAIN")
        result = DeepWiki().execute(dict(heads=heads,attribute=attr),ctx)
        if result.ok and heads[0] in result.text and attr in result.text and float(thought(brain.mc,result.text)@ctx.vec)>=.6:
            train.append(dict(kind="discovery",prompt=q,target=result.text))
            entry.update(promoted=True,evidence=result.text,head=heads[0])
    return train, dict(discoveries=audit,oracle_total=len(mined),held_out=held)

