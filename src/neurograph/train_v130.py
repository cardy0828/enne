"""ツール選択ノードを弱教師で学習する。train の末尾200会話は剪定用に分ける（v13.0）。"""
from collections import Counter
import json
import os
import pickle
import re
import resource
import unicodedata
from pathlib import Path

import numpy as np
import torch

from neurograph.settings import SFT_DIR, require_gpu
from neurograph.tool_cortex_v130 import Calc, Clock, DeepWiki, MODEL, arithmetic, feature, grams, norm, thought, op_features, kernel, predict_op
from neurograph.thought_cortex_v50 import ContinuousThoughtCortexV50


def mine(q,a):
    """二つの入力整数に対して、回答に一意の結果が含まれる演算を探す。"""
    ns=re.findall(r"\d+",norm(q))
    if len(ns)!=2 or max(map(len,ns))>6:
        return None,False
    tokens=re.findall(r"-?\d+(?:\.\d+)?",norm(a))
    matches=[]
    for op in "+-*/":
        try:
            value=arithmetic(op.join(ns))
        except (ValueError,SyntaxError,ArithmeticError):
            continue
        if value in tokens:
            matches.append((op,value))
    if len(matches)!=1:
        return None,False
    op,v=matches[0]
    clean=(v not in ns and len(q)<=160 and len(a)<=240 and
           (tokens[-1]==v or tokens==[v]) and all(t in ns+[v] for t in tokens))
    return op,clean


def fit(rows):
    """対角正則化をクラス頻度に比例させ、少数演算を均衡化する。"""
    features = [f for q,f,op in rows]
    labels = [op for q,f,op in rows]
    counts = Counter(labels)
    matrix = np.array([[kernel(a,b) for b in features] for a in features])
    penalty = np.diag([.15*4*counts[op]/len(rows) for op in labels])
    targets = np.eye(4)[["+-*/".index(op) for op in labels]]
    return dict(features=features,weights=np.linalg.solve(matrix+penalty,targets))


def quoted_number(q):
    """括弧・引用符の内側にある数字を検出する。"""
    depth = 0
    for ch in q:
        kind = unicodedata.category(ch)
        if kind in ("Ps","Pi"):
            depth += 1
        elif kind in ("Pe","Pf"):
            depth = max(0,depth-1)
        elif depth and ch.isdigit():
            return True
    return False


def train():
    """出生は訓練だけで許し、Devで未発火の冗長中心を削除する。"""
    require_gpu()
    mc=ContinuousThoughtCortexV50.load("macro_v70")
    data=[json.loads(line) for line in (SFT_DIR / "train.jsonl").open()]
    split=len(data)-200
    prototypes=np.stack([thought(mc,t.description) for t in (DeepWiki,Calc,Clock)])
    nodes=[]
    model=dict(ops=[],same_unit=[],clock=[],prototypes=prototypes)
    matched,accepted=Counter(),Counter()
    examples, dev_ops = {}, []
    sources = Counter()
    # ISO・時刻形式を含む応答から種を作り、質問と回答双方の共有表現で拡張する。
    seeds = []
    pool = []
    df = Counter()
    for record in data[:split]:
        if record["source"]=="wiki_conv_ja":
            continue
        for q,a in record["turns"]:
            if len(q)>48 or re.search(r"\d",q):
                continue
            bits = grams(q)
            df.update(bits)
            pool.append((q,a,bits))
            if re.fullmatch(r"(?:[01]?\d|2[0-3])[^\W\d_][0-5]?\d[^\W\d_]{1,5}\W?",norm(a)) or (
                    len(q)<=24 and "UNIX" in a.upper() and re.search(r"(?<!\d)\d{10}(?!\d)",a)):
                seeds.append((q,a,bits))
    clock_seeds = {q for q,_,_ in seeds}
    anchors = {g for q,a,bits in seeds for g in bits if df[g]<=max(3,len(pool)*.001) and all(ch.isalpha() for ch in g)}
    hints = {g for q,a,bits in seeds for g in grams(a) if 1<=df[g]<=len(pool)*.005 and all(ch.isalpha() for ch in g)}
    model["clock_anchors"] = anchors
    for q,a,bits in pool:
        if (bits & anchors and len(q)<=22) or (bits & hints and len(grams(a) & hints)>=3 and len(q)<=32):
            clock_seeds.add(q)
    for i,record in enumerate(data):
        for j,(q,a) in enumerate(record["turns"]):
            op,clean=mine(q,a)
            clean = clean and record["source"]!="wiki_conv_ja" and not quoted_number(q) and len(a)<=220 and len(q)<=180
            if clean:
                if i<split:
                    examples[norm(q)] = op
                    sources[str(j)] += 1
                else:
                    dev_ops.append((q,op))
            if i<split and record["source"]!="wiki_conv_ja":
                # 数量の列挙で偶然一致する例を避け、同じ単位の短い記述で
                # 二つの値に続く異なる結果を一意に検算できる場合だけを採る。
                for passage in (q,a):
                    passage = norm(passage)
                    digits = list(re.finditer(r"\d+",passage))
                    for left,right,total in zip(digits,digits[1:],digits[2:]):
                        if max(len(m[0]) for m in (left,right,total))>3:
                            continue
                        segment = passage[left.start():total.end()+1]
                        units = re.findall(r"\d+([^\W\d_])",segment)
                        if len(segment)>70 or len(units)!=3 or len(set(units))!=1 or len(segment)<11:
                            continue
                        if "\n" in segment or quoted_number(segment):
                            continue
                        x,y,z = (int(m[0]) for m in (left,right,total))
                        if min(x,y,z)<2 or z in (x,y):
                            continue
                        ops = [sym for sym,val in zip("+-*/",(x+y,x-y,x*y,x/y)) if val==z]
                        if len(ops)==1:
                            context = segment[:total.start()-left.start()]
                            examples.setdefault(context,ops[0])
            if i<split:
                # 回答中の検算可能な等式も、式そのものの文脈と演算の教師になる。
                for l,sign,r,v in re.findall(r"(?<![.\d])(\d{1,4})\s*([+*/-])\s*(\d{1,4})\s*=\s*(\d+)(?![.\d])",norm(a)):
                    expr=l+sign+r
                    try:
                        if arithmetic(expr)==v:
                            examples.setdefault(expr,sign)
                            at = norm(a).find(expr)
                            if at>0:
                                before = re.split(r"[\n!?。]",norm(a)[:at].strip())[-1].strip()
                                if 5<=len(before)<=70 and not re.search(r"\d",before):
                                    examples.setdefault(before+" "+expr,sign)
                    except (SyntaxError,ValueError,ArithmeticError):
                        pass
            label=None
            if i<split and op:
                matched[op]+=1
            if clean:
                if i<split:
                    accepted[op]+=1
                label="calc"
            elif q in clock_seeds:
                label="clock"
            if i<split and not label and (i%32 or j):
                continue
            if i>=split and j:
                continue
            vec=thought(mc,q[:500])
            if i<split and label:
                model["ops" if clean else "clock"].append((grams(q),vec,op or label))
                units=re.findall(r"\d+([^\W\d_])",norm(q))
                if clean and len(units)==2 and units[0]==units[1]:
                    model["same_unit"].append((grams(q),vec,op))
            # 来歴と答えから学習用にのみ記事の詳細要求・短い対話を弱ラベル化する。
            wiki=record["source"]=="wiki_conv_ja"
            multi=q.count("、")>0
            detail=wiki and len(a)>100
            action=label or ("deep_wiki" if detail else "GROUNDED" if wiki else "SOCIAL" if len(q)<30 else "ABSTAIN")
            x=feature(vec,prototypes,len(re.findall(r"\d+",q)),detail,multi)
            near=max(nodes,key=lambda n:float(n["center"]@x),default=None)
            if i<split:
                if near is None or (float(near["center"]@x)<.73 or
                     (label and label not in near["counts"])):
                    near=dict(id=len(nodes),center=x.copy(),total=np.zeros_like(x),counts=Counter(),dev=0)
                    nodes.append(near)
                near["counts"][action]+=1
                near["total"]+=x
                near["center"]=near["total"]/np.linalg.norm(near["total"])
            else:
                near["dev"]+=1
    dominant=lambda n:n["counts"].most_common(1)[0][0]
    # 希少ツール専用の唯一の中心は保存する。頻出用途の冗長中心を睡眠時に除く。
    model["nodes"]=[n for n in nodes if n["dev"] or
                    (dominant(n) in ("calc","clock") and n is max(
                        (m for m in nodes if dominant(m)==dominant(n)),key=lambda m:m["counts"][dominant(m)]))]
    rows = [(q,op_features(mc,q),op) for q,op in examples.items()]
    model.update(fit(rows))
    cv = []
    # 交差検証では同型の数式を同時に除外し、カーネル行列だけを再利用する。
    matrix = np.array([[kernel(left[1],right[1]) for right in rows] for left in rows])
    groups = [re.sub(r"\d+","#",r[0]) for r in rows]
    for i,(q,feat,op) in enumerate(rows):
        keep = [j for j,g in enumerate(groups) if g!=groups[i]]
        labels = [rows[j][2] for j in keep]
        counts = Counter(labels)
        penalty = np.diag([.15*4*counts[s]/len(keep) for s in labels])
        targets = np.eye(4)[["+-*/".index(s) for s in labels]]
        weights = np.linalg.solve(matrix[np.ix_(keep,keep)]+penalty,targets)
        pred = "+-*/"[np.argmax(matrix[i,keep]@weights)]
        cv.append(dict(q=q,expected=op,predicted=pred))
    dev_predictions = [dict(q=q,expected=op,predicted=predict_op(model,op_features(mc,q))[0]) for q,op in dev_ops]
    with MODEL.open("wb") as f:
        pickle.dump(model,f)
    report=dict(start=0,born=len(nodes),pruned=len(nodes)-len(model["nodes"]),alive=len(model["nodes"]),
                operator_counts=Counter(examples.values()),cv=cv,dev_ops=dev_predictions,mined_turns=sources,split=split,dev=200,matched=matched,accepted=accepted,clock=len(model["clock"]),
                nodes=[dict(id=n["id"],action=dominant(n),counts=n["counts"],dev=n["dev"]) for n in model["nodes"]],
                rss_kib=resource.getrusage(0).ru_maxrss,cuda_bytes=torch.cuda.max_memory_allocated())
    MODEL.with_name("training_v131_tools.json").write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!="nodes"},flush=True)


if __name__=="__main__":
    train()
