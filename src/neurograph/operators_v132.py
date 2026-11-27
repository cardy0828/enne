"""演算の根拠を数量次元、文字特徴のラベル間頻度差、カーネルで集約する。"""
from collections import Counter
import math
import re
import unicodedata
from neurograph.tool_cortex_v130 import kernel
from neurograph.train_v130 import fit


def content(text):
    """単一文字でも内容を持ち得る表意文字を Unicode 属性で抽出する。"""
    return {ch for ch in text if unicodedata.name(ch,"").startswith("CJK UNIFIED IDEOGRAPH")}


def train_operator(rows):
    """各クラス内で特徴を含む例数を数え、弱教師の多いクラスへの偏りを補正する。"""
    model=fit(rows)
    documents={op:Counter() for op in "+-*/"}
    totals=Counter()
    for q,_,op in rows:
        documents[op].update(content(q))
        totals[op]+=1
    model.update(documents=documents,totals=totals)
    return model


def ranked_operators(model,q,features):
    """単位保存の候補内で平滑化した文字根拠と近傍の学習出力を比較する。"""
    neighbours=[kernel(features,f) for f in model["features"]]
    scores=neighbours @ model["weights"]
    units=re.findall(r"\d+([^\W\d_])",q)
    choices="+-*/"
    if len(units)==2 and units[0]==units[1] and q.count(units[0])>2:
        choices="+-"
    docs=model["documents"]
    for ch in sorted(content(q)):
        count=sum(docs[o][ch] for o in choices)
        if count<3 or max(docs[o][ch] for o in choices)<.75*count:
            continue
        for op in choices:
            rate=(docs[op][ch]+1)/(model["totals"][op]+1)
            scores["+-*/".index(op)]+=.12*math.log(rate)
    return sorted(((op,float(scores["+-*/".index(op)])) for op in choices),key=lambda pair:-pair[1]),max(neighbours)


def classify(model,q,features):
    """従来の呼び出しには上位の演算子と近傍支持度を返す。"""
    ranking,support=ranked_operators(model,q,features)
    return ranking[0][0],support



