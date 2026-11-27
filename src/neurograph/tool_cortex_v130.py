"""CPU 上のプラグイン候補、学習済みノードおよび根拠の検証。"""
import ast
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
import math
import operator
from pathlib import Path
import pickle
import re
import sqlite3
from typing import Protocol
import unicodedata

import numpy as np

from neurograph.build_v131_body_index import CORPUS, INDEX, signature
from neurograph.consensus_arbiter_v125 import normalize, grams as alias_grams

MODEL = Path("memory_gate_v120/tool_nodes_v131.pkl")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
SIGNS = str.maketrans({"×":"*", "÷":"/", "−":"-"})


def norm(text):
    """全角数式を ASCII に正規化する。例 １＋２ → 1+2。"""
    return unicodedata.normalize("NFKC", text).translate(SIGNS).strip()


def thought(mc, text):
    """UTF-8 を既存モーターで符号化し、単位長の CPU ベクトルを返す。"""
    motors = mc.encode(np.frombuffer(text.encode(), dtype=np.uint8))
    vec = mc.compute_root_thought_vec(motors)
    return vec / max(1e-9, np.linalg.norm(vec))


def arithmetic(expr):
    """数値と四則演算だけの有限 AST を有理数で計算する。"""
    if len(expr)>120 or not re.fullmatch(r"[0-9.()+*/\s-]+",expr):
        raise ValueError("syntax")
    tree = ast.parse(expr, mode="eval")
    if len(list(ast.walk(tree)))>48 or not isinstance(tree.body, ast.BinOp):
        raise ValueError("size or shape")

    def visit(node):
        """数値リテラルの十進表記を保存し、部分式を再帰評価する。"""
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            return Fraction(ast.get_source_segment(expr,node))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd,ast.USub):
            value = visit(node.operand)
            return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left),visit(node.right))
        raise ValueError("operator")
    return str(visit(tree.body))


@dataclass
class ToolResult:
    ok: bool
    text: str = ""
    meta: dict = field(default_factory=dict)


class ToolPlugin(Protocol):
    name: str
    description: str

    def propose(self, ctx):
        """状況から信頼度と引数の組を返す。適用不能なら信頼度は0。"""

    def execute(self, args, ctx):
        """根拠を検査し、一時作業記憶へ載せる結果を返す。"""


class ToolRegistry:
    def __init__(self):
        """空のレジストリを作る。"""
        self.tools = {}

    def register(self, tool: ToolPlugin):
        """重複した名前を拒否してプラグインを追加する。"""
        if tool.name in self.tools:
            raise ValueError(tool.name)
        self.tools[tool.name] = tool


def similarity(q, vec, examples):
    """数値をマスクした文字 bigram と意味余弦で近傍を計算する。"""
    bits = grams(q)
    return sorted(((.65*len(bits & g)/max(1,len(bits | g))+.35*float(vec@v), label)
                   for g,v,label in examples), reverse=True)


def grams(q):
    """数字の値に依存しない隣接文字対を返す。"""
    s = re.sub(r"\d+", "#", norm(q))
    return set(zip(s,s[1:]))


def op_features(mc, q):
    """数値マスク済みのモーター・1～3文字の集合と思考をCPUで符号化する。"""
    masked = re.sub(r"\d+", "#", norm(q))
    ids = mc.encode(np.frombuffer(masked.encode(), dtype=np.uint8))
    bits = {masked[i:i+w] for w in (1,2,3) for i in range(len(masked)-w+1)}
    suffixes = re.findall(r"#([^\W\d_])",masked)
    if len(suffixes)==2 and suffixes[0]==suffixes[1]:
        bits.update(("equal-unit",i) for i in range(7))
    bits.update(int(m) for m in ids if m >= 256)
    return bits, thought(mc, masked)


def kernel(left, right):
    """共有特徴のコサインと思考の内積を混合する。"""
    a,x = left
    b,y = right
    return .8*len(a & b)/math.sqrt(max(1,len(a)*len(b))) + .2*float(x@y)


def predict_op(model, features):
    """クラス均衡 ridge の4出力から演算と最近傍類似度を返す。"""
    sims = np.array([kernel(features, row) for row in model["features"]])
    return "+-*/"[np.argmax(sims @ model["weights"])], float(sims.max())


class Calc:
    name = "calc"
    description = "arithmetic expression, quantities, sum, difference, product and quotient"

    def propose(self,c):
        """完全な数式または十分に近い学習文章に演算を割り当てる。"""
        q = norm(c.q)
        if re.fullmatch(r"[0-9.()+*/\s-]+",q) and any(op in q for op in "+-*/"):
            return 1., {"expression":q}
        ns = re.findall(r"\d+", q)
        if len(ns)==2 and c.mode!="GROUNDED" and max(map(len,ns))<=3:
            op, score = predict_op(c.model, op_features(c.brain.mc, q))
            if score >= .22:
                return min(1., score+.40), {"expression":op.join(ns)}
        return 0., {}

    def execute(self,args,c):
        """不正構文、ゼロ除算を失敗として返す。"""
        try:
            return ToolResult(True,arithmetic(args["expression"]))
        except (ValueError, SyntaxError, ArithmeticError):
            return ToolResult(False)


class Clock:
    name = "clock"
    description = "current local date, weekday, wall clock time, UTC, UNIX timestamp"

    def __init__(self,now=None):
        """テストでは now に固定日時を返す関数を渡す。"""
        self.now = now or (lambda: datetime.now().astimezone())

    def propose(self,c):
        """訓練中の時刻応答に近い短い問い合わせだけを受理する。"""
        near = similarity(c.q,c.vec,c.model["clock"])
        support = max((len(grams(c.q.replace(h,"")) & grams(body.decode()[:120])) for h,body,*_ in c.rows), default=0)
        threshold = .40 if c.mode != "GROUNDED" or support <= 2 else .70
        anchored = bool(grams(c.q) & c.model["clock_anchors"])
        if near and ((anchored and near[0][0]>.28 and support<=2) or near[0][0]>max(threshold,.62)) and not re.search(r"\d",c.q):
            return min(1.,near[0][0]+(.45 if anchored else .25)), {}
        return 0., {}

    def execute(self,args,c):
        """タイムゾーン付きローカル時刻と曜日を返す。"""
        return ToolResult(True,self.now().strftime("%Y-%m-%d %H:%M:%S %z %A"))


class DeepWiki:
    name = "deep_wiki"
    description = "retrieve a deeper article passage for a missing attribute or comparison"

    def propose(self,c):
        """不足属性、未解決の質問、複数の直接主題を索引に渡す。"""
        heads = list(dict.fromkeys(c.ents))
        if c.mode=="ABSTAIN" or c.gap:
            return .67, {"heads":heads[:3],"attribute":c.gap}
        if len(heads)==2 and all(h in c.q for h in heads) and all(
                any(r[0]==h for r in c.rows) for h in heads):
            # 共通の分類語を持つ定義同士は一段目で既に比較できる。
            leads = [next(r[1].decode() for r in c.rows if r[0]==h) for h in heads]
            if len(grams(leads[0][:65]) & grams(leads[1][:65]))<4:
                return .66, {"heads":heads,"attribute":None}
        return 0., {}

    def execute(self,args,c):
        """主語を索引で確定し、出典のある完結文だけを採用する。"""
        heads = []
        graph = c.brain.arbiter_graph
        spans = c.brain.parser._raw_split_content_spans(c.q)
        for head in args["heads"]:
            aliases = [w for w in spans if len(w)==1 and any(
                h==head and len(bits & alias_grams(c.q.replace(w,""))) >= 2
                for h,bits in graph.aliases.get(w,()))]
            if not ((len(head)>1 and head in c.q) or aliases or head in c.carried):
                return ToolResult(False)
            heads.append(head)
        if not heads and not args["heads"] and not c.gap:
            # 語義候補は質問先頭の完全見出しに限る。短い別名は既存の照合器に任せる。
            for end in range(min(30,len(c.q)),1,-1):
                candidate = c.q[:end]
                if c.brain.lead_hip.has_exact_lead(candidate,min_quality=3):
                    heads = [candidate]
                    break
        if not heads:
            return ToolResult(False)
        gap = args["attribute"]
        linker = c.brain.memory_gate.noun_linker
        if gap and c.carried and linker and linker in c.q:
            tail = c.q.split(linker)[-1]
            spans = c.brain.parser._raw_split_content_spans(tail)
            if spans:
                gap = spans[0]
        if gap and linker and c.q.count(linker)>1:
            return ToolResult(False)
        evidence = []
        with sqlite3.connect(f"file:{INDEX}?mode=ro",uri=True) as db, CORPUS.open("rb") as f:
            for head in heads:
                key = normalize(head)
                senses = db.execute("select preview,pos,chars from bodies where head=?", (key,)).fetchall()
                verified = c.brain.lead_hip.conn.execute(
                    "select substr(lead,1,180) from leads where head_norm=? union select substr(lead,1,180) from senses where head_norm=?",(key,key))
                previews = {r[0] for r in verified}
                senses = [row for row in senses if row[0] in previews]
                context = grams(c.q.replace(head,""))
                companions = [h for h in heads if h!=head]
                for other in companions:
                    context.update(grams(next((r[1].decode()[:120] for r in c.rows if r[0]==other),"")))
                def alignment(item):
                    """同名語義の冒頭を他の主題・属性との重なり順に比較する。"""
                    bits = grams(item[0][:220])
                    return len(context & bits)/math.sqrt(max(1,len(bits))), float(thought(c.brain.mc,item[0][:180])@c.vec)
                if not senses:
                    return ToolResult(False)
                lead,offset,size = max(senses,key=alignment)
                if len(senses)>1 and companions and not any(grams(lead) & grams(r[1].decode()[:160])
                        for r in c.rows if r[0] in companions):
                    return ToolResult(False)
                # 属格で結ばれた二主題では、属性側の本文に主題が明記された箇所のみを使う。
                relation = next(((h,other) for h in heads for other in heads if h!=other and
                                linker and h+linker+other in c.q),None)
                if relation and head==relation[0]:
                    continue
                row = (offset,size)
                f.seek(row[0])
                data = f.read(row[1]*4).decode("utf-32-le")
                candidates = []
                first = True
                for line in data.splitlines():
                    start, depth = 0, 0
                    for i,ch in enumerate(line):
                        cat = unicodedata.category(ch)
                        if cat in ("Ps","Pi"):
                            depth+=1
                        elif cat in ("Pe","Pf"):
                            depth-=1
                        if ch not in c.brain.memory_gate.terminators or depth:
                            continue
                        sent = line[start:i+1].strip()
                        start=i+1
                        if first:
                            first=False
                            continue
                        if sent in args.get("exclude_snippets",()):
                            continue
                        if not 5<=len(sent)<=110:
                            continue
                        if relation and relation[0] not in sent:
                            continue
                        if gap:
                            if not sent.startswith((head+linker+gap,gap)) or not re.search(r"\d",sent[len(sent.split(gap,1)[0])+len(gap):][:12]):
                                continue
                        elif len(heads)==1 and c.mode=="ABSTAIN":
                            detail={ch for ch in c.q if ch not in head and "CJK UNIFIED" in unicodedata.name(ch,"")}
                            if not (detail.intersection(sent) and len(grams(c.q.replace(head,"")) & grams(sent))>=2):
                                continue
                        elif c.mode=="ABSTAIN":
                            continue
                        mc=c.brain.mc
                        ids=mc.encode(np.frombuffer(sent.encode(),dtype=np.uint8))
                        novel=sum(float(mc.motor_idfs[m]) for m in set(ids) if m>=256 and mc.motor_bytes[m] not in c.rows_text)
                        score=sum(1 for other in heads if other!=head and other in sent)+float(thought(mc,sent)@c.vec)+len(grams(sent)&grams(c.q))/max(1,len(grams(c.q)))+.005*novel/math.sqrt(len(sent))
                        if args.get("exclude_snippets"):
                            score+=len(grams(args.get("residue",""))&grams(sent))
                        candidates.append((score,sent))
                if not candidates:
                    return ToolResult(False)
                evidence.append(max(candidates)[1])
        return ToolResult(True,"".join(evidence),{"heads":heads,"snippets":evidence})


@dataclass
class Context:
    brain: object
    q: str
    ents: list
    rows: list
    mode: str
    gap: str
    carried: list
    model: dict
    vec: np.ndarray
    rows_text: bytes


def feature(vec, prototypes, count, gap, multi):
    """思考・記号密度・属性欠落・複数対象・記述余弦を一つの単位ベクトルにする。"""
    x=np.r_[vec, min(count,3)/2,float(bool(gap)),float(multi),prototypes@vec,.3]
    return x/max(np.linalg.norm(x),1e-9)


class ToolUseArbiterV130:
    def __init__(self, now=None):
        """学習済み中心とプラグインを CPU 上で準備する。"""
        with MODEL.open("rb") as f:
            self.model=pickle.load(f)
        self.registry=ToolRegistry()
        for tool in (DeepWiki(),Calc(),Clock(now)):
            self.registry.register(tool)

    def decide(self,brain,q,ents,rows,mode,gap,carried,custom):
        """近傍ノードの事前重みと候補の根拠から一つの行動を決定する。"""
        vec=thought(brain.mc,q)
        c=Context(brain,q,ents,rows,mode,gap,carried,self.model,vec,b" ".join(r[1] for r in rows))
        x=feature(vec,self.model["prototypes"],len(re.findall(r"\d+",q)),gap,len(ents)>1)
        n=max(self.model["nodes"],key=lambda n: float(x@n["center"]))
        scores={mode:.50}
        proposals={}
        if not custom:
            for name, tool in self.registry.tools.items():
                confidence,args=tool.propose(c)
                if confidence:
                    scores[name]=confidence*(.85+.15*n["counts"][name]/sum(n["counts"].values()))
                    proposals[name]=args
        action=max(scores,key=scores.get)
        trace=dict(tool_node_id=n["id"],tool_action_probs={k:v/sum(scores.values()) for k,v in scores.items()},
                   called_tool=None,tool_args={},tool_ok=None,tool_result_preview="",tool_result="",fallback=None)
        if action not in proposals:
            return trace,None
        result=self.registry.tools[action].execute(proposals[action],c)
        trace.update(called_tool=action,tool_args=proposals[action],tool_ok=result.ok,tool_result=result.text,
                     tool_result_preview=result.text[:160],fallback=None if result.ok else mode)
        return trace,result

