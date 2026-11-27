"""決定済みの雑談と数値の発話候補を CPU で組み立てる（v13.1）。"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import pickle
import re
import unicodedata

import numpy as np
from neurograph.sampling_v123 import clauses

MODEL = Path("memory_gate_v120/utterance_nodes_v131.pkl")


def encode(mc, q):
    """最終節の意図を重視し、文字対集合と正規化済み思考を返す。"""
    parts = clauses(q) or [q]
    last = parts[-1]
    bits = set(zip(last,last[1:]))
    ids = mc.encode(np.frombuffer(last.encode(),dtype=np.uint8))
    vec = mc.compute_root_thought_vec(ids)
    return bits, vec/max(np.linalg.norm(vec),1e-9)


def overlap(a,b):
    """二集合の共有特徴を長さで補正する。"""
    return len(a & b)/math.sqrt(max(1,len(a)*len(b)))


@dataclass
class SocialUtteranceNode:
    index: int
    anchor: set
    center: np.ndarray
    examples: list = field(default_factory=list)
    dev: int = 0

    def score(self,bits,vec):
        """代表節と思考との二種類の近さを返す。"""
        covered = bool(self.anchor) and self.anchor.issubset(bits)
        return .65*overlap(bits,self.anchor)+.20*float(vec@self.center)+.55*covered


def unit_from(q):
    """複数数値の直後に共通する接頭単位、または異なる単位のうち問いの末尾で再言及される単位を抽出する。"""
    text = unicodedata.normalize("NFKC", q)
    matches = list(re.finditer(r"\d+", text))
    if len(matches) < 2:
        return ""
    units = []
    for number in matches:
        suffix = text[number.end():]
        if not suffix or not suffix[0].isalpha():
            return ""
        script = unicodedata.name(suffix[0], "").split(" ")[0]
        end = 1
        while end < len(suffix) and suffix[end].isalpha() and (
                unicodedata.category(suffix[end]) == "Lm" or
                unicodedata.name(suffix[end], "").split(" ")[0] == script):
            end += 1
        units.append(suffix[:end])
    # 1) 全数値の直後文字列の最長共通接頭辞（例: 「枚焼」「枚渡」→「枚」、「冊」「冊渡」→「冊」）
    prefix = units[0]
    for u in units[1:]:
        k = 0
        limit = min(len(prefix), len(u))
        while k < limit and prefix[k] == u[k]:
            k += 1
        prefix = prefix[:k]
        if not prefix:
            break
    if 1 <= len(prefix) <= 6:
        return prefix
    # 2) 異単位（例: 「7台」と「4枚」）の場合は、最後の数値以降の疑問節に唯一再登場する接頭単位を採用
    tail = text[matches[-1].end() + len(units[-1]):]
    matched = []
    for u in units:
        hit = ""
        for k in range(min(6, len(u)), 0, -1):
            if u[:k] in tail:
                hit = u[:k]
                break
        if hit and hit not in matched:
            matched.append(hit)
    return matched[0] if len(matched) == 1 else ""


class UtteranceCortex:
    def __init__(self):
        """学習済み文末・社会発話中心のみを復元する。"""
        with MODEL.open("rb") as stream:
            self.nodes,self.ending = pickle.load(stream)

    def frame(self,mc,q,mode,value,tool,sampler):
        """確定したモードの発話を作り、候補数と実際の選択を返す。"""
        unit = ""
        node_id = None
        count = 1
        if mode=="SOCIAL":
            bits,vec = encode(mc,q)
            node = max(self.nodes,key=lambda n:n.score(bits,vec))
            node_id = node.index
            ranked = {}
            for other,answer in node.examples:
                score = overlap(bits,other)+.5*bool(other and other.issubset(bits))
                ranked[answer] = max(ranked.get(answer,0),score)
            best = max(ranked.values())
            pool = sorted(((s,a) for a,s in ranked.items() if s>=max(.28,best-.20)),reverse=True)[:5]
            if pool:
                options = [a for s,a in pool]
                value = sampler(options,[s*4 for s,a in pool]) if sampler else options[0]
            count = len(pool)
        elif tool in ("clock","calc"):
            unit = unit_from(q) if tool=="calc" else ""
            value += unit+self.ending
        return value,dict(utterance_mode=mode,utterance_node_id=node_id,unit_extracted=unit,
                          candidates_count=count,chosen_candidate=value)
