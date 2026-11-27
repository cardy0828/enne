"""CPU 上でオンライン成長する局所閉形式の合議ノード。"""
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from neurograph.panel_v124 import ConsensusPanel, grams, unit
from neurograph.memory_gate_v120 import normalize
from neurograph.sampling_v123 import clauses

ROOT = Path("memory_gate_v120")
MODEL = ROOT / "arbiter_v125.pkl"
MODES = ("GROUNDED", "ABSTAIN", "SOCIAL")
JUDGES = ("j_lex", "j_sem", "j_prag", "j_canon", "cross_slot", "disagreement", "min_vote")
MAX_NODES = 64
GROW_SIM = .88


def ideograph(word):
    """単一の Unicode 統合漢字か調べる。"""
    return len(word) == 1 and unicodedata.name(word, "").startswith("CJK UNIFIED IDEOGRAPH")


class ArbiterNode:
    def __init__(self, index, context):
        """出生時のルーティング中心と、ridge の局所十分統計を初期化する。"""
        self.index = index
        self.center = context.copy()
        self.fires = self.dev = 0
        self.counts = np.zeros(3, dtype=int)
        self.xx = np.eye(8) * 4
        self.xy = 4 * np.array([0., .5, .5, .1, .3, 1., -.1, .2])
        self.mm = np.eye(len(context))
        self.my = np.zeros((len(context), 3))

    def absorb(self, context, features, labels, mode):
        """一問の候補群を一括加算し、質問のモードは一度だけ数える。"""
        self.center += context
        self.fires += 1
        self.counts[mode] += 1
        self.mm += np.outer(context, context)
        self.my[:, mode] += context
        if len(features):
            self.xx += features.T @ features
            self.xy += features.T @ labels

    def fit(self):
        """局所 ridge 正規方程式から候補選好と三択の係数を得る。"""
        self.weights = np.linalg.solve(self.xx, self.xy)
        self.policy = np.linalg.solve(self.mm, self.my)


class ConsensusArbiterGraphV125:
    def __init__(self):
        """空のノード列と、train だけから構築する語用記憶を用意する。"""
        self.nodes = []
        self.panel = ConsensusPanel.load()
        self.aliases = defaultdict(list)
        self.pairs = []
        self.postings = defaultdict(list)
        self.lift = {}
        self.openers = {}

    @classmethod
    def load(cls):
        """CPU に凍結済みグラフを復元する。"""
        with MODEL.open("rb") as stream:
            return pickle.load(stream)

    def discourse(self, prompt):
        """各節と短い学習会話との bigram cosine から、応答と連続類似度を得る。"""
        best = (0., -1)
        parts = clauses(prompt)
        last = (0., -1)
        for pos, part in enumerate(parts):
            bits = grams(part)
            counts = Counter(i for g in sorted(bits) for i in self.postings.get(g, ()))
            for i, shared in counts.items():
                query, reply, length = self.pairs[i]
                score = shared / math.sqrt(max(1, len(bits) * length))
                if pos == len(parts)-1:
                    last = max(last,(score,i))
                best = max(best, (score * (.92 if pos == 0 else 1.), i))
        score, index = best
        reply = self.pairs[index][1] if index >= 0 else None
        parts = clauses(prompt)
        if len(parts)>1 and len(parts[0])>=3:
            for opener,echo in self.openers.items():
                if opener.startswith(parts[0]):
                    similarity = math.sqrt(len(parts[0])/len(opener))
                    if similarity > score:
                        score,reply = similarity,echo
        if len(parts)>1 and last[0]>=.55:
            reply=self.pairs[last[1]][1]
        return score,reply

    def entities(self, brain, prompt, raw):
        """境界上の二つの漢字名を、辞書の高品質定義と共通分野で検証する。"""
        spans = brain.parser._raw_split_content_spans(prompt)
        groups = []
        for word in spans:
            if not ideograph(word) or word in brain.parser.boilerplate_frames:
                continue
            if brain.parser.chunk_wiki_df.get(word, 0) < 8:
                continue
            motors = brain.mc.encode(np.frombuffer(word.encode(),dtype=np.uint8))
            if max((brain.rc.cg_vec[m] for m in motors if m>=256),default=0.)<.45:
                continue
            options = [(head,bits) for head,bits in self.aliases.get(word, ())
                       if brain.lead_hip.has_exact_lead(head,3,True)]
            if options:
                groups.append((word, options))
        if len(groups)==1:
            word, options = groups[0]
            context = grams(prompt.replace(word,""))
            if options:
                overlap, head = max((len(bits & context)/math.sqrt(max(1,len(bits)*len(context))),head) for head,bits in options)
                if overlap>.15:
                    return [head.encode()]
        if len(groups) == 2 and len(prompt.split(groups[0][0], 1)[1].partition(groups[1][0])[0]) == 1 and prompt.split(groups[0][0],1)[1][:1]!=self.linker:
            left, right = groups
            choices = [(len(a & b) / math.sqrt(max(1,len(a)*len(b))), ha,hb)
                       for ha,a in left[1] for hb,b in right[1] if ha != hb]
            if choices:
                sim, ha,hb = max(choices)
                if sim > .08:
                    return [ha.encode(),hb.encode()]
        # 所有連鎖で、主題に比べ百科質問への lift が極端に低い中間修飾語を除く。
        if self.linker and prompt.count(self.linker)>=2:
            primary,modifier,*tail = prompt.split(self.linker)
            if primary.encode() in raw and self.lift.get(modifier,0.) < self.lift.get(primary,0.)-.75:
                return [b for b in raw if b!=modifier.encode()]
        return raw

    def describe(self, gate, brain, prompt, ents, rows, qa, proto, gap):
        """質問と候補からルーティング特徴・各判事票・応答候補を決定的に作る。"""
        social, reply = self.discourse(prompt)
        x = self.panel.describe(gate,brain,prompt,ents,rows,qa,proto) if rows else []
        old = self.panel.votes(x) if rows else []
        profile = []
        for (head,lead,source,quality), f, vote in zip(rows,x,old):
            main = brain.lead_hip.conn.execute("select lead from leads where head_norm=?", (normalize(head),)).fetchone()
            primary = bool(main and main[0] == lead.decode())
            first = re.split("[" + re.escape("".join(sorted(gate.terminators))) + "]",lead.decode(),1)[0]
            # 見出しを定義中に再引用する度合いと、独立した主題索引の一致。
            canon = (.5 * primary + .5 * min(1.,quality/5)) / max(1,first.count(head))
            prag = float(1-social) * (.5 + .5 * vote[2])
            four = [vote[0],vote[1],prag,canon]
            profile.append([1.,*four,f[12],np.ptp(four),min(four)])
        features = np.asarray(profile).reshape(-1,8)
        support = any(f[1] and (f[7]<2 or f[9]==0) for f in x)
        custom = any(f[10] for f in x)
        numeric = len(re.findall(r"\d+",prompt)) >= 2
        query = gate.is_ungrounded_knowledge_query(brain,prompt)
        vec = brain.mc.compute_root_thought_vec(brain.mc.encode(np.frombuffer(prompt.encode(),dtype=np.uint8)))
        compact = unit(np.array([s.mean() for s in np.array_split(vec,8)]))
        regime = [1., bool(ents),len(set(ents))>1,support,custom,bool(gap),numeric,query,social,
                  max((self.lift.get(s,0.) for s in ents),default=0.)]
        context = unit(np.r_[regime,features[:,1:].max(axis=0) if rows else np.zeros(7), .3*compact])
        return context, features, reply, social, support

    def route(self, context, grow=False):
        """最も近い中心へ送る。train 中のみ、離れた入力から新しい中心を生やす。"""
        sims = [float(unit(n.center) @ context) for n in self.nodes]
        if grow and len(sims)<MAX_NODES and (not sims or max(sims)<GROW_SIM):
            self.nodes.append(ArbiterNode(len(sims),context))
            return self.nodes[-1], 1.
        index = int(np.argmax(sims))
        return self.nodes[index], sims[index]

    def decide(self, gate, brain, prompt, ents, rows, qa, proto, gap):
        """局所三択と候補順位を計算し、根拠欠落などの構造制約を適用する。"""
        ctx, z, reply, social, support = self.describe(gate,brain,prompt,ents,rows,qa,proto,gap)
        node, sim = self.route(ctx)
        logits = ctx @ node.policy
        eligible = np.array([support and not gap, True, reply is not None])
        if rows and support and social < .65 and not gap:
            eligible[2] = False
        if gap or len(re.findall(r"\d+",prompt)) >= 2 or any(unicodedata.category(c) in ("Ps","Pi") for c in prompt):
            eligible[2] = False
        if not gap and social >= .65 and eligible[2] and not any(s.startswith("custom") for _,_,s,_ in rows):
            logits[2] += 2 * social
        logits[~eligible] = -np.inf
        probs = np.exp(logits-np.max(logits))
        probs /= probs.sum()
        mode = MODES[int(np.argmax(probs))]
        scores = z @ node.weights
        trace = []
        for i,(h,b,s,q) in enumerate(rows):
            trace.append(dict(head=h, source=s,sense=b.decode()[:120],**dict(zip(JUDGES,z[i,1:])),
                              arbiter_score=float(scores[i]),score=float(scores[i]),active=False,chosen_sense=False))
        if mode == "GROUNDED":
            for h in dict.fromkeys(r[0] for r in rows):
                best = max((i for i,r in enumerate(rows) if r[0]==h),key=lambda i:scores[i])
                trace[best].update(active=True,chosen_sense=True)
        details = dict(arbiter_node_id=node.index,arbiter_sim=sim,arbiter_mode_probs=dict(zip(MODES,probs)),
                       arbiter_judge_weights=dict(zip(JUDGES,node.weights[1:])),chosen_decision=mode,candidates=trace)
        return trace, reply if mode=="SOCIAL" else None, details


def attribute_gap(gate,brain,prompt,ents,leads,custom):
    """所有連鎖の末端属性を検査し、中間の時間語に検査を遮らせない。"""
    if not custom and leads and gate.noun_linker and any(prompt.startswith(e+gate.noun_linker) for e in ents):
        parts = prompt.split(gate.noun_linker)
        if len(parts)>2 and brain.parser._raw_split_content_spans(parts[1])==[parts[1]]:
            spans = brain.parser._raw_split_content_spans(parts[-1])
            if spans and parts[-1].startswith(spans[0]) and all(spans[0].encode() not in lead for lead in leads):
                return spans[0]
    # 比較対象が別名に正規化された場合、両定義にその表記があることを確かめる。
    if len(ents)==2 and all(e not in prompt for e in ents) and all(any(e.encode() in b for b in leads) for e in ents):
        return None
    return gate.detect_attribute_gap(brain,prompt,ents,[],ents,leads,custom)

