"""CPU 上で語義候補と対話意図を評価する閉形式パネル。"""
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from neurograph.memory_gate_v120 import normalize

MODEL = Path("memory_gate_v120/consensus_v124.pkl")
GROUPS = ([0, 1, 2, 3, 6, 10, 11], [0, 4, 5, 12], [0, 6, 7, 8, 9, 13, 14, 15])


def grams(text):
    """正規化済み文字列の重複しない二文字組を返す。"""
    text = normalize(text)
    return set(zip(text, text[1:]))


def unit(vector):
    """ゼロベクトルも扱える単位ベクトル化。"""
    return vector / max(np.linalg.norm(vector), np.finfo(float).eps)


class ConsensusPanel:
    @classmethod
    def load(cls):
        """学習済みパネルのみを CPU にロードする。"""
        with MODEL.open("rb") as stream:
            return pickle.load(stream)

    def discourse(self, prompt):
        """質問の二文字組から学習会話の対数尤度比を求める。"""
        tokens = grams(prompt)
        return sum(self.log_odds.get(t, 0.) for t in tokens) / max(1, len(tokens))

    def vectors(self, brain, rows):
        """候補の定義文を CPU の思考空間と文字 bigram 集合へ写像する。"""
        result = []
        for head, lead, source, quality in rows:
            text = lead.decode()
            end = next((i+1 for i,c in enumerate(text) if c in self.terminators),len(text))
            sentence = text[:end]
            motors = brain.mc.encode(np.frombuffer(sentence.encode(),dtype=np.uint8))
            result.append((unit(brain.mc.compute_root_thought_vec(motors)), grams(sentence)))
        return result

    def describe(self, gate, brain, prompt, ents, rows, qa, proto):
        """各語義と別スロットとの最大一致度を求め、判事が使う特徴行列を返す。"""
        vecs = self.vectors(brain, rows)
        pragmatic = self.discourse(prompt)
        desc = []
        for i,(head,lead,source,quality) in enumerate(rows):
            v,g = vecs[i]
            peers = [(w,h) for j,(w,h) in enumerate(vecs) if rows[j][0] != head]
            coherence = max((float(v @ w) * len(g & h)/max(1, math.sqrt(len(g)*len(h)))
                             for w,h in peers),default=0.)
            x = gate.features(brain,prompt,ents,head,lead,source,qa,proto)
            idf = brain.parser.chunk_u_idf.get(head,math.log(brain.parser.n_user_docs+10))
            desc.append(np.r_[x,quality,coherence,pragmatic,idf,math.log1p(len(prompt))])
        return desc

    def votes(self, x):
        """三つの線形スコアを学習標本のスコア区間ごとの経験確率に変換する。"""
        x = np.atleast_2d(x)
        columns = []
        for group, weight, (edges,rates) in zip(GROUPS,self.weights,self.calibration):
            columns.append(rates[np.searchsorted(edges,x[:,group] @ weight)])
        return np.column_stack(columns)

    def evaluate(self, features):
        """学習済みの重みで三票を集約し、不一致と弱い票を反映する。"""
        votes = self.votes(features)
        spread = np.ptp(votes,axis=1)
        design = np.column_stack((np.ones(len(votes)), votes, -spread, votes.min(axis=1)))
        scores = np.clip(design @ self.aggregate,0.,1.)
        return votes, spread, scores

