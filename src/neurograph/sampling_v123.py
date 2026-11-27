"""受理された発話候補の温度抽選と、短い SFT 発話の検索。"""
import json
import math
import unicodedata
from functools import lru_cache
from pathlib import Path
from neurograph.settings import SFT_DIR


# コーパスを走査するときの候補長（Unicode 文字数）と発話検索の制限。
MIN_TURN_CHARS = 3
MAX_PROMPT_CHARS = 40
MAX_REPLY_CHARS = 48
MIN_CLAUSE_OVERLAP = 0.55
COVERAGE_WEIGHT = 0.2
MAX_SOCIAL_OPTIONS = 8


def choose(options, scores, rng, temperature, top_p, min_p):
    """例として二つの完結経路から、相対確率床と累積確率を満たす一つを返す。"""
    order = sorted(range(len(options)), key=scores.__getitem__, reverse=True)
    peak = scores[order[0]]
    weighted = [(i, math.exp((scores[i] - peak) / temperature)) for i in order]
    eligible = [(i, w) for i, w in weighted if w >= min_p and w > 0]
    limit = top_p * sum(w for _, w in eligible)
    kept, weights = [], []
    mass = 0.
    for i, weight in eligible:
        kept.append(options[i])
        weights.append(weight)
        mass += weight
        if mass >= limit:
            break
    return rng.choices(kept, weights=weights)[0]


def clauses(text):
    """句読点で発話を分け、Unicode 正規化済みの比較用部分列を返す。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c if c.isalnum() else " " for c in text).split()


@lru_cache(maxsize=1)
def short_pairs(path):
    """train の先頭ターンから短い一行の完結応答を集める。欠損時は空集合。"""
    pairs = []
    if Path(path).is_file():
        with open(path, encoding="utf8") as stream:
            for line in stream:
                turns = json.loads(line)["turns"]
                if not turns:
                    continue
                q, a = turns[0]
                if (MIN_TURN_CHARS <= len(q) <= MAX_PROMPT_CHARS
                        and MIN_TURN_CHARS <= len(a) <= MAX_REPLY_CHARS and a.isprintable()
                        and unicodedata.category(a[-1]) == "Po" and not any(c.isdigit() for c in a)):
                    pairs.append((clauses(q), a))
    return pairs


@lru_cache(maxsize=128)
def social_options(prompt):
    """質問の各句に対応する短い学習例を探し、上位の重複しない回答と類似度を返す。"""
    observed = clauses(prompt)
    ranked = {}
    matching_question = {}
    for parts, answer in short_pairs(SFT_DIR / "train.jsonl"):
        if not parts or not observed:
            continue
        # 全句の対応を要求する。挨拶が共通でも後ろに無関係な質問が付く例は除く。
        similarities = []
        for part in parts:
            overlap = max((min(len(q), len(part)) / max(len(q), len(part))
                           for q in observed if q in part or part in q), default=0.)
            similarities.append(overlap)
        # 前置きの挨拶だけが違う場合は、末尾の質問句を照合する。
        # 未対応の前置きが回答にも現れる例は持ち込まない。
        if len(parts) >= 2 and len(observed) >= 2 and len(observed[-1]) >= 4:
            last, query = parts[-1], observed[-1]
            if last in query or query in last:
                similarity = min(len(last), len(query)) / max(len(last), len(query))
                if similarity >= 0.8 and all(part not in answer for part in parts[:-1]):
                    matching_question[answer] = max(matching_question.get(answer, 0.), similarity)
        if min(similarities) >= MIN_CLAUSE_OVERLAP:
            score = sum(similarities) / len(parts) + COVERAGE_WEIGHT * min(1., len(parts) / len(observed))
            ranked[answer] = max(ranked.get(answer, 0.), score)
    if not ranked:
        ranked = matching_question
    return sorted(ranked.items(), key=lambda x: -x[1])[:MAX_SOCIAL_OPTIONS]

