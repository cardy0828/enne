"""検索した根拠を使うかどうかを決める記憶ゲート。重みとモーター接続頻度を CPU に保持する（v12.0）。"""
import json
import math
import pickle
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from neurograph.settings import SFT_DIR

MODEL = Path("memory_gate_v120/memory_gate_v120.pkl")
FEATURES = ("bias", "primary_hit", "idf_coverage", "subject", "qa_cos", "answer_cos",
            "salience", "numbers", "operators", "unmatched_numbers", "custom")


def normalize(value):
    """文字列を見出し照合用に正規化する。"""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def subject(head, passage):
    """冒頭の見出しに括弧または短い役割語と休止が続くかを調べる。"""
    text = normalize(passage)
    if text and unicodedata.category(text[0]) in ("Ps", "Pi"):
        text = text[1:]
    key = normalize(head)
    if not key or not text.startswith(key):
        return 0.
    tail = text[len(key):].lstrip()
    return float(any(unicodedata.category(ch).startswith("P") for ch in tail[:2]))


class ThalamicMemoryGateV120:
    @classmethod
    def load(cls):
        """保存済み統計を CPU にロードする。"""
        obj = cls()
        with MODEL.open("rb") as stream:
            obj.__dict__.update(pickle.load(stream))
        return obj

    def features(self, brain, prompt, entities, head, lead, source, qa, proto):
        """検索スロットと質問から、回帰と推論で共通の特徴量を作る。"""
        passage = lead.decode()
        maximum = math.log(brain.parser.n_user_docs + 10)
        weights = [brain.parser.chunk_u_idf.get(e, maximum) for e in entities]
        hits = [normalize(e) in normalize(passage) or normalize(e) == normalize(head) for e in entities]
        primary = max(range(len(entities)), key=lambda i: (weights[i], len(entities[i])), default=None)
        motors = brain.mc.encode(np.frombuffer(lead, dtype=np.uint8))
        vec = brain.mc.compute_root_thought_vec(motors)
        def cosine(other):
            """スロットと応答意図ベクトル間の正規化内積を返す。"""
            return float(np.dot(other, vec) / max(1e-8, np.linalg.norm(other)*np.linalg.norm(vec)))
        content = brain.parser._raw_split_content_spans(prompt)
        size = sum(len(s.encode()) for s in content if s not in brain.parser.boilerplate_frames)
        import re
        numbers = set(re.findall(r"\d+",prompt))
        unmatched = numbers.difference(re.findall(r"\d+",passage))
        return np.array([1., primary is not None and hits[primary],
                         sum(w*h for w,h in zip(weights,hits))/max(1.,sum(weights)),
                         subject(head, passage), cosine(qa), cosine(proto),
                         min(1., sum(len(e.encode()) for e in entities)/max(1,size)),
                         sum(ch.isdecimal() and (i == 0 or not prompt[i-1].isdecimal()) for i,ch in enumerate(prompt)),
                         sum(unicodedata.category(ch) == "Sm" for ch in prompt), len(unmatched), source.startswith("custom")])

    def score_turn_gate(self, features):
        """重複を持たない根拠を閉じ、リッジ予測を確信度に変換する。"""
        estimate = float(np.clip(features @ self.weights, 0., 1.))
        # 複数の数値を持つ質問では、根拠が実際に説明する数量の割合を使う。
        numeric_mentions, missing = features[7], features[9]
        support = 1. if numeric_mentions < 2 else max(0., 1. - missing/numeric_mentions)
        return estimate * support if features[2] else 0.

    def continuity(self, previous, motor):
        """応答内の条件付き頻度を独立出現頻度と比較する。"""
        prior = (self.unigram.get(motor,0)+1)/(self.total+len(self.unigram))
        count = self.pairs.get((previous,motor),0)
        odds = (count + prior) / ((1+self.outgoing.get(previous,0))*prior)
        return odds/(1+odds)

    def score_step_gate(self, syntax, uncertainty, confidence):
        """構文受理率とコアの不確実性で外部チャネルを調節する。"""
        return (syntax * uncertainty * confidence) ** (1. / 3.)

    def is_ungrounded_knowledge_query(self, brain, prompt):
        """挨拶などの雑談開始句を含まず、高IDF内容語が複数ある無根拠質問を検出する。"""
        p_norm = normalize(prompt)
        openers = getattr(self, "social_openers", [])
        if any(p_norm.startswith(op) for op in openers if op):
            return False
        raw = brain.parser._raw_split_content_spans(prompt)
        max_u = math.log(brain.parser.n_user_docs + 10.0) if brain.parser.n_user_docs > 0 else 10.9
        high_idf = [
            s for s in raw
            if len(s) >= 2
            and normalize(s) not in brain.parser.boilerplate_frames
            and brain.parser.chunk_u_idf.get(s, max_u) >= 8.5
        ]
        return len(high_idf) >= 2

    def detect_attribute_gap(self, brain, prompt, ent_strs, carried, pre_carry_ents, agenda_bytes, has_custom):
        """質問が要求する百科事典的属性名詞が、取得済み根拠に全く含まれないかを検査する。"""
        if has_custom or not agenda_bytes:
            return None
        # 属性は同じ主題の定義文で検証する。後段で別地域等に帰属する属性を流用しない。
        definitions = []
        for lead in agenda_bytes:
            passage = normalize(lead.decode())
            end = min((passage.find(t) for t in self.terminators if t in passage), default=len(passage))
            definitions.append(passage[:end])
        joined = " ".join(definitions)
        linker = self.noun_linker
        if linker and (carried or any(prompt.startswith(e + linker) for e in ent_strs)):
            tail = prompt.partition(linker)[2]
            spans = brain.parser._raw_split_content_spans(tail)
            if spans and tail.startswith(spans[0]) and spans[0] not in joined and spans[0] not in brain.parser.boilerplate_frames:
                return spans[0]
        skip = {normalize(e) for e in ent_strs} | {normalize(e) for e in carried} | {normalize(e) for e in pre_carry_ents}
        max_u = math.log(brain.parser.n_user_docs + 10.0) if brain.parser.n_user_docs > 0 else 10.9
        for span in brain.parser._raw_split_content_spans(prompt):
            sn = normalize(span)
            if len(sn) < 2 or sn in skip:
                continue
            if not brain.lead_hip.has_exact_lead(span):
                continue
            w_df = brain.parser.chunk_wiki_df.get(span, 0)
            u_idf = brain.parser.chunk_u_idf.get(span, max_u)
            if w_df >= 10 and u_idf >= 4.5 and sn not in joined:
                return span
        return None

    def build_abstention_paths(self, brain, prompt, ent_strs, carried, missing_attr=None):
        """学習済みの棄権句だけから、根拠不足時の候補発話経路を組み立てる。"""
        import re
        ent_sufs = getattr(self, "abst_entity_suffixes", [])
        gen_paths = getattr(self, "abst_generic_paths", [])
        linker = getattr(self, "noun_linker", "")
        numbers = re.findall(r"\d+", prompt)
        paths = []
        if missing_attr:
            subj = carried[0] if carried else (ent_strs[0] if ent_strs else "")
            anchor = f"{subj}{linker}{missing_attr}" if (subj and linker) else (missing_attr or subj)
            for suf in ent_sufs[:2]:
                paths.append(f"{anchor}{suf}".encode("utf-8"))
        elif ent_strs and len(numbers) < 2:
            # 辞書に載らない実体（架空語・未知語）を優先して主題に据える。
            target = next((e for e in ent_strs if not brain.lead_hip.has_exact_lead(e)), ent_strs[0])
            for suf in ent_sufs[:2]:
                paths.append(f"{target}{suf}".encode("utf-8"))
        for gp in gen_paths[:2]:
            paths.append(gp.encode("utf-8"))
        return paths


def _is_question(text):
    """末尾が Unicode の疑問符カテゴリ名かどうかを言語非依存に判定する。"""
    s = text.strip()
    return bool(s) and "QUESTION MARK" in unicodedata.name(s[-1], "")


def _discover_abstention_memory(brain, train_turns, terminators):
    """train のみから言語ルールゼロ・統計的基準で認識的棄権表現を自動発見する。"""
    from collections import defaultdict
    # Pass 1: Assistant 側モーター文書頻度 + 雑談開始句 + 名詞間接続子（1文字役割語）を収集
    ans_motor_df = Counter()
    social_opener_cnt = Counter()
    linker_cnt = Counter()
    role_chars = {
        brain.mc.motor_bytes[int(m)].decode("utf-8", errors="ignore")
        for m in brain.rc.role_motor_ids.tolist()
        if len(brain.mc.motor_bytes[int(m)].decode("utf-8", errors="ignore")) == 1
    }
    short_qa = []
    for is_first, u, a in train_turns:
        u = u.strip()
        a = a.strip()
        if not u or not a:
            continue
        am = brain.mc.encode(np.frombuffer(a.encode("utf-8"), dtype=np.uint8)).tolist()
        ans_motor_df.update(set(am))
        if len(u) <= 85 and len(a) <= 85 and "\n" not in u and "\n" not in a:
            short_qa.append((is_first, u, a, am))

    # Pass 2: 質問を「無根拠・数字なし」vs「有根拠または数字あり」に分割し、
    ungr_vecs = defaultdict(list)
    ungr_topics = defaultdict(set)
    ungr_cnt = Counter()
    ungr_zrare = Counter()
    gr_cnt = Counter()
    zero_rare_turns = []
    n_ungr = 0
    n_gr = 0

    for is_first, u, a, am in short_qa:
        if sum(unicodedata.category(ch) == "Po" and ch != u[-1] for ch in u) >= 2:
            continue
        if any(unicodedata.category(c) in ("Ps", "Pi") for c in u):
            continue
        u_ents = [x.decode("utf-8", errors="ignore") for x in brain.extract_verified_entities(u, True)[0]]
        if not u_ents:
            if is_first and len(u) <= 20 and len(a) <= 50:
                um0 = brain.mc.encode(np.frombuffer(u.encode("utf-8"), dtype=np.uint8)).tolist()
                if um0:
                    op = normalize(brain.mc.motor_bytes[um0[0]].decode("utf-8", errors="ignore"))
                    if len(op) >= 3:
                        social_opener_cnt[op] += 1
            continue
        # 2実体間に挟まる1文字役割語の最頻値から名詞接続子を統計学習する。
        if len(u_ents) >= 2:
            for e1, e2 in zip(u_ents, u_ents[1:]):
                p1 = u.find(e1)
                if p1 >= 0:
                    mid = u[p1 + len(e1): p1 + len(e1) + 1]
                    if mid in role_chars and u[p1 + len(e1) + 1:].startswith(e2):
                        linker_cnt[mid] += 1
        if not _is_question(u) or _is_question(a) or len(a) < 10 or any(len(e) > 12 for e in u_ents):
            continue
        if not any(a.endswith(t) for t in terminators):
            continue
        um_arr = brain.mc.encode(np.frombuffer(u.encode("utf-8"), dtype=np.uint8))
        um = set(um_arr.tolist())
        novel = {m for m in set(am).difference(um) if m >= 256 and brain.mc.motor_idfs[m] >= brain.mc.tau_idf}
        slots = brain.lead_hip.query_leads(u_ents, u, top_k=1, prefer_subject=True)
        has_wiki_hit = False
        if slots:
            lm = set(brain.mc.encode(np.frombuffer(slots[0][3], dtype=np.uint8)).tolist())
            has_wiki_hit = bool(novel.intersection(lm))
        has_digit = any(ch.isdecimal() for ch in a)
        sufs = set()
        for k in range(2, min(9, len(am) + 1)):
            sb = b"".join(brain.mc.motor_bytes[m] for m in am[-k:])
            if 15 <= len(sb) <= 54:
                sufs.add(sb)
        if (not has_wiki_hit) and (not has_digit) and len(a) <= 68:
            rare_novel = [
                m for m in set(am).difference(um)
                if m >= 256 and ans_motor_df[m] < 35 and brain.mc.motor_idfs[m] >= brain.mc.tau_idf
            ]
            # 応答が質問外の新規辞書実体（人名・地名・組織名等）を含まないことも確認する。
            a_novel_ents = [
                x.decode("utf-8", errors="ignore")
                for x in brain.extract_verified_entities(a, True)[0]
                if x.decode("utf-8", errors="ignore") not in u
            ]
            is_zrare = (len(rare_novel) == 0 and len(a_novel_ents) == 0)
            vu = brain.mc.compute_root_thought_vec(um_arr)
            nv = float(np.linalg.norm(vu))
            if nv > 1e-6:
                vu = vu / nv
            n_ungr += 1
            primary = u_ents[0]
            for sb in sufs:
                ungr_topics[sb].add(primary)
                ungr_cnt[sb] += 1
                if is_zrare:
                    ungr_zrare[sb] += 1
                if len(ungr_vecs[sb]) < 60:
                    ungr_vecs[sb].append(vu)
            if is_zrare:
                zero_rare_turns.append((u, a, primary, am))
        elif has_wiki_hit or has_digit:
            n_gr += 1
            for sb in sufs:
                gr_cnt[sb] += 1

    scored = []
    for sb, topics in ungr_topics.items():
        nt = len(topics)
        if nt < 8:
            continue
        cu = ungr_cnt[sb]
        cg = gr_cnt[sb]
        lift_g = math.log(((cu + 0.5) / (n_ungr + 1.0)) / ((cg + 0.5) / (n_gr + 1.0)))
        zrate = ungr_zrare[sb] / max(1, cu)
        V = np.stack(ungr_vecs[sb], axis=0)
        mean_v = V.mean(axis=0)
        disp = float(1.0 - np.dot(mean_v, mean_v))
        if lift_g < 1.2 or disp < 0.60 or zrate < 0.80:
            continue
        sc = (disp ** 2) * (zrate ** 2) * lift_g * math.log1p(nt) * math.sqrt(len(sb) / 18.0)
        scored.append((sc, disp, zrate, lift_g, nt, sb.decode("utf-8", errors="ignore")))

    scored.sort(reverse=True)
    clauses = []
    clause_meta = []
    for sc, disp, zrate, lift_g, nt, txt in scored:
        if any(txt in c or c in txt for c in clauses):
            continue
        clauses.append(txt)
        clause_meta.append(dict(clause=txt, score=round(sc, 3), disp=round(disp, 3), zrate=round(zrate, 3), lift=round(lift_g, 2), topics=nt))
        if len(clauses) >= 6:
            break

    # 発見された純粋棄権句から、主題連結形（役割語始まりの句を最優先）と汎用単独形を自動構築する。
    abst_entity_suffixes = [c for c in clauses if c and c[0] in role_chars]
    ent_suf_topics = defaultdict(set)
    ent_suf_cnt = Counter()
    gen_path_topics = defaultdict(set)
    gen_path_cnt = Counter()
    for u, a, primary, am in zero_rare_turns:
        if not any(a.endswith(c) for c in clauses):
            continue
        if sum(a.count(t) for t in terminators) != 1:
            continue
        if a.startswith(primary):
            suf = a[len(primary):]
            if 8 <= len(suf) <= 28 and suf and suf[0] in role_chars:
                sm = brain.mc.encode(np.frombuffer(suf.encode("utf-8"), dtype=np.uint8)).tolist()
                if all(m < 256 or ans_motor_df[m] >= 35 for m in sm):
                    ent_suf_topics[suf].add(primary)
                    ent_suf_cnt[suf] += 1
        elif primary not in a and 8 <= len(a) <= 28:
            matched_c = next((c for c in clauses if a.endswith(c)), "")
            prefix_a = a[:-len(matched_c)] if matched_c else a
            if not brain.extract_verified_entities(prefix_a, True)[0]:
                if all(m < 256 or ans_motor_df[m] >= 35 for m in am):
                    gen_path_topics[a].add(primary)
                    gen_path_cnt[a] += 1

    # 質問から応答に転記された先頭の主題を差し引き、別々の主題で再現した接尾部を数える。
    for _, prompt, reply, _ in short_qa:
        if not any(reply.endswith(c) for c in clauses) or sum(reply.count(t) for t in terminators) != 1:
            continue
        for end in range(len(reply) - 1, 1, -1):
            anchor, suffix = reply[:end], reply[end:]
            if anchor in prompt and suffix[0] in role_chars and any(suffix.endswith(c) for c in clauses):
                ent_suf_topics[suffix].add(anchor)
                ent_suf_cnt[suffix] += 1
                break

    for suf, _ in sorted(ent_suf_cnt.items(), key=lambda kv: (len(ent_suf_topics[kv[0]]), kv[1]), reverse=True):
        if len(ent_suf_topics[suf]) >= 2 and suf not in abst_entity_suffixes:
            abst_entity_suffixes.append(suf)
        if len(abst_entity_suffixes) >= 2:
            break

    # 汎用単独文は、トピック固有語を含まない純粋棄権文を優先し、発見済みの純粋棄権句で補完する。
    abst_generic_paths = [
        gp for gp, _ in sorted(gen_path_cnt.items(), key=lambda kv: (len(gen_path_topics[kv[0]]), kv[1]), reverse=True)[:5]
    ]
    for c in clauses:
        if c and c[0] not in role_chars and len(c) >= 8 and c not in abst_generic_paths:
            abst_generic_paths.append(c)
        if len(abst_generic_paths) >= 5:
            break

    social_openers = [op for op, cnt in social_opener_cnt.most_common(12) if cnt >= 5]
    noun_linker = linker_cnt.most_common(1)[0][0] if linker_cnt else ""
    return {
        "abst_clauses": clauses,
        "abst_clause_meta": clause_meta,
        "abst_entity_suffixes": abst_entity_suffixes,
        "abst_generic_paths": abst_generic_paths,
        "social_openers": social_openers,
        "noun_linker": noun_linker,
    }


def main():
    """train の決定的な部分標本で回帰し、最後の200会話で閾値を選ぶ。"""
    import resource
    import torch
    from neurograph.wiki_lead_hippocampus_v112 import (EntityLookupContext, WikiLeadHippocampusV112,
        ContinuousThoughtCortexV50, RoleBoundThoughtCortexV60, InstructAdapterV80,
        PrefrontalReasoningGraphV90, BrocaSyntacticParserV100)
    started = time.perf_counter()
    brain = EntityLookupContext()
    brain.mc = ContinuousThoughtCortexV50.load("macro_v70")
    brain.rc = RoleBoundThoughtCortexV60(brain.mc,num_roles=64,seed=60)
    brain.rc.load_weights("macro_v70/role_hrr_v60.pkl")
    brain.adapter = InstructAdapterV80.load("instruct_v80",engram_dir=None)
    brain.rgraph = PrefrontalReasoningGraphV90.load("reasoning_v90",mc=brain.mc)
    brain.parser = BrocaSyntacticParserV100.load("broca_v100")
    brain.lead_hip = WikiLeadHippocampusV112()
    gate = ThalamicMemoryGateV120()
    gate.unigram, gate.outgoing, pairs, endings = Counter(), Counter(), Counter(), Counter()
    path = (SFT_DIR / "train.jsonl")
    with path.open() as stream:
        count = sum(1 for _ in stream)
    split = count - 200
    train, dev = [], []
    train_turns = []
    previous = None
    with path.open() as stream:
        for i,line in enumerate(stream):
            turns = json.loads(line)["turns"]
            prompt, answer = turns[0]
            am = brain.mc.encode(np.frombuffer(answer.encode(), dtype=np.uint8)).tolist()
            rows = train if i < split else dev
            if i == split:
                previous = None
            if i < split:
                for ti, (u_turn,reply) in enumerate(turns):
                    train_turns.append((ti == 0, u_turn, reply))
                    seq = brain.mc.encode(np.frombuffer(reply.encode(),dtype=np.uint8)).tolist()
                    pairs.update(zip(seq,seq[1:]))
                    gate.outgoing.update(seq[:-1])
                    gate.unigram.update(seq)
                    if reply.strip():
                        endings[reply.strip()[-1]] += 1
                if i % 5:
                    continue
            ents = [x.decode() for x in brain.extract_verified_entities(prompt, True)[0]]
            um = brain.mc.encode(np.frombuffer(prompt.encode(), dtype=np.uint8))
            qa = brain.adapter.predict_qa_intent(brain.rc, um)
            node, _ = brain.rgraph.route(brain.mc, brain.rc, brain.adapter, um, um)
            slots = brain.lead_hip.query_leads(ents, prompt, prefer_subject=True)
            novel = {m for m in set(am).difference(um) if m >= 256 and brain.mc.motor_idfs[m]>=brain.mc.tau_idf}
            for _,src,head,lead in slots:
                lm = brain.mc.encode(np.frombuffer(lead,dtype=np.uint8))
                x = gate.features(brain,prompt,ents,head,lead,src,qa,node.ans_concept_proto)
                y = int(bool(novel.intersection(lm)) and x[2]>0)
                rows.append((x,y))
                if i < split:
                    custom = x.copy()
                    custom[3], custom[-1] = 0., 1.
                    rows.append((custom,y))
            if previous:
                _,src,head,lead = previous
                rows.append((gate.features(brain,prompt,ents,head,lead,src,qa,node.ans_concept_proto),0))
            if i < split and slots and previous and ents:
                # 汎用語だけ一致する負例。元の質問と根拠にない別見出しを対に追加する。
                wrong = previous[2]
                _,src,head,lead = slots[0]
                if normalize(wrong) not in normalize(lead.decode()):
                    mixed = gate.features(brain,prompt+" "+wrong,ents+[wrong],head,lead,src,qa,node.ans_concept_proto)
                    rows.append((mixed,0))
            if slots:
                previous = slots[0]
    gate.pairs = {k:v for k,v in pairs.items() if v>=2}
    gate.total = sum(gate.unigram.values())
    gate.terminators = {k for k,v in endings.items() if v >= sum(endings.values())*.01 and unicodedata.category(k).startswith("P")}
    x,y = map(np.array,zip(*train))
    sample = np.where(y, .5/np.mean(y), .5/(1-np.mean(y)))
    gate.weights = np.linalg.solve(x.T@(x*sample[:,None])+np.eye(x.shape[1]), x.T@(sample*y))
    scores = np.array([gate.score_turn_gate(v) for v,_ in dev])
    labels = np.array([y for _,y in dev],dtype=bool)
    def balanced(t):
        """正負両方の再現率を均等に評価する。"""
        return ((scores[labels]>=t).mean() + (scores[~labels]<t).mean())/2
    gate.tau_turn = float(max(np.linspace(.05,.95,91),key=balanced))
    abst_info = _discover_abstention_memory(brain, train_turns, gate.terminators)
    gate.__dict__.update(abst_info)
    MODEL.parent.mkdir(exist_ok=True)
    with MODEL.open("wb") as stream:
        pickle.dump(gate.__dict__,stream)
    print(json.dumps(dict(train_sec=time.perf_counter()-started, train_pairs=len(y), positive=int(y.sum()),
                    dev_pairs=len(dev),dev_positive=int(labels.sum()),tau_turn=gate.tau_turn,
                    weights=dict(zip(FEATURES,gate.weights)), terminators=sorted(gate.terminators),
                    abst_clauses=gate.abst_clauses, abst_clause_meta=gate.abst_clause_meta,
                    abst_entity_suffixes=gate.abst_entity_suffixes, abst_generic_paths=gate.abst_generic_paths,
                    social_openers=gate.social_openers, noun_linker=gate.noun_linker,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20),ensure_ascii=False))


if __name__ == "__main__":
    main()

class ThalamicConsensusGateV124:
    @classmethod
    def load(cls):
        """三判事の学習済み状態を読み、旧ゲートと共用する CPU パネルを返す。"""
        from neurograph.panel_v124 import ConsensusPanel
        instance = cls()
        instance.panel = ConsensusPanel.load()
        return instance

    def decide(self, gate, brain, prompt, entities, rows, qa, proto):
        """最大三語義の票・語義選択・拒否理由と、会話に一致した学習応答を返す。"""
        panel = self.panel
        x = panel.describe(gate,brain,prompt,entities,rows,qa,proto)
        votes,spread,scores = panel.evaluate(x)
        question = normalize(prompt)
        reply = None
        # 質問の末尾に既知の会話ターンが丸ごとあり、取得見出しがその外にある場合。
        for start in range(len(question)):
            suffix = question[start:]
            if suffix in panel.social_replies and not any(normalize(e) in suffix for e in entities):
                reply = panel.social_replies[suffix]
                break
        trace = []
        for i,(head,lead,source,quality) in enumerate(rows):
            lex,sem,prag = map(float,votes[i])
            veto = []
            if reply:
                adjusted = np.array([lex,sem,0.])
                design = np.r_[1.,adjusted,-np.ptp(adjusted),adjusted.min()]
                scores[i] = np.clip(design @ panel.aggregate,0.,1.)
            if reply or prag < panel.prag_floor:
                veto.append("j_prag")
            if not x[i][2] or not any(f[1] for f in x):
                veto.append("j_lex")
            if scores[i] < panel.threshold:
                veto.append("consensus")
            trace.append(dict(head=head,source=source,sense=lead.decode()[:120],j_lex=lex,j_sem=sem,
                              j_prag=0. if reply else prag,prag_before_exact_match=prag,
                              disagreement=float(np.ptp([lex,sem,0. if reply else prag])),
                              consensus_score=float(scores[i]),score=float(scores[i]),active=False,
                              chosen_sense=False,vetoed_by=veto,cross_slot=float(x[i][12])))
        for head in dict.fromkeys(row[0] for row in rows):
            group = [i for i,row in enumerate(rows) if row[0]==head]
            eligible = [i for i in group if not trace[i]["vetoed_by"]]
            if eligible:
                best = max(eligible,key=lambda i:scores[i]*(1+x[i][12]))
                trace[best]["chosen_sense"] = True
                trace[best]["active"] = True
        return trace,reply
