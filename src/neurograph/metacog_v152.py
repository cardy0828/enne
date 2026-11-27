"""根拠候補を CPU 上で監査し、発話前に最大2回だけ再検討するメタ認知（v15.2）。"""
from dataclasses import dataclass
import json
import math
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path("memory_gate_v120")
MODEL = ROOT / "metacog_v152.pkl"


def grams(text):
    """例「東京駅」から隣接二文字の集合を求める。"""
    return {text[i:i+2] for i in range(len(text)-1) if text[i:i+2].isalnum()}


def geometry(q, evidence, df, n):
    """質問と本文について IDF 加重被覆率、最大一致、文字被覆を返す。"""
    bits = grams(q)
    common = bits & grams(evidence)
    weights = {b: math.log1p(n/(1+df.get(b,0))) for b in bits}
    return np.array([sum(weights[b] for b in sorted(common))/max(1,sum(weights[b] for b in sorted(bits))),
                     max((weights[b] for b in common),default=0)/math.log1p(n),
                     len(set(q)&set(evidence))/max(1,len(set(q)))])


def prior(store, mode):
    """最終世代の検証済み採点と、既知の失敗・訂正をモード別に集計する。"""
    hits = total = 0
    with store.log.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("generation_id") != store.current() or event.get("status")!="committed":
                continue
            for group,rows in event.get("after",{}).get("cases",{}).items():
                for row in rows:
                    key = row.get("tool") or ("ABSTAIN" if group=="traps" else "GROUNDED")
                    if key==mode:
                        hits += int(bool(row["ok"]))
                        total += 1
    with store.buffer.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"].startswith("tool:") and row["kind"].split(":")[1]==mode:
                total += 1
    return (hits+1)/(total+2)


@dataclass
class Plan:
    mode: str
    entities: list
    heads: list
    evidence: list
    sources: list
    judges: list
    tool: dict
    rescue_coverage: float | None = None


def calibrated_score(model,mode,score):
    """モード均衡重みで学習した Dev 校正曲線を線形補間する。"""
    slope, intercept = model["calibration"]
    return 1 / (1 + math.exp(-(slope * score + intercept)))


class MetacogMonitorV152:
    def __init__(self):
        """固定済み CPU ノードと校正曲線をロードする。"""
        with MODEL.open("rb") as stream:
            self.model = pickle.load(stream)

    def inspect(self, brain, q, plan):
        """例 GROUNDED の主題・残差・語義を検査し、根拠と改訂指示を返す。"""
        m = self.model
        evidence = "".join(plan.evidence)
        tool = (plan.tool or {}).get("called_tool") if (plan.tool or {}).get("tool_ok") else None
        key = tool or plan.mode
        store=brain.evolution
        signature=(store.root,brain.generation_id,store.log.stat().st_mtime_ns,store.buffer.stat().st_mtime_ns)
        if signature!=getattr(self,"prior_signature",None):
            self.prior_signature=signature
            self.priors={}
        if key not in self.priors:
            self.priors[key]=prior(store,key)
        capability=self.priors[key]
        flags, repair = [], {}
        anchors = [e for e in plan.entities if e and e in q]
        def specificity(word):
            """文字頻度と訓練質問頻度で主題候補を順位付けする。"""
            return sum(math.log1p(m["n"]/(1+m["df"].get(b,0))) for b in sorted(grams(word)) if not any(c.isdecimal() for c in b))
        # 数字主体の見出しだけが残った場合、最大幅を固定して別の完全一致見出しを検証する。
        numeric_only=anchors and all(any(ch.isdecimal() for ch in e) for e in anchors)
        if plan.mode=="ABSTAIN" or numeric_only and plan.mode=="GROUNDED":
            for start in range(len(q)):
                for width in range(min(16,len(q)-start),2,-1):
                    phrase=q[start:start+width]
                    if phrase.isalpha() and brain.lead_hip.has_exact_lead(phrase,min_quality=3):
                        if phrase not in anchors:
                            anchors.append(phrase)
                        break
        primary = max(anchors,key=specificity,default="")
        if plan.mode=="ABSTAIN":
            candidates=[a for a in anchors if 3<=len(a)<=16 and a.isalpha()
                        and brain.lead_hip.has_exact_lead(a,min_quality=3)]
            if candidates:
                primary=max(candidates,key=specificity)
                repair["rescue"]=primary
        lexical = geometry(q,evidence,m["df"],m["n"])
        trusted = any(s.startswith(("custom","learned")) for s in plan.sources)
        partial_coverage = False
        if (plan.mode=="GROUNDED" or tool=="deep_wiki") and not trusted:
            if primary and len(anchors)>1 and primary not in evidence:
                flags.append("subject_mismatch")
                repair["exclude_entities"]=[e for e in plan.entities if e!=primary]
                repair["primary"]=primary
            residue=q
            for e in sorted(anchors,key=len,reverse=True):
                residue=residue.replace(e," ")
            terms={b for b in grams(residue) if any(unicodedata.name(c,"").startswith("CJK UNIFIED") for c in b)
                   and m["df"].get(b,0)<=m["rare_df"]}
            unsupported=terms-grams(evidence)
            matched=terms & grams(evidence)
            if (plan.rescue_coverage is None or plan.rescue_coverage<.60) and len(unsupported)>=3 and len(unsupported)*2>len(terms) and not (
                    primary in plan.heads and len(matched)>=2 and any(ch.isdecimal() for ch in q)):

                flags.append("off_target_snippet" if tool=="deep_wiki" else "unanswered_query")
                if tool=="deep_wiki":
                    repair["exclude_snippets"]=plan.tool.get("snippets",[evidence])
            for i,(head, passage) in enumerate(zip(plan.heads,plan.evidence)):
                if not head or len(plan.heads)<2 and not terms:
                    continue
                context=residue+" "+" ".join(p for j,p in enumerate(plan.evidence) if i!=j)
                options=[j["sense"] for j in plan.judges if j.get("head")==head and not j.get("active")]
                def coherence(text):
                    """他の根拠・修飾語との IDF 加重一致を長さで正規化する。"""
                    bits=grams(text[:160].replace(head,""))
                    return sum(specificity(b) for b in sorted(bits & grams(context)))/math.sqrt(max(1,len(bits)))
                best=max(options,key=coherence,default=passage)
                if coherence(best)>coherence(passage)+2:
                    flags.append("sense_incoherent")
                    repair.update(exclude_senses=[passage],preferred=best,head=head)
                    break
            active = [j for j in plan.judges if j.get("active")]
            if (plan.mode == "GROUNDED" and "unanswered_query" in flags
                    and len(set(plan.heads)) >= 2 and anchors and matched
                    and all(a in evidence for a in anchors)
                    and set(plan.heads) == {j["head"] for j in active}
                    and flags == ["unanswered_query"]
                    and min(j["j_sem"] for j in active) > m["semantic_floor"]
                    and min(j["cross_slot"] for j in active) > m["cross_floor"]):
                flags.remove("unanswered_query")
                partial_coverage = True
        if tool=="calc":
            ranking=plan.tool.get("operators",[])
            expression=plan.tool["tool_args"].get("expression","")
            operator=next((op for op,_ in ranking if op in expression),None)
            signs=m["signs"].get(operator,{})
            atypical=signs and (1+signs.get(True,0))/(2+sum(signs.values()))<.05
            if ranking and float(plan.tool["tool_result"])<0 and (ranking[0][0]=="+" or atypical):
                flags.append("tool_anomaly")
            elif len(ranking)>1 and ranking[0][1]-ranking[1][1]<m["margin"] and not plan.tool.get("reconsidered"):
                flags.append("tool_low_margin")
        raw=float(lexical @ m["weights"][1:]+m["weights"][0])
        leaf=min(m["nodes"],key=lambda c:np.linalg.norm(lexical-c[0]))
        raw=(raw+leaf[1])/2
        active=[j for j in plan.judges if j.get("active")]
        disagreement=max((j["disagreement"] for j in active),default=0.)
        geometry_score=float(np.interp(raw,m["x"],m["y"]))
        if trusted:
            score=(2+lexical[1]+geometry_score)/4
        elif tool in ("clock","calc"):
            from neurograph.tool_cortex_v130 import norm
            attempt=max((a["score"] for a in plan.tool.get("attempts",[]) if a.get("ok")),default=0.)
            if tool=="clock":
                score=1-(1-attempt)**2
            elif re.fullmatch(r"[0-9.()+*/\s-]+",norm(q)):
                score=attempt
            else:
                ranking=plan.tool.get("operators",[])
                margin=max(0.,ranking[0][1]-ranking[1][1]) if len(ranking)>1 else 0.
                score=(attempt+margin/(margin+.25)+plan.tool.get("experience_support",0.))/3
        elif plan.mode=="SOCIAL":
            from neurograph.utterance_cortex_v131 import encode,overlap
            bits,vec=encode(brain.mc,q)
            node=max(brain.utterance_cortex.nodes,key=lambda n:n.score(bits,vec))
            best=max((overlap(bits,b)+.5*bool(b and b.issubset(bits)) for b,_ in node.examples),default=0.)
            score=(min(1.,best)+min(1.,max(0.,node.score(bits,vec))))/2 if best>=.28 else 0.
            repair.update(node_score=float(node.score(bits,vec)),example_match=float(best))
            if best<.28:
                flags.append("no_social_match")
        elif plan.mode=="ABSTAIN":
            score=capability/(2+len(anchors))
        else:
            votes=[(1-j["disagreement"]+j["min_vote"]+j["j_canon"])/3 for j in active]
            score=.65*geometry_score+.35*min(votes,default=0.)
            # 各実体が本文に現れ、意味票と正典票も支持するときのみ残りの不確実性を減らす。
            if anchors and active:
                coverage=sum(a in evidence for a in anchors)/len(anchors)
                consensus=min(j["j_sem"]*j["j_canon"] for j in active)
                score+=(1-score)*coverage*consensus
        if partial_coverage:
            score *= (1 + len(matched) / len(terms)) / 2
            repair["partial_coverage"] = len(matched) / len(terms)
        if plan.rescue_coverage is not None:
            # 独立した見出し完全一致と、訓練文字頻度で補正した修飾語の被覆。
            score=(score+plan.rescue_coverage+1)/3
        if any(f!="tool_low_margin" for f in flags):
            score*=.5
        return dict(mode=plan.mode,raw_confidence=round(score,6),confidence=round(calibrated_score(m,plan.mode,score),6),audit_flags=flags,self_prior=capability,
                    primary=primary,repair=repair,heads=plan.heads,evidence=plan.evidence,
                    entities=plan.entities,tool=plan.tool,judges=plan.judges,judge_disagreement=float(disagreement))


class MetacogControllerV152:
    max_revisions=2
    tau_high=.75

    def review(self,brain,q,plan,revise):
        """候補ごとの監査履歴を蓄積し、許可された回数だけ CPU 検索する。"""
        history=[]
        actions=[]
        while True:
            audit=brain.metacog_monitor.inspect(brain,q,plan)
            history.append(audit)
            flags=audit["audit_flags"]
            if plan.mode=="ABSTAIN" and audit["repair"].get("rescue") and len(actions)<self.max_revisions:
                updated=revise(plan,audit)
                if updated:
                    actions.append("REFOCUS")
                    plan=updated
                    continue
            if all(f in ("tool_low_margin", "no_social_match") for f in flags):
                actions.append("PASS")
                break
            action="REFOCUS" if "subject_mismatch" in flags else "RETRY_NEXT"
            updated=revise(plan,audit) if len(actions)<self.max_revisions else None
            if updated is None:
                if "subject_mismatch" in flags and len(actions)<self.max_revisions:
                    # 最初の実体集合をすべて棄却した場合にも完全一致索引から探索する。
                    abstention=Plan("ABSTAIN",plan.entities,[],[],[],[],{})
                    fallback=brain.metacog_monitor.inspect(brain,q,abstention)
                    if fallback["repair"].get("rescue"):
                        candidate=revise(abstention,fallback)
                        audit["fallback_audit"]=fallback
                        if candidate:
                            actions.append("REFOCUS")
                            plan=candidate
                            continue
                actions.append("DEMOTE_ABSTAIN")
                plan=Plan("ABSTAIN",plan.entities,[],[],[],[],{})
                history.append(brain.metacog_monitor.inspect(brain,q,plan))
                break
            actions.append(action)
            plan=updated
        first,last=history[0],history[-1]
        if len(history)>1 and plan.mode!="ABSTAIN" :
            # 改訂時には棄却された初期推定にも重みを残す。
            last["confidence"]=round((2*last["confidence"]+first["confidence"])/3,6)
        band="HIGH" if last["confidence"]>=self.tau_high else "LOW" if last["confidence"]<.45 else "MEDIUM"
        return plan,dict(enabled=True,initial_mode=first["mode"],initial_confidence=first["confidence"],
                         final_mode=plan.mode,final_confidence=last["confidence"],confidence_band=band,
                         audit_flags=sorted({f for h in history for f in h["audit_flags"]}),actions_taken=actions,
                         revisions=min(self.max_revisions,len(history)-1),self_prior=last["self_prior"],candidate_history=history)


def deliberate(brain,q,plan,qa,proto,carried,gap,tool_result,allow_leads=True,allow_body=True):
    """検索と演算の代替案を構成し、全候補を同じ監視器で再評価する。"""
    from neurograph.tool_cortex_v130 import Context,thought,op_features,norm,arithmetic
    from neurograph.operators_v132 import ranked_operators
    from neurograph.consensus_arbiter_v125 import attribute_gap
    from neurograph.memory_gate_v120 import normalize
    if not hasattr(brain,"metacog_monitor"):
        brain.metacog_monitor=MetacogMonitorV152()
    router=getattr(brain,"orthogonal_router",None)
    if plan.tool and plan.tool.get("tool_ok"):
        plan.tool=dict(plan.tool)
        if tool_result:
            plan.tool.update(tool_result.meta)
        if plan.tool["called_tool"]=="calc" and len(re.findall(r"\d+",norm(q)))==2 and any(c.isalpha() for c in q):
            text=norm(q)
            from neurograph.tool_cortex_v130 import kernel
            features=op_features(brain.mc,text)
            ranking,_=ranked_operators(router.model,text,features)
            expression=plan.tool["tool_args"]["expression"]
            support=[kernel(features,f) for kind,_,_,f in router.model.get("experience",{}).values()
                     if kind.startswith("tool:calc:") and kind[-1] in expression
                     and (("equal-unit",0) in features[0])==(("equal-unit",0) in f[0])]
            plan.tool["experience_support"]=max(support,default=0.)
            plan.tool["operators"]=ranking
    rejected=set()
    def revise(old,audit):
        """REFOCUS は主題を再検索し、RETRY_NEXT は却下済みの語義・文・演算を外す。"""
        repair=audit["repair"]
        if repair.get("rescue"):
            head=repair["rescue"]
            if not allow_leads or head in rejected:
                return None
            rejected.add(head)
            hits=brain.lead_hip.query_leads([head],q,2,True,True,True)
            rows=[(h,b,src,3.) for _,src,h,b in hits if h==head]
            if not rows:
                return None
            residue=q.replace(head," ")
            bits=grams(residue)
            model=brain.metacog_monitor.model
            rare={b for b in bits if model["df"].get(b,0)<=model["rare_df"]}
            def verified(candidate):
                """再検索結果に対し、残差被覆・意味票・同じ監視器の確信度を要求する。"""
                passage="".join(candidate.evidence)
                covered=len(rare & grams(passage))/max(1,len(rare))
                chars={c for c in residue if unicodedata.name(c,"").startswith("CJK UNIFIED")}
                weights={c:math.log1p(model["n"]/(1+model["char_df"].get(c,0))) for c in sorted(chars)}
                char_coverage=sum(w for c,w in weights.items() if c in passage)/max(1,sum(weights.values()))
                candidate.rescue_coverage=char_coverage
                review=brain.metacog_monitor.inspect(brain,q,candidate)
                audit.setdefault("rescue_attempts",[]).append(dict(source=candidate.sources,coverage=covered,
                    character_coverage=char_coverage,confidence=review["confidence"],flags=review["audit_flags"],
                    semantic=[j["j_sem"] for j in candidate.judges]))
                return ((not gap or gap in passage) and not review["audit_flags"] and review["raw_confidence"]>=2/3 and (covered>=.5 or char_coverage>=.60)
                        and any(j["j_sem"]>=.30 for j in candidate.judges))
            from neurograph.wiki_lead_hippocampus_v112 import _next_sentence_end_bytes
            unresolved = set()
            for row in rows:
                missing = attribute_gap(brain.memory_gate,brain,q,[head],[row[1]],False)
                span = row[1][:_next_sentence_end_bytes(row[1],brain.memory_gate.terminators)]
                span_gap = attribute_gap(brain.memory_gate,brain,q,[head],[span],False)
                unresolved.update(x for x in (gap, missing, span_gap) if x)
                if missing or span_gap or (gap and gap.encode() not in span):
                    continue
                judges,_,_=brain.arbiter_graph.decide(brain.memory_gate,brain,q,[head],[row],qa,proto,missing)
                candidate=Plan("GROUNDED",[head],[head],[row[1].decode()],[row[2]],judges,{})
                if verified(candidate):
                    return candidate
            if not allow_body or router is None:
                return None
            args=dict(heads=[head],attribute=gap or next(iter(sorted(unresolved)),None),residue=residue,exclude_snippets=[r[1].decode() for r in rows])
            ctx=Context(brain,q,[head],rows,"ABSTAIN",None,carried,router.model,thought(brain.mc,q),rows[0][1])
            result=router.registry.tools["deep_wiki"].execute(args,ctx)
            if (result.ok and all(x in result.text for x in unresolved)
                    and not attribute_gap(brain.memory_gate,brain,q,[head],[result.text.encode()],False)):
                body=[(head,result.text.encode(),"lead:wikipedia",3.)]
                judges,_,_=brain.arbiter_graph.decide(brain.memory_gate,brain,q,[head],body,qa,proto,None)
                trace=dict(called_tool="deep_wiki",category="memory_hop2",tool_ok=True,tool_args=args,
                           tool_result=result.text,tool_result_preview=result.text[:160],**result.meta)
                candidate=Plan("TOOL_CALL",[head],[head],[result.text],["deep_wiki"],judges,trace)
                if verified(candidate):
                    return candidate
            return None
        if repair.get("exclude_entities") or repair.get("exclude_senses"):
            ents=[e for e in old.entities if e not in repair.get("exclude_entities",())]
            if repair.get("primary") and repair["primary"] not in ents:
                ents.append(repair["primary"])
            rejected.update(repair.get("exclude_senses",()))
            hits=brain.lead_hip.query_leads(ents,q,2,True,True,True,
                    exclude_entities=repair.get("exclude_entities",()),exclude_senses=rejected)
            rows=[]
            for _,src,h,b in hits:
                passage=b.decode()
                if h==repair.get("head") and not passage.startswith(repair["preferred"]):
                    continue
                quality=brain.lead_hip.conn.execute("SELECT quality FROM senses WHERE head_norm=? AND lead=?",(normalize(h),passage)).fetchone()
                rows.append((h,b,src,float(quality[0]) if quality else 0.))
            missing=attribute_gap(brain.memory_gate,brain,q,ents,[r[1] for r in rows],False)
            judges,_,decision=brain.arbiter_graph.decide(brain.memory_gate,brain,q,ents,rows,qa,proto,missing)
            accepted=[r for r,j in zip(rows,judges) if j["active"]]
            if not accepted:
                return None
            return Plan("GROUNDED",ents,[r[0] for r in accepted],[r[1].decode() for r in accepted],
                        [r[2] for r in accepted],judges,{})
        if repair.get("exclude_snippets"):
            rejected.update(repair["exclude_snippets"])
            args=dict(old.tool["tool_args"])
            residue=q
            for ent in old.entities:
                residue=residue.replace(ent," ")
            rows=[(j["head"],j["sense"].encode(),j["source"],0.) for j in old.judges]
            ctx=Context(brain,q,old.entities,rows,"ABSTAIN" if gap else "GROUNDED",gap,carried,router.model,thought(brain.mc,q),b" ".join(r[1] for r in rows))
            args.update(exclude_snippets=sorted(rejected),residue=residue)
            result=router.registry.tools["deep_wiki"].execute(args,ctx)
            if result.ok:
                trace=dict(old.tool,tool_args=args,tool_result=result.text,tool_result_preview=result.text[:160],**result.meta)
                return Plan(old.mode,old.entities,old.heads,[result.text],old.sources,old.judges,trace)
        if any(flag.startswith("tool_") for flag in audit["audit_flags"]):
            ranking=old.tool["operators"]
            if not old.tool.get("reconsidered") and len(ranking)>1:
                numbers=re.findall(r"\d+",norm(q))
                current=old.tool["tool_args"]["expression"]
                alternative=next((op for op,_ in ranking[:2] if op not in current),None)
                if alternative and float(old.tool["tool_result"])<0:
                    expression=alternative.join(numbers)
                    try:
                        value=arithmetic(expression)
                    except (ValueError,ArithmeticError):
                        return None
                    if float(value)<0:
                        return None
                    trace=dict(old.tool,tool_result=value,tool_result_preview=value,tool_args={"expression":expression},reconsidered=True)
                    return Plan(old.mode,old.entities,old.heads,[value],old.sources,old.judges,trace)
        return None
    return MetacogControllerV152().review(brain,q,plan,revise)
