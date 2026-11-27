"""外部行動を独立採択し、記憶の二段検索を別に実行する CPU ルータ。"""
from neurograph.operators_v132 import classify
import pickle
import re
from neurograph.tool_cortex_v130 import (Calc, Clock, DeepWiki, ToolRegistry, Context,
    MODEL, thought, norm, grams, similarity, op_features, kernel)


class ActionCalc(Calc):
    category = "action"

    def propose(self,c):
        """式は構文で受け、文章では二つの数量と学習近傍を要求する。"""
        q = norm(c.q)
        if re.fullmatch(r"[0-9.()+*/\s-]+",q):
            return super().propose(c)
        quantities = re.findall(r"(\d+)([^\W\d_])",q)
        if len(quantities)!=2 or len(re.findall(r"\d+",q))!=2:
            return 0.,{}
        if any(len(n)>3 for n,_ in quantities):
            return 0.,{}
        features = op_features(c.brain.mc,q)
        op,score = classify(c.model,q,features)
        # 同一単位の加算経験を異次元の積へ転移しない。旧モデルは局所領域外で保持する。
        same_unit = quantities[0][1]==quantities[1][1]
        corrections = [(kernel(features,f),k[-1]) for k,_,_,f in c.model.get("experience",{}).values()
                       if k.startswith("tool:calc:") and (("equal-unit",0) in f[0])==same_unit]
        if corrections:
            support,label = max(corrections)
            if support >= .50:
                closed_form,_ = classify(c.model["experience_operator"],q,features)
                op = label if support>=.75 or label==closed_form else closed_form
                score = max(score,support)
        if score < c.model["calc_floor"]:
            return 0.,{}
        return score,dict(expression=op.join(n for n,_ in quantities))


class ActionClock(Clock):
    category = "action"

    def propose(self,c):
        """訓練質問との類似度と低頻度アンカーで現在時刻要求を認識する。"""
        near = similarity(c.q,c.vec,c.model["clock"])
        corrected = [(g,v,k) for k,g,v,_ in c.model.get("experience",{}).values() if k=="tool:clock"]
        if corrected and not re.search(r"\d",c.q):
            support = similarity(c.q,c.vec,corrected)[0][0]
            if support >= .30 and any(len(grams(c.q)&g)>=2 for g,_,_ in corrected):
                return support,{}
        anchored = bool(grams(c.q) & c.model["clock_anchors"])
        if near and not re.search(r"\d",c.q) and near[0][0]>(.28 if anchored else .62):
            return near[0][0],{}
        return 0.,{}


class MemoryWiki(DeepWiki):
    category = "memory_hop2"

    def propose(self,c):
        """百科主題を検証できた比較・欠落属性にだけ二段検索を許す。"""
        verified = {r[0] for r in c.rows if r[0] in c.q or r[0] in c.carried}
        if not verified or (c.mode=="ABSTAIN" and (not c.gap or c.gap not in c.rows_text.decode())):
            return 0.,{}
        return super().propose(c)


class OrthogonalRouter:
    def __init__(self):
        """独立行動のモデルとカテゴリ付きプラグインを CPU に復元する。"""
        with MODEL.open("rb") as stream:
            self.model = pickle.load(stream)
        with MODEL.with_name("action_nodes_v132.pkl").open("rb") as stream:
            self.model.update(pickle.load(stream))
        self.registry = ToolRegistry()
        for tool in (ActionCalc(),ActionClock(),MemoryWiki()):
            self.registry.register(tool)

    def select(self,category,brain,q,ents,rows,mode,gap,carried,guard):
        """同じカテゴリの適格候補を順に実行し、成功時点で観測を返す。"""
        trace=dict(category=category,guard=guard,attempts=[],called_tool=None,tool_args={},
                   tool_ok=None,tool_result="",tool_result_preview="",fallback=None)
        if guard:
            return trace,None
        ctx=Context(brain,q,ents,rows,mode,gap,carried,self.model,thought(brain.mc,q),b" ".join(r[1] for r in rows))
        eligible=[]
        for name,tool in self.registry.tools.items():
            if tool.category==category:
                score,args=tool.propose(ctx)
                if score>0:
                    eligible.append((score,name,args))
        for score,name,args in sorted(eligible,key=lambda item:(-item[0],item[1])):
            result=self.registry.tools[name].execute(args,ctx)
            trace["attempts"].append(dict(name=name,score=score,args=args,ok=result.ok,result=result.text))
            trace.update(called_tool=name,tool_args=args,tool_ok=result.ok,tool_result=result.text,tool_result_preview=result.text[:160])
            if result.ok:
                return trace,result
        trace["fallback"]=mode
        return trace,None




