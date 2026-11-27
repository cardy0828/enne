"""固定回帰と経験による改善を、実際の発話とツール観測で判定する（v14.0）。"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re

from neurograph.settings import GPU, STRICT_DEVICE

ROOT = Path("memory_gate_v120")


def check_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA を使える GPU が必要です")
    if GPU is not None and os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU:
        raise RuntimeError("CUDA_VISIBLE_DEVICES に ENNE_GPU_UUID と同じ UUID を指定してください")
    if STRICT_DEVICE and (os.environ.get("PYTORCH_CUDA_ALLOC_CONF")!="expandable_segments:True" or
        torch.cuda.device_count()!=1 or "RTX 3060" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("RTX 3060 の UUID とアロケータを固定してください")


def read(name):
    """固定資料をロードし、作成時のハッシュを照合する。"""
    path = ROOT/name
    manifest = json.loads((ROOT/"guard_v140.sha256.json").read_bytes())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest[name]:
        raise ValueError("固定ゲート資産が変更されています")
    return json.loads(data)


def probe(brain, row, **kwargs):
    """質問・期待値から本番応答を採点する。時計は同じ呼出の観測と照合する。"""
    q,k,t = row["prompt"],row["kind"],row["target"]
    a,valid,info = brain.respond(q,score_critic=False,**kwargs)
    tool = info["tool_trace"] or {}
    if k=="trap":
        ok = info["epistemic_abstain"] and a in info["agenda_passages"] and not tool.get("tool_ok")
    elif k.startswith("tool:"):
        name = k.split(":")[1]
        value = tool.get("tool_result","")
        ok = tool.get("called_tool")==name and tool.get("tool_ok") and value and value in a
        if name=="calc":
            ok = ok and value==row.get("value",t) and t in a
            if k.count(":")==2:
                ok = ok and k[-1] in tool["tool_args"]["expression"]
    elif k=="gold":
        ok = sum(word in a for word in t)
    else:
        ok = t in a and not info["epistemic_abstain"]
    return dict(prompt=q,answer=a,ok=int(bool(ok)) if k!="gold" else ok,
                valid=valid,eot=info["stopped_eot"],sources=info["agenda_sources"],
                tool=tool.get("called_tool"),value=tool.get("tool_result"))


def evaluate(brain, probes):
    """親と候補で同一の Core・Custom・既知罠・算術を実行する。"""
    core = [r for r in read("results_v123_temp.json")["rows"] if r["temperature"]==0]
    blind = read("blind_v132.json")
    cases = {}
    for group, subset in (("core",core[:6]),("custom",core[6:8])):
        if group=="custom":
            brain.lead_hip.mount_custom("astra","\n".join(r["info"]["agenda_passages"][0] for r in subset))
        try:
            cases[group] = [probe(brain,dict(prompt=r["q"],target=r["gold"],kind="gold")) for r in subset]
        finally:
            if group=="custom":
                brain.lead_hip.unmount_custom("astra")
    traps = [r["q"] for r in core if r["info"].get("epistemic_abstain") and r["q"].startswith("架空")]
    for v in (124,125,130,131,132):
        for r in read(f"blind_v{v}.json"):
            if "架空" in r.get("type","") or "未来" in r.get("type","") or (r.get("trap") and ("架空" in r["q"] or "未来" in r["criterion"])):
                traps.append(r["q"])
    cases["traps"] = [probe(brain,dict(prompt=q,target="",kind="trap")) for q in dict.fromkeys(traps)]
    cases["arithmetic"] = [probe(brain,dict(prompt=r["q"],kind="tool:calc",value=v,target=t))
        for r,v,t in zip(blind[:4],("11","43","29","28"),("11","43枚","29匹","28枚"))]
    cases["progress"] = [probe(brain,row) for row in probes]
    return dict(cases=cases,scores={k:sum(r["ok"] for r in rows) for k,rows in cases.items()})


def failures(before, after):
    """各項目の後退、絶対下限、発話完結と正味進歩を検査する。"""
    reasons = []
    for k,rows in after["cases"].items():
        for prior,row in zip(before["cases"][k], rows):
            if not row["valid"] or not row["eot"] or row["ok"]<prior["ok"]:
                reasons.append(k+" "+row["prompt"])
            elif k in ("traps","arithmetic") and not row["ok"]:
                reasons.append(k+" "+row["prompt"])
    for k,minimum in (("core",12),("custom",6)):
        if after["scores"][k]<minimum:
            reasons.append(k)
    if after["scores"]["progress"]<=before["scores"]["progress"]:
        reasons.append("検証済みの正味改善なし")
    return reasons
