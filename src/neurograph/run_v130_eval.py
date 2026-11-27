"""同じ統合脳で tools のみ切り替える。盲検は凍結検証後の一回に限定する。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import subprocess
import time

import torch

from neurograph.settings import SFT_DIR
from neurograph.run_v112_lead_qa import load_brain
from neurograph.tool_cortex_v130 import ToolUseArbiterV130, MODEL, INDEX, CORPUS
import sqlite3

ROOT=MODEL.parent
RESULT=ROOT/"results_v130_tools.json"


def read(path):
    """UTF-8の既存実験記録を復元する。"""
    return json.loads(path.read_text(encoding="utf8"))


def save(data):
    """全回答が書き終わってから結果を置換する。"""
    tmp=RESULT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    tmp.replace(RESULT)


def turn(brain,c,enabled,**kw):
    """回答、完全なトレース、正解語の一致数を保存する。"""
    a,valid,info=brain.respond(c["q"],use_tools=enabled,**kw)
    return dict(q=c["q"],answer=a,valid=valid,info=info,hits=sum(k in a for k in c.get("gold",[])),
                gold=c.get("gold",[]),criterion=c.get("criterion"),exact_abstention=bool(
                info["epistemic_abstain"] and a in info["agenda_passages"]))


def evaluate(blind):
    """31+16の回帰または排他的マーカー付きの新規10問を生成する。"""
    if blind:
        subprocess.run(["sha256sum","--quiet","-c",str(ROOT/"freeze_v130.sha256")],check=True)
        cases=read(ROOT/"blind_v130.json")
        prompts={c["q"] for c in cases}
        if len(cases)!=10 or len(prompts)!=10:
            raise ValueError("blind size")
        with (SFT_DIR / "train.jsonl").open() as f:
            if any(q in prompts for line in f for q,a in json.loads(line)["turns"]):
                raise ValueError("exact match")
        out=read(RESULT)
        with (ROOT/"blind_v130.started").open("x") as f:
            f.write(datetime.now().isoformat())
    brain,_=load_brain()
    allocated=lambda:torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
    before=allocated()
    cortex=ToolUseArbiterV130()
    deltas=[allocated()-before]
    original=cortex.decide

    def measured(*args):
        """割り当て総量カウンタの差には解放済みの一時テンソルも含む。"""
        start=allocated()
        result=original(*args)
        deltas.append(allocated()-start)
        return result

    cortex.decide=measured
    brain.tool_cortex=cortex
    try:
        if blind:
            out["blind"]={}
            out["exact_train_matches"]=0
            for enabled in (False,True):
                rows=[]
                out["blind"][str(enabled)]=rows
                for c in cases:
                    rows.append(turn(brain,c,enabled))
                    save(out)
        else:
            cases=[r for r in read(ROOT/"results_v123_temp.json")["rows"] if r["temperature"]==0]
            former=read(ROOT/"blind_v124.json")+read(ROOT/"blind_v125.json")
            facts="\n".join(cases[i]["info"]["agenda_passages"][0] for i in (6,7))
            out=dict(regression={},former={},temperature=[],divergences=[])
            for enabled in (False,True):
                rows=[]
                for i,c in enumerate(cases):
                    if i==6:
                        brain.lead_hip.mount_custom("astra",facts)
                    if i==8:
                        brain.lead_hip.unmount_custom("astra")
                    history=[(cases[5]["q"],rows[5]["answer"])] if i==13 else None
                    rows.append(turn(brain,c,enabled,history=history))
                out["regression"][str(enabled)]=rows
                out["former"][str(enabled)]=[turn(brain,c,enabled) for c in former]
                save(out)
            fixed=datetime.now().astimezone()
            cortex.registry.tools["clock"].now=lambda:fixed
            for c in [dict(q="128 * 37"),cases[2],dict(q="今、何時ですか?"),cases[11],cases[1]]:
                baseline=None
                for seed in (1,2,3):
                    row=turn(brain,c,True,temperature=.7,seed=seed)
                    out["temperature"].append(row)
                    current=[row["info"][k] for k in ("tool_trace","arbiter_trace","epistemic_abstain","entities")]
                    if baseline is not None and baseline!=current:
                        out["divergences"].append([c["q"],seed])
                    baseline=current
            timings=[]
            with sqlite3.connect(f"file:{INDEX}?mode=ro",uri=True) as db, CORPUS.open("rb") as f:
                for h, in db.execute("select head from bodies limit 1000"):
                    t=time.perf_counter_ns()
                    offset,size=db.execute("select pos,chars from bodies where head=?",(h,)).fetchone()
                    f.seek(offset)
                    f.read(size*4)
                    timings.append((time.perf_counter_ns()-t)/1e6)
            out["seek_ms"]={k:sorted(timings)[i] for k,i in (("median",500),("p95",950),("max",999))}
        out["blind_cuda_delta" if blind else "tool_cuda_delta_max"]=max(deltas)
        out["rss_kib"]=max(out.get("rss_kib",0),resource.getrusage(0).ru_maxrss)
        save(out)
    finally:
        brain.lead_hip.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--blind",action="store_true")
    evaluate(parser.parse_args().blind)

