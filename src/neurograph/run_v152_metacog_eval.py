"""メタ認知の有無だけを切り替えて評価する。新規問題は凍結検証の後に1回だけ実行する（v15.2）。"""
from datetime import datetime
import json
import os
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import torch
from neurograph.settings import SFT_DIR
from neurograph.gate_v140 import check_device
from neurograph.run_v130_eval import ROOT,read,load_brain,turn
import neurograph.metacog_v152 as metacog

OUT=ROOT/"results_v152_metacog.json"
TARGETS=[
    ("1964年に開業した東海道新幹線は、どの駅からどの駅までを結ぶ路線ですか？",["東京駅","新大阪駅"]),
    ("名古屋城で来月最初の日曜日に訪れる人の正確な人数は？",[]),
    ("琵琶湖から自然に流れ出す川の名前を教えて。",["瀬田川"]),
    ("動物のヤマネコとリンクスについて、どんな仲間なのか知りたいです。",["ネコ科"]),
    ("昨日5キロメートル走り、今日も7キロメートル走りました。合計で何キロメートル走りましたか？",["12キロメートル"])]


def save(out):
    """実行結果を UTF-8 で原子的に保存する。"""
    tmp=OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    tmp.replace(OUT)


def freeze():
    """全コード・学習済み資産・検索 DB と再現用の入力ファイルを列挙して凍結する。"""
    manifest=ROOT/"freeze_v152.sha256"
    paths={Path(line[66:]) for line in (ROOT/"freeze_v151.sha256").read_text().splitlines()}
    for extension in ("*.py","*.pt","*.pkl","*.sqlite","*.safetensors","*.npz","*.bin","*.sh","*.pth","*.db","*.ckpt","*.go"):
        paths.update(Path(".").rglob(extension))
    paths.update([ROOT/"fit_judges_v152.json",ROOT/"blind_v151.json",ROOT/"training_v152.json",ROOT/"dev_answers_v151.json",ROOT/"blind_v150.json",
                  ROOT/"reference_v150_gen002.json",ROOT/"results_v123_temp.json"])
    hashes=subprocess.check_output(["sha256sum",*sorted(str(p) for p in paths)])
    with manifest.open("xb") as stream:
        stream.write(hashes)


def calibration(rows):
    """実際の最終信頼度・採点の組から、空帯を保持して精度と校正誤差を返す。"""
    import numpy as np
    scores = np.array([r[0] for r in rows])
    labels = np.array([int(r[1]) for r in rows])
    masks = {"HIGH":scores >= .75, "MEDIUM":(.45 <= scores) & (scores < .75),
             "LOW":scores < .45, "NON-HIGH":scores < .75}
    bands = {k:dict(correct=int(labels[m].sum()), total=int(m.sum()),
                    accuracy=float(labels[m].mean()) if m.any() else None) for k,m in masks.items()}
    bins = np.minimum(9,(scores*10).astype(int))
    positive, negative = scores[labels==1], scores[labels==0]
    ece = sum(np.mean(bins==i)*abs((scores-labels)[bins==i].mean()) for i in set(bins))
    return dict(bands=bands,ECE=float(ece),Brier=float(np.mean((scores-labels)**2)),
                AUROC=float(np.mean((positive[:,None]>negative)+.5*(positive[:,None]==negative)))
                if len(positive) and len(negative) else None)


def evaluate(phase):
    """dev は対照・既知例、blind は凍結検証後、sleep は隔離世代で実行する。"""
    check_device()
    subprocess.run(["sha256sum","--quiet","-c",str(ROOT/"live_before_v150.sha256")],check=True)
    out=read(OUT) if OUT.exists() else {}
    if phase=="blind":
        subprocess.run(["sha256sum","--quiet","-c",str(ROOT/"freeze_v152.sha256")],check=True)
        cases=read(ROOT/"blind_v152.json")
        qs={r["q"] for r in cases}
        assert len(qs)==len(cases)==10
        assert not qs.intersection(q for q,_ in TARGETS)
        assert not qs.intersection(r["q"] for r in read(ROOT/"reference_v150_gen002.json"))
        assert not qs.intersection(r["q"] for r in read(ROOT/"dev_answers_v151.json"))
        for path in ROOT.glob("blind_v*.json"):
            if path.name!="blind_v152.json":
                assert not qs.intersection(r["q"] for r in read(path))
        with (SFT_DIR / "train.jsonl").open() as stream:
            assert not any(t in qs for line in stream for pair in json.loads(line)["turns"] for t in pair)
        with (ROOT/"blind_v152.started").open("x") as f:
            f.write(json.dumps(dict(started=datetime.now().isoformat(),sha256=hashlib.sha256((ROOT/"blind_v152.json").read_bytes()).hexdigest())))
        out["exact_overlap"]=dict(train=0,prior_blind=0,regression=0,known=0,calibration=0)
    with tempfile.TemporaryDirectory() as dest:
        isolated=Path(dest)/"evolution"
        shutil.copytree("evolution_v140",isolated)
        os.environ["NEUROGRAPH_EVOLUTION_DIR"]=str(isolated)
        brain,_=load_brain()
        assert brain.generation_id=="gen_002"
        deltas=[]
        original=metacog.deliberate
        def measured(*args):
            """監視器ロード・監査・改訂全体で累積 CUDA 割当差を測る。"""
            before=torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
            value=original(*args)
            deltas.append(torch.cuda.memory_stats()["allocated_bytes.all.allocated"]-before)
            return value
        metacog.deliberate=measured
        def ask(case,enabled,**kw):
            """一問を実行し全文・完全な情報・実経過時間を保存する。"""
            start=time.perf_counter()
            row=turn(brain,case,True,use_metacog=enabled,score_critic=False,**kw)
            row["seconds"]=time.perf_counter()-start
            return row
        try:
            if phase in ("dev","blind"):
                regression=[r for r in read(ROOT/"results_v123_temp.json")["rows"] if r["temperature"]==0]
                suites={"blind":cases} if phase=="blind" else {"targeted":[dict(q=q,gold=g) for q,g in TARGETS],"retrospective":read(ROOT/"blind_v150.json"),"prior151":read(ROOT/"blind_v151.json"),"regression":regression}
                for label,data in suites.items():
                    out[label]={}
                    for enabled in (False,True):
                        rows=[]
                        for i,c in enumerate(data):
                            history=None
                            if label=="regression":
                                if i==6:
                                    brain.lead_hip.mount_custom("astra","\n".join(r["info"]["agenda_passages"][0] for r in data[6:8]))
                                if i==8:
                                    brain.lead_hip.unmount_custom("astra")
                                if i==13:
                                    history=[(data[5]["q"],rows[5]["answer"])]
                            rows.append(ask(c,enabled,history=history))
                        out[label]["ON" if enabled else "OFF"]=rows
                        save(out)
                if phase=="dev":
                    out["reference_matches"]=sum(r["answer"]==s["answer"] for r,s in zip(read(ROOT/"reference_v150_gen002.json"),out["regression"]["OFF"]))
                    assert out["reference_matches"]==31
                    out["temperature"]=[]
                    fixed=datetime.now().astimezone()
                    brain.orthogonal_router.registry.tools["clock"].now=lambda:fixed
                    for q in [TARGETS[0][0],TARGETS[1][0],regression[23]["q"],"128 * 37","今、何時ですか?"]:
                        for seed in (1,2,3):
                            out["temperature"].append(ask(dict(q=q),True,temperature=.7,seed=seed))
            elif phase=="dev_validation":
                rows=[]
                for row in read(ROOT/"dev_answers_v151.json"):
                    if row["group"]=="test":
                        actual=ask(dict(q=row["q"]),True)
                        rows.append(dict(q=row["q"],answer=actual["answer"],
                                         reference=row["reference"],trace=actual["info"]["metacog_trace"],
                                         same_answer=row["answer"]==actual["answer"]))
                out["dev_validation"]=rows
            elif phase=="fit_features":
                # Dev 校正集合の裁定票のみを取得する。発話の採点は既存の同一発話ラベルを使う。
                samples=[]
                for row in read(ROOT/"dev_answers_v151.json"):
                    if row["group"]=="fit":
                        actual=ask(dict(q=row["q"]),False)
                        samples.extend(actual["info"].get("gate_decisions") or [])
                (ROOT/"fit_judges_v152.json").write_text(json.dumps(samples,ensure_ascii=False))
            elif phase=="sleep":
                for q,_ in TARGETS:
                    ask(dict(q=q),True,record_episode=True)
                rows=[json.loads(line) for line in brain.evolution.buffer.read_text().splitlines()]
                recovered=[r for r in rows if r["kind"]=="metacog_recovery"]
                from neurograph.gate_v140 import evaluate as fixed_gate
                out["fixed_gate"]=fixed_gate(brain,[])
                sticker=read(ROOT/"blind_v132.json")[3]
                out["sticker"]=[ask(sticker,on) for on in (False,True)]
                out["isolated_sleep"]=dict(records=recovered,result=brain.sleep_evolve())
            if phase=="dev":
                temp=out["temperature"]
                out["temperature_summary"]=dict(
                    divergent_pairs=sum(temp[i]["info"]["metacog_trace"]!=temp[i-i%3]["info"]["metacog_trace"]
                                        for i in range(len(temp)) if i%3),
                    pairs=10, valid_eot=sum(r["valid"] and r["info"]["stopped_eot"] for r in temp))
            out[phase+"_resources"]=dict(calls=len(deltas),cuda_extra_bytes=max(deltas,default=None),
                rss_kib=resource.getrusage(0).ru_maxrss,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            save(out)
        finally:
            metacog.deliberate=original
            brain.lead_hip.close()
    subprocess.run(["sha256sum","--quiet","-c",str(ROOT/"live_before_v150.sha256")],check=True)


if __name__=="__main__":
    if sys.argv[1]=="freeze":
        freeze()
    else:
        evaluate(sys.argv[1])
