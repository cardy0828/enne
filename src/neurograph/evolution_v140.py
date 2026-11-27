"""経験を記録し、固定ゲートを通った CPU スナップショットだけを新しい世代として公開する（v14.0）。"""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sqlite3
import tempfile
import unicodedata
import uuid

from neurograph.tool_cortex_v130 import arithmetic, grams, norm, thought, op_features
from neurograph.utterance_cortex_v131 import SocialUtteranceNode, encode

FILES = ("learned_memory.sqlite", "evolved_axis1.pkl", "evolved_utterance.pkl")


def digest(path):
    """指定ファイルの内容を SHA-256 で識別する。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    """同一ディレクトリの一時ファイルを同期してから置換する。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def append(path, row):
    """ロックを保持する呼出元から JSON 一行を永続化する。"""
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def clean(s):
    """空文字、制御文字、過大な記録を拒否する。"""
    if not isinstance(s, str) or not s.strip() or len(s) > 2048 or any(
            unicodedata.category(ch) in ("Cc", "Cs") for ch in s):
        raise ValueError("空欄・制御文字・2048文字を超える入力は登録できません")
    return s.strip()


def sealed(folder, gen, parent, offset, probes=()):
    """世代の全 CPU 資産と消費済みバッファ位置を固定する。"""
    meta = dict(generation_id=gen, parent=parent, offset=offset, probes=probes,
                sha256={p:digest(folder/p) for p in FILES})
    atomic(folder/"meta.json", json.dumps(meta).encode())
    return meta


class SleepConsolidatorV140:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("NEUROGRAPH_EVOLUTION_DIR", "evolution_v140"))
        self.gens = self.root/"generations"
        self.gens.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root/"active_gen.json"
        self.log = self.root/"ledger.jsonl"
        self.buffer = self.root/"episodes.jsonl"
        with self.lock():
            if not self.pointer.exists():
                folder = self.gens/"gen_000"
                if not folder.exists():
                    tmp = Path(tempfile.mkdtemp(dir=self.gens))
                    with sqlite3.connect(tmp/FILES[0]) as db:
                        db.execute("CREATE TABLE facts(topic TEXT PRIMARY KEY, fact TEXT NOT NULL, source TEXT NOT NULL)")
                    for name in FILES[1:]:
                        atomic(tmp/name, pickle.dumps({}))
                    sealed(tmp, folder.name, None, 0)
                    tmp.rename(folder)
                self.activate(folder.name)
            self.buffer.touch(exist_ok=True)
            self.log.touch(exist_ok=True)

    @contextlib.contextmanager
    def lock(self):
        """複数の Linux CLI 間で睡眠・追記・世代移動を直列化する。"""
        import fcntl
        with (self.root/".lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def current(self):
        """アトミックな世代ポインタを読み取る。"""
        return json.loads(self.pointer.read_bytes())["generation_id"]

    def verify(self, gen):
        """ロード前にパスと全チェックサムを照合する。"""
        if not re.fullmatch(r"gen_[0-9]{3,}", gen):
            raise ValueError("世代名が不正です")
        folder = self.gens/gen
        meta = json.loads((folder/"meta.json").read_bytes())
        if meta["generation_id"] != gen or meta["sha256"] != {p:digest(folder/p) for p in FILES}:
            raise ValueError("世代の整合性検査に失敗")
        return meta

    def activate(self, gen):
        """例 gen_001 の検証後に参照先を切り替える。"""
        self.verify(gen)
        atomic(self.pointer, json.dumps(dict(generation_id=gen)).encode())

    def mount(self, brain, folder):
        """検証済み SQLite とノードを CPU 上で復元する。"""
        from neurograph.action_axis_v132 import OrthogonalRouter
        from neurograph.utterance_cortex_v131 import UtteranceCortex
        with sqlite3.connect(f"file:{folder/FILES[0]}?immutable=1", uri=True) as db:
            facts = db.execute("SELECT topic,fact,source FROM facts ORDER BY length(topic) DESC,topic").fetchall()
        axis = pickle.loads((folder/FILES[1]).read_bytes())
        social = pickle.loads((folder/FILES[2]).read_bytes())
        router = OrthogonalRouter()
        router.model["experience"] = axis
        calc = [(q,features,k[-1]) for q,(k,_,_,features) in axis.items() if k.startswith("tool:calc:")]
        if calc:
            from neurograph.operators_v132 import train_operator
            from neurograph.tool_cortex_v130 import MODEL
            base = json.loads(MODEL.with_name("training_v132.json").read_bytes())["cv"]
            original = [(r["q"],f,r["expected"]) for r,f in zip(base,router.model["features"],strict=True)]
            router.model["experience_operator"] = train_operator(original+calc)
        cortex = UtteranceCortex()
        cortex.nodes.extend(social.values())
        brain.orthogonal_router, brain.utterance_cortex = router, cortex
        brain.lead_hip.learned_facts = facts
        brain.generation_id = json.loads((folder/"meta.json").read_bytes())["generation_id"]

    def record(self, kind, prompt, target="", **extra):
        """例 tool:calc:+ の入力を次回睡眠のために追記する。"""
        with self.lock():
            append(self.buffer, dict(kind=kind, prompt=clean(prompt), target=target, **extra))

    def correction(self, prompt, kind, target):
        """訂正ラベルと対象を検証し、学習前の監査記録にする。"""
        kind, _, suffix = kind.partition(":") if kind.startswith(("social:", "fact:")) else (kind,"","")
        if kind not in ("fact", "social", "tool:clock", *("tool:calc:"+op for op in "+-*/")):
            raise ValueError("未知の訂正種別")
        if kind in ("fact","social"):
            target = clean(target or suffix)
        self.record(kind, prompt, target)

    def rollback(self, brain, gen):
        """公開済み世代だけを手動復元する。"""
        with self.lock():
            old = self.current()
            self.verify(gen)
            self.mount(brain, self.gens/gen)
            self.activate(gen)
            append(self.log, dict(status="manual_rollback", parent=old, generation_id=gen))

    def sleep(self, brain, run_curiosity=True, guard_fn=None):
        """親世代と隔離した候補に全固定検査と追加検査を適用する。"""
        from neurograph.gate_v140 import evaluate, failures, check_device
        from neurograph.curiosity_v140 import explore
        check_device()
        with self.lock():
            parent = self.current()
            meta = self.verify(parent)
            number = max(int(p.name[4:]) for p in self.gens.glob("gen_*"))+1
            gen = f"gen_{number:03d}"
            cycle = dict(parent=parent, generation_id=gen, cycle=uuid.uuid4().hex)
            folder = Path(tempfile.mkdtemp(dir=self.root))
            try:
                shutil.copytree(self.gens/parent, folder, dirs_exist_ok=True)
                append(self.log, dict(cycle, status="proposed"))
                with self.buffer.open("rb") as f:
                    offset = meta["offset"]
                    with self.log.open(encoding="utf8") as history:
                        for line in history:
                            event = json.loads(line)
                            if event.get("parent")==parent and event["status"]=="rolled_back":
                                offset = max(offset,event["offset"])
                    f.seek(offset)
                    rows = [json.loads(line) for line in f]
                    offset = f.tell()
                shadow = copy.copy(brain)
                shadow.lead_hip = copy.copy(brain.lead_hip)
                shadow.lead_hip.custom_facts = []
                self.mount(shadow, self.gens/parent)
                discovery = {}
                if run_curiosity:
                    from neurograph.memory_gate_v120 import ThalamicMemoryGateV120
                    from neurograph.consensus_arbiter_v125 import ConsensusArbiterGraphV125
                    if shadow.memory_gate is None:
                        shadow.memory_gate = ThalamicMemoryGateV120.load()
                    if not hasattr(shadow,"arbiter_graph"):
                        shadow.arbiter_graph = ConsensusArbiterGraphV125.load()
                    extra, discovery = explore(shadow, rows)
                    rows.extend(extra)
                from neurograph.gate_v140 import probe
                useful = []
                redundant = 0
                for row in rows:
                    if row.get("origin")=="oracle" and probe(shadow,row)["ok"]:
                        redundant += 1
                    else:
                        useful.append(row)
                probes, counts = self.consolidate(folder, shadow, useful, gen)
                counts["oracle_skipped_correct"] = redundant
                retained = meta.get("probes",[])
                sealed(folder, gen, parent, offset, retained+probes)
                before = evaluate(shadow, probes)
                held = discovery.get("held_out",[])
                before["retained"] = [probe(shadow,r) for r in retained]
                before["oracle_holdout"] = [probe(shadow,r) for r in held]
                self.mount(shadow, folder)
                after = evaluate(shadow, probes)
                after["oracle_holdout"] = [probe(shadow,r) for r in held]
                after["retained"] = [probe(shadow,r) for r in retained]
                reasons = failures(before, after)
                if any(b["ok"]>a["ok"] or not a["eot"] or not a["valid"] for b,a in zip(before["retained"],after["retained"])):
                    reasons.append("過去の経験に後退")
                if any(b["ok"]>a["ok"] or not a["eot"] or not a["valid"] for b,a in zip(before["oracle_holdout"],after["oracle_holdout"])):
                    reasons.append("算術保留集合の後退")
                for row,observed in zip(probes,after["cases"]["progress"]):
                    if not observed["ok"]:
                        reasons.append("未確認の更新 "+row["prompt"])
                if guard_fn and not guard_fn(shadow):
                    reasons.append("追加検査")
                tested = dict(cycle, offset=offset, before=before, after=after, nodes=counts, curiosity=discovery,
                              delta={k:after["scores"][k]-v for k,v in before["scores"].items()},
                              reasons=reasons)
                append(self.log, dict(tested,status="tested"))
                if reasons:
                    status = "rolled_back"
                else:
                    folder.rename(self.gens/gen)
                    self.activate(gen)
                    self.mount(brain,self.gens/gen)
                    status = "committed"
                result = dict(tested,status=status)
                append(self.log,result)
                return result
            except Exception as exc:
                append(self.log,dict(cycle,status="error",reason=str(exc)))
                raise
            finally:
                if folder.exists():
                    shutil.rmtree(folder)

    def consolidate(self, folder, brain, rows, gen):
        """承認候補の事実、演算・時計・社会発話の経験を閉形式で追加する。"""
        axis = pickle.loads((folder/FILES[1]).read_bytes())
        social = pickle.loads((folder/FILES[2]).read_bytes())
        probes = []
        before = len(axis)+len(social)
        born = pruned = 0
        with sqlite3.connect(folder/FILES[0]) as db:
            for row in rows:
                q,k,t = row["prompt"],row["kind"],row["target"]
                if k in ("fact","teach","discovery"):
                    db.execute("INSERT INTO facts VALUES(?,?,?) ON CONFLICT(topic) DO UPDATE SET fact=excluded.fact,source=excluded.source WHERE fact<>excluded.fact",
                               (clean(q),clean(t),"learned:"+gen))
                elif k=="social":
                    bits,vec = encode(brain.mc,q)
                    born += 1
                    pruned += int(q in social)
                    node_id = social[q].index if q in social else max((node.index for node in social.values()),default=99999)+1
                    social[q] = SocialUtteranceNode(node_id,bits,vec,[(bits,t)])
                elif k=="tool:clock" or k.startswith("tool:calc:"):
                    key = re.sub(r"\d+", "#", norm(q))
                    if k.startswith("tool:calc:"):
                        nums = re.findall(r"\d+",norm(q))
                        if len(nums)!=2:
                            continue
                        try:
                            value = arithmetic(k[-1].join(nums))
                        except (ValueError, SyntaxError, ArithmeticError):
                            continue
                        if t and value!=t:
                            continue
                        row = dict(row,target=value)
                    born += 1
                    pruned += int(key in axis)
                    axis[key] = (k,grams(q),thought(brain.mc,q),op_features(brain.mc,q))
                else:
                    continue
                probes.append(row)
        for name, nodes in zip(FILES[1:],(axis,social)):
            atomic(folder/name,pickle.dumps(nodes))
        return probes, dict(before=before,born=born,pruned=pruned,alive=len(axis)+len(social))
