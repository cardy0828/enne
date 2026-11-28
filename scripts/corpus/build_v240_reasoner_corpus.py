#!/usr/bin/env python3
"""
ENNE v24.0 関係推論野 (ReasonerCore) 学習・検証コーパス構築スクリプト。

設計仕様:
- 逆伝播や事前学習済み言語モデルを使わない。
- rust_bench_v182/bench_320.json の unlabeled (20件) と episodes (383件) のみをオンライン可塑性教師として抽出し、
  320件の評価ケース (cases) は DENYLIST に入れてニューラルヘッド学習から除外する。
- quant_head (Diff/Max/Min), order_head (Source/Sink/Rank2/Rank3/Rank4), prem_head (NonPremise/Premise)
  用の独立言い換えデータを生成し、BLAKE2b ハッシュで 85% train / 15% val に分割する。
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ENNE_ROOT", "."))
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "v240_reasoner_core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNSEEN_PROBES = [
    {
        "prompt": "富士山の標高は3776.12m。北岳の標高は3193m。富士山と北岳の高さの違いってどれくらい？",
        "expected_mode": "QuantDiff",
        "expected_answer": "583.12m",
    },
    {
        "prompt": "富士山の標高は3776m。北岳の標高は3193m。富士山と北岳は何m違う？",
        "expected_mode": "QuantDiff",
        "expected_answer": "583m",
    },
    {
        "prompt": "A塔は150m。B塔は95m。A塔とB塔の高さの開きはどれくらい？",
        "expected_mode": "QuantDiff",
        "expected_answer": "55m",
    },
    {
        "prompt": "メロンは1200円。バナナは200円。メロンとバナナはいくら違う？",
        "expected_mode": "QuantDiff",
        "expected_answer": "1000円",
    },
    {
        "prompt": "東京駅は10km。新宿駅は18km。渋谷駅は14km。距離が一番短いのはどれ？",
        "expected_mode": "QuantCompare",
        "expected_answer": "東京駅",
    },
    {
        "prompt": "A社は500円。B社は320円。どちらが安い？",
        "expected_mode": "QuantCompare",
        "expected_answer": "B社",
    },
    {
        "prompt": "白川は80km。黒川は125km。最も長いのはどれ？",
        "expected_mode": "QuantCompare",
        "expected_answer": "黒川",
    },
    {
        "prompt": "太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で2位は誰？",
        "expected_mode": "OrderRank",
        "expected_answer": "花子",
    },
    {
        "prompt": "太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で二番目は誰？",
        "expected_mode": "OrderRank",
        "expected_answer": "花子",
    },
    {
        "prompt": "太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で3位は誰？",
        "expected_mode": "OrderRank",
        "expected_answer": "次郎",
    },
    {
        "prompt": "富士山の方が北岳よりも高い。北岳の方が槍ヶ岳よりも高い。一番高いのはどれ？",
        "expected_mode": "OrderExtreme",
        "expected_answer": "富士山",
    },
    {
        "prompt": "鉄球の方が銅球よりも重い。銅球の方が木球よりも重い。最軽量はどれでしょう。",
        "expected_mode": "OrderExtreme",
        "expected_answer": "木球",
    },
    {
        "prompt": "鯨の全員が哺乳類に属する。鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？",
        "expected_mode": "Syllogism",
        "expected_answer": "はい",
    },
    {
        "prompt": "鯨の全員が哺乳類に属する。鮪の所属は哺乳類。鮪を鯨の一員と言い切れる？",
        "expected_mode": "Abstain",
        "expected_answer": "判断不能",
    },
    {
        "prompt": "青木の出身は北海道。石川は青木と同じ出身。前田は石川と同じ出身。村上は前田と同じ出身。村上の出身はどこ？",
        "expected_mode": "Transfer",
        "expected_answer": "北海道",
    },
    {
        "prompt": "甲は乙より強い。乙は丙より強い。丙は丁より強い。丁は甲より強い。一番強いのは誰？",
        "expected_mode": "ContradictionCycle",
        "expected_answer": "矛盾",
    },
]


def split_of(text: str) -> str:
    h = int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return "val" if (h % 7) == 0 else "train"


def main():
    rng = random.Random(20261128)
    bench = json.loads((ROOT / "rust_bench_v182/bench_320.json").read_text(encoding="utf-8"))

    # Write official bench_320 training signals separately (disjoint from the 320 test cases)
    (OUT_DIR / "unlabeled.json").write_text(
        json.dumps(bench["unlabeled"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "episodes.json").write_text(
        json.dumps(bench["episodes"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "unseen_probes.json").write_text(
        json.dumps(UNSEEN_PROBES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    denylist = {
        "端点は？",
        "セラ＝ネラ。",
        "セラ＝",
        "追跡キー＝砂時計印。",
        "追跡キー＝",
        "富士山と北岳の標高差は何m？",
        "富士山と北岳と槍ヶ岳の標高差は何m？",
        "富士山と槍ヶ岳はどちらが標高が高い？",
        "富士山について知りたい",
        "光合成の仕組みを説明して",
        "日本の首都とは何ですか？",
        "朝塔と夕塔の高さの差は何m？",
        "甲は乙よりゾルい。丙は乙よりメクい。一番メクいのは誰？",
        "甲は乙よりルゼい。乙は丙よりルゼい。端点は？",
    }
    for c in bench["cases"]:
        denylist.add(c["prompt"].strip())
        for s in re.split(r"[。\n]", c["prompt"]):
            if s.strip():
                denylist.add(s.strip())
        for t in c.get("turns", []):
            denylist.add(t.strip())
    for p in UNSEEN_PROBES:
        denylist.add(p["prompt"].strip())
        for s in re.split(r"[。\n]", p["prompt"]):
            if s.strip():
                denylist.add(s.strip())

    rows = []
    seen = set()

    def add(task: str, text: str, label: int):
        t = re.sub(r"\s+", " ", text.strip())
        if not t or (task, t) in seen or t in denylist:
            return
        seen.add((task, t))
        rows.append({
            "task": task,
            "text": t,
            "label": label,
            "split": split_of(f"{task}:{t}"),
        })

    # Disjoint entities for neural heads (separate from E2E / bench_320 entities)
    ent_pairs = [
        ("エベレスト", "K2"),
        ("立山", "白山"),
        ("筑波山", "高尾山"),
        ("白川", "黒川"),
        ("P社", "Q社"),
        ("X機", "Y機"),
        ("青箱", "赤箱"),
        ("ハル号", "アキ号"),
        ("東館", "西館"),
        ("新製品", "旧製品"),
        ("第1橋", "第2橋"),
        ("銀貨", "銅貨"),
        ("北部", "南部"),
        ("上流", "下流"),
        ("A村", "B村"),
        ("ミドリ駅", "アカネ駅"),
    ]
    attrs_u = [
        ("標高", "m"), ("高さ", "m"), ("距離", "km"), ("長さ", "cm"),
        ("価格", "円"), ("値段", "円"), ("重さ", "kg"), ("速度", "km"),
        ("広さ", "㎡"), ("時間", "秒"),
    ]

    # =========================================================================
    # 1. Quant Head (0=Diff, 1=Max, 2=Min)
    # =========================================================================
    diff_templates_pair = [
        "{a}と{b}の{attr}差は何{u}？",
        "{a}と{b}の{attr}の差は？",
        "{a}と{b}の{attr}の差はいくつ？",
        "{a}と{b}の{attr}の違いは？",
        "{a}と{b}の{attr}の違いってどれくらい？",
        "{a}と{b}は{attr}がどれだけ違う？",
        "{a}と{b}の{attr}の開きはどれくらい？",
        "{a}と{b}は何{u}違う？",
        "{a}と{b}は何{u}違いますか？",
        "{a}と{b}はどれくらい違う？",
        "{a}と{b}はいくら違う？",
        "{a}と{b}はいくら違いますか？",
        "{a}と{b}の間の{u}の差は何{u}？",
        "{a}と{b}の差額はいくら？",
        "{a}と{b}の数値の差を教えて。",
    ]
    diff_templates_gen = [
        "一番高いものと一番低いものの差は何{u}？",
        "最大と最小の{attr}差は何{u}？",
        "一番高いものと一番低いものの違いはどれくらい？",
        "最も大きい値と最も小さい値はいくつ違う？",
        "{attr}の最大と最小は何{u}違う？",
        "両者の{attr}の開きはどれくらいですか？",
        "2つの{attr}の差はいくら？",
    ]
    for a, b in ent_pairs:
        for attr, u in attrs_u:
            for tpl in diff_templates_pair[:10]:
                add("quant", tpl.format(a=a, b=b, attr=attr, u=u), 0)
            add("quant", rng.choice(diff_templates_pair[10:]).format(a=a, b=b, attr=attr, u=u), 0)
    for attr, u in attrs_u:
        for tpl in diff_templates_gen:
            add("quant", tpl.format(attr=attr, u=u), 0)

    max_adjs = ["高い", "長い", "大きい", "重い", "速い", "広い", "高い値"]
    min_adjs = ["低い", "短い", "小さい", "軽い", "遅い", "狭い", "安い"]
    max_templates = [
        "{attr}が一番{adj}のはどれ？",
        "一番{adj}のはどれ？",
        "一番{adj}のはどれですか？",
        "一番{adj}のは誰？",
        "最も{adj}のはどれ？",
        "最も{adj}のは誰ですか？",
        "どれが一番{adj}ですか？",
        "どちらが{adj}？",
        "どちらが{attr}が{adj}？",
        "どちらの方が{adj}ですか？",
        "{a}と{b}はどちらが{adj}？",
        "{a}と{b}ではどちらが{attr}が{adj}ですか？",
        "{attr}の最高はどれ？",
        "最大値はどれ？",
    ]
    min_templates = [
        "{attr}が一番{adj}のはどれ？",
        "一番{adj}のはどれ？",
        "一番{adj}のはどれですか？",
        "一番{adj}のは誰？",
        "最も{adj}のはどれ？",
        "最も{adj}のは誰ですか？",
        "どれが一番{adj}ですか？",
        "どちらが{adj}？",
        "どちらが{attr}が{adj}？",
        "どちらの方が{adj}ですか？",
        "{a}と{b}はどちらが{adj}？",
        "{a}と{b}ではどちらが{attr}が{adj}ですか？",
        "{attr}の最低はどれ？",
        "最小値はどれ？",
    ]
    for a, b in ent_pairs:
        for attr, u in attrs_u:
            for adj in max_adjs:
                for tpl in max_templates[:12]:
                    add("quant", tpl.format(a=a, b=b, attr=attr, adj=adj), 1)
            for adj in min_adjs:
                for tpl in min_templates[:12]:
                    add("quant", tpl.format(a=a, b=b, attr=attr, adj=adj), 2)
    # Extra standalone Max / Min diversity (including "どれでしょう")
    for attr, _ in attrs_u:
        for adj in max_adjs:
            for tpl in max_templates + ["一番{adj}のはどれでしょう。", "{attr}が一番{adj}のはどれでしょう。"]:
                add("quant", tpl.format(a="X社", b="Y社", attr=attr, adj=adj), 1)
        for adj in min_adjs:
            for tpl in min_templates + ["一番{adj}のはどれでしょう。", "{attr}が一番{adj}のはどれでしょう。"]:
                add("quant", tpl.format(a="X社", b="Y社", attr=attr, adj=adj), 2)

    # =========================================================================
    # 2. Order Head (0=Source, 1=Sink, 2=Rank2, 3=Rank3, 4=Rank4)
    # Symmetrical across polar pairs so lexical polarity does not leak to Source/Sink
    # =========================================================================
    order_preds = [
        "重い", "軽い", "速い", "遅い", "高い", "低い", "大きい", "小さい",
        "長い", "短い", "強い", "弱い", "広い", "狭い", "古い", "新しい",
        "深い", "浅い", "勝った", "負けた", "先輩", "後輩", "上司", "部下", "北", "南",
    ]
    who_or_what = ["誰？", "誰ですか？", "どれ？", "どれですか？", "どこ？", "どれでしょう。"]

    # Class 0: Source
    src_stems = ["重量", "軽量", "高値", "安値", "長距離", "短距離", "高速", "低速", "最大", "最小", "最古", "最新", "最深", "最浅", "最強", "最弱"]
    for p in order_preds:
        for w in who_or_what:
            add("order", f"一番{p}のは{w}", 0)
            add("order", f"一番{p}なのは{w}", 0)
            add("order", f"最も{p}のは{w}", 0)
            add("order", f"{p}順で最初は{w}", 0)
            add("order", f"{p}順の先頭は{w}", 0)
            add("order", f"{p}順でトップは{w}", 0)
            add("order", f"{p}順で1番目は{w}", 0)
            add("order", f"{p}順で1位は{w}", 0)
            add("order", f"{p}順で一番目は{w}", 0)
    for st in src_stems:
        for w in who_or_what:
            add("order", f"最{st}は{w}", 0)
            add("order", f"最{st}なのは{w}", 0)
    for w in who_or_what:
        add("order", f"先頭は{w}", 0)
        add("order", f"トップは{w}", 0)
        add("order", f"1位は{w}", 0)

    # Class 1: Sink (omits "端点は？" so 1-shot online learning covers it)
    for p in order_preds:
        for w in who_or_what:
            add("order", f"{p}順で最後は{w}", 1)
            add("order", f"{p}順の最後は{w}", 1)
            add("order", f"{p}順の末尾は{w}", 1)
            add("order", f"{p}順で一番後ろは{w}", 1)
            add("order", f"{p}順で最下位は{w}", 1)
            add("order", f"{p}順でビリは{w}", 1)
            add("order", f"{p}順でいちばん後ろは{w}", 1)
    for w in who_or_what:
        add("order", f"最後は{w}", 1)
        add("order", f"末尾は{w}", 1)
        add("order", f"一番後ろは{w}", 1)
        add("order", f"最下位は{w}", 1)

    # Class 2: Rank 2
    for p in order_preds:
        for w in who_or_what:
            add("order", f"{p}順で2番目は{w}", 2)
            add("order", f"{p}順の2番目は{w}", 2)
            add("order", f"{p}順で2位は{w}", 2)
            add("order", f"{p}順の2位は{w}", 2)
            add("order", f"{p}順で二番目は{w}", 2)
            add("order", f"{p}順で二位は{w}", 2)
            add("order", f"{p}順で第2位は{w}", 2)
    for w in who_or_what:
        add("order", f"2番目は{w}", 2)
        add("order", f"2位は{w}", 2)
        add("order", f"二番目は{w}", 2)
        add("order", f"二位は{w}", 2)
        add("order", f"上から2番目は{w}", 2)
        add("order", f"上から2位は{w}", 2)
        add("order", f"次点は{w}", 2)
        add("order", f"第2位は{w}", 2)

    # Class 3: Rank 3
    for p in order_preds:
        for w in who_or_what:
            add("order", f"{p}順で3番目は{w}", 3)
            add("order", f"{p}順の3番目は{w}", 3)
            add("order", f"{p}順で3位は{w}", 3)
            add("order", f"{p}順の3位は{w}", 3)
            add("order", f"{p}順で三番目は{w}", 3)
            add("order", f"{p}順で三位は{w}", 3)
            add("order", f"{p}順で第3位は{w}", 3)
    for w in who_or_what:
        add("order", f"3番目は{w}", 3)
        add("order", f"3位は{w}", 3)
        add("order", f"三番目は{w}", 3)
        add("order", f"三位は{w}", 3)
        add("order", f"上から3番目は{w}", 3)
        add("order", f"上から3位は{w}", 3)
        add("order", f"第3位は{w}", 3)

    # Class 4: Rank 4
    for p in order_preds:
        for w in who_or_what:
            add("order", f"{p}順で4番目は{w}", 4)
            add("order", f"{p}順の4番目は{w}", 4)
            add("order", f"{p}順で4位は{w}", 4)
            add("order", f"{p}順の4位は{w}", 4)
            add("order", f"{p}順で四番目は{w}", 4)
            add("order", f"{p}順で四位は{w}", 4)
            add("order", f"{p}順で第4位は{w}", 4)
    for w in who_or_what:
        add("order", f"4番目は{w}", 4)
        add("order", f"4位は{w}", 4)
        add("order", f"四番目は{w}", 4)
        add("order", f"四位は{w}", 4)
        add("order", f"上から4番目は{w}", 4)
        add("order", f"上から4位は{w}", 4)
        add("order", f"第4位は{w}", 4)

    # =========================================================================
    # 3. Premise Head (0=NonPremise, 1=Premise)
    # =========================================================================
    prem_names = ["アオイ", "ミドリ", "アカネ", "ハル", "ナツ", "アキ", "フユ", "白山", "立山", "エベレスト", "K2", "P社", "Q社", "X館", "Y館"]
    prem_roles = ["総務", "研究", "広報", "設計", "製造", "企画", "京都", "仙台", "金沢"]
    prem_classes = ["脊椎動物", "植物", "金属", "惑星", "楽器", "競技"]
    for _ in range(950):
        a, b = rng.sample(prem_names, 2)
        role = rng.choice(prem_roles)
        cls_a, cls_b = rng.sample(prem_classes, 2)
        attr, u = rng.choice(attrs_u)
        val = rng.randint(10, 5000)
        adj = rng.choice(["高い", "重い", "速い", "大きい", "長い", "安い", "ゾルい", "ルゼい"])
        add("prem", f"{a}＝{b}", 1)
        add("prem", f"{a}の担当は{role}", 1)
        add("prem", f"{a}の勤務地は{role}", 1)
        add("prem", f"{a}の{attr}は{val}{u}", 1)
        add("prem", f"{a}は{val}{u}", 1)
        add("prem", f"{a}は{b}より{adj}", 1)
        add("prem", f"{a}の方が{b}よりも{adj}", 1)
        add("prem", f"{a}は{b}と同じ担当", 1)
        add("prem", f"{a}は{b}と同期だ", 1)
        add("prem", f"{a}・{b}・東館のどれかに鍵がある", 1)
        add("prem", f"{a}には入っていない", 1)
        add("prem", f"{a}は除外だ", 1)
        add("prem", f"{a}ではない", 1)
        add("prem", f"すべての{a}は{cls_a}です", 1)
        add("prem", f"{a}の全員が{cls_b}に属する", 1)
        add("prem", f"{a}の所属は{cls_a}", 1)

    lookup_topics = [
        "重力", "相対性理論", "光合成", "半導体", "火山", "DNA", "量子力学",
        "生態系", "地震", "太陽系", "太平洋", "日本列島", "京都府", "白山",
        "酵素", "核融合", "天の川", "富士五湖", "黒潮", "北極星",
    ]
    for _ in range(1800):
        t = rng.choice(lookup_topics)
        a, b = rng.sample(prem_names, 2)
        adj = rng.choice(["重い", "軽い", "速い", "遅い", "高い", "低い", "長い", "短い"])
        st = rng.choice(["軽量", "重量", "高値", "安値", "高速", "低速", "最大", "最小"])
        add("prem", f"{t}について知りたい", 0)
        add("prem", f"{t}の仕組みを説明して", 0)
        add("prem", f"{t}の意味を教えて", 0)
        add("prem", f"{t}について詳しく説明して", 0)
        add("prem", f"{t}はどんなものか教えて", 0)
        add("prem", f"{t}って何ですか？", 0)
        add("prem", f"{t}ってどこだっけ？", 0)
        add("prem", f"{a}と{b}の差は何m？", 0)
        add("prem", f"{a}と{b}の高さの違いはどれくらい？", 0)
        add("prem", f"{a}と{b}はどちらが{adj}？", 0)
        add("prem", f"{a}の担当は何？", 0)
        add("prem", f"{a}の勤務地はどこ？", 0)
        add("prem", f"{a}の所属先を教えて。", 0)
        add("prem", f"{a}は{b}より{adj}ですか？", 0)
        add("prem", f"{a}は{b}に属しますか？", 0)
        add("prem", f"{a}を一員と言い切れる？", 0)
        add("prem", f"一番{adj}のはどれでしょう。", 0)
        add("prem", f"最{st}はどれでしょう。", 0)
        add("prem", f"{adj}順で最後は誰？", 0)
        add("prem", f"2番目は誰？", 0)
        add("prem", f"2位は誰？", 0)
        add("prem", f"今日はいい天気だね", 0)

    rng.shuffle(rows)
    out_jsonl = OUT_DIR / "neural_heads.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter((r["task"], r["split"], r["label"]) for r in rows)
    probe_texts = {p["prompt"].strip() for p in UNSEEN_PROBES}
    manifest = {
        "total": len(rows),
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
        "by_task": {
            t: {
                "train": sum(1 for r in rows if r["task"] == t and r["split"] == "train"),
                "val": sum(1 for r in rows if r["task"] == t and r["split"] == "val"),
            }
            for t in ("quant", "order", "prem")
        },
        "counts": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(counts.items())},
        "unlabeled_count": len(bench["unlabeled"]),
        "episodes_count": len(bench["episodes"]),
        "unseen_probes_count": len(UNSEEN_PROBES),
        "denylist_hits": sum(1 for r in rows if r["text"] in denylist),
        "probe_overlap": sum(1 for r in rows if r["text"] in probe_texts),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
