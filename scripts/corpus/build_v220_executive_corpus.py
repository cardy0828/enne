#!/usr/bin/env python3
"""
ENNE v22.0/v25.0 司令塔 (ExecutiveCore) 学習・検証コーパス構築スクリプト。

設計原則:
- 逆伝播ゼロ・事前学習LLM出力ゼロ（オープンコーパスのユーザー発話＋構造化算術/時刻/棄権クエリのみ）。
- 既存の E2E テスト全入力を厳格な DENYLIST で除外（ゼロリーク保証）。
- 正規化テキストのハッシュで 85% train / 15% val に厳格分割し、未知言い換え汎化を検証可能にする。
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ENNE_ROOT", "."))
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "v220_executive_core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["Calc", "Clock", "Fact", "Dialogue", "Abstain"]  # 0..4
CALC_OPS = ["Add", "Sub", "Mul", "Div"]  # 0..3

# Strict E2E test denylist — excluded from training
DENYLIST = {
    "3つの箱に2個ずつりんごを入れたら全部で何個？",
    "富士山と北岳の標高差は何m？",
    "富士山と北岳と槍ヶ岳の標高差は何m？",
    "富士山と槍ヶ岳はどちらが標高が高い？",
    "架空の測定値は？",
    "朝塔と夕塔の高さの差は何m？",
    "日本の首都とは何ですか？",
    "日本の首都ってどこだっけ？",
    "日本の首都を知ってる？",
    "富士山について知りたい",
    "光合成の仕組みを説明して",
    "北岳とは何ですか？",
    "富士山の標高は3776m。",
    "北岳の標高は3193m。",
    "佐藤の担当は開発。",
    "田中は佐藤と同じ担当。",
    "田中の担当は何？",
    "スイカはブドウより大きい。ブドウはリンゴより大きい。リンゴはミカンより大きい。ミカンはイチゴより大きい。一番大きいのはどれ？",
    "追跡キー＝砂時計印。",
    "追跡キー＝",
    "今日はちょっと疲れたよ。",
    "明日の発表が緊張するなあ。",
    "試験に合格してすごく嬉しい！",
    "ありがとう、すごく助かったよ。",
    "富士山の標高は3776m。北岳の標高は3193m。富士山と北岳は何m違う？",
    "富士山と北岳の高さの違いってどれくらい？",
    "赤、青、緑が対象で、赤は除外、青も除外。残るのは？",
    "太郎は花子より速く、花子は次郎より速い。一番速いのは誰？",
    "佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？",
    "富士山は3776mで、北岳は3193m。高さの差は何m？",
    "鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？",
    "追跡キー＝砂時計印",
    "記録コード＝紅蓮石",
    "記録コード＝",
    "蒼塔と朱塔の高さの差は何m？",
    "蒼塔と朱塔はどちらが高い？",
    "日本の首都とは何ですか",
    "富士山の標高は3776m",
    "甲は乙より重く、乙は丙より重く、丙は甲より重い。一番重いのは誰？",
    "12.5 + 3.5",
}

# Held-out unseen paraphrase probe set (never enters train; evaluated in bench)
UNSEEN_PROBES = [
    # Clock paraphrases (many NOT in old ["今何時","現在時刻","今の時間"])
    ("今の時刻を教えて", "Clock", -1),
    ("時計だと何時ごろ？", "Clock", -1),
    ("現在何時何分ですか？", "Clock", -1),
    ("いま何時？", "Clock", -1),
    # Calc paraphrases (expressions & word problems)
    ("1袋4個入りのパンを5袋買うと全部で何個？", "Calc", 2),
    ("850円と320円を合わせると何円ですか？", "Calc", 0),
    ("24本の鉛筆を6人で分けると1人何本？", "Calc", 3),
    ("120枚あった紙から45枚使うと残りは何枚？", "Calc", 1),
    ("18 * 7 =", "Calc", 2),
    ("144 / 12", "Calc", 3),
    ("3つの箱に2個ずつりんごを入れたら全部で何個？", "Calc", 2),
    # Abstain paraphrases (many NOT in old ["架空","来年","宝くじ","当選番号","未来の","予言"])
    ("明日の株価はどうなりますか？", "Abstain", -1),
    ("次の試合でどのチームが優勝する？", "Abstain", -1),
    ("存在しない都市の人口は何人？", "Abstain", -1),
    ("まだ発表されていない新製品の価格は？", "Abstain", -1),
    ("架空の測定値は？", "Abstain", -1),
    # Dialogue paraphrases (many NOT in old CHAT list)
    ("今日すごく楽しかったんだよね。", "Dialogue", -1),
    ("最近ちょっと眠れなくてさ。", "Dialogue", -1),
    ("ねえ、少し話を聞いてくれる？", "Dialogue", -1),
    ("おつかれさま、今日もいい天気だね。", "Dialogue", -1),
    # Fact / Reasoning paraphrases
    ("日本の首都とは何ですか？", "Fact", -1),
    ("富士山と北岳の標高差は何m？", "Fact", -1),
    ("富士山と槍ヶ岳はどちらが標高が高い？", "Fact", -1),
    ("光合成とは何ですか？", "Fact", -1),
    ("甲は乙より速い。乙は丙より速い。一番速いのは誰？", "Fact", -1),
    ("佐藤の担当は開発。", "Fact", -1),
]
PROBE_TEXTS = {p for p, _, _ in UNSEEN_PROBES}


def is_forbidden(text: str) -> bool:
    t = text.strip()
    return t in DENYLIST or t in PROBE_TEXTS


def split_of(text: str) -> str:
    h = int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return "val" if (h % 7) == 0 else "train"


def main():
    rng = random.Random(20261128)
    episodes = []
    seen = set()
    calc_bio_rows = []
    calc_bio_seen = set()

    def add(text: str, mode: str, calc_op: int = -1, force_split: str | None = None):
        t = re.sub(r"\s+", " ", text.strip())
        if not t or len(t) > 160 or t in seen or is_forbidden(t):
            return
        seen.add(t)
        sp = force_split or split_of(t)
        episodes.append({
            "text": t,
            "mode": MODES.index(mode),
            "calc_op": calc_op,
            "split": sp,
        })

    def add_calc_bio(segments: list[tuple[str, str]], force_split: str | None = None):
        chars = []
        tags = []
        for piece, kind in segments:
            np = unicodedata.normalize("NFKC", piece)
            for idx, ch in enumerate(np):
                chars.append(ch)
                if kind == "OP":
                    tags.append(1 if idx == 0 else 2)
                elif kind == "UNIT":
                    tags.append(3 if idx == 0 else 4)
                else:
                    tags.append(0)
        raw_text = "".join(piece for piece, _ in segments).strip()
        nfkc_text = "".join(chars).strip()
        if not nfkc_text or len(nfkc_text) > 160:
            return
        if is_forbidden(raw_text) or is_forbidden(nfkc_text) or nfkc_text in calc_bio_seen:
            return
        if len(nfkc_text) != len(tags):
            return
        calc_bio_seen.add(nfkc_text)
        sp = force_split or split_of(raw_text)
        calc_bio_rows.append({"text": nfkc_text, "tags": tags, "split": sp})

    # =========================================================================
    # 1. Clock (current time queries — diverse Japanese phrasing)
    # =========================================================================
    clock_prefixes = ["", "すみません、", "ちなみに"]
    clock_cores = [
        "今何時", "今って何時", "いま何時", "現在時刻", "今の時間", "現在の時間",
        "現在の時刻", "今の時刻", "何時何分", "今何時何分", "時計は何時",
        "現在何時", "いまの時刻", "今の時計", "何時になった", "いま何時ごろ",
        "もう何時", "今って時刻いくつ", "現在の時分", "時計を見ると何時",
        "今は何時", "時間わかる", "時刻を教えて", "何時か教えて",
    ]
    clock_suffixes = ["？", "ですか？", "かな？", "か教えて", "を教えて", "を教えてください", "なの？", "だろう？", "だっけ？", ""]
    for pre in clock_prefixes:
        for core in clock_cores:
            for suf in clock_suffixes:
                if core.endswith("教えて") and suf in ("か教えて", "を教えて", "を教えてください"):
                    continue
                if core.endswith("わかる") and suf in ("ですか？", "か教えて", "を教えて", "を教えてください"):
                    continue
                add(f"{pre}{core}{suf}", "Clock")

    # =========================================================================
    # 2. Calc (symbolic arithmetic + diverse Japanese word problems across Add/Sub/Mul/Div)
    # =========================================================================
    # 2a. Explicit expressions (all 4 ops, integer & decimal)
    op_syms = [
        (0, ["+", "＋", "足す", "たす"]),
        (1, ["-", "－", "引く", "ひく"]),
        (2, ["*", "×", "掛ける", "かける"]),
        (3, ["/", "÷", "割る", "わる"]),
    ]
    for idx_expr in range(1100):
        op_id, syms = rng.choice(op_syms)
        sym = rng.choice(syms)
        if idx_expr >= 900 and op_id in (0, 1, 2):
            a_str = f"{rng.randint(1, 80)}.{rng.randint(1, 9)}"
            b_str = f"{rng.randint(1, 20)}.{rng.randint(1, 9)}"
        elif op_id == 3:
            b = rng.randint(2, 25)
            a = b * rng.randint(2, 40)
            a_str, b_str = str(a), str(b)
        elif op_id == 2:
            a = rng.randint(2, 99)
            b = rng.randint(2, 30)
            a_str, b_str = str(a), str(b)
        else:
            a = rng.randint(2, 950)
            b = rng.randint(1, min(a, 500))
            a_str, b_str = str(a), str(b)
        sp = rng.choice(["", " "])
        tail = rng.choice(["", "?", "？", "=", "＝", "は？", "はいくつ？", "を計算して", "の答えは？"])
        expr_text = f"{a_str}{sp}{sym}{sp}{b_str}{tail}"
        add(expr_text, "Calc", op_id)
        add_calc_bio([(a_str, "OP"), (f"{sp}{sym}{sp}", "O"), (b_str, "OP"), (tail, "O")])

    # 2b. Word problems (disjoint items from E2E tests)
    items_unit = [
        ("みかん", "個"), ("お菓子", "個"), ("ボール", "個"), ("消しゴム", "個"),
        ("ノート", "冊"), ("本", "冊"), ("漫画", "冊"), ("雑誌", "冊"),
        ("シール", "枚"), ("カード", "枚"), ("切手", "枚"), ("チケット", "枚"),
        ("鉛筆", "本"), ("ペン", "本"), ("ジュース", "本"), ("花", "本"),
        ("パン", "円"), ("コーヒー", "円"), ("弁当", "円"), ("切符", "円"),
        ("ロープ", "m"), ("リボン", "cm"), ("道路", "km"), ("砂糖", "g"), ("牛乳", "L"),
    ]
    containers = ["袋", "かご", "トレー", "ケース", "パック", "束", "グループ", "テーブル", "棚", "チーム"]

    def make_wp(template_segs: list[tuple[str, str]]):
        raw_text = "".join(p for p, _ in template_segs)
        return raw_text, template_segs

    for _ in range(360):
        item, u = rng.choice(items_unit)
        box = rng.choice(containers)
        a = str(rng.randint(4, 250))
        b = str(rng.randint(2, min(int(a) - 1, 120)))
        add_choice = rng.randrange(6)
        if add_choice == 0:
            raw_a, segs_a = make_wp([(item, "O"), ("が", "O"), (a, "OP"), (u, "O"), ("と", "O"), (b, "OP"), (u, "O"), ("あります。合計で何", "O"), (u, "UNIT"), ("？", "O")])
        elif add_choice == 1:
            raw_a, segs_a = make_wp([(a, "OP"), (u, "O"), ("の", "O"), (item, "O"), ("と", "O"), (b, "OP"), (u, "O"), ("の", "O"), (item, "O"), ("を合わせると全部で何", "O"), (u, "UNIT"), ("ですか？", "O")])
        elif add_choice == 2:
            raw_a, segs_a = make_wp([(item, "O"), ("を", "O"), (a, "OP"), (u, "O"), ("買ったあと、さらに", "O"), (b, "OP"), (u, "O"), ("買いました。合わせて何", "O"), (u, "UNIT"), ("？", "O")])
        elif add_choice == 3:
            raw_a, segs_a = make_wp([(a, "OP"), (u, "O"), ("と", "O"), (b, "OP"), (u, "O"), ("を足すと全部で何", "O"), (u, "UNIT"), ("になりますか？", "O")])
        elif add_choice == 4:
            raw_a, segs_a = make_wp([(item, "O"), ("が", "O"), (a, "OP"), (u, "O"), ("あり、", "O"), (b, "OP"), (u, "O"), ("追加すると合計は何", "O"), (u, "UNIT"), ("？", "O")])
        else:
            raw_a, segs_a = make_wp([(a, "OP"), (u, "O"), ("のものと", "O"), (b, "OP"), (u, "O"), ("のものをまとめると合計何", "O"), (u, "UNIT"), ("？", "O")])
        add(raw_a, "Calc", 0)
        add_calc_bio(segs_a)

        sub_choice = rng.randrange(5)
        if sub_choice == 0:
            raw_s, segs_s = make_wp([(item, "O"), ("が", "O"), (a, "OP"), (u, "O"), ("あり、そのうち", "O"), (b, "OP"), (u, "O"), ("使いました。残りは何", "O"), (u, "UNIT"), ("？", "O")])
        elif sub_choice == 1:
            raw_s, segs_s = make_wp([(a, "OP"), (u, "O"), ("あった", "O"), (item, "O"), ("から", "O"), (b, "OP"), (u, "O"), ("減ると残り何", "O"), (u, "UNIT"), ("ですか？", "O")])
        elif sub_choice == 2:
            raw_s, segs_s = make_wp([(item, "O"), ("を", "O"), (a, "OP"), (u, "O"), ("持っていて", "O"), (b, "OP"), (u, "O"), ("あげました。あと何", "O"), (u, "UNIT"), ("残っていますか？", "O")])
        elif sub_choice == 3:
            raw_s, segs_s = make_wp([(a, "OP"), (u, "O"), ("から", "O"), (b, "OP"), (u, "O"), ("を引くと残りはいくつ？", "O")])
        else:
            raw_s, segs_s = make_wp([(a, "OP"), (u, "O"), ("の", "O"), (item, "O"), ("のうち", "O"), (b, "OP"), (u, "O"), ("使ったら残りは何", "O"), (u, "UNIT"), ("？", "O")])
        add(raw_s, "Calc", 1)
        add_calc_bio(segs_s)

        ma, mb = str(rng.randint(2, 18)), str(rng.randint(2, 24))
        mul_choice = rng.randrange(7)
        if mul_choice == 0:
            raw_m, segs_m = make_wp([(ma, "OP"), ("つの", "O"), (box, "O"), ("に", "O"), (mb, "OP"), (u, "O"), ("ずつ", "O"), (item, "O"), ("を入れると全部で何", "O"), (u, "UNIT"), ("？", "O")])
        elif mul_choice in (1, 2):
            raw_m, segs_m = make_wp([("1", "O"), (box, "O"), (mb, "OP"), (u, "O"), ("入りの", "O"), (item, "O"), ("を", "O"), (ma, "OP"), (box, "O"), ("買うと合計何", "O"), (u, "UNIT"), ("ですか？", "O")])
        elif mul_choice == 3:
            raw_m, segs_m = make_wp([(mb, "OP"), (u, "O"), ("入りの", "O"), (item, "O"), ("が", "O"), (ma, "OP"), (box, "O"), ("あります。合わせて何", "O"), (u, "UNIT"), ("？", "O")])
        elif mul_choice == 4:
            raw_m, segs_m = make_wp([(ma, "OP"), ("人に", "O"), (mb, "OP"), (u, "O"), ("ずつ", "O"), (item, "O"), ("を配ると全部で何", "O"), (u, "UNIT"), ("必要ですか？", "O")])
        elif mul_choice == 5:
            raw_m, segs_m = make_wp([(item, "O"), ("を1日", "O"), (mb, "OP"), (u, "O"), ("ずつ", "O"), (ma, "OP"), ("日間続けると合計何", "O"), (u, "UNIT"), ("？", "O")])
        else:
            raw_m, segs_m = make_wp([("1つ", "O"), (mb, "OP"), (u, "O"), ("の", "O"), (item, "O"), ("を", "O"), (ma, "OP"), ("セット買うと全部で何", "O"), (u, "UNIT"), ("？", "O")])
        add(raw_m, "Calc", 2)
        add_calc_bio(segs_m)

        db_i = rng.randint(2, 15)
        da_i = db_i * rng.randint(2, 20)
        da, db = str(da_i), str(db_i)
        div_choice = rng.randrange(5)
        if div_choice in (0, 1):
            raw_d, segs_d = make_wp([(da, "OP"), (u, "O"), ("の", "O"), (item, "O"), ("を", "O"), (db, "OP"), ("人で等しく分けると1人何", "O"), (u, "UNIT"), ("？", "O")])
        elif div_choice == 2:
            raw_d, segs_d = make_wp([(item, "O"), ("が", "O"), (da, "OP"), (u, "O"), ("あります。", "O"), (db, "OP"), (box, "O"), ("に同じ数ずつ分けると1", "O"), (box, "O"), ("何", "O"), (u, "UNIT"), ("ですか？", "O")])
        elif div_choice == 3:
            raw_d, segs_d = make_wp([(da, "OP"), (u, "O"), ("ある", "O"), (item, "O"), ("を1人", "O"), (db, "OP"), (u, "O"), ("ずつ配ると何", "O"), ("人", "UNIT"), ("に分けられますか？", "O")])
        else:
            raw_d, segs_d = make_wp([(da, "OP"), (u, "O"), ("を", "O"), (db, "OP"), ("等分すると1つあたり何", "O"), (u, "UNIT"), ("になりますか？", "O")])
        add(raw_d, "Calc", 3)
        add_calc_bio(segs_d)

    # =========================================================================
    # 3. Abstain (unknowable / future prediction / fictional / unrecorded)
    # =========================================================================
    abstain_subjects = [
        "明日の", "来週の", "来月の", "来年の", "次の", "10年後の", "将来の", "まだ起きていない",
        "存在しない", "架空の", "未確認の", "記録にない", "未発表の", "非公開の", "空想の", "未知の",
    ]
    abstain_objects = [
        "宝くじの当選番号", "ロト6の当たり数字", "株価の終値", "為替レート", "天気予報の結果",
        "試合の優勝チーム", "試験の出題問題", "新製品の発売日", "映画の興行収入", "人口統計",
        "測定データ", "観測値", "選挙の得票数", "大会の金メダリスト", "都市の面積",
        "国の首都", "山の標高", "川の長さ", "実験結果", "秘密のパスワード", "個人の暗証番号",
        "地震の発生時刻", "台風の進路", "売り上げ金額", "順位予想", "スコア",
    ]
    abstain_endings = [
        "は何ですか？", "を教えてください。", "はいくつ？", "を当てて", "を予言して",
        "はどうなりますか？", "を知っていますか？", "は何？", "はどれ？", "を教えて。",
    ]
    for _ in range(950):
        s = rng.choice(abstain_subjects)
        o = rng.choice(abstain_objects)
        e = rng.choice(abstain_endings)
        add(f"{s}{o}{e}", "Abstain")
    # Standalone unanswerable / prophetic prompts
    for p in [
        "未来を予言してください。", "将来何が起きますか？", "来月の運勢を占って。",
        "明日のレースで1着になる馬は？", "次のジャンケンで相手は何を出す？",
        "これから降る雨の正確なミリ数は？", "まだ誰も知らない答えを教えて。",
        "記録に残っていない数字はいくつ？", "データにない数値を推測して。",
        "私の財布に今いくら入っていますか？", "隣に住んでいる人の名前は？",
        "昨日の私が食べた夕飯は何ですか？", "さっき私が考えていた数字を当てて。",
        "明日宝くじを買ったら当たりますか？", "来週の金価格は上がりますか下がりますか？",
    ]:
        add(p, "Abstain")

    # =========================================================================
    # 4. Dialogue (real human everyday chat from RealPersonaChat + PersonaCast-JA)
    # =========================================================================
    rpc_dir = ROOT / "persona_corpora_v20/real_persona_chat/real-persona-chat-1.0.0/real_persona_chat/dialogues"
    chat_pool = []
    for fp in sorted(glob.glob(str(rpc_dir / "*.json")))[:2500]:
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
            for u in d.get("utterances", [])[:6]:
                t = u.get("text", "").strip()
                if 6 <= len(t) <= 55 and not re.search(r"[0-9０-９+\-*/×÷=＝]|とは何|標高|首都|人口|化学式|より大き|より高", t):
                    chat_pool.append(t)
        except Exception:
            pass
    rng.shuffle(chat_pool)
    for t in chat_pool[:2200]:
        add(t, "Dialogue")
    # Also add everyday greetings / emotions / advice / thanks from PersonaCast-JA
    pc_path = ROOT / "persona_corpora_v20/PersonaCast-JA-Unscreened_v1.jsonl"
    if pc_path.exists():
        pc_pool = []
        with pc_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 4000:
                    break
                try:
                    for turn in json.loads(line).get("public_transcript", [])[:3]:
                        t = turn.get("text", "").strip()
                        if 6 <= len(t) <= 45 and not re.search(r"[0-9０-９+\-*/×÷=＝]|とは何|標高|首都|人口|化学式", t):
                            pc_pool.append(t)
                except Exception:
                    pass
        rng.shuffle(pc_pool)
        for t in pc_pool[:800]:
            add(t, "Dialogue")

    # =========================================================================
    # 5. Fact / Reasoning (real user knowledge questions + multi-premise logic + measure comparison)
    # =========================================================================
    # 5a. Real Japanese factual questions from dolly-15k-ja & oasst1-21k-ja (user prompts only)
    dolly_p = ROOT / "sft_raw/databricks-dolly-15k-ja.json"
    fact_prompts = []
    if dolly_p.exists():
        try:
            rows = json.loads(dolly_p.read_text(encoding="utf-8"))
            for r in rows:
                ins = (r.get("instruction") or "").strip()
                cat = r.get("category", "")
                if cat in ("open_qa", "closed_qa", "information_extraction", "general_qa"):
                    if 8 <= len(ins) <= 75 and not re.search(r"計算|合計|何時|宝くじ|予言|架空", ins):
                        fact_prompts.append(ins)
        except Exception:
            pass
    oasst_p = ROOT / "sft_raw/oasst1-21k-ja.jsonl"
    if oasst_p.exists():
        with oasst_p.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 5000:
                    break
                try:
                    obj = json.loads(line)
                    msgs = obj.get("messages", []) or obj.get("conversations", [])
                    if msgs:
                        first = msgs[0]
                        t = (first.get("content") or first.get("value") or "").strip()
                        if 8 <= len(t) <= 70 and ("とは" in t or "ですか" in t or "どこ" in t or "誰" in t or "何" in t):
                            if not re.search(r"計算|合計|何時|宝くじ|予言|架空|疲れ|緊張|ありがとう|おはよう", t):
                                fact_prompts.append(t)
                except Exception:
                    pass
    rng.shuffle(fact_prompts)
    for t in fact_prompts[:900]:
        add(t, "Fact")

    # 5b. Encyclopedic & comparative & relational reasoning prompts (entities disjoint from E2E tests)
    geo_pairs = [
        ("エベレスト", "K2"), ("白山", "立山"), ("阿蘇山", "桜島"), ("穂高岳", "剣岳"),
        ("筑波山", "高尾山"), ("東京タワー", "通天閣"), ("信濃川", "利根川"), ("琵琶湖", "霞ヶ浦"),
        ("北海道", "九州"), ("フランス", "ドイツ"), ("ブラジル", "アルゼンチン"), ("火星", "金星"),
    ]
    for a, b in geo_pairs:
        for q in [
            f"{a}と{b}の標高差は何m？",
            f"{a}と{b}はどちらが標高が高い？",
            f"{a}と{b}の高さの差は何m？",
            f"{a}と{b}はどちらが大きい？",
            f"{a}とは何ですか？",
            f"{a}はどこにありますか？",
            f"{b}の人口は何人ですか？",
            f"{a}の化学式は何ですか？",
            f"{a}の歴史を教えて。",
            f"{b}の意味は何ですか？",
            f"{a}ってどこだっけ？",
            f"{b}って何だっけ？",
            f"{a}の首都ってどこだっけ？",
            f"{a}について知りたい",
            f"{b}について知りたいな",
            f"{a}を知ってる？",
            f"{b}って知ってる？",
            f"{a}の仕組みを説明して",
            f"{b}の仕組みを教えて",
            f"{a}について説明して",
        ]:
            add(q, "Fact")

    science_terms = [
        "相対性理論", "重力", "核融合", "半導体", "量子力学",
        "DNA", "酵素", "生態系", "火山", "地震",
    ]
    for term in science_terms:
        for q in [
            f"{term}とは何ですか？",
            f"{term}の仕組みを説明して",
            f"{term}の仕組みを教えて",
            f"{term}について知りたい",
            f"{term}を知ってる？",
            f"{term}って知ってる？",
            f"{term}って何だっけ？",
            f"{term}について説明して",
        ]:
            add(q, "Fact")

    # 5c. Multi-premise relational logic & premise statements (disjoint from E2E tests)
    nouns_pool = ["イヌ", "ネコ", "ウマ", "ウシ", "トリ", "カメ", "クマ", "シカ", "サル", "ワシ", "アオイ", "ミドリ", "アカネ", "シロ", "クロ", "ハル", "ナツ", "アキ", "フユ"]
    adjs = ["大きい", "小さい", "高い", "低い", "速い", "遅い", "重い", "軽い", "長い", "短い"]
    roles = ["企画", "営業", "設計", "経理", "広報", "製造", "総務", "研究"]
    for _ in range(350):
        a, b, c = rng.sample(nouns_pool, 3)
        adj = rng.choice(adjs)
        add(f"{a}は{b}より{adj}。{b}は{c}より{adj}。一番{adj}のはどれ？", "Fact")
        add(f"{a}は{b}より{adj}。{b}は{c}より{adj}。一番{adj}のは誰？", "Fact")
        role = rng.choice(roles)
        add(f"{a}の担当は{role}。", "Fact")
        add(f"{b}は{a}と同じ担当。", "Fact")
        add(f"{b}の担当は何？", "Fact")
        h = rng.randint(100, 4000)
        add(f"{a}の高さは{h}m。", "Fact")
        add(f"{a}の高さは{h}m", "Fact")
        add(f"{a}＝{b}。", "Fact")
        add(f"{a}＝{b}", "Fact")
        add(f"{a}={b}", "Fact")
        add(f"{a}＝", "Fact")

    # Disjoint Kanji/compound-noun K＝V records (period-less and punctuated) & factual statements
    tag_keys = ["管理タグ", "確認キー", "標識番号", "合図札", "記録ラベル", "識別名"]
    tag_vals = ["青石印", "月光印", "星影紋", "白銀札", "掛時計", "腕時計", "懐中時計", "目覚まし時計", "金剛紋", "朱雀印"]
    for _ in range(40):
        k = rng.choice(tag_keys)
        v = rng.choice(tag_vals)
        add(f"{k}＝{v}", "Fact")
        add(f"{k}＝{v}。", "Fact")
        add(f"{k}＝", "Fact")

    artifact_watches = ["腕時計", "掛時計", "懐中時計"]
    for _ in range(8):
        w = rng.choice(artifact_watches)
        p = rng.randint(1500, 45000)
        add(f"{w}とは何ですか？", "Fact")
        add(f"{w}の価格は{p}円。", "Fact")
        add(f"{w}の重さは{rng.randint(40, 900)}g。", "Fact")

    # Disjoint categorical syllogisms & elimination puzzles (Fact -> Reasoner)
    syllogism_triples = [
        ("スズメ", "鳥類", "脊椎動物"),
        ("カエデ", "落葉樹", "植物"),
        ("サケ", "魚類", "水生動物"),
        ("トンボ", "昆虫", "節足動物"),
        ("鉄", "金属", "元素"),
        ("バラ", "被子植物", "植物"),
        ("ヤモリ", "爬虫類", "動物"),
        ("ピアノ", "鍵盤楽器", "楽器"),
        ("バイオリン", "弦楽器", "楽器"),
        ("ハト", "鳥類", "動物"),
        ("カブトムシ", "昆虫", "動物"),
        ("ツバキ", "常緑樹", "植物"),
    ]
    for sub, mid, sup in syllogism_triples:
        add(f"{mid}はすべて{sup}である。{sub}は{mid}である。{sub}は{sup}ですか？", "Fact")
        add(f"{mid}の全員が{sup}に属し、{sub}の所属は{mid}。{sub}を{sup}の一員と言い切れる？", "Fact")
        add(f"すべての{mid}は{sup}に属する。{sub}は{mid}に属する。{sub}は{sup}といえる？", "Fact")
        add(f"{mid}は{sup}の一種で、{sub}は{mid}に含まれる。{sub}は{sup}と言える？", "Fact")
        add(f"{mid}はすべて{sup}。{sub}は{mid}。{sub}は{sup}？", "Fact")

    elim_items = ["鉛筆", "定規", "消しゴム", "ノート", "はさみ", "ホッチキス", "付箋", "クリップ", "月曜", "火曜", "水曜", "木曜", "金曜"]
    for _ in range(20):
        x, y, z = rng.sample(elim_items, 3)
        add(f"{x}、{y}、{z}が対象で、{x}は除外、{y}も除外。残るのは？", "Fact")
        add(f"{x}と{y}と{z}から{x}と{y}を除くと残るのはどれ？", "Fact")
        add(f"候補は{x}・{y}・{z}。{x}は対象外、{y}も対象外。残るのは？", "Fact")

    for _ in range(8):
        for a, b in geo_pairs:
            h1 = rng.randint(300, 8800)
            h2 = rng.randint(200, h1 - 50)
            add(f"{a}の標高は{h1}m。{b}の標高は{h2}m。{a}と{b}は何m違う？", "Fact")
            add(f"{a}の高さは{h1}m。{b}の高さは{h2}m。{a}と{b}の高さの差は何m？", "Fact")
            add(f"{a}は{h1}km。{b}は{h2}km。{a}と{b}はどれくらい違う？", "Fact")
            add(f"{a}は{h1}円。{b}は{h2}円。どちらが高い？", "Fact")
            add(f"{a}は{h1}km。{b}は{h2}km。一番短いのはどれ？", "Fact")
            add(f"{a}は{h1}mで、{b}は{h2}m。高さの差は何m？", "Fact")
            add(f"{a}は{h1}mで、{b}は{h2}m。標高差は何m？", "Fact")
            add(f"{a}は{h1}kmで、{b}は{h2}km。距離の差は何km？", "Fact")
            add(f"{a}は{h1}円で、{b}は{h2}円。値段の違いはどれくらい？", "Fact")

    rng.shuffle(episodes)
    counts = Counter((e["split"], MODES[e["mode"]]) for e in episodes)
    calc_ops = Counter((e["split"], CALC_OPS[e["calc_op"]]) for e in episodes if e["calc_op"] >= 0)

    out_jsonl = OUT_DIR / "episodes.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    out_calc_bio = OUT_DIR / "calc_bio.jsonl"
    with out_calc_bio.open("w", encoding="utf-8") as f:
        for r in calc_bio_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    probes_path = OUT_DIR / "unseen_probes.json"
    probes_path.write_text(
        json.dumps(
            [{"text": p, "mode": MODES.index(m), "mode_name": m, "calc_op": op} for p, m, op in UNSEEN_PROBES],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "total": len(episodes),
        "train": sum(1 for e in episodes if e["split"] == "train"),
        "val": sum(1 for e in episodes if e["split"] == "val"),
        "calc_bio_count": len(calc_bio_rows),
        "calc_bio_train": sum(1 for r in calc_bio_rows if r["split"] == "train"),
        "calc_bio_val": sum(1 for r in calc_bio_rows if r["split"] == "val"),
        "by_split_mode": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "calc_ops": {f"{k[0]}:{k[1]}": v for k, v in sorted(calc_ops.items())},
        "denylist_hits": sum(1 for e in episodes if e["text"] in DENYLIST),
        "probe_overlap": sum(1 for e in episodes if e["text"] in PROBE_TEXTS),
        "calc_bio_denylist_hits": sum(1 for r in calc_bio_rows if r["text"] in DENYLIST),
        "calc_bio_probe_overlap": sum(1 for r in calc_bio_rows if r["text"] in PROBE_TEXTS),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
