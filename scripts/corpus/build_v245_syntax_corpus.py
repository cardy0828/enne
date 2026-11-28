#!/usr/bin/env python3
"""
ENNE v24.5 文字単位構文・節境界コア (SyntaxCore) 学習・検証コーパス構築スクリプト。

設計仕様:
- 逆伝播や事前学習済みトークナイザ/言語モデルを使わず、文字単位10タグBIOと遷移行列用コーパスを構築する。
- タグ一覧:
  0: SEP    (節・文境界: 。 / \\n / 複文接続の読点 、 / で、)
  1: FUNC   (機能語・格助詞・区切り記号: は / の / が / に / を / と / より / よりも / の方が / ＝ / ・ / 並列読点 、 / ？)
  2: B-ENT, 3: I-ENT   (実体・値・キー・項目)
  4: B-ATTR, 5: I-ATTR (属性名・全称修飾語・集合種別語: 標高 / 担当 / 所属 / すべて / 全員 / 箱)
  6: B-NUM,  7: I-NUM  (数値＋単位スパン: 3776m / 10km / 1200円)
  8: B-PRED, 9: I-PRED (述語・関係・コピュラ・否定抑制: 高い / 速く / 勝った / 同じ / 除外だ / 属する / です)
- bench_320 の評価ケース全320件、v240/v245 未知プローブ、E2E テスト登場実体・文は除外する。
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
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "v245_syntax_core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG_NAMES = [
    "SEP", "FUNC", "B-ENT", "I-ENT", "B-ATTR", "I-ATTR", "B-NUM", "I-NUM", "B-PRED", "I-PRED"
]
ROLE_MAP = {
    "SEP": (0, 0),
    "FUNC": (1, 1),
    "ENT": (2, 3),
    "ATTR": (4, 5),
    "NUM": (6, 7),
    "PRED": (8, 9),
}

UNSEEN_SYNTAX_PROBES = [
    {
        "prompt": "太郎は花子より速く、花子は次郎より速い。一番速いのは誰？",
        "expected_mode": "OrderExtreme",
        "expected_answer": "太郎",
    },
    {
        "prompt": "富士山は3776mで、北岳は3193m。高さの差は何m？",
        "expected_mode": "QuantDiff",
        "expected_answer": "583m",
    },
    {
        "prompt": "佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？",
        "expected_mode": "Transfer",
        "expected_answer": "開発",
    },
    {
        "prompt": "赤、青、緑が対象で、赤は除外、青も除外。残るのは？",
        "expected_mode": "Elimination",
        "expected_answer": "緑",
    },
    {
        "prompt": "富士山の標高は3776mで、北岳の標高は3193m。富士山と北岳の高さの違いってどれくらい？",
        "expected_mode": "QuantDiff",
        "expected_answer": "583m",
    },
    {
        "prompt": "鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？",
        "expected_mode": "Syllogism",
        "expected_answer": "はい",
    },
    {
        "prompt": "甲は乙より重く、乙は丙より重く、丙は甲より重い。一番重いのは誰？",
        "expected_mode": "ContradictionCycle",
        "expected_answer": "矛盾",
    },
    {
        "prompt": "リンゴはミカンより大きく、ブドウはミカンより小さい。一番大きいのはどれ？",
        "expected_mode": "OrderExtreme",
        "expected_answer": "リンゴ",
    },
    {
        "prompt": "A塔は150mで、B塔は95m、C塔は210m。一番高いのはどれ？",
        "expected_mode": "QuantCompare",
        "expected_answer": "C塔",
    },
    {
        "prompt": "東京駅は10km、新宿駅は18km、渋谷駅は14km。距離が一番短いのはどれ？",
        "expected_mode": "QuantCompare",
        "expected_answer": "東京駅",
    },
    {
        "prompt": "青木の配属は札幌で、石川は青木と同期だ。石川の配属は？",
        "expected_mode": "Transfer",
        "expected_answer": "札幌",
    },
    {
        "unlabeled": "春は夏よりノビい。夏は春よりチヂい。",
        "prompt": "甲は乙よりノビい。丙は乙よりチヂい。一番チヂいのは誰？",
        "expected_mode": "OrderExtreme",
        "expected_answer": "丙",
    },
]


def split_of(text: str) -> str:
    h = int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return "val" if (h % 7) == 0 else "train"


def build_seq(parts: list[tuple[str, str]]) -> tuple[str, list[int]]:
    text_chunks = []
    tags = []
    for tok, role in parts:
        if not tok:
            continue
        b_tag, i_tag = ROLE_MAP[role]
        text_chunks.append(tok)
        for idx, _ch in enumerate(tok):
            tags.append(b_tag if idx == 0 else i_tag)
    text = "".join(text_chunks)
    assert len(text) == len(tags)
    return text, tags


def main():
    rng = random.Random(20261128)
    bench = json.loads((ROOT / "rust_bench_v182/bench_320.json").read_text(encoding="utf-8"))
    v240_probes = json.loads((ROOT / "v240_reasoner_core/unseen_probes.json").read_text(encoding="utf-8"))

    (OUT_DIR / "unseen_syntax_probes.json").write_text(
        json.dumps(UNSEEN_SYNTAX_PROBES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    denylist = set()
    for c in bench["cases"]:
        denylist.add(c["prompt"].strip())
        for s in re.split(r"[。\n]", c["prompt"]):
            if s.strip():
                denylist.add(s.strip())
        for t in c.get("turns", []):
            denylist.add(t.strip())
    for p in v240_probes:
        denylist.add(p["prompt"].strip())
    for p in UNSEEN_SYNTAX_PROBES:
        denylist.add(p["prompt"].strip())
        if "unlabeled" in p:
            denylist.add(p["unlabeled"].strip())

    # Forbidden entities/predicates that must NEVER enter syntax training
    forbidden_tokens = {
        "富士山", "北岳", "槍ヶ岳", "朝塔", "夕塔", "スイカ", "ブドウ", "リンゴ", "ミカン", "イチゴ",
        "佐藤", "田中", "太郎", "花子", "次郎", "健太", "青木", "石川", "セラ", "ネラ",
        "追跡キー", "砂時計印", "ゾルい", "メクい", "ルゼい", "端点", "鯨", "鯱", "鮪", "哺乳類",
        "東京駅", "新宿駅", "渋谷駅", "A塔", "B塔", "C塔", "ノビい", "チヂい",
    }

    rows = []
    seen = set()

    def add(parts: list[tuple[str, str]]):
        text, tags = build_seq(parts)
        if not text or text in seen or text in denylist:
            return
        if any(ft in text for ft in forbidden_tokens):
            return
        seen.add(text)
        rows.append({
            "text": text,
            "tags": tags,
            "split": split_of(text),
        })

    # Disjoint synthetic / Japanese entities across multiple scripts
    kanji_ents = [
        "エベレスト", "白山", "立山", "筑波山", "高尾山", "阿蘇山", "桜島", "白川", "黒川",
        "吉川", "中村", "岡田", "藤田", "村上", "森", "鈴木", "高橋", "渡辺", "小林", "山本",
        "山田", "斉藤", "松田", "前田", "上原", "下村", "東山", "西村", "南雲", "北野",
        "P社", "Q社", "R社", "X機", "Y機", "Z機", "D館", "E館", "F館",
        "青葉", "紅葉", "朝霧", "夕凪", "銀貨", "銅貨", "金貨",
        "八ヶ岳", "駒ヶ岳", "市ヶ谷", "霞ヶ関", "鶴ヶ岡", "四ッ谷",
    ]
    short_kanji = ["甲", "乙", "丙", "丁", "戌", "己", "春", "夏", "秋", "冬", "紫", "橙", "紺", "黄", "白", "黒", "銀", "銅", "鉄"]
    kata_pool = [
        "アオイ", "ミドリ", "アカネ", "イワシ", "サバ", "アジ", "ツバメ", "ワシ", "ハヤブサ", "クジラ",
        "イルカ", "パンダ", "コアラ", "キリン", "シマウマ", "カラス", "スズメ", "ツル", "カメ", "シカ",
        "ハルカ", "カナタ", "ミナト", "ヒカリ", "ソラ", "リク", "カイ", "レン", "ユイ", "アキ",
    ]
    # Remove any accidental overlap with forbidden tokens
    kata_pool = [k for k in kata_pool if not any(ft in k for ft in forbidden_tokens)]
    ascii_pool = ["K1", "K2", "P1", "P2", "P3", "V1", "V2", "M1", "M2", "N1", "N2", "N3"]
    ascii_1char = ["E", "F", "G", "H", "J", "K", "L", "M", "N", "T", "U", "V", "W", "Z"]
    all_ents = kanji_ents + kata_pool + short_kanji + ascii_pool + ascii_1char

    pred_pairs = [
        ("重い", "重く", "軽い", "軽く"),
        ("速い", "速く", "遅い", "遅く"),
        ("高い", "高く", "低い", "低く"),
        ("大きい", "大きく", "小さい", "小さく"),
        ("長い", "長く", "短い", "短く"),
        ("強い", "強く", "弱い", "弱く"),
        ("広い", "広く", "狭い", "狭く"),
        ("古い", "古く", "新しい", "新しく"),
        ("深い", "深く", "浅い", "浅く"),
        ("安い", "安く", "高い", "高く"),
    ]
    verb_pairs = [
        ("勝った", "負けた", "に"),
        ("上回った", "下回った", "を"),
        ("抜いた", "遅れた", "を"),
    ]
    rel_pairs = [
        ("上司だ", "部下だ", "の"),
        ("先輩だ", "後輩だ", "の"),
        ("兄だ", "弟だ", "の"),
        ("東だ", "西だ", "の"),
    ]
    alien_adjs = ["ガボい", "トメい", "ヌマい", "ヒケい", "ラピい", "relAした", "relBした"]

    attrs_num = [
        ("標高", "m"), ("高さ", "m"), ("距離", "km"), ("長さ", "cm"),
        ("価格", "円"), ("値段", "円"), ("重さ", "kg"), ("速さ", "km"), ("容量", "L"),
    ]
    attrs_assign = [
        "担当", "勤務地", "出身", "配属", "配属先", "拠点", "専門", "役割", "部門", "教室",
        "メモ先", "ルタ先", "カポ地", "トソ系", "ナミ係", "ペコ番",
    ]
    sym_adjs = ["同じ", "同一の", "共通の"]
    sym_nouns = ["同期だ", "同僚だ", "相棒だ", "仲間だ", "メナだ", "カポだ", "トソだ", "ルキだ", "ヌハだ", "パネだ"]

    # =========================================================================
    # 1. Key-value ledger sequences (`K＝V` / `K＝`) across Kata, ASCII, Kanji, mixed scripts
    # =========================================================================
    ledger_keys = kata_pool + ascii_pool + ascii_1char + [
        "記録キー", "確認キー", "管理タグ", "合図印", "標識番号", "暗号名", "白山", "立山", "甲", "乙"
    ]
    ledger_vals = kata_pool + ascii_pool + [
        "青石印", "月光印", "星影紋", "白銀札", "紅玉印", "金剛紋", "蒼空板", "朱雀印", "緑青札"
    ]
    for _ in range(1300):
        pool_use = ledger_keys if rng.random() < 0.55 else (kata_pool + ascii_pool)
        val_use = ledger_vals if rng.random() < 0.55 else (kata_pool + ascii_pool)
        k1, k2 = rng.sample(pool_use, 2)
        v1, v2 = rng.sample(val_use, 2)
        sep = rng.choice(["。", "\n"])
        sym = rng.choice(["＝", "="])
        parts = []
        if rng.random() < 0.65:
            parts.extend([("台帳", "ENT"), (sep, "SEP")])
        parts.extend([(k1, "ENT"), (sym, "FUNC"), (v1, "ENT"), (sep, "SEP")])
        if rng.random() < 0.5:
            parts.extend([(rng.choice(["点検", "休憩", "記録", "晴天"]), "ENT"), (sep, "SEP")])
        parts.extend([(k2, "ENT"), (sym, "FUNC"), (v2, "ENT"), (sep, "SEP")])
        if rng.random() < 0.5:
            parts.extend([(rng.choice(["点検", "記録"]), "ENT"), (sep, "SEP")])
        target = rng.choice([k1, k2])
        parts.extend([(target, "ENT"), (sym, "FUNC")])
        add(parts)
        add([(k1, "ENT"), (sym, "FUNC"), (v1, "ENT"), (sep, "SEP")])
        add([(k1, "ENT"), (sym, "FUNC"), (v1, "ENT")])
        add([(target, "ENT"), (sym, "FUNC")])

    # =========================================================================
    # 2. Quantitative clauses + comma-joined numeric clauses + questions
    # =========================================================================
    q_diff_suffixes = [
        [("高さ", "ATTR"), ("の", "FUNC"), ("差", "ATTR"), ("は", "FUNC"), ("何m", "PRED"), ("？", "FUNC")],
        [("標高差", "ATTR"), ("は", "FUNC"), ("何m", "PRED"), ("？", "FUNC")],
        [("値段", "ATTR"), ("の", "FUNC"), ("違い", "ATTR"), ("ってどれくらい", "PRED"), ("？", "FUNC")],
        [("何m違う", "PRED"), ("？", "FUNC")],
        [("いくら違う", "PRED"), ("？", "FUNC")],
        [("開きはどれくらい", "PRED"), ("？", "FUNC")],
    ]
    q_cmp_suffixes = [
        [("一番", "ATTR"), ("高い", "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC")],
        [("一番", "ATTR"), ("低い", "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC")],
        [("距離", "ATTR"), ("が", "FUNC"), ("一番", "ATTR"), ("短い", "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC")],
        [("どちらが", "FUNC"), ("安い", "PRED"), ("？", "FUNC")],
        [("最も", "ATTR"), ("長い", "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC")],
    ]
    for _ in range(1400):
        e1, e2, e3 = rng.sample(all_ents, 3)
        attr, u = rng.choice(attrs_num)
        v1 = rng.randint(10, 4500)
        v2 = rng.randint(10, 4500)
        v3 = rng.randint(10, 4500)
        sep = rng.choice(["。", "\n"])
        mode = rng.randint(0, 4)
        if mode == 0:
            # Bare 。 separated
            parts = [
                (e1, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), (sep, "SEP"),
                (e2, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), (sep, "SEP"),
            ]
        elif mode == 1:
            # Comma `で、` joined attribute measurements
            parts = [
                (e1, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), ("で、", "SEP"),
                (e2, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), (sep, "SEP"),
            ]
        elif mode == 2:
            # Bare measurement comma `で、` joined
            parts = [
                (e1, "ENT"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), ("で、", "SEP"),
                (e2, "ENT"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), (sep, "SEP"),
            ]
        elif mode == 3:
            # 3 entities comma `、` / `で、`
            conn = rng.choice(["で、", "、"])
            parts = [
                (e1, "ENT"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), (conn, "SEP"),
                (e2, "ENT"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), ("、", "SEP"),
                (e3, "ENT"), ("は", "FUNC"), (f"{v3}{u}", "NUM"), (sep, "SEP"),
            ]
        else:
            # Mixed + query mention
            parts = [
                (e1, "ENT"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), ("で、", "SEP"),
                (e2, "ENT"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), (sep, "SEP"),
            ]
        q_suf = rng.choice(q_diff_suffixes if mode in (0, 1, 2, 4) and rng.random() < 0.6 else q_cmp_suffixes)
        if rng.random() < 0.5:
            parts.extend([(e1, "ENT"), ("と", "FUNC"), (e2, "ENT"), ("の", "FUNC")] + q_suf)
        elif q_suf == q_diff_suffixes[3] or q_suf == q_diff_suffixes[4]:
            parts.extend([(e1, "ENT"), ("と", "FUNC"), (e2, "ENT"), ("は", "FUNC")] + q_suf)
        else:
            parts.extend(q_suf)
        add(parts)
        # Single premise too
        add([(e1, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), (sep, "SEP")])
        if rng.random() < 0.5:
            add([(e1, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v1}{u}", "NUM")])
        add([(e2, "ENT"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), ("で、", "SEP")])

    # =========================================================================
    # 3. Order / Transitivity / Cycle (single + comma-joined `、` + alien + unlabeled)
    # =========================================================================
    for _ in range(1800):
        a, b, c = rng.sample(all_ents, 3)
        p1, p1_ku, p2, p2_ku = rng.choice(pred_pairs)
        sep = rng.choice(["。", "\n"])
        style = rng.randint(0, 6)
        if style == 0:
            # Standard より
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("より", "FUNC"), (p1, "PRED"), (sep, "SEP"),
                (b, "ENT"), ("は", "FUNC"), (c, "ENT"), ("より", "FUNC"), (p1 if rng.random() < 0.6 else p2, "PRED"), (sep, "SEP"),
                ("一番", "ATTR"), (p1, "PRED"), ("のは", "FUNC"), (rng.choice(["誰", "どれ"]), "ATTR"), ("？", "FUNC"),
            ]
        elif style == 1:
            # Comma-joined `〜く、〜`
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("より", "FUNC"), (p1_ku, "PRED"), ("、", "SEP"),
                (b, "ENT"), ("は", "FUNC"), (c, "ENT"), ("より", "FUNC"), (p1 if rng.random() < 0.5 else p2, "PRED"), (sep, "SEP"),
                (rng.choice(["一番", "最も"]), "ATTR"), (p1, "PRED"), ("のは", "FUNC"), (rng.choice(["誰", "どれ"]), "ATTR"), ("？", "FUNC"),
            ]
        elif style == 2:
            # 3-clause cycle comma-joined `〜く、〜く、〜`
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("より", "FUNC"), (p1_ku, "PRED"), ("、", "SEP"),
                (b, "ENT"), ("は", "FUNC"), (c, "ENT"), ("より", "FUNC"), (p1_ku, "PRED"), ("、", "SEP"),
                (c, "ENT"), ("は", "FUNC"), (a, "ENT"), ("より", "FUNC"), (p1, "PRED"), (sep, "SEP"),
                ("一番", "ATTR"), (p1, "PRED"), ("のは", "FUNC"), ("誰", "ATTR"), ("？", "FUNC"),
            ]
        elif style == 3:
            # の方が ... よりも pattern (without は)
            parts = [
                (a, "ENT"), ("の方が", "FUNC"), (b, "ENT"), ("よりも", "FUNC"), (p1, "PRED"), (sep, "SEP"),
                (b, "ENT"), ("の方が", "FUNC"), (c, "ENT"), ("よりも", "FUNC"), (p1, "PRED"), (sep, "SEP"),
                (rng.choice([("一番", "ATTR"), ("最も", "ATTR")]),),
            ]
            parts.pop()
            parts.extend([("一番", "ATTR"), (p1, "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC")])
        elif style == 4:
            # Verb / Relational particle (`に`, `を`, `の`)
            if rng.random() < 0.5:
                v1, v2, cm = rng.choice(verb_pairs)
            else:
                v1, v2, cm = rng.choice(rel_pairs)
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), (cm, "FUNC"), (v1, "PRED"), (sep, "SEP"),
                (c, "ENT"), ("は", "FUNC"), (b, "ENT"), (cm, "FUNC"), (v2, "PRED"), (sep, "SEP"),
                ("一番", "ATTR"), (v1, "PRED"), ("のは", "FUNC"), ("誰", "ATTR"), ("？", "FUNC"),
            ]
        elif style == 5:
            # Alien predicates + unlabeled symmetric swap + rank queries
            ap1, ap2 = rng.sample(alien_adjs, 2)
            cm = "を" if "した" in ap1 else "より"
            if cm == "を" and "した" not in ap2:
                ap2 = "relBした"
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), (cm, "FUNC"), (ap1, "PRED"), (sep, "SEP"),
                (b, "ENT"), ("は", "FUNC"), (a, "ENT"), (cm, "FUNC"), (ap2, "PRED"), (sep, "SEP"),
            ]
            add([
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), (cm, "FUNC"), (ap1, "PRED"), (sep, "SEP"),
                (c, "ENT"), ("は", "FUNC"), (b, "ENT"), (cm, "FUNC"), (ap2, "PRED"), (sep, "SEP"),
                ("一番", "ATTR"), (ap2, "PRED"), ("のは", "FUNC"), ("誰", "ATTR"), ("？", "FUNC"),
            ])
        else:
            # Rank / Pairwise / Extreme / Unknown-token order queries
            unk_q = rng.choice(["終端", "境目", "頂点", "末尾", "先端"])
            rank_q = rng.choice([
                [(p1, "PRED"), ("順で", "FUNC"), ("2位", "ATTR"), ("は", "FUNC"), ("誰", "ATTR"), ("？", "FUNC")],
                [(p1, "PRED"), ("順で", "FUNC"), ("二番目", "ATTR"), ("は", "FUNC"), ("誰", "ATTR"), ("？", "FUNC")],
                [(p1, "PRED"), ("順で", "FUNC"), ("3位", "ATTR"), ("は", "FUNC"), ("誰", "ATTR"), ("？", "FUNC")],
                [(p1, "PRED"), ("順で", "FUNC"), ("最後", "ATTR"), ("は", "FUNC"), ("誰", "ATTR"), ("？", "FUNC")],
                [(p1, "PRED"), ("順で", "FUNC"), ("最初", "ATTR"), ("は", "FUNC"), ("誰", "ATTR"), ("？", "FUNC")],
                [(a, "ENT"), ("は", "FUNC"), (c, "ENT"), ("より", "FUNC"), (p1, "PRED"), ("ですか？", "FUNC")],
                [(unk_q, "ATTR"), ("は？", "FUNC")],
            ])
            parts = [
                (a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("より", "FUNC"), (p1, "PRED"), (sep, "SEP"),
                (b, "ENT"), ("は", "FUNC"), (c, "ENT"), ("より", "FUNC"), (p1, "PRED"), (sep, "SEP"),
            ] + rank_q
            add([(unk_q, "ATTR"), ("は？", "FUNC")])
        add(parts)
        if rng.random() < 0.35:
            add([(a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("より", "FUNC"), (p1, "PRED")])

    # =========================================================================
    # 4. Attribute assignment & Alias transfer (including comma `で、`)
    # =========================================================================
    for _ in range(1300):
        a, b, v, v2 = rng.sample(all_ents, 4)
        attr = rng.choice(attrs_assign)
        sep = rng.choice(["。", "\n"])
        st = rng.randint(0, 3)
        if st == 0:
            sym = rng.choice(sym_adjs)
            parts = [
                (a, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v, "ENT"), (rng.choice([sep, "で、"]), "SEP"),
                (b, "ENT"), ("は", "FUNC"), (a, "ENT"), ("と", "FUNC"), (sym, "PRED"), (attr, "ATTR"), (sep, "SEP"),
                (b, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (rng.choice(["何？", "どこ？", "を教えて。"]), "FUNC"),
            ]
        elif st == 1:
            sn = rng.choice(sym_nouns)
            parts = [
                (a, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v, "ENT"), ("で、", "SEP"),
                (b, "ENT"), ("は", "FUNC"), (a, "ENT"), ("と", "FUNC"), (sn, "PRED"), (sep, "SEP"),
                (b, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), ("？", "FUNC"),
            ]
        elif st == 2:
            # Inverted alias phrasing: "Aと同じ部署なのはB"
            parts = [
                (a, "ENT"), ("と", "FUNC"), ("同じ", "PRED"), (attr, "ATTR"), ("なのは", "FUNC"), (b, "ENT"), (sep, "SEP"),
                (a, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v, "ENT"), (sep, "SEP"),
                (b, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("を", "FUNC"), ("教えて", "PRED"), (sep, "SEP"),
            ]
        else:
            # Right-hand alias / attribute value overlap
            parts = [
                (a, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v, "ENT"), (sep, "SEP"),
                (b, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v2, "ENT"), (sep, "SEP"),
                (v, "ENT"), ("の", "FUNC"), (attr + "者", "ATTR"), ("は", "FUNC"), (b, "ENT"), ("と", "FUNC"), ("同じ", "PRED"), (sep, "SEP"),
                (v, "ENT"), ("の", "FUNC"), (attr + "者", "ATTR"), ("は？", "FUNC"),
            ]
        add(parts)
        if rng.random() < 0.35:
            add([(a, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (v, "ENT")])

    # Focused structural reinforcement for 1-char uppercase ASCII entities in order/cycle chains
    for _ in range(300):
        e1, e2, e3, e4 = rng.sample(ascii_1char, 4)
        p1, _, _, _ = rng.choice(pred_pairs)
        add([
            (e1, "ENT"), ("は", "FUNC"), (e2, "ENT"), ("より", "FUNC"), (p1, "PRED"), ("。", "SEP"),
            (e2, "ENT"), ("は", "FUNC"), (e3, "ENT"), ("より", "FUNC"), (p1, "PRED"), ("。", "SEP"),
            (e3, "ENT"), ("は", "FUNC"), (e4, "ENT"), ("より", "FUNC"), (p1, "PRED"), ("。", "SEP"),
            (e4, "ENT"), ("は", "FUNC"), (e1, "ENT"), ("より", "FUNC"), (p1, "PRED"), ("。", "SEP"),
            ("一番", "ATTR"), (p1, "PRED"), ("のは", "FUNC"), (rng.choice(["誰", "どれ"]), "ATTR"), ("？", "FUNC"),
        ])

    # Focused structural reinforcement for mixed-script Kanji+ヶ/ッ entities & Katakana+Kanji attrs & Katakana+だ sym-aliases
    mixed_kanji = ["八ヶ岳", "駒ヶ岳", "市ヶ谷", "霞ヶ関", "鶴ヶ岡", "四ッ谷"]
    kata_attrs = ["メモ先", "ルタ先", "カポ地", "トソ系", "ナミ係", "ペコ番"]
    kata_syms = ["メナだ", "カポだ", "トソだ", "ルキだ", "ヌハだ", "パネだ"]
    for _ in range(350):
        m1, m2, m3 = rng.sample(mixed_kanji, 3)
        attr, u = rng.choice(attrs_num)
        v1, v2, v3 = rng.randint(50, 3500), rng.randint(50, 3500), rng.randint(50, 3500)
        add([
            (m1, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v1}{u}", "NUM"), ("。", "SEP"),
            (m2, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v2}{u}", "NUM"), ("。", "SEP"),
            (m3, "ENT"), ("の", "FUNC"), (attr, "ATTR"), ("は", "FUNC"), (f"{v3}{u}", "NUM"), ("。", "SEP"),
            ("一番", "ATTR"), ("低い", "PRED"), ("のは", "FUNC"), ("どれ", "ATTR"), ("？", "FUNC"),
        ])
        a, b, c, v, v2s = rng.sample(all_ents, 5)
        ka = rng.choice(kata_attrs)
        ks = rng.choice(kata_syms)
        add([
            (a, "ENT"), ("の", "FUNC"), (ka, "ATTR"), ("は", "FUNC"), (v, "ENT"), ("。", "SEP"),
            (b, "ENT"), ("の", "FUNC"), (ka, "ATTR"), ("は", "FUNC"), (v2s, "ENT"), ("。", "SEP"),
            (c, "ENT"), ("は", "FUNC"), (a, "ENT"), ("と", "FUNC"), (ks, "PRED"), ("。", "SEP"),
            (c, "ENT"), ("の", "FUNC"), (ka, "ATTR"), ("は？", "FUNC"),
        ])
        add([
            (a, "ENT"), ("は", "FUNC"), (b, "ENT"), ("と", "FUNC"), (ks, "PRED"), ("。", "SEP"),
            (b, "ENT"), ("は", "FUNC"), (a, "ENT"), ("と", "FUNC"), (ks, "PRED"), ("。", "SEP"),
        ])

    # =========================================================================
    # 5. Syllogisms (`です`, `すべての`, `全員が...に属する/属し、`, distractors)
    # =========================================================================
    cls_names = ["爬虫類", "両生類", "魚類", "鳥類", "脊椎動物", "植物", "金属", "惑星", "楽器", "競技"]
    for _ in range(1100):
        x, y, z = rng.sample(all_ents + cls_names, 3)
        if x == y or y == z or x == z:
            continue
        sep = rng.choice(["。", "\n"])
        if rng.random() < 0.5:
            parts = [
                ("すべて", "ATTR"), ("の", "FUNC"), (y, "ENT"), ("は", "FUNC"), (z, "ENT"), ("です", "PRED"), (sep, "SEP"),
                (x, "ENT"), ("は", "FUNC"), (y, "ENT"), ("です", "PRED"), (sep, "SEP"),
                (x, "ENT"), ("は", "FUNC"), (z, "ENT"), ("ですか？", "FUNC"),
            ]
        else:
            parts = [
                (y, "ENT"), ("の", "FUNC"), ("全員", "ATTR"), ("が", "FUNC"), (z, "ENT"), ("に", "FUNC"),
                ("属し", "PRED"), ("、", "SEP") if rng.random() < 0.6 else ("属する", "PRED"),
            ]
            if parts[-1] == ("属する", "PRED"):
                parts.append((sep, "SEP"))
            parts.extend([
                (x, "ENT"), ("の", "FUNC"), ("所属", "ATTR"), ("は", "FUNC"), (y, "ENT"), (sep, "SEP"),
            ])
            if rng.random() < 0.35:
                parts.extend([
                    (rng.choice(["机の上", "窓辺", "庭", "棚の上"]), "ENT"), ("には", "FUNC"),
                    (rng.choice(["本", "時計", "鍵", "帽子"]), "ENT"), ("が", "FUNC"), ("ある", "PRED"), (sep, "SEP"),
                ])
            parts.extend([
                (x, "ENT"), ("を", "FUNC"), (z, "ENT"), ("の", "FUNC"), ("一員と言い切れる？", "PRED"),
            ])
        add(parts)

    # =========================================================================
    # 6. Pool & Inhibition (`・` and `,` enumeration `、` as FUNC vs clause `、` / `で、` as SEP)
    # =========================================================================
    short_pools = [
        ("金", "銀", "銅"), ("松", "竹", "梅"), ("春", "夏", "秋"),
        ("紫", "橙", "紺"), ("東", "西", "南"), ("鳥", "魚", "虫"),
        ("甲", "乙", "丙"), ("桜", "桃", "椿"), ("北山", "南山", "東山"),
        ("青葉", "紅葉", "朝霧"), ("P社", "Q社", "R社"), ("D館", "E館", "F館"),
    ]
    inh_preds = ["除外だ", "除外", "外れだ", "外れ", "無実だ", "ザビだ", "ではない", "には入っていない"]
    for _ in range(1300):
        p1, p2, p3 = rng.choice(short_pools) if rng.random() < 0.55 else tuple(rng.sample(all_ents, 3))
        if len({p1, p2, p3}) < 3:
            continue
        sep = rng.choice(["。", "\n"])
        style = rng.randint(0, 2)
        if style == 0:
            # Dot `・` pool + standard inhibition
            inh = rng.choice(inh_preds)
            parts = [
                (p1, "ENT"), ("・", "FUNC"), (p2, "ENT"), ("・", "FUNC"), (p3, "ENT"),
                ("が", "FUNC"), (rng.choice(["対象だ", "抽選だ", "容疑者だ", "ポクだ"]), "PRED"), (sep, "SEP"),
                (p1, "ENT"), ("は", "FUNC"), (inh, "PRED"), (sep, "SEP"),
                (p2, "ENT"), ("も" if rng.random() < 0.5 else "は", "FUNC"), (inh, "PRED"), (sep, "SEP"),
                (rng.choice([("残るのは", "PRED"), ("当たりは", "PRED"), ("犯人は", "ATTR")]),),
            ]
            parts.pop()
            parts.extend([("残るのは", "PRED"), ("？", "FUNC")])
        elif style == 1:
            # Comma-listed pool `A、B、C` (`、`=FUNC) + `で、` (`SEP`) + comma-separated inhibitions (`、`=SEP)
            pool_w = rng.choice(["対象", "抽選", "候補", "容疑者"])
            inh_s = rng.choice(["除外", "外れ", "無実"])
            parts = [
                (p1, "ENT"), ("、", "FUNC"), (p2, "ENT"), ("、", "FUNC"), (p3, "ENT"),
                ("が", "FUNC"), (pool_w, "PRED"), ("で、", "SEP"),
                (p1, "ENT"), ("は", "FUNC"), (inh_s, "PRED"), ("、", "SEP"),
                (p2, "ENT"), ("も", "FUNC"), (inh_s, "PRED"), (sep, "SEP"),
                (rng.choice(["残るのは", "当たりは", "残りは"]), "PRED"), ("？", "FUNC"),
            ]
        else:
            # `A・B・Cの箱のどれかに鍵がある` OR `A・B・Cのどれかが担当。Aではない。`
            if rng.random() < 0.5:
                mid = [("の", "FUNC"), ("箱", "ATTR")] if rng.random() < 0.6 else []
                parts = [
                    (p1, "ENT"), ("・", "FUNC"), (p2, "ENT"), ("・", "FUNC"), (p3, "ENT"),
                    ("の", "FUNC"), ("箱", "ATTR"), ("のどれかに鍵がある", "PRED"), (sep, "SEP"),
                    (p1, "ENT"), *mid, ("には入っていない", "PRED"), (sep, "SEP"),
                    (p2, "ENT"), *mid, ("にも入っていない", "PRED"), (sep, "SEP"),
                    ("鍵", "ATTR"), ("はどの", "FUNC"), ("箱", "ATTR"), ("？", "FUNC"),
                ]
            else:
                parts = [
                    (p1, "ENT"), ("・", "FUNC"), (p2, "ENT"), ("・", "FUNC"), (p3, "ENT"),
                    ("のどれかが担当", "PRED"), (sep, "SEP"),
                    (p1, "ENT"), ("ではない", "PRED"), (sep, "SEP"),
                    (p2, "ENT"), ("ではない", "PRED"), (sep, "SEP"),
                    ("担当", "ATTR"), ("は誰？", "FUNC"),
                ]
        add(parts)

    rng.shuffle(rows)
    out_jsonl = OUT_DIR / "syntax_bio.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tag_counts = Counter()
    for r in rows:
        for t in r["tags"]:
            tag_counts[TAG_NAMES[t]] += 1

    manifest = {
        "sequences": len(rows),
        "train_sequences": sum(1 for r in rows if r["split"] == "train"),
        "val_sequences": sum(1 for r in rows if r["split"] == "val"),
        "total_chars": sum(len(r["tags"]) for r in rows),
        "tag_counts": dict(sorted(tag_counts.items())),
        "unseen_syntax_probes_count": len(UNSEEN_SYNTAX_PROBES),
        "denylist_hits": sum(1 for r in rows if r["text"] in denylist),
        "probe_overlap": sum(1 for r in rows if any(r["text"] == p["prompt"] for p in UNSEEN_SYNTAX_PROBES)),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
