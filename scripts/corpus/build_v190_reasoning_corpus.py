#!/usr/bin/env python3
"""
ENNE v19.0 推論コーパス生成・漏洩検査スクリプト。

方針:
  1. SFTデータセット（sft_raw/）からは「ユーザー命令文（人:）」側のみを抽出し、
     他社LLMの回答文（答:）は一切使用しない。
  2. 12の推論モード × 任意長実体鎖（N=0..6） × 多様な日本語命令表現・推論展開スタイルを組み合わせ、
     実体・数値・結論をキャッシュポインター (<P>, <V>, <ANS>) で抽象化した
     「命令 → 思考モード → 推論過程」エピソードを生成する。
  3. bench_320.json および実SQLite Wikipediaテスト（富士山・北岳・槍ヶ岳 等）に含まれる
     全実体・全質問文との衝突をプログラムで全件監査し、衝突0件を保証する。
"""
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ENNE_ROOT", "."))
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "v190_reasoning_core"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load forbidden test entities & prompts from bench_320.json + real Wiki test
BENCH_PATH = ROOT / "rust_bench_v182" / "bench_320.json"
bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
forbidden_prompts = set()
forbidden_tokens = {
    "富士山", "北岳", "槍ヶ岳", "583.12m", "596.12m", "3776.12m", "3193m", "3180m",
    "東京タワー", "スカイツリー", "エベレスト", "阿蘇山", "剣岳",
    "健太", "太郎", "花子", "次郎", "佐藤", "鈴木", "高橋", "田中", "山本", "伊藤",
    "スイカ", "ブドウ", "リンゴ", "ミカン", "イチゴ", "メロン", "バナナ",
    "東京", "大阪", "名古屋", "札幌", "福岡", "仙台", "金沢", "京都",
    "火星", "金星", "木星", "地球", "新幹線", "砂時計印",
}
for c in bench["cases"]:
    forbidden_prompts.add(c["prompt"].strip())
    g = c["gold"].strip()
    if len(g) >= 2 and g not in ("はい", "いいえ", "矛盾", "不明", "判断不可"):
        forbidden_tokens.add(g)
    # Extract 2+ char katakana spans from turns and prompt for entity overlap filtering
    for txt in list(c.get("turns", [])) + [c["prompt"]]:
        for m in re.findall(r"[ァ-ヶー]{2,}", txt):
            forbidden_tokens.add(m)

# 2. Mine genuine human Japanese instruction prefixes/suffixes ONLY from prompt side of sft_raw
human_openers = []
human_closers = []
sft_files = [
    ROOT / "sft_raw" / "oasst1-21k-ja.jsonl",
    ROOT / "sft_raw" / "databricks-dolly-15k-ja.json",
    ROOT / "sft_raw" / "magpie-sft-v1.0.jsonl",
    ROOT / "sft_raw" / "ja_conv_gpt3.5_100K_fix.jsonl",
]
opener_pat = re.compile(r"^(こんにちは[！。、]?|すみません[、。]?|教えてください[。、]?|質問です[。、]?|ちょっと確認したいのですが[、]?|次の条件[をに][^。]{0,16}[。、]?|以下の[情報条件内容記述文章][^。]{0,16}[。、]?)")
closer_pat = re.compile(r"(教えてください[。？]?|教えて[。？]?|答えてください[。？]?|分かりますか？|わかりますか？|どれですか？|何ですか？|どちらですか？|いくつですか？|説明してください[。？]?|整理してください[。？]?|確認してください[。？]?)$")

mined_prompts = []
for sf in sft_files:
    if not sf.exists():
        continue
    if sf.suffix == ".json":
        try:
            for obj in json.loads(sf.read_text(encoding="utf-8")):
                q = (obj.get("instruction") or "").strip()
                if 6 <= len(q) <= 80:
                    mined_prompts.append(q)
        except Exception:
            pass
    else:
        with sf.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 15000:
                    break
                try:
                    obj = json.loads(line)
                    for m in obj.get("conversations", []):
                        role = m.get("role") or m.get("from")
                        if role in ("user", "human"):
                            q = (m.get("content") or m.get("value") or "").strip()
                            if 6 <= len(q) <= 80:
                                mined_prompts.append(q)
                            break
                except Exception:
                    pass

opener_counts = {}
closer_counts = {}
for q in mined_prompts:
    mo = opener_pat.search(q)
    if mo:
        s = mo.group(1)
        if not any(ft in s for ft in forbidden_tokens):
            opener_counts[s] = opener_counts.get(s, 0) + 1
    mc = closer_pat.search(q)
    if mc:
        s = mc.group(1)
        closer_counts[s] = closer_counts.get(s, 0) + 1

mined_openers = [k for k, _ in sorted(opener_counts.items(), key=lambda x: -x[1])[:24]]
if not mined_openers:
    mined_openers = ["", "質問です。", "次の条件をもとに、", "以下の内容から、"]
mined_openers = ["", "", "", ""] + mined_openers  # weight plain prompt naturally

# 3. Disjoint synthetic entity pools (100% checked against forbidden_tokens)
RAW_ENTITIES = [
    "青嵐", "白波", "琥珀", "翠玉", "月影", "星屑", "朝霧", "夕凪", "銀嶺", "蒼穹",
    "雪華", "露草", "風鈴", "雲海", "霞草", "水紋", "砂丘", "白帆", "藍晶", "珊瑚",
    "碧空", "緋色", "深緑", "黄昏", "暁光", "氷河", "草原", "渓谷", "岬灯", "松風",
    "竹林", "梅香", "桜並", "楓葉", "杉並", "桧山", "柳川", "藤棚", "桐箱", "楠木",
    "黒曜", "白磁", "青銅", "真珠", "翡翠", "瑪瑙", "柘榴", "胡桃", "杏子", "柚子",
    "栗山", "柿木", "梨園", "桃源", "早蕨", "若葉", "秋霜", "冬嶺", "春霞", "夏雲",
    "小鳥", "海燕", "白鷺", "千鳥", "蛍火", "蝉時", "蜻蛉", "鈴虫", "駒鳥", "鶴橋",
    "東雲", "西風", "南星", "北極", "中洲", "上流", "下町", "内海", "外洋", "高台",
    "バルン", "メルタ", "ソルナ", "リベル", "ヴァル", "ノクス", "テラム", "ルミナ",
    "オルカ", "ゼフィ", "フェリ", "ミラン", "シエル", "カノン", "ロアン", "エリス",
    "甲組", "乙組", "丙組", "丁組", "戊組", "己組", "一号", "二号", "三号", "四号",
    "白峰", "黒岳", "赤城", "青嶺", "翠岳", "銀峰", "霧ヶ峰", "雲取", "笠岳", "鷲羽",
]
ENTITIES = [e for e in RAW_ENTITIES if e not in forbidden_tokens and not any(ft in e for ft in forbidden_tokens)]
assert len(ENTITIES) >= 80, f"Too few safe entities: {len(ENTITIES)}"

RAW_VALUES_TEXT = [
    "翡翠", "青銅", "白磁", "真珠", "琥珀", "黒曜", "銀貨", "木彫", "和紙", "絹糸",
    "北館", "南館", "東棟", "西棟", "本館", "別館", "港町", "山里", "湖畔", "高原",
    "総務", "企画", "研究", "法務", "資材", "品質", "運用", "保守", "調査", "広報部",
]
VALUES_TEXT = [v for v in RAW_VALUES_TEXT if v not in forbidden_tokens and not any(ft in v for ft in forbidden_tokens)]

CATS = [
    "鉱物", "植物", "器物", "楽器", "織物", "書物", "建築", "船舶", "星雲", "結晶",
    "果実", "香草", "鳥類", "魚類", "標本", "道具", "素材", "装置", "記録", "領域",
]

# Relations (both natural Japanese & alien synthetic predicates for OOD generalization)
ORDER_PAIRS = [
    ("大きい", "小さい", "一番大きい", "一番小さい"),
    ("高い", "低い", "一番高い", "一番低い"),
    ("重い", "軽い", "一番重い", "一番軽い"),
    ("速い", "遅い", "一番速い", "一番遅い"),
    ("長い", "短い", "一番長い", "一番短い"),
    ("古い", "新しい", "一番古い", "一番新しい"),
    ("強い", "弱い", "一番強い", "一番弱い"),
    ("広い", "狭い", "一番広い", "一番狭い"),
    ("深い", "浅い", "一番深い", "一番浅い"),
    ("ゾルい", "メクい", "一番ゾルい", "一番メクい"),
    ("ザミい", "ノムい", "一番ザミい", "一番ノムい"),
    ("ルゼい", "タポい", "一番ルゼい", "一番タポい"),
]

QUANT_ATTRS = [
    ("標高", "m", 400, 4200, "高い", "低い"),
    ("高さ", "m", 20, 650, "高い", "低い"),
    ("全長", "km", 5, 450, "長い", "短い"),
    ("距離", "km", 2, 180, "遠い", "近い"),
    ("価格", "円", 120, 9800, "高い", "安い"),
    ("重さ", "g", 50, 5000, "重い", "軽い"),
    ("身長", "cm", 145, 198, "高い", "低い"),
    ("面積", "km2", 10, 950, "広い", "狭い"),
    ("容量", "ml", 100, 2500, "多い", "少ない"),
    ("温度", "度", 5, 95, "高い", "低い"),
]

# 4. Compositional Japanese reasoning discourse patterns (multi-register, multi-structure)
# Note: Only <P>, <V>, <ANS> pointer markers are used; zero domain entities are hardcoded in reasoning.

def render_p_list(n: int, style: int) -> str:
    if n <= 0:
        return ""
    if style == 0:
        return "、".join(["<P>"] * n)
    elif style == 1:
        return "と".join(["<P>"] * n)
    else:
        return "→".join(["<P>"] * n)

def render_pv_list(n: int, style: int) -> str:
    items = ["<P>（<V>）"] * n
    return "と".join(items) if style == 1 else "、".join(items)

def build_reasoning(mode: str, n_p: int, n_v: int, rng: random.Random) -> str:
    # Choose from 6 Japanese discourse registers/structures
    reg = rng.randrange(6)
    p_sep = rng.choice([0, 1, 2]) if mode in ("OrderExtreme", "OrderRank", "OrderPairwise", "Syllogism", "ContradictionCycle", "Transfer") else rng.choice([0, 1])
    plist = render_p_list(n_p, p_sep)
    pvlist = render_pv_list(n_v, rng.choice([0, 1]))

    if mode == "Copy":
        opts = [
            f"記録を照合します。対象は{plist}で、対応する値は<ANS>です。",
            f"与えられた対応関係を確認すると、{plist}に結び付く記録は<ANS>となります。",
            f"まず{plist}の記載を参照します。そのまま読み取れる内容は<ANS>です。",
            f"{plist}の登録内容を辿った結果、該当する値は<ANS>と分かります。",
            f"前提の記録から{plist}を引き当てます。結論は<ANS>です。",
            f"{plist}について記録を確かめると、対応先は<ANS>になっています。",
        ]
        return opts[reg]

    if mode == "OrderExtreme":
        opts = [
            f"まず順序関係を一列に並べます。根拠となる並びは{plist}であり、端に位置する<ANS>が答えになります。",
            f"大小の比較を順に辿ると、{plist}という序列が得られます。したがって最も端にあるのは<ANS>です。",
            f"条件を整理すると、{plist}の順に関係がつながります。ここから求める対象は<ANS>と決まります。",
            f"比較の連鎖を確認します。{plist}の並びから端点を特定すると、結果は<ANS>となります。",
            f"{plist}の順序を端から検証していくと、条件を満たすのは<ANS>です。",
            f"順番に比較を重ねます。関係は{plist}と連なるため、結論は<ANS>です。",
        ]
        return opts[reg]

    if mode == "OrderRank":
        opts = [
            f"順位を特定するため全体を並べ替えます。{plist}の順になるため、指定された位置は<ANS>です。",
            f"序列を端から数えて確認します。並びは{plist}であり、該当する順位は<ANS>となります。",
            f"まず比較条件から{plist}という順列を作ります。位置を数え上げると答えは<ANS>です。",
            f"{plist}の並び順を基準にして順番を辿ると、求める対象は<ANS>に当たります。",
            f"順序の列を整理すると{plist}です。指定の番目を読み取ると<ANS>になります。",
            f"比較結果を一直線に並べた{plist}から順位を引くと、結論は<ANS>です。",
        ]
        return opts[reg]

    if mode == "OrderPairwise":
        opts = [
            f"二者間の大小関係を連鎖から調べます。{plist}の並びにより位置関係が定まるため、答えは<ANS>です。",
            f"比較の経路を{plist}と辿ることで両者の優劣が分かります。したがって結果は<ANS>となります。",
            f"まず{plist}の順序を確認します。この並びから二者の関係を読み取ると<ANS>です。",
            f"与えられた比較を繋ぐと{plist}の順になります。両者を照らし合わせると<ANS>と判断できます。",
            f"{plist}という序列上で二者の位置を比べた結果、結論は<ANS>です。",
            f"条件を連結して{plist}の順を得ました。これより判定結果は<ANS>になります。",
        ]
        return opts[reg]

    if mode == "Transfer":
        opts = [
            f"別名や同一視のつながりを順に辿ります。{plist}の経路で属性が引き継がれるため、答えは<ANS>です。",
            f"対応の連鎖を確認すると、{plist}と繋がっています。元の値を引き継ぐと結果は<ANS>となります。",
            f"まず{plist}の参照関係を追跡します。同じ属性を共有しているため、結論は<ANS>です。",
            f"条件より{plist}が順に結び付いています。この転移関係を辿ると<ANS>が得られます。",
            f"{plist}の対応先を段階的に確認していくと、最終的に<ANS>へ行き着きます。",
            f"同一関係の鎖である{plist}を辿って値を引き継ぐと、求める内容は<ANS>です。",
        ]
        return opts[reg]

    if mode == "Elimination":
        opts = [
            f"消去法で選択肢を絞り込みます。条件から外れる{plist}を除くと、残る<ANS>が答えになります。",
            f"まず該当しない要素を確認します。{plist}が除外されるため、最終的に残るのは<ANS>です。",
            f"前提の条件で{plist}が候補から外れます。残った一つを特定すると結果は<ANS>となります。",
            f"除外条件を順に適用します。{plist}を取り除いた結果、結論は<ANS>に決まります。",
            f"{plist}は条件を満たさないため除きます。消去の末に残る対象は<ANS>です。",
            f"一つずつ外していくと{plist}が消え、条件に合うものとして<ANS>が残ります。",
        ]
        return opts[reg]

    if mode == "Syllogism":
        opts = [
            f"分類の包含関係を段階的に辿ります。{plist}の順で上位概念へ繋がるため、判定は<ANS>です。",
            f"三段論法で所属関係を確認します。{plist}という連鎖が成り立つことから、結論は<ANS>となります。",
            f"まず{plist}の所属経路を整理します。包含の推移により答えは<ANS>です。",
            f"上位分類への繋がりを調べると{plist}となります。この論理関係から結果は<ANS>と判断できます。",
            f"{plist}の包含関係を順に確認した結果、問いに対する答えは<ANS>になります。",
            f"集合のつながりである{plist}を辿ると、成立する結論は<ANS>です。",
        ]
        return opts[reg]

    if mode == "ContradictionCycle":
        opts = [
            f"順序関係に閉路がないか検証します。{plist}と辿ると出発点へ戻ってしまうため、結果は<ANS>です。",
            f"比較の向きを順に追うと{plist}で一周してしまいます。序列が成り立たないため結論は<ANS>となります。",
            f"まず{plist}の連鎖を確認します。大小関係が循環しているため、判定は<ANS>です。",
            f"条件を繋ぎ合わせると{plist}というループが生じます。整合しないため答えは<ANS>になります。",
            f"{plist}の経路で関係が元の対象へ戻ることを確認しました。したがって結果は<ANS>です。",
            f"序列の整合性を調べたところ{plist}で閉じており両立しません。結論は<ANS>です。",
        ]
        return opts[reg]

    if mode == "ContradictionConflict":
        opts = [
            f"同一対象に対する記述の整合性を確認します。{plist}に関する条件が食い違うため、結果は<ANS>です。",
            f"前提同士を照合すると、{plist}で異なる内容が同時に主張されています。両立しないため結論は<ANS>となります。",
            f"まず{plist}の記録を突き合わせます。値が一致せず衝突しているため、判定は<ANS>です。",
            f"条件の重複を点検したところ、{plist}の間で食い違いが見つかりました。よって答えは<ANS>になります。",
            f"{plist}の前提が互いに反しており同時には成り立ちません。したがって結論は<ANS>です。",
            f"記述の整合チェックを行うと{plist}で衝突が生じています。最終判定は<ANS>です。",
        ]
        return opts[reg]

    if mode == "QuantDiff":
        opts = [
            f"数値の隔たりを求めるため両者の測定値を確認します。{pvlist}の差を計算すると、結果は<ANS>です。",
            f"まず対象の数値を揃えて比較します。{pvlist}について大きい値から小さい値を引くと、<ANS>となります。",
            f"根拠となる値は{pvlist}です。二つの数値の差を算出すると、答えは<ANS>になります。",
            f"{pvlist}の値を引き算して隔たりを求めます。計算の結果、結論は<ANS>です。",
            f"比較対象の数値を整理すると{pvlist}です。差分を取ることで<ANS>が得られます。",
            f"それぞれの値である{pvlist}を確認し、差を計算した結果、<ANS>と求まります。",
        ]
        return opts[reg]

    if mode == "QuantCompare":
        opts = [
            f"同じ単位で測定値の大小を見比べます。{pvlist}を比較した結果、条件に合うのは<ANS>です。",
            f"まず双方の数値を並べて確認します。{pvlist}の大小関係から、答えは<ANS>となります。",
            f"根拠となる測定値は{pvlist}です。数値を直接比べると、結論は<ANS>になります。",
            f"{pvlist}の値を照らし合わせて比較します。大きい側・小さい側を判定すると結果は<ANS>です。",
            f"数値を並べて検証すると{pvlist}です。この比較により該当するのは<ANS>と分かります。",
            f"対象の値である{pvlist}を見比べた結果、求める答えは<ANS>です。",
        ]
        return opts[reg]

    # Abstain
    opts = [
        "与えられた前提を点検しましたが、結論を一つに決める手掛かりが足りません。したがって<ANS>です。",
        "条件を整理しても対象を特定する根拠が不足しています。無理に断定せず、結果は<ANS>となります。",
        "まず必要な情報を確認しましたが、判断に足る記述が揃っていません。結論は<ANS>です。",
        "前提の範囲だけでは一意に導けないため、保留として<ANS>と判断します。",
        "根拠となる関係が欠けていることを確認しました。このため答えは<ANS>になります。",
        "手掛かりが不足しており確定できないため、最終的な判定は<ANS>です。",
    ]
    return opts[reg]


def gen_episode(mode: str, rng: random.Random) -> dict:
    opener = rng.choice(mined_openers)
    if mode == "Copy":
        e = rng.choice(ENTITIES)
        v = rng.choice(VALUES_TEXT)
        sep = rng.choice(["＝", "は", "の担当は", "の場所は", "の値は", "→"])
        if sep == "＝":
            q = rng.choice([f"{e}＝{v}。{e}＝", f"{e}＝{v}。\n{e}＝何？", f"{e}＝{v}。{e}は何ですか？"])
        elif sep == "→":
            q = f"{e}→{v}。{e}の対応先は何ですか？"
        else:
            q = rng.choice([
                f"{e}{sep}{v}です。{e}{sep}何ですか？",
                f"{e}{sep}{v}。{e}について教えてください。",
                f"{e}{sep}{v}。\n{e}の値を答えてください。",
            ])
        prompt = opener + q
        return {"mode": mode, "prompt": prompt, "n_p": 1, "n_v": 0, "reasoning": build_reasoning(mode, 1, 0, rng)}

    if mode in ("OrderExtreme", "OrderRank", "OrderPairwise"):
        n = rng.choice([2, 3, 3, 4, 4, 5, 6])
        if mode == "OrderRank":
            n = rng.choice([3, 4, 4, 5])
        ents = rng.sample(ENTITIES, n)
        hi, lo, sup_hi, sup_lo = rng.choice(ORDER_PAIRS)
        clauses = []
        for i in range(n - 1):
            if rng.random() < 0.7:
                clauses.append(f"{ents[i]}は{ents[i+1]}より{hi}。")
            else:
                clauses.append(f"{ents[i+1]}は{ents[i]}より{lo}。")
        rng.shuffle(clauses)
        body = "".join(clauses)
        if mode == "OrderExtreme":
            ask = rng.choice([
                f"{sup_hi}のはどれですか？", f"{sup_lo}のは誰ですか？",
                f"{sup_hi}のを教えてください。", f"もっとも{hi}のはどれ？",
                f"一番{lo}のはどれになりますか？",
            ])
        elif mode == "OrderRank":
            rank = rng.choice(["2番目", "二番目", "3番目", "真ん中"])
            ask = rng.choice([
                f"{hi}方から{rank}はどれですか？",
                f"{lo}方から{rank}は誰？",
                f"順番に並べたとき{rank}を教えてください。",
            ])
        else:
            a, b = rng.sample(ents, 2)
            ask = rng.choice([
                f"{a}と{b}ではどちらが{hi}ですか？",
                f"{a}は{b}より{hi}ですか？",
                f"{a}と{b}を比べるとどちらが{lo}？",
            ])
        prompt = opener + body + rng.choice(["", "\n"]) + ask
        return {"mode": mode, "prompt": prompt, "n_p": n, "n_v": 0, "reasoning": build_reasoning(mode, n, 0, rng)}

    if mode == "Transfer":
        n = rng.choice([2, 2, 3, 3, 4, 5])
        ents = rng.sample(ENTITIES, n)
        val = rng.choice(VALUES_TEXT)
        attr = rng.choice(["の拠点は", "の担当は", "の分類は", "の色は", "の所属は"])
        clauses = [f"{ents[0]}{attr}{val}です。"]
        for i in range(n - 1):
            clauses.append(rng.choice([
                f"{ents[i+1]}は{ents[i]}と同じです。",
                f"{ents[i+1]}は{ents[i]}の別名です。",
                f"{ents[i+1]}も{ents[i]}と同じ所属です。",
            ]))
        rng.shuffle(clauses)
        ask = rng.choice([
            f"{ents[-1]}{attr}何ですか？",
            f"{ents[-1]}の対応先を教えてください。",
            f"{ents[-1]}はどこですか？",
        ])
        prompt = opener + "".join(clauses) + rng.choice(["", "\n"]) + ask
        return {"mode": mode, "prompt": prompt, "n_p": n, "n_v": 0, "reasoning": build_reasoning(mode, n, 0, rng)}

    if mode == "Elimination":
        n_total = rng.choice([3, 4, 5])
        ents = rng.sample(ENTITIES, n_total)
        elim = ents[:-1]
        cand_str = "・".join(ents)
        clauses = [f"候補は{cand_str}です。"]
        for e in elim:
            clauses.append(rng.choice([f"{e}ではない。", f"{e}は除外する。", f"{e}以外が正解です。"]))
        ask = rng.choice(["残るのはどれですか？", "正解はどれ？", "該当するものを教えてください。"])
        prompt = opener + "".join(clauses) + ask
        return {"mode": mode, "prompt": prompt, "n_p": len(elim), "n_v": 0, "reasoning": build_reasoning(mode, len(elim), 0, rng)}

    if mode == "Syllogism":
        n = rng.choice([3, 3, 4, 5])
        e0 = rng.choice(ENTITIES)
        cs = rng.sample(CATS, n - 1)
        chain = [e0] + cs
        clauses = []
        for i in range(n - 1):
            clauses.append(rng.choice([
                f"{chain[i]}はすべて{chain[i+1]}です。",
                f"{chain[i]}は{chain[i+1]}の一種です。",
                f"すべての{chain[i]}は{chain[i+1]}に属します。",
            ]))
        rng.shuffle(clauses)
        ask = rng.choice([
            f"{chain[0]}は{chain[-1]}ですか？",
            f"{chain[0]}は{chain[-1]}と言えますか？",
            f"{chain[0]}は{chain[-1]}に属するか判定してください。",
        ])
        prompt = opener + "".join(clauses) + ask
        return {"mode": mode, "prompt": prompt, "n_p": n, "n_v": 0, "reasoning": build_reasoning(mode, n, 0, rng)}

    if mode == "ContradictionCycle":
        n = rng.choice([3, 4, 5])
        ents = rng.sample(ENTITIES, n)
        hi, lo, sup_hi, _ = rng.choice(ORDER_PAIRS)
        clauses = [f"{ents[i]}は{ents[(i+1)%n]}より{hi}。" for i in range(n)]
        rng.shuffle(clauses)
        ask = rng.choice([
            f"{sup_hi}のはどれですか？",
            "この条件に矛盾はありますか？",
            "一番端はどれになりますか？",
        ])
        prompt = opener + "".join(clauses) + ask
        return {"mode": mode, "prompt": prompt, "n_p": n + 1, "n_v": 0, "reasoning": build_reasoning(mode, n + 1, 0, rng)}

    if mode == "ContradictionConflict":
        e = rng.choice(ENTITIES)
        v1, v2 = rng.sample(VALUES_TEXT, 2)
        n = rng.choice([2, 3])
        attr = rng.choice(["の担当は", "の場所は", "の分類は", "＝"])
        if attr == "＝":
            body = f"{e}＝{v1}。{e}＝{v2}。"
        else:
            body = f"{e}{attr}{v1}です。しかし{e}{attr}{v2}です。"
        ask = rng.choice([
            f"{e}はどれですか？", "矛盾がありますか？", "正しく判定できますか？",
        ])
        prompt = opener + body + ask
        return {"mode": mode, "prompt": prompt, "n_p": n, "n_v": 0, "reasoning": build_reasoning(mode, n, 0, rng)}

    if mode in ("QuantDiff", "QuantCompare"):
        attr, unit, lo_v, hi_v, adj_hi, adj_lo = rng.choice(QUANT_ATTRS)
        n_m = rng.choice([2, 2, 2, 3])
        ents = rng.sample(ENTITIES, n_m)
        vals = []
        while len(vals) < n_m:
            cand = rng.randint(lo_v, hi_v)
            if f"{cand}{unit}" not in forbidden_tokens and not any(ft in f"{cand}{unit}" for ft in forbidden_tokens):
                vals.append(cand)
        # Sometimes omit inline values (matching SQLite lookup questions)
        if rng.random() < 0.45:
            body = ""
        else:
            body = "".join(f"{e}の{attr}は{v}{unit}。" for e, v in zip(ents, vals))
        ent_join = "と".join(ents)
        if mode == "QuantDiff":
            ask = rng.choice([
                f"{ent_join}の{attr}差は何{unit}？",
                f"{ent_join}の{attr}の差はいくつですか？",
                f"{ent_join}の{attr}の違いはどれくらい？",
                f"{ent_join}の{attr}の差を求めてください。",
            ])
        else:
            ask = rng.choice([
                f"{ent_join}はどちらが{attr}が{adj_hi}？",
                f"{ent_join}ではどちらが{adj_lo}ですか？",
                f"{ent_join}の{attr}を比べて{adj_hi}方を教えてください。",
                f"{ent_join}のうち{adj_hi}のはどっち？",
            ])
        prompt = opener + body + ask
        return {"mode": mode, "prompt": prompt, "n_p": 2, "n_v": 2, "reasoning": build_reasoning(mode, 2, 2, rng)}

    # Abstain
    e1, e2, e3 = rng.sample(ENTITIES, 3)
    hi, _, sup_hi, _ = rng.choice(ORDER_PAIRS)
    body = rng.choice([
        f"{e1}は{e2}より{hi}。",
        f"{e1}の記録だけがあります。",
        f"{e1}と{e2}の関係は分かりません。",
        "",
    ])
    ask = rng.choice([
        f"{e3}はどれより{hi}ですか？",
        f"{e3}の値を教えてください。",
        f"{e1}と{e3}ではどちらが{hi}？",
        f"{e3}の所属はどこですか？",
    ])
    prompt = opener + body + ask
    return {"mode": mode, "prompt": prompt, "n_p": 0, "n_v": 0, "reasoning": build_reasoning(mode, 0, 0, rng)}


MODES = [
    "Copy", "OrderExtreme", "OrderRank", "OrderPairwise",
    "Transfer", "Elimination", "Syllogism",
    "ContradictionCycle", "ContradictionConflict",
    "QuantDiff", "QuantCompare", "Abstain",
]

rng = random.Random(20261128)
N_TRAIN = 96000   # 8,000 per mode -> ~30-35 MB instruction+reasoning corpus
N_VAL = 3600      # 300 per mode held out

episodes = []
for m in MODES:
    for _ in range(N_TRAIN // len(MODES)):
        ep = gen_episode(m, rng)
        ep["split"] = "train"
        episodes.append(ep)
    for _ in range(N_VAL // len(MODES)):
        ep = gen_episode(m, rng)
        ep["split"] = "val"
        episodes.append(ep)

rng.shuffle(episodes)

# Strict zero-leakage audit
collisions = set()
prompt_overlap = 0
for ep in episodes:
    if ep["prompt"].strip() in forbidden_prompts:
        prompt_overlap += 1
    for ft in forbidden_tokens:
        if ft in ep["prompt"] or ft in ep["reasoning"]:
            collisions.add(ft)

assert prompt_overlap == 0, f"Prompt overlap detected: {prompt_overlap}"
assert not collisions, f"Entity leakage detected: {sorted(collisions)}"

tr_path = OUT_DIR / "train.jsonl"
va_path = OUT_DIR / "val.jsonl"
n_tr = n_va = 0
with tr_path.open("w", encoding="utf-8") as ft, va_path.open("w", encoding="utf-8") as fv:
    for ep in episodes:
        line = json.dumps(ep, ensure_ascii=False) + "\n"
        if ep["split"] == "train":
            ft.write(line)
            n_tr += 1
        else:
            fv.write(line)
            n_va += 1

audit = {
    "mined_human_prompts": len(mined_prompts),
    "mined_openers_used": len(mined_openers),
    "train_episodes": n_tr,
    "val_episodes": n_va,
    "train_mb": round(tr_path.stat().st_size / 1e6, 2),
    "val_mb": round(va_path.stat().st_size / 1e6, 2),
    "forbidden_tokens_checked": len(forbidden_tokens),
    "entity_collisions": sorted(collisions),
    "prompt_overlaps": prompt_overlap,
    "llm_assistant_answers_used": 0,
}
(OUT_DIR / "corpus_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
