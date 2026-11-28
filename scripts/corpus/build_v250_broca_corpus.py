#!/usr/bin/env python3
"""
ENNE v25.0 メタ認知・ブローカ発話野 (MetacogCore + BrocaCore) コーパス構築スクリプト。

設計仕様:
- 逆伝播や事前学習済み言語モデルを使わず訓練コーパスを構築する。
- すべての E2E テスト文・検証用Wikipedia見出し・対話プロンプトを訓練集合から除外する。
- 出力 (`ROOT/v250_broca_metacog/`):
  1. `metacog_act.jsonl`: 単一節前提保持ゲート (store=1) vs 質問・情報要求・空右辺・対話・算術 (store=0)
  2. `metacog_conf.jsonl`: 海馬連想想起スコア・マージン・種別特徴 → 採択 (accept=1/0)
  3. `fact_bio.jsonl`: `v124.sqlite` の実Wikipediaリード文に対する4値BIO (`0=SKIP, 1=B-DEF, 2=I-DEF, 3=TAIL`)
  4. `dialogue_intents.jsonl`: RealPersonaChat および合成日常発話による中立日常対話 7意図分類
  5. `broca_episodes.jsonl`: `BrocaDAG` (14発話行動 + ネイティブポインター `<P0>`, `<P1>`, `<ANS>`, `<END>`) 学習エピソード
  6. `unseen_fact_probes.json` / `unseen_broca_probes.json` / `manifest.json`
"""
import glob
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ENNE_ROOT", "."))
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "v250_broca_metacog"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DENYLIST_ENTS = {
    "富士山", "北岳", "槍ヶ岳", "東京都", "東京", "光合成", "朝塔", "夕塔",
    "平城京", "架空", "追跡", "はち", "ありがとう", "スイカ", "ブドウ", "リンゴ",
    "ミカン", "イチゴ", "佐藤", "田中", "追跡キー", "砂時計印", "セラ", "ネラ",
    "相対性理論", "量子力学", "蒼塔", "朱塔", "記録コード", "紅蓮石",
}

DENYLIST_PROMPTS = {
    "3つの箱に2個ずつりんごを入れたら全部で何個？",
    "富士山と北岳の標高差は何m？",
    "富士山と北岳と槍ヶ岳の標高差は何m？",
    "富士山と槍ヶ岳はどちらが標高が高い？",
    "架空の測定値は？",
    "朝塔と夕塔の高さの差は何m？",
    "朝塔の高さは340m",
    "朝塔の高さは340m。",
    "夕塔の高さは200m。",
    "朝塔とは何ですか",
    "朝塔とは何ですか？",
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
    "追跡キー＝砂時計印。",
    "追跡キー＝",
    "セラ＝ネラ。",
    "セラ＝",
    "スイカはブドウより大きい。ブドウはリンゴより大きい。リンゴはミカンより大きい。ミカンはイチゴより大きい。一番大きいのはどれ？",
    "今日はちょっと疲れたよ。",
    "明日の発表が緊張するなあ。",
    "試験に合格してすごく嬉しい！",
    "ありがとう、すごく助かったよ。",
    "今日すごく楽しかったんだよね。",
    "最近ちょっと眠れなくてさ。",
    "ねえ、少し話を聞いてくれる？",
    "おつかれさま、今日もいい天気だね。",
    "こえしゅーどっく",
    "今の時刻を教えて",
    "1袋4個入りのパンを5袋買うと全部で何個？",
    "850円と320円を合わせると何円ですか？",
    "24本の鉛筆を6人で分けると1人何本？",
    "120枚あった紙から45枚使うと残りは何枚？",
    "18 * 7 =",
    "144 / 12",
    "明日の株価はどうなりますか？",
    "存在しない都市の人口は何人？",
    "赤、青、緑が対象で、赤は除外、青も除外。残るのは？",
    "太郎は花子より速く、花子は次郎より速い。一番速いのは誰？",
    "佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？",
    "富士山は3776mで、北岳は3193m。高さの差は何m？",
    "鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？",
    "12.5 + 3.5",
    "日本の首都とは何ですか",
    "蒼塔と朱塔の高さの差は何m？",
    "蒼塔と朱塔はどちらが高い？",
    "甲は乙より重く、乙は丙より重く、丙は甲より重い。一番重いのは誰？",
    "富士山の標高は3776m",
    "追跡キー＝砂時計印",
    "記録コード＝紅蓮石",
    "記録コード＝",
}

INTENT_NAMES = ["greeting", "thanks", "tired", "anxious", "positive", "question_open", "general"]

NEUTRAL_REPLIES = {
    0: [
        "こんにちは。今日はどのようなことをお手伝いできますか。",
        "おつかれさまです。何か知りたいことや気になることはありますか。",
        "おはようございます。どうぞお気軽にお話しください。",
    ],
    1: [
        "どういたしまして。またいつでもお声がけください。",
        "お役に立ててよかったです。ほかに必要なことはありますか。",
        "こちらこそありがとうございます。引き続きお手伝いします。",
    ],
    2: [
        "お疲れさまです。無理をなさらず、少し休んでくださいね。",
        "大変でしたね。温かいものでも飲んで一息ついてください。",
        "よく頑張られましたね。今日はゆっくり体を休められるとよいですね。",
    ],
    3: [
        "緊張しますよね。これまで準備したことを信じて落ち着いて進みましょう。",
        "不安になるお気持ちもよくわかります。一歩ずつ整理して進めていきましょう。",
        "大丈夫ですよ。深呼吸をして、できるところから確認してみましょう。",
    ],
    4: [
        "それは素晴らしいですね。努力が実を結んで本当によかったです。",
        "おめでとうございます。嬉しい出来事があると気持ちも明るくなりますね。",
        "とてもよいお知らせですね。頑張ってきた成果が出ましたね。",
    ],
    5: [
        "もちろんです。差し支えない範囲で詳しく教えていただけますか。",
        "ええ、ぜひお伺いします。どのようなことでしょうか。",
        "はい、喜んでお聞きします。気になっていることをお話しください。",
    ],
    6: [
        "そうなんですね。よろしければ、もう少し経緯をお聞かせください。",
        "お話しいただきありがとうございます。気になる点があればおっしゃってください。",
        "承知しました。ほかにも確認したいことがあればお知らせください。",
    ],
}

OPEN_PAREN = set("(（[【")
CLOSE_PAREN = set(")）]】")
PAREN_PAIR = {"(": ")", "（": "）", "[": "]", "【": "】"}


def split_of(text: str) -> str:
    h = int(hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return "val" if (h % 7) == 0 else "train"


def is_clean_text(text: str) -> bool:
    t = text.strip()
    if not t or t in DENYLIST_PROMPTS:
        return False
    return not any(ent in t for ent in DENYLIST_ENTS)


def label_wiki_lead(head: str, lead: str):
    text = lead.strip()
    if len(text) < 15 or len(text) > 220 or not head:
        return None
    if not is_clean_text(text) or not is_clean_text(head):
        return None
    chars = list(text)
    n = len(chars)
    pdepth = [0] * n
    depth = 0
    for i, c in enumerate(chars):
        if c in OPEN_PAREN:
            depth += 1
            pdepth[i] = depth
        elif c in CLOSE_PAREN:
            pdepth[i] = depth
            depth = max(0, depth - 1)
        else:
            pdepth[i] = depth

    # First sentence-ending '。' at paren_depth == 0
    s1_end = -1
    for i, c in enumerate(chars):
        if pdepth[i] == 0 and c == "。":
            s1_end = i
            break
    if s1_end < 6:
        return None

    # Walk outside-paren chars up to s1_end to locate leading "<head>とは、" / "<head>は、" / "<head>は"
    out_positions = [i for i in range(s1_end) if pdepth[i] == 0]
    out_str = "".join(chars[i] for i in out_positions)
    m = re.match(r"^(?:『[^』]{1,40}』|「[^」]{1,40}」)?\s*(.+?)(とは、|とは|は、|は)\s*", out_str)
    if not m:
        return None
    subj_raw = m.group(1).strip()
    if subj_raw != head and head not in subj_raw and subj_raw not in head:
        return None
    topic_end_out_idx = m.end()
    if topic_end_out_idx >= len(out_positions) - 3:
        return None
    def_start = out_positions[topic_end_out_idx]

    # Exclude sentence-final copula before/at s1_end at paren_depth == 0
    def_end = s1_end
    body_out_before_end = "".join(chars[i] for i in range(def_start, s1_end) if pdepth[i] == 0)
    tail_end_out_pos = None
    for cop in ("である", "をいう", "を指す", "のこと", "です", "だ"):
        if body_out_before_end.endswith(cop) and len(body_out_before_end) > len(cop) + 3:
            # Find start index in chars of the suffix copula
            target_len = len(body_out_before_end) - len(cop)
            cnt = 0
            for idx in range(def_start, s1_end):
                if pdepth[idx] == 0:
                    if cnt == target_len:
                        tail_end_out_pos = idx
                        break
                    cnt += 1
            break
    if tail_end_out_pos is not None:
        def_end = tail_end_out_pos

    tags = [0] * n
    started = False
    def_count = 0
    for i in range(n):
        if i >= def_end:
            tags[i] = 3  # TAIL
        elif i >= def_start and pdepth[i] == 0:
            if not started:
                tags[i] = 1  # B-DEF
                started = True
            else:
                tags[i] = 2  # I-DEF
            def_count += 1
        else:
            tags[i] = 0  # SKIP

    if def_count < 4 or def_count > 110:
        return None
    return {"head": head, "body": text, "tags": tags, "split": split_of(text)}


def classify_dialogue_intent(text: str) -> int:
    t = text.strip()
    if any(k in t for k in ("ありがとう", "感謝", "助かり", "助かった")):
        return 1  # thanks
    if any(k in t for k in ("おはよう", "こんにちは", "こんばんは", "おつかれ", "お疲れ")):
        return 0  # greeting
    if any(k in t for k in ("疲れ", "つかれ", "眠い", "ねむい", "だるい", "眠れなく")):
        return 2  # tired
    if any(k in t for k in ("緊張", "不安", "心配", "ドキドキ", "怖い", "迷って")):
        return 3  # anxious
    if any(k in t for k in ("嬉しい", "うれしい", "楽しい", "たのしい", "合格", "成功", "最高")):
        return 4  # positive
    if t.endswith(("？", "?")) or any(k in t for k in ("聞いて", "相談", "どう思う", "でしょうか", "教えて")):
        return 5  # question_open
    return 6  # general


def main():
    rng = random.Random(20261128)
    unseen_broca = {
        "metacog_store": [
            {"text": "朝塔の高さは340m。", "store": 1},
            {"text": "朝塔の高さは340m", "store": 1},
            {"text": "追跡キー＝砂時計印。", "store": 1},
            {"text": "追跡キー＝", "store": 0},
            {"text": "朝塔とは何ですか", "store": 0},
            {"text": "日本の首都とは何ですか？", "store": 0},
            {"text": "富士山について知りたい", "store": 0},
        ],
        "dialogue": [
            {"text": "今日はちょっと疲れたよ。", "expected_intent": 2},
            {"text": "明日の発表が緊張するなあ。", "expected_intent": 3},
            {"text": "試験に合格してすごく嬉しい！", "expected_intent": 4},
            {"text": "ありがとう、すごく助かったよ。", "expected_intent": 1},
            {"text": "こえしゅーどっく", "expected_intent": 6},
        ],
    }
    denylist_prompts_norm = {norm_text(p) for p in DENYLIST_PROMPTS}
    unseen_prompts_norm = {
        norm_text(x["text"])
        for x in unseen_broca["metacog_store"] + unseen_broca["dialogue"]
    }
    blocked_prompts_norm = denylist_prompts_norm | unseen_prompts_norm

    # =========================================================================
    # 1. Metacog store-gate corpus (`metacog_act.jsonl`)
    # =========================================================================
    ents = [
        "エベレスト", "白山", "立山", "筑波山", "高尾山", "阿蘇山", "桜島",
        "白川", "黒川", "鈴木", "高橋", "渡辺", "小林", "山本", "斉藤", "松田",
        "P社", "Q社", "R社", "X機", "Y機", "Z機", "D館", "E館", "F館",
        "青葉", "紅葉", "朝霧", "夕凪", "銀貨", "銅貨", "金貨",
        "イヌ", "ネコ", "ワシ", "クジラ", "イルカ", "パンダ", "ハルカ", "ミナト",
    ]
    key_names = ["記録キー", "確認キー", "管理タグ", "合図印", "標識番号", "暗号名", "識別コード", "参照ラベル"]
    val_names = ["青石印", "月光印", "星影紋", "白銀札", "紅玉印", "金剛紋", "蒼空板", "朱雀印", "緑青札"]
    attrs_num = [("標高", "m"), ("高さ", "m"), ("距離", "km"), ("長さ", "cm"), ("重さ", "kg"), ("価格", "円")]
    attrs_assign = ["担当", "勤務地", "出身", "配属", "配属先", "拠点", "専門", "役割", "部門"]
    roles = ["企画", "営業", "設計", "経理", "広報", "製造", "総務", "研究", "開発"]
    adjs = ["大きい", "小さい", "高い", "低い", "速い", "遅い", "重い", "軽い", "長い", "短い"]
    sym_nouns = ["同期だ", "同僚だ", "相棒だ", "仲間だ"]

    metacog_rows = []
    metacog_seen = set()

    def add_mc(text: str, store: int):
        t = text.strip()
        if not t or t in metacog_seen or not is_clean_text(t) or norm_text(t) in blocked_prompts_norm:
            return
        metacog_seen.add(t)
        metacog_rows.append({"text": t, "store": store, "split": split_of(t)})

    for _ in range(1800):
        a, b = rng.sample(ents, 2)
        attr, u = rng.choice(attrs_num)
        v = rng.randint(10, 8800)
        end = rng.choice(["。", ""])
        # Positive declarative premises (with and without '。')
        add_mc(f"{a}の{attr}は{v}{u}{end}", 1)
        add_mc(f"{a}は{v}{u}{end}", 1)
        asg = rng.choice(attrs_assign)
        rl = rng.choice(roles)
        add_mc(f"{a}の{asg}は{rl}{end}", 1)
        add_mc(f"{b}は{a}と同じ{asg}{end}", 1)
        add_mc(f"{b}は{a}と{rng.choice(sym_nouns)}{end}", 1)
        adj = rng.choice(adjs)
        add_mc(f"{a}は{b}より{adj}{end}", 1)
        k = rng.choice(key_names + ents[:12])
        vv = rng.choice(val_names + ents[12:24])
        sym = rng.choice(["＝", "="])
        add_mc(f"{k}{sym}{vv}{end}", 1)

        # Negatives: queries, unpunctuated questions, empty RHS '=' / '＝', Wikipedia requests, multi-clause
        qend = rng.choice(["？", "?", ""])
        add_mc(f"{k}{sym}", 0)
        add_mc(f"{a}とは何ですか{qend}", 0)
        add_mc(f"{a}とは何{qend}", 0)
        add_mc(f"{a}の{attr}は何{u}{qend}", 0)
        add_mc(f"{a}の{asg}は何{qend}", 0)
        add_mc(f"{a}の{asg}を教えて{end}", 0)
        add_mc(f"{a}について知りたい", 0)
        add_mc(f"{a}の仕組みを説明して", 0)
        add_mc(f"{a}ってどこだっけ{qend}", 0)
        add_mc(f"{a}を知ってる{qend}", 0)
        add_mc(f"{a}と{b}の{attr}差は何{u}{qend}", 0)
        add_mc(f"{a}と{b}はどちらが{adj}{qend}", 0)
        add_mc(f"{a}の{attr}は{v}{u}。{b}の{attr}は{v+10}{u}。{a}と{b}は何{u}違う？", 0)

    # =========================================================================
    # 2. Metacog confidence corpus (`metacog_conf.jsonl`)
    # =========================================================================
    conf_rows = []
    for i in range(2400):
        is_pred = rng.randint(0, 1)
        is_taught = 1 if (i % 8 == 0) else 0
        is_exact = 1 if (not is_pred and not is_taught) else 0
        if is_taught:
            top = rng.uniform(0.65, 0.99)
            margin = rng.uniform(0.05, 0.35)
            acc = 1
        elif i % 2 == 0:
            top = rng.uniform(0.49, 0.98)
            margin = rng.uniform(0.01, 0.35)
            acc = 1
        else:
            top = rng.uniform(0.05, 0.41)
            margin = rng.uniform(0.0, 0.12)
            acc = 0
        split = "val" if (i % 7) == 0 else "train"
        conf_rows.append({
            "top_score": round(top, 5),
            "margin": round(margin, 5),
            "is_predicate": is_pred,
            "is_taught": is_taught,
            "is_exact": is_exact,
            "accept": acc,
            "split": split,
        })

    # =========================================================================
    # 3. Wikipedia fact distillation corpus (`fact_bio.jsonl` + `unseen_fact_probes.json`)
    # =========================================================================
    db_path = ROOT / "wiki_lead_v112/v124.sqlite"
    fact_rows = []
    unseen_fact_probes = []
    if db_path.exists():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for p_head in ("富士山", "光合成", "北岳", "相対性理論", "量子力学"):
            r = conn.execute(
                "SELECT head, lead FROM leads WHERE head = ? ORDER BY prio DESC LIMIT 1",
                (p_head,),
            ).fetchone()
            if r:
                unseen_fact_probes.append({"head": r[0], "body": r[1]})
        unseen_heads_norm = {norm_text(x["head"]) for x in unseen_fact_probes}

        cur = conn.execute(
            "SELECT head, lead FROM leads WHERE def_quality >= 1 ORDER BY def_quality DESC, prio DESC LIMIT 40000"
        )
        for head, lead in cur:
            if norm_text(head) in unseen_heads_norm or any(e in head or e in lead for e in DENYLIST_ENTS):
                continue
            row = label_wiki_lead(head, lead)
            if row is not None:
                fact_rows.append(row)
                if len(fact_rows) >= 10000:
                    break
        conn.close()
    else:
        unseen_heads_norm = set()

    # =========================================================================
    # 4. Dialogue intent corpus (`dialogue_intents.jsonl`) + `BrocaDAG` (`broca_episodes.jsonl`)
    # =========================================================================
    intent_examples = {
        0: [
            "おはようございます。今日もよろしく。", "こんにちは！調子はどうですか？", "こんばんは、今日もお疲れさまでした。",
            "おつかれさま、よく頑張ったね。", "こんにちは、また話せて嬉しいです。", "おはよう、今朝は涼しいですね。",
            "お疲れ様です！今日もいい一日になりますように。", "こんばんは、夜は冷えますね。",
        ],
        1: [
            "本当にありがとうございます。", "とても助かりました、感謝します。", "教えてくれてありがとう！",
            "おかげでうまくいきました、感謝です。", "丁寧に対応してくれて嬉しいです。", "わざわざ調べてくれてありがとう。",
            "先ほどは助かりました。", "ご親切に感謝いたします。",
        ],
        2: [
            "今日はずっと作業してて疲れたな。", "なんだかとても眠いです。", "体がだるくて休みたい気分です。",
            "連日の作業でくたくたです。", "最近ちょっと寝不足気味でつらい。", "集中力が切れて疲れました。",
            "少し休憩したいなあ。", "朝から動き通しで疲れがたまっています。",
        ],
        3: [
            "来週のプレゼンが少し不安です。", "大事なテスト前でドキドキしています。", "失敗しないか心配になってきました。",
            "緊張して落ち着かないです。", "うまくできるか少し迷っています。", "明日の面談がちょっと怖いです。",
            "初めての挑戦なので不安が大きい。", "緊張感が抜けなくてそわそわする。",
        ],
        4: [
            "目標を達成できてすごく嬉しいです！", "今日の実験が大成功しました！", "とても楽しい時間を過ごせました。",
            "念願の資格に合格しました！", "褒められて最高の気分です。", "結果が良く出てほっとしました！",
            "おいしいものが食べられて幸せです。", "うまくいったので本当に嬉しい！",
        ],
        5: [
            "少し相談に乗ってもらえますか？", "ちょっと話を聞いてもらってもいい？", "このやり方で合っているかどう思いますか？",
            "おすすめの進め方を教えてください。", "何か良いアイデアはありませんか。", "少しだけ質問してもよろしいですか？",
            "こういう時はどうすればいいでしょうか。", "意見を聞かせてもらえますか。",
        ],
        6: [
            "週末は本を読んで過ごす予定です。", "最近の天気は変わりやすいですね。", "コーヒーを淹れて一息ついています。",
            "そろそろ昼食の時間ですね。", "電車が少し混んでいました。", "新しいノートを買ってきました。",
            "静かな音楽を聴きながら作業しています。", "季節の変わり目になってきましたね。",
        ],
    }

    # Augment intents with real human utterances from RealPersonaChat
    rpc_dir = ROOT / "persona_corpora_v20/real_persona_chat/real-persona-chat-1.0.0/real_persona_chat/dialogues"
    rpc_pool = []
    for fp in sorted(glob.glob(str(rpc_dir / "*.json")))[:2500]:
        try:
            d = json.loads(Path(fp).read_text(encoding="utf-8"))
            for u in d.get("utterances", [])[:6]:
                t = u.get("text", "").strip()
                if 5 <= len(t) <= 60 and is_clean_text(t) and not re.search(r"[0-9０-９+\-*/×÷=＝]", t):
                    rpc_pool.append(t)
        except Exception:
            pass
    rng.shuffle(rpc_pool)

    dlg_rows = []
    dlg_seen = set()
    dlg_counts = Counter()

    def add_dlg(text: str, intent: int, max_per_intent: int = 360):
        t = re.sub(r"\s+", " ", text.strip())
        if (
            not t
            or t in dlg_seen
            or not is_clean_text(t)
            or norm_text(t) in blocked_prompts_norm
            or dlg_counts[intent] >= max_per_intent
        ):
            return
        dlg_seen.add(t)
        dlg_counts[intent] += 1
        dlg_rows.append({"text": t, "intent": intent, "split": split_of(t)})

    for intent_id, seeds in intent_examples.items():
        for s in seeds:
            add_dlg(s, intent_id, 400)

    # Rich diverse everyday phrasing per intent (balanced across intents 0..5 so general=6 never dominates)
    advs = ["", "今日は", "なんだか", "ちょっと", "すごく", "本当に", "最近", "少し"]
    ends_calm = ["。", "よ。", "なあ。", "な。", "です。"]
    ends_excl = ["！", "よ！", "なあ！"]
    events = ["発表", "仕事", "試験", "作業", "研究", "課題", "計画", "準備", "勉強", "練習"]
    for _ in range(180):
        ad = rng.choice(advs)
        ad2 = rng.choice(["", "ちょっと", "すごく", "とても", "かなり"])
        ev = rng.choice(events)
        add_dlg(f"{ad}{ad2}疲れた{rng.choice(ends_calm)}", 2)
        add_dlg(f"{ad}体がだるい{rng.choice(ends_calm)}", 2)
        add_dlg(f"{ev}で{ad2}疲れてしまった{rng.choice(ends_calm)}", 2)
        add_dlg(f"来週の{ev}が緊張する{rng.choice(ends_calm)}", 3)
        add_dlg(f"{ev}がうまくいくか不安{rng.choice(['だなあ。', 'です。', 'だよ。'])}", 3)
        add_dlg(f"{ad2}ドキドキして心配{rng.choice(['だよ。', 'です。', 'だな。'])}", 3)
        add_dlg(f"{ev}が成功して{ad2}嬉しい{rng.choice(ends_excl)}", 4)
        add_dlg(f"{ev}で結果が出て楽しい{rng.choice(ends_excl)}", 4)
        add_dlg(f"{ad2}助かった{rng.choice(['よ。', '！', '、ありがとう。'])}", 1)
        add_dlg(f"丁寧に教えてくれてありがとう{rng.choice(['！', 'ございます。', '。'])}", 1)
        add_dlg(f"{ev}について相談してもいい{rng.choice(['？', 'ですか？'])}", 5)
        add_dlg(f"ちょっと話を聞いてもらえますか？", 5)
        add_dlg(f"{rng.choice(['おはようございます', 'こんにちは', 'おつかれさま'])}、良い天気{rng.choice(['だね。', 'ですね。'])}", 0)

    for t in rpc_pool:
        intent = classify_dialogue_intent(t)
        add_dlg(t, intent, 320)

    # =========================================================================
    # 5. BrocaDAG utterance episodes (`broca_episodes.jsonl`, acts 0..14)
    # =========================================================================
    # Acts:
    #   0: SHORT_ANS ("<ANS>")
    #   1: ACK ("了解しました。一時記憶に保持しました。")
    #   2: ABSTAIN ("情報がありません。")
    #   3: CLOCK ("現在は<P0>時<P1>分です。")
    #   4: CALC ("<P0><P1>です。")
    #   5: FACT_EXPLAIN ("<P0>は、<P1>です。")
    #   6: FACT_TAUGHT ("<ANS>")
    #   7..13: DLG_0..DLG_6 (neutral everyday replies conditioned on h_prompt)
    broca_eps = []

    def add_broca(act: int, prompt: str, target: str):
        p = prompt.strip()
        if not p or not target or norm_text(p) in blocked_prompts_norm:
            return
        sp = split_of(f"{act}:{p}:{target}")
        broca_eps.append({"act": act, "prompt": p, "target": target, "split": sp})

    for _ in range(220):
        a, b = rng.sample(ents, 2)
        attr, u = rng.choice(attrs_num)
        v = rng.randint(20, 4000)
        asg = rng.choice(attrs_assign)
        rl = rng.choice(roles)
        add_broca(0, f"{a}と{b}の{attr}差は何{u}？", "<ANS>")
        add_broca(0, f"{a}の{asg}は何？", "<ANS>")
        add_broca(0, f"{a}は{b}より大きい。一番大きいのはどれ？", "<ANS>")
        add_broca(1, f"{a}の{attr}は{v}{u}。", "了解しました。一時記憶に保持しました。")
        add_broca(1, f"{a}の{asg}は{rl}。", "了解しました。一時記憶に保持しました。")
        add_broca(1, f"{rng.choice(key_names)}＝{rng.choice(val_names)}。", "了解しました。一時記憶に保持しました。")
        add_broca(2, f"存在しない{a}の値は？", "情報がありません。")
        add_broca(2, f"来週の{a}の予想は？", "情報がありません。")
        add_broca(3, rng.choice(["今何時ですか？", "現在時刻を教えて", "いま何時？"]), "現在は<P0>時<P1>分です。")
        add_broca(4, f"{v} + 15 =", "<P0><P1>です。")
        add_broca(4, f"2袋に{v}個ずつ入れると何個？", "<P0><P1>です。")
        add_broca(5, f"{a}とは何ですか？", "<P0>は、<P1>です。")
        add_broca(5, f"{b}について知りたい", "<P0>は、<P1>です。")
        add_broca(6, f"{a}とは何ですか？", "<ANS>")
        add_broca(6, f"{b}の高さは？", "<ANS>")

    # Dialogue act episodes: map punctuation cues inside each intent cleanly so W_prompt learns real signals
    by_intent = Counter()
    for row in dlg_rows:
        it = row["intent"]
        candidates = NEUTRAL_REPLIES[it]
        t_txt = row["text"]
        if "！" in t_txt or "!" in t_txt:
            idx = 1 % len(candidates)
        elif "？" in t_txt or "?" in t_txt:
            idx = 2 % len(candidates)
        else:
            idx = 0
        if by_intent[it] < 180:
            broca_eps.append({
                "act": 7 + it,
                "prompt": row["text"],
                "target": candidates[idx],
                "split": row["split"],
            })
            by_intent[it] += 1

    # Write all artifacts
    def write_jsonl(path: Path, items: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "metacog_act.jsonl", metacog_rows)
    write_jsonl(OUT_DIR / "metacog_conf.jsonl", conf_rows)
    write_jsonl(OUT_DIR / "fact_bio.jsonl", fact_rows)
    write_jsonl(OUT_DIR / "dialogue_intents.jsonl", dlg_rows)
    write_jsonl(OUT_DIR / "broca_episodes.jsonl", broca_eps)

    (OUT_DIR / "unseen_fact_probes.json").write_text(
        json.dumps(unseen_fact_probes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "unseen_broca_probes.json").write_text(
        json.dumps(unseen_broca, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    denylist_hits = (
        sum(1 for r in metacog_rows if norm_text(r["text"]) in denylist_prompts_norm)
        + sum(1 for r in dlg_rows if norm_text(r["text"]) in denylist_prompts_norm)
        + sum(1 for r in broca_eps if norm_text(r["prompt"]) in denylist_prompts_norm)
        + sum(1 for r in fact_rows if any(e in r["head"] or e in r["body"] for e in DENYLIST_ENTS))
    )
    probe_overlap = (
        sum(1 for r in metacog_rows if norm_text(r["text"]) in unseen_prompts_norm)
        + sum(1 for r in dlg_rows if norm_text(r["text"]) in unseen_prompts_norm)
        + sum(1 for r in broca_eps if norm_text(r["prompt"]) in unseen_prompts_norm)
        + sum(1 for r in fact_rows if norm_text(r["head"]) in unseen_heads_norm)
    )
    manifest = {
        "metacog_act_total": len(metacog_rows),
        "metacog_conf_total": len(conf_rows),
        "fact_bio_total": len(fact_rows),
        "dialogue_intents_total": len(dlg_rows),
        "broca_episodes_total": len(broca_eps),
        "unseen_fact_probes": len(unseen_fact_probes),
        "denylist_hits": denylist_hits,
        "probe_overlap": probe_overlap,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
