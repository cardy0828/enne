//! ENNE v25.0 テキスト経路の結合テストおよびソースコード監査テスト。
use anyhow::Result;
use enne_rs::{brain::Brain, core_dag::NeurogenesisGraph, reasoner::Reasoner};
use std::{
    fs,
    path::{Path, PathBuf},
};
/// 凍結データ・学習済み重みの基点ディレクトリを返す（ENNE_ROOT / NEUROGRAPH_ROOT を優先）。
fn root() -> PathBuf {
    std::env::var_os("ENNE_ROOT")
        .or_else(|| std::env::var_os("NEUROGRAPH_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_path_buf()
        })
}
#[test]
/// 算術と複数件の知識検索を公開インターフェースから検証する。
fn grounded_questions() -> Result<()> {
    let mut b = Brain::open(&root())?;
    for (q, a) in [
        ("3つの箱に2個ずつりんごを入れたら全部で何個？", "6個です。"),
        ("富士山と北岳の標高差は何m？", "583.12m"),
        ("富士山と北岳と槍ヶ岳の標高差は何m？", "596.12m"),
        ("富士山と槍ヶ岳はどちらが標高が高い？", "富士山"),
        ("架空の測定値は？", "情報がありません。"),
    ] {
        assert_eq!(b.think(q)?, a)
    }
    b.teach("朝塔", "高さは340m。")?;
    b.teach("夕塔", "高さは200m。")?;
    assert_eq!(b.think("朝塔と夕塔の高さの差は何m？")?, "140m");
    Ok(())
}
#[test]
/// 未知の反意対を観測し、保存・復元とセッション終了後の独立性を確認する。
fn learning_and_reset() -> Result<()> {
    let mut b = Brain::open(&root())?;
    b.learn_relation("春は夏よりゾルい。夏は春よりメクい。", None);
    let q = "甲は乙よりゾルい。丙は乙よりメクい。一番メクいのは誰？";
    assert_eq!(b.reasoner.answer(q), "丙");
    b.learn_relation("春は夏よりルゼい。夏は秋よりルゼい。端点は？", Some("秋"));
    assert_eq!(
        b.reasoner
            .answer("甲は乙よりルゼい。乙は丙よりルゼい。端点は？"),
        "丙"
    );
    b.reasoner.observe_turn("セラ＝ネラ。");
    assert_eq!(b.reasoner.answer("セラ＝"), "ネラ");
    assert!(b.think("日本の首都とは何ですか？")?.starts_with("東京都"));
    b.reset_history();
    assert_ne!(b.reasoner.answer("セラ＝"), "ネラ");
    let file = std::env::temp_dir().join(format!("ng-test-{}.json", std::process::id()));
    b.reasoner.save_synapses(&file)?;
    let loaded = Reasoner::load_synapses(&file)?;
    fs::remove_file(file)?;
    assert_eq!(loaded.answer(q), "丙");
    Ok(())
}
#[test]
/// 同じ初期重みから2回学習し、分裂・棄却・推論専用モードを確認する。
fn growth() -> Result<()> {
    let out = root().join("rust_bench_v182");
    let data = fs::read(out.join("wiki.bin"))?;
    let mut results = vec![];
    for _ in 0..2 {
        let mut graph = NeurogenesisGraph::<128>::load(&out.join("basis_120000.bin"), 1024)?;
        let before = graph.run_stream(&data[..1300], 4, false)?;
        assert_eq!(before.2, 0);
        let learned = graph.run_stream(&data[..1300], 4, true)?;
        assert!(learned.2 > 0);
        graph.apoptosis();
        let after = graph.run_stream(&data[..1300], 4, false)?;
        assert!(after.2 <= learned.2);
        results.push((learned, after));
    }
    assert_eq!(results[0], results[1]);
    Ok(())
}
#[test]
/// 320件の独立した入力と、複数ターンの前提を順序通りに実行する。
fn frozen_suite() -> Result<()> {
    let mut b = Brain::open(&root())?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("rust_bench_v182/bench_320.json"))?)?;
    let rows = fixture["cases"].as_array().unwrap();
    assert_eq!(rows.len(), 320);
    for c in rows {
        b.reset_history();
        if let Some(turns) = c["turns"].as_array() {
            for t in turns {
                b.reasoner.observe_turn(t.as_str().unwrap())
            }
        }
        assert_eq!(
            b.reasoner.answer(c["prompt"].as_str().unwrap()),
            c["gold"],
            "{}",
            c["prompt"]
        );
    }
    Ok(())
}

#[test]
/// 会話で受け取った仮の測定値を優先し、終了後はSQLiteの実測値へ戻る。
fn brain_turns() -> Result<()> {
    let mut b = Brain::open(&root())?;
    for fact in ["富士山の標高は3776m。", "北岳の標高は3193m。"] {
        assert_eq!(b.think(fact)?, "了解しました。一時記憶に保持しました。");
    }
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583m");
    b.reset_history();
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583.12m");
    b.think("佐藤の担当は開発。")?;
    b.think("田中は佐藤と同じ担当。")?;
    assert_eq!(b.think("田中の担当は何？")?, "開発");
    Ok(())
}

#[test]
/// v19.0 推論言語コアを Brain に直結し、発話側は答えのみ・内部思考に全根拠トレースを分離できることを検証する。
fn v190_mounted_brain_e2e() -> Result<()> {
    let ckpt1 = root().join("v190_reasoning_core/reasoning_core_v190.bin");
    let ckpt2 = root().join("rust_bench_v182/v190/reasoning_core_v190.bin");
    if !ckpt1.exists() && !ckpt2.exists() {
        return Ok(());
    }
    let mut b = Brain::open(&root())?;
    b.mount_reasoning_core()?;
    assert!(b.has_reasoning_core());

    // 発話側は答えのみ、内部トレース側に根拠と推論過程が入る
    let (diff, diff_trace) = b.think_with_trace("富士山と北岳の標高差は何m？")?;
    assert_eq!(diff, "583.12m");
    let dt = diff_trace.expect("trace");
    assert!(dt.contains("北岳") && dt.contains("富士山") && dt.contains("3193m") && dt.contains("3776.12m") && dt.contains("583.12m"), "trace={dt}");

    let (cmp, cmp_trace) = b.think_with_trace("富士山と槍ヶ岳はどちらが標高が高い？")?;
    assert_eq!(cmp, "富士山");
    let ct = cmp_trace.expect("trace");
    assert!(ct.contains("槍ヶ岳") && ct.contains("富士山") && ct.contains("3180m") && ct.contains("3776.12m"), "trace={ct}");

    let (chain5, chain5_trace) = b.think_with_trace("スイカはブドウより大きい。ブドウはリンゴより大きい。リンゴはミカンより大きい。ミカンはイチゴより大きい。一番大きいのはどれ？")?;
    assert_eq!(chain5, "スイカ");
    let c5t = chain5_trace.expect("trace");
    for tok in ["スイカ", "ブドウ", "リンゴ", "ミカン", "イチゴ"] {
        assert!(c5t.contains(tok), "missing {tok} in {c5t}");
    }

    let abstain = b.think("架空の測定値は？")?;
    assert_eq!(abstain, "情報がありません。");

    // Multi-turn ephemeral memory & reset zero-bleed
    b.reset_history();
    assert_eq!(b.think("追跡キー＝砂時計印。")?, "了解しました。一時記憶に保持しました。");
    assert_eq!(b.think("追跡キー＝")?, "砂時計印");
    b.reset_history();
    assert!(!b.think("追跡キー＝")?.contains("砂時計印"));

    // Wikipedia 述語検索は定義文そのままではなく答えのみを返す
    assert_eq!(b.think("日本の首都とは何ですか？")?, "東京都");
    Ok(())
}

#[test]
/// v22.0 純ヘブ司令塔: executive.rs に固定日本語キーワード表が存在しないことを監査し、
/// 未知の言い換え（時計・四則文章題・棄権・雑談・事実）が純ヘブ NN で正確にルーティングされることを検証する。
fn v220_neural_executive_audit_and_e2e() -> Result<()> {
    // 1. ソースコード監査: 旧来の固定日本語キーワード表が除かれていること
    let src = include_str!("../src/executive.rs");
    for forbidden in [
        "今何時",
        "現在時刻",
        "今の時間",
        "架空",
        "宝くじ",
        "当選番号",
        "予言",
        "合計",
        "合わせて",
        "全部で",
        "ずつ",
        "緊張",
        "化学式",
        "ありがとう",
        "おはよう",
    ] {
        assert!(!src.contains(forbidden), "forbidden keyword literal in executive.rs: {forbidden}");
    }

    // 2. 未知言い換えの E2E 動作検証（旧キーワード表では失敗していた表現を含む）
    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    // 時刻（旧キーワード表にない言い換え）
    let clk = b.think("今の時刻を教えて")?;
    assert!(clk.starts_with("現在は") && clk.ends_with("分です。"), "clk={clk}");

    // 四則文章題（加算・減算・乗算・除算すべて純ヘブ NN の演算分類で解く）
    assert_eq!(b.think("1袋4個入りのパンを5袋買うと全部で何個？")?, "20個です。");
    assert_eq!(b.think("850円と320円を合わせると何円ですか？")?, "1170円です。");
    assert_eq!(b.think("24本の鉛筆を6人で分けると1人何本？")?, "4本です。");
    assert_eq!(b.think("120枚あった紙から45枚使うと残りは何枚？")?, "75枚です。");
    assert_eq!(b.think("18 * 7 =")?, "126です。");
    assert_eq!(b.think("144 / 12")?, "12です。");

    // 棄権（旧キーワード表にない未知表現）
    assert_eq!(b.think("明日の株価はどうなりますか？")?, "情報がありません。");
    assert_eq!(b.think("存在しない都市の人口は何人？")?, "情報がありません。");

    // 事実・推論（誤って Calc/Abstain に吸われないこと）
    assert_eq!(b.think("日本の首都とは何ですか？")?, "東京都");
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583.12m");
    Ok(())
}

#[test]
/// v23.0 純ヘブ連想海馬: hippocampus.rs に SQLite・CLEAN/MEASURE 正規表現・比較キーワードが
/// 一切存在しないことを監査し、未知言い換え想起・比較・1回教示・棄権を E2E で検証する。
fn v230_neural_hippocampus_audit_and_e2e() -> Result<()> {
    // 1. ソースコード監査: SQLite と固定正規表現ルールが除かれていること
    let src = include_str!("../src/hippocampus.rs");
    for forbidden in [
        "rusqlite",
        "SELECT",
        "SQLITE",
        "Regex",
        "とは何ですか",
        "について教えて",
        "標高",
        "海抜",
        "メートル",
        "高い",
        "低い",
    ] {
        assert!(
            !src.contains(forbidden),
            "forbidden literal in hippocampus.rs: {forbidden}"
        );
    }

    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    // 2. 旧 CLEAN 正規表現では剥離できなかった未知言い換えの直接想起
    let h1 = b.hip.lookup_hits("日本の首都ってどこだっけ？", 3)?;
    assert_eq!(h1[0].head, "東京都");
    let h2 = b.hip.lookup_hits("日本の首都を知ってる？", 3)?;
    assert_eq!(h2[0].head, "東京都");
    let h3 = b.hip.lookup_hits("富士山について知りたい", 3)?;
    assert_eq!(h3[0].head, "富士山");
    let h4 = b.hip.lookup_hits("光合成の仕組みを説明して", 3)?;
    assert_eq!(h4[0].head, "光合成");

    // End-to-end Brain::think on unseen paraphrases (where Executive routes to Fact)
    assert_eq!(b.think("日本の首都ってどこだっけ？")?, "東京都");
    assert!(b.think("富士山について知りたい")?.starts_with("富士山"));
    assert!(b.think("光合成の仕組みを説明して")?.starts_with("光合成"));

    // 3. 学習データから厳格除外したホールドアウト実体の量的比較・差異
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583.12m");
    assert_eq!(b.think("富士山と北岳と槍ヶ岳の標高差は何m？")?, "596.12m");
    assert_eq!(b.think("富士山と槍ヶ岳はどちらが標高が高い？")?, "富士山");

    // 4. オンライン1回教示 (fast-weight teach) + 神経数値抽出
    b.teach("朝塔", "高さは340m。")?;
    b.teach("夕塔", "高さは200m。")?;
    assert_eq!(b.think("朝塔と夕塔の高さの差は何m？")?, "140m");

    // 5. 棄権の維持
    assert_eq!(b.think("架空の測定値は？")?, "情報がありません。");
    Ok(())
}

#[test]
/// v24.0 純ヘブ関係推論野: reasoner.rs / learning.rs / reasoner_core.rs に
/// BFS (VecDeque/reach/path)・文字種ハック・固定日本語規則が存在しないことを監査し、
/// 未知言い換え推論および白紙からの suite::refit 320/320 を E2E で検証する。
fn v240_neural_reasoner_audit_and_e2e() -> Result<()> {
    let r_src = include_str!("../src/reasoner.rs");
    let l_src = include_str!("../src/learning.rs");
    let c_src = include_str!("../src/reasoner_core.rs");
    for (name, src) in [
        ("reasoner.rs", r_src),
        ("learning.rs", l_src),
        ("reasoner_core.rs", c_src),
    ] {
        for forbidden in [
            "VecDeque",
            "reach(",
            "path(",
            "差",
            "番目",
            "ですか？",
            "katakana",
            "synapses_v182",
            "SCALAR_PAIRS",
            "_STRUCTURAL_REL_NOUNS",
            "同じ",
            "ではない",
            "勝っ",
            "上司",
            "モグ",
            "ペラ",
            "チロ",
            "ポク",
            "ザビ",
            "ゾル",
            "メク",
            "負け",
            "より",
            "どれか",
            "ない",
        ] {
            assert!(
                !src.contains(forbidden),
                "forbidden literal in {name}: {forbidden}"
            );
        }
    }

    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    for (q, expected) in [
        ("富士山と北岳の高さの違いってどれくらい？", "583.12m"),
        ("富士山の標高は3776m。北岳の標高は3193m。富士山と北岳は何m違う？", "583m"),
        ("東京駅は10km。新宿駅は18km。渋谷駅は14km。距離が一番短いのはどれ？", "東京駅"),
        ("太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で2位は誰？", "花子"),
        ("太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で二番目は誰？", "花子"),
        ("富士山の方が北岳よりも高い。北岳の方が槍ヶ岳よりも高い。一番高いのはどれ？", "富士山"),
        ("鯨の全員が哺乳類に属する。鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？", "はい"),
        ("鯨の全員が哺乳類に属する。鮪の所属は哺乳類。鮪を鯨の一員と言い切れる？", "判断不能"),
    ] {
        assert_eq!(b.think(q)?, expected, "prompt={q}");
    }

    let refit = enne_rs::suite::refit(&root().join("rust_bench_v182"))?;
    assert_eq!(refit["total"].as_u64(), Some(320));
    assert_eq!(refit["correct"].as_u64(), Some(320));
    Ok(())
}

#[test]
/// v24.5 純ヘブ文字単位構文コア: syntax_core.rs / reasoner_core.rs に
/// 日本語助詞・句読点の固定文字分割や禁止語句が一切存在しないことを監査し、
/// 読点連結の複文・並列リスト読点と節境界読点の識別を E2E で検証する。
fn v245_neural_syntax_audit_and_e2e() -> Result<()> {
    let sx_src = include_str!("../src/syntax_core.rs");
    let rc_src = include_str!("../src/reasoner_core.rs");
    let r_src = include_str!("../src/reasoner.rs");
    let l_src = include_str!("../src/learning.rs");

    for (name, src) in [
        ("syntax_core.rs", sx_src),
        ("reasoner_core.rs", rc_src),
    ] {
        for forbidden_delim in [
            "'は'", "'の'", "'が'", "'に'", "'を'", "'と'", "'。'", "'、'", "'も'", "'で'", "'＝'", "'・'",
            "\"は\"", "\"の\"", "\"が\"", "\"に\"", "\"を\"", "\"と\"", "\"。\"", "\"、\"",
        ] {
            assert!(
                !src.contains(forbidden_delim),
                "forbidden delimiter literal in {name}: {forbidden_delim}"
            );
        }
    }

    for (name, src) in [
        ("syntax_core.rs", sx_src),
        ("reasoner_core.rs", rc_src),
        ("reasoner.rs", r_src),
        ("learning.rs", l_src),
    ] {
        for forbidden in [
            "VecDeque",
            "reach(",
            "path(",
            "差",
            "番目",
            "ですか？",
            "katakana",
            "synapses_v182",
            "SCALAR_PAIRS",
            "_STRUCTURAL_REL_NOUNS",
            "同じ",
            "ではない",
            "勝っ",
            "上司",
            "モグ",
            "ペラ",
            "チロ",
            "ポク",
            "ザビ",
            "ゾル",
            "メク",
            "負け",
            "より",
            "どれか",
            "ない",
        ] {
            assert!(
                !src.contains(forbidden),
                "forbidden literal in {name}: {forbidden}"
            );
        }
    }

    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    for (q, expected) in [
        ("赤、青、緑が対象で、赤は除外、青も除外。残るのは？", "緑"),
        ("太郎は花子より速く、花子は次郎より速い。一番速いのは誰？", "太郎"),
        ("佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？", "開発"),
        ("富士山は3776mで、北岳は3193m。高さの差は何m？", "583m"),
        ("鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？", "はい"),
    ] {
        assert_eq!(b.reasoner.answer(q), expected, "reasoner prompt={q}");
        assert_eq!(b.think(q)?, expected, "brain prompt={q}");
    }
    Ok(())
}

#[test]
/// v25.0 純ヘブ ブローカ発話野・メタ認知制御・零正規表現:
/// executive / metacog / planner / broca_core / brain / hippocampus_core から
/// 固定返答文字列・正規表現・Wikipedia定義そのまま引用・文字ブリッジが除かれていることを監査し、
/// 未知Wikipedia要約・中立対話多様性・句点なし前提ゲートを E2E で検証する。
fn v250_neural_broca_metacog_audit_and_e2e() -> Result<()> {
    let ex_src = include_str!("../src/executive.rs");
    let mc_full = include_str!("../src/metacog.rs");
    let mc_rt = mc_full.split("#[cfg(test)]").next().unwrap();
    let pl_src = include_str!("../src/planner.rs");
    let bc_full = include_str!("../src/broca_core.rs");
    let bc_rt = bc_full.split("#[cfg(test)]").next().unwrap();
    let br_src = include_str!("../src/brain.rs");
    let hp_full = include_str!("../src/hippocampus_core.rs");
    let hp_rt = hp_full
        .split("pub fn consolidate_and_bench")
        .next()
        .unwrap();

    for forbidden_exec in [
        "Regex",
        "EXPR",
        "NUM",
        "\\u{4F55}",
        "現在は",
        "了解しました",
        "情報がありません",
        "なるほど",
        "もう少し詳しく",
        "です。",
        "split('。')",
        "0.45",
    ] {
        assert!(
            !ex_src.contains(forbidden_exec),
            "forbidden literal in executive.rs: {forbidden_exec}"
        );
    }

    for (name, src) in [
        ("metacog.rs", mc_rt),
        ("planner.rs", pl_src),
        ("broca_core.rs", bc_rt),
        ("brain.rs", br_src),
        ("hippocampus_core.rs", hp_rt),
    ] {
        for forbidden in [
            "Regex",
            "static EXPR",
            "static NUM",
            "\\u{4F55}",
            "現在は",
            "了解しました",
            "情報がありません",
            "なるほど",
            "もう少し詳しく",
            "concise_hippocampus_answer",
            "split('。')",
            "trim_end_matches('。')",
            "ends_with(\"ですか\")",
            "score >= 0.45",
            "{name}は{val}",
        ] {
            assert!(
                !src.contains(forbidden),
                "forbidden literal in {name}: {forbidden}"
            );
        }
    }

    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    // 1. Unseen Wikipedia distillation (parenthetical readings stripped, full first definition preserved)
    let fuji = b.think("富士山について知りたい")?;
    assert!(
        fuji.starts_with("富士山") && !fuji.contains("（ふじさん）"),
        "fuji={fuji}"
    );
    let photo = b.think("光合成の仕組みを説明して")?;
    assert!(
        photo.starts_with("光合成")
            && !photo.contains("（こうごうせい")
            && (photo.contains("エネルギー") || photo.contains("反応")),
        "photo={photo}"
    );
    let kita = b.think("北岳とは何ですか？")?;
    assert!(
        kita.starts_with("北岳") && !kita.contains("（きただけ）"),
        "kita={kita}"
    );

    // 2. Unseen neutral dialogues: distinct natural polite replies (no robotic fallback)
    let mut dlg_set = std::collections::BTreeSet::new();
    for q in [
        "今日はちょっと疲れたよ。",
        "ありがとう、すごく助かったよ。",
        "明日の発表が緊張するなあ。",
        "試験に合格してすごく嬉しい！",
    ] {
        let ans = b.think(q)?;
        assert!(
            !ans.is_empty() && !ans.contains("なるほど、もう少し詳しく"),
            "q={q}, ans={ans}"
        );
        dlg_set.insert(ans);
    }
    assert!(
        dlg_set.len() >= 3,
        "dialogue replies should vary by intent: {dlg_set:?}"
    );

    // 3. Metacog store-gate without trailing period vs unpunctuated question
    b.reset_history();
    let ack = b.think("朝塔の高さは340m")?;
    assert!(ack.contains("了解しました"), "ack={ack}");
    b.reset_history();
    let q_noperiod = b.think("朝塔とは何ですか")?;
    assert!(
        !q_noperiod.contains("了解しました"),
        "question stored erroneously: {q_noperiod}"
    );

    Ok(())
}

#[test]
/// v25.0 テキスト全パイプライン通貫結合テスト (v22.0〜v25.0 + v19.0 trace):
/// 音声器官を触らず、単一 Brain セッションで四則・時計・海馬/teach・Wikipedia要約・複雑推論・
/// マルチターン作業記憶・棄権・中立雑談・クロスモード連続発話までを通貫検証する。
fn v250_full_text_pipeline_integration() -> Result<()> {
    let mut b = Brain::open(&root())?;
    let _ = b.mount_reasoning_core();

    // 1. Calc (CalcSpanTagger + BrocaDAG)
    for (q, expected) in [
        ("18 * 7 =", "126です。"),
        ("144 / 12", "12です。"),
        ("12.5 + 3.5", "16です。"),
        ("250 - 75 =", "175です。"),
        ("3つの箱に2個ずつりんごを入れたら全部で何個？", "6個です。"),
        ("1袋4個入りのパンを5袋買うと全部で何個？", "20個です。"),
        ("24本の鉛筆を6人で分けると1人何本？", "4本です。"),
        ("120枚あった紙から45枚使うと残りは何枚？", "75枚です。"),
        ("850円と320円を合わせると何円ですか？", "1170円です。"),
    ] {
        assert_eq!(b.think(q)?, expected, "calc q={q}");
    }

    // 2. Clock (ExecutiveCore + BrocaDAG)
    for q in ["今の時刻を教えて", "時計だと何時ごろ？", "いま何時？"] {
        let ans = b.think(q)?;
        assert!(
            ans.starts_with("現在は") && ans.ends_with("分です。"),
            "clock q={q} ans={ans}"
        );
    }

    // 3. Hippocampus recall, FactDistiller & Online teach + compare_plan
    assert_eq!(b.think("日本の首都とは何ですか？")?, "東京都");
    assert_eq!(b.think("日本の首都ってどこだっけ？")?, "東京都");

    let fuji = b.think("富士山について知りたい")?;
    assert!(
        fuji.starts_with("富士山") && !fuji.contains("（ふじさん）"),
        "fuji={fuji}"
    );
    let photo = b.think("光合成の仕組みを説明して")?;
    assert!(
        photo.starts_with("光合成")
            && !photo.contains("（こうごうせい")
            && (photo.contains("エネルギー") || photo.contains("反応")),
        "photo={photo}"
    );
    let kita = b.think("北岳とは何ですか？")?;
    assert!(
        kita.starts_with("北岳") && !kita.contains("（きただけ）"),
        "kita={kita}"
    );

    b.teach("蒼塔", "高さは420m。")?;
    b.teach("朱塔", "高さは180m。")?;
    let (tower_diff, tower_trace) = b.think_with_trace("蒼塔と朱塔の高さの差は何m？")?;
    assert_eq!(tower_diff, "240m");
    let tt = tower_trace.expect("tower_trace");
    assert!(
        tt.contains("420m") && tt.contains("180m") && tt.contains("240m"),
        "tt={tt}"
    );
    assert_eq!(b.think("蒼塔と朱塔はどちらが高い？")?, "蒼塔");

    // 4. Complex Single-Turn Reasoning + Trace
    let (d1, t1) = b.think_with_trace("富士山は3776mで、北岳は3193m。高さの差は何m？")?;
    assert_eq!(d1, "583m");
    assert!(t1.is_some());

    for (q, expected) in [
        ("太郎は花子より速く、花子は次郎より速い。一番速いのは誰？", "太郎"),
        (
            "太郎は花子より速い。花子は次郎より速い。次郎は健太より速い。速い順で2位は誰？",
            "花子",
        ),
        ("赤、青、緑が対象で、赤は除外、青も除外。残るのは？", "緑"),
        ("佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？", "開発"),
        (
            "鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？",
            "はい",
        ),
        (
            "鯨の全員が哺乳類に属する。鮪の所属は哺乳類。鮪を鯨の一員と言い切れる？",
            "判断不能",
        ),
        (
            "甲は乙より重く、乙は丙より重く、丙は甲より重い。一番重いのは誰？",
            "矛盾",
        ),
    ] {
        assert_eq!(b.think(q)?, expected, "reason q={q}");
    }

    // 5. Multi-Turn Working Memory, Unpunctuated Store/Query & Online learn_relation
    b.reset_history();
    assert_eq!(
        b.think("富士山の標高は3776m")?,
        "了解しました。一時記憶に保持しました。"
    );
    assert_eq!(
        b.think("北岳の標高は3193m。")?,
        "了解しました。一時記憶に保持しました。"
    );
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583m");
    b.reset_history();
    assert_eq!(b.think("富士山と北岳の標高差は何m？")?, "583.12m");

    assert_eq!(
        b.think("追跡キー＝砂時計印")?,
        "了解しました。一時記憶に保持しました。"
    );
    assert_eq!(b.think("追跡キー＝")?, "砂時計印");
    b.reset_history();
    assert!(!b.think("追跡キー＝")?.contains("砂時計印"));

    assert_eq!(b.think("日本の首都とは何ですか")?, "東京都");

    b.learn_relation("春は夏よりノビい。夏は春よりチヂい。", None);
    assert_eq!(
        b.reasoner
            .answer("甲は乙よりノビい。丙は乙よりチヂい。一番チヂいのは誰？"),
        "丙"
    );

    // 6. Abstain
    for q in [
        "架空の測定値は？",
        "明日の株価はどうなりますか？",
        "存在しない都市の人口は何人？",
    ] {
        assert_eq!(b.think(q)?, "情報がありません。", "abstain q={q}");
    }

    // 7. Neutral Everyday Dialogue
    let mut dlg_set = std::collections::BTreeSet::new();
    for q in [
        "おつかれさま、今日もいい天気だね。",
        "ありがとう、すごく助かったよ。",
        "今日はちょっと疲れたよ。",
        "明日の発表が緊張するなあ。",
        "試験に合格してすごく嬉しい！",
        "ねえ、少し話を聞いてくれる？",
    ] {
        let ans = b.think(q)?;
        assert!(
            !ans.is_empty() && !ans.contains("なるほど、もう少し詳しく"),
            "dlg q={q}, ans={ans}"
        );
        dlg_set.insert(ans);
    }
    assert!(
        dlg_set.len() >= 4,
        "expected >=4 distinct dialogue replies: {dlg_set:?}"
    );

    // 8. Interleaved Cross-Mode Session Stress (no cross-turn bleed)
    b.reset_history();
    let d_pre = b.think("おつかれさま、今日もいい天気だね。")?;
    assert!(!d_pre.is_empty());
    assert_eq!(b.think("18 * 7 =")?, "126です。");
    assert_eq!(
        b.think("記録コード＝紅蓮石")?,
        "了解しました。一時記憶に保持しました。"
    );
    assert_eq!(b.think("記録コード＝")?, "紅蓮石");
    assert_eq!(b.think("架空の測定値は？")?, "情報がありません。");
    assert_eq!(b.think("日本の首都とは何ですか？")?, "東京都");
    assert_eq!(b.think("記録コード＝")?, "紅蓮石");
    b.reset_history();
    assert_ne!(b.think("記録コード＝")?, "紅蓮石");
    assert_eq!(
        b.think("スイカはブドウより大きい。ブドウはリンゴより大きい。一番大きいのはどれ？")?,
        "スイカ"
    );

    Ok(())
}
