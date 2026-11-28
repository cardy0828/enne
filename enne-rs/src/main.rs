//! ENNE v25.0 テキスト経路 CLI エントリポイント（学習・ベンチマーク・対話検証）。
use anyhow::{Context, Result, bail, ensure};
use enne_rs::{brain::Brain, core_dag::NeurogenesisGraph, inference::Inference};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
/// 起動済み器官への同一入力を反復し、中央値と最終結果を返す。
fn measure<T>(mut f: impl FnMut() -> Result<T>, n: usize) -> Result<(T, f64)> {
    let mut last = f()?;
    let mut samples = vec![];
    for _ in 0..n {
        let start = Instant::now();
        last = std::hint::black_box(f()?);
        samples.push(start.elapsed().as_secs_f64() * 1000.)
    }
    samples.sort_by(f64::total_cmp);
    Ok((last, samples[n / 2]))
}
/// 全320問をセッション単位で評価し、事実・計算も同じプロセスで測定する。
fn bench(root: &Path, brain: &mut Brain) -> Result<Value> {
    let out = root.join("rust_bench_v182");
    let suite: Value = serde_json::from_slice(&fs::read(out.join("bench_320.json"))?)?;
    let mut rows = vec![];
    let mut right = 0;
    let mut total = 0.;
    for c in suite["cases"].as_array().unwrap() {
        let q = c["prompt"].as_str().unwrap();
        let (pred, ms) = measure(
            || {
                brain.reset_history();
                if let Some(turns) = c["turns"].as_array() {
                    for turn in turns {
                        brain.reasoner.observe_turn(turn.as_str().unwrap())
                    }
                }
                Ok(brain.reasoner.answer(q))
            },
            5,
        )?;
        total += ms;
        right += usize::from(c["gold"] == pred);
        rows.push(json!({"pred":pred,"ms":ms}));
    }
    brain.reset_history();
    let mut results = json!({"reason":{"correct":right,"total":rows.len(),"ms_per_query":total/rows.len() as f64,"rows":rows}});
    for (key, q) in [
        ("calc", "3つの箱に2個ずつりんごを入れたら全部で何個？"),
        ("fact", "日本の首都とは何ですか？"),
        ("bridge", "富士山と北岳の標高差は何m？"),
    ] {
        let (answer, ms) = measure(|| brain.think(q), 51)?;
        results[key] = json!({"answer":answer,"median_ms":ms});
    }
    fs::write(
        out.join("rust_organs.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    Ok(results)
}
/// CLI の status / think / bench / train / infer を実行する。
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let root = std::env::var_os("ENNE_ROOT")
        .or_else(|| std::env::var_os("NEUROGRAPH_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_path_buf()
        });
    let out = root.join("rust_bench_v182");
    match args.first().map(String::as_str).unwrap_or("status") {
        "v183-bench" => enne_rs::cache_bench::run(&root)?,
        "v183-think" => println!(
            "{}",
            enne_rs::cache_bench::think(&root, &args[1..].join(" "))?
        ),
        "v190-save" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v190_reasoning_core"));
            let ckpt_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| corpus_dir.join("reasoning_core_v190.bin"));
            let epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            enne_rs::reasoning_core::train_and_save(
                &corpus_dir,
                &ckpt_path,
                5_000_000,
                epochs,
            )?;
        }
        "v190-bench" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v190_reasoning_core"));
            let out_json = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| corpus_dir.join("results_v190.json"));
            let cap: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
            enne_rs::reasoning_core::run_v190_bench(&root, &corpus_dir, &out_json, cap)?;
        }
        "v220-train" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v220_executive_core"));
            let ckpt_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| corpus_dir.join("executive_core_v22.bin"));
            let out_json = args
                .get(3)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("v220/results_v220.json"));
            let epochs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
            let rep = enne_rs::executive_core::train_and_bench(
                &corpus_dir,
                &ckpt_path,
                &out_json,
                epochs,
            )?;
            println!("{}", serde_json::to_string_pretty(&rep)?);
        }
        "v230-build" => {
            let ckpt_path = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v230_hippocampus_core/hippocampus_core_v23.bin"));
            let out_json = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("v230/results_v230.json"));
            let rep =
                enne_rs::hippocampus_core::consolidate_and_bench(&root, &ckpt_path, &out_json)?;
            println!("{}", serde_json::to_string_pretty(&rep)?);
        }
        "v245-train" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v245_syntax_core"));
            let ckpt_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| corpus_dir.join("syntax_core_v245.bin"));
            let out_json = args
                .get(3)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("v245/results_v245.json"));
            let epochs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
            let rep = enne_rs::syntax_core::train_and_bench(
                &corpus_dir,
                &ckpt_path,
                &out_json,
                epochs,
            )?;
            println!("{}", serde_json::to_string_pretty(&rep)?);
        }
        "v240-train" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v240_reasoner_core"));
            let ckpt_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| corpus_dir.join("reasoner_core_v24.bin"));
            let out_json = args
                .get(3)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("v240/results_v240.json"));
            let epochs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let rep = enne_rs::reasoner_core::train_and_bench(
                &root,
                &corpus_dir,
                &ckpt_path,
                &out_json,
                epochs,
            )?;
            println!("{}", serde_json::to_string_pretty(&rep)?);
        }
        "v250-train" => {
            let corpus_dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("v250_broca_metacog"));
            let out_json = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("v250/results_v250.json"));
            let rep = enne_rs::broca_core::train_and_bench(&corpus_dir, &out_json)?;
            println!("{}", serde_json::to_string_pretty(&rep)?);
        }
        "v250-verify" => {
            let t_load = Instant::now();
            let mut b = Brain::open(&root)?;
            b.mount_reasoning_core()?;
            let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

            let mut rows = Vec::new();
            let mut passed = 0usize;
            let mut run_case = |cat: &str,
                                prompt: &str,
                                b: &mut Brain,
                                check: &dyn Fn(&str, Option<&str>) -> bool|
             -> Result<()> {
                let t0 = Instant::now();
                let (reply, trace) = b.think_with_trace(prompt)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ok = check(&reply, trace.as_deref());
                if ok {
                    passed += 1;
                }
                rows.push(json!({
                    "category": cat,
                    "prompt": prompt,
                    "reply": reply,
                    "internal_trace": trace,
                    "ms": ms,
                    "ok": ok,
                }));
                Ok(())
            };

            // 1. Calc
            run_case("Calc", "18 * 7 =", &mut b, &|r, _| r == "126です。")?;
            run_case("Calc", "12.5 + 3.5", &mut b, &|r, _| r == "16です。")?;
            run_case("Calc", "3つの箱に2個ずつりんごを入れたら全部で何個？", &mut b, &|r, _| r == "6個です。")?;
            run_case("Calc", "1袋4個入りのパンを5袋買うと全部で何個？", &mut b, &|r, _| r == "20個です。")?;
            run_case("Calc", "24本の鉛筆を6人で分けると1人何本？", &mut b, &|r, _| r == "4本です。")?;

            // 2. Clock
            run_case("Clock", "今の時刻を教えて", &mut b, &|r, _| {
                r.starts_with("現在は") && r.ends_with("分です。")
            })?;

            // 3. Hippocampus + FactDistiller + online teach
            run_case("Wiki", "日本の首都とは何ですか？", &mut b, &|r, _| r == "東京都")?;
            run_case("Wiki", "日本の首都とは何ですか", &mut b, &|r, _| r == "東京都")?;
            run_case("Wiki", "富士山について知りたい", &mut b, &|r, _| {
                r.starts_with("富士山") && !r.contains("（ふじさん）")
            })?;
            run_case("Wiki", "光合成の仕組みを説明して", &mut b, &|r, _| {
                r.starts_with("光合成") && !r.contains("（こうごうせい") && (r.contains("エネルギー") || r.contains("反応"))
            })?;
            run_case("Wiki", "北岳とは何ですか？", &mut b, &|r, _| {
                r.starts_with("北岳") && !r.contains("（きただけ）")
            })?;
            b.teach("蒼塔", "高さは420m。")?;
            b.teach("朱塔", "高さは180m。")?;
            run_case("TeachCompare", "蒼塔と朱塔の高さの差は何m？", &mut b, &|r, t| {
                r == "240m" && t.is_some_and(|s| s.contains("240m"))
            })?;
            run_case("TeachCompare", "蒼塔と朱塔はどちらが高い？", &mut b, &|r, _| r == "蒼塔")?;

            // 4. Reasoning
            run_case("Reasoning", "富士山は3776mで、北岳は3193m。高さの差は何m？", &mut b, &|r, t| {
                r == "583m" && t.is_some()
            })?;
            run_case("Reasoning", "太郎は花子より速く、花子は次郎より速い。一番速いのは誰？", &mut b, &|r, _| r == "太郎")?;
            run_case("Reasoning", "赤、青、緑が対象で、赤は除外、青も除外。残るのは？", &mut b, &|r, _| r == "緑")?;
            run_case("Reasoning", "佐藤の担当は開発で、田中は佐藤と同じ担当。田中の担当は何？", &mut b, &|r, _| r == "開発")?;
            run_case("Reasoning", "鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？", &mut b, &|r, _| r == "はい")?;
            run_case("Reasoning", "甲は乙より重く、乙は丙より重く、丙は甲より重い。一番重いのは誰？", &mut b, &|r, _| r == "矛盾")?;

            // 5. Multi-turn memory & Copy & Reset
            b.reset_history();
            run_case("MultiTurn", "富士山の標高は3776m", &mut b, &|r, _| {
                r == "了解しました。一時記憶に保持しました。"
            })?;
            run_case("MultiTurn", "北岳の標高は3193m。", &mut b, &|r, _| {
                r == "了解しました。一時記憶に保持しました。"
            })?;
            run_case("MultiTurn", "富士山と北岳の標高差は何m？", &mut b, &|r, _| r == "583m")?;
            run_case("Copy", "追跡キー＝砂時計印", &mut b, &|r, _| {
                r == "了解しました。一時記憶に保持しました。"
            })?;
            run_case("Copy", "追跡キー＝", &mut b, &|r, _| r == "砂時計印")?;
            b.reset_history();
            run_case("PostReset", "富士山と北岳の標高差は何m？", &mut b, &|r, _| r == "583.12m")?;

            // 6. Abstain
            run_case("Abstain", "架空の測定値は？", &mut b, &|r, _| r == "情報がありません。")?;
            run_case("Abstain", "明日の株価はどうなりますか？", &mut b, &|r, _| r == "情報がありません。")?;

            // 7. Dialogue
            let check_dlg = |r: &str, _: Option<&str>| {
                !r.is_empty() && !r.contains("なるほど、もう少し詳しく")
            };
            run_case("Dialogue", "今日はちょっと疲れたよ。", &mut b, &check_dlg)?;
            run_case("Dialogue", "ありがとう、すごく助かったよ。", &mut b, &check_dlg)?;
            run_case("Dialogue", "明日の発表が緊張するなあ。", &mut b, &check_dlg)?;
            run_case("Dialogue", "試験に合格してすごく嬉しい！", &mut b, &check_dlg)?;
            let dialogue_unique = rows
                .iter()
                .filter(|row| row["category"] == "Dialogue")
                .filter_map(|row| row["reply"].as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len();

            let (rss, peak) = enne_rs::memory::working_set();
            let total = rows.len();
            let report = json!({
                "load_ms": load_ms,
                "rss_mib": rss,
                "peak_rss_mib": peak,
                "passed": passed,
                "total": total,
                "dialogue_unique": dialogue_unique,
                "cases": rows,
            });
            fs::create_dir_all(out.join("v250"))?;
            fs::write(
                out.join("v250/integration_report.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            ensure!(
                passed == total && dialogue_unique >= 3,
                "v250-verify failed: {passed}/{total}, dialogue_unique={dialogue_unique}"
            );
        }
        "explain" => {
            let mut b = Brain::open(&root)?;
            b.mount_reasoning_core()?;
            println!("{}", b.explain(&args[1..].join(" "))?);
        }
        "v190-demo" => {
            let t_load = Instant::now();
            let mut b = Brain::open(&root)?;
            b.mount_reasoning_core()?;
            let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

            let mut demo_rows = vec![];
            for q in [
                "富士山と北岳の標高差は何m？",
                "富士山と北岳と槍ヶ岳の標高差は何m？",
                "富士山と槍ヶ岳はどちらが標高が高い？",
                "日本の首都とは何ですか？",
                "スイカはブドウより大きい。ブドウはリンゴより大きい。リンゴはミカンより大きい。ミカンはイチゴより大きい。一番大きいのはどれ？",
                "架空の測定値は？",
            ] {
                b.reset_history();
                let t0 = Instant::now();
                let (plain, trace) = b.think_with_trace(q)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                demo_rows.push(json!({"prompt": q, "reply": plain, "internal_trace": trace, "ms": ms}));
            }
            // Multi-turn premise -> reasoned answer -> reset check
            b.reset_history();
            let ack = b.think("朝塔の高さは340m。")?;
            let ack2 = b.think("夕塔の高さは200m。")?;
            let mt_ans = b.think("朝塔と夕塔の高さの差は何m？")?;
            b.reset_history();
            let (rss, peak) = enne_rs::memory::working_set();
            let payload = json!({
                "load_ms": load_ms,
                "rss_mib": rss,
                "peak_rss_mib": peak,
                "queries": demo_rows,
                "multiturn": {"ack1": ack, "ack2": ack2, "answer": mt_ans},
            });
            fs::create_dir_all(out.join("v190"))?;
            fs::write(
                out.join("v190/e2e_demo.json"),
                serde_json::to_vec_pretty(&payload)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        "mega-train" => {
            let train_p = PathBuf::from(args.get(1).context("train_bin path")?);
            let val_p = PathBuf::from(args.get(2).context("val_bin path")?);
            let out_p = PathBuf::from(args.get(3).context("out_json path")?);
            let cap: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(35_000_000);
            let batch: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(512);
            let max_bytes: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
            let ckpt_p = args
                .get(7)
                .filter(|s| *s != "none" && !s.is_empty())
                .map(PathBuf::from);
            let rep = enne_rs::mega_core::run_mega_experiment(
                &train_p,
                &val_p,
                &out_p,
                ckpt_p.as_deref(),
                cap,
                batch,
                max_bytes,
            )?;
            println!(
                "{}",
                json!({
                    "alive": rep.final_alive_neurons,
                    "synaptic_params": rep.final_synaptic_params,
                    "val_bpc": rep.final_val_bpc,
                    "elapsed_s": rep.total_elapsed_s,
                    "avg_bytes_per_s": rep.avg_bytes_per_s,
                    "infer_b1_ms": rep.infer_b1_median_ms,
                    "peak_rss_mib": rep.peak_rss_mib
                })
            );
        }
        "mega-ref-old" => {
            // Minimal reference: evaluate frozen 4.58M core on first 32KB of val_bin + 6 prompts
            let val_p = PathBuf::from(args.get(1).context("val_bin path")?);
            let out_p = PathBuf::from(args.get(2).context("out_json path")?);
            let val = fs::read(&val_p)?;
            let slice = &val[..val.len().min(32768)];
            let core = Inference::load(&out.join("core_index.json"), 6000000)?;
            let mut st = core.init_state();
            let mut nll = 0.0f64;
            for i in 0..slice.len() - 1 {
                let (logits, _) = core.step_byte(&mut st, slice[i]);
                nll += enne_rs::cache_gen::bits(&logits, slice[i + 1] as usize);
            }
            let val_bpc = nll / (slice.len() - 1) as f64;
            let prompts = [
                "日本の首都は",
                "富士山は",
                "人工知能とは",
                "京都府は",
                "江戸時代は",
                "物理学において",
            ];
            let mut samples = vec![];
            for p in prompts {
                let generated = core.generate(p.as_bytes(), 120);
                samples.push(json!({
                    "prompt": p,
                    "greedy": String::from_utf8_lossy(&generated)
                }));
            }
            let payload = json!({
                "alive_neurons": core.alive.iter().filter(|&&v| v != 0).count(),
                "val_bytes_scored": slice.len() - 1,
                "val_bpc": val_bpc,
                "samples": samples
            });
            fs::write(&out_p, serde_json::to_vec_pretty(&payload)?)?;
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        "train" => {
            let size = args.get(1).context("入力サイズ")?.parse()?;
            let batch = args.get(2).context("ストリーム数")?.parse()?;
            let cap: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(150000);
            let data_path = args
                .get(4)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join("wiki.bin"));
            let basis_path = args
                .get(5)
                .map(PathBuf::from)
                .unwrap_or_else(|| out.join(format!("basis_{size}.bin")));
            let input = fs::read(&data_path)?;
            let mut g = NeurogenesisGraph::<128>::load(&basis_path, cap)?;
            let start = Instant::now();
            let (n, bpc, alive) =
                g.run_stream(input.get(..size).context("入力サイズ超過")?, batch, true)?;
            let time = start.elapsed().as_secs_f64();
            let (rss, peak) = enne_rs::memory::working_set();
            println!(
                "{}",
                json!({"elapsed_s":time,"input_bytes_per_s":size as f64/time,"predictions":n,"alive_hidden":alive,"bpc":bpc,"pool_cap":cap,"rss_mib":rss,"peak_rss_mib":peak})
            );
        }
        "infer" => {
            let cap = args.get(1).context("プールサイズ")?.parse()?;
            let start = Instant::now();
            let core = Inference::load(&out.join("core_index.json"), cap)?;
            let load = start.elapsed().as_secs_f64() * 1000.;
            let input = fs::read(out.join("wiki.bin"))?;
            let (bytes, ms) = measure(|| Ok(core.generate(&input[..75], 60)), 9)?;
            println!(
                "{}",
                json!({"median_ms":ms,"load_ms":load,"rss_mib":enne_rs::memory::working_set().0,"peak_rss_mib":enne_rs::memory::working_set().1,"ms_per_generated_byte":ms/60.,"ms_per_step":ms/134.,"alive":core.alive.iter().filter(|x|**x!=0).count(),"generated":bytes})
            );
        }
        cmd => {
            let mut b = Brain::open(&root)?;
            match cmd {
                "status" => println!("{}", b.describe()),
                "think" => println!("{}", b.think(&args[1..].join(" "))?),
                "boot" => {
                    let (mem, peak) = enne_rs::memory::working_set();
                    println!("{}", json!({"rss_mib":mem,"peak_rss_mib":peak}));
                }
                "organs" | "bench" => {
                    let results = bench(&root, &mut b)?;
                    println!("{} / 320", results["reason"]["correct"]);
                    if cmd == "bench" {
                        enne_rs::suite::run(&out, results)?
                    }
                }
                _ => bail!("unknown command {cmd}"),
            }
        }
    }
    Ok(())
}
