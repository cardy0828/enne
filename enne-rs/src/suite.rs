//! ENNE v25.0 の学習・推論・器官ベンチマークを比較集計する。
use crate::reasoner::Reasoner;
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
#[derive(Deserialize)]
struct Case {
    prompt: String,
    gold: String,
    #[serde(default)]
    turns: Vec<String>,
}
#[derive(Deserialize)]
struct Fixture {
    unlabeled: Vec<String>,
    episodes: Vec<(String, String)>,
    cases: Vec<Case>,
}
/// 保存済み学習例だけで空状態から学習し、独立した評価入力への予測を保存する。
pub fn refit(out: &Path) -> Result<Value> {
    let fixture: Fixture = serde_json::from_slice(&fs::read(out.join("bench_320.json"))?)?;
    let mut brain = Reasoner::load_synapses(&out.join("blank_synapses.json"))?;
    for doc in &fixture.unlabeled {
        brain.observe_unlabeled(doc)
    }
    for (q, a) in &fixture.episodes {
        brain.learn_episode(q, a)
    }
    let mut rows = vec![];
    let mut correct = 0;
    for c in &fixture.cases {
        brain.apoptosis();
        for t in &c.turns {
            brain.observe_turn(t)
        }
        let pred = brain.answer(&c.prompt);
        correct += usize::from(pred == c.gold);
        rows.push(pred);
    }
    let data = json!({"episodes":fixture.episodes.len(),"unlabeled":fixture.unlabeled.len(),"total":fixture.cases.len(),"correct":correct,"predictions":rows});
    fs::write(
        out.join("refit_rust.json"),
        serde_json::to_vec_pretty(&data)?,
    )?;
    Ok(data)
}

/// 独立した子プロセスで計測し、終了コードと単一のJSON結果を検証する。
fn child(args: &[&str]) -> Result<Value> {
    child_with_threads(args, None)
}
/// 計測対象プロセスだけにRayonのワーカー数を指定する。
fn child_with_threads(args: &[&str], threads: Option<usize>) -> Result<Value> {
    let mut command = Command::new(std::env::current_exe()?);
    if let Some(n) = threads {
        command.env("RAYON_NUM_THREADS", n.to_string());
    }
    let output = command.args(args).output()?;
    ensure!(
        output.status.success(),
        "{}: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}
/// Rust 全測定を実行し、同じ入力に対する Python の記録と一緒に保存する。
pub fn run(out: &Path, organs: Value) -> Result<()> {
    let reference =
        |name: &str| -> Result<Value> { Ok(serde_json::from_slice(&fs::read(out.join(name))?)?) };
    let mut training = json!({});
    for (size, batch) in [
        ("120000", "1"),
        ("120000", "32"),
        ("120000", "256"),
        ("120000", "1024"),
        ("1000000", "256"),
    ] {
        let row = child(&["train", size, batch])?;
        training[format!("{size}_{batch}")] = row;
    }
    let mut parallel = json!({});
    for b in ["32", "256", "1024"] {
        parallel[b] = child_with_threads(&["train", "120000", b], Some(16))?;
    }
    let mut infer = json!({});
    for cap in ["150000", "6000000"] {
        let rust = child(&["infer", cap])?;
        let py = reference(&format!("python_infer_{cap}.json"))?;
        infer[cap] = json!({"generated_equal":rust["generated"]==py["generated"],"speedup":py["median_ms"].as_f64().unwrap()/rust["median_ms"].as_f64().unwrap(),"rust":rust,"python":py});
    }
    let mut boot = vec![];
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let mem = child(&["boot"])?;
        boot.push(json!({"process_elapsed_ms":start.elapsed().as_secs_f64()*1000.,"memory":mem}));
    }
    let py = reference("python_organs.json")?;
    let train_py = reference("python_training.json")?;
    let mut train_comparison = json!({});
    for b in ["1", "32", "256", "1024"] {
        let r = &training[format!("120000_{b}")];
        let p = &train_py[b];
        train_comparison[b] = json!({"speedup":p["elapsed_s"].as_f64().unwrap()/r["elapsed_s"].as_f64().unwrap(),"delta_bpc":r["bpc"].as_f64().unwrap()-p["bpc"].as_f64().unwrap(),"delta_hidden":r["alive_hidden"].as_i64().unwrap()-p["alive_hidden"].as_i64().unwrap()});
    }
    let mut organ_comparison = json!({});
    for key in ["calc", "fact", "bridge"] {
        let r = &organs[key];
        let p = &py[key];
        let matches = r["answer"] == p["answer"];
        organ_comparison[key] = json!({"output_fields_match":matches,"speedup":p["median_ms"].as_f64().unwrap()/r["median_ms"].as_f64().unwrap()});
    }
    let audit_words = [
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
    ];
    let text = concat!(include_str!("reasoner.rs"), include_str!("learning.rs"));
    let matches: Vec<_> = audit_words
        .iter()
        .filter(|word| text.contains(**word))
        .collect();
    let results = json!({"rust":organs,"python":py,"refit":{"rust":refit(out)?,"python":reference("refit_python.json")?},"organ_comparison":organ_comparison,"training_rust":training,"training_python":train_py,"training_comparison":train_comparison,"rust_parallel_16":parallel,"inference":infer,"boot_rust":boot,"boot_python":reference("python_boot.json")?,"reference_user_supplied_ms":{"rtx3060":252.7,"python_cpu_earlier":394.6},"audit":{"scanned":["src/reasoner.rs","src/learning.rs"],"literals_checked":audit_words,"hits":matches},"runtime":{"rust_threads":rayon::current_num_threads(),"python_training_torch_threads":1,"blas_threads":1,"os":std::env::consts::OS,"arch":std::env::consts::ARCH}});
    fs::write(
        out.join("results_rust_vs_python.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    Ok(())
}
