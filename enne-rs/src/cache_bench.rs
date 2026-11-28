//! v18.3 の凍結 CPU 推論、生成条件別評価とセッション監査。
use crate::{brain::Brain, cache_gen::*, inference::Inference, reasoner::ReasoningPlan};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path, time::Instant};
/// 実重みを読み込み、開発分割で混合係数を選択する。
fn load(root: &Path) -> Result<(Brain, Inference, Bridge)> {
    let brain = Brain::open(root)?;
    let dir = root.join("rust_bench_v182");
    let mut core = brain.core(6000000)?;
    core.load_adapter(&dir.join("core_adapter.bin"))?;
    let examples: Vec<Example> =
        serde_json::from_slice(&fs::read(dir.join("discourse_train.json"))?)?;
    let bridge = Bridge::train(&core, &examples)?;
    Ok((brain, core, bridge))
}
/// 単発の質問を実際の SQLite と関係推論へ接続する。
pub fn think(root: &Path, q: &str) -> Result<String> {
    let (brain, core, b) = load(root)?;
    let plan = brain
        .hip
        .compare_plan(q, &brain.reasoner)?
        .unwrap_or_else(|| brain.reasoner.reason_plan(q));
    Ok(generate(&core, &b, &plan, q.as_bytes(), config("D", false)).text)
}
/// A〜Dの比較条件を固定する。
fn config(arm: &str, short: bool) -> Config {
    Config {
        core: arm != "C",
        bridge: arm != "A",
        cache: arm == "C" || arm == "D",
        short,
        temperature: 0.,
        seed: 1,
    }
}
/// 評価フィクスチャの収録順に対応する六区分。
fn partition(i: usize) -> &'static str {
    match i {
        0..30 => "Tier1",
        30..50 => "Tier2",
        50..140 => "Copy",
        140..230 => "Reason",
        230..260 => "OOD",
        _ => "Hard",
    }
}
/// 中央にある二観測の平均も考慮した中央値。
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2] + v[(v.len() - 1) / 2]) * 0.5
}
/// 独立したターンを復元し、コアにも同一の過去前提を与える。
fn prepare(brain: &mut Brain, c: &Value) -> String {
    brain.reset_history();
    let mut context = String::new();
    if let Some(ts) = c["turns"].as_array() {
        for t in ts {
            let s = t.as_str().unwrap();
            brain.reasoner.observe_turn(s);
            context.push_str(s);
            context.push('\n')
        }
    }
    context.push_str(c["prompt"].as_str().unwrap());
    context
}
/// ゴールド回答と全根拠の文字列包含を評価する。
fn anchors(p: &ReasoningPlan, gold: &str, out: &str) -> bool {
    out.contains(gold)
        && p.path_nodes
            .iter()
            .chain(&p.auxiliary_values)
            .all(|n| out.contains(std::str::from_utf8(n).unwrap()))
}
/// 全件の生出力を保存し、独立の凍結コアで BPC を再測定する。
pub fn run(root: &Path) -> Result<()> {
    let dir = root.join("rust_bench_v182");
    let (mut brain, core, b) = load(root)?;
    let fixture: Value = serde_json::from_slice(&fs::read(dir.join("bench_320.json"))?)?;
    let cases = fixture["cases"].as_array().unwrap();
    ensure!(cases.len() == 320);
    let examples: Vec<Example> =
        serde_json::from_slice(&fs::read(dir.join("discourse_train.json"))?)?;
    let train: BTreeSet<_> = examples
        .iter()
        .filter(|e| e.split == "train")
        .map(|e| e.text.as_str())
        .collect();
    ensure!(
        examples
            .iter()
            .filter(|e| e.split == "dev")
            .all(|e| !train.contains(e.text.as_str()))
    );
    let corpus = examples
        .iter()
        .map(|e| e.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    let surface = regex::Regex::new(r"<(?:P[0-3]|V[01]|ANS)>")?.replace_all(&corpus, " ");
    let mut collisions = BTreeSet::new();
    for c in cases {
        prepare(&mut brain, c);
        let prompt = c["prompt"].as_str().unwrap();
        ensure!(!corpus.contains(prompt.trim()));
        let plan = brain.reasoner.reason_plan(prompt);
        for span in plan
            .path_nodes
            .iter()
            .chain(&plan.auxiliary_values)
            .chain([&plan.answer_bytes])
        {
            let token = std::str::from_utf8(span)?;
            if !token.is_empty() && surface.contains(token) {
                collisions.insert(token.to_owned());
            }
        }
    }
    let mut rows = vec![];
    let mut summary = json!({});
    let mut diversity = vec![];
    let mut wiki = vec![];
    for arm in ["A", "B", "C", "D"] {
        for short in [true, false] {
            let mut subset = json!({});
            let mut lats = vec![];
            let mut rates = vec![];
            let (mut answer, mut exact, mut all, mut byte_count, mut end) = (0, 0, 0, 0, 0);
            let (mut loss, mut expr, mut expr_count) = (0., 0., 0);
            for (i, c) in cases.iter().enumerate() {
                let t = Instant::now();
                let prompt = prepare(&mut brain, c);
                let q = c["prompt"].as_str().unwrap();
                let p = brain.reasoner.reason_plan(q);
                let o = generate(&core, &b, &p, prompt.as_bytes(), config(arm, short));
                let ms = t.elapsed().as_secs_f64() * 1000.;
                let gold = c["gold"].as_str().unwrap();
                let length = o.text.len();
                let hit = o.text.contains(gold);
                let eq = o.text == gold;
                let grounded = anchors(&p, gold, &o.text);
                if arm == "D" {
                    ensure!(p.answer_bytes == gold.as_bytes());
                }
                if arm != "C" {
                    ensure!(length == o.generated_core_steps);
                }
                let (bits, e, n) = score_text(&core, prompt.as_bytes(), &o);
                loss += bits;
                expr += e;
                expr_count += n;
                byte_count += length;
                lats.push(ms);
                if length > 0 {
                    rates.push(ms / length as f64)
                }
                end += o.stopped as usize;
                answer += hit as usize;
                exact += eq as usize;
                all += grounded as usize;
                let bucket = &mut subset[partition(i)];
                for (k, v) in [
                    ("n", 1),
                    ("contains", hit as usize),
                    ("exact", eq as usize),
                    ("all", grounded as usize),
                ] {
                    bucket[k] = json!(bucket[k].as_u64().unwrap_or(0) + v as u64)
                }
                rows.push(json!({"index":i,"group":c["group"],"arm":arm,"short":short,"gold":gold,"prompt":prompt,"mode":p.mode,"plan":p,"out":o,"ms":ms,"core_bits":bits,"expression_bits":e,"expression_bytes":n,"all_anchors":grounded}));
            }
            let values = json!({"exact":exact,"contains":answer,"all_anchors":all,"utf8":320,"stopped":end,"median_ms":median(lats),"median_ms_per_byte":median(rates),"bpc":loss/byte_count as f64,"expression_bpc":expr/expr_count as f64,"partitions":subset});
            eprintln!("{arm} short={short} {values}");
            summary[arm][if short { "short" } else { "full" }] = values;
        }
    }
    for i in [
        0, 12, 22, 29, 30, 36, 45, 50, 87, 135, 140, 145, 150, 230, 260, 270, 280, 290, 300, 310,
    ] {
        let c = &cases[i];
        let prompt = prepare(&mut brain, c);
        let plan = brain.reasoner.reason_plan(c["prompt"].as_str().unwrap());
        for temperature in [0., 0.45] {
            let mut samples = vec![];
            let mut intact = 0;
            for seed in 1..=5 {
                let mut cfg = config("D", false);
                cfg.temperature = temperature;
                cfg.seed = seed;
                let o = generate(&core, &b, &plan, prompt.as_bytes(), cfg);
                intact += anchors(&plan, c["gold"].as_str().unwrap(), &o.text) as usize;
                samples.push(o.text)
            }
            let unique = samples.iter().collect::<BTreeSet<_>>().len();
            if temperature == 0. {
                ensure!(unique == 1)
            }
            diversity.push(json!({"index":i,"temperature":temperature,"unique":unique,"anchors":intact,"samples":samples}));
        }
    }
    brain.reset_history();
    for (q, gold) in [
        ("富士山と北岳の標高差は何m？", "583.12m"),
        ("富士山と北岳と槍ヶ岳の標高差は何m？", "596.12m"),
        ("富士山と槍ヶ岳はどちらが標高が高い？", "富士山"),
    ] {
        let p = brain.hip.compare_plan(q, &brain.reasoner)?.unwrap();
        let o = generate(&core, &b, &p, q.as_bytes(), config("D", false));
        wiki.push(
            json!({"prompt":q,"gold":gold,"all_anchors":anchors(&p,gold,&o.text),"out":o,"plan":p}),
        );
    }
    let q = "追跡キー＝";
    let clean = brain.reasoner.reason_plan(q);
    let baseline = generate(&core, &b, &clean, q.as_bytes(), config("D", false));
    brain.reasoner.observe_turn("追跡キー＝砂時計印");
    let dirty = brain.reasoner.reason_plan(q);
    let before = generate(&core, &b, &dirty, q.as_bytes(), config("D", false));
    brain.reset_history();
    let after = generate(
        &core,
        &b,
        &brain.reasoner.reason_plan(q),
        q.as_bytes(),
        config("D", false),
    );
    let bleed = after.text != baseline.text || after.text.contains("砂時計印");
    ensure!(before.text.contains("砂時計印"));
    let forbidden = [
        "なので",
        "答えは",
        "循環",
        "否定",
        "候補",
        "該当する",
        "同じ関係",
    ];
    let hits: Vec<_> = forbidden
        .iter()
        .filter(|s| include_str!("cache_gen.rs").contains(**s))
        .collect();
    let previous: Value =
        serde_json::from_slice(&fs::read(dir.join("results_rust_vs_python.json"))?)?;
    let source = concat!(include_str!("reasoner.rs"), include_str!("learning.rs"));
    let lexicon: Vec<_> = previous["audit"]["literals_checked"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| source.contains(s.as_str().unwrap()))
        .collect();
    let results = json!({"summary":summary,"rows":rows,"diversity":diversity,"wiki":wiki,"audit":{"corpus_anchor_collisions":collisions,"generator_hits":hits,"reasoner_hits":lexicon,"bleed":bleed,"baseline":baseline,"before":before,"after":after},"alive":core.alive.iter().filter(|&&v|v!=0).count(),"dev_losses":b.dev,"w_core":b.weight,"peak_mib":crate::memory::working_set().1});
    fs::write(
        dir.join("results_v183_cache_core_gen.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    Ok(())
}
