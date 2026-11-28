//! ENNE v25.0 統合脳インタフェース（テキスト経路のロード・推論・記憶・説明トレース）。
use crate::{
    executive::{self, ExecAction},
    executive_core::ExecutiveCore,
    hippocampus::Hippocampus,
    metacog::MetacogCore,
    planner::Planner,
    reasoner::{ReasonMode, Reasoner, ReasoningPlan},
    reasoning_core::{GenConfig, ReasoningCore},
};
use anyhow::Result;
use std::path::{Path, PathBuf};

fn resolve_artifact(root: &Path, primary: &str, fallback: &str) -> PathBuf {
    let p1 = root.join(primary);
    if p1.exists() {
        p1
    } else {
        root.join(fallback)
    }
}

pub struct Brain {
    pub hip: Hippocampus,
    pub reasoner: Reasoner,
    pub executive: ExecutiveCore,
    pub metacog: MetacogCore,
    pub planner: Planner,
    root: PathBuf,
    language_core: Option<crate::inference::Inference>,
    reasoning_core: Option<ReasoningCore>,
}

impl Brain {
    /// ディスクからテキスト経路の全 CPU 器官を読み込む。
    pub fn open(root: &Path) -> Result<Self> {
        let exec_path = resolve_artifact(
            root,
            "v220_executive_core/executive_core_v22.bin",
            "rust_bench_v182/v220/executive_core_v22.bin",
        );
        let hip_path = resolve_artifact(
            root,
            "v230_hippocampus_core/hippocampus_core_v23.bin",
            "rust_bench_v182/v230/hippocampus_core_v23.bin",
        );
        let reasoner_path = resolve_artifact(
            root,
            "v240_reasoner_core/reasoner_core_v24.bin",
            "rust_bench_v182/v240/reasoner_core_v24.bin",
        );
        let metacog_path = resolve_artifact(
            root,
            "v250_broca_metacog/metacog_core_v25.bin",
            "rust_bench_v182/v250/metacog_core_v25.bin",
        );
        let broca_path = resolve_artifact(
            root,
            "v250_broca_metacog/broca_core_v25.bin",
            "rust_bench_v182/v250/broca_core_v25.bin",
        );
        Ok(Self {
            hip: Hippocampus::open(&hip_path)?,
            reasoner: Reasoner::open(&reasoner_path)?,
            executive: ExecutiveCore::load(&exec_path)?,
            metacog: MetacogCore::load(&metacog_path)?,
            planner: Planner::load(&broca_path)?,
            root: root.into(),
            language_core: None,
            reasoning_core: None,
        })
    }

    pub fn teach(&mut self, s: &str, b: &str) -> Result<()> {
        self.hip.teach(s, b)
    }

    pub fn learn_relation(&mut self, text: &str, answer: Option<&str>) {
        if let Some(a) = answer {
            self.reasoner.learn_episode(text, a)
        } else {
            self.reasoner.observe_unlabeled(text)
        }
    }

    pub fn reset_history(&mut self) {
        self.reasoner.apoptosis()
    }

    pub fn think(&mut self, q: &str) -> Result<String> {
        Ok(self.think_with_trace(q)?.0)
    }

    /// Routes input purely via Hebbian cores and separates final reply from optional internal trace.
    pub fn think_with_trace(&mut self, q: &str) -> Result<(String, Option<String>)> {
        let trace_cfg = GenConfig {
            use_cue_bank: true,
            use_cache: true,
            short: false,
            temperature: 0.0,
            seed: 1,
        };
        let (_mode, action) = executive::route_with_core(&self.executive, q);
        match action {
            ExecAction::Clock { hour, minute } => {
                Ok((self.planner.emit_clock(q, &hour, &minute), None))
            }
            ExecAction::Calc { value, unit } => {
                Ok((self.planner.emit_calc(q, &value, &unit), None))
            }
            ExecAction::Dialogue => {
                if let Some(core) = &self.language_core {
                    let bytes = core.generate(q.as_bytes(), 60);
                    Ok((String::from_utf8_lossy(&bytes).into_owned(), None))
                } else {
                    Ok((self.planner.emit_dialogue(q), None))
                }
            }
            ExecAction::Abstain => Ok((self.planner.emit_abstain(q), None)),
            ExecAction::Fact => {
                let multi = self.reasoner.has_multi_clauses(q);
                if !multi && self.metacog.should_store_premise(q) && self.reasoner.is_premise(q) {
                    self.reasoner.observe_turn(q);
                    return Ok((self.planner.emit_ack(q), None));
                }
                if multi || self.reasoner.has_memory() {
                    let plan = self.reasoner.reason_plan(q);
                    let cand = String::from_utf8_lossy(&plan.answer_bytes);
                    if !self.reasoner.is_unknown(&cand) || multi {
                        let reply = self.planner.emit_short_ans(q, &plan.answer_bytes);
                        let trace = self
                            .reasoning_core
                            .as_ref()
                            .map(|rc| rc.generate(q, &plan, trace_cfg).text);
                        return Ok((reply, trace));
                    }
                }
                if let Some(plan) = self.hip.compare_plan(q, &self.reasoner)? {
                    let reply = self.planner.emit_short_ans(q, &plan.answer_bytes);
                    let trace = self
                        .reasoning_core
                        .as_ref()
                        .map(|rc| rc.generate(q, &plan, trace_cfg).text);
                    return Ok((reply, trace));
                }
                let hits = self.hip.lookup_hits(q, 3)?;
                if self.metacog.accept_memory_hit(&hits) {
                    let hit = &hits[0];
                    if hit.source == "wiki_predicate"
                        && !hit.head.is_empty()
                        && !q.contains(&hit.head)
                    {
                        let reply = self.planner.emit_short_ans(q, hit.head.as_bytes());
                        let trace = self.reasoning_core.as_ref().map(|rc| {
                            let plan = ReasoningPlan {
                                mode: ReasonMode::Copy,
                                answer_bytes: hit.head.as_bytes().to_vec(),
                                path_nodes: vec![hit.head.as_bytes().to_vec()],
                                auxiliary_values: vec![],
                            };
                            rc.generate(q, &plan, trace_cfg).text
                        });
                        return Ok((reply, trace));
                    }
                    if hit.source == "taught" {
                        let reply = self.planner.emit_fact_taught(q, hit.body.as_bytes());
                        let trace = self.reasoning_core.as_ref().map(|rc| {
                            let plan = ReasoningPlan {
                                mode: ReasonMode::Copy,
                                answer_bytes: hit.body.as_bytes().to_vec(),
                                path_nodes: vec![hit.head.as_bytes().to_vec()],
                                auxiliary_values: vec![],
                            };
                            rc.generate(q, &plan, trace_cfg).text
                        });
                        return Ok((reply, trace));
                    }
                    let reply = self.planner.emit_fact_explain(q, &hit.head, &hit.body);
                    let trace = self.reasoning_core.as_ref().map(|rc| {
                        let plan = ReasoningPlan {
                            mode: ReasonMode::Copy,
                            answer_bytes: reply.as_bytes().to_vec(),
                            path_nodes: vec![hit.head.as_bytes().to_vec()],
                            auxiliary_values: vec![],
                        };
                        rc.generate(q, &plan, trace_cfg).text
                    });
                    return Ok((reply, trace));
                }
                let reply = self.planner.emit_abstain(q);
                let trace = self.reasoning_core.as_ref().map(|rc| {
                    let abstain = ReasoningPlan {
                        mode: ReasonMode::Abstain,
                        answer_bytes: reply.as_bytes().to_vec(),
                        path_nodes: vec![],
                        auxiliary_values: vec![],
                    };
                    rc.generate(q, &abstain, trace_cfg).text
                });
                Ok((reply, trace))
            }
        }
    }

    pub fn explain(&mut self, q: &str) -> Result<String> {
        let (reply, trace) = self.think_with_trace(q)?;
        Ok(trace.unwrap_or(reply))
    }

    pub fn core(&self, cap: usize) -> Result<crate::inference::Inference> {
        crate::inference::Inference::load(&self.root.join("rust_bench_v182/core_index.json"), cap)
    }

    pub fn mount_core(&mut self, cap: usize) -> Result<()> {
        self.language_core = Some(self.core(cap)?);
        Ok(())
    }

    pub fn reasoning_core(&self) -> Result<ReasoningCore> {
        let path = resolve_artifact(
            &self.root,
            "v190_reasoning_core/reasoning_core_v190.bin",
            "rust_bench_v182/v190/reasoning_core_v190.bin",
        );
        ReasoningCore::load(&path)
    }

    pub fn mount_reasoning_core(&mut self) -> Result<()> {
        self.reasoning_core = Some(self.reasoning_core()?);
        Ok(())
    }

    pub fn has_reasoning_core(&self) -> bool {
        self.reasoning_core.is_some()
    }

    pub fn describe(&self) -> &str {
        "CPU: BF16 DAG / v22.0 司令塔 / v23.0 連想海馬 / v24.5 文字構文解析 / v24.0 関係推論野 / v25.0 メタ認知・ブローカ発話野 / v19.0 推論言語コア"
    }
}
