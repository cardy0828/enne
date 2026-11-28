//! ENNE v25.0 発話生成ファサード (`BrocaCore`, `NGBC2501`)。
//!
//! 認知応答・時刻・算術整形・定義文蒸留・日常対話応答を `BrocaCore` に委譲する。
use crate::broca_core::BrocaCore;
use anyhow::Result;
use std::path::Path;

pub struct Planner {
    core: BrocaCore,
}

impl Planner {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            core: BrocaCore::load(path)?,
        })
    }

    pub fn emit_short_ans(&self, prompt: &str, ans_bytes: &[u8]) -> String {
        self.core.emit_short_ans(prompt, ans_bytes)
    }

    pub fn emit_ack(&self, prompt: &str) -> String {
        self.core.emit_ack(prompt)
    }

    pub fn emit_abstain(&self, prompt: &str) -> String {
        self.core.emit_abstain(prompt)
    }

    pub fn emit_clock(&self, prompt: &str, hour: &str, minute: &str) -> String {
        self.core.emit_clock(prompt, hour, minute)
    }

    pub fn emit_calc(&self, prompt: &str, value: &str, unit: &str) -> String {
        self.core.emit_calc(prompt, value, unit)
    }

    pub fn emit_fact_explain(&self, prompt: &str, head: &str, body: &str) -> String {
        self.core.emit_fact_explain(prompt, head, body)
    }

    pub fn emit_fact_taught(&self, prompt: &str, body_bytes: &[u8]) -> String {
        self.core.emit_fact_taught(prompt, body_bytes)
    }

    pub fn emit_dialogue(&self, prompt: &str) -> String {
        self.core.emit_dialogue(prompt)
    }

    pub fn core(&self) -> &BrocaCore {
        &self.core
    }
}
