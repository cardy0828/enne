//! ENNE v25.0 メタ認知制御コア (`MetacogCore`, `NGMC2501`)。
//!
//! 1) 記憶保持判定 (`should_store_premise`): 2層 オヤ・ヘブ k-WTA + 局所デルタ 2値分類器。
//! 2) 根拠採択判定 (`accept_memory_hit`): 想起テレメトリ上の局所デルタパーセプトロン。
use crate::{core_dag::mix, hippocampus::Fact};
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

pub const MC_DIN: usize = 128;
pub const MC_HIDDEN: usize = 64;
pub const MC_TOP_K: usize = 8;
pub const MC_BUCKETS: usize = 16384;
pub const MC_CONF_DIM: usize = 7;
const MAGIC: &[u8; 8] = b"NGMC2501";

#[derive(Deserialize)]
pub struct MetacogActRow {
    pub text: String,
    pub store: usize,
    #[serde(default)]
    pub split: String,
}

#[derive(Deserialize)]
pub struct MetacogConfRow {
    pub top_score: f64,
    pub margin: f64,
    pub is_predicate: usize,
    pub is_taught: usize,
    pub is_exact: usize,
    pub accept: usize,
    #[serde(default)]
    pub split: String,
}

#[inline(always)]
fn script_class(c: char) -> u64 {
    if c.is_whitespace() {
        1
    } else if c.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&c) {
        2
    } else if c.is_ascii_alphabetic() {
        3
    } else if ('\u{3040}'..='\u{309F}').contains(&c) {
        4
    } else if ('\u{30A0}'..='\u{30FF}').contains(&c) {
        5
    } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
        6
    } else {
        7
    }
}

fn extract_gate_features(raw: &str) -> (Vec<usize>, [f32; MC_DIN]) {
    let q: String = raw.trim().nfkc().collect();
    let chars: Vec<char> = q.chars().collect();
    let n = chars.len();
    let mask = MC_BUCKETS - 1;
    let mut feats = Vec::with_capacity(n * 3 + 24);
    let push_f = |feats: &mut Vec<usize>, tag: u64, val: u64| {
        feats.push((mix(tag ^ val.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize) & mask);
    };

    for i in 0..n {
        let c0 = chars[i] as u64;
        push_f(&mut feats, 0x101, c0);
        if i + 1 < n {
            let c1 = chars[i + 1] as u64;
            push_f(&mut feats, 0x202, (c0 << 21) ^ c1);
            if i + 2 < n {
                let c2 = chars[i + 2] as u64;
                push_f(&mut feats, 0x303, mix((c0 << 21) ^ c1) ^ c2);
            }
        }
    }

    // Explicit trailing boundary cues (zero language literals: captures any trailing symbol/ending)
    for k in 1..=6usize {
        if let Some(idx) = n.checked_sub(k) {
            push_f(&mut feats, 0x700 + k as u64, chars[idx] as u64);
            push_f(&mut feats, 0x800 + k as u64, script_class(chars[idx]));
        }
    }
    if n >= 2 {
        push_f(
            &mut feats,
            0x901,
            ((chars[n - 2] as u64) << 21) ^ (chars[n - 1] as u64),
        );
    }
    if n >= 3 {
        push_f(
            &mut feats,
            0x902,
            mix(((chars[n - 3] as u64) << 21) ^ (chars[n - 2] as u64)) ^ (chars[n - 1] as u64),
        );
    }
    if n >= 4 {
        push_f(
            &mut feats,
            0x903,
            mix(((chars[n - 4] as u64) << 21) ^ (chars[n - 3] as u64))
                ^ ((chars[n - 2] as u64) << 21)
                ^ (chars[n - 1] as u64),
        );
    }

    feats.sort_unstable();
    feats.dedup();

    let mut x = [0.0f32; MC_DIN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x517c_c1b7_2722_0a95);
        for _ in 0..4 {
            let d = (h as usize) & (MC_DIN - 1);
            let sign = if ((h >> 8) & 1) == 0 { 1.0f32 } else { -1.0f32 };
            x[d] += sign;
            h = mix(h ^ 0x9e37_79b9_7f4a_7c15);
        }
    }
    let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for v in &mut x {
            *v *= inv;
        }
    }
    (feats, x)
}

pub struct MetacogCore {
    pub trained: bool,
    pub w1: Vec<f32>,
    pub theta: [f32; MC_HIDDEN],
    pub w2: Vec<f32>,
    pub w_dir: Vec<f32>,
    pub bias: [f32; 2],
    pub w_conf: [f32; MC_CONF_DIM],
}

impl Default for MetacogCore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetacogCore {
    pub fn new() -> Self {
        let mut w1 = vec![0.0f32; MC_HIDDEN * MC_DIN];
        let mut st = 0x1319_8a2e_0370_7344u64;
        for h in 0..MC_HIDDEN {
            let mut n2 = 0.0f32;
            for d in 0..MC_DIN {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let u = ((st >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
                w1[h * MC_DIN + d] = u;
                n2 += u * u;
            }
            let inv = 1.0 / (n2.sqrt() + 1e-8);
            for d in 0..MC_DIN {
                w1[h * MC_DIN + d] *= inv;
            }
        }
        Self {
            trained: false,
            w1,
            theta: [0.0f32; MC_HIDDEN],
            w2: vec![0.0f32; MC_HIDDEN * 2],
            w_dir: vec![0.0f32; MC_BUCKETS * 2],
            bias: [0.0f32; 2],
            w_conf: [-2.0, 4.0, 0.5, 1.0, 0.5, 2.0, 0.5],
        }
    }

    #[inline]
    fn hidden_act(&self, x: &[f32; MC_DIN]) -> [(usize, f32); MC_TOP_K] {
        let mut scores = [(0usize, 0.0f32); MC_HIDDEN];
        for h in 0..MC_HIDDEN {
            let row = &self.w1[h * MC_DIN..(h + 1) * MC_DIN];
            let mut s = 0.0f32;
            for d in 0..MC_DIN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); MC_TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..MC_TOP_K {
            top[i] = scores[i];
            norm2 += scores[i].1 * scores[i].1;
        }
        if norm2 > 1e-8 {
            let inv = 1.0 / norm2.sqrt();
            for item in &mut top {
                item.1 *= inv;
            }
        }
        top
    }

    fn store_logits(&self, feats: &[usize], x: &[f32; MC_DIN]) -> [f32; 2] {
        let winners = self.hidden_act(x);
        let mut logits = self.bias;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * 2;
                logits[0] += act * self.w2[base];
                logits[1] += act * self.w2[base + 1];
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in feats {
            let base = f * 2;
            logits[0] += inv_f * self.w_dir[base];
            logits[1] += inv_f * self.w_dir[base + 1];
        }
        logits
    }

    /// Pure Hebbian gate classifying whether a single-clause utterance should be kept in working memory.
    pub fn should_store_premise(&self, raw: &str) -> bool {
        let q = raw.trim();
        if q.is_empty() || !self.trained {
            return false;
        }
        let (feats, x) = extract_gate_features(q);
        let logits = self.store_logits(&feats, &x);
        logits[1] > logits[0]
    }

    #[inline]
    fn conf_score(&self, feat: &[f32; MC_CONF_DIM]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..MC_CONF_DIM {
            s += self.w_conf[i] * feat[i];
        }
        s
    }

    /// Neural confidence gate deciding whether Hippocampus hits pass grounding threshold.
    pub fn accept_memory_hit(&self, hits: &[Fact]) -> bool {
        let Some(top) = hits.first() else {
            return false;
        };
        let top_score = top.score as f32;
        let margin = if hits.len() >= 2 {
            (top.score - hits[1].score).max(0.0) as f32
        } else {
            top_score
        };
        let is_pred = if top.source == "wiki_predicate" { 1.0 } else { 0.0 };
        let is_taught = if top.source == "taught" { 1.0 } else { 0.0 };
        let is_exact = if is_pred == 0.0 && is_taught == 0.0 {
            1.0
        } else {
            0.0
        };
        let feat = [
            1.0f32,
            top_score,
            margin,
            top_score * top_score,
            is_pred,
            is_taught,
            is_exact,
        ];
        self.conf_score(&feat) > 0.0
    }

    /// Trains both pure-Hebbian Metacog heads and returns `(val_act_acc, val_conf_acc)`.
    pub fn train(&mut self, act_rows: &[MetacogActRow], conf_rows: &[MetacogConfRow]) -> (f64, f64) {
        let train_act: Vec<&MetacogActRow> = act_rows.iter().filter(|r| r.split != "val").collect();
        let val_act: Vec<&MetacogActRow> = act_rows.iter().filter(|r| r.split == "val").collect();

        if !train_act.is_empty() {
            let prep: Vec<(Vec<usize>, [f32; MC_DIN], usize)> = train_act
                .iter()
                .map(|r| {
                    let (f, x) = extract_gate_features(&r.text);
                    (f, x, usize::from(r.store > 0))
                })
                .collect();

            let target_rate = MC_TOP_K as f32 / MC_HIDDEN as f32;
            for pass in 0..2 {
                let eta = 0.05 / (1.0 + 0.5 * pass as f32);
                for (_, x, _) in &prep {
                    let winners = self.hidden_act(x);
                    let mut won = [false; MC_HIDDEN];
                    for &(h, act) in &winners {
                        if act <= 0.0 {
                            continue;
                        }
                        won[h] = true;
                        let row = &mut self.w1[h * MC_DIN..(h + 1) * MC_DIN];
                        let mut n2 = 0.0f32;
                        for d in 0..MC_DIN {
                            let nv = row[d] + eta * act * (x[d] - act * row[d]);
                            row[d] = nv;
                            n2 += nv * nv;
                        }
                        let inv = 1.0 / (n2.sqrt() + 1e-8);
                        for d in 0..MC_DIN {
                            row[d] *= inv;
                        }
                    }
                    for h in 0..MC_HIDDEN {
                        let fired = if won[h] { 1.0 } else { 0.0 };
                        self.theta[h] =
                            (self.theta[h] + 0.001 * (fired - target_rate)).clamp(-0.35, 0.35);
                    }
                }
            }

            for epoch in 0..6 {
                let lr = 0.25 / (1.0 + 0.25 * epoch as f32);
                for (feats, x, gold) in &prep {
                    let winners = self.hidden_act(x);
                    let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                    let logits = self.store_logits(feats, x);
                    let mx = logits[0].max(logits[1]);
                    let e0 = (logits[0] - mx).exp();
                    let e1 = (logits[1] - mx).exp();
                    let p0 = e0 / (e0 + e1 + 1e-12);
                    let p1 = e1 / (e0 + e1 + 1e-12);
                    let d0 = if *gold == 0 { 1.0 - p0 } else { -p0 };
                    let d1 = if *gold == 1 { 1.0 - p1 } else { -p1 };
                    self.bias[0] += lr * 0.05 * d0;
                    self.bias[1] += lr * 0.05 * d1;
                    for &(h, act) in &winners {
                        if act > 0.0 {
                            let base = h * 2;
                            self.w2[base] += lr * 0.35 * act * d0;
                            self.w2[base + 1] += lr * 0.35 * act * d1;
                        }
                    }
                    for &f in feats {
                        let base = f * 2;
                        self.w_dir[base] += lr * inv_f * d0;
                        self.w_dir[base + 1] += lr * inv_f * d1;
                    }
                }
            }
            self.trained = true;
        }

        let train_conf: Vec<&MetacogConfRow> = conf_rows.iter().filter(|r| r.split != "val").collect();
        let val_conf: Vec<&MetacogConfRow> = conf_rows.iter().filter(|r| r.split == "val").collect();
        if !train_conf.is_empty() {
            self.w_conf = [0.0f32; MC_CONF_DIM];
            for epoch in 0..24 {
                let lr = 0.40 / (1.0 + 0.1 * epoch as f32);
                for r in &train_conf {
                    let s = r.top_score as f32;
                    let feat = [
                        1.0f32,
                        s,
                        r.margin as f32,
                        s * s,
                        r.is_predicate as f32,
                        r.is_taught as f32,
                        r.is_exact as f32,
                    ];
                    let score = self.conf_score(&feat);
                    let prob = 1.0 / (1.0 + (-score.clamp(-12.0, 12.0)).exp());
                    let target = if r.accept > 0 { 1.0f32 } else { 0.0f32 };
                    let delta = target - prob;
                    for i in 0..MC_CONF_DIM {
                        self.w_conf[i] += lr * delta * feat[i];
                    }
                }
            }
        }

        let act_acc = if val_act.is_empty() {
            1.0
        } else {
            let ok = val_act
                .iter()
                .filter(|r| self.should_store_premise(&r.text) == (r.store > 0))
                .count();
            ok as f64 / val_act.len() as f64
        };
        let conf_acc = if val_conf.is_empty() {
            1.0
        } else {
            let ok = val_conf
                .iter()
                .filter(|r| {
                    let s = r.top_score as f32;
                    let feat = [
                        1.0f32,
                        s,
                        r.margin as f32,
                        s * s,
                        r.is_predicate as f32,
                        r.is_taught as f32,
                        r.is_exact as f32,
                    ];
                    (self.conf_score(&feat) > 0.0) == (r.accept > 0)
                })
                .count();
            ok as f64 / val_conf.len() as f64
        };
        (act_acc, conf_acc)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&[u8::from(self.trained)])?;
        w.write_all(bytemuck::cast_slice(&self.w1))?;
        w.write_all(bytemuck::cast_slice(&self.theta))?;
        w.write_all(bytemuck::cast_slice(&self.w2))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir))?;
        w.write_all(bytemuck::cast_slice(&self.bias))?;
        w.write_all(bytemuck::cast_slice(&self.w_conf))?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid MetacogCore magic");
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        let mut w1 = vec![0.0f32; MC_HIDDEN * MC_DIN];
        let mut theta = [0.0f32; MC_HIDDEN];
        let mut w2 = vec![0.0f32; MC_HIDDEN * 2];
        let mut w_dir = vec![0.0f32; MC_BUCKETS * 2];
        let mut bias = [0.0f32; 2];
        let mut w_conf = [0.0f32; MC_CONF_DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w2))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w_dir))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut bias))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w_conf))?;
        Ok(Self {
            trained: flag[0] != 0,
            w1,
            theta,
            w2,
            w_dir,
            bias,
            w_conf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metacog_core_store_conf_and_roundtrip() {
        let acts = vec![
            MetacogActRow { text: "alpha=100m.".into(), store: 1, split: "train".into() },
            MetacogActRow { text: "alpha=100m".into(), store: 1, split: "train".into() },
            MetacogActRow { text: "alpha=?".into(), store: 0, split: "train".into() },
            MetacogActRow { text: "alpha=".into(), store: 0, split: "train".into() },
        ];
        let confs = vec![
            MetacogConfRow { top_score: 0.85, margin: 0.1, is_predicate: 0, is_taught: 0, is_exact: 1, accept: 1, split: "train".into() },
            MetacogConfRow { top_score: 0.25, margin: 0.02, is_predicate: 0, is_taught: 0, is_exact: 1, accept: 0, split: "train".into() },
        ];
        let mut core = MetacogCore::new();
        let _ = core.train(&acts, &confs);
        assert!(core.should_store_premise("alpha=100m"));
        assert!(!core.should_store_premise("alpha="));
        assert!(core.accept_memory_hit(&[Fact { head: "a".into(), body: "b".into(), source: "wiki_exact", score: 0.85 }]));
        assert!(!core.accept_memory_hit(&[Fact { head: "a".into(), body: "b".into(), source: "wiki_exact", score: 0.25 }]));
        let tmp = std::env::temp_dir().join(format!("ng-mc-{}.bin", std::process::id()));
        core.save(&tmp).unwrap();
        let loaded = MetacogCore::load(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert!(loaded.should_store_premise("alpha=100m"));
        assert!(!loaded.should_store_premise("alpha="));
    }
}
