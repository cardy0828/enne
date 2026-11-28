//! ENNE v22.0/v25.0 2層 司令塔コア (`ExecutiveCore`, `NGEC2201` / `NGEC2501`)。
//!
//! 設計仕様:
//! - 逆伝播・事前学習済み言語モデル・固定日本語キーワード表を使わず学習・推論する。
//! - Layer 1: 構造特徴 + UTF-8文字 1/2/3/4-gram ランダム射影 (D_IN=256) →
//!            k-WTA 側方抑制付き競合オヤ・ヘブ隠れ層 (H=128, top-16発火 + 恒常性閾値)
//! - Layer 2: 隠れ層活性 + 直接感覚経路 (F=32768バケット) の局所デルタ則による
//!            5モード分類 (Calc/Clock/Fact/Dialogue/Abstain) & 4演算分類 (Add/Sub/Mul/Div) & Spanタガー (`CalcSpanTagger`)
use crate::core_dag::mix;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use unicode_normalization::UnicodeNormalization;

pub const D_IN: usize = 256;
pub const HIDDEN: usize = 128;
pub const TOP_K: usize = 16;
pub const FEAT_BUCKETS: usize = 32768;
pub const NUM_MODES: usize = 5;
pub const NUM_OPS: usize = 4;

pub const CT_DIN: usize = 64;
pub const CT_HIDDEN: usize = 32;
pub const CT_TOP_K: usize = 6;
pub const CT_BUCKETS: usize = 16384;
pub const CALC_TAGS: usize = 5;
pub const CT_O: usize = 0;
pub const CT_BOP: usize = 1;
pub const CT_IOP: usize = 2;
pub const CT_BUNIT: usize = 3;
pub const CT_IUNIT: usize = 4;

const MAGIC: &[u8; 8] = b"NGEC2501";

pub const MODE_NAMES: [&str; NUM_MODES] = ["Calc", "Clock", "Fact", "Dialogue", "Abstain"];
pub const OP_NAMES: [&str; NUM_OPS] = ["Add", "Sub", "Mul", "Div"];

#[derive(Deserialize)]
pub struct ExecEpisode {
    pub text: String,
    pub mode: usize,
    #[serde(default = "neg_one")]
    pub calc_op: i32,
    #[serde(default)]
    pub split: String,
}
fn neg_one() -> i32 {
    -1
}

#[derive(Deserialize)]
pub struct ProbeCase {
    pub text: String,
    pub mode: usize,
    pub mode_name: String,
    pub calc_op: i32,
}

#[derive(Deserialize, Clone)]
pub struct CalcBioRow {
    pub text: String,
    pub tags: Vec<usize>,
    #[serde(default)]
    pub split: String,
}

#[inline(always)]
fn calc_script_class(c: char) -> u64 {
    if c.is_whitespace() {
        1
    } else if c.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&c) {
        2
    } else if c == '.' {
        3
    } else if matches!(c, '+' | '-' | '*' | '/' | '×' | '÷' | '=' | '＝' | '＋' | '－') {
        4
    } else if c.is_ascii_alphabetic() {
        5
    } else if ('\u{3040}'..='\u{309F}').contains(&c) {
        6
    } else if ('\u{30A0}'..='\u{30FF}').contains(&c) {
        7
    } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
        8
    } else {
        9
    }
}

fn extract_calc_char_features(chars: &[char], pos: usize) -> (Vec<usize>, [f32; CT_DIN]) {
    let n = chars.len();
    let mask = CT_BUCKETS - 1;
    let mut feats = Vec::with_capacity(28);
    let ch = |offset: isize| -> u64 {
        let p = pos as isize + offset;
        if p < 0 || p >= n as isize {
            0xFFFF_FFu64
        } else {
            chars[p as usize] as u64
        }
    };
    let sc = |offset: isize| -> u64 {
        let p = pos as isize + offset;
        if p < 0 || p >= n as isize {
            0
        } else {
            calc_script_class(chars[p as usize])
        }
    };
    let push_f = |feats: &mut Vec<usize>, tag: u64, val: u64| {
        feats.push((mix(tag ^ val.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize) & mask);
    };

    for (idx, off) in (-2isize..=2).enumerate() {
        push_f(&mut feats, 0xA100 + idx as u64, ch(off));
        push_f(&mut feats, 0xA200 + idx as u64, sc(off));
    }
    push_f(&mut feats, 0xB001, (ch(-1) << 21) ^ ch(0));
    push_f(&mut feats, 0xB002, (ch(0) << 21) ^ ch(1));
    push_f(&mut feats, 0xB003, (ch(-2) << 21) ^ ch(-1));
    push_f(&mut feats, 0xB004, (ch(1) << 21) ^ ch(2));
    push_f(&mut feats, 0xB005, mix((ch(-1) << 21) ^ ch(0)) ^ ch(1));

    push_f(&mut feats, 0xC001, (sc(-1) << 8) ^ (sc(0) << 4) ^ sc(1));
    push_f(&mut feats, 0xC002, (sc(-2) << 8) ^ (sc(-1) << 4) ^ sc(0));
    push_f(&mut feats, 0xC003, (sc(0) << 8) ^ (sc(1) << 4) ^ sc(2));
    push_f(&mut feats, 0xC004, (ch(-1) << 8) ^ sc(0));
    push_f(&mut feats, 0xC005, (sc(0) << 8) ^ ch(1));
    push_f(&mut feats, 0xC006, (ch(-2) << 16) ^ (ch(-1) << 4) ^ sc(0));

    let is_one = (chars[pos] == '1') as u64;
    if is_one == 1 {
        push_f(&mut feats, 0xD001, (1u64 << 32) ^ ch(1));
        push_f(&mut feats, 0xD002, (1u64 << 32) ^ ch(-1));
        push_f(&mut feats, 0xD003, (1u64 << 16) ^ (sc(-1) << 4) ^ sc(1));
    } else {
        push_f(&mut feats, 0xD001, 0);
    }

    push_f(&mut feats, 0xE001, (pos == 0) as u64);
    push_f(&mut feats, 0xE002, (pos + 1 == n) as u64);
    push_f(&mut feats, 0xE003, (pos + 2 == n) as u64);
    push_f(&mut feats, 0xE004, (pos + 3 == n) as u64);
    push_f(&mut feats, 0xE005, ((pos * 4) / n.max(1)) as u64);

    feats.sort_unstable();
    feats.dedup();

    let mut x = [0.0f32; CT_DIN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x517c_c1b7_2722_0a95);
        for _ in 0..4 {
            let d = (h as usize) & (CT_DIN - 1);
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

/// 構造特徴 + UTF-8 文字 1..=4-gram ハッシュを抽出する（語彙表に依存しない設計）。
pub fn extract_features(raw: &str) -> (Vec<usize>, [f32; D_IN]) {
    let q: String = raw.trim().nfkc().collect();
    let chars: Vec<char> = q.chars().collect();
    let mask = FEAT_BUCKETS - 1;
    let mut feats = Vec::with_capacity(chars.len() * 4 + 16);

    // 1. Structural and character-class features (language-neutral counters)
    let mut n_digits = 0usize;
    let mut in_num = false;
    let mut num_count = 0usize;
    let mut n_hiragana = 0usize;
    let mut n_katakana = 0usize;
    let mut n_kanji = 0usize;
    let mut n_math_sym = 0usize;
    let mut n_periods = 0usize;
    for &c in &chars {
        if c.is_ascii_digit() {
            n_digits += 1;
            if !in_num {
                num_count += 1;
                in_num = true;
            }
        } else if c != '.' {
            in_num = false;
        }
        if ('\u{3040}'..='\u{309F}').contains(&c) {
            n_hiragana += 1;
        } else if ('\u{30A0}'..='\u{30FF}').contains(&c) {
            n_katakana += 1;
        } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
            n_kanji += 1;
        }
        if matches!(c, '+' | '-' | '*' | '/' | '×' | '÷' | '=' | '＝') {
            n_math_sym += 1;
        }
        if c == '。' {
            n_periods += 1;
        }
    }
    let has_q = q.ends_with(['?', '？']);
    let len_b = match chars.len() {
        0..=7 => 0u64,
        8..=18 => 1,
        19..=40 => 2,
        _ => 3,
    };
    let struct_tags = [
        (0xF001u64, num_count.min(4) as u64),
        (0xF002, (n_math_sym > 0) as u64),
        (0xF003, has_q as u64),
        (0xF004, n_periods.min(3) as u64),
        (0xF005, len_b),
        (0xF006, (n_digits > 0 && n_math_sym > 0) as u64),
        (0xF007, (num_count >= 2) as u64),
        (0xF008, (n_kanji * 2 > chars.len().max(1)) as u64),
        (0xF009, (n_hiragana * 3 > chars.len().max(1) * 2) as u64),
        (0xF00A, (n_katakana >= 2) as u64),
    ];
    for (tag, val) in struct_tags {
        feats.push((mix(tag ^ (val.wrapping_mul(0x9e3779b9))) as usize) & mask);
    }

    // 2. UTF-8 char 1/2/3/4-grams (whitespace stripped)
    let clean: Vec<char> = chars.into_iter().filter(|c| !c.is_whitespace()).collect();
    for i in 0..clean.len() {
        let c0 = clean[i] as u64;
        feats.push((mix(c0 ^ 0x101) as usize) & mask);
        if i + 1 < clean.len() {
            let c1 = clean[i + 1] as u64;
            feats.push((mix((c0 << 21) ^ c1 ^ 0x202) as usize) & mask);
            if i + 2 < clean.len() {
                let c2 = clean[i + 2] as u64;
                feats.push((mix((c0 << 42) ^ (c1 << 21) ^ c2 ^ 0x303) as usize) & mask);
                if i + 3 < clean.len() {
                    let c3 = clean[i + 3] as u64;
                    feats.push((mix(mix((c0 << 21) ^ c1) ^ ((c2 << 21) ^ c3) ^ 0x404) as usize) & mask);
                }
            }
        }
    }
    feats.sort_unstable();
    feats.dedup();

    // 3. Deterministic sparse Johnson-Lindenstrauss projection into D_IN=256 (L2 normalized)
    let mut x = [0.0f32; D_IN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x517cc1b727220a95);
        for _ in 0..4 {
            let d = (h as usize) & (D_IN - 1);
            let sign = if ((h >> 8) & 1) == 0 { 1.0f32 } else { -1.0f32 };
            x[d] += sign;
            h = mix(h ^ 0x9e3779b97f4a7c15);
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

pub struct ExecutiveCore {
    // Layer 1: Competitive Oja-Hebbian hidden prototypes (HIDDEN * D_IN) + homeostatic thresholds
    pub w1: Vec<f32>,
    pub theta: [f32; HIDDEN],
    // Layer 2: Hidden-to-Mode (HIDDEN * NUM_MODES) + Direct-to-Mode (FEAT_BUCKETS * NUM_MODES)
    pub w2_mode: Vec<f32>,
    pub w_dir_mode: Vec<f32>,
    pub b_mode: [f32; NUM_MODES],
    // Layer 2 (CalcOp head): Hidden-to-Op (HIDDEN * NUM_OPS) + Direct-to-Op (FEAT_BUCKETS * NUM_OPS)
    pub w2_op: Vec<f32>,
    pub w_dir_op: Vec<f32>,
    pub b_op: [f32; NUM_OPS],
    // CalcSpanTagger (5-tag character BIO + Viterbi for operand & unit extraction)
    pub ct_trained: bool,
    pub ct_w1: Vec<f32>,
    pub ct_theta: [f32; CT_HIDDEN],
    pub ct_w2: Vec<f32>,
    pub ct_w_dir: Vec<f32>,
    pub ct_bias: [f32; CALC_TAGS],
    pub ct_init_log: [f32; CALC_TAGS],
    pub ct_trans_log: [f32; CALC_TAGS * CALC_TAGS],
}

impl Default for ExecutiveCore {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutiveCore {
    pub fn new() -> Self {
        // Initialize Layer 1 unit prototypes deterministically
        let mut w1 = vec![0.0f32; HIDDEN * D_IN];
        let mut st = 0x243f6a8885a308d3u64;
        for h in 0..HIDDEN {
            let mut n2 = 0.0f32;
            for d in 0..D_IN {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let u = ((st >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
                w1[h * D_IN + d] = u;
                n2 += u * u;
            }
            let inv = 1.0 / (n2.sqrt() + 1e-8);
            for d in 0..D_IN {
                w1[h * D_IN + d] *= inv;
            }
        }
        let mut ct_w1 = vec![0.0f32; CT_HIDDEN * CT_DIN];
        for h in 0..CT_HIDDEN {
            let mut n2 = 0.0f32;
            for d in 0..CT_DIN {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let u = ((st >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
                ct_w1[h * CT_DIN + d] = u;
                n2 += u * u;
            }
            let inv = 1.0 / (n2.sqrt() + 1e-8);
            for d in 0..CT_DIN {
                ct_w1[h * CT_DIN + d] *= inv;
            }
        }
        Self {
            w1,
            theta: [0.0f32; HIDDEN],
            w2_mode: vec![0.0f32; HIDDEN * NUM_MODES],
            w_dir_mode: vec![0.0f32; FEAT_BUCKETS * NUM_MODES],
            b_mode: [0.0f32; NUM_MODES],
            w2_op: vec![0.0f32; HIDDEN * NUM_OPS],
            w_dir_op: vec![0.0f32; FEAT_BUCKETS * NUM_OPS],
            b_op: [0.0f32; NUM_OPS],
            ct_trained: false,
            ct_w1,
            ct_theta: [0.0f32; CT_HIDDEN],
            ct_w2: vec![0.0f32; CT_HIDDEN * CALC_TAGS],
            ct_w_dir: vec![0.0f32; CT_BUCKETS * CALC_TAGS],
            ct_bias: [0.0f32; CALC_TAGS],
            ct_init_log: [0.0f32; CALC_TAGS],
            ct_trans_log: [0.0f32; CALC_TAGS * CALC_TAGS],
        }
    }

    /// Layer 1: k-WTA (top-16) 側方抑制付き競合隠れ表現を計算する。
    fn hidden_act(&self, x: &[f32; D_IN]) -> [(usize, f32); TOP_K] {
        let mut scores = [(0usize, 0.0f32); HIDDEN];
        for h in 0..HIDDEN {
            let row = &self.w1[h * D_IN..(h + 1) * D_IN];
            let mut s = 0.0f32;
            for d in 0..D_IN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..TOP_K {
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

    /// 5モード (Calc/Clock/Fact/Dialogue/Abstain) のロジットと最有力モードを予測する。
    pub fn predict_mode(&self, q: &str) -> (usize, [f32; NUM_MODES]) {
        let (feats, x) = extract_features(q);
        let winners = self.hidden_act(&x);
        let mut logits = self.b_mode;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * NUM_MODES;
                for c in 0..NUM_MODES {
                    logits[c] += act * self.w2_mode[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * NUM_MODES;
            for c in 0..NUM_MODES {
                logits[c] += inv_f * self.w_dir_mode[base + c];
            }
        }
        let mut best = 0;
        for c in 1..NUM_MODES {
            if logits[c] > logits[best] {
                best = c;
            }
        }
        (best, logits)
    }

    /// 計算文章題/式の4演算 (Add/Sub/Mul/Div) を純ヘブ NN で予測する。
    pub fn predict_calc_op(&self, q: &str) -> (usize, [f32; NUM_OPS]) {
        let (feats, x) = extract_features(q);
        let winners = self.hidden_act(&x);
        let mut logits = self.b_op;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * NUM_OPS;
                for c in 0..NUM_OPS {
                    logits[c] += act * self.w2_op[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * NUM_OPS;
            for c in 0..NUM_OPS {
                logits[c] += inv_f * self.w_dir_op[base + c];
            }
        }
        let mut best = 0;
        for c in 1..NUM_OPS {
            if logits[c] > logits[best] {
                best = c;
            }
        }
        (best, logits)
    }

    /// 純ヘブ学習（Phase 1: Oja-Hebbian + k-WTA 恒常性隠れ層、Phase 2: 局所ヘブデルタ読み出し層）。
    pub fn train(&mut self, episodes: &[ExecEpisode], epochs: usize) -> (f64, f64, usize, usize) {
        let train: Vec<&ExecEpisode> = episodes.iter().filter(|e| e.split != "val").collect();
        let val: Vec<&ExecEpisode> = episodes.iter().filter(|e| e.split == "val").collect();

        // Pre-extract features once for fast CPU training
        let train_feats: Vec<(Vec<usize>, [f32; D_IN], usize, i32)> = train
            .iter()
            .map(|e| {
                let (f, x) = extract_features(&e.text);
                (f, x, e.mode, e.calc_op)
            })
            .collect();

        // Phase 1: Unsupervised competitive Oja-Hebbian learning on Layer 1
        let target_rate = TOP_K as f32 / HIDDEN as f32;
        for pass in 0..3 {
            let eta = 0.06 / (1.0 + 0.4 * pass as f32);
            for (_, x, _, _) in &train_feats {
                let winners = self.hidden_act(x);
                let mut won = [false; HIDDEN];
                for &(h, act) in &winners {
                    if act <= 0.0 {
                        continue;
                    }
                    won[h] = true;
                    let row = &mut self.w1[h * D_IN..(h + 1) * D_IN];
                    let mut n2 = 0.0f32;
                    for d in 0..D_IN {
                        let nv = row[d] + eta * act * (x[d] - act * row[d]);
                        row[d] = nv;
                        n2 += nv * nv;
                    }
                    let inv = 1.0 / (n2.sqrt() + 1e-8);
                    for d in 0..D_IN {
                        row[d] *= inv;
                    }
                }
                // Homeostatic threshold adaptation so all 128 prototypes specialize evenly
                for h in 0..HIDDEN {
                    let fired = if won[h] { 1.0f32 } else { 0.0f32 };
                    self.theta[h] = (self.theta[h] + 0.002 * (fired - target_rate)).clamp(-0.35, 0.35);
                }
            }
        }

        // Phase 2: Class-balanced local Hebbian delta rule for Layer 2 (Mode + CalcOp)
        let mut mode_cnt = [1usize; NUM_MODES];
        for (_, _, m, _) in &train_feats {
            mode_cnt[*m] += 1;
        }
        let max_m = *mode_cnt.iter().max().unwrap_or(&1) as f32;
        let mut mode_wt = [1.0f32; NUM_MODES];
        for c in 0..NUM_MODES {
            mode_wt[c] = (max_m / mode_cnt[c] as f32).sqrt().min(2.5);
        }

        for epoch in 0..epochs.max(1) {
            let lr = 0.25 / (1.0 + 0.25 * epoch as f32);
            for (feats, x, mode, calc_op) in &train_feats {
                let winners = self.hidden_act(x);
                let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                let cw = mode_wt[*mode];

                // Mode head forward + local delta update
                let mut logits = self.b_mode;
                for &(h, act) in &winners {
                    if act > 0.0 {
                        let base = h * NUM_MODES;
                        for c in 0..NUM_MODES {
                            logits[c] += act * self.w2_mode[base + c];
                        }
                    }
                }
                for &f in feats {
                    let base = f * NUM_MODES;
                    for c in 0..NUM_MODES {
                        logits[c] += inv_f * self.w_dir_mode[base + c];
                    }
                }
                let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = [0.0f32; NUM_MODES];
                let mut sum_e = 0.0f32;
                for c in 0..NUM_MODES {
                    probs[c] = (logits[c] - max_l).exp();
                    sum_e += probs[c];
                }
                for c in 0..NUM_MODES {
                    probs[c] /= sum_e + 1e-12;
                    let delta = if c == *mode { 1.0 - probs[c] } else { -probs[c] };
                    self.b_mode[c] += lr * cw * 0.05 * delta;
                }
                for &(h, act) in &winners {
                    if act > 0.0 {
                        let base = h * NUM_MODES;
                        for c in 0..NUM_MODES {
                            let delta = if c == *mode { 1.0 - probs[c] } else { -probs[c] };
                            self.w2_mode[base + c] += lr * cw * 0.35 * act * delta;
                        }
                    }
                }
                for &f in feats {
                    let base = f * NUM_MODES;
                    for c in 0..NUM_MODES {
                        let delta = if c == *mode { 1.0 - probs[c] } else { -probs[c] };
                        self.w_dir_mode[base + c] += lr * cw * inv_f * delta;
                    }
                }

                // CalcOp head forward + local delta update (for Calc episodes with op 0..4)
                if *calc_op >= 0 && (*calc_op as usize) < NUM_OPS {
                    let op = *calc_op as usize;
                    let mut ologits = self.b_op;
                    for &(h, act) in &winners {
                        if act > 0.0 {
                            let base = h * NUM_OPS;
                            for c in 0..NUM_OPS {
                                ologits[c] += act * self.w2_op[base + c];
                            }
                        }
                    }
                    for &f in feats {
                        let base = f * NUM_OPS;
                        for c in 0..NUM_OPS {
                            ologits[c] += inv_f * self.w_dir_op[base + c];
                        }
                    }
                    let max_o = ologits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut oprobs = [0.0f32; NUM_OPS];
                    let mut osum = 0.0f32;
                    for c in 0..NUM_OPS {
                        oprobs[c] = (ologits[c] - max_o).exp();
                        osum += oprobs[c];
                    }
                    for c in 0..NUM_OPS {
                        oprobs[c] /= osum + 1e-12;
                        let delta = if c == op { 1.0 - oprobs[c] } else { -oprobs[c] };
                        self.b_op[c] += lr * 0.05 * delta;
                    }
                    for &(h, act) in &winners {
                        if act > 0.0 {
                            let base = h * NUM_OPS;
                            for c in 0..NUM_OPS {
                                let delta = if c == op { 1.0 - oprobs[c] } else { -oprobs[c] };
                                self.w2_op[base + c] += lr * 0.35 * act * delta;
                            }
                        }
                    }
                    for &f in feats {
                        let base = f * NUM_OPS;
                        for c in 0..NUM_OPS {
                            let delta = if c == op { 1.0 - oprobs[c] } else { -oprobs[c] };
                            self.w_dir_op[base + c] += lr * inv_f * delta;
                        }
                    }
                }
            }
        }

        // Validation accuracy on held-out split
        let mut mode_ok = 0usize;
        let mut op_ok = 0usize;
        let mut op_tot = 0usize;
        for ep in &val {
            let (pm, _) = self.predict_mode(&ep.text);
            if pm == ep.mode {
                mode_ok += 1;
            }
            if ep.calc_op >= 0 {
                op_tot += 1;
                let (po, _) = self.predict_calc_op(&ep.text);
                if po == ep.calc_op as usize {
                    op_ok += 1;
                }
            }
        }
        (
            mode_ok as f64 / val.len().max(1) as f64,
            op_ok as f64 / op_tot.max(1) as f64,
            train.len(),
            val.len(),
        )
    }

    #[inline]
    fn ct_hidden_act(&self, x: &[f32; CT_DIN]) -> [(usize, f32); CT_TOP_K] {
        let mut scores = [(0usize, 0.0f32); CT_HIDDEN];
        for h in 0..CT_HIDDEN {
            let row = &self.ct_w1[h * CT_DIN..(h + 1) * CT_DIN];
            let mut s = 0.0f32;
            for d in 0..CT_DIN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.ct_theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); CT_TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..CT_TOP_K {
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

    #[inline]
    fn ct_emit_logits(&self, feats: &[usize], x: &[f32; CT_DIN]) -> [f32; CALC_TAGS] {
        let winners = self.ct_hidden_act(x);
        let mut logits = self.ct_bias;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * CALC_TAGS;
                for c in 0..CALC_TAGS {
                    logits[c] += act * self.ct_w2[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in feats {
            let base = f * CALC_TAGS;
            for c in 0..CALC_TAGS {
                logits[c] += inv_f * self.ct_w_dir[base + c];
            }
        }
        logits
    }

    /// Calc BIO タガーを Oja-Hebbian + 局所デルタ則 + 対数遷移カウントで学習し、(val_char_acc, val_span_f1) を返す。
    pub fn train_calc_tagger(&mut self, rows: &[CalcBioRow], epochs: usize) -> (f64, f64) {
        let train: Vec<&CalcBioRow> = rows.iter().filter(|r| r.split != "val").collect();
        let val: Vec<&CalcBioRow> = rows.iter().filter(|r| r.split == "val").collect();
        if train.is_empty() {
            return (0.0, 0.0);
        }

        let mut prepared = Vec::with_capacity(train.len());
        let mut init_cnt = [1.0f64; CALC_TAGS];
        let mut trans_cnt = [[1.0f64; CALC_TAGS]; CALC_TAGS];
        let mut tag_cnt = [1usize; CALC_TAGS];

        for row in &train {
            let chars: Vec<char> = row.text.chars().collect();
            if chars.is_empty() || chars.len() != row.tags.len() {
                continue;
            }
            init_cnt[row.tags[0].min(CALC_TAGS - 1)] += 1.0;
            for w in row.tags.windows(2) {
                trans_cnt[w[0].min(CALC_TAGS - 1)][w[1].min(CALC_TAGS - 1)] += 1.0;
            }
            let mut cf = Vec::with_capacity(chars.len());
            for (i, &tg) in row.tags.iter().enumerate() {
                let t = tg.min(CALC_TAGS - 1);
                tag_cnt[t] += 1;
                let (feats, x) = extract_calc_char_features(&chars, i);
                cf.push((feats, x, t));
            }
            prepared.push(cf);
        }

        let init_sum: f64 = init_cnt.iter().sum();
        for c in 0..CALC_TAGS {
            self.ct_init_log[c] = (init_cnt[c] / init_sum).ln() as f32 * 0.35;
        }
        for prev in 0..CALC_TAGS {
            let rsum: f64 = trans_cnt[prev].iter().sum();
            for cur in 0..CALC_TAGS {
                self.ct_trans_log[prev * CALC_TAGS + cur] =
                    (trans_cnt[prev][cur] / rsum).ln() as f32 * 0.35;
            }
        }

        let target_rate = CT_TOP_K as f32 / CT_HIDDEN as f32;
        for pass in 0..2 {
            let eta = 0.05 / (1.0 + 0.5 * pass as f32);
            for seq in &prepared {
                for (_, x, _) in seq {
                    let winners = self.ct_hidden_act(x);
                    let mut won = [false; CT_HIDDEN];
                    for &(h, act) in &winners {
                        if act <= 0.0 {
                            continue;
                        }
                        won[h] = true;
                        let row = &mut self.ct_w1[h * CT_DIN..(h + 1) * CT_DIN];
                        let mut n2 = 0.0f32;
                        for d in 0..CT_DIN {
                            let nv = row[d] + eta * act * (x[d] - act * row[d]);
                            row[d] = nv;
                            n2 += nv * nv;
                        }
                        let inv = 1.0 / (n2.sqrt() + 1e-8);
                        for d in 0..CT_DIN {
                            row[d] *= inv;
                        }
                    }
                    for h in 0..CT_HIDDEN {
                        let fired = if won[h] { 1.0f32 } else { 0.0f32 };
                        self.ct_theta[h] =
                            (self.ct_theta[h] + 0.001 * (fired - target_rate)).clamp(-0.35, 0.35);
                    }
                }
            }
        }

        let max_t = *tag_cnt.iter().max().unwrap_or(&1) as f32;
        let mut tag_wt = [1.0f32; CALC_TAGS];
        for c in 0..CALC_TAGS {
            tag_wt[c] = (max_t / tag_cnt[c] as f32).sqrt().min(2.5);
        }

        for epoch in 0..epochs.max(1) {
            let lr = 0.25 / (1.0 + 0.3 * epoch as f32);
            for seq in &prepared {
                for (feats, x, gold) in seq {
                    let winners = self.ct_hidden_act(x);
                    let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                    let cw = tag_wt[*gold];
                    let logits = self.ct_emit_logits(feats, x);
                    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs = [0.0f32; CALC_TAGS];
                    let mut sum_e = 0.0f32;
                    for c in 0..CALC_TAGS {
                        probs[c] = (logits[c] - max_l).exp();
                        sum_e += probs[c];
                    }
                    for c in 0..CALC_TAGS {
                        probs[c] /= sum_e + 1e-12;
                        let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                        self.ct_bias[c] += lr * cw * 0.04 * delta;
                    }
                    for &(h, act) in &winners {
                        if act > 0.0 {
                            let base = h * CALC_TAGS;
                            for c in 0..CALC_TAGS {
                                let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                                self.ct_w2[base + c] += lr * cw * 0.35 * act * delta;
                            }
                        }
                    }
                    for &f in feats {
                        let base = f * CALC_TAGS;
                        for c in 0..CALC_TAGS {
                            let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                            self.ct_w_dir[base + c] += lr * cw * inv_f * delta;
                        }
                    }
                }
            }
        }
        self.ct_trained = true;

        if val.is_empty() {
            return (1.0, 1.0);
        }
        let mut char_ok = 0usize;
        let mut char_tot = 0usize;
        let mut span_tp = 0usize;
        let mut span_fp = 0usize;
        let mut span_fn = 0usize;
        for row in &val {
            let pred = self.tag_calc_chars(&row.text);
            for (&p, &g) in pred.iter().zip(row.tags.iter()) {
                char_tot += 1;
                if p == g {
                    char_ok += 1;
                }
            }
            let gold_spans = extract_spans_from_tags(&row.tags);
            let pred_spans = extract_spans_from_tags(&pred);
            for s in &pred_spans {
                if gold_spans.contains(s) {
                    span_tp += 1;
                } else {
                    span_fp += 1;
                }
            }
            for s in &gold_spans {
                if !pred_spans.contains(s) {
                    span_fn += 1;
                }
            }
        }
        let span_f1 = if span_tp + span_fp + span_fn == 0 {
            1.0
        } else {
            (2.0 * span_tp as f64) / (2 * span_tp + span_fp + span_fn) as f64
        };
        (char_ok as f64 / char_tot.max(1) as f64, span_f1)
    }

    /// 1次対数ヘブ遷移付き Viterbi DP で数値オペランド・単位 BIO タグ列を復号する。
    pub fn tag_calc_chars(&self, raw: &str) -> Vec<usize> {
        let q: String = raw.trim().nfkc().collect();
        let chars: Vec<char> = q.chars().collect();
        let n = chars.len();
        if n == 0 || !self.ct_trained {
            return vec![CT_O; n];
        }
        let mut dp = vec![[f32::NEG_INFINITY; CALC_TAGS]; n];
        let mut back = vec![[0u8; CALC_TAGS]; n];
        let (f0, x0) = extract_calc_char_features(&chars, 0);
        let l0 = self.ct_emit_logits(&f0, &x0);
        for c in 0..CALC_TAGS {
            dp[0][c] = l0[c] + self.ct_init_log[c];
        }
        for t in 1..n {
            let (ft, xt) = extract_calc_char_features(&chars, t);
            let lt = self.ct_emit_logits(&ft, &xt);
            for cur in 0..CALC_TAGS {
                let mut best_prev = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for prev in 0..CALC_TAGS {
                    let s = dp[t - 1][prev] + self.ct_trans_log[prev * CALC_TAGS + cur];
                    if s > best_score {
                        best_score = s;
                        best_prev = prev;
                    }
                }
                dp[t][cur] = best_score + lt[cur];
                back[t][cur] = best_prev as u8;
            }
        }
        let mut tags = vec![CT_O; n];
        let mut best_last = 0usize;
        for c in 1..CALC_TAGS {
            if dp[n - 1][c] > dp[n - 1][best_last] {
                best_last = c;
            }
        }
        tags[n - 1] = best_last;
        for t in (1..n).rev() {
            tags[t - 1] = back[t][tags[t]] as usize;
        }
        tags
    }

    /// Calc BIO タガーを用いて入力から数値オペランド列と単位文字列を神経抽出する。
    pub fn extract_calc_operands_and_unit(&self, raw: &str) -> (Vec<String>, String) {
        let q: String = raw.trim().nfkc().collect();
        let chars: Vec<char> = q.chars().collect();
        let tags = self.tag_calc_chars(&q);
        let mut ops = Vec::new();
        let mut cur_op = String::new();
        let mut unit = String::new();
        let mut cur_unit = String::new();
        for (i, &tg) in tags.iter().enumerate() {
            let ch = chars[i];
            match tg {
                CT_BOP => {
                    if !cur_op.is_empty() {
                        ops.push(std::mem::take(&mut cur_op));
                    }
                    cur_op.push(ch);
                }
                CT_IOP => {
                    if cur_op.is_empty() {
                        cur_op.push(ch);
                    } else {
                        cur_op.push(ch);
                    }
                }
                CT_BUNIT => {
                    if !cur_op.is_empty() {
                        ops.push(std::mem::take(&mut cur_op));
                    }
                    if !cur_unit.is_empty() {
                        unit = std::mem::take(&mut cur_unit);
                    }
                    cur_unit.push(ch);
                }
                CT_IUNIT => {
                    if !cur_op.is_empty() {
                        ops.push(std::mem::take(&mut cur_op));
                    }
                    cur_unit.push(ch);
                }
                _ => {
                    if !cur_op.is_empty() {
                        ops.push(std::mem::take(&mut cur_op));
                    }
                    if !cur_unit.is_empty() {
                        unit = std::mem::take(&mut cur_unit);
                    }
                }
            }
        }
        if !cur_op.is_empty() {
            ops.push(cur_op);
        }
        if !cur_unit.is_empty() {
            unit = cur_unit;
        }
        (ops, unit)
    }

    /// バイナリチェックポイント `NGEC2501` を保存する。
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(bytemuck::cast_slice(&self.w1))?;
        w.write_all(bytemuck::cast_slice(&self.theta))?;
        w.write_all(bytemuck::cast_slice(&self.w2_mode))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir_mode))?;
        w.write_all(bytemuck::cast_slice(&self.b_mode))?;
        w.write_all(bytemuck::cast_slice(&self.w2_op))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir_op))?;
        w.write_all(bytemuck::cast_slice(&self.b_op))?;
        w.write_all(&[u8::from(self.ct_trained)])?;
        w.write_all(bytemuck::cast_slice(&self.ct_w1))?;
        w.write_all(bytemuck::cast_slice(&self.ct_theta))?;
        w.write_all(bytemuck::cast_slice(&self.ct_w2))?;
        w.write_all(bytemuck::cast_slice(&self.ct_w_dir))?;
        w.write_all(bytemuck::cast_slice(&self.ct_bias))?;
        w.write_all(bytemuck::cast_slice(&self.ct_init_log))?;
        w.write_all(bytemuck::cast_slice(&self.ct_trans_log))?;
        w.flush()?;
        Ok(())
    }

    /// `NGEC2501` バイナリチェックポイントをロードする。
    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid ExecutiveCore checkpoint magic");
        let mut w1 = vec![0.0f32; HIDDEN * D_IN];
        let mut theta = [0.0f32; HIDDEN];
        let mut w2_mode = vec![0.0f32; HIDDEN * NUM_MODES];
        let mut w_dir_mode = vec![0.0f32; FEAT_BUCKETS * NUM_MODES];
        let mut b_mode = [0.0f32; NUM_MODES];
        let mut w2_op = vec![0.0f32; HIDDEN * NUM_OPS];
        let mut w_dir_op = vec![0.0f32; FEAT_BUCKETS * NUM_OPS];
        let mut b_op = [0.0f32; NUM_OPS];
        r.read_exact(bytemuck::cast_slice_mut(&mut w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w2_mode))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w_dir_mode))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut b_mode))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w2_op))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut w_dir_op))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut b_op))?;
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        let mut ct_w1 = vec![0.0f32; CT_HIDDEN * CT_DIN];
        let mut ct_theta = [0.0f32; CT_HIDDEN];
        let mut ct_w2 = vec![0.0f32; CT_HIDDEN * CALC_TAGS];
        let mut ct_w_dir = vec![0.0f32; CT_BUCKETS * CALC_TAGS];
        let mut ct_bias = [0.0f32; CALC_TAGS];
        let mut ct_init_log = [0.0f32; CALC_TAGS];
        let mut ct_trans_log = [0.0f32; CALC_TAGS * CALC_TAGS];
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_w2))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_w_dir))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_bias))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_init_log))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut ct_trans_log))?;
        Ok(Self {
            w1,
            theta,
            w2_mode,
            w_dir_mode,
            b_mode,
            w2_op,
            w_dir_op,
            b_op,
            ct_trained: flag[0] != 0,
            ct_w1,
            ct_theta,
            ct_w2,
            ct_w_dir,
            ct_bias,
            ct_init_log,
            ct_trans_log,
        })
    }
}

fn extract_spans_from_tags(tags: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < tags.len() {
        let t = tags[i];
        if t == CT_BOP || t == CT_IOP {
            let start = i;
            i += 1;
            while i < tags.len() && tags[i] == CT_IOP {
                i += 1;
            }
            spans.push((1, start, i));
        } else if t == CT_BUNIT || t == CT_IUNIT {
            let start = i;
            i += 1;
            while i < tags.len() && tags[i] == CT_IUNIT {
                i += 1;
            }
            spans.push((3, start, i));
        } else {
            i += 1;
        }
    }
    spans
}

#[derive(Serialize)]
struct ProbeResult {
    text: String,
    gold_mode: String,
    pred_mode: String,
    mode_ok: bool,
    gold_op: Option<String>,
    pred_op: Option<String>,
    op_ok: Option<bool>,
}

/// コーパスから ExecutiveCore を学習・保存し、未知言い換えプローブ含む検証レポートを出力する。
pub fn train_and_bench(
    corpus_dir: &Path,
    ckpt_path: &Path,
    out_json: &Path,
    epochs: usize,
) -> Result<Value> {
    let raw = fs::read_to_string(corpus_dir.join("episodes.jsonl"))?;
    let episodes: Vec<ExecEpisode> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let calc_bio_path = corpus_dir.join("calc_bio.jsonl");
    let calc_rows: Vec<CalcBioRow> = if calc_bio_path.exists() {
        fs::read_to_string(&calc_bio_path)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let probes: Vec<ProbeCase> =
        serde_json::from_slice(&fs::read(corpus_dir.join("unseen_probes.json"))?)?;

    let t0 = Instant::now();
    let mut core = ExecutiveCore::new();
    let (val_mode_acc, val_op_acc, n_train, n_val) = core.train(&episodes, epochs);
    let (val_calc_char_acc, val_calc_span_f1) = core.train_calc_tagger(&calc_rows, 4);
    let train_s = t0.elapsed().as_secs_f64();
    core.save(ckpt_path)?;
    let ckpt_mib = fs::metadata(ckpt_path)?.len() as f64 / (1024.0 * 1024.0);

    let tl = Instant::now();
    let loaded = ExecutiveCore::load(ckpt_path)?;
    let load_ms = tl.elapsed().as_secs_f64() * 1000.0;

    let mut probe_rows = Vec::new();
    let mut p_mode_ok = 0usize;
    let mut p_op_ok = 0usize;
    let mut p_op_tot = 0usize;
    let mut p_ext_ok = 0usize;
    let mut times_us = Vec::new();
    for p in &probes {
        let ti = Instant::now();
        let (pm, _) = loaded.predict_mode(&p.text);
        let us = ti.elapsed().as_secs_f64() * 1e6;
        times_us.push(us);
        let mok = pm == p.mode;
        if mok {
            p_mode_ok += 1;
        }
        let (gold_op, pred_op, op_ok) = if p.calc_op >= 0 {
            p_op_tot += 1;
            let (po, _) = loaded.predict_calc_op(&p.text);
            let (ops, _) = loaded.extract_calc_operands_and_unit(&p.text);
            if ops.len() >= 2 && ops[0].parse::<f64>().is_ok() && ops[1].parse::<f64>().is_ok() {
                p_ext_ok += 1;
            }
            let ok = po == p.calc_op as usize;
            if ok {
                p_op_ok += 1;
            }
            (
                Some(OP_NAMES[p.calc_op as usize].to_owned()),
                Some(OP_NAMES[po].to_owned()),
                Some(ok),
            )
        } else {
            (None, None, None)
        };
        probe_rows.push(ProbeResult {
            text: p.text.clone(),
            gold_mode: p.mode_name.clone(),
            pred_mode: MODE_NAMES[pm].to_owned(),
            mode_ok: mok,
            gold_op,
            pred_op,
            op_ok,
        });
    }
    let avg_us = times_us.iter().sum::<f64>() / times_us.len().max(1) as f64;
    let payload = json!({
        "train_episodes": n_train,
        "val_episodes": n_val,
        "val_mode_acc": val_mode_acc,
        "val_op_acc": val_op_acc,
        "val_calc_char_acc": val_calc_char_acc,
        "val_calc_span_f1": val_calc_span_f1,
        "unseen_probe_mode": format!("{p_mode_ok}/{}", probes.len()),
        "unseen_probe_op": format!("{p_op_ok}/{p_op_tot}"),
        "calc_probe_extract_ok": format!("{p_ext_ok}/{p_op_tot}"),
        "train_elapsed_s": train_s,
        "ckpt_mib": ckpt_mib,
        "load_ms": load_ms,
        "avg_infer_us": avg_us,
        "hidden_neurons": HIDDEN,
        "top_k": TOP_K,
        "probes": probe_rows,
    });
    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_json, serde_json::to_vec_pretty(&payload)?)?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn executive_core_oja_hebbian_learns_modes_and_ops() {
        let eps = vec![
            ExecEpisode { text: "今の時刻を教えて".into(), mode: 1, calc_op: -1, split: "train".into() },
            ExecEpisode { text: "3つの袋に4個ずつ入れると何個？".into(), mode: 0, calc_op: 2, split: "train".into() },
            ExecEpisode { text: "50円と30円を合わせると何円？".into(), mode: 0, calc_op: 0, split: "train".into() },
            ExecEpisode { text: "明日の宝くじの番号は？".into(), mode: 4, calc_op: -1, split: "train".into() },
            ExecEpisode { text: "今日は楽しかったね。".into(), mode: 3, calc_op: -1, split: "train".into() },
            ExecEpisode { text: "富士山の標高は何m？".into(), mode: 2, calc_op: -1, split: "train".into() },
        ];
        let mut core = ExecutiveCore::new();
        let _ = core.train(&eps, 8);
        assert_eq!(core.predict_mode("今の時刻を教えて").0, 1);
        assert_eq!(core.predict_mode("3つの袋に4個ずつ入れると何個？").0, 0);
        assert_eq!(core.predict_calc_op("3つの袋に4個ずつ入れると何個？").0, 2);
        assert_eq!(core.predict_calc_op("50円と30円を合わせると何円？").0, 0);
        assert_eq!(core.predict_mode("明日の宝くじの番号は？").0, 4);

        let bio_rows = vec![
            CalcBioRow {
                text: "18 * 7 =".into(),
                tags: vec![1, 2, 0, 0, 0, 1, 0, 0],
                split: "train".into(),
            },
            CalcBioRow {
                text: "18 + 7 =".into(),
                tags: vec![1, 2, 0, 0, 0, 1, 0, 0],
                split: "train".into(),
            },
            CalcBioRow {
                text: "28 / 4 =".into(),
                tags: vec![1, 2, 0, 0, 0, 1, 0, 0],
                split: "train".into(),
            },
            CalcBioRow {
                text: "1f4x5=?".into(),
                tags: vec![0, 0, 1, 0, 1, 0, 0],
                split: "train".into(),
            },
        ];
        let (ca, sf) = core.train_calc_tagger(&bio_rows, 8);
        assert!(ca > 0.9 && sf > 0.9);
        let (ops, unit) = core.extract_calc_operands_and_unit("18 * 7 =");
        assert_eq!(ops, vec!["18".to_owned(), "7".to_owned()]);
        assert_eq!(unit, "");
        let tmp = std::env::temp_dir().join(format!("ng-ec-{}.bin", std::process::id()));
        core.save(&tmp).unwrap();
        let loaded = ExecutiveCore::load(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert_eq!(loaded.extract_calc_operands_and_unit("18 * 7 =").0, ops);
    }
}
