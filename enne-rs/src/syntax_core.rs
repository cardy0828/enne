//! ENNE v24.5 文字単位構文・節境界知覚コア (`SyntaxCore`, `NGSX2451`)。
//!
//! 設計仕様:
//! - 逆伝播や事前学習済み言語モデル・外部形態素解析器を使わず、文字窓特徴のみを入力とする。
//! - 各文字位置の局所窓特徴 (文字n-gram + Unicode文字種遷移 + 相対境界手がかり) を入力とし、
//!   1層 オヤ・ヘブ k-WTA 競合隠れ表現 (`SYN_HIDDEN=64`, `SYN_TOP_K=8`) と
//!   2層 局所デルタ放出重みおよび 1次対数ヘブ遷移行列上の Viterbi 復号で 10種 BIO タグを推定する。
//! - タグ体系:
//!   0=SEP, 1=FUNC, 2=B-ENT, 3=I-ENT, 4=B-ATTR, 5=I-ATTR, 6=B-NUM, 7=I-NUM, 8=B-PRED, 9=I-PRED。

use crate::core_dag::mix;
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};

pub const SYN_DIN: usize = 128;
pub const SYN_HIDDEN: usize = 64;
pub const SYN_TOP_K: usize = 8;
pub const SYN_BUCKETS: usize = 32768;
pub const NUM_TAGS: usize = 10;

pub const TAG_SEP: usize = 0;
pub const TAG_FUNC: usize = 1;
pub const TAG_BENT: usize = 2;
pub const TAG_IENT: usize = 3;
pub const TAG_BATTR: usize = 4;
pub const TAG_IATTR: usize = 5;
pub const TAG_BNUM: usize = 6;
pub const TAG_INUM: usize = 7;
pub const TAG_BPRED: usize = 8;
pub const TAG_IPRED: usize = 9;

const MAGIC: &[u8; 8] = b"NGSX2451";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Func,
    Entity,
    Attr,
    Num,
    Pred,
}

#[derive(Clone, Debug)]
pub struct SyntaxSpan<'a> {
    pub kind: SpanKind,
    pub text: &'a str,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Clone, Debug)]
pub struct ParsedClause<'a> {
    pub raw: &'a str,
    pub byte_start: usize,
    pub byte_end: usize,
    pub spans: Vec<SyntaxSpan<'a>>,
}

#[inline(always)]
fn tag_to_kind(tag: usize) -> Option<SpanKind> {
    match tag {
        TAG_FUNC => Some(SpanKind::Func),
        TAG_BENT | TAG_IENT => Some(SpanKind::Entity),
        TAG_BATTR | TAG_IATTR => Some(SpanKind::Attr),
        TAG_BNUM | TAG_INUM => Some(SpanKind::Num),
        TAG_BPRED | TAG_IPRED => Some(SpanKind::Pred),
        _ => None,
    }
}

#[inline(always)]
fn is_begin_tag(tag: usize) -> bool {
    matches!(tag, TAG_BENT | TAG_BATTR | TAG_BNUM | TAG_BPRED)
}

#[inline(always)]
fn script_class(c: char) -> u64 {
    if c.is_whitespace() {
        1
    } else if c.is_ascii_digit() || ('\u{FF10}'..='\u{FF19}').contains(&c) || c == '.' {
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

/// 注目文字位置 `pos` に対する前後4文字の非固定・言語非依存窓特徴量を抽出する。
pub fn extract_char_window_features(chars: &[char], pos: usize) -> (Vec<usize>, [f32; SYN_DIN]) {
    let n = chars.len();
    let mask = SYN_BUCKETS - 1;
    let mut feats = Vec::with_capacity(48);

    let ch_at = |off: isize| -> u64 {
        let p = pos as isize + off;
        if p < 0 {
            0xFFFF_0001u64
        } else if p >= n as isize {
            0xFFFF_0002u64
        } else {
            chars[p as usize] as u64
        }
    };
    let sc_at = |off: isize| -> u64 {
        let p = pos as isize + off;
        if p < 0 || p >= n as isize {
            0u64
        } else {
            script_class(chars[p as usize])
        }
    };

    // 1. Positional unigrams (-4..=+4) + script classes
    for off in -4isize..=4 {
        let oi = (off + 10) as u64;
        let ch = ch_at(off);
        let sc = sc_at(off);
        feats.push((mix(0xA101u64 ^ (oi << 32) ^ ch) as usize) & mask);
        feats.push((mix(0xA202u64 ^ (oi << 16) ^ sc) as usize) & mask);
    }

    // 2. Positional char bigrams (-3..=+2) & skip bigram (-1, +1)
    for off in -3isize..=2 {
        let oi = (off + 10) as u64;
        let c0 = ch_at(off);
        let c1 = ch_at(off + 1);
        feats.push((mix(0xB101u64 ^ (oi << 44) ^ (c0 << 21) ^ c1) as usize) & mask);
    }
    feats.push((mix(0xB202u64 ^ (ch_at(-1) << 21) ^ ch_at(1)) as usize) & mask);

    // 3. Positional char trigrams (-2..=0)
    for off in -2isize..=0 {
        let oi = (off + 10) as u64;
        let c0 = ch_at(off);
        let c1 = ch_at(off + 1);
        let c2 = ch_at(off + 2);
        feats.push((mix(0xC101u64 ^ (oi << 48) ^ (c0 << 32) ^ (c1 << 16) ^ c2) as usize) & mask);
    }

    // 4. Script bigrams, trigrams & script-transition boundaries
    for off in -2isize..=1 {
        let oi = (off + 10) as u64;
        let s0 = sc_at(off);
        let s1 = sc_at(off + 1);
        feats.push((mix(0xD101u64 ^ (oi << 12) ^ (s0 << 4) ^ s1) as usize) & mask);
    }
    let s_prev = sc_at(-1);
    let s_cur = sc_at(0);
    let s_next = sc_at(1);
    feats.push((mix(0xD202u64 ^ (s_prev << 8) ^ (s_cur << 4) ^ s_next) as usize) & mask);
    feats.push((mix(0xD303u64 ^ ((s_cur != s_prev) as u64)) as usize) & mask);
    feats.push((mix(0xD404u64 ^ ((s_cur != s_next) as u64)) as usize) & mask);

    // 5. Relative structural anchors: boundary / digit / symbol proximity
    let at_start = u64::from(pos == 0);
    let at_end = u64::from(pos + 1 == n);
    let near_end = u64::from(pos + 2 == n);
    let digit_left = (-3isize..=-1).any(|o| sc_at(o) == 2);
    let digit_right = (1isize..=3).any(|o| sc_at(o) == 2);
    let sym_left = (1usize..=3).find(|&d| sc_at(-(d as isize)) == 7).unwrap_or(0) as u64;
    let sym_right = (1usize..=3).find(|&d| sc_at(d as isize) == 7).unwrap_or(0) as u64;
    for (tag, val) in [
        (0xE001u64, at_start),
        (0xE002, at_end),
        (0xE003, near_end),
        (0xE004, u64::from(digit_left)),
        (0xE005, u64::from(digit_right)),
        (0xE006, sym_left),
        (0xE007, sym_right),
    ] {
        feats.push((mix(tag ^ val.wrapping_mul(0x9e3779b9)) as usize) & mask);
    }

    feats.sort_unstable();
    feats.dedup();

    let mut x = [0.0f32; SYN_DIN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x243f6a8885a308d3);
        for _ in 0..4 {
            let d = (h as usize) & (SYN_DIN - 1);
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

#[derive(Deserialize)]
pub struct SyntaxRow {
    pub text: String,
    pub tags: Vec<usize>,
    #[serde(default)]
    pub split: String,
}

pub struct SyntaxCore {
    pub trained: bool,
    pub w1: Vec<f32>,
    pub theta: [f32; SYN_HIDDEN],
    pub w2: Vec<f32>,
    pub w_dir: Vec<f32>,
    pub bias: [f32; NUM_TAGS],
    pub init_trans: [f32; NUM_TAGS],
    pub trans: [f32; NUM_TAGS * NUM_TAGS],
}

impl Default for SyntaxCore {
    fn default() -> Self {
        Self::blank()
    }
}

impl SyntaxCore {
    pub fn blank() -> Self {
        let mut w1 = vec![0.0f32; SYN_HIDDEN * SYN_DIN];
        let mut st = 0x13198a2e03707344u64;
        for h in 0..SYN_HIDDEN {
            let mut n2 = 0.0f32;
            for d in 0..SYN_DIN {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let u = ((st >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
                w1[h * SYN_DIN + d] = u;
                n2 += u * u;
            }
            let inv = 1.0 / (n2.sqrt() + 1e-8);
            for d in 0..SYN_DIN {
                w1[h * SYN_DIN + d] *= inv;
            }
        }
        Self {
            trained: false,
            w1,
            theta: [0.0f32; SYN_HIDDEN],
            w2: vec![0.0f32; SYN_HIDDEN * NUM_TAGS],
            w_dir: vec![0.0f32; SYN_BUCKETS * NUM_TAGS],
            bias: [0.0f32; NUM_TAGS],
            init_trans: [0.0f32; NUM_TAGS],
            trans: [0.0f32; NUM_TAGS * NUM_TAGS],
        }
    }

    #[inline]
    fn hidden_act(&self, x: &[f32; SYN_DIN]) -> [(usize, f32); SYN_TOP_K] {
        let mut scores = [(0usize, 0.0f32); SYN_HIDDEN];
        for h in 0..SYN_HIDDEN {
            let row = &self.w1[h * SYN_DIN..(h + 1) * SYN_DIN];
            let mut s = 0.0f32;
            for d in 0..SYN_DIN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); SYN_TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..SYN_TOP_K {
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
    fn emit_logits(&self, feats: &[usize], x: &[f32; SYN_DIN]) -> [f32; NUM_TAGS] {
        let winners = self.hidden_act(x);
        let mut logits = self.bias;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * NUM_TAGS;
                for c in 0..NUM_TAGS {
                    logits[c] += act * self.w2[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in feats {
            let base = f * NUM_TAGS;
            for c in 0..NUM_TAGS {
                logits[c] += inv_f * self.w_dir[base + c];
            }
        }
        logits
    }

    /// 1次対数ヘブ遷移行列つき Viterbi DP を使って文字列全体の BIO タグ列を復号する。
    pub fn tag_chars(&self, text: &str) -> Vec<usize> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return vec![];
        }
        if !self.trained {
            return chars
                .iter()
                .map(|&c| {
                    if c == '\n' {
                        TAG_SEP
                    } else if c.is_ascii_digit() {
                        TAG_BNUM
                    } else if c.is_ascii_punctuation() {
                        TAG_FUNC
                    } else {
                        TAG_IENT
                    }
                })
                .collect();
        }

        let mut dp = vec![[f32::NEG_INFINITY; NUM_TAGS]; n];
        let mut back = vec![[0u8; NUM_TAGS]; n];

        let (f0, x0) = extract_char_window_features(&chars, 0);
        let l0 = self.emit_logits(&f0, &x0);
        for c in 0..NUM_TAGS {
            dp[0][c] = l0[c] + self.init_trans[c];
        }

        for t in 1..n {
            let (ft, xt) = extract_char_window_features(&chars, t);
            let lt = self.emit_logits(&ft, &xt);
            for cur in 0..NUM_TAGS {
                let mut best_prev = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for prev in 0..NUM_TAGS {
                    let s = dp[t - 1][prev] + self.trans[prev * NUM_TAGS + cur];
                    if s > best_score {
                        best_score = s;
                        best_prev = prev;
                    }
                }
                dp[t][cur] = best_score + lt[cur];
                back[t][cur] = best_prev as u8;
            }
        }

        let mut tags = vec![0usize; n];
        let mut best_last = 0usize;
        for c in 1..NUM_TAGS {
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

    /// 復号された文字 BIO タグ列から、節リストと型付きスパン (`Entity`/`Attr`/`Num`/`Pred`/`Func`) を構築する。
    pub fn parse<'a>(&self, text: &'a str) -> Vec<ParsedClause<'a>> {
        let info: Vec<(usize, usize, char)> = text
            .char_indices()
            .map(|(bs, ch)| (bs, bs + ch.len_utf8(), ch))
            .collect();
        let n = info.len();
        if n == 0 {
            return vec![];
        }
        let tags = self.tag_chars(text);
        let mut clauses = Vec::new();
        let mut i = 0usize;
        while i < n {
            while i < n && (tags[i] == TAG_SEP || info[i].2.is_whitespace()) {
                i += 1;
            }
            if i >= n {
                break;
            }
            let c_start_idx = i;
            while i < n && tags[i] != TAG_SEP {
                i += 1;
            }
            let mut c_end_idx = i;
            while c_end_idx > c_start_idx && info[c_end_idx - 1].2.is_whitespace() {
                c_end_idx -= 1;
            }
            if c_end_idx <= c_start_idx {
                continue;
            }
            let c_bs = info[c_start_idx].0;
            let c_be = info[c_end_idx - 1].1;

            let mut spans = Vec::new();
            let mut k = c_start_idx;
            while k < c_end_idx {
                while k < c_end_idx && info[k].2.is_whitespace() {
                    k += 1;
                }
                if k >= c_end_idx {
                    break;
                }
                let Some(kind) = tag_to_kind(tags[k]) else {
                    k += 1;
                    continue;
                };
                let s_bs = info[k].0;
                let mut s_be = info[k].1;
                k += 1;
                while k < c_end_idx {
                    if info[k].2.is_whitespace() {
                        break;
                    }
                    let next_tag = tags[k];
                    let Some(next_kind) = tag_to_kind(next_tag) else {
                        break;
                    };
                    if next_kind != kind || is_begin_tag(next_tag) {
                        break;
                    }
                    s_be = info[k].1;
                    k += 1;
                }
                spans.push(SyntaxSpan {
                    kind,
                    text: &text[s_bs..s_be],
                    byte_start: s_bs,
                    byte_end: s_be,
                });
            }

            if !spans.is_empty() {
                clauses.push(ParsedClause {
                    raw: &text[c_bs..c_be],
                    byte_start: c_bs,
                    byte_end: c_be,
                    spans,
                });
            }
        }
        clauses
    }

    pub fn train_from_rows(&mut self, rows: &[SyntaxRow], epochs: usize) -> (f64, f64, f64) {
        let mut tr_rows = Vec::new();
        let mut va_rows = Vec::new();
        for r in rows {
            if r.split == "val" {
                va_rows.push(r);
            } else {
                tr_rows.push(r);
            }
        }

        // Collect all training character windows once + co-occurrence transition counts
        let mut char_samples: Vec<(Vec<usize>, [f32; SYN_DIN], usize)> = Vec::new();
        let mut init_cnt = [0.0f64; NUM_TAGS];
        let mut trans_cnt = [[0.0f64; NUM_TAGS]; NUM_TAGS];
        let mut tag_cnt = [1usize; NUM_TAGS];

        for r in &tr_rows {
            let chars: Vec<char> = r.text.chars().collect();
            if chars.len() != r.tags.len() || chars.is_empty() {
                continue;
            }
            init_cnt[r.tags[0]] += 1.0;
            for t in 0..chars.len() {
                let y = r.tags[t];
                tag_cnt[y] += 1;
                if t > 0 {
                    trans_cnt[r.tags[t - 1]][y] += 1.0;
                }
                let (f, x) = extract_char_window_features(&chars, t);
                char_samples.push((f, x, y));
            }
        }

        // Phase 1: Competitive Oja-Hebbian Layer 1 (subsampled for fast CPU pass)
        let target_rate = SYN_TOP_K as f32 / SYN_HIDDEN as f32;
        let stride = (char_samples.len() / 120_000).max(1);
        for pass in 0..3 {
            let eta = 0.06 / (1.0 + 0.4 * pass as f32);
            for (idx, (_, x, _)) in char_samples.iter().enumerate() {
                if idx % stride != pass % stride {
                    continue;
                }
                let winners = self.hidden_act(x);
                let mut won = [false; SYN_HIDDEN];
                for &(h, act) in &winners {
                    if act <= 0.0 {
                        continue;
                    }
                    won[h] = true;
                    let row = &mut self.w1[h * SYN_DIN..(h + 1) * SYN_DIN];
                    let mut n2 = 0.0f32;
                    for d in 0..SYN_DIN {
                        let nv = row[d] + eta * act * (x[d] - act * row[d]);
                        row[d] = nv;
                        n2 += nv * nv;
                    }
                    let inv = 1.0 / (n2.sqrt() + 1e-8);
                    for d in 0..SYN_DIN {
                        row[d] *= inv;
                    }
                }
                for h in 0..SYN_HIDDEN {
                    let fired = if won[h] { 1.0f32 } else { 0.0f32 };
                    self.theta[h] =
                        (self.theta[h] + 0.002 * (fired - target_rate)).clamp(-0.35, 0.35);
                }
            }
        }

        // Phase 2: Class-balanced Local Hebbian Delta for character emissions
        let max_c = *tag_cnt.iter().max().unwrap_or(&1) as f32;
        let mut cwt = [1.0f32; NUM_TAGS];
        for c in 0..NUM_TAGS {
            cwt[c] = (max_c / tag_cnt[c] as f32).sqrt().min(3.0);
        }

        let tr_hidden: Vec<[(usize, f32); SYN_TOP_K]> = char_samples
            .iter()
            .map(|(_, x, _)| self.hidden_act(x))
            .collect();

        for epoch in 0..epochs.max(1) {
            let lr = 0.22 / (1.0 + 0.30 * epoch as f32);
            for ((feats, _, y), winners) in char_samples.iter().zip(&tr_hidden) {
                let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                let wt = cwt[*y];

                let mut logits = self.bias;
                for &(h, act) in winners {
                    if act > 0.0 {
                        let base = h * NUM_TAGS;
                        for c in 0..NUM_TAGS {
                            logits[c] += act * self.w2[base + c];
                        }
                    }
                }
                for &f in feats {
                    let base = f * NUM_TAGS;
                    for c in 0..NUM_TAGS {
                        logits[c] += inv_f * self.w_dir[base + c];
                    }
                }

                let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = [0.0f32; NUM_TAGS];
                let mut sum_e = 0.0f32;
                for c in 0..NUM_TAGS {
                    probs[c] = (logits[c] - max_l).exp();
                    sum_e += probs[c];
                }
                for c in 0..NUM_TAGS {
                    probs[c] /= sum_e + 1e-12;
                    let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                    self.bias[c] += lr * wt * 0.05 * delta;
                }
                for &(h, act) in winners {
                    if act > 0.0 {
                        let base = h * NUM_TAGS;
                        for c in 0..NUM_TAGS {
                            let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                            self.w2[base + c] += lr * wt * 0.35 * act * delta;
                        }
                    }
                }
                for &f in feats {
                    let base = f * NUM_TAGS;
                    for c in 0..NUM_TAGS {
                        let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                        self.w_dir[base + c] += lr * wt * inv_f * delta;
                    }
                }
            }
        }

        // Phase 3: Log-Hebbian co-occurrence transition matrix
        let init_sum: f64 = init_cnt.iter().sum::<f64>().max(1.0) + 1e-2;
        for c in 0..NUM_TAGS {
            self.init_trans[c] = (0.7 * ((init_cnt[c] + 1e-3) / init_sum).ln()) as f32;
        }
        for prev in 0..NUM_TAGS {
            let row_sum: f64 = trans_cnt[prev].iter().sum::<f64>().max(1.0) + 1e-2;
            for cur in 0..NUM_TAGS {
                self.trans[prev * NUM_TAGS + cur] =
                    (0.7 * ((trans_cnt[prev][cur] + 1e-3) / row_sum).ln()) as f32;
            }
        }
        self.trained = true;

        // Evaluate held-out val set: char acc, SEP boundary F1, exact span F1
        let mut char_ok = 0usize;
        let mut char_tot = 0usize;
        let (mut sep_tp, mut sep_fp, mut sep_fn) = (0usize, 0usize, 0usize);
        let (mut sp_tp, mut sp_fp, mut sp_fn) = (0usize, 0usize, 0usize);

        let gold_spans = |tags: &[usize]| -> Vec<(usize, usize, Option<SpanKind>)> {
            let mut out = Vec::new();
            let mut k = 0usize;
            while k < tags.len() {
                let Some(kind) = tag_to_kind(tags[k]) else {
                    k += 1;
                    continue;
                };
                let s0 = k;
                k += 1;
                while k < tags.len() {
                    let t = tags[k];
                    if tag_to_kind(t) != Some(kind) || is_begin_tag(t) {
                        break;
                    }
                    k += 1;
                }
                out.push((s0, k, Some(kind)));
            }
            out
        };

        for r in &va_rows {
            let pred = self.tag_chars(&r.text);
            if pred.len() != r.tags.len() {
                continue;
            }
            for (&p, &g) in pred.iter().zip(&r.tags) {
                char_tot += 1;
                if p == g {
                    char_ok += 1;
                }
                if p == TAG_SEP && g == TAG_SEP {
                    sep_tp += 1;
                } else if p == TAG_SEP && g != TAG_SEP {
                    sep_fp += 1;
                } else if p != TAG_SEP && g == TAG_SEP {
                    sep_fn += 1;
                }
            }
            let ps = gold_spans(&pred);
            let gs = gold_spans(&r.tags);
            for item in &ps {
                if gs.contains(item) {
                    sp_tp += 1;
                } else {
                    sp_fp += 1;
                }
            }
            for item in &gs {
                if !ps.contains(item) {
                    sp_fn += 1;
                }
            }
        }

        let char_acc = char_ok as f64 / char_tot.max(1) as f64;
        let sep_f1 = if 2 * sep_tp + sep_fp + sep_fn == 0 {
            1.0
        } else {
            (2 * sep_tp) as f64 / (2 * sep_tp + sep_fp + sep_fn) as f64
        };
        let span_f1 = if 2 * sp_tp + sp_fp + sp_fn == 0 {
            1.0
        } else {
            (2 * sp_tp) as f64 / (2 * sp_tp + sp_fp + sp_fn) as f64
        };
        (char_acc, sep_f1, span_f1)
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[u8::from(self.trained)])?;
        w.write_all(bytemuck::cast_slice(&self.w1))?;
        w.write_all(bytemuck::cast_slice(&self.theta))?;
        w.write_all(bytemuck::cast_slice(&self.w2))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir))?;
        w.write_all(bytemuck::cast_slice(&self.bias))?;
        w.write_all(bytemuck::cast_slice(&self.init_trans))?;
        w.write_all(bytemuck::cast_slice(&self.trans))?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid SyntaxCore checkpoint magic");
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        let mut core = Self::blank();
        core.trained = flag[0] != 0;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w2))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w_dir))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.bias))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.init_trans))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.trans))?;
        Ok(core)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::new(File::create(path)?);
        self.write_to(&mut w)?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        Self::read_from(&mut r)
    }
}

/// v24.5 文字単位純ヘブ構文・節境界知覚コアの学習と単体精度検証を実行する。
pub fn train_and_bench(
    corpus_dir: &Path,
    ckpt_path: &Path,
    out_json: &Path,
    epochs: usize,
) -> Result<Value> {
    let t0 = Instant::now();
    let txt = fs::read_to_string(corpus_dir.join("syntax_bio.jsonl"))?;
    let mut rows = Vec::new();
    for line in txt.lines().filter(|l| !l.trim().is_empty()) {
        rows.push(serde_json::from_str::<SyntaxRow>(line)?);
    }

    let mut core = SyntaxCore::blank();
    let (val_char_acc, val_sep_f1, val_span_f1) = core.train_from_rows(&rows, epochs);
    core.save(ckpt_path)?;
    let train_elapsed_s = t0.elapsed().as_secs_f64();

    let tl = Instant::now();
    let loaded = SyntaxCore::load(ckpt_path)?;
    let load_ms = tl.elapsed().as_secs_f64() * 1000.0;
    let ckpt_mib = fs::metadata(ckpt_path)?.len() as f64 / (1024.0 * 1024.0);

    let tp = Instant::now();
    let mut val_seqs = 0usize;
    for r in rows.iter().filter(|r| r.split == "val").take(200) {
        let _ = std::hint::black_box(loaded.parse(&r.text));
        val_seqs += 1;
    }
    let avg_parse_us = tp.elapsed().as_secs_f64() * 1_000_000.0 / val_seqs.max(1) as f64;
    let (rss, peak) = crate::memory::working_set();

    let report = json!({
        "ckpt_mib": ckpt_mib,
        "load_ms": load_ms,
        "train_elapsed_s": train_elapsed_s,
        "avg_parse_us": avg_parse_us,
        "rss_mib": rss,
        "peak_rss_mib": peak,
        "train_sequences": rows.iter().filter(|r| r.split != "val").count(),
        "val_sequences": rows.iter().filter(|r| r.split == "val").count(),
        "val_char_acc": val_char_acc,
        "val_sep_f1": val_sep_f1,
        "val_span_f1": val_span_f1,
    });
    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_json, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_syntax_bio_parse_and_roundtrip_work() {
        let rows: Vec<SyntaxRow> = [
            ("AA|BB:rel1;CC:10m;", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,6,7,7,0], "train"),
            ("DD|EE:rel2;FF:20m;", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,6,7,7,0], "train"),
            ("GG|HH:rel1;II:30m;", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,6,7,7,0], "train"),
            ("AA|BB:rel2;", vec![2,3,1,2,3,1,8,9,9,9,0], "train"),
            ("KK|LL:rel1;MM:45m;", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,6,7,7,0], "val"),
        ]
        .into_iter()
        .map(|(text, tags, split)| SyntaxRow {
            text: text.to_owned(),
            tags,
            split: split.to_owned(),
        })
        .collect();

        let mut core = SyntaxCore::blank();
        let (char_acc, sep_f1, _) = core.train_from_rows(&rows, 8);
        assert!(char_acc >= 0.85 && sep_f1 >= 0.85, "acc={char_acc} sep={sep_f1}");

        let parsed = core.parse("AA|BB:rel1;CC:10m;");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].spans[0].kind, SpanKind::Entity);
        assert_eq!(parsed[0].spans[0].text, "AA");
        assert_eq!(parsed[1].spans.last().unwrap().kind, SpanKind::Num);

        let tmp = std::env::temp_dir().join(format!("ng-sx245-{}.bin", std::process::id()));
        core.save(&tmp).unwrap();
        let loaded = SyntaxCore::load(&tmp).unwrap();
        std::fs::remove_file(tmp).unwrap();
        assert_eq!(loaded.tag_chars("KK|LL:rel1;"), core.tag_chars("KK|LL:rel1;"));
    }
}
