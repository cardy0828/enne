//! ENNE v25.0 発話生成・定義文蒸留コア (`BrocaCore`, `NGBC2501`)。
//!
//! 1) `FactDistiller`: 括弧深度・主題境界を含む4タグ文字 BIO オヤ・ヘブ k-WTA + Viterbi 復号。
//! 2) `DialogueIntentHead`: 日常発話 7意図に対するオヤ・ヘブ分類器。
//! 3) `BrocaDAG`: 14行動の BF16 神経新生バイトDAG（ネイティブスロット `<P0>`, `<P1>`, `<ANS>`, `<END>`）。
use crate::{core_dag::mix, metacog::MetacogCore};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use unicode_normalization::UnicodeNormalization;

pub const FD_DIN: usize = 64;
pub const FD_HIDDEN: usize = 32;
pub const FD_TOP_K: usize = 6;
pub const FD_BUCKETS: usize = 16384;
pub const FD_TAGS: usize = 4; // 0=SKIP, 1=B-DEF, 2=I-DEF, 3=TAIL

pub const DLG_DIN: usize = 128;
pub const DLG_HIDDEN: usize = 64;
pub const DLG_TOP_K: usize = 8;
pub const DLG_BUCKETS: usize = 16384;
pub const DLG_INTENT_CNT: usize = 7;

pub const DAG_DIM: usize = 128;
pub const P0_TOK: usize = 256;
pub const P1_TOK: usize = 257;
pub const ANS_TOK: usize = 258;
pub const END_TOK: usize = 259;
pub const DAG_VOCAB: usize = 260;
const BOS_SYM: u16 = 260;
const DAG_HOPS: usize = 12;
const DAG_CHAR_HIST: usize = 14;
const DAG_SHARP_K: usize = 8;
const DAG_PROMPT_BUCKETS: usize = 16384;

pub const ACT_SHORT_ANS: usize = 0;
pub const ACT_ACK: usize = 1;
pub const ACT_ABSTAIN: usize = 2;
pub const ACT_CLOCK: usize = 3;
pub const ACT_CALC: usize = 4;
pub const ACT_FACT_EXPLAIN: usize = 5;
pub const ACT_FACT_TAUGHT: usize = 6;
pub const ACT_DLG_BASE: usize = 7;
pub const BROCA_ACT_CNT: usize = 14;

const MAGIC: &[u8; 8] = b"NGBC2501";

#[derive(Deserialize)]
pub struct FactBioRow {
    pub head: String,
    pub body: String,
    pub tags: Vec<usize>,
    #[serde(default)]
    pub split: String,
}

#[derive(Deserialize)]
pub struct DlgIntentRow {
    pub text: String,
    pub intent: usize,
    #[serde(default)]
    pub split: String,
}

#[derive(Deserialize)]
pub struct BrocaEpisode {
    pub act: usize,
    pub prompt: String,
    pub target: String,
    #[serde(default)]
    pub split: String,
}

#[inline(always)]
fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[inline(always)]
fn to_bf16_sr(x: f32, rng: &mut u64) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let rand16 = (*rng as u32) & 0xffff;
    (bits.wrapping_add(rand16) >> 16) as u16
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

#[inline(always)]
fn is_open_paren(c: char) -> bool {
    matches!(c, '(' | '\u{FF08}' | '[' | '\u{3010}')
}

#[inline(always)]
fn is_close_paren(c: char) -> bool {
    matches!(c, ')' | '\u{FF09}' | ']' | '\u{3011}')
}

fn compute_paren_depths(chars: &[char]) -> Vec<usize> {
    let mut depths = Vec::with_capacity(chars.len());
    let mut depth = 0usize;
    for &c in chars {
        if is_open_paren(c) {
            depth += 1;
            depths.push(depth);
        } else if is_close_paren(c) {
            depths.push(depth);
            depth = depth.saturating_sub(1);
        } else {
            depths.push(depth);
        }
    }
    depths
}

fn compute_head_mask(
    head: &str,
    body_chars: &[char],
    depths: &[usize],
) -> (Vec<bool>, Vec<usize>, usize) {
    let hchars: Vec<char> = head.chars().collect();
    let mut in_head = vec![false; body_chars.len()];
    let mut out_pos = vec![0usize; body_chars.len()];
    let mut op = 0usize;
    let mut first_period_out = body_chars.len();
    for (i, &c) in body_chars.iter().enumerate() {
        out_pos[i] = op;
        if depths[i] == 0 {
            op += 1;
            if first_period_out == body_chars.len() && c == '\u{3002}' {
                first_period_out = i;
            }
        }
    }
    if !hchars.is_empty() && body_chars.len() >= hchars.len() {
        'outer: for start in 0..=body_chars.len().saturating_sub(hchars.len()).min(36) {
            if depths[start] > 0 {
                continue;
            }
            for k in 0..hchars.len() {
                if body_chars[start + k] != hchars[k] {
                    continue 'outer;
                }
            }
            for k in 0..hchars.len() {
                in_head[start + k] = true;
            }
            break;
        }
    }
    (in_head, out_pos, first_period_out)
}

fn extract_fact_char_features(
    chars: &[char],
    depths: &[usize],
    in_head: &[bool],
    out_pos: &[usize],
    first_period_out: usize,
    pos: usize,
) -> (Vec<usize>, [f32; FD_DIN]) {
    let n = chars.len();
    let mask = FD_BUCKETS - 1;
    let mut feats = Vec::with_capacity(32);
    let push_f = |feats: &mut Vec<usize>, tag: u64, val: u64| {
        feats.push((mix(tag ^ val.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize) & mask);
    };
    let ch = |off: isize| -> u64 {
        let p = pos as isize + off;
        if p < 0 || p >= n as isize {
            0xFFFF_FFu64
        } else {
            chars[p as usize] as u64
        }
    };
    let sc = |off: isize| -> u64 {
        let p = pos as isize + off;
        if p < 0 || p >= n as isize {
            0
        } else {
            script_class(chars[p as usize])
        }
    };

    for (idx, off) in (-2isize..=2).enumerate() {
        push_f(&mut feats, 0x1100 + idx as u64, ch(off));
        push_f(&mut feats, 0x1200 + idx as u64, sc(off));
    }
    push_f(&mut feats, 0x1301, (ch(-1) << 21) ^ ch(0));
    push_f(&mut feats, 0x1302, (ch(0) << 21) ^ ch(1));
    push_f(&mut feats, 0x1303, (ch(1) << 21) ^ ch(2));
    push_f(&mut feats, 0x1304, mix((ch(-1) << 21) ^ ch(0)) ^ ch(1));
    push_f(&mut feats, 0x1305, (sc(-1) << 8) ^ (sc(0) << 4) ^ sc(1));

    let cur_paren = depths[pos] > 0;
    let past_period = pos >= first_period_out;
    let dist_to_end = first_period_out.saturating_sub(pos).min(6) as u64;
    let mut head_end_out = 0usize;
    for i in 0..n {
        if in_head[i] {
            head_end_out = out_pos[i] + 1;
        }
    }
    let after_head_out = out_pos[pos].saturating_sub(head_end_out).min(5) as u64;
    let past_head = head_end_out > 0 && out_pos[pos] >= head_end_out && !in_head[pos];
    let early_out = (out_pos[pos] < 18) as u64;
    let sig_flags = ((cur_paren as u64) << 4)
        | ((past_period as u64) << 3)
        | ((in_head[pos] as u64) << 2)
        | ((past_head as u64) << 1)
        | early_out;
    push_f(&mut feats, 0x1401, sig_flags);
    push_f(&mut feats, 0x1402, (sig_flags << 12) ^ ch(0));
    push_f(&mut feats, 0x1403, (sig_flags << 8) ^ (ch(0) << 21) ^ ch(1));
    push_f(&mut feats, 0x1404, (past_period as u64) << 4 | dist_to_end);
    push_f(&mut feats, 0x1405, (dist_to_end << 21) ^ ch(0));
    push_f(&mut feats, 0x1406, ((past_head as u64) << 8) ^ (after_head_out << 24) ^ ch(0));
    push_f(&mut feats, 0x1407, ((past_head as u64) << 8) ^ (after_head_out << 24) ^ ((ch(0) << 21) ^ ch(1)));

    feats.sort_unstable();
    feats.dedup();

    let mut x = [0.0f32; FD_DIN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x517c_c1b7_2722_0a95);
        for _ in 0..4 {
            let d = (h as usize) & (FD_DIN - 1);
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

fn extract_text_ngrams(raw: &str, buckets: usize) -> (Vec<usize>, [f32; DLG_DIN]) {
    let q: String = raw.trim().nfkc().collect();
    let chars: Vec<char> = q.chars().filter(|c| !c.is_whitespace()).collect();
    let mask = buckets - 1;
    let mut feats = Vec::with_capacity(chars.len() * 3 + 4);
    for i in 0..chars.len() {
        let c0 = chars[i] as u64;
        feats.push((mix(c0 ^ 0x101) as usize) & mask);
        if i + 1 < chars.len() {
            let c1 = chars[i + 1] as u64;
            feats.push((mix((c0 << 21) ^ c1 ^ 0x202) as usize) & mask);
            if i + 2 < chars.len() {
                let c2 = chars[i + 2] as u64;
                feats.push((mix(mix((c0 << 21) ^ c1) ^ c2 ^ 0x303) as usize) & mask);
            }
        }
    }
    if !chars.is_empty() {
        feats.push((mix(0x8888u64 ^ (*chars.last().unwrap() as u64)) as usize) & mask);
    }
    feats.sort_unstable();
    feats.dedup();

    let mut x = [0.0f32; DLG_DIN];
    for &f in &feats {
        let mut h = mix(f as u64 ^ 0x243f_6a88_85a3_08d3);
        for _ in 0..4 {
            let d = (h as usize) & (DLG_DIN - 1);
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

pub fn tokenize_broca_target(mut s: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(s.len() + 2);
    while !s.is_empty() {
        if let Some(rest) = s.strip_prefix("<P0>") {
            out.push(P0_TOK as u16);
            s = rest;
        } else if let Some(rest) = s.strip_prefix("<P1>") {
            out.push(P1_TOK as u16);
            s = rest;
        } else if let Some(rest) = s.strip_prefix("<ANS>") {
            out.push(ANS_TOK as u16);
            s = rest;
        } else {
            let ch = s.chars().next().unwrap();
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                out.push(b as u16);
            }
            s = &s[ch.len_utf8()..];
        }
    }
    out.push(END_TOK as u16);
    out
}

#[derive(Clone, Copy, Default)]
struct Utf8State {
    rem: u8,
}
impl Utf8State {
    fn feed(&mut self, tok: u16) {
        if tok >= 256 {
            self.rem = 0;
            return;
        }
        let b = tok as u8;
        if self.rem > 0 {
            if (b & 0xC0) == 0x80 {
                self.rem -= 1;
            } else {
                self.rem = 0;
            }
        } else if b < 0x80 {
            self.rem = 0;
        } else if (0xC2..=0xDF).contains(&b) {
            self.rem = 1;
        } else if (0xE0..=0xEF).contains(&b) {
            self.rem = 2;
        } else if (0xF0..=0xF4).contains(&b) {
            self.rem = 3;
        }
    }
    fn allows(&self, tok: usize) -> bool {
        if self.rem > 0 {
            tok < 256 && ((tok as u8) & 0xC0) == 0x80
        } else if tok >= 256 {
            true
        } else {
            let b = tok as u8;
            b < 0x80 || (0xC2..=0xF4).contains(&b)
        }
    }
}

#[derive(Clone)]
struct CharHist {
    items: [u64; DAG_CHAR_HIST],
    len: usize,
    pending: [u8; 4],
    plen: usize,
    need: usize,
}
impl CharHist {
    fn new() -> Self {
        Self {
            items: [0; DAG_CHAR_HIST],
            len: 0,
            pending: [0; 4],
            plen: 0,
            need: 0,
        }
    }
    fn push_raw(&mut self, sym: u64) {
        if self.len < DAG_CHAR_HIST {
            self.items[self.len] = sym;
            self.len += 1;
        } else {
            self.items.copy_within(1..DAG_CHAR_HIST, 0);
            self.items[DAG_CHAR_HIST - 1] = sym;
        }
    }
    fn push_tok(&mut self, tok: u16) {
        if tok >= 256 {
            self.plen = 0;
            self.need = 0;
            self.push_raw(0x1_0000_0000u64 | tok as u64);
            return;
        }
        let b = tok as u8;
        if self.plen == 0 {
            if b < 0x80 {
                self.push_raw(b as u64);
            } else {
                self.need = if (0xC2..=0xDF).contains(&b) {
                    2
                } else if (0xE0..=0xEF).contains(&b) {
                    3
                } else if (0xF0..=0xF4).contains(&b) {
                    4
                } else {
                    1
                };
                self.pending[0] = b;
                self.plen = 1;
                if self.need == 1 {
                    self.push_raw(b as u64);
                    self.plen = 0;
                    self.need = 0;
                }
            }
        } else {
            self.pending[self.plen] = b;
            self.plen += 1;
            if self.plen >= self.need {
                let mut code = 0u64;
                for i in 0..self.plen {
                    code = (code << 8) | self.pending[i] as u64;
                }
                self.push_raw(code);
                self.plen = 0;
                self.need = 0;
            }
        }
    }
    #[inline(always)]
    fn hop_key(&self, depth: usize) -> Option<u64> {
        if depth == 0 && self.plen > 0 {
            let mut code = 0x2_0000_0000u64;
            for i in 0..self.plen {
                code = (code << 8) | self.pending[i] as u64;
            }
            return Some(code);
        }
        let off = if self.plen > 0 { depth - 1 } else { depth };
        if off < self.len {
            Some(self.items[self.len - 1 - off])
        } else {
            None
        }
    }
    #[inline(always)]
    fn byte_phase(&self) -> u8 {
        self.plen as u8
    }
}

#[inline(always)]
fn dag_sig64(p: u32, phase: u8, c: u64) -> u64 {
    mix(
        mix(mix(p as u64 + 31) ^ ((phase as u64 + 1) * 1_000_033))
            ^ mix(c.wrapping_add(2_000_039)),
    )
    .max(1)
}

#[inline(always)]
fn dag_gain(depth: usize) -> f32 {
    match depth {
        0 => 0.25,
        1 => 0.90,
        2 => 1.10,
        _ => 1.30,
    }
}

#[inline(always)]
fn hebb_update_bf16_128(w: &mut [u16], err: &[f32; DAG_DIM], lr: f32, rng: &mut [u32; 8]) -> f32 {
    let mut align = [0.0f32; 8];
    let mut j = 0;
    while j < DAG_DIM {
        for lane in 0..8 {
            let idx = j + lane;
            let wv = from_bf16(w[idx]);
            let ev = err[idx];
            align[lane] += wv * ev;
            let nv = wv + lr * ev;
            let mut r = rng[lane];
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            rng[lane] = r;
            let bits = nv.to_bits();
            let rounded = if (bits & 0x7f80_0000) == 0x7f80_0000 {
                bits >> 16
            } else {
                bits.wrapping_add(r & 0xffff) >> 16
            };
            w[idx] = rounded as u16;
        }
        j += 8;
    }
    align.iter().sum()
}

fn build_dag_basis() -> Vec<f32> {
    let mut e = vec![0.0f32; DAG_VOCAB * DAG_DIM];
    let mut state = 0x517c_c1b7_2722_0a95u64;
    for t in 0..DAG_VOCAB {
        let mut norm2 = 0.0f32;
        for d in 0..DAG_DIM {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
            e[t * DAG_DIM + d] = u;
            norm2 += u * u;
        }
        let inv = 1.0 / (norm2.sqrt() + 1e-8);
        for d in 0..DAG_DIM {
            e[t * DAG_DIM + d] *= inv;
        }
    }
    e
}

fn rand_unit_vec(hidden: usize, dim: usize, mut st: u64) -> Vec<f32> {
    let mut w = vec![0.0f32; hidden * dim];
    for h in 0..hidden {
        let mut n2 = 0.0f32;
        for d in 0..dim {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let u = ((st >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
            w[h * dim + d] = u;
            n2 += u * u;
        }
        let inv = 1.0 / (n2.sqrt() + 1e-8);
        for d in 0..dim {
            w[h * dim + d] *= inv;
        }
    }
    w
}

pub struct BrocaCore {
    // 1. FactDistiller
    pub fd_trained: bool,
    pub fd_w1: Vec<f32>,
    pub fd_theta: [f32; FD_HIDDEN],
    pub fd_w2: Vec<f32>,
    pub fd_w_dir: Vec<f32>,
    pub fd_bias: [f32; FD_TAGS],
    pub fd_init_log: [f32; FD_TAGS],
    pub fd_trans_log: [f32; FD_TAGS * FD_TAGS],

    // 2. DialogueIntentHead
    pub dlg_w1: Vec<f32>,
    pub dlg_theta: [f32; DLG_HIDDEN],
    pub dlg_w2: Vec<f32>,
    pub dlg_w_dir: Vec<f32>,
    pub dlg_bias: [f32; DLG_INTENT_CNT],

    // 3. BrocaDAG
    pub dag_cap: usize,
    pub dag_alive: usize,
    pub dag_e: Vec<f32>,
    pub dag_bias: Vec<f32>,
    pub dag_w_node: Vec<u16>,
    pub dag_sharp_dst: Vec<i16>,
    pub dag_sharp_w: Vec<u16>,
    pub dag_fires: Vec<u32>,
    pub dag_edge_buckets: usize,
    pub dag_edge_sigs: Vec<u64>,
    pub dag_edge_children: Vec<i32>,
    pub dag_w_prompt: Vec<u16>,
}

impl Default for BrocaCore {
    fn default() -> Self {
        Self::new(45_000)
    }
}

impl BrocaCore {
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(BROCA_ACT_CNT + 256);
        let edge_buckets = (cap * 2).next_power_of_two();
        Self {
            fd_trained: false,
            fd_w1: rand_unit_vec(FD_HIDDEN, FD_DIN, 0x3a39_8031_912b_5511),
            fd_theta: [0.0f32; FD_HIDDEN],
            fd_w2: vec![0.0f32; FD_HIDDEN * FD_TAGS],
            fd_w_dir: vec![0.0f32; FD_BUCKETS * FD_TAGS],
            fd_bias: [0.0f32; FD_TAGS],
            fd_init_log: [0.0f32; FD_TAGS],
            fd_trans_log: [0.0f32; FD_TAGS * FD_TAGS],
            dlg_w1: rand_unit_vec(DLG_HIDDEN, DLG_DIN, 0x6a09_e667_f3bc_c908),
            dlg_theta: [0.0f32; DLG_HIDDEN],
            dlg_w2: vec![0.0f32; DLG_HIDDEN * DLG_INTENT_CNT],
            dlg_w_dir: vec![0.0f32; DLG_BUCKETS * DLG_INTENT_CNT],
            dlg_bias: [0.0f32; DLG_INTENT_CNT],
            dag_cap: cap,
            dag_alive: BROCA_ACT_CNT,
            dag_e: build_dag_basis(),
            dag_bias: vec![0.0f32; BROCA_ACT_CNT * DAG_VOCAB],
            dag_w_node: vec![0u16; cap * DAG_DIM],
            dag_sharp_dst: vec![-1i16; cap * DAG_SHARP_K],
            dag_sharp_w: vec![0u16; cap * DAG_SHARP_K],
            dag_fires: vec![0u32; cap],
            dag_edge_buckets: edge_buckets,
            dag_edge_sigs: vec![0u64; edge_buckets * 4],
            dag_edge_children: vec![-1i32; edge_buckets * 4],
            dag_w_prompt: vec![0u16; DAG_PROMPT_BUCKETS * DAG_DIM],
        }
    }

    // -------------------------------------------------------------------------
    // FactDistiller
    // -------------------------------------------------------------------------

    #[inline]
    fn fd_hidden_act(&self, x: &[f32; FD_DIN]) -> [(usize, f32); FD_TOP_K] {
        let mut scores = [(0usize, 0.0f32); FD_HIDDEN];
        for h in 0..FD_HIDDEN {
            let row = &self.fd_w1[h * FD_DIN..(h + 1) * FD_DIN];
            let mut s = 0.0f32;
            for d in 0..FD_DIN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.fd_theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); FD_TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..FD_TOP_K {
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
    fn fd_emit_logits(&self, feats: &[usize], x: &[f32; FD_DIN]) -> [f32; FD_TAGS] {
        let winners = self.fd_hidden_act(x);
        let mut logits = self.fd_bias;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * FD_TAGS;
                for c in 0..FD_TAGS {
                    logits[c] += act * self.fd_w2[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in feats {
            let base = f * FD_TAGS;
            for c in 0..FD_TAGS {
                logits[c] += inv_f * self.fd_w_dir[base + c];
            }
        }
        logits
    }

    pub fn tag_fact_chars(&self, head: &str, body: &str) -> Vec<usize> {
        let chars: Vec<char> = body.chars().collect();
        let n = chars.len();
        if n == 0 || !self.fd_trained {
            return vec![0; n];
        }
        let depths = compute_paren_depths(&chars);
        let (in_head, out_pos, first_period_out) = compute_head_mask(head, &chars, &depths);

        let mut dp = vec![[f32::NEG_INFINITY; FD_TAGS]; n];
        let mut back = vec![[0u8; FD_TAGS]; n];
        let (f0, x0) =
            extract_fact_char_features(&chars, &depths, &in_head, &out_pos, first_period_out, 0);
        let l0 = self.fd_emit_logits(&f0, &x0);
        for c in 0..FD_TAGS {
            dp[0][c] = l0[c] + self.fd_init_log[c];
        }
        for t in 1..n {
            let (ft, xt) =
                extract_fact_char_features(&chars, &depths, &in_head, &out_pos, first_period_out, t);
            let lt = self.fd_emit_logits(&ft, &xt);
            for cur in 0..FD_TAGS {
                let mut best_prev = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for prev in 0..FD_TAGS {
                    let s = dp[t - 1][prev] + self.fd_trans_log[prev * FD_TAGS + cur];
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
        for c in 1..FD_TAGS {
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

    /// Distills the concise outer-sentence definitional core of a Wikipedia lead using learned BIO tags.
    pub fn distill_fact(&self, head: &str, body: &str) -> String {
        let chars: Vec<char> = body.chars().collect();
        if chars.is_empty() {
            return String::new();
        }
        let depths = compute_paren_depths(&chars);
        let tags = self.tag_fact_chars(head, body);
        let mut out = String::new();
        let mut seen_def = false;
        let mut count = 0usize;
        for (i, &tg) in tags.iter().enumerate() {
            if tg == 1 || tg == 2 {
                if depths[i] == 0 {
                    out.push(chars[i]);
                    seen_def = true;
                    count += 1;
                    if count >= 110 {
                        break;
                    }
                }
            } else if tg == 3 && seen_def {
                break;
            }
        }
        let trimmed = out.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
        // Fallback if model weights are untrained: strip parentheticals and cap cleanly
        let mut fb = String::new();
        for (i, &c) in chars.iter().enumerate() {
            if depths[i] == 0 {
                fb.push(c);
                if fb.chars().count() >= 80 {
                    break;
                }
            }
        }
        fb.trim().to_owned()
    }

    pub fn train_fact_distiller(&mut self, rows: &[FactBioRow], epochs: usize) -> f64 {
        let train: Vec<&FactBioRow> = rows.iter().filter(|r| r.split != "val").collect();
        let val: Vec<&FactBioRow> = rows.iter().filter(|r| r.split == "val").collect();
        if train.is_empty() {
            return 0.0;
        }

        let mut init_cnt = [1.0f64; FD_TAGS];
        let mut trans_cnt = [[1.0f64; FD_TAGS]; FD_TAGS];
        let mut tag_cnt = [1usize; FD_TAGS];
        let mut prepared = Vec::with_capacity(train.len());

        for row in &train {
            let chars: Vec<char> = row.body.chars().collect();
            if chars.is_empty() || chars.len() != row.tags.len() {
                continue;
            }
            let depths = compute_paren_depths(&chars);
            let (in_head, out_pos, first_p) = compute_head_mask(&row.head, &chars, &depths);
            init_cnt[row.tags[0].min(FD_TAGS - 1)] += 1.0;
            for w in row.tags.windows(2) {
                trans_cnt[w[0].min(FD_TAGS - 1)][w[1].min(FD_TAGS - 1)] += 1.0;
            }
            let mut seq = Vec::with_capacity(chars.len());
            for (i, &tg) in row.tags.iter().enumerate() {
                let t = tg.min(FD_TAGS - 1);
                tag_cnt[t] += 1;
                let (feats, x) =
                    extract_fact_char_features(&chars, &depths, &in_head, &out_pos, first_p, i);
                seq.push((feats, x, t));
            }
            prepared.push(seq);
        }

        let init_sum: f64 = init_cnt.iter().sum();
        for c in 0..FD_TAGS {
            self.fd_init_log[c] = (init_cnt[c] / init_sum).ln() as f32 * 0.4;
        }
        for prev in 0..FD_TAGS {
            let rsum: f64 = trans_cnt[prev].iter().sum();
            for cur in 0..FD_TAGS {
                self.fd_trans_log[prev * FD_TAGS + cur] =
                    (trans_cnt[prev][cur] / rsum).ln() as f32 * 0.4;
            }
        }

        let target_rate = FD_TOP_K as f32 / FD_HIDDEN as f32;
        for pass in 0..2 {
            let eta = 0.04 / (1.0 + 0.5 * pass as f32);
            for seq in &prepared {
                for (_, x, _) in seq.iter().step_by(2) {
                    let winners = self.fd_hidden_act(x);
                    let mut won = [false; FD_HIDDEN];
                    for &(h, act) in &winners {
                        if act <= 0.0 {
                            continue;
                        }
                        won[h] = true;
                        let row = &mut self.fd_w1[h * FD_DIN..(h + 1) * FD_DIN];
                        let mut n2 = 0.0f32;
                        for d in 0..FD_DIN {
                            let nv = row[d] + eta * act * (x[d] - act * row[d]);
                            row[d] = nv;
                            n2 += nv * nv;
                        }
                        let inv = 1.0 / (n2.sqrt() + 1e-8);
                        for d in 0..FD_DIN {
                            row[d] *= inv;
                        }
                    }
                    for h in 0..FD_HIDDEN {
                        let fired = if won[h] { 1.0 } else { 0.0 };
                        self.fd_theta[h] =
                            (self.fd_theta[h] + 0.001 * (fired - target_rate)).clamp(-0.35, 0.35);
                    }
                }
            }
        }

        let max_t = *tag_cnt.iter().max().unwrap_or(&1) as f32;
        let mut tag_wt = [1.0f32; FD_TAGS];
        for c in 0..FD_TAGS {
            tag_wt[c] = (max_t / tag_cnt[c] as f32).sqrt().min(2.2);
        }

        for epoch in 0..epochs.max(1) {
            let lr = 0.22 / (1.0 + 0.35 * epoch as f32);
            for seq in &prepared {
                for (feats, x, gold) in seq {
                    let winners = self.fd_hidden_act(x);
                    let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                    let cw = tag_wt[*gold];
                    let logits = self.fd_emit_logits(feats, x);
                    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs = [0.0f32; FD_TAGS];
                    let mut sum_e = 0.0f32;
                    for c in 0..FD_TAGS {
                        probs[c] = (logits[c] - mx).exp();
                        sum_e += probs[c];
                    }
                    for c in 0..FD_TAGS {
                        probs[c] /= sum_e + 1e-12;
                        let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                        self.fd_bias[c] += lr * cw * 0.03 * delta;
                    }
                    for &(h, act) in &winners {
                        if act > 0.0 {
                            let base = h * FD_TAGS;
                            for c in 0..FD_TAGS {
                                let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                                self.fd_w2[base + c] += lr * cw * 0.30 * act * delta;
                            }
                        }
                    }
                    for &f in feats {
                        let base = f * FD_TAGS;
                        for c in 0..FD_TAGS {
                            let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                            self.fd_w_dir[base + c] += lr * cw * inv_f * delta;
                        }
                    }
                }
            }
        }
        self.fd_trained = true;

        if val.is_empty() {
            return 1.0;
        }
        let mut ok = 0usize;
        let mut tot = 0usize;
        for row in val.iter().take(400) {
            let pred = self.tag_fact_chars(&row.head, &row.body);
            for (&p, &g) in pred.iter().zip(row.tags.iter()) {
                tot += 1;
                if p == g {
                    ok += 1;
                }
            }
        }
        ok as f64 / tot.max(1) as f64
    }

    // -------------------------------------------------------------------------
    // DialogueIntentHead
    // -------------------------------------------------------------------------

    #[inline]
    fn dlg_hidden_act(&self, x: &[f32; DLG_DIN]) -> [(usize, f32); DLG_TOP_K] {
        let mut scores = [(0usize, 0.0f32); DLG_HIDDEN];
        for h in 0..DLG_HIDDEN {
            let row = &self.dlg_w1[h * DLG_DIN..(h + 1) * DLG_DIN];
            let mut s = 0.0f32;
            for d in 0..DLG_DIN {
                s += row[d] * x[d];
            }
            scores[h] = (h, (s - self.dlg_theta[h]).max(0.0));
        }
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut top = [(0usize, 0.0f32); DLG_TOP_K];
        let mut norm2 = 0.0f32;
        for i in 0..DLG_TOP_K {
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

    pub fn predict_dialogue_intent(&self, prompt: &str) -> (usize, [f32; DLG_INTENT_CNT]) {
        let (feats, x) = extract_text_ngrams(prompt, DLG_BUCKETS);
        let winners = self.dlg_hidden_act(&x);
        let mut logits = self.dlg_bias;
        for &(h, act) in &winners {
            if act > 0.0 {
                let base = h * DLG_INTENT_CNT;
                for c in 0..DLG_INTENT_CNT {
                    logits[c] += act * self.dlg_w2[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * DLG_INTENT_CNT;
            for c in 0..DLG_INTENT_CNT {
                logits[c] += inv_f * self.dlg_w_dir[base + c];
            }
        }
        let mut best = 0;
        for c in 1..DLG_INTENT_CNT {
            if logits[c] > logits[best] {
                best = c;
            }
        }
        (best, logits)
    }

    pub fn train_dialogue_intent(&mut self, rows: &[DlgIntentRow], epochs: usize) -> f64 {
        let train: Vec<&DlgIntentRow> = rows.iter().filter(|r| r.split != "val").collect();
        let val: Vec<&DlgIntentRow> = rows.iter().filter(|r| r.split == "val").collect();
        if train.is_empty() {
            return 0.0;
        }
        let prep: Vec<(Vec<usize>, [f32; DLG_DIN], usize)> = train
            .iter()
            .map(|r| {
                let (f, x) = extract_text_ngrams(&r.text, DLG_BUCKETS);
                (f, x, r.intent.min(DLG_INTENT_CNT - 1))
            })
            .collect();

        let target_rate = DLG_TOP_K as f32 / DLG_HIDDEN as f32;
        for pass in 0..2 {
            let eta = 0.05 / (1.0 + 0.5 * pass as f32);
            for (_, x, _) in &prep {
                let winners = self.dlg_hidden_act(x);
                let mut won = [false; DLG_HIDDEN];
                for &(h, act) in &winners {
                    if act <= 0.0 {
                        continue;
                    }
                    won[h] = true;
                    let row = &mut self.dlg_w1[h * DLG_DIN..(h + 1) * DLG_DIN];
                    let mut n2 = 0.0f32;
                    for d in 0..DLG_DIN {
                        let nv = row[d] + eta * act * (x[d] - act * row[d]);
                        row[d] = nv;
                        n2 += nv * nv;
                    }
                    let inv = 1.0 / (n2.sqrt() + 1e-8);
                    for d in 0..DLG_DIN {
                        row[d] *= inv;
                    }
                }
                for h in 0..DLG_HIDDEN {
                    let fired = if won[h] { 1.0 } else { 0.0 };
                    self.dlg_theta[h] =
                        (self.dlg_theta[h] + 0.001 * (fired - target_rate)).clamp(-0.35, 0.35);
                }
            }
        }

        let mut cnt = [1usize; DLG_INTENT_CNT];
        for (_, _, it) in &prep {
            cnt[*it] += 1;
        }
        let max_c = *cnt.iter().max().unwrap_or(&1) as f32;
        let mut wt = [1.0f32; DLG_INTENT_CNT];
        for c in 0..DLG_INTENT_CNT {
            wt[c] = (max_c / cnt[c] as f32).sqrt().min(2.5);
        }

        for epoch in 0..epochs.max(1) {
            let lr = 0.25 / (1.0 + 0.25 * epoch as f32);
            for (feats, x, gold) in &prep {
                let winners = self.dlg_hidden_act(x);
                let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
                let cw = wt[*gold];
                let mut logits = self.dlg_bias;
                for &(h, act) in &winners {
                    if act > 0.0 {
                        let base = h * DLG_INTENT_CNT;
                        for c in 0..DLG_INTENT_CNT {
                            logits[c] += act * self.dlg_w2[base + c];
                        }
                    }
                }
                for &f in feats {
                    let base = f * DLG_INTENT_CNT;
                    for c in 0..DLG_INTENT_CNT {
                        logits[c] += inv_f * self.dlg_w_dir[base + c];
                    }
                }
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = [0.0f32; DLG_INTENT_CNT];
                let mut sum_e = 0.0f32;
                for c in 0..DLG_INTENT_CNT {
                    probs[c] = (logits[c] - mx).exp();
                    sum_e += probs[c];
                }
                for c in 0..DLG_INTENT_CNT {
                    probs[c] /= sum_e + 1e-12;
                    let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                    self.dlg_bias[c] += lr * cw * 0.05 * delta;
                }
                for &(h, act) in &winners {
                    if act > 0.0 {
                        let base = h * DLG_INTENT_CNT;
                        for c in 0..DLG_INTENT_CNT {
                            let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                            self.dlg_w2[base + c] += lr * cw * 0.35 * act * delta;
                        }
                    }
                }
                for &f in feats {
                    let base = f * DLG_INTENT_CNT;
                    for c in 0..DLG_INTENT_CNT {
                        let delta = if c == *gold { 1.0 - probs[c] } else { -probs[c] };
                        self.dlg_w_dir[base + c] += lr * cw * inv_f * delta;
                    }
                }
            }
        }

        if val.is_empty() {
            return 1.0;
        }
        let ok = val
            .iter()
            .filter(|r| self.predict_dialogue_intent(&r.text).0 == r.intent)
            .count();
        ok as f64 / val.len() as f64
    }

    // -------------------------------------------------------------------------
    // BrocaDAG (BF16 neural autoregressive generation with slot pointers)
    // -------------------------------------------------------------------------

    #[inline(always)]
    fn edge_lookup(&self, parent: u32, phase: u8, key: u64) -> Option<usize> {
        let s = dag_sig64(parent, phase, key);
        let j = ((s as usize) & (self.dag_edge_buckets - 1)) * 4;
        for k in j..j + 4 {
            if self.dag_edge_sigs[k] == s && self.dag_edge_children[k] >= 0 {
                return Some(self.dag_edge_children[k] as usize);
            }
        }
        None
    }

    #[inline(always)]
    fn edge_insert(&mut self, parent: u32, phase: u8, key: u64, child: usize) -> bool {
        let s = dag_sig64(parent, phase, key);
        let j = ((s as usize) & (self.dag_edge_buckets - 1)) * 4;
        for k in j..j + 4 {
            if self.dag_edge_sigs[k] == 0 || self.dag_edge_sigs[k] == s {
                self.dag_edge_sigs[k] = s;
                self.dag_edge_children[k] = child as i32;
                return true;
            }
        }
        false
    }

    fn dag_prompt_vec(&self, prompt: &str) -> [f32; DAG_DIM] {
        let (feats, _) = extract_text_ngrams(prompt, DAG_PROMPT_BUCKETS);
        let mut h = [0.0f32; DAG_DIM];
        if feats.is_empty() {
            return h;
        }
        let inv = 1.0 / (feats.len() as f32).sqrt();
        for &f in &feats {
            let row = &self.dag_w_prompt[f * DAG_DIM..(f + 1) * DAG_DIM];
            for d in 0..DAG_DIM {
                h[d] += from_bf16(row[d]) * inv;
            }
        }
        let norm = h.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1.0 {
            let s = 1.0 / norm;
            for x in &mut h {
                *x *= s;
            }
        }
        h
    }

    #[inline(always)]
    fn accum_node(&self, node: usize, g: f32, h: &mut [f32; DAG_DIM], logits: &mut [f32; DAG_VOCAB]) {
        let wrow = &self.dag_w_node[node * DAG_DIM..(node + 1) * DAG_DIM];
        for d in 0..DAG_DIM {
            h[d] += from_bf16(wrow[d]) * g;
        }
        let sbase = node * DAG_SHARP_K;
        for k in 0..DAG_SHARP_K {
            let dst = self.dag_sharp_dst[sbase + k];
            if dst >= 0 {
                logits[dst as usize] += from_bf16(self.dag_sharp_w[sbase + k]) * g * 0.75;
            }
        }
    }

    fn forward_step(
        &self,
        act: usize,
        hist: &CharHist,
        h_prompt: &[f32; DAG_DIM],
        prompt_scale: f32,
        path: &mut Vec<usize>,
    ) -> [f32; DAG_VOCAB] {
        path.clear();
        let mut h = [0.0f32; DAG_DIM];
        let mut logits = [0.0f32; DAG_VOCAB];
        let root = act.min(BROCA_ACT_CNT - 1);
        let bbase = root * DAG_VOCAB;
        logits.copy_from_slice(&self.dag_bias[bbase..bbase + DAG_VOCAB]);

        let mut cur = root;
        path.push(cur);
        self.accum_node(cur, dag_gain(0), &mut h, &mut logits);

        let phase = hist.byte_phase();
        for depth in 0..DAG_HOPS {
            let Some(key) = hist.hop_key(depth) else {
                break;
            };
            if let Some(child) = self.edge_lookup(cur as u32, phase, key) {
                cur = child;
                path.push(cur);
                self.accum_node(cur, dag_gain(depth + 1), &mut h, &mut logits);
            } else {
                break;
            }
        }

        if prompt_scale > 0.0 && phase == 0 {
            let decay = if path.len() <= 2 {
                1.0
            } else {
                1.0 / (1.0 + 0.70 * (path.len().saturating_sub(1) as f32))
            };
            let s = prompt_scale * decay;
            for d in 0..DAG_DIM {
                h[d] += s * h_prompt[d];
            }
        }

        for t in 0..DAG_VOCAB {
            let erow = &self.dag_e[t * DAG_DIM..(t + 1) * DAG_DIM];
            let mut s = 0.0f32;
            for d in 0..DAG_DIM {
                s += erow[d] * h[d];
            }
            logits[t] += s;
        }
        logits
    }

    fn learn_dag_episode(
        &mut self,
        ep: &BrocaEpisode,
        lr_base: f32,
        rng: &mut [u32; 8],
        sr_rng: &mut u64,
        allow_grow: bool,
    ) {
        let root = ep.act.min(BROCA_ACT_CNT - 1);
        let toks = tokenize_broca_target(&ep.target);
        if toks.is_empty() {
            return;
        }
        let is_chat = root >= ACT_DLG_BASE;
        let p_feats = if is_chat {
            extract_text_ngrams(&ep.prompt, DAG_PROMPT_BUCKETS).0
        } else {
            Vec::new()
        };
        let p_scale = if is_chat { 0.25f32 } else { 0.0f32 };
        let h_prompt = if is_chat {
            self.dag_prompt_vec(&ep.prompt)
        } else {
            [0.0f32; DAG_DIM]
        };

        let mut hist = CharHist::new();
        hist.push_tok(BOS_SYM);
        let mut path = Vec::with_capacity(DAG_HOPS + 1);
        let mut probs = [0.0f32; DAG_VOCAB];
        let mut err_vec = [0.0f32; DAG_DIM];

        for &target_u16 in &toks {
            let target = target_u16 as usize;
            let logits = self.forward_step(root, &hist, &h_prompt, p_scale, &mut path);
            let mut max_l = f32::NEG_INFINITY;
            let mut pred = 0usize;
            for t in 0..DAG_VOCAB {
                if logits[t] > max_l {
                    max_l = logits[t];
                    pred = t;
                }
            }
            let mut sum_e = 0.0f32;
            for t in 0..DAG_VOCAB {
                let v = (logits[t] - max_l).exp();
                probs[t] = v;
                sum_e += v;
            }
            let inv_sum = 1.0 / (sum_e + 1e-12);
            for t in 0..DAG_VOCAB {
                probs[t] *= inv_sum;
            }
            let p_tgt = probs[target];

            err_vec.copy_from_slice(&self.dag_e[target * DAG_DIM..(target + 1) * DAG_DIM]);
            for t in 0..DAG_VOCAB {
                let pt = probs[t];
                if pt > 1e-4 {
                    let erow = &self.dag_e[t * DAG_DIM..(t + 1) * DAG_DIM];
                    for d in 0..DAG_DIM {
                        err_vec[d] -= pt * erow[d];
                    }
                }
            }

            let bbase = root * DAG_VOCAB;
            let lr_b = lr_base * 0.03;
            for t in 0..DAG_VOCAB {
                let delta = if t == target { 1.0 - probs[t] } else { -probs[t] };
                self.dag_bias[bbase + t] += lr_b * delta;
            }

            let plen = path.len();
            for (depth, &node) in path.iter().enumerate() {
                self.dag_fires[node] = self.dag_fires[node].saturating_add(1);
                let g = dag_gain(depth);
                let lr = if depth == 0 {
                    lr_base * 0.16
                } else {
                    lr_base * g * (1.0 / (1.0 + 0.002 * (self.dag_fires[node] as f32).sqrt()))
                };
                let _ = hebb_update_bf16_128(
                    &mut self.dag_w_node[node * DAG_DIM..(node + 1) * DAG_DIM],
                    &err_vec,
                    lr,
                    rng,
                );
                if depth == 0 {
                    continue;
                }
                let sbase = node * DAG_SHARP_K;
                let lr_s = lr * 1.15;
                let mut found = None;
                let mut empty = None;
                let mut weakest = sbase;
                let mut weak_abs = f32::INFINITY;
                for k in sbase..sbase + DAG_SHARP_K {
                    if self.dag_sharp_dst[k] == target as i16 {
                        found = Some(k);
                        break;
                    }
                    if self.dag_sharp_dst[k] < 0 && empty.is_none() {
                        empty = Some(k);
                    }
                    let aw = from_bf16(self.dag_sharp_w[k]).abs();
                    if aw < weak_abs {
                        weak_abs = aw;
                        weakest = k;
                    }
                }
                let slot = found.or(empty).unwrap_or(weakest);
                let prev_w = if self.dag_sharp_dst[slot] == target as i16 {
                    from_bf16(self.dag_sharp_w[slot])
                } else {
                    0.0
                };
                self.dag_sharp_dst[slot] = target as i16;
                self.dag_sharp_w[slot] =
                    to_bf16_sr((prev_w + lr_s * (1.0 - p_tgt)).clamp(-6.0, 6.0), sr_rng);
                if pred != target {
                    for k in sbase..sbase + DAG_SHARP_K {
                        if self.dag_sharp_dst[k] == pred as i16 {
                            let nw =
                                (from_bf16(self.dag_sharp_w[k]) - lr_s * probs[pred]).clamp(-6.0, 6.0);
                            self.dag_sharp_w[k] = to_bf16_sr(nw, sr_rng);
                            break;
                        }
                    }
                }
            }

            if is_chat && !p_feats.is_empty() && hist.byte_phase() == 0 {
                let lr_p = lr_base * 0.08 / (p_feats.len() as f32).sqrt();
                for &f in &p_feats {
                    let _ = hebb_update_bf16_128(
                        &mut self.dag_w_prompt[f * DAG_DIM..(f + 1) * DAG_DIM],
                        &err_vec,
                        lr_p,
                        rng,
                    );
                }
            }

            if allow_grow && pred != target && plen <= DAG_HOPS && self.dag_alive < self.dag_cap {
                let leaf = path[plen - 1];
                let next_depth = plen - 1;
                if self.dag_fires[leaf] >= 1 {
                    if let Some(key) = hist.hop_key(next_depth) {
                        let phase = hist.byte_phase();
                        let child = self.dag_alive;
                        if self.edge_insert(leaf as u32, phase, key, child) {
                            self.dag_alive += 1;
                            self.dag_fires[child] = 1;
                            let _ = hebb_update_bf16_128(
                                &mut self.dag_w_node[child * DAG_DIM..(child + 1) * DAG_DIM],
                                &err_vec,
                                lr_base * 0.90,
                                rng,
                            );
                            let sbase = child * DAG_SHARP_K;
                            self.dag_sharp_dst[sbase] = target as i16;
                            self.dag_sharp_w[sbase] = to_bf16_sr(1.1, sr_rng);
                        }
                    }
                }
            }

            hist.push_tok(target_u16);
        }
    }

    pub fn train_dag(&mut self, eps: &[BrocaEpisode], epochs: usize) -> f64 {
        let train: Vec<&BrocaEpisode> = eps.iter().filter(|e| e.split != "val").collect();
        let val: Vec<&BrocaEpisode> = eps.iter().filter(|e| e.split == "val").collect();
        if train.is_empty() {
            return 0.0;
        }
        let mut rng = [1u32, 3, 5, 7, 11, 13, 17, 19];
        let mut sr_rng = 0x9e37_79b9_7f4a_7c15u64;
        for epoch in 0..epochs.max(1) {
            let lr = 0.50 / (1.0 + 0.18 * epoch as f32);
            let allow_grow = epoch + 1 < epochs.max(1);
            for (idx, ep) in train.iter().enumerate() {
                if allow_grow && self.dag_alive >= self.dag_cap {
                    self.learn_dag_episode(ep, lr, &mut rng, &mut sr_rng, false);
                } else {
                    self.learn_dag_episode(ep, lr, &mut rng, &mut sr_rng, allow_grow || idx % 2 == 0);
                }
            }
        }
        let eval_set = if val.is_empty() { &train } else { &val };
        let mut bits = 0.0f64;
        let mut toks_tot = 0usize;
        let mut path = Vec::with_capacity(DAG_HOPS + 1);
        for ep in eval_set {
            let root = ep.act.min(BROCA_ACT_CNT - 1);
            let toks = tokenize_broca_target(&ep.target);
            let is_chat = root >= ACT_DLG_BASE;
            let p_scale = if is_chat { 0.25f32 } else { 0.0f32 };
            let h_prompt = if is_chat {
                self.dag_prompt_vec(&ep.prompt)
            } else {
                [0.0f32; DAG_DIM]
            };
            let mut hist = CharHist::new();
            hist.push_tok(BOS_SYM);
            for &t in &toks {
                let logits = self.forward_step(root, &hist, &h_prompt, p_scale, &mut path);
                let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_e: f32 = logits.iter().map(|&v| (v - mx).exp()).sum();
                let lp = (logits[t as usize] - mx) - sum_e.ln();
                bits += (-lp as f64) / std::f64::consts::LN_2;
                toks_tot += 1;
                hist.push_tok(t);
            }
        }
        bits / toks_tot.max(1) as f64
    }

    /// Greedy autoregressive generation expanding `<P0>`, `<P1>`, and `<ANS>` slots inline.
    pub fn generate(
        &self,
        act: usize,
        prompt: &str,
        p0: &str,
        p1: &str,
        ans: &str,
        has_p0: bool,
        has_p1: bool,
        has_ans: bool,
    ) -> String {
        let root = act.min(BROCA_ACT_CNT - 1);
        let is_chat = root >= ACT_DLG_BASE;
        let p_scale = if is_chat { 0.25f32 } else { 0.0f32 };
        let h_prompt = if is_chat {
            self.dag_prompt_vec(prompt)
        } else {
            [0.0f32; DAG_DIM]
        };

        let mut hist = CharHist::new();
        hist.push_tok(BOS_SYM);
        let mut utf8 = Utf8State::default();
        let mut path = Vec::with_capacity(DAG_HOPS + 1);
        let mut bytes: Vec<u8> = Vec::with_capacity(256);
        let mut p0_used = false;
        let mut p1_used = false;
        let mut ans_used = false;

        for _ in 0..220 {
            let mut logits = self.forward_step(root, &hist, &h_prompt, p_scale, &mut path);
            let boundary = utf8.rem == 0;
            let allow_p0 = boundary && has_p0 && !p0_used;
            let allow_p1 = boundary && has_p1 && !p1_used && (!has_p0 || p0_used);
            let allow_ans = boundary && has_ans && !ans_used;
            let slots_done =
                (!has_p0 || p0_used) && (!has_p1 || p1_used) && (!has_ans || ans_used);
            for t in 0..256 {
                if t == 0 || !utf8.allows(t) {
                    logits[t] = f32::NEG_INFINITY;
                }
            }
            if !allow_p0 {
                logits[P0_TOK] = f32::NEG_INFINITY;
            }
            if !allow_p1 {
                logits[P1_TOK] = f32::NEG_INFINITY;
            }
            if !allow_ans {
                logits[ANS_TOK] = f32::NEG_INFINITY;
            }
            if !boundary || !slots_done {
                logits[END_TOK] = f32::NEG_INFINITY;
            }

            let pick = (0..DAG_VOCAB)
                .reduce(|a, b| if logits[b] > logits[a] { b } else { a })
                .unwrap_or(END_TOK);
            if pick == END_TOK {
                break;
            }
            hist.push_tok(pick as u16);
            utf8.feed(pick as u16);
            if pick < 256 {
                bytes.push(pick as u8);
            } else if pick == P0_TOK {
                p0_used = true;
                bytes.extend_from_slice(p0.as_bytes());
            } else if pick == P1_TOK {
                p1_used = true;
                bytes.extend_from_slice(p1.as_bytes());
            } else if pick == ANS_TOK {
                ans_used = true;
                bytes.extend_from_slice(ans.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn emit_short_ans(&self, prompt: &str, ans_bytes: &[u8]) -> String {
        let ans = String::from_utf8_lossy(ans_bytes);
        self.generate(ACT_SHORT_ANS, prompt, "", "", &ans, false, false, true)
    }

    pub fn emit_ack(&self, prompt: &str) -> String {
        self.generate(ACT_ACK, prompt, "", "", "", false, false, false)
    }

    pub fn emit_abstain(&self, prompt: &str) -> String {
        self.generate(ACT_ABSTAIN, prompt, "", "", "", false, false, false)
    }

    pub fn emit_clock(&self, prompt: &str, hour: &str, minute: &str) -> String {
        self.generate(ACT_CLOCK, prompt, hour, minute, "", true, true, false)
    }

    pub fn emit_calc(&self, prompt: &str, value: &str, unit: &str) -> String {
        self.generate(ACT_CALC, prompt, value, unit, "", true, true, false)
    }

    pub fn emit_fact_explain(&self, prompt: &str, head: &str, body: &str) -> String {
        let distilled = self.distill_fact(head, body);
        self.generate(
            ACT_FACT_EXPLAIN,
            prompt,
            head,
            &distilled,
            "",
            true,
            true,
            false,
        )
    }

    pub fn emit_fact_taught(&self, prompt: &str, body_bytes: &[u8]) -> String {
        let ans = String::from_utf8_lossy(body_bytes);
        self.generate(ACT_FACT_TAUGHT, prompt, "", "", &ans, false, false, true)
    }

    pub fn emit_dialogue(&self, prompt: &str) -> String {
        let (intent, _) = self.predict_dialogue_intent(prompt);
        self.generate(
            ACT_DLG_BASE + intent,
            prompt,
            "",
            "",
            "",
            false,
            false,
            false,
        )
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::with_capacity(2 * 1024 * 1024, File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&[u8::from(self.fd_trained)])?;
        w.write_all(bytemuck::cast_slice(&self.fd_w1))?;
        w.write_all(bytemuck::cast_slice(&self.fd_theta))?;
        w.write_all(bytemuck::cast_slice(&self.fd_w2))?;
        w.write_all(bytemuck::cast_slice(&self.fd_w_dir))?;
        w.write_all(bytemuck::cast_slice(&self.fd_bias))?;
        w.write_all(bytemuck::cast_slice(&self.fd_init_log))?;
        w.write_all(bytemuck::cast_slice(&self.fd_trans_log))?;
        w.write_all(bytemuck::cast_slice(&self.dlg_w1))?;
        w.write_all(bytemuck::cast_slice(&self.dlg_theta))?;
        w.write_all(bytemuck::cast_slice(&self.dlg_w2))?;
        w.write_all(bytemuck::cast_slice(&self.dlg_w_dir))?;
        w.write_all(bytemuck::cast_slice(&self.dlg_bias))?;
        w.write_all(&(self.dag_cap as u64).to_le_bytes())?;
        w.write_all(&(self.dag_alive as u64).to_le_bytes())?;
        w.write_all(&(self.dag_edge_buckets as u64).to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(&self.dag_e))?;
        w.write_all(bytemuck::cast_slice(&self.dag_bias))?;
        w.write_all(bytemuck::cast_slice(&self.dag_w_node))?;
        w.write_all(bytemuck::cast_slice(&self.dag_sharp_dst))?;
        w.write_all(bytemuck::cast_slice(&self.dag_sharp_w))?;
        w.write_all(bytemuck::cast_slice(&self.dag_fires))?;
        w.write_all(bytemuck::cast_slice(&self.dag_edge_sigs))?;
        w.write_all(bytemuck::cast_slice(&self.dag_edge_children))?;
        w.write_all(bytemuck::cast_slice(&self.dag_w_prompt))?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(2 * 1024 * 1024, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid BrocaCore checkpoint magic");
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;
        let mut fd_w1 = vec![0.0f32; FD_HIDDEN * FD_DIN];
        let mut fd_theta = [0.0f32; FD_HIDDEN];
        let mut fd_w2 = vec![0.0f32; FD_HIDDEN * FD_TAGS];
        let mut fd_w_dir = vec![0.0f32; FD_BUCKETS * FD_TAGS];
        let mut fd_bias = [0.0f32; FD_TAGS];
        let mut fd_init_log = [0.0f32; FD_TAGS];
        let mut fd_trans_log = [0.0f32; FD_TAGS * FD_TAGS];
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_w2))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_w_dir))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_bias))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_init_log))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut fd_trans_log))?;

        let mut dlg_w1 = vec![0.0f32; DLG_HIDDEN * DLG_DIN];
        let mut dlg_theta = [0.0f32; DLG_HIDDEN];
        let mut dlg_w2 = vec![0.0f32; DLG_HIDDEN * DLG_INTENT_CNT];
        let mut dlg_w_dir = vec![0.0f32; DLG_BUCKETS * DLG_INTENT_CNT];
        let mut dlg_bias = [0.0f32; DLG_INTENT_CNT];
        r.read_exact(bytemuck::cast_slice_mut(&mut dlg_w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dlg_theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dlg_w2))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dlg_w_dir))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dlg_bias))?;

        let read_u64 = |r: &mut BufReader<File>| -> Result<u64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };
        let dag_cap = read_u64(&mut r)? as usize;
        let dag_alive = read_u64(&mut r)? as usize;
        let dag_edge_buckets = read_u64(&mut r)? as usize;
        let mut dag_e = vec![0.0f32; DAG_VOCAB * DAG_DIM];
        let mut dag_bias = vec![0.0f32; BROCA_ACT_CNT * DAG_VOCAB];
        let mut dag_w_node = vec![0u16; dag_cap * DAG_DIM];
        let mut dag_sharp_dst = vec![-1i16; dag_cap * DAG_SHARP_K];
        let mut dag_sharp_w = vec![0u16; dag_cap * DAG_SHARP_K];
        let mut dag_fires = vec![0u32; dag_cap];
        let mut dag_edge_sigs = vec![0u64; dag_edge_buckets * 4];
        let mut dag_edge_children = vec![-1i32; dag_edge_buckets * 4];
        let mut dag_w_prompt = vec![0u16; DAG_PROMPT_BUCKETS * DAG_DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_e))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_bias))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_w_node))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_sharp_dst))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_sharp_w))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_fires))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_edge_sigs))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_edge_children))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut dag_w_prompt))?;
        Ok(Self {
            fd_trained: flag[0] != 0,
            fd_w1,
            fd_theta,
            fd_w2,
            fd_w_dir,
            fd_bias,
            fd_init_log,
            fd_trans_log,
            dlg_w1,
            dlg_theta,
            dlg_w2,
            dlg_w_dir,
            dlg_bias,
            dag_cap,
            dag_alive,
            dag_e,
            dag_bias,
            dag_w_node,
            dag_sharp_dst,
            dag_sharp_w,
            dag_fires,
            dag_edge_buckets,
            dag_edge_sigs,
            dag_edge_children,
            dag_w_prompt,
        })
    }
}

#[derive(Deserialize)]
struct UnseenFactProbe {
    head: String,
    body: String,
}

#[derive(Deserialize)]
struct StoreProbe {
    text: String,
    store: usize,
}

#[derive(Deserialize)]
struct DlgProbe {
    text: String,
    expected_intent: usize,
}

#[derive(Deserialize)]
struct UnseenBrocaProbes {
    metacog_store: Vec<StoreProbe>,
    dialogue: Vec<DlgProbe>,
}

/// Offline pure-Hebbian training & benchmark for MetacogCore (`v25.0`) + BrocaCore (`v25.0`).
pub fn train_and_bench(corpus_dir: &Path, out_json: &Path) -> Result<Value> {
    let t0 = Instant::now();
    let act_rows: Vec<crate::metacog::MetacogActRow> =
        fs::read_to_string(corpus_dir.join("metacog_act.jsonl"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<_>, _>>()?;
    let conf_rows: Vec<crate::metacog::MetacogConfRow> =
        fs::read_to_string(corpus_dir.join("metacog_conf.jsonl"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<_>, _>>()?;
    let fact_rows: Vec<FactBioRow> = fs::read_to_string(corpus_dir.join("fact_bio.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let dlg_rows: Vec<DlgIntentRow> =
        fs::read_to_string(corpus_dir.join("dialogue_intents.jsonl"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<_>, _>>()?;
    let broca_eps: Vec<BrocaEpisode> =
        fs::read_to_string(corpus_dir.join("broca_episodes.jsonl"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut mc = MetacogCore::new();
    let (mc_val_act_acc, mc_val_conf_acc) = mc.train(&act_rows, &conf_rows);
    let mc_path = corpus_dir.join("metacog_core_v25.bin");
    mc.save(&mc_path)?;
    let tl_mc = Instant::now();
    let mc_loaded = MetacogCore::load(&mc_path)?;
    let mc_load_ms = tl_mc.elapsed().as_secs_f64() * 1000.0;

    let mut bc = BrocaCore::new(45_000);
    let fact_val_char_acc = bc.train_fact_distiller(&fact_rows, 4);
    let dlg_val_intent_acc = bc.train_dialogue_intent(&dlg_rows, 10);
    let broca_val_bpt = bc.train_dag(&broca_eps, 5);
    let bc_path = corpus_dir.join("broca_core_v25.bin");
    bc.save(&bc_path)?;
    let tl_bc = Instant::now();
    let bc_loaded = BrocaCore::load(&bc_path)?;
    let bc_load_ms = tl_bc.elapsed().as_secs_f64() * 1000.0;

    let fact_probes: Vec<UnseenFactProbe> =
        serde_json::from_slice(&fs::read(corpus_dir.join("unseen_fact_probes.json"))?)?;
    let mut fact_probe_out = Vec::new();
    for p in &fact_probes {
        let ex = bc_loaded.emit_fact_explain(&p.head, &p.head, &p.body);
        fact_probe_out.push(json!({
            "head": p.head,
            "distilled": bc_loaded.distill_fact(&p.head, &p.body),
            "explain": ex,
        }));
    }

    let broca_probes: UnseenBrocaProbes =
        serde_json::from_slice(&fs::read(corpus_dir.join("unseen_broca_probes.json"))?)?;
    let mc_probe_ok = broca_probes
        .metacog_store
        .iter()
        .filter(|p| mc_loaded.should_store_premise(&p.text) == (p.store > 0))
        .count();
    let mut dlg_probe_out = Vec::new();
    let mut dlg_probe_ok = 0usize;
    for p in &broca_probes.dialogue {
        let (intent, _) = bc_loaded.predict_dialogue_intent(&p.text);
        if intent == p.expected_intent {
            dlg_probe_ok += 1;
        }
        dlg_probe_out.push(json!({
            "text": p.text,
            "expected_intent": p.expected_intent,
            "pred_intent": intent,
            "reply": bc_loaded.emit_dialogue(&p.text),
        }));
    }

    let ack_out = bc_loaded.emit_ack("dummy");
    let abs_out = bc_loaded.emit_abstain("dummy");
    let clk_out = bc_loaded.emit_clock("dummy", "12", "34");
    let calc_u_out = bc_loaded.emit_calc("dummy", "6", "unit");
    let calc_b_out = bc_loaded.emit_calc("dummy", "126", "");
    let short_out = bc_loaded.emit_short_ans("dummy", b"OK");

    let mc_mib = fs::metadata(&mc_path)?.len() as f64 / (1024.0 * 1024.0);
    let bc_mib = fs::metadata(&bc_path)?.len() as f64 / (1024.0 * 1024.0);
    let payload = json!({
        "metacog_val_act_acc": mc_val_act_acc,
        "metacog_val_conf_acc": mc_val_conf_acc,
        "metacog_unseen_store_ok": format!("{mc_probe_ok}/{}", broca_probes.metacog_store.len()),
        "metacog_ckpt_mib": mc_mib,
        "metacog_load_ms": mc_load_ms,
        "fact_val_char_acc": fact_val_char_acc,
        "dlg_val_intent_acc": dlg_val_intent_acc,
        "dlg_unseen_intent_ok": format!("{dlg_probe_ok}/{}", broca_probes.dialogue.len()),
        "broca_val_bpt": broca_val_bpt,
        "broca_alive_nodes": bc_loaded.dag_alive,
        "broca_ckpt_mib": bc_mib,
        "broca_load_ms": bc_load_ms,
        "train_elapsed_s": t0.elapsed().as_secs_f64(),
        "cog_outputs": {
            "short_ans": short_out,
            "ack": ack_out,
            "abstain": abs_out,
            "clock": clk_out,
            "calc_with_unit": calc_u_out,
            "calc_bare": calc_b_out,
        },
        "fact_probes": fact_probe_out,
        "dialogue_probes": dlg_probe_out,
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
    fn broca_core_distill_dag_and_roundtrip() {
        let mut core = BrocaCore::new(2048);
        let fact_rows = vec![FactBioRow {
            head: "Alpha".into(),
            body: "Alpha(a) wa x.".into(),
            tags: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
            split: "train".into(),
        }];
        let acc = core.train_fact_distiller(&fact_rows, 4);
        assert!(acc > 0.5);
        let dlg_rows = vec![
            DlgIntentRow { text: "hello good".into(), intent: 0, split: "train".into() },
            DlgIntentRow { text: "thanks kind".into(), intent: 1, split: "train".into() },
        ];
        let d_acc = core.train_dialogue_intent(&dlg_rows, 6);
        assert!(d_acc > 0.5);
        let eps = vec![
            BrocaEpisode { act: ACT_SHORT_ANS, prompt: "q".into(), target: "<ANS>".into(), split: "train".into() },
            BrocaEpisode { act: ACT_CLOCK, prompt: "t".into(), target: "<P0>:<P1>".into(), split: "train".into() },
            BrocaEpisode { act: ACT_DLG_BASE, prompt: "hello good".into(), target: "hi!".into(), split: "train".into() },
        ];
        let _ = core.train_dag(&eps, 6);
        assert_eq!(core.emit_short_ans("q", b"42"), "42");
        assert_eq!(core.emit_clock("t", "10", "25"), "10:25");
        assert_eq!(core.emit_dialogue("hello good"), "hi!");
        let tmp = std::env::temp_dir().join(format!("ng-bc-{}.bin", std::process::id()));
        core.save(&tmp).unwrap();
        let loaded = BrocaCore::load(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert_eq!(loaded.emit_clock("t", "10", "25"), "10:25");
    }
}
