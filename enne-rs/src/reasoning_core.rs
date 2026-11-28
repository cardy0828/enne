//! ENNE v19.0 推論・命令特化 CPU 言語コア（ネイティブポインター語彙260・UTF-8文字14ホップ・直接ヘブ命令キューバンク）。
//!
//! 設計仕様:
//! - 逆伝播・事前学習済み言語モデル・外部API・ランタイム固定テンプレート語彙を使わず動作する。
//! - 12の推論モード推定（命令→思考）と、キャッシュポインター (<P>, <V>, <ANS>, <END>) 付きの
//!   推論過程生成（推論→結論）を単一の神経新生コアで実行する。
use crate::{
    brain::Brain,
    core_dag::mix,
    reasoner::{ReasonMode, ReasoningPlan},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};

pub const DIM: usize = 256;
pub const BYTE_COUNT: usize = 256;
pub const NEXT_P: usize = 256;
pub const ANS: usize = 257;
pub const NEXT_V: usize = 258;
pub const END: usize = 259;
pub const VOCAB: usize = 260;
const BOS_SYM: u16 = 260;
const HOPS: usize = 14;
const CHAR_HIST: usize = 16;
const SHARP_K: usize = 8;
const CUE_BUCKETS: usize = 65536;
const NUM_MODES: usize = 12;

pub const ALL_MODES: [ReasonMode; NUM_MODES] = [
    ReasonMode::Copy,
    ReasonMode::OrderExtreme,
    ReasonMode::OrderRank,
    ReasonMode::OrderPairwise,
    ReasonMode::Transfer,
    ReasonMode::Elimination,
    ReasonMode::ContradictionCycle,
    ReasonMode::ContradictionConflict,
    ReasonMode::QuantDiff,
    ReasonMode::QuantCompare,
    ReasonMode::Syllogism,
    ReasonMode::Abstain,
];

pub fn mode_idx(m: ReasonMode) -> usize {
    match m {
        ReasonMode::Copy => 0,
        ReasonMode::OrderExtreme => 1,
        ReasonMode::OrderRank => 2,
        ReasonMode::OrderPairwise => 3,
        ReasonMode::Transfer => 4,
        ReasonMode::Elimination => 5,
        ReasonMode::ContradictionCycle => 6,
        ReasonMode::ContradictionConflict => 7,
        ReasonMode::QuantDiff => 8,
        ReasonMode::QuantCompare => 9,
        ReasonMode::Syllogism => 10,
        ReasonMode::Abstain => 11,
    }
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
fn sig64(p: u32, qstate: u8, c: u64) -> u64 {
    mix(
        mix(mix(p as u64 + 17) ^ ((qstate as u64 + 1) * 1000003))
            ^ mix(c.wrapping_add(2000029)),
    )
    .max(1)
}

#[inline(always)]
fn gain(depth: usize) -> f32 {
    1.0 / (1.0 + 0.10 * depth as f32)
}

#[inline(always)]
fn dot256(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 8レーン並列の AVX2 自動ベクトル化対応 BF16 確率丸めヘブ更新。
#[inline(always)]
fn hebb_update_bf16_256(w: &mut [u16], err: &[f32], lr: f32, rng: &mut [u32; 8]) -> f32 {
    let mut align = [0.0f32; 8];
    let mut j = 0;
    while j < DIM {
        for lane in 0..8 {
            let idx = j + lane;
            let wv = f32::from_bits((w[idx] as u32) << 16);
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

/// 動的キュー状態を 5 bit (0..32) に符号化する。
#[inline(always)]
fn pack_qstate(p_idx: usize, p_len: usize, v_idx: usize, v_len: usize, ans_rem: bool) -> u8 {
    let p_more = u8::from(p_idx < p_len);
    let p_started = u8::from(p_idx > 0);
    let v_more = u8::from(v_idx < v_len);
    let v_pending = u8::from(v_idx < v_len && p_idx > v_idx);
    let ans = u8::from(ans_rem);
    p_more | (p_started << 1) | (v_more << 2) | (v_pending << 3) | (ans << 4)
}

#[derive(Deserialize)]
pub struct Episode {
    pub mode: ReasonMode,
    pub prompt: String,
    pub n_p: usize,
    pub n_v: usize,
    pub reasoning: String,
    #[serde(default)]
    pub split: String,
}

/// `<P>`, `<V>`, `<ANS>` をネイティブ制御シンボル (256..259) に畳み、他は UTF-8 バイトへ展開する。
pub fn tokenize_reasoning(mut s: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(s.len() + 1);
    while !s.is_empty() {
        if s.starts_with("<P>") {
            out.push(NEXT_P as u16);
            s = &s[3..];
        } else if s.starts_with("<V>") {
            out.push(NEXT_V as u16);
            s = &s[3..];
        } else if s.starts_with("<ANS>") {
            out.push(ANS as u16);
            s = &s[5..];
        } else {
            let c = s.chars().next().unwrap();
            for &b in &s.as_bytes()[..c.len_utf8()] {
                out.push(b as u16);
            }
            s = &s[c.len_utf8()..];
        }
    }
    out.push(END as u16);
    out
}

/// 命令文から自己組織化文字2-gram・3-gramおよび構造特徴をハッシュキュー列へ抽出する（ハードコード語彙ゼロ）。
pub fn extract_prompt_cues(prompt: &str) -> Vec<usize> {
    let chars: Vec<char> = prompt.chars().collect();
    let mut cues = Vec::with_capacity(chars.len() * 2 + 8);
    // Single characters (captures particles, punctuation, symbols like ＝, ？, m, %)
    for &c in &chars {
        if c > ' ' {
            let h = mix((c as u64).wrapping_mul(0x9e37_79b9) ^ 0x1111) as usize;
            cues.push(h & (CUE_BUCKETS - 1));
        }
    }
    // Character bigrams (captures 標高, 比較, 矛盾, 以外, 一番, 順番, 同じ, etc.)
    for w in chars.windows(2) {
        if w[0] > ' ' && w[1] > ' ' {
            let k = ((w[0] as u64) << 21) | (w[1] as u64);
            cues.push((mix(k ^ 0x2222) as usize) & (CUE_BUCKETS - 1));
        }
    }
    // Character trigrams (captures より高, どちら, すべて, ではない, etc.)
    for w in chars.windows(3) {
        if w[0] > ' ' && w[2] > ' ' {
            let k = ((w[0] as u64) << 42) ^ ((w[1] as u64) << 21) ^ (w[2] as u64);
            cues.push((mix(k ^ 0x3333) as usize) & (CUE_BUCKETS - 1));
        }
    }
    // Structural signals: presence of digits, question mark, multi-clause count
    let has_digit = chars.iter().any(|c| c.is_ascii_digit()) as u64;
    let clause_bucket = (prompt.matches('。').count().min(6)) as u64;
    let has_eq = prompt.contains('＝') as u64;
    let has_newline = prompt.contains('\n') as u64;
    for (tag, val) in [
        (0x9001u64, has_digit),
        (0x9002, clause_bucket),
        (0x9003, has_eq),
        (0x9004, has_newline),
        (0x9005, (has_digit << 4) | clause_bucket),
    ] {
        cues.push((mix(tag ^ (val * 1000003)) as usize) & (CUE_BUCKETS - 1));
    }
    cues.sort_unstable();
    cues.dedup();
    cues
}

#[derive(Clone)]
struct CharRing {
    hist: [u64; CHAR_HIST],
    len: usize,
    buf: [u8; 4],
    blen: u8,
    bexp: u8,
    /// 文頭の最初の1文字または制御記号（長距離整合のため）
    opening: u64,
}

impl Default for CharRing {
    fn default() -> Self {
        Self {
            hist: [0; CHAR_HIST],
            len: 0,
            buf: [0; 4],
            blen: 0,
            bexp: 0,
            opening: 0,
        }
    }
}

impl CharRing {
    #[inline]
    fn push_sym(&mut self, sym: u16) {
        if sym >= BYTE_COUNT as u16 {
            self.blen = 0;
            self.bexp = 0;
            let code = 0x200000 + sym as u64;
            if self.opening == 0 && sym != BOS_SYM {
                self.opening = code;
            }
            self.hist[self.len % CHAR_HIST] = code;
            self.len += 1;
            return;
        }
        let b = sym as u8;
        if b < 0x80 {
            self.blen = 0;
            self.bexp = 0;
            let code = b as u64;
            if self.opening == 0 {
                self.opening = code;
            }
            self.hist[self.len % CHAR_HIST] = code;
            self.len += 1;
        } else if b >= 0xc0 {
            self.buf[0] = b;
            self.blen = 1;
            self.bexp = if b < 0xe0 {
                2
            } else if b < 0xf0 {
                3
            } else {
                4
            };
        } else if self.blen > 0 && self.blen < self.bexp {
            self.buf[self.blen as usize] = b;
            self.blen += 1;
            if self.blen == self.bexp {
                if let Ok(s) = std::str::from_utf8(&self.buf[..self.blen as usize]) {
                    if let Some(ch) = s.chars().next() {
                        let code = ch as u64;
                        if self.opening == 0 {
                            self.opening = code;
                        }
                        self.hist[self.len % CHAR_HIST] = code;
                        self.len += 1;
                    }
                }
                self.blen = 0;
                self.bexp = 0;
            }
        } else {
            self.blen = 0;
            self.bexp = 0;
        }
    }

    #[inline]
    fn prev(&self, back: usize) -> Option<u64> {
        let skip = usize::from(self.blen == 0);
        let off = back + skip;
        if off == 0 || off > CHAR_HIST || off > self.len {
            return None;
        }
        let v = self.hist[(self.len - off) % CHAR_HIST];
        (v != 0).then_some(v)
    }
}

/// 命令理解（意図・モード予測）とネイティブポインター推論生成を統合した CPU 神経新生コア。
pub struct ReasoningCore {
    pub cap: usize,
    pub buckets: usize,
    pub len: usize,
    pub alive_count: usize,
    // Sensory roots: (VOCAB + 1) symbols * 32 queue states = 8,352 roots
    n_roots: usize,
    w: Vec<u16>,
    sharp_dst: Vec<i16>,
    sharp_w: Vec<u16>,
    parent: Vec<u32>,
    span: Vec<u8>,
    depth: Vec<u8>,
    alive: Vec<u8>,
    updates: Vec<u32>,
    pain: Vec<f32>,
    sigs: Vec<u64>,
    children: Vec<u32>,
    // Shared output readout basis [VOCAB, DIM] and bias [VOCAB]
    basis: Vec<f32>,
    bias: [f32; VOCAB],
    // Direct Hebbian Instruction-Cue Readout Bank (Pathway 2)
    cue_w: Vec<u16>,         // [CUE_BUCKETS, DIM] BF16
    cue_updates: Vec<u32>,   // [CUE_BUCKETS]
    mode_emb: Vec<f32>,      // [NUM_MODES, DIM] f32
    // Direct Hebbian Mode Classifier (Stage 1: raw prompt -> 12 ReasonModes)
    mode_w: Vec<f32>,        // [CUE_BUCKETS, NUM_MODES]
    mode_bias: [f32; NUM_MODES],
    rng: u64,
}

#[derive(Clone, Copy)]
pub struct GenConfig {
    pub use_cue_bank: bool,
    pub use_cache: bool,
    pub short: bool,
    pub temperature: f32,
    pub seed: u64,
}

#[derive(Serialize)]
pub struct GenOutput {
    pub text: String,
    pub predicted_mode: ReasonMode,
    pub pointer_ranges: Vec<(usize, usize)>,
    pub pointer_bytes: usize,
    pub forced_budget_gates: usize,
    pub stopped: bool,
}

impl ReasoningCore {
    /// 推論コーパスから基底と構造を初期化する（隠れ細胞ゼロ開始）。
    pub fn new(cap: usize) -> Self {
        let n_roots = (VOCAB + 1) * 32;
        let buckets = ((cap / 2).max(262_144)).next_power_of_two();
        // Orthogonal-ish deterministic initial basis for VOCAB=260, DIM=256
        let mut basis = vec![0.0f32; VOCAB * DIM];
        for i in 1..VOCAB {
            let mut norm2 = 0.0f32;
            for j in 0..DIM {
                let h = mix(((i as u64) << 16) ^ (j as u64 + 1));
                let v = ((h & 0xffff) as f32 / 32768.0) - 1.0;
                basis[i * DIM + j] = v;
                norm2 += v * v;
            }
            let scale = 1.25 / norm2.sqrt().max(1e-5);
            for j in 0..DIM {
                basis[i * DIM + j] *= scale;
            }
        }
        let mut g = Self {
            cap: cap.max(n_roots + 1024),
            buckets,
            len: n_roots,
            alive_count: 0,
            n_roots,
            w: vec![0u16; n_roots * DIM],
            sharp_dst: vec![-1i16; n_roots * SHARP_K],
            sharp_w: vec![0u16; n_roots * SHARP_K],
            parent: vec![0u32; n_roots],
            span: vec![0u8; n_roots],
            depth: vec![0u8; n_roots],
            alive: vec![1u8; n_roots],
            updates: vec![0u32; n_roots],
            pain: vec![0.0f32; n_roots],
            sigs: vec![0u64; buckets * 4],
            children: vec![0u32; buckets * 4],
            basis,
            bias: [0.0; VOCAB],
            cue_w: vec![0u16; CUE_BUCKETS * DIM],
            cue_updates: vec![0u32; CUE_BUCKETS],
            mode_emb: vec![0.0f32; NUM_MODES * DIM],
            mode_w: vec![0.0f32; CUE_BUCKETS * NUM_MODES],
            mode_bias: [0.0; NUM_MODES],
            rng: 0x9e37_79b9_7f4a_7c15,
        };
        let pre = cap.min(1_000_000);
        if pre > n_roots {
            let add = pre - n_roots;
            g.w.reserve(add * DIM);
            g.sharp_dst.reserve(add * SHARP_K);
            g.sharp_w.reserve(add * SHARP_K);
            g.parent.reserve(add);
            g.span.reserve(add);
            g.depth.reserve(add);
            g.alive.reserve(add);
            g.updates.reserve(add);
            g.pain.reserve(add);
        }
        g
    }

    #[inline(always)]
    fn root_id(&self, sym: u16, qstate: u8) -> u32 {
        (sym as usize * 32 + (qstate as usize & 31)) as u32
    }

    #[inline]
    fn alloc_node(&mut self, parent: u32, span: u8, depth: u8) -> Option<u32> {
        if self.len >= self.cap {
            return None;
        }
        let id = self.len as u32;
        self.len += 1;
        self.w.resize(self.len * DIM, 0);
        self.sharp_dst.resize(self.len * SHARP_K, -1);
        self.sharp_w.resize(self.len * SHARP_K, 0);
        self.parent.push(parent);
        self.span.push(span);
        self.depth.push(depth);
        self.alive.push(1);
        self.updates.push(0);
        self.pain.push(0.0);
        self.alive_count += 1;
        Some(id)
    }

    /// Stage 1: 生の日本語命令文から推論モード（12種）を直接ヘブ分類器で予測する（正規表現ゼロ）。
    pub fn predict_mode(&self, prompt: &str) -> (ReasonMode, [f32; NUM_MODES]) {
        let cues = extract_prompt_cues(prompt);
        let mut logits = self.mode_bias;
        let inv = 1.0 / (cues.len().max(1) as f32).sqrt();
        for &c in &cues {
            let row = &self.mode_w[c * NUM_MODES..(c + 1) * NUM_MODES];
            for m in 0..NUM_MODES {
                logits[m] += row[m] * inv;
            }
        }
        let best = (0..NUM_MODES)
            .reduce(|a, b| if logits[b] > logits[a] { b } else { a })
            .unwrap();
        (ALL_MODES[best], logits)
    }

    /// Stage 1 局所デルタ・ヘブ更新（逆伝播なし）。
    fn train_mode_step(&mut self, cues: &[usize], gold_mode: ReasonMode, lr: f32) -> bool {
        let gold = mode_idx(gold_mode);
        let mut logits = self.mode_bias;
        let inv = 1.0 / (cues.len().max(1) as f32).sqrt();
        for &c in cues {
            let row = &self.mode_w[c * NUM_MODES..(c + 1) * NUM_MODES];
            for m in 0..NUM_MODES {
                logits[m] += row[m] * inv;
            }
        }
        let best = (0..NUM_MODES)
            .reduce(|a, b| if logits[b] > logits[a] { b } else { a })
            .unwrap();
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = [0.0f32; NUM_MODES];
        let mut sum = 0.0f32;
        for m in 0..NUM_MODES {
            probs[m] = (logits[m] - top).exp();
            sum += probs[m];
        }
        for m in 0..NUM_MODES {
            probs[m] /= sum;
        }
        let mut diff = [0.0f32; NUM_MODES];
        for m in 0..NUM_MODES {
            diff[m] = if m == gold { 1.0 } else { 0.0 } - probs[m];
            self.mode_bias[m] += lr * 0.2 * diff[m];
        }
        let step_lr = lr * inv;
        for &c in cues {
            let row = &mut self.mode_w[c * NUM_MODES..(c + 1) * NUM_MODES];
            for m in 0..NUM_MODES {
                row[m] += step_lr * diff[m];
            }
        }
        best == gold
    }

    #[inline]
    fn cue_vec(&self, cues: &[usize], mode: ReasonMode, use_cue_bank: bool) -> [f32; DIM] {
        let mut out = [0.0f32; DIM];
        let mrow = &self.mode_emb[mode_idx(mode) * DIM..(mode_idx(mode) + 1) * DIM];
        for j in 0..DIM {
            out[j] += mrow[j] * 0.70;
        }
        if use_cue_bank && !cues.is_empty() {
            let inv = 0.55 / (cues.len() as f32).sqrt();
            for &c in cues {
                let row = &self.cue_w[c * DIM..(c + 1) * DIM];
                for j in 0..DIM {
                    out[j] += from_bf16(row[j]) * inv;
                }
            }
        }
        out
    }

    #[inline]
    fn local_cond(hist_syms: &[u16], span: usize, ring: &CharRing) -> Option<u64> {
        if span < 2 {
            hist_syms
                .len()
                .checked_sub(span + 2)
                .map(|i| hist_syms[i] as u64 + 1)
        } else {
            ring.prev(span - 1).map(|cp| 0x400 + cp)
        }
    }

    /// 1ステップの推論DAG探索＋命令キュー・モード結合。
    #[inline]
    fn forward_step(
        &self,
        hist_syms: &[u16],
        ring: &CharRing,
        qstate: u8,
        mode: ReasonMode,
        cue_h: &[f32; DIM],
        vis_out: &mut Vec<(u8, u32)>,
        h_out: &mut [f32; DIM],
        logits_out: &mut [f32; VOCAB],
    ) -> u32 {
        let last_sym = *hist_syms.last().unwrap_or(&BOS_SYM);
        let root = self.root_id(last_sym, qstate);
        vis_out.clear();
        vis_out.push((0, root));
        let mut cur = root;
        let mut leaf = root;
        let mask = self.buckets - 1;
        // Branch 1: exact sequential char-hop path (depth 1..HOPS)
        for hop in 1..=HOPS {
            let Some(cond) = Self::local_cond(hist_syms, self.span[cur as usize] as usize, ring)
            else {
                break;
            };
            // Include mode tag in early hops so each reasoning mode owns crisp discourse paths
            let c_eff = if hop == 1 {
                cond ^ ((mode_idx(mode) as u64 + 1) << 40)
            } else if hop == 3 && ring.opening != 0 {
                cond ^ (ring.opening << 24)
            } else {
                cond
            };
            let s = sig64(cur, qstate, c_eff);
            let base = ((s as usize) & mask) * 4;
            let row = &self.sigs[base..base + 4];
            let mut found = None;
            for m in 0..4 {
                if row[m] == s {
                    found = Some(self.children[base + m]);
                    break;
                }
            }
            if let Some(id) = found {
                vis_out.push((hop as u8, id));
                cur = id;
                leaf = id;
            } else {
                break;
            }
        }

        // Accumulate visited DAG weights + direct Hebbian instruction cue readout
        *h_out = *cue_h;
        *logits_out = self.bias;
        for &(d, nid) in vis_out.iter() {
            let g = gain(d as usize);
            let wslice = &self.w[nid as usize * DIM..(nid as usize + 1) * DIM];
            for j in 0..DIM {
                h_out[j] += from_bf16(wslice[j]) * g;
            }
            let sbase = nid as usize * SHARP_K;
            let dst = &self.sharp_dst[sbase..sbase + SHARP_K];
            let sw = &self.sharp_w[sbase..sbase + SHARP_K];
            let gs = g * 0.75;
            for j in 0..SHARP_K {
                let dd = dst[j];
                if dd >= 0 {
                    logits_out[dd as usize] += from_bf16(sw[j]) * gs;
                }
            }
        }
        for v in h_out.iter_mut() {
            *v = (*v * 0.5).tanh() * 2.0;
        }
        // Output projection: logits += h @ basis^T
        for i in 0..VOCAB {
            let row: &[f32; DIM] = self.basis[i * DIM..(i + 1) * DIM].try_into().unwrap();
            logits_out[i] += dot256(h_out, row);
        }
        leaf
    }

    /// 1エピソードの学習（Stage 1 モード分類 ＋ Stage 2 推論生成のヘブ更新＆有糸分裂）。
    pub fn train_episode(&mut self, ep: &Episode, use_cue_bank: bool, epoch_lr: f32) -> (bool, f64, usize) {
        let cues = extract_prompt_cues(&ep.prompt);
        let mode_ok = self.train_mode_step(&cues, ep.mode, 0.08 * epoch_lr);
        let syms = tokenize_reasoning(&ep.reasoning);
        let mut hist_syms: Vec<u16> = Vec::with_capacity(syms.len() + 1);
        hist_syms.push(BOS_SYM);
        let mut ring = CharRing::default();
        ring.push_sym(BOS_SYM);
        let (mut p_idx, mut v_idx, mut ans_used) = (0usize, 0usize, false);

        let mut vis = Vec::with_capacity(HOPS + 2);
        let mut h = [0.0f32; DIM];
        let mut logits = [0.0f32; VOCAB];
        let mut nll = 0.0f64;
        let mut rng8 = std::array::from_fn(|l| {
            let z = mix(self.rng ^ (l as u64 + 1));
            (z as u32).max(1)
        });
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);

        let mut cue_err_acc = [0.0f32; DIM];
        let mut mode_err_acc = [0.0f32; DIM];
        let cue_h = self.cue_vec(&cues, ep.mode, use_cue_bank);

        for &target_u16 in &syms {
            let target = target_u16 as usize;
            let qstate = pack_qstate(p_idx, ep.n_p, v_idx, ep.n_v, !ans_used);
            let leaf = self.forward_step(
                &hist_syms,
                &ring,
                qstate,
                ep.mode,
                &cue_h,
                &mut vis,
                &mut h,
                &mut logits,
            );

            // Softmax + NLL
            let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut prob = [0.0f32; VOCAB];
            let mut sum = 0.0f32;
            for i in 0..VOCAB {
                prob[i] = (logits[i] - top).exp();
                sum += prob[i];
            }
            let inv_sum = 1.0 / sum;
            let mut bad = usize::from(target == 0);
            let mut bad_p = -1.0f32;
            for i in 0..VOCAB {
                prob[i] *= inv_sum;
                if i != target && prob[i] > bad_p {
                    bad_p = prob[i];
                    bad = i;
                }
            }
            let pt = prob[target];
            let surp = 1.0 - pt;
            nll += -(pt.max(1e-30) as f64).log2();

            // Local Hebbian error projection: err = basis[target] - sum_j prob[j] * basis[j]
            let mut err = [0.0f32; DIM];
            let trow = &self.basis[target * DIM..(target + 1) * DIM];
            err.copy_from_slice(trow);
            for j in 0..VOCAB {
                let pj = prob[j];
                if pj > 1e-5 {
                    let brow = &self.basis[j * DIM..(j + 1) * DIM];
                    for k in 0..DIM {
                        err[k] -= pj * brow[k];
                    }
                }
            }
            for k in 0..DIM {
                let hv = h[k] * 0.5;
                err[k] *= (1.0 - hv * hv).max(0.15);
                cue_err_acc[k] += err[k];
                mode_err_acc[k] += err[k];
            }

            // Update visited DAG nodes
            for &(d, nid) in &vis {
                let idx = nid as usize;
                let g = gain(d as usize);
                self.updates[idx] = self.updates[idx].saturating_add(1);
                let up = self.updates[idx];
                let lr = 0.24 * epoch_lr * g / (1.0 + 0.02 * up as f32).sqrt();
                let wslice = &mut self.w[idx * DIM..(idx + 1) * DIM];
                let _ = hebb_update_bf16_256(wslice, &err, lr, &mut rng8);

                let slr = 0.36 * epoch_lr * g / (1.0 + 0.004 * up as f32).sqrt();
                let sbase = idx * SHARP_K;
                let dst = &mut self.sharp_dst[sbase..sbase + SHARP_K];
                let sw = &mut self.sharp_w[sbase..sbase + SHARP_K];
                let t16 = target as i16;
                if let Some(pos) = dst.iter().position(|&v| v == t16) {
                    let nv = (from_bf16(sw[pos]) + slr * surp).clamp(-4.0, 5.0);
                    sw[pos] = to_bf16_sr(nv, &mut self.rng);
                } else if surp > 0.22 {
                    let pos = dst.iter().position(|&v| v < 0).unwrap_or_else(|| {
                        (0..SHARP_K)
                            .min_by(|&a, &c| {
                                from_bf16(sw[a])
                                    .abs()
                                    .total_cmp(&from_bf16(sw[c]).abs())
                            })
                            .unwrap()
                    });
                    dst[pos] = t16;
                    sw[pos] = to_bf16_sr((slr * surp).clamp(0.08, 1.8), &mut self.rng);
                }
                if bad_p > 0.06 {
                    let b16 = bad as i16;
                    if let Some(pos) = dst.iter().position(|&v| v == b16) {
                        let nv = (from_bf16(sw[pos]) - slr * 0.85 * bad_p).clamp(-4.0, 5.0);
                        sw[pos] = to_bf16_sr(nv, &mut self.rng);
                    }
                }
            }

            // Light readout bias & basis update
            let brate = 0.004 * epoch_lr;
            for j in 0..VOCAB {
                let d = if j == target { 1.0 } else { 0.0 } - prob[j];
                self.bias[j] += brate * 0.4 * d;
                if d.abs() > 0.005 {
                    let brow = &mut self.basis[j * DIM..(j + 1) * DIM];
                    let scale = brate * d;
                    for k in 0..DIM {
                        brow[k] += scale * h[k];
                    }
                }
            }

            // Mitosis on leaf when surprised
            let li = leaf as usize;
            if surp > 0.38 {
                self.pain[li] += surp;
                if self.pain[li] >= 1.6 && (self.depth[li] as usize) < HOPS {
                    if let Some(cond) =
                        Self::local_cond(&hist_syms, self.span[li] as usize, &ring)
                    {
                        let next_hop = self.depth[li] as usize + 1;
                        let c_eff = if next_hop == 1 {
                            cond ^ ((mode_idx(ep.mode) as u64 + 1) << 40)
                        } else if next_hop == 3 && ring.opening != 0 {
                            cond ^ (ring.opening << 24)
                        } else {
                            cond
                        };
                        let s = sig64(leaf, qstate, c_eff);
                        let base = ((s as usize) & (self.buckets - 1)) * 4;
                        if !self.sigs[base..base + 4].contains(&s) {
                            if let Some(off) = self.sigs[base..base + 4].iter().position(|&e| e == 0)
                            {
                                let nspan = self.span[li].saturating_add(1);
                                let ndepth = self.depth[li].saturating_add(1);
                                if let Some(id) = self.alloc_node(leaf, nspan, ndepth) {
                                    self.sigs[base + off] = s;
                                    self.children[base + off] = id;
                                    self.pain[li] *= 0.35;
                                }
                            }
                        }
                    }
                }
            }

            // Advance episode state
            hist_syms.push(target_u16);
            ring.push_sym(target_u16);
            if target == NEXT_P {
                p_idx += 1;
            } else if target == NEXT_V {
                v_idx += 1;
            } else if target == ANS {
                ans_used = true;
            }
        }

        // Episode-level Hebbian update on Direct Instruction-Cue Readout Bank & Mode Embedding
        let inv_steps = 1.0 / (syms.len().max(1) as f32);
        for k in 0..DIM {
            cue_err_acc[k] *= inv_steps;
            mode_err_acc[k] *= inv_steps;
        }
        let mrow = &mut self.mode_emb[mode_idx(ep.mode) * DIM..(mode_idx(ep.mode) + 1) * DIM];
        for k in 0..DIM {
            mrow[k] = (mrow[k] + 0.08 * epoch_lr * mode_err_acc[k]).clamp(-3.0, 3.0);
        }
        if use_cue_bank && !cues.is_empty() {
            let inv_c = 1.0 / (cues.len() as f32).sqrt();
            for &c in &cues {
                self.cue_updates[c] = self.cue_updates[c].saturating_add(1);
                let clr = 0.14 * epoch_lr * inv_c / (1.0 + 0.01 * self.cue_updates[c] as f32).sqrt();
                let crow = &mut self.cue_w[c * DIM..(c + 1) * DIM];
                let _ = hebb_update_bf16_256(crow, &cue_err_acc, clr, &mut rng8);
            }
        }

        (mode_ok, nll, syms.len())
    }

    /// ホールドアウト評価（モード分類精度 + 推論記号BPC）。
    pub fn eval_episodes(&self, eps: &[Episode], use_cue_bank: bool) -> (f64, f64) {
        let (mode_hits, nll, steps): (usize, f64, usize) = eps
            .par_iter()
            .map(|ep| {
                let (pred_m, _) = self.predict_mode(&ep.prompt);
                let m_ok = usize::from(pred_m == ep.mode);
                let cues = extract_prompt_cues(&ep.prompt);
                let cue_h = self.cue_vec(&cues, ep.mode, use_cue_bank);
                let syms = tokenize_reasoning(&ep.reasoning);
                let mut hist_syms = Vec::with_capacity(syms.len() + 1);
                hist_syms.push(BOS_SYM);
                let mut ring = CharRing::default();
                ring.push_sym(BOS_SYM);
                let (mut p_idx, mut v_idx, mut ans_used) = (0usize, 0usize, false);
                let mut vis = Vec::with_capacity(HOPS + 2);
                let mut h = [0.0f32; DIM];
                let mut logits = [0.0f32; VOCAB];
                let mut ep_nll = 0.0f64;
                for &t16 in &syms {
                    let target = t16 as usize;
                    let qstate = pack_qstate(p_idx, ep.n_p, v_idx, ep.n_v, !ans_used);
                    let _ = self.forward_step(
                        &hist_syms,
                        &ring,
                        qstate,
                        ep.mode,
                        &cue_h,
                        &mut vis,
                        &mut h,
                        &mut logits,
                    );
                    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f64 = logits.iter().map(|&x| ((x - top) as f64).exp()).sum();
                    ep_nll += ((top - logits[target]) as f64 + sum.ln()) / std::f64::consts::LN_2;
                    hist_syms.push(t16);
                    ring.push_sym(t16);
                    if target == NEXT_P {
                        p_idx += 1;
                    } else if target == NEXT_V {
                        v_idx += 1;
                    } else if target == ANS {
                        ans_used = true;
                    }
                }
                (m_ok, ep_nll, syms.len())
            })
            .reduce(
                || (0, 0.0, 0),
                |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3),
            );
        (
            mode_hits as f64 / eps.len().max(1) as f64,
            nll / steps.max(1) as f64,
        )
    }

    /// エンドツーエンド生成: 生の日本語命令 → (1)推論モード推定 → (2)キャッシュポインター付き推論言語生成。
    pub fn generate(&self, prompt: &str, plan: &ReasoningPlan, cfg: GenConfig) -> GenOutput {
        let (predicted_mode, _) = self.predict_mode(prompt);
        let cues = extract_prompt_cues(prompt);
        let active_mode = plan.mode;
        let cue_h = self.cue_vec(&cues, active_mode, cfg.use_cue_bank);

        let path_items: Vec<&[u8]> = plan
            .path_nodes
            .iter()
            .map(Vec::as_slice)
            .filter(|s| !s.is_empty())
            .collect();
        let val_items: Vec<&[u8]> = plan
            .auxiliary_values
            .iter()
            .map(Vec::as_slice)
            .filter(|s| !s.is_empty())
            .collect();
        let ans_bytes: &[u8] = &plan.answer_bytes;
        let has_ans = !ans_bytes.is_empty();

        if cfg.short && cfg.use_cache {
            let text = String::from_utf8_lossy(ans_bytes).into_owned();
            let len = text.len();
            return GenOutput {
                text,
                predicted_mode,
                pointer_ranges: if len > 0 { vec![(0, len)] } else { vec![] },
                pointer_bytes: len,
                forced_budget_gates: 0,
                stopped: true,
            };
        }

        let (mut p_idx, mut v_idx, mut ans_used) = (0usize, 0usize, false);
        let mut bytes: Vec<u8> = Vec::with_capacity(256);
        let mut hist_syms: Vec<u16> = Vec::with_capacity(256);
        hist_syms.push(BOS_SYM);
        let mut ring = CharRing::default();
        ring.push_sym(BOS_SYM);
        let mut pointer_ranges = Vec::new();
        let mut pointer_bytes = 0usize;
        let mut forced_budget_gates = 0usize;
        let mut stopped = false;
        let mut seed = cfg.seed;

        let total_payload: usize = path_items.iter().map(|s| s.len()).sum::<usize>()
            + val_items.iter().map(|s| s.len()).sum::<usize>()
            + ans_bytes.len();
        let limit = if cfg.use_cache {
            380.max(total_payload + 160)
        } else {
            320
        };

        let mut vis = Vec::with_capacity(HOPS + 2);
        let mut h = [0.0f32; DIM];
        let mut logits = [0.0f32; VOCAB];

        while bytes.len() < limit {
            let qstate = pack_qstate(
                p_idx,
                path_items.len(),
                v_idx,
                val_items.len(),
                has_ans && !ans_used,
            );
            let _ = self.forward_step(
                &hist_syms,
                &ring,
                qstate,
                active_mode,
                &cue_h,
                &mut vis,
                &mut h,
                &mut logits,
            );

            let boundary = std::str::from_utf8(&bytes).is_ok();
            let reserve: usize = if cfg.use_cache {
                path_items[p_idx..].iter().map(|s| s.len()).sum::<usize>()
                    + val_items[v_idx..].iter().map(|s| s.len()).sum::<usize>()
                    + if has_ans && !ans_used {
                        ans_bytes.len()
                    } else {
                        0
                    }
            } else {
                0
            };
            let available = limit.saturating_sub(bytes.len() + reserve);

            for s in 0..BYTE_COUNT {
                if s == 0 || available == 0 || !legal_utf8(&bytes, s as u8, available) {
                    logits[s] = f32::NEG_INFINITY;
                }
            }
            let allow_p = boundary
                && p_idx < path_items.len()
                && (!cfg.use_cache || path_items[p_idx].len() <= limit - bytes.len())
                && !(v_idx < val_items.len() && p_idx > v_idx);
            let allow_v = boundary
                && v_idx < val_items.len()
                && (!cfg.use_cache || val_items[v_idx].len() <= limit - bytes.len())
                && p_idx > v_idx;
            let allow_ans = boundary
                && has_ans
                && !ans_used
                && (!cfg.use_cache || ans_bytes.len() <= limit - bytes.len());
            let all_emitted =
                p_idx == path_items.len() && v_idx == val_items.len() && (!has_ans || ans_used);
            if !allow_p {
                logits[NEXT_P] = f32::NEG_INFINITY;
            }
            if !allow_v {
                logits[NEXT_V] = f32::NEG_INFINITY;
            }
            if !allow_ans {
                logits[ANS] = f32::NEG_INFINITY;
            }
            if !boundary || !all_emitted {
                logits[END] = f32::NEG_INFINITY;
            }

            let pick = choose(&logits, cfg.temperature, &mut seed);
            hist_syms.push(pick as u16);
            ring.push_sym(pick as u16);

            if pick == END {
                stopped = true;
                break;
            }
            if pick < BYTE_COUNT {
                bytes.push(pick as u8);
            } else {
                let span: &[u8] = if pick == NEXT_P {
                    let item = path_items[p_idx];
                    p_idx += 1;
                    item
                } else if pick == NEXT_V {
                    let item = val_items[v_idx];
                    v_idx += 1;
                    item
                } else {
                    ans_used = true;
                    ans_bytes
                };
                if !cfg.use_cache {
                    continue;
                }
                if available == 0 {
                    forced_budget_gates += 1;
                }
                pointer_ranges.push((bytes.len(), bytes.len() + span.len()));
                bytes.extend_from_slice(span);
                pointer_bytes += span.len();
            }
        }

        GenOutput {
            text: String::from_utf8(bytes).expect("UTF-8"),
            predicted_mode,
            pointer_ranges,
            pointer_bytes,
            forced_budget_gates,
            stopped,
        }
    }

    /// 推論言語コアのバイナリチェックポイント（NGRC1901）を保存する。
    pub fn save(&self, path: &Path) -> Result<u64> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::with_capacity(2 * 1024 * 1024, File::create(path)?);
        w.write_all(b"NGRC1901")?;
        w.write_all(&(self.cap as u32).to_le_bytes())?;
        w.write_all(&(self.buckets as u32).to_le_bytes())?;
        w.write_all(&(self.len as u32).to_le_bytes())?;
        w.write_all(&(self.alive_count as u32).to_le_bytes())?;
        w.write_all(&(self.n_roots as u32).to_le_bytes())?;
        w.write_all(&self.rng.to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(&self.basis))?;
        w.write_all(bytemuck::cast_slice(&self.bias))?;
        w.write_all(&self.alive)?;
        w.write_all(&self.span)?;
        w.write_all(&self.depth)?;
        w.write_all(bytemuck::cast_slice(&self.parent))?;
        w.write_all(bytemuck::cast_slice(&self.updates))?;
        w.write_all(bytemuck::cast_slice(&self.w))?;
        w.write_all(bytemuck::cast_slice(&self.sharp_dst))?;
        w.write_all(bytemuck::cast_slice(&self.sharp_w))?;
        let edge_count: u64 = self.sigs.iter().filter(|&&s| s != 0).count() as u64;
        w.write_all(&edge_count.to_le_bytes())?;
        for m in 0..self.sigs.len() {
            if self.sigs[m] != 0 {
                w.write_all(&self.sigs[m].to_le_bytes())?;
                w.write_all(&self.children[m].to_le_bytes())?;
            }
        }
        w.write_all(bytemuck::cast_slice(&self.cue_w))?;
        w.write_all(bytemuck::cast_slice(&self.cue_updates))?;
        w.write_all(bytemuck::cast_slice(&self.mode_emb))?;
        w.write_all(bytemuck::cast_slice(&self.mode_w))?;
        w.write_all(bytemuck::cast_slice(&self.mode_bias))?;
        w.flush()?;
        Ok(fs::metadata(path)?.len())
    }

    /// 保存済みチェックポイント（NGRC1901）を高速ロードする。
    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::with_capacity(2 * 1024 * 1024, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == b"NGRC1901", "invalid ReasoningCore magic");
        let read_u32 = |r: &mut BufReader<File>| -> Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let read_u64 = |r: &mut BufReader<File>| -> Result<u64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };
        let cap = read_u32(&mut r)? as usize;
        let buckets = read_u32(&mut r)? as usize;
        let len = read_u32(&mut r)? as usize;
        let alive_count = read_u32(&mut r)? as usize;
        let n_roots = read_u32(&mut r)? as usize;
        let rng = read_u64(&mut r)?;

        let mut basis = vec![0.0f32; VOCAB * DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut basis))?;
        let mut bias = [0.0f32; VOCAB];
        r.read_exact(bytemuck::cast_slice_mut(&mut bias))?;
        let mut alive = vec![0u8; len];
        r.read_exact(&mut alive)?;
        let mut span = vec![0u8; len];
        r.read_exact(&mut span)?;
        let mut depth = vec![0u8; len];
        r.read_exact(&mut depth)?;
        let mut parent = vec![0u32; len];
        r.read_exact(bytemuck::cast_slice_mut(&mut parent))?;
        let mut updates = vec![0u32; len];
        r.read_exact(bytemuck::cast_slice_mut(&mut updates))?;
        let mut w = vec![0u16; len * DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut w))?;
        let mut sharp_dst = vec![-1i16; len * SHARP_K];
        r.read_exact(bytemuck::cast_slice_mut(&mut sharp_dst))?;
        let mut sharp_w = vec![0u16; len * SHARP_K];
        r.read_exact(bytemuck::cast_slice_mut(&mut sharp_w))?;

        let edge_count = read_u64(&mut r)? as usize;
        let mut sigs = vec![0u64; buckets * 4];
        let mut children = vec![0u32; buckets * 4];
        let mask = buckets - 1;
        for _ in 0..edge_count {
            let s = read_u64(&mut r)?;
            let ch = read_u32(&mut r)?;
            let base = ((s as usize) & mask) * 4;
            if let Some(off) = sigs[base..base + 4].iter().position(|&e| e == 0 || e == s) {
                sigs[base + off] = s;
                children[base + off] = ch;
            }
        }
        let mut cue_w = vec![0u16; CUE_BUCKETS * DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut cue_w))?;
        let mut cue_updates = vec![0u32; CUE_BUCKETS];
        r.read_exact(bytemuck::cast_slice_mut(&mut cue_updates))?;
        let mut mode_emb = vec![0.0f32; NUM_MODES * DIM];
        r.read_exact(bytemuck::cast_slice_mut(&mut mode_emb))?;
        let mut mode_w = vec![0.0f32; CUE_BUCKETS * NUM_MODES];
        r.read_exact(bytemuck::cast_slice_mut(&mut mode_w))?;
        let mut mode_bias = [0.0f32; NUM_MODES];
        r.read_exact(bytemuck::cast_slice_mut(&mut mode_bias))?;

        Ok(Self {
            cap,
            buckets,
            len,
            alive_count,
            n_roots,
            w,
            sharp_dst,
            sharp_w,
            parent,
            span,
            depth,
            alive,
            updates,
            pain: vec![0.0f32; len],
            sigs,
            children,
            basis,
            bias,
            cue_w,
            cue_updates,
            mode_emb,
            mode_w,
            mode_bias,
            rng,
        })
    }
}

fn choose(v: &[f32], t: f32, seed: &mut u64) -> usize {
    let best = (0..v.len())
        .reduce(|a, b| if v[b] > v[a] { b } else { a })
        .unwrap();
    if t <= 0.0 {
        return best;
    }
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let sum: f64 = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| (((x - v[best]) / t) as f64).exp())
        .sum();
    let mut residual = ((*seed >> 11) as f64 / (1u64 << 53) as f64) * sum;
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            continue;
        }
        residual -= (((x - v[best]) / t) as f64).exp();
        if residual <= 0.0 {
            return i;
        }
    }
    best
}

fn legal_utf8(bytes: &[u8], next: u8, remaining: usize) -> bool {
    let offset = bytes.len().saturating_sub(3);
    let start = (offset..bytes.len())
        .rev()
        .find(|&i| bytes[i] & 0xc0 != 0x80);
    let mut tail = [0u8; 4];
    let mut n = 0;
    if let Some(i) = start {
        let prev = &bytes[i..];
        if std::str::from_utf8(prev).is_err() {
            tail[..prev.len()].copy_from_slice(prev);
            n = prev.len();
        }
    }
    tail[n] = next;
    n += 1;
    let width = match tail[0] {
        0..=127 => 1,
        194..=223 => 2,
        224..=239 => 3,
        240..=244 => 4,
        _ => return false,
    };
    if width > n + remaining - 1 {
        return false;
    }
    match std::str::from_utf8(&tail[..n]) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none(),
    }
}

fn load_jsonl(path: &Path) -> Result<Vec<Episode>> {
    let txt = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in txt.lines() {
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}

fn anchors_ok(plan: &ReasoningPlan, gold: &str, text: &str) -> bool {
    text.contains(gold)
        && plan
            .path_nodes
            .iter()
            .chain(&plan.auxiliary_values)
            .all(|n| text.contains(std::str::from_utf8(n).unwrap()))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2] + v[(v.len() - 1) / 2]) * 0.5
}

/// 推論言語コアを指定エポック学習し、バイナリチェックポイントを保存・再読込検証する。
pub fn train_and_save(corpus_dir: &Path, ckpt_path: &Path, cap: usize, epochs: usize) -> Result<u64> {
    let train = load_jsonl(&corpus_dir.join("train.jsonl"))?;
    let val = load_jsonl(&corpus_dir.join("val.jsonl"))?;
    let t0 = Instant::now();
    let mut core = ReasoningCore::new(cap);
    for epoch in 0..epochs.max(1) {
        let lr = if epoch == 0 { 1.0 } else { 0.6 };
        for (i, ep) in train.iter().enumerate() {
            let _ = core.train_episode(ep, true, lr);
            if (i + 1) % 48000 == 0 || i + 1 == train.len() {
                let (vm, vbpc) = core.eval_episodes(&val, true);
                eprintln!(
                    "[v19.0-save] ep={} {:6}/{} | {:.1}s | alive={} | val_mode={:.4} val_bpc={:.4}",
                    epoch + 1,
                    i + 1,
                    train.len(),
                    t0.elapsed().as_secs_f64(),
                    core.alive_count,
                    vm,
                    vbpc
                );
            }
        }
    }
    let bytes = core.save(ckpt_path)?;
    let rt = Instant::now();
    let reloaded = ReasoningCore::load(ckpt_path)?;
    let (vm, vbpc) = reloaded.eval_episodes(&val, true);
    eprintln!(
        "[v19.0-save] saved {} ({:.2} MiB), reload {:.2} ms, verified val_mode={:.4} val_bpc={:.4}",
        ckpt_path.display(),
        bytes as f64 / (1024.0 * 1024.0),
        rt.elapsed().as_secs_f64() * 1000.0,
        vm,
        vbpc
    );
    Ok(bytes)
}

/// v19.0 フル学習・ホールドアウト評価・未知320問・実SQLite・言い換え・アブレーション・監査を一貫実行する。
pub fn run_v190_bench(root: &Path, corpus_dir: &Path, out_json: &Path, cap: usize) -> Result<()> {
    let train = load_jsonl(&corpus_dir.join("train.jsonl"))?;
    let val = load_jsonl(&corpus_dir.join("val.jsonl"))?;
    ensure!(!train.is_empty() && !val.is_empty());

    let t0 = Instant::now();
    let mut core = ReasoningCore::new(cap);
    let mut curve = Vec::new();

    // 2 epochs with gentle LR decay
    for epoch in 0..2 {
        let lr = if epoch == 0 { 1.0 } else { 0.6 };
        let mut m_ok = 0usize;
        let mut nll = 0.0f64;
        let mut steps = 0usize;
        for (i, ep) in train.iter().enumerate() {
            let (ok, e_nll, e_steps) = core.train_episode(ep, true, lr);
            m_ok += usize::from(ok);
            nll += e_nll;
            steps += e_steps;
            if (i + 1) % 32000 == 0 || i + 1 == train.len() {
                let (vm, vbpc) = core.eval_episodes(&val, true);
                let (rss, _) = crate::memory::working_set();
                eprintln!(
                    "[v19.0] ep={} {:6}/{} | {:.1}s | alive={:7} | tr_mode={:.3} tr_bpc={:.4} | val_mode={:.4} val_bpc={:.4} | rss={:.0} MiB",
                    epoch + 1,
                    i + 1,
                    train.len(),
                    t0.elapsed().as_secs_f64(),
                    core.alive_count,
                    m_ok as f64 / (i + 1) as f64,
                    nll / steps as f64,
                    vm,
                    vbpc,
                    rss
                );
                curve.push(json!({
                    "epoch": epoch + 1,
                    "episodes_seen": epoch * train.len() + i + 1,
                    "elapsed_s": t0.elapsed().as_secs_f64(),
                    "alive_neurons": core.alive_count,
                    "train_mode_acc": m_ok as f64 / (i + 1) as f64,
                    "train_bpc": nll / steps as f64,
                    "val_mode_acc": vm,
                    "val_bpc": vbpc,
                    "rss_mib": rss
                }));
            }
        }
    }
    let train_elapsed_s = t0.elapsed().as_secs_f64();

    // Save compact checkpoint & verify reload roundtrip
    let ckpt_path = corpus_dir.join("reasoning_core_v190.bin");
    let ckpt_bytes = core.save(&ckpt_path)?;
    let reload_t = Instant::now();
    let reloaded = ReasoningCore::load(&ckpt_path)?;
    let reload_ms = reload_t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[v19.0] saved checkpoint {} ({:.2} MiB), reloaded in {:.2} ms",
        ckpt_path.display(),
        ckpt_bytes as f64 / (1024.0 * 1024.0),
        reload_ms
    );
    let core = reloaded;

    // Ablation on held-out val: Cue Bank ON vs OFF
    let (val_mode_on, val_bpc_on) = core.eval_episodes(&val, true);
    let (_, val_bpc_off) = core.eval_episodes(&val, false);

    // Evaluate on held-out bench_320.json (no entity overlap with training)
    let mut brain = Brain::open(root)?;
    let fixture: Value =
        serde_json::from_slice(&fs::read(root.join("rust_bench_v182/bench_320.json"))?)?;
    let cases = fixture["cases"].as_array().unwrap();

    let mut rows = Vec::new();
    let mut mode_correct_320 = 0usize;
    let mut exact_short = 0usize;
    let mut contains_full = 0usize;
    let mut all_anchors_full = 0usize;
    let mut stopped_full = 0usize;
    let mut forced_gates_full = 0usize;
    let mut nocache_contains = 0usize;
    let mut cue_diff_count = 0usize;
    let mut lats_full = Vec::new();
    let mut lats_short = Vec::new();

    for (i, c) in cases.iter().enumerate() {
        brain.reset_history();
        let mut full_prompt = String::new();
        if let Some(ts) = c["turns"].as_array() {
            for t in ts {
                let s = t.as_str().unwrap();
                brain.reasoner.observe_turn(s);
                full_prompt.push_str(s);
                full_prompt.push('\n');
            }
        }
        let q = c["prompt"].as_str().unwrap();
        full_prompt.push_str(q);
        let gold = c["gold"].as_str().unwrap();
        let plan = brain.reasoner.reason_plan(q);

        // Short
        let ts = Instant::now();
        let out_s = core.generate(
            &full_prompt,
            &plan,
            GenConfig {
                use_cue_bank: true,
                use_cache: true,
                short: true,
                temperature: 0.0,
                seed: 1,
            },
        );
        lats_short.push(ts.elapsed().as_secs_f64() * 1000.0);
        if out_s.text == gold {
            exact_short += 1;
        }

        // Full (Cue Bank ON + Cache ON)
        let tf = Instant::now();
        let out_f = core.generate(
            &full_prompt,
            &plan,
            GenConfig {
                use_cue_bank: true,
                use_cache: true,
                short: false,
                temperature: 0.0,
                seed: 1,
            },
        );
        let ms = tf.elapsed().as_secs_f64() * 1000.0;
        lats_full.push(ms);

        // Ablation: Cue Bank OFF vs ON phrasing diff
        let out_nocue = core.generate(
            &full_prompt,
            &plan,
            GenConfig {
                use_cue_bank: false,
                use_cache: true,
                short: false,
                temperature: 0.0,
                seed: 1,
            },
        );
        if out_nocue.text != out_f.text {
            cue_diff_count += 1;
        }

        // Ablation: Cache OFF
        let out_nocache = core.generate(
            &full_prompt,
            &plan,
            GenConfig {
                use_cue_bank: true,
                use_cache: false,
                short: false,
                temperature: 0.0,
                seed: 1,
            },
        );
        if out_nocache.text.contains(gold) {
            nocache_contains += 1;
        }

        let m_ok = out_f.predicted_mode == plan.mode;
        let hit = out_f.text.contains(gold);
        let anc = anchors_ok(&plan, gold, &out_f.text);
        mode_correct_320 += usize::from(m_ok);
        contains_full += usize::from(hit);
        all_anchors_full += usize::from(anc);
        stopped_full += usize::from(out_f.stopped);
        forced_gates_full += out_f.forced_budget_gates;

        rows.push(json!({
            "index": i,
            "group": c["group"],
            "mode": plan.mode,
            "predicted_mode": out_f.predicted_mode,
            "mode_match": m_ok,
            "gold": gold,
            "all_anchors": anc,
            "contains": hit,
            "ms": ms,
            "text": out_f.text,
            "n_p": plan.path_nodes.len(),
            "n_v": plan.auxiliary_values.len()
        }));
    }

    // Diversity & Determinism (20 questions x 5 seeds at T=0.0 and T=0.55)
    let mut diversity = Vec::new();
    for i in [
        0, 12, 22, 29, 30, 36, 45, 50, 87, 135, 140, 145, 150, 230, 260, 270, 280, 290, 300, 310,
    ] {
        let c = &cases[i];
        brain.reset_history();
        let mut full_prompt = String::new();
        if let Some(ts) = c["turns"].as_array() {
            for t in ts {
                let s = t.as_str().unwrap();
                brain.reasoner.observe_turn(s);
                full_prompt.push_str(s);
                full_prompt.push('\n');
            }
        }
        let q = c["prompt"].as_str().unwrap();
        full_prompt.push_str(q);
        let gold = c["gold"].as_str().unwrap();
        let plan = brain.reasoner.reason_plan(q);
        for temp in [0.0f32, 0.55] {
            let mut samples = Vec::new();
            let mut intact = 0usize;
            for seed in 1..=5 {
                let o = core.generate(
                    &full_prompt,
                    &plan,
                    GenConfig {
                        use_cue_bank: true,
                        use_cache: true,
                        short: false,
                        temperature: temp,
                        seed,
                    },
                );
                intact += usize::from(anchors_ok(&plan, gold, &o.text));
                samples.push(o.text);
            }
            let unique = samples.iter().collect::<BTreeSet<_>>().len();
            if temp == 0.0 {
                ensure!(unique == 1);
            }
            diversity.push(json!({
                "index": i,
                "temperature": temp,
                "unique": unique,
                "anchors": intact,
                "samples": samples
            }));
        }
    }

    // Wikipedia queries and held-out paraphrases
    brain.reset_history();
    let mut wiki = Vec::new();
    for (q, gold) in [
        ("富士山と北岳の標高差は何m？", "583.12m"),
        ("富士山と北岳と槍ヶ岳の標高差は何m？", "596.12m"),
        ("富士山と槍ヶ岳はどちらが標高が高い？", "富士山"),
        ("富士山と北岳の高さの違いってどれくらい？", "583.12m"),
    ] {
        if let Some(plan) = brain.hip.compare_plan(q, &brain.reasoner)? {
            let o = core.generate(
                q,
                &plan,
                GenConfig {
                    use_cue_bank: true,
                    use_cache: true,
                    short: false,
                    temperature: 0.0,
                    seed: 1,
                },
            );
            wiki.push(json!({
                "prompt": q,
                "gold": gold,
                "mode": plan.mode,
                "predicted_mode": o.predicted_mode,
                "mode_match": o.predicted_mode == plan.mode,
                "all_anchors": anchors_ok(&plan, gold, &o.text),
                "text": o.text
            }));
        }
    }

    // Held-out natural paraphrase intent classification check
    let paraphrases = [
        ("富士山と北岳の高さの差ってどれくらい？", ReasonMode::QuantDiff),
        ("東京タワーとスカイツリーはどっちが高いですか？", ReasonMode::QuantCompare),
        ("健太は太郎より背が高く、太郎は次郎より背が高い。一番背が高いのは？", ReasonMode::OrderExtreme),
        ("A・B・Cのうち、AとBは条件から外れる。残るのは？", ReasonMode::Elimination),
        ("すべての鳥は動物で、スズメは鳥です。スズメは動物ですか？", ReasonMode::Syllogism),
        ("甲は乙より大きく、乙は丙より大きく、丙は甲より大きい。一番大きいのは？", ReasonMode::ContradictionCycle),
        ("佐藤の所属は大阪です。田中は佐藤と同じ所属です。田中の所属は？", ReasonMode::Transfer),
        ("Xの値は42ですが、Yの値は何ですか？", ReasonMode::Abstain),
    ];
    let mut para_rows = Vec::new();
    let mut para_ok = 0usize;
    for (q, expected) in paraphrases {
        let (pred, _) = core.predict_mode(q);
        let ok = pred == expected;
        para_ok += usize::from(ok);
        para_rows.push(json!({"prompt": q, "expected": expected, "predicted": pred, "ok": ok}));
    }

    // Session reset / apoptosis bleed audit
    let aq = "追跡キー＝";
    let base_plan = brain.reasoner.reason_plan(aq);
    let baseline = core.generate(
        aq,
        &base_plan,
        GenConfig {
            use_cue_bank: true,
            use_cache: true,
            short: false,
            temperature: 0.0,
            seed: 1,
        },
    );
    brain.reasoner.observe_turn("追跡キー＝砂時計印");
    let dirty_plan = brain.reasoner.reason_plan(aq);
    let before = core.generate(
        aq,
        &dirty_plan,
        GenConfig {
            use_cue_bank: true,
            use_cache: true,
            short: false,
            temperature: 0.0,
            seed: 1,
        },
    );
    brain.reset_history();
    let clean_plan = brain.reasoner.reason_plan(aq);
    let after = core.generate(
        aq,
        &clean_plan,
        GenConfig {
            use_cue_bank: true,
            use_cache: true,
            short: false,
            temperature: 0.0,
            seed: 1,
        },
    );
    let bleed = after.text != baseline.text || after.text.contains("砂時計印");
    ensure!(before.text.contains("砂時計印"));

    // Forbidden hardcoded template strings audit on THIS source file
    let forbidden = ["なので", "答えは", "循環", "否定", "候補", "該当する", "同じ関係"];
    // Note: check only outside tests/audit literal line by checking a simple pattern list
    let src_code = include_str!("reasoning_core.rs")
        .split("// Forbidden hardcoded")
        .next()
        .unwrap_or("");
    let gen_hits: Vec<_> = forbidden
        .iter()
        .filter(|&&s| src_code.contains(s))
        .collect();

    let (rss, peak) = crate::memory::working_set();
    let summary = json!({
        "train_episodes": train.len(),
        "val_episodes": val.len(),
        "train_elapsed_s": train_elapsed_s,
        "alive_neurons": core.alive_count,
        "synaptic_params": (core.alive_count + core.n_roots + CUE_BUCKETS) * (DIM + SHARP_K),
        "val_mode_acc": val_mode_on,
        "val_bpc_cue_on": val_bpc_on,
        "val_bpc_cue_off": val_bpc_off,
        "bench320": {
            "short_exact": exact_short,
            "full_contains": contains_full,
            "full_all_anchors": all_anchors_full,
            "full_stopped": stopped_full,
            "full_forced_gates": forced_gates_full,
            "zero_regex_mode_acc": mode_correct_320,
            "nocache_contains": nocache_contains,
            "cue_bank_phrasing_diffs": cue_diff_count,
            "short_median_ms": median(lats_short),
            "full_median_ms": median(lats_full),
        },
        "paraphrase_intent_acc": format!("{para_ok}/{}", para_rows.len()),
        "rss_mib": rss,
        "peak_rss_mib": peak,
        "bleed": bleed,
        "generator_template_hits": gen_hits,
    });
    eprintln!("=== v19.0 SUMMARY ===\n{}", serde_json::to_string_pretty(&summary)?);

    let payload = json!({
        "summary": summary,
        "curve": curve,
        "wiki": wiki,
        "paraphrases": para_rows,
        "diversity": diversity,
        "rows": rows,
        "audit": {
            "bleed": bleed,
            "before": before.text,
            "after": after.text,
            "generator_template_hits": gen_hits
        }
    });
    if let Some(parent) = out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_json, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_core_pointers_and_cues() {
        assert_eq!(
            tokenize_reasoning("<P>（<V>）→<ANS>!"),
            vec![NEXT_P as u16, 0xef, 0xbc, 0x88, NEXT_V as u16, 0xef, 0xbc, 0x89, 0xe2, 0x86, 0x92, ANS as u16, b'!' as u16, END as u16]
        );
        let mut c = ReasoningCore::new(65536);
        let ep = Episode {
            mode: ReasonMode::QuantDiff,
            prompt: "白峰と黒岳の標高差は何m？".into(),
            n_p: 2,
            n_v: 2,
            reasoning: "測定値は<P>（<V>）と<P>（<V>）です。差は<ANS>です。".into(),
            split: "train".into(),
        };
        let mut last_bpc = 99.0;
        for _ in 0..12 {
            let (_, nll, st) = c.train_episode(&ep, true, 1.0);
            last_bpc = nll / st as f64;
        }
        assert!(last_bpc < 1.5, "bpc={last_bpc}");
        let (pred, _) = c.predict_mode("白峰と黒岳の標高差は何m？");
        assert_eq!(pred, ReasonMode::QuantDiff);
        let plan = ReasoningPlan {
            mode: ReasonMode::QuantDiff,
            answer_bytes: b"583.12m".to_vec(),
            path_nodes: vec!["北岳".as_bytes().to_vec(), "富士山".as_bytes().to_vec()],
            auxiliary_values: vec![b"3193m".to_vec(), b"3776.12m".to_vec()],
        };
        let out = c.generate(
            "富士山と北岳の標高差は何m？",
            &plan,
            GenConfig {
                use_cue_bank: true,
                use_cache: true,
                short: false,
                temperature: 0.0,
                seed: 1,
            },
        );
        assert!(out.text.contains("北岳") && out.text.contains("富士山") && out.text.contains("583.12m"), "out={}", out.text);
        let tmp = std::env::temp_dir().join(format!("ng-rc-test-{}.bin", std::process::id()));
        c.save(&tmp).unwrap();
        let reloaded = ReasoningCore::load(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        let out2 = reloaded.generate(
            "富士山と北岳の標高差は何m？",
            &plan,
            GenConfig {
                use_cue_bank: true,
                use_cache: true,
                short: false,
                temperature: 0.0,
                seed: 1,
            },
        );
        assert_eq!(out.text, out2.text);
    }
}
