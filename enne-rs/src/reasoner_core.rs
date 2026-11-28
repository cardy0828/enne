//! ENNE v24.5 関係推論野コア (`ReasonerCore`, `NGRC2451` + `SyntaxCore`)。
//!
//! 設計仕様:
//! - 逆伝播や事前学習済み言語モデル、固定JSON辞書・記号グラフ探索を使わず動作する。
//! - 前段 文字単位構文・節境界知覚コア (`SyntaxCore`) が文字列を節リストと型付きスパン
//!   (`Entity`/`Attr`/`Num`/`Pred`/`Func`) へ復号し、後段 `ReasonerCore` が構造化推論を担う。
//! - 再帰スパイキング TTFS 作業記憶グラフ (`NeuralWorkingGraph`):
//!   シナプス行列積の再帰伝播で初到達スパイク電位 $H_{ij} = \gamma^{t-1}$ ($\gamma = 0.75$) を求め、
//!   閉路判定 ($H_{ii} > 0$)・到達性 ($H_{uv} > 0$)・優越順位・最急降下トレース復元を実行する。
//! - 対照ヘブ述語SDR記憶と 2層オヤ・ヘブ k-WTA / 局所デルタ読み出しヘッド (`quant_head`, `order_head`, `prem_head`)。

use crate::{
    core_dag::mix,
    reasoner::{ReasonMode, ReasoningPlan},
    syntax_core::{ParsedClause, SpanKind, SyntaxCore},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use unicode_normalization::UnicodeNormalization;

pub const D_IN: usize = 128;
pub const HIDDEN: usize = 64;
pub const TOP_K: usize = 8;
pub const FEAT_BUCKETS: usize = 16384;

pub const NUM_QUANT: usize = 3; // 0=Diff, 1=Max, 2=Min
pub const NUM_ORDER: usize = 5; // 0=Source, 1=Sink, 2=Rank2, 3=Rank3, 4=Rank4
pub const NUM_PREM: usize = 2;  // 0=NonPremise, 1=Premise

const ROLE_ORDER: usize = 0;
const ROLE_SYM_ALIAS: usize = 1;
const ROLE_POOL: usize = 2;
const ROLE_INHIBIT: usize = 3;
const ROLE_CLASS_MOD: usize = 4;
const ROLE_CLASS_COPULA: usize = 5;
const NUM_ROLES: usize = 6;

const MAGIC: &[u8; 8] = b"NGRC2451";
const TTFS_GAMMA: f32 = 0.75;

#[inline]
fn is_hira_block(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

fn stem_prefix_without_trailing_hira(tok: &str) -> Option<&str> {
    let mut chars = tok.chars();
    let c0 = chars.next()?;
    let c1 = chars.next()?;
    if is_hira_block(c0) || is_hira_block(c1) {
        return None;
    }
    let last = tok.chars().last()?;
    if tok.chars().count() >= 3 && is_hira_block(last) {
        Some(&tok[..tok.len() - last.len_utf8()])
    } else {
        None
    }
}

/// UTF-8 文字 1/2/3-gram の厳密64-bit符号集合（対照ヘブ反意判定用）。
fn char_ngrams_123(raw: &str) -> Vec<u64> {
    let clean: Vec<char> = raw
        .trim()
        .nfkc()
        .filter(|c| !c.is_whitespace())
        .collect();
    let mut out = Vec::with_capacity(clean.len() * 3);
    for i in 0..clean.len() {
        let c0 = clean[i] as u64;
        out.push((1u64 << 60) | c0);
        if i + 1 < clean.len() {
            let c1 = clean[i + 1] as u64;
            out.push((2u64 << 60) | (c0 << 21) | c1);
            if i + 2 < clean.len() {
                let c2 = clean[i + 2] as u64;
                out.push((3u64 << 60) | (c0 << 40) | (c1 << 20) | c2);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 共有接辞を自動相殺する対照ヘブ表現内積: $\langle \phi(q), \phi(\text{anti}) - \phi(\text{base}) \rangle$。
pub fn contrastive_sdr_dot(q: &str, anti: &str, base: &str) -> f32 {
    if anti == base {
        return 0.0;
    }
    let qg = char_ngrams_123(q);
    let ag = char_ngrams_123(anti);
    let bg = char_ngrams_123(base);
    let mut score = 0.0f32;
    for g in ag {
        if qg.binary_search(&g).is_ok() {
            score += 1.0;
        }
    }
    for g in bg {
        if qg.binary_search(&g).is_ok() {
            score -= 1.0;
        }
    }
    score
}

/// 言語非依存の構造特徴と UTF-8 文字 1..=4-gram ハッシュ抽出（固定日本語語彙・区切り文字ゼロ）。
pub fn extract_neural_features(raw: &str) -> (Vec<usize>, [f32; D_IN]) {
    let q: String = raw.trim().nfkc().collect();
    let chars: Vec<char> = q.chars().collect();
    let mask = FEAT_BUCKETS - 1;
    let mut feats = Vec::with_capacity(chars.len() * 4 + 16);

    let mut n_digits = 0usize;
    let mut in_num = false;
    let mut num_count = 0usize;
    let mut n_hira = 0usize;
    let mut n_kata = 0usize;
    let mut n_kanji = 0usize;
    let mut n_punct = 0usize;
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
            n_hira += 1;
        } else if ('\u{30A0}'..='\u{30FF}').contains(&c) {
            n_kata += 1;
        } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
            n_kanji += 1;
        } else if !c.is_alphanumeric() && !c.is_whitespace() && c != '.' {
            n_punct += 1;
        }
    }
    let ends_punct = chars
        .last()
        .is_some_and(|&c| !c.is_alphanumeric() && !c.is_whitespace());
    let has_digit_rank = chars
        .windows(2)
        .any(|w| w[0].is_ascii_digit() && !w[1].is_ascii_digit())
        || n_digits > 0;
    let len_b = match chars.len() {
        0..=6 => 0u64,
        7..=14 => 1,
        15..=28 => 2,
        _ => 3,
    };
    let struct_tags = [
        (0xE101u64, num_count.min(4) as u64),
        (0xE102, ends_punct as u64),
        (0xE103, n_punct.min(6) as u64),
        (0xE106, len_b),
        (0xE107, (n_digits > 0) as u64),
        (0xE108, has_digit_rank as u64),
        (0xE109, (n_kanji * 2 > chars.len().max(1)) as u64),
        (0xE10A, (n_hira * 2 > chars.len().max(1)) as u64),
        (0xE10B, (n_kata >= 2) as u64),
    ];
    for (tag, val) in struct_tags {
        feats.push((mix(tag ^ val.wrapping_mul(0x9e3779b9)) as usize) & mask);
    }

    let clean: Vec<char> = chars.into_iter().filter(|c| !c.is_whitespace()).collect();
    for i in 0..clean.len() {
        let c0 = clean[i] as u64;
        feats.push((mix(c0 ^ 0x111) as usize) & mask);
        if i + 1 < clean.len() {
            let c1 = clean[i + 1] as u64;
            feats.push((mix((c0 << 21) ^ c1 ^ 0x222) as usize) & mask);
            if i + 2 < clean.len() {
                let c2 = clean[i + 2] as u64;
                feats.push((mix((c0 << 42) ^ (c1 << 21) ^ c2 ^ 0x333) as usize) & mask);
                if i + 3 < clean.len() {
                    let c3 = clean[i + 3] as u64;
                    feats.push((mix(mix((c0 << 21) ^ c1) ^ ((c2 << 21) ^ c3) ^ 0x444) as usize) & mask);
                }
            }
        }
    }
    feats.sort_unstable();
    feats.dedup();

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

/// 再帰スパイキング TTFS 神経作業記憶グラフ。
pub struct NeuralWorkingGraph<'a> {
    pub nodes: Vec<&'a str>,
    pub adj: Vec<f32>,
    pub ttfs: Vec<f32>,
    edges: Vec<(usize, usize)>,
    propagated: bool,
}

impl<'a> NeuralWorkingGraph<'a> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            adj: Vec::new(),
            ttfs: Vec::new(),
            edges: Vec::new(),
            propagated: false,
        }
    }

    pub fn ensure_node(&mut self, label: &'a str) -> usize {
        if let Some(pos) = self.nodes.iter().position(|&n| n == label) {
            pos
        } else {
            let id = self.nodes.len();
            self.nodes.push(label);
            self.propagated = false;
            id
        }
    }

    pub fn add_edge(&mut self, u: &'a str, v: &'a str) {
        let iu = self.ensure_node(u);
        let iv = self.ensure_node(v);
        if !self.edges.contains(&(iu, iv)) {
            self.edges.push((iu, iv));
        }
        self.propagated = false;
    }

    /// シナプス行列積の再帰伝播によって TTFS 初到達電位行列を計算する。
    pub fn propagate(&mut self) {
        let n = self.nodes.len();
        self.adj.clear();
        self.adj.resize(n * n, 0.0);
        self.ttfs.clear();
        self.ttfs.resize(n * n, 0.0);
        if n == 0 {
            self.propagated = true;
            return;
        }
        let mut active = vec![0.0f32; n * n];
        for &(u, v) in &self.edges {
            self.adj[u * n + v] = 1.0;
            self.ttfs[u * n + v] = 1.0;
            active[u * n + v] = 1.0;
        }
        let mut next_active = vec![0.0f32; n * n];
        let mut g = TTFS_GAMMA;
        for _step in 1..n {
            next_active.fill(0.0);
            let mut any = false;
            for i in 0..n {
                let row_a = i * n;
                for k in 0..n {
                    if active[row_a + k] <= 0.5 {
                        continue;
                    }
                    let row_k = k * n;
                    for j in 0..n {
                        if self.adj[row_k + j] > 0.5 && self.ttfs[row_a + j] == 0.0 {
                            self.ttfs[row_a + j] = g;
                            next_active[row_a + j] = 1.0;
                            any = true;
                        }
                    }
                }
            }
            if !any {
                break;
            }
            std::mem::swap(&mut active, &mut next_active);
            g *= TTFS_GAMMA;
        }
        self.propagated = true;
    }

    pub fn reachable(&self, u: &str, v: &str) -> bool {
        let n = self.nodes.len();
        let (Some(iu), Some(iv)) = (
            self.nodes.iter().position(|&x| x == u),
            self.nodes.iter().position(|&x| x == v),
        ) else {
            return false;
        };
        self.ttfs[iu * n + iv] > 0.0
    }

    pub fn has_cycle_at(&self, i: usize) -> bool {
        let n = self.nodes.len();
        i < n && self.ttfs[i * n + i] > 0.0
    }

    pub fn dom_count(&self, i: usize) -> usize {
        let n = self.nodes.len();
        if i >= n {
            return 0;
        }
        (0..n).filter(|&j| j != i && self.ttfs[i * n + j] > 0.0).count()
    }

    pub fn in_deg(&self, i: usize) -> usize {
        let n = self.nodes.len();
        (0..n).filter(|&u| self.adj[u * n + i] > 0.5).count()
    }

    pub fn out_deg(&self, i: usize) -> usize {
        let n = self.nodes.len();
        (0..n).filter(|&v| self.adj[i * n + v] > 0.5).count()
    }

    /// TTFS 電位の最急勾配に沿った局所シナプス発火経路の復元（閉路・連鎖共通）。
    pub fn trace_ttfs(&self, start: &str, end: &str) -> Vec<&'a str> {
        let n = self.nodes.len();
        let (Some(is), Some(ie)) = (
            self.nodes.iter().position(|&x| x == start),
            self.nodes.iter().position(|&x| x == end),
        ) else {
            return vec![];
        };
        if self.ttfs[is * n + ie] <= 0.0 {
            return vec![];
        }
        let mut route = vec![self.nodes[is]];
        let mut cur = is;
        let mut visited_mid = vec![false; n];
        visited_mid[is] = true;

        for step in 0..=n {
            if (step > 0 || cur != ie) && self.adj[cur * n + ie] > 0.5 {
                route.push(self.nodes[ie]);
                return route;
            }
            let mut best_nxt = None;
            let mut best_pot = 0.0f32;
            for nxt in 0..n {
                if self.adj[cur * n + nxt] <= 0.5 || visited_mid[nxt] {
                    continue;
                }
                let pot = self.ttfs[nxt * n + ie];
                if pot > best_pot {
                    best_pot = pot;
                    best_nxt = Some(nxt);
                }
            }
            if let Some(nxt) = best_nxt {
                visited_mid[nxt] = true;
                route.push(self.nodes[nxt]);
                cur = nxt;
            } else {
                break;
            }
        }
        vec![]
    }
}

/// 側方抑制 NMS（最長非重複スパン優先・左から右）を通じて質問中の活性化実体を抽出する。
pub fn find_wm_entities<'a>(q: &str, candidates: &[&'a str]) -> Vec<&'a str> {
    let mut uniq: Vec<&'a str> = candidates.iter().copied().filter(|s| !s.is_empty()).collect();
    uniq.sort_unstable();
    uniq.dedup();
    uniq.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

    let mut chosen: Vec<(usize, usize, &'a str)> = Vec::new();
    for ent in uniq {
        for (pos, _) in q.match_indices(ent) {
            let s0 = pos;
            let e0 = pos + ent.len();
            if !chosen.iter().any(|&(cs, ce, _)| s0 < ce && cs < e0) {
                chosen.push((s0, e0, ent));
            }
        }
    }
    chosen.sort_by_key(|&(s, _, _)| s);
    chosen.into_iter().map(|(_, _, ent)| ent).collect()
}

fn shared_tail(a: &str, b: &str) -> String {
    a.chars()
        .rev()
        .zip(b.chars().rev())
        .take_while(|(x, y)| x == y)
        .map(|p| p.0)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

/// 連続する同一 Func 区切りで並んだ実体候補リストの先頭塊（長さ >= 3）を取り出す。
fn extract_pool_list<'a>(cl: &ParsedClause<'a>) -> Option<(Vec<&'a str>, usize)> {
    let sp = &cl.spans;
    if sp.len() < 5 || sp[0].kind != SpanKind::Entity {
        return None;
    }
    if sp[1].kind != SpanKind::Func {
        return None;
    }
    let sep = sp[1].text;
    let mut items = vec![sp[0].text];
    let mut end_byte = sp[0].byte_end;
    let mut i = 1usize;
    while i + 1 < sp.len()
        && sp[i].kind == SpanKind::Func
        && sp[i].text == sep
        && sp[i + 1].kind == SpanKind::Entity
    {
        items.push(sp[i + 1].text);
        end_byte = sp[i + 1].byte_end;
        i += 2;
    }
    if items.len() >= 3 {
        Some((items, end_byte))
    } else {
        None
    }
}

pub struct ReasonerCore {
    pub neural_trained: bool,
    pub syntax: SyntaxCore,
    // Shared Layer 1: Competitive Oja-Hebbian prototypes
    pub w1: Vec<f32>,
    pub theta: [f32; HIDDEN],
    // Layer 2 Heads: Quant (Diff/Max/Min), Order (Source/Sink/R2/R3/R4), Premise (No/Yes)
    pub w2_quant: Vec<f32>,
    pub w_dir_quant: Vec<f32>,
    pub b_quant: [f32; NUM_QUANT],
    pub w2_order: Vec<f32>,
    pub w_dir_order: Vec<f32>,
    pub b_order: [f32; NUM_ORDER],
    pub w2_prem: Vec<f32>,
    pub w_dir_prem: Vec<f32>,
    pub b_prem: [f32; NUM_PREM],
    // Contrastive Hebbian Predicate SDR Memory + E/I Plasticity Channels
    pub preds: Vec<String>,
    pub anti_edges: Vec<(u32, u32, f32)>,
    pub role_w: Vec<[f32; NUM_ROLES]>,
    pub pol_order_ei: Vec<[f32; 2]>,
    pub pol_quant_ei: Vec<[f32; 2]>,
    // Learned response tokens: [yes, no, unknown, contradiction]
    pub labels: [String; 4],
    // Ephemeral session working-memory buffer (cleared on apoptosis)
    pub turns: Vec<String>,
}

impl Default for ReasonerCore {
    fn default() -> Self {
        Self::blank()
    }
}

impl ReasonerCore {
    pub fn blank() -> Self {
        let mut w1 = vec![0.0f32; HIDDEN * D_IN];
        let mut st = 0x517cc1b727220a95u64;
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
        Self {
            neural_trained: false,
            syntax: SyntaxCore::blank(),
            w1,
            theta: [0.0f32; HIDDEN],
            w2_quant: vec![0.0f32; HIDDEN * NUM_QUANT],
            w_dir_quant: vec![0.0f32; FEAT_BUCKETS * NUM_QUANT],
            b_quant: [0.0f32; NUM_QUANT],
            w2_order: vec![0.0f32; HIDDEN * NUM_ORDER],
            w_dir_order: vec![0.0f32; FEAT_BUCKETS * NUM_ORDER],
            b_order: [0.0f32; NUM_ORDER],
            w2_prem: vec![0.0f32; HIDDEN * NUM_PREM],
            w_dir_prem: vec![0.0f32; FEAT_BUCKETS * NUM_PREM],
            b_prem: [0.0f32; NUM_PREM],
            preds: Vec::new(),
            anti_edges: Vec::new(),
            role_w: Vec::new(),
            pol_order_ei: Vec::new(),
            pol_quant_ei: Vec::new(),
            labels: [
                "はい".into(),
                "いいえ".into(),
                "判断不能".into(),
                "矛盾".into(),
            ],
            turns: Vec::new(),
        }
    }

    pub fn from_blank_seed_json(bytes: &[u8]) -> Result<Self> {
        Self::from_blank_seed_json_with_syntax(bytes, SyntaxCore::blank())
    }

    pub fn from_blank_seed_json_with_syntax(bytes: &[u8], syntax: SyntaxCore) -> Result<Self> {
        let v: Value = serde_json::from_slice(bytes)?;
        let st = &v["state"];
        let mut core = Self::blank();
        core.syntax = syntax;
        if let Some(s) = st["yes_label"].as_str() {
            core.labels[0] = s.to_owned();
        }
        if let Some(s) = st["no_label"].as_str() {
            core.labels[1] = s.to_owned();
        }
        if let Some(s) = st["unknown_label"].as_str() {
            core.labels[2] = s.to_owned();
        }
        if let Some(s) = st["contradiction_label"].as_str() {
            core.labels[3] = s.to_owned();
        }
        Ok(core)
    }

    pub fn yes_label(&self) -> &str {
        &self.labels[0]
    }
    pub fn no_label(&self) -> &str {
        &self.labels[1]
    }
    pub fn unknown_label(&self) -> &str {
        &self.labels[2]
    }
    pub fn contradiction_label(&self) -> &str {
        &self.labels[3]
    }

    fn intern_pred(&mut self, s: &str) -> usize {
        if let Some(pos) = self.preds.iter().position(|p| p == s) {
            pos
        } else {
            let id = self.preds.len();
            self.preds.push(s.to_owned());
            self.role_w.push([0.0f32; NUM_ROLES]);
            self.pol_order_ei.push([0.0f32; 2]);
            self.pol_quant_ei.push([0.0f32; 2]);
            id
        }
    }

    fn reinforce_role(&mut self, tok: &str, role: usize, delta: f32) {
        if tok.is_empty() {
            return;
        }
        let id = self.intern_pred(tok);
        self.role_w[id][role] += delta;
    }

    fn reinforce_anti(&mut self, a: &str, b: &str) {
        if a.is_empty() || b.is_empty() || a == b {
            return;
        }
        let ia = self.intern_pred(a) as u32;
        let ib = self.intern_pred(b) as u32;
        self.role_w[ia as usize][ROLE_ORDER] += 1.0;
        self.role_w[ib as usize][ROLE_ORDER] += 1.0;
        for (u, v) in [(ia, ib), (ib, ia)] {
            if let Some(edge) = self
                .anti_edges
                .iter_mut()
                .find(|(eu, ev, _)| *eu == u && *ev == v)
            {
                edge.2 += 1.0;
            } else {
                self.anti_edges.push((u, v, 1.0));
            }
        }
    }

    fn has_anti(&self, a: &str, b: &str) -> bool {
        let (Some(ia), Some(ib)) = (
            self.preds.iter().position(|p| p == a),
            self.preds.iter().position(|p| p == b),
        ) else {
            return false;
        };
        let u = ia as u32;
        let v = ib as u32;
        self.anti_edges
            .iter()
            .any(|&(eu, ev, w)| eu == u && ev == v && w > 0.0)
    }

    fn anti_partners(&self, base: &str) -> Vec<&str> {
        let Some(ib) = self.preds.iter().position(|p| p == base) else {
            return vec![];
        };
        let b = ib as u32;
        self.anti_edges
            .iter()
            .filter(|&&(eu, _, w)| eu == b && w > 0.0)
            .map(|&(_, ev, _)| self.preds[ev as usize].as_str())
            .collect()
    }

    fn has_role(&self, s: &str, role: usize) -> bool {
        self.preds
            .iter()
            .enumerate()
            .any(|(id, tok)| self.role_w[id][role] > 0.0 && s.contains(tok.as_str()))
    }

    fn has_role_exact(&self, s: &str, role: usize) -> bool {
        self.preds
            .iter()
            .enumerate()
            .any(|(id, tok)| self.role_w[id][role] > 0.0 && tok == s)
    }

    fn is_class_mod(&self, modifier: &str) -> bool {
        self.has_role_exact(modifier, ROLE_CLASS_MOD)
    }

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

    fn eval_head<const C: usize>(
        &self,
        feats: &[usize],
        winners: &[(usize, f32); TOP_K],
        w2: &[f32],
        w_dir: &[f32],
        bias: &[f32; C],
    ) -> (usize, [f32; C]) {
        let mut logits = *bias;
        for &(h, act) in winners {
            if act > 0.0 {
                let base = h * C;
                for c in 0..C {
                    logits[c] += act * w2[base + c];
                }
            }
        }
        let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in feats {
            let base = f * C;
            for c in 0..C {
                logits[c] += inv_f * w_dir[base + c];
            }
        }
        let mut best = 0;
        for c in 1..C {
            if logits[c] > logits[best] {
                best = c;
            }
        }
        (best, logits)
    }

    pub fn predict_quant(&self, q: &str) -> (usize, [f32; NUM_QUANT]) {
        let (feats, x) = extract_neural_features(q);
        let winners = self.hidden_act(&x);
        self.eval_head(&feats, &winners, &self.w2_quant, &self.w_dir_quant, &self.b_quant)
    }

    pub fn predict_order(&self, q: &str) -> (usize, [f32; NUM_ORDER]) {
        let (feats, x) = extract_neural_features(q);
        let winners = self.hidden_act(&x);
        self.eval_head(&feats, &winners, &self.w2_order, &self.w_dir_order, &self.b_order)
    }

    pub fn predict_prem(&self, s: &str) -> (usize, [f32; NUM_PREM]) {
        let (feats, x) = extract_neural_features(s);
        let winners = self.hidden_act(&x);
        self.eval_head(&feats, &winners, &self.w2_prem, &self.w_dir_prem, &self.b_prem)
    }

    /// 型付き構文節から2実体＋関係述語（大小・順序候補）を抽出する。
    fn extract_binary_rel<'a>(cl: &ParsedClause<'a>) -> Option<(&'a str, &'a str, &'a str)> {
        let mut ents = Vec::new();
        let mut attrs = Vec::new();
        let mut preds = Vec::new();
        for (idx, sp) in cl.spans.iter().enumerate() {
            match sp.kind {
                SpanKind::Entity => ents.push((idx, sp.text)),
                SpanKind::Attr => attrs.push((idx, sp.text)),
                SpanKind::Pred => preds.push((idx, sp.text)),
                _ => {}
            }
        }
        if ents.len() == 2 && !preds.is_empty() {
            let has_mid_attr = attrs.iter().any(|&(k, _)| k > ents[0].0 && k < ents[1].0);
            if !has_mid_attr {
                return Some((ents[0].1, ents[1].1, preds.last().unwrap().1));
            }
        }
        None
    }

    /// 活用形（例: 連用形）を学習済み `ROLE_ORDER` または文脈述語の標準形へ寄せる。
    fn canonical_order_pred<'a>(&'a self, raw_p: &'a str, prompt_preds: &[&'a str]) -> &'a str {
        if self.has_role_exact(raw_p, ROLE_ORDER) {
            return raw_p;
        }
        let rp_len = raw_p.chars().count();
        if rp_len < 2 {
            return raw_p;
        }
        // Check known ROLE_ORDER predicates first
        let mut best: Option<(usize, &str)> = None;
        for (id, tok) in self.preds.iter().enumerate() {
            if self.role_w[id][ROLE_ORDER] <= 0.0 || tok == raw_p || self.has_anti(raw_p, tok) {
                continue;
            }
            let clen = tok.chars().count();
            if clen < 2 || clen.abs_diff(rp_len) > 1 {
                continue;
            }
            let shared = raw_p
                .chars()
                .zip(tok.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if shared >= 1 && shared + 1 >= rp_len.max(clen) {
                if best.is_none_or(|(bs, _)| shared > bs) {
                    best = Some((shared, tok.as_str()));
                }
            }
        }
        if let Some((_, matched)) = best {
            return matched;
        }
        // Fallback: align to another predicate appearing in the same prompt (prefer later/canonical form)
        for &cand in prompt_preds.iter().rev() {
            if cand == raw_p || self.has_anti(raw_p, cand) {
                continue;
            }
            let clen = cand.chars().count();
            if clen < 2 || clen.abs_diff(rp_len) > 1 {
                continue;
            }
            let shared = raw_p
                .chars()
                .zip(cand.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if shared >= 1 && shared + 1 >= rp_len.max(clen) {
                return cand;
            }
        }
        raw_p
    }

    pub fn observe_turn(&mut self, s: &str) {
        self.turns.push(s.to_owned());
    }

    pub fn has_memory(&self) -> bool {
        !self.turns.is_empty()
    }

    pub fn has_multi_clauses(&self, text: &str) -> bool {
        self.syntax.parse(text).len() >= 2
    }

    pub fn apoptosis(&mut self) {
        self.turns.clear();
    }

    pub fn is_unknown(&self, answer: &str) -> bool {
        self.labels[2] == answer
    }

    pub fn is_premise(&self, text: &str) -> bool {
        let clauses = self.syntax.parse(text);
        if clauses.is_empty() {
            return false;
        }
        let is_copy_cl = |cl: &ParsedClause<'_>| {
            cl.spans.len() == 3
                && cl.spans[0].kind == SpanKind::Entity
                && cl.spans[1].kind == SpanKind::Func
                && cl.spans[2].kind == SpanKind::Entity
        };
        if self.neural_trained {
            return clauses.iter().all(|cl| {
                is_copy_cl(cl)
                    || self.has_role(cl.raw, ROLE_INHIBIT)
                    || self.has_role(cl.raw, ROLE_POOL)
                    || self.has_role(cl.raw, ROLE_SYM_ALIAS)
                    || self.predict_prem(cl.raw).0 == 1
            });
        }
        clauses.iter().any(|cl| {
            is_copy_cl(cl)
                || cl.spans.iter().filter(|s| s.kind == SpanKind::Entity).count() >= 2
                || cl.spans.iter().any(|s| s.kind == SpanKind::Num)
                || self.has_role(cl.raw, ROLE_INHIBIT)
                || self.has_role(cl.raw, ROLE_POOL)
                || self.has_role(cl.raw, ROLE_SYM_ALIAS)
        })
    }

    /// 無教師の対称・反意ペアから同義述語・反意シナプスを1回で獲得する（構文解析に SyntaxCore を使用）。
    pub fn observe_unlabeled(&mut self, text: &str) {
        let clauses = self.syntax.parse(text);
        let facts: Vec<_> = clauses
            .iter()
            .filter_map(Self::extract_binary_rel)
            .collect();
        for (i, &(a, b, r)) in facts.iter().enumerate() {
            for (j, &(c, d, t)) in facts.iter().enumerate() {
                if j > i && a == d && b == c {
                    if r == t {
                        self.reinforce_role(r, ROLE_SYM_ALIAS, 2.0);
                    } else {
                        self.reinforce_anti(r, t);
                    }
                }
                if b == c && a != d && t == r {
                    for &(u, v, p) in &facts {
                        if u == d && v == a && p != r {
                            self.reinforce_anti(r, p);
                        }
                    }
                }
            }
        }
    }

    fn learn_class(&mut self, facts: &[ParsedClause<'_>], answer: &str) {
        if self.labels[0] != answer {
            return;
        }
        let mut mod_candidates = Vec::new();
        let mut cop_candidates = Vec::new();
        for cl in facts {
            let mut ents = Vec::new();
            let mut attrs = Vec::new();
            let mut preds = Vec::new();
            for (idx, sp) in cl.spans.iter().enumerate() {
                match sp.kind {
                    SpanKind::Entity => ents.push((idx, sp.text)),
                    SpanKind::Attr => attrs.push((idx, sp.text)),
                    SpanKind::Pred => preds.push((idx, sp.text)),
                    _ => {}
                }
            }
            if ents.len() == 2 && !preds.is_empty() {
                let first_e = ents[0].0;
                for &(a_idx, a_txt) in &attrs {
                    if a_idx < first_e {
                        mod_candidates.push(a_txt.to_owned());
                    }
                }
                cop_candidates.push(preds.last().unwrap().1.to_owned());
            }
        }
        for m in mod_candidates {
            self.reinforce_role(&m, ROLE_CLASS_MOD, 1.0);
        }
        for c in cop_candidates {
            self.reinforce_role(&c, ROLE_CLASS_COPULA, 1.0);
        }
    }

    fn parse_num_clause<'a>(cl: &ParsedClause<'a>) -> Option<(f64, &'a str, &'a str)> {
        let mut subj = None;
        let mut num_sp = None;
        for sp in &cl.spans {
            if sp.kind == SpanKind::Entity && subj.is_none() {
                subj = Some(sp.text);
            } else if sp.kind == SpanKind::Num && num_sp.is_none() {
                num_sp = Some(sp);
            }
        }
        let subj = subj?;
        let nsp = num_sp?;
        let s = nsp.text;
        let end = s
            .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
            .unwrap_or(s.len());
        if end == 0 {
            return None;
        }
        let val = s[..end].parse::<f64>().ok()?;
        let tail_start = (nsp.byte_start - cl.byte_start) + end;
        let unit = cl.raw[tail_start..].trim();
        Some((val, subj, unit))
    }

    fn learn_numeric(&mut self, facts: &[ParsedClause<'_>], q_raw: &str, answer: &str) -> bool {
        let mut values = vec![];
        for cl in facts {
            if let Some(item) = Self::parse_num_clause(cl) {
                values.push(item);
            }
        }
        if values.len() < 2 || values.iter().any(|v| v.2 != values[0].2) {
            return false;
        }
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let lo = values[0];
        let hi = *values.last().unwrap();
        if lo.0 == hi.0 {
            return false;
        }
        let is_min = if answer == lo.1 {
            true
        } else if answer == hi.1 {
            false
        } else {
            return false;
        };
        let toks: Vec<String> = self
            .preds
            .iter()
            .enumerate()
            .filter(|(id, tok)| self.role_w[*id][ROLE_ORDER] > 0.0 && q_raw.contains(tok.as_str()))
            .map(|(_, tok)| tok.clone())
            .collect();
        for t in toks {
            let id = self.intern_pred(&t);
            if is_min {
                self.pol_quant_ei[id][1] += 1.0;
            } else {
                self.pol_quant_ei[id][0] += 1.0;
            }
        }
        true
    }

    fn learn_pool(&mut self, facts: &[ParsedClause<'_>], answer: &str) -> bool {
        for (fi, cl) in facts.iter().enumerate() {
            let Some((list, list_end)) = extract_pool_list(cl) else {
                continue;
            };
            if !list.contains(&answer) {
                continue;
            }
            let mut marker_tails = Vec::new();
            let mut stem_candidates = Vec::new();
            for item in list.iter().filter(|m| **m != answer) {
                for (oi, ocl) in facts.iter().enumerate() {
                    if oi == fi {
                        continue;
                    }
                    if let Some(first_ent) = ocl.spans.iter().find(|s| s.kind == SpanKind::Entity) {
                        if first_ent.text == *item {
                            let rest_spans: Vec<_> = ocl
                                .spans
                                .iter()
                                .skip_while(|s| s.byte_start <= first_ent.byte_end && (s.kind == SpanKind::Entity || s.kind == SpanKind::Func))
                                .collect();
                            let tail = if let Some(first_content) = rest_spans.first() {
                                &ocl.raw[first_content.byte_start - ocl.byte_start..]
                            } else {
                                ocl.raw.strip_prefix(item).unwrap_or("")
                            };
                            if !tail.is_empty() {
                                marker_tails.push(tail);
                                if let Some(pred) = ocl.spans.iter().rev().find(|s| s.kind == SpanKind::Pred) {
                                    if let Some(stem) = stem_prefix_without_trailing_hira(pred.text) {
                                        stem_candidates.push(stem.to_owned());
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            if marker_tails.len() + 1 != list.len() {
                continue;
            }
            let tail = &cl.raw[list_end - cl.byte_start..];
            let old_pools: Vec<String> = self
                .preds
                .iter()
                .enumerate()
                .filter(|(id, _)| self.role_w[*id][ROLE_POOL] > 0.0)
                .map(|(_, s)| s.clone())
                .collect();
            self.reinforce_role(tail, ROLE_POOL, 1.0);
            if let Some(pred) = cl.spans.iter().rev().find(|s| s.kind == SpanKind::Pred) {
                if let Some(stem) = stem_prefix_without_trailing_hira(pred.text) {
                    self.reinforce_role(stem, ROLE_POOL, 1.0);
                }
            }
            for other in old_pools {
                let chars: Vec<_> = other.chars().collect();
                let mut best = String::new();
                for start in 0..chars.len() {
                    for stop in start + 2..=chars.len() {
                        let part: String = chars[start..stop].iter().collect();
                        if tail.contains(&part) && part.len() > best.len() {
                            best = part;
                        }
                    }
                }
                if best.chars().count() >= 3 {
                    self.reinforce_role(&best, ROLE_POOL, 1.0);
                }
            }
            for key in marker_tails {
                let old_inhs: Vec<String> = self
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|(id, _)| self.role_w[*id][ROLE_INHIBIT] > 0.0)
                    .map(|(_, s)| s.clone())
                    .collect();
                self.reinforce_role(key, ROLE_INHIBIT, 1.0);
                for other in old_inhs {
                    let suffix = shared_tail(&other, key);
                    if suffix.chars().count() >= 2 {
                        self.reinforce_role(&suffix, ROLE_INHIBIT, 1.0);
                    }
                }
            }
            for st in stem_candidates {
                self.reinforce_role(&st, ROLE_INHIBIT, 1.0);
            }
            return true;
        }
        false
    }

    fn learn_transfer(
        &mut self,
        facts: &[ParsedClause<'_>],
        q_cl: &ParsedClause<'_>,
        answer: &str,
    ) -> bool {
        let Some(requested) = q_cl
            .spans
            .iter()
            .find(|s| s.kind == SpanKind::Entity)
            .map(|s| s.text)
        else {
            return false;
        };
        for fact in facts {
            let ents: Vec<&str> = fact
                .spans
                .iter()
                .filter(|s| s.kind == SpanKind::Entity)
                .map(|s| s.text)
                .collect();
            if ents.len() < 2 || *ents.last().unwrap() != answer {
                continue;
            }
            let owner = ents[0];
            if owner == requested {
                continue;
            }
            for line in facts {
                let line_ents: Vec<&str> = line
                    .spans
                    .iter()
                    .filter(|s| s.kind == SpanKind::Entity)
                    .map(|s| s.text)
                    .collect();
                if !line_ents.contains(&requested) || !line_ents.contains(&owner) {
                    continue;
                }
                if let Some(pred) = line.spans.iter().find(|s| s.kind == SpanKind::Pred) {
                    self.reinforce_role(pred.text, ROLE_SYM_ALIAS, 1.0);
                    return true;
                }
            }
        }
        false
    }

    fn learn_order(&mut self, parts: &[ParsedClause<'_>], q_cl: &ParsedClause<'_>, answer: &str) {
        let facts: Vec<_> = parts.iter().filter_map(Self::extract_binary_rel).collect();
        let Some(&(_, _, base)) = facts.first() else {
            return;
        };
        let mut graph = NeuralWorkingGraph::new();
        for &(a, b, r) in &facts {
            if r != base && self.has_anti(base, r) {
                graph.add_edge(b, a);
            } else {
                graph.add_edge(a, b);
            }
        }
        graph.propagate();
        let n = graph.nodes.len();
        if n < 2 {
            return;
        }
        let Some(ia) = graph.nodes.iter().position(|&x| x == answer) else {
            return;
        };
        let is_source = if graph.dom_count(ia) + 1 == n {
            true
        } else if (0..n).all(|u| u == ia || graph.ttfs[u * n + ia] > 0.0) {
            false
        } else {
            return;
        };

        let antis: Vec<String> = self.anti_partners(base).into_iter().map(str::to_owned).collect();
        let mut flipped = false;
        for sp in &q_cl.spans {
            if antis.iter().any(|a| sp.text == a) {
                flipped = true;
                break;
            }
        }
        let effective_source = if flipped { !is_source } else { is_source };
        let toks: Vec<String> = q_cl
            .spans
            .iter()
            .filter(|s| {
                s.kind != SpanKind::Func
                    && s.kind != SpanKind::Num
                    && s.text != base
                    && !antis.iter().any(|a| a == s.text)
            })
            .map(|s| s.text.to_owned())
            .collect();
        for tok in toks {
            let id = self.intern_pred(&tok);
            if effective_source {
                self.pol_order_ei[id][0] += 1.0;
            } else {
                self.pol_order_ei[id][1] += 1.0;
            }
        }
    }

    /// 1エピソード（背景前提＋問い＋正答）から関係役割と質問極性を学習する。
    pub fn learn_episode(&mut self, prompt: &str, answer: &str) {
        let mut clauses = self.syntax.parse(prompt);
        let Some(q_cl) = clauses.pop() else {
            return;
        };
        self.learn_class(&clauses, answer);
        if !self.learn_numeric(&clauses, q_cl.raw, answer)
            && !self.learn_pool(&clauses, answer)
            && !self.learn_transfer(&clauses, &q_cl, answer)
        {
            self.learn_order(&clauses, &q_cl, answer);
        }
        self.apoptosis();
    }

    fn online_quant_polarity(&self, q: &str) -> f32 {
        let pol = |id: usize| -> f32 {
            let hi = self.pol_quant_ei[id][0];
            let lo = self.pol_quant_ei[id][1];
            (hi - lo) / (hi + lo).max(1.0)
        };
        let mut score = 0.0f32;
        for (id, tok) in self.preds.iter().enumerate() {
            if self.role_w[id][ROLE_ORDER] <= 0.0 || !q.contains(tok.as_str()) {
                continue;
            }
            let direct = pol(id);
            if direct.abs() > 0.35 {
                score += direct;
            } else {
                let mut total = 0.0f32;
                let mut delta = 0.0f32;
                let u = id as u32;
                for &(eu, ev, w) in &self.anti_edges {
                    if eu == u && w > 0.0 {
                        total += w;
                        delta += w * pol(ev as usize);
                    }
                }
                if total > 0.0 {
                    score -= delta / total;
                }
            }
        }
        score
    }

    fn online_order_polarity(&self, q: &str, base: &str) -> f32 {
        let mut score = 0.0f32;
        for (id, tok) in self.preds.iter().enumerate() {
            let src = self.pol_order_ei[id][0];
            let sink = self.pol_order_ei[id][1];
            if src == 0.0 && sink == 0.0 {
                continue;
            }
            if tok == base || self.has_anti(base, tok) {
                continue;
            }
            if !q.contains(tok.as_str()) {
                continue;
            }
            let s0 = src + 0.5;
            let s1 = sink + 0.5;
            if ((s0 - s1) / (s0 + s1)).abs() > 0.45 {
                score += (s0 / s1).ln();
            }
        }
        score
    }

    fn same_pred_family(&self, a: &str, b: &str) -> bool {
        a == b || self.has_anti(a, b)
    }

    pub fn answer(&self, prompt: &str) -> String {
        String::from_utf8(self.reason_plan(prompt).answer_bytes).unwrap()
    }

    /// 数値付き実体タプル列に対して定量推論プラン (`QuantDiff` / `QuantCompare`) を構築する。
    pub fn reason_quantities_plan(&self, q: &str, items: &[(f64, &str, &str)]) -> ReasoningPlan {
        use ReasonMode::*;
        let plan = |mode, answer: &str, nodes: Vec<&str>| ReasoningPlan {
            mode,
            answer_bytes: answer.as_bytes().to_vec(),
            path_nodes: nodes.iter().map(|n| n.as_bytes().to_vec()).collect(),
            auxiliary_values: vec![],
        };
        let unknown = || plan(Abstain, self.unknown_label(), vec![]);
        if items.len() < 2 {
            return unknown();
        }
        let mut quantities = items.to_vec();
        quantities.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q_names: Vec<&str> = quantities.iter().map(|r| r.1).collect();
        let named_ents = find_wm_entities(q, &q_names);
        let named: Vec<_> = quantities
            .iter()
            .filter(|r| named_ents.contains(&r.1))
            .collect();
        let (lo, hi) = if named.len() >= 2 {
            (*named[0], **named.last().unwrap())
        } else {
            (quantities[0], *quantities.last().unwrap())
        };
        if lo.2 != hi.2 {
            return unknown();
        }
        let quantify = |mode, answer: &str| {
            let mut p = plan(mode, answer, vec![lo.1, hi.1]);
            p.auxiliary_values = [lo, hi]
                .iter()
                .map(|r| format!("{}{}", r.0, r.2).into_bytes())
                .collect();
            p
        };

        let (q_cls, _) = if self.neural_trained {
            self.predict_quant(q)
        } else {
            let want_diff = !lo.2.is_empty() && q.contains(lo.2);
            (usize::from(!want_diff), [0.0; NUM_QUANT])
        };
        if q_cls == 0 {
            let val_str = format!("{:.6}", hi.0 - lo.0);
            let clean_val = val_str.trim_end_matches('0').trim_end_matches('.');
            return quantify(QuantDiff, &format!("{clean_val}{}", lo.2));
        }
        let online_pol = self.online_quant_polarity(q);
        let low = if online_pol.abs() > 0.35 {
            online_pol < 0.0
        } else {
            q_cls == 2
        };
        quantify(QuantCompare, if low { lo.1 } else { hi.1 })
    }

    /// 文字単位純ヘブ構文解析＋再帰 TTFS 作業記憶グラフ＋神経読み出しヘッドで推論を実行する。
    pub fn reason_plan(&self, prompt: &str) -> ReasoningPlan {
        use ReasonMode::*;
        let plan = |mode, answer: &str, nodes: Vec<&str>| ReasoningPlan {
            mode,
            answer_bytes: answer.as_bytes().to_vec(),
            path_nodes: nodes.iter().map(|n| n.as_bytes().to_vec()).collect(),
            auxiliary_values: vec![],
        };
        let unknown = || plan(Abstain, self.unknown_label(), vec![]);

        let turn_clauses: Vec<ParsedClause<'_>> = self
            .turns
            .iter()
            .flat_map(|t| self.syntax.parse(t.as_str()))
            .collect();
        let mut p_clauses = self.syntax.parse(prompt);
        let Some(q_cl) = p_clauses.pop() else {
            return unknown();
        };
        let q = q_cl.raw;

        let mut parts: Vec<ParsedClause<'_>> = turn_clauses;
        parts.extend(p_clauses);

        let mut copy_pairs: Vec<(&str, &str, &str)> = Vec::new();
        let mut pool_observed = false;
        let mut pool_order: Vec<&str> = Vec::new();
        let mut pool_act: BTreeMap<&str, f32> = BTreeMap::new();
        let mut inhibit_clauses: Vec<(&str, Vec<&str>)> = Vec::new();
        let mut quantities: Vec<(f64, &str, &str)> = Vec::new();
        let mut aliases: Vec<(&str, &str)> = Vec::new();
        let mut assign: BTreeMap<&str, &str> = BTreeMap::new();
        let mut attr_links: Vec<(&str, &str)> = Vec::new();
        let mut classes: Vec<(&str, &str)> = Vec::new();
        let mut raw_order: Vec<(&str, &str, &str)> = Vec::new();

        for cl in &parts {
            let sp = &cl.spans;

            // 1. Copy record: single Func separating non-empty key Entity(ies) and val Entity(ies), no Attr/Pred/Num
            let n_funcs = sp.iter().filter(|s| s.kind == SpanKind::Func).count();
            let has_attr_or_pred_or_num = sp
                .iter()
                .any(|s| matches!(s.kind, SpanKind::Attr | SpanKind::Pred | SpanKind::Num));
            if n_funcs == 1 && !has_attr_or_pred_or_num {
                let f_sp = sp.iter().find(|s| s.kind == SpanKind::Func).unwrap();
                let lhs = cl.raw[..f_sp.byte_start - cl.byte_start].trim();
                let rhs = cl.raw[f_sp.byte_end - cl.byte_start..].trim();
                if !lhs.is_empty() && !rhs.is_empty() {
                    copy_pairs.push((lhs, f_sp.text, rhs));
                    continue;
                }
            }

            // 2. Pool introduction: >=3 Func-separated entities + learned ROLE_POOL
            if let Some((list, _)) = extract_pool_list(cl) {
                if self.has_role(cl.raw, ROLE_POOL) {
                    pool_observed = true;
                    for item in list {
                        if !pool_act.contains_key(item) {
                            pool_order.push(item);
                        }
                        pool_act.insert(item, 1.0);
                    }
                    continue;
                }
            }

            // 3. GABAergic inhibition line
            if self.has_role(cl.raw, ROLE_INHIBIT) {
                let ents: Vec<&str> = sp
                    .iter()
                    .filter(|s| s.kind == SpanKind::Entity)
                    .map(|s| s.text)
                    .collect();
                inhibit_clauses.push((cl.raw, ents));
                continue;
            }

            let mut ents = Vec::new();
            let mut attrs = Vec::new();
            let mut preds = Vec::new();
            for (idx, item) in sp.iter().enumerate() {
                match item.kind {
                    SpanKind::Entity => ents.push((idx, item.text)),
                    SpanKind::Attr => attrs.push((idx, item.text)),
                    SpanKind::Pred => preds.push((idx, item.text)),
                    _ => {}
                }
            }

            // 4. Symmetric alias
            if ents.len() >= 2 && self.has_role(cl.raw, ROLE_SYM_ALIAS) {
                aliases.push((ents[0].1, ents[1].1));
                continue;
            }

            // 5. Quantity measurement
            if let Some(q_item) = Self::parse_num_clause(cl) {
                quantities.push(q_item);
                continue;
            }

            // 6. AttrAssign vs ClassIsA vs Order
            if ents.len() == 2 {
                let (a_idx, a_txt) = ents[0];
                let (b_idx, b_txt) = ents[1];
                let has_mid_attr = attrs.iter().any(|&(k, _)| k > a_idx && k < b_idx);
                let has_pre_mod = attrs.iter().any(|&(k, m)| k < a_idx && self.is_class_mod(m));

                if preds.is_empty() && has_mid_attr && !has_pre_mod {
                    if let Some(old) = assign.insert(a_txt, b_txt).filter(|old| *old != b_txt) {
                        return plan(
                            ContradictionConflict,
                            self.contradiction_label(),
                            vec![a_txt, old, b_txt],
                        );
                    }
                    attr_links.push((a_txt, b_txt));
                    continue;
                }

                if !preds.is_empty() {
                    let p_txt = preds.last().unwrap().1;
                    if self.has_role_exact(p_txt, ROLE_CLASS_COPULA) || has_pre_mod {
                        classes.push((a_txt, b_txt));
                        continue;
                    }
                    if has_mid_attr
                        && !self.has_role(p_txt, ROLE_ORDER)
                        && !self.has_role(p_txt, ROLE_SYM_ALIAS)
                    {
                        classes.push((a_txt, b_txt));
                        continue;
                    }
                    raw_order.push((a_txt, b_txt, p_txt));
                }
            }
        }

        // 0. Check ledger Copy query (single trailing Func matching recorded Copy Func symbol, no Attr/Pred/Num)
        let q_funcs = q_cl.spans.iter().filter(|s| s.kind == SpanKind::Func).count();
        let q_has_other = q_cl
            .spans
            .iter()
            .any(|s| matches!(s.kind, SpanKind::Attr | SpanKind::Pred | SpanKind::Num));
        if q_funcs == 1 && !q_has_other {
            let f_sp = q_cl.spans.iter().find(|s| s.kind == SpanKind::Func).unwrap();
            let key = q_cl.raw[..f_sp.byte_start - q_cl.byte_start].trim();
            let rhs = q_cl.raw[f_sp.byte_end - q_cl.byte_start..].trim();
            if !key.is_empty() && rhs.is_empty() {
                let sym = f_sp.text;
                if copy_pairs.iter().any(|&(_, s, _)| s == sym) {
                    for &(k, s, v) in copy_pairs.iter().rev() {
                        if k == key && s == sym {
                            return plan(Copy, v, vec![k]);
                        }
                    }
                    return unknown();
                }
            }
        }

        // Apply GABAergic inhibition signals to active pool neurons
        let mut removed = Vec::new();
        for (line_raw, line_ents) in &inhibit_clauses {
            for &item in &pool_order {
                if line_raw.starts_with(item) || line_ents.contains(&item) {
                    if let Some(act) = pool_act.get_mut(item) {
                        if *act > 0.0 {
                            removed.push(item);
                        }
                        *act -= 1.5;
                    }
                }
            }
        }

        // A. Quantitative reasoning (neural QuantHead + online polarity)
        if quantities.len() >= 2 {
            return self.reason_quantities_plan(q, &quantities);
        }

        // B. Elimination by lateral/GABAergic inhibition
        if pool_observed {
            let survivors: Vec<&str> = pool_order
                .iter()
                .copied()
                .filter(|item| pool_act.get(item).copied().unwrap_or(0.0) > 0.0)
                .collect();
            return if survivors.is_empty() {
                plan(ContradictionConflict, self.contradiction_label(), removed)
            } else if survivors.len() == 1 {
                plan(Elimination, survivors[0], removed)
            } else {
                unknown()
            };
        }

        // C. Syllogism over TTFS working graph (promoting contextual attribute links when class edges exist)
        if !classes.is_empty() {
            let mut all_cls = classes;
            all_cls.extend(attr_links);
            let mut cls_graph = NeuralWorkingGraph::new();
            for &(u, v) in &all_cls {
                cls_graph.add_edge(u, v);
            }
            cls_graph.propagate();
            let ents = find_wm_entities(q, &cls_graph.nodes);
            if ents.len() == 2 {
                let (u, v) = (ents[0], ents[1]);
                if cls_graph.reachable(u, v) {
                    return plan(
                        Syllogism,
                        self.yes_label(),
                        cls_graph.trace_ttfs(u, v),
                    );
                }
            }
            return unknown();
        }

        // D. Transfer / Attribute aliasing over symmetric TTFS graph
        if !assign.is_empty() {
            let mut sym_graph = NeuralWorkingGraph::new();
            for &(a, b) in &aliases {
                sym_graph.add_edge(a, b);
                sym_graph.add_edge(b, a);
            }
            for &k in assign.keys() {
                sym_graph.ensure_node(k);
            }
            sym_graph.propagate();
            let subj_opt = q_cl
                .spans
                .iter()
                .find(|s| s.kind == SpanKind::Entity)
                .map(|s| s.text)
                .or_else(|| find_wm_entities(q, &sym_graph.nodes).into_iter().next());
            if let Some(subject) = subj_opt {
                if let Some(&val) = assign.get(subject) {
                    return plan(Transfer, val, vec![subject]);
                }
                if let Some(isubj) = sym_graph.nodes.iter().position(|&x| x == subject) {
                    let n = sym_graph.nodes.len();
                    let mut best: Option<(f32, &str, &str)> = None;
                    for (j, &cand) in sym_graph.nodes.iter().enumerate() {
                        let pot = sym_graph.ttfs[isubj * n + j];
                        if pot > 0.0 {
                            if let Some(&val) = assign.get(cand) {
                                if best.is_none_or(|(bp, _, _)| pot > bp) {
                                    best = Some((pot, cand, val));
                                }
                            }
                        }
                    }
                    if let Some((_, cand, val)) = best {
                        return plan(Transfer, val, sym_graph.trace_ttfs(subject, cand));
                    }
                }
            }
        }

        // E. Canonicalize inflected order predicates & filter non-order distractor clauses
        if raw_order.is_empty() {
            return unknown();
        }
        let prompt_preds: Vec<&str> = raw_order.iter().map(|&(_, _, p)| p).collect();
        let norm_order: Vec<(&str, &str, &str)> = raw_order
            .iter()
            .map(|&(a, b, p)| (a, b, self.canonical_order_pred(p, &prompt_preds)))
            .collect();

        // Group into predicate/antonym families and keep dominant family
        let mut fam_scores: Vec<(&str, usize, bool)> = Vec::new();
        for &(_, _, p) in &norm_order {
            if let Some(pos) = fam_scores.iter().position(|&(rep, _, _)| self.same_pred_family(rep, p)) {
                fam_scores[pos].1 += 1;
                if self.has_role_exact(p, ROLE_ORDER) {
                    fam_scores[pos].2 = true;
                }
            } else {
                fam_scores.push((p, 1, self.has_role_exact(p, ROLE_ORDER)));
            }
        }
        let Some(&(dom_rep, dom_cnt, dom_known)) = fam_scores
            .iter()
            .max_by_key(|&&(_, cnt, known)| (usize::from(known) * 100 + cnt, cnt))
        else {
            return unknown();
        };
        if !dom_known && dom_cnt < 2 {
            return unknown();
        }
        let order: Vec<(&str, &str, &str)> = norm_order
            .into_iter()
            .filter(|&(_, _, p)| self.same_pred_family(dom_rep, p))
            .collect();

        let base = order[0].2;
        let mut ord_graph = NeuralWorkingGraph::new();
        for &(a, b, r) in &order {
            if r != base && self.has_anti(base, r) {
                ord_graph.add_edge(b, a);
            } else {
                ord_graph.add_edge(a, b);
            }
        }
        ord_graph.propagate();
        let n_nodes = ord_graph.nodes.len();
        for i in 0..n_nodes {
            if ord_graph.has_cycle_at(i) {
                let node = ord_graph.nodes[i];
                return plan(
                    ContradictionCycle,
                    self.contradiction_label(),
                    ord_graph.trace_ttfs(node, node),
                );
            }
        }

        // Contrastive Hebbian SDR query inversion
        let partners = self.anti_partners(base);
        let max_anti_dot = partners
            .iter()
            .map(|&anti| contrastive_sdr_dot(q, anti, base))
            .fold(0.0f32, f32::max);
        let flipped = max_anti_dot > 0.0;

        // Pairwise comparison check using SyntaxCore Entity spans / active WM nodes
        let pair_ents = find_wm_entities(q, &ord_graph.nodes);
        if pair_ents.len() == 2 {
            let (u, v) = if flipped {
                (pair_ents[1], pair_ents[0])
            } else {
                (pair_ents[0], pair_ents[1])
            };
            if ord_graph.reachable(u, v) {
                return plan(
                    OrderPairwise,
                    self.yes_label(),
                    ord_graph.trace_ttfs(u, v),
                );
            }
            if ord_graph.reachable(v, u) {
                return plan(
                    OrderPairwise,
                    self.no_label(),
                    ord_graph.trace_ttfs(v, u),
                );
            }
        }

        // Polarity & Rank decoding (neural OrderHead + online 1-shot E/I channel)
        let mut cleaned_q = q
            .trim()
            .trim_end_matches(|c: char| !c.is_alphanumeric() && !('\u{3040}'..='\u{9FFF}').contains(&c))
            .to_owned();
        for &anti in &partners {
            cleaned_q = cleaned_q.replace(anti, " ");
        }
        cleaned_q = cleaned_q.replace(base, " ");
        let online_pol = self.online_order_polarity(&cleaned_q, base);

        let (base_sink, rank) = if self.neural_trained {
            let (ord_cls, _) = self.predict_order(&cleaned_q);
            match ord_cls {
                2 => (false, 2usize),
                3 => (false, 3usize),
                4 => (false, 4usize),
                _ => {
                    if online_pol.abs() > 0.35 {
                        (online_pol < -0.35, 1usize)
                    } else {
                        (ord_cls == 1, 1usize)
                    }
                }
            }
        } else {
            let r = q
                .chars()
                .find_map(|c| c.to_digit(10))
                .filter(|&d| d >= 2)
                .unwrap_or(1) as usize;
            (online_pol < -0.35, r)
        };

        let sink = flipped ^ base_sink;
        if rank > 1 {
            let mut sorted: Vec<(usize, &str)> = (0..n_nodes)
                .map(|i| (ord_graph.dom_count(i), ord_graph.nodes[i]))
                .collect();
            sorted.sort_unstable();
            if !sink {
                sorted.reverse();
            }
            if rank <= sorted.len() && sorted.windows(2).all(|p| p[0].0 != p[1].0) {
                return plan(
                    OrderRank,
                    sorted[rank - 1].1,
                    sorted.iter().map(|r| r.1).collect(),
                );
            }
            return unknown();
        }

        let candidates: Vec<&str> = (0..n_nodes)
            .filter(|&i| {
                if sink {
                    ord_graph.out_deg(i) == 0
                } else {
                    ord_graph.in_deg(i) == 0
                }
            })
            .map(|i| ord_graph.nodes[i])
            .collect();
        if candidates.len() == 1 {
            let endpoint = candidates[0];
            let route = ord_graph
                .nodes
                .iter()
                .map(|&other| {
                    if sink {
                        ord_graph.trace_ttfs(other, endpoint)
                    } else {
                        ord_graph.trace_ttfs(endpoint, other)
                    }
                })
                .max_by_key(Vec::len)
                .unwrap_or_default();
            plan(OrderExtreme, endpoint, route)
        } else {
            unknown()
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&[u8::from(self.neural_trained)])?;
        self.syntax.write_to(&mut w)?;
        w.write_all(bytemuck::cast_slice(&self.w1))?;
        w.write_all(bytemuck::cast_slice(&self.theta))?;
        w.write_all(bytemuck::cast_slice(&self.w2_quant))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir_quant))?;
        w.write_all(bytemuck::cast_slice(&self.b_quant))?;
        w.write_all(bytemuck::cast_slice(&self.w2_order))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir_order))?;
        w.write_all(bytemuck::cast_slice(&self.b_order))?;
        w.write_all(bytemuck::cast_slice(&self.w2_prem))?;
        w.write_all(bytemuck::cast_slice(&self.w_dir_prem))?;
        w.write_all(bytemuck::cast_slice(&self.b_prem))?;

        let write_str = |wr: &mut BufWriter<File>, s: &str| -> Result<()> {
            let b = s.as_bytes();
            wr.write_all(&(b.len() as u32).to_le_bytes())?;
            wr.write_all(b)?;
            Ok(())
        };
        for l in &self.labels {
            write_str(&mut w, l)?;
        }
        w.write_all(&(self.preds.len() as u32).to_le_bytes())?;
        for id in 0..self.preds.len() {
            write_str(&mut w, &self.preds[id])?;
            w.write_all(bytemuck::cast_slice(&self.role_w[id]))?;
            w.write_all(bytemuck::cast_slice(&self.pol_order_ei[id]))?;
            w.write_all(bytemuck::cast_slice(&self.pol_quant_ei[id]))?;
        }
        w.write_all(&(self.anti_edges.len() as u32).to_le_bytes())?;
        for &(u, v, wt) in &self.anti_edges {
            w.write_all(&u.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            w.write_all(&wt.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid ReasonerCore checkpoint magic");
        let mut flag = [0u8; 1];
        r.read_exact(&mut flag)?;

        let mut core = Self::blank();
        core.neural_trained = flag[0] != 0;
        core.syntax = SyntaxCore::read_from(&mut r)?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w1))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.theta))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w2_quant))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w_dir_quant))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.b_quant))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w2_order))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w_dir_order))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.b_order))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w2_prem))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.w_dir_prem))?;
        r.read_exact(bytemuck::cast_slice_mut(&mut core.b_prem))?;

        let read_str = |rd: &mut BufReader<File>| -> Result<String> {
            let mut lb = [0u8; 4];
            rd.read_exact(&mut lb)?;
            let len = u32::from_le_bytes(lb) as usize;
            let mut buf = vec![0u8; len];
            rd.read_exact(&mut buf)?;
            Ok(String::from_utf8(buf)?)
        };
        let read_u32 = |rd: &mut BufReader<File>| -> Result<u32> {
            let mut b = [0u8; 4];
            rd.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        for l in &mut core.labels {
            *l = read_str(&mut r)?;
        }
        let n_preds = read_u32(&mut r)? as usize;
        core.preds.reserve(n_preds);
        core.role_w.reserve(n_preds);
        core.pol_order_ei.reserve(n_preds);
        core.pol_quant_ei.reserve(n_preds);
        for _ in 0..n_preds {
            let s = read_str(&mut r)?;
            let mut rw = [0.0f32; NUM_ROLES];
            let mut po = [0.0f32; 2];
            let mut pq = [0.0f32; 2];
            r.read_exact(bytemuck::cast_slice_mut(&mut rw))?;
            r.read_exact(bytemuck::cast_slice_mut(&mut po))?;
            r.read_exact(bytemuck::cast_slice_mut(&mut pq))?;
            core.preds.push(s);
            core.role_w.push(rw);
            core.pol_order_ei.push(po);
            core.pol_quant_ei.push(pq);
        }
        let n_anti = read_u32(&mut r)? as usize;
        core.anti_edges.reserve(n_anti);
        for _ in 0..n_anti {
            let u = read_u32(&mut r)?;
            let v = read_u32(&mut r)?;
            let mut fb = [0u8; 4];
            r.read_exact(&mut fb)?;
            core.anti_edges.push((u, v, f32::from_le_bytes(fb)));
        }
        Ok(core)
    }
}

#[derive(Deserialize)]
struct HeadRow {
    task: String,
    text: String,
    label: usize,
    split: String,
}

#[derive(Deserialize)]
struct UnseenProbe {
    prompt: String,
    expected_mode: ReasonMode,
    expected_answer: String,
    #[serde(default)]
    unlabeled: Option<String>,
}

#[derive(Deserialize)]
struct BenchCase {
    prompt: String,
    gold: String,
    #[serde(default)]
    turns: Vec<String>,
}

#[derive(Deserialize)]
struct BenchFixture {
    unlabeled: Vec<String>,
    episodes: Vec<(String, String)>,
    cases: Vec<BenchCase>,
}

fn train_head<const C: usize>(
    core: &ReasonerCore,
    train_items: &[(Vec<usize>, [f32; D_IN], usize)],
    val_items: &[(Vec<usize>, [f32; D_IN], usize)],
    epochs: usize,
    w2: &mut [f32],
    w_dir: &mut [f32],
    bias: &mut [f32; C],
) -> f64 {
    let mut cnt = [1usize; C];
    for &(_, _, y) in train_items {
        cnt[y] += 1;
    }
    let max_c = *cnt.iter().max().unwrap_or(&1) as f32;
    let mut cwt = [1.0f32; C];
    for c in 0..C {
        cwt[c] = (max_c / cnt[c] as f32).sqrt().min(3.0);
    }

    for epoch in 0..epochs.max(1) {
        let lr = 0.25 / (1.0 + 0.25 * epoch as f32);
        for (feats, x, y) in train_items {
            let winners = core.hidden_act(x);
            let inv_f = 1.0 / (feats.len().max(1) as f32).sqrt();
            let wt = cwt[*y];
            let (_, logits) = core.eval_head(feats, &winners, w2, w_dir, bias);
            let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = [0.0f32; C];
            let mut sum_e = 0.0f32;
            for c in 0..C {
                probs[c] = (logits[c] - max_l).exp();
                sum_e += probs[c];
            }
            for c in 0..C {
                probs[c] /= sum_e + 1e-12;
                let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                bias[c] += lr * wt * 0.05 * delta;
            }
            for &(h, act) in &winners {
                if act > 0.0 {
                    let base = h * C;
                    for c in 0..C {
                        let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                        w2[base + c] += lr * wt * 0.35 * act * delta;
                    }
                }
            }
            for &f in feats {
                let base = f * C;
                for c in 0..C {
                    let delta = if c == *y { 1.0 - probs[c] } else { -probs[c] };
                    w_dir[base + c] += lr * wt * inv_f * delta;
                }
            }
        }
    }

    let mut ok = 0usize;
    for (feats, x, y) in val_items {
        let winners = core.hidden_act(x);
        let (pred, _) = core.eval_head(feats, &winners, w2, w_dir, bias);
        ok += usize::from(pred == *y);
    }
    ok as f64 / val_items.len().max(1) as f64
}

/// v24.5 関係推論野のオフライン学習（Oja-Hebbian + 局所デルタ + オンライン固定化 + SyntaxCore統合）と全自動評価。
pub fn train_and_bench(
    root: &Path,
    corpus_dir: &Path,
    ckpt_path: &Path,
    out_json: &Path,
    epochs: usize,
) -> Result<Value> {
    let t0 = Instant::now();
    let head_lines = fs::read_to_string(corpus_dir.join("neural_heads.jsonl"))?;
    let mut quant_tr = Vec::new();
    let mut quant_va = Vec::new();
    let mut order_tr = Vec::new();
    let mut order_va = Vec::new();
    let mut prem_tr = Vec::new();
    let mut prem_va = Vec::new();
    let mut all_x = Vec::new();

    for line in head_lines.lines().filter(|l| !l.trim().is_empty()) {
        let row: HeadRow = serde_json::from_str(line)?;
        let (f, x) = extract_neural_features(&row.text);
        let is_val = row.split == "val";
        if !is_val {
            all_x.push(x);
        }
        match row.task.as_str() {
            "quant" => {
                if is_val {
                    quant_va.push((f, x, row.label));
                } else {
                    quant_tr.push((f, x, row.label));
                }
            }
            "order" => {
                if is_val {
                    order_va.push((f, x, row.label));
                } else {
                    order_tr.push((f, x, row.label));
                }
            }
            "prem" => {
                if is_val {
                    prem_va.push((f, x, row.label));
                } else {
                    prem_tr.push((f, x, row.label));
                }
            }
            _ => {}
        }
    }

    let mut core = ReasonerCore::blank();
    let sx1 = root.join("v245_syntax_core/syntax_core_v245.bin");
    let sx2 = root.join("rust_bench_v182/v245/syntax_core_v245.bin");
    let sx_path = if sx1.exists() { sx1 } else { sx2 };
    ensure!(sx_path.exists(), "missing syntax_core_v245.bin at {}", sx_path.display());
    core.syntax = SyntaxCore::load(&sx_path)?;

    // Phase 1: Unsupervised competitive Oja-Hebbian learning on Layer 1
    let target_rate = TOP_K as f32 / HIDDEN as f32;
    for pass in 0..3 {
        let eta = 0.06 / (1.0 + 0.4 * pass as f32);
        for x in &all_x {
            let winners = core.hidden_act(x);
            let mut won = [false; HIDDEN];
            for &(h, act) in &winners {
                if act <= 0.0 {
                    continue;
                }
                won[h] = true;
                let row = &mut core.w1[h * D_IN..(h + 1) * D_IN];
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
            for h in 0..HIDDEN {
                let fired = if won[h] { 1.0f32 } else { 0.0f32 };
                core.theta[h] = (core.theta[h] + 0.002 * (fired - target_rate)).clamp(-0.35, 0.35);
            }
        }
    }

    // Phase 2: Local Hebbian delta readout heads
    let mut w2_q = std::mem::take(&mut core.w2_quant);
    let mut wd_q = std::mem::take(&mut core.w_dir_quant);
    let mut b_q = core.b_quant;
    let val_quant_acc = train_head::<NUM_QUANT>(&core, &quant_tr, &quant_va, epochs, &mut w2_q, &mut wd_q, &mut b_q);
    core.w2_quant = w2_q;
    core.w_dir_quant = wd_q;
    core.b_quant = b_q;

    let mut w2_o = std::mem::take(&mut core.w2_order);
    let mut wd_o = std::mem::take(&mut core.w_dir_order);
    let mut b_o = core.b_order;
    let val_order_acc = train_head::<NUM_ORDER>(&core, &order_tr, &order_va, epochs, &mut w2_o, &mut wd_o, &mut b_o);
    core.w2_order = w2_o;
    core.w_dir_order = wd_o;
    core.b_order = b_o;

    let mut w2_p = std::mem::take(&mut core.w2_prem);
    let mut wd_p = std::mem::take(&mut core.w_dir_prem);
    let mut b_p = core.b_prem;
    let val_prem_acc = train_head::<NUM_PREM>(&core, &prem_tr, &prem_va, epochs, &mut w2_p, &mut wd_p, &mut b_p);
    core.w2_prem = w2_p;
    core.w_dir_prem = wd_p;
    core.b_prem = b_p;

    // Phase 3: Consolidate relational plasticity from official bench_320 unlabeled + episodes
    let unlabeled: Vec<String> =
        serde_json::from_slice(&fs::read(corpus_dir.join("unlabeled.json"))?)?;
    let episodes: Vec<(String, String)> =
        serde_json::from_slice(&fs::read(corpus_dir.join("episodes.json"))?)?;
    for doc in &unlabeled {
        core.observe_unlabeled(doc);
    }
    for (q, a) in &episodes {
        core.learn_episode(q, a);
    }
    core.neural_trained = true;
    core.save(ckpt_path)?;
    let train_elapsed_s = t0.elapsed().as_secs_f64();

    // Reload checkpoint & measure latency/size/accuracy
    let tl = Instant::now();
    let mut loaded = ReasonerCore::load(ckpt_path)?;
    let load_ms = tl.elapsed().as_secs_f64() * 1000.0;
    let ckpt_mib = fs::metadata(ckpt_path)?.len() as f64 / (1024.0 * 1024.0);

    let fixture: BenchFixture =
        serde_json::from_slice(&fs::read(root.join("rust_bench_v182/bench_320.json"))?)?;
    let mut bench_ok = 0usize;
    let mut bench_fails = Vec::new();
    let mut reason_times_us = Vec::new();
    for (i, c) in fixture.cases.iter().enumerate() {
        loaded.apoptosis();
        for t in &c.turns {
            loaded.observe_turn(t);
        }
        let ts = Instant::now();
        let pred = loaded.answer(&c.prompt);
        reason_times_us.push(ts.elapsed().as_secs_f64() * 1_000_000.0);
        if pred == c.gold {
            bench_ok += 1;
        } else {
            bench_fails.push(json!({"idx": i, "prompt": c.prompt, "gold": c.gold, "pred": pred}));
        }
    }
    loaded.apoptosis();

    // Blank scratch refit check (paired with SyntaxCore)
    let mut refit_core = ReasonerCore::from_blank_seed_json_with_syntax(
        &fs::read(root.join("rust_bench_v182/blank_synapses.json"))?,
        SyntaxCore::load(&sx_path)?,
    )?;
    for doc in &fixture.unlabeled {
        refit_core.observe_unlabeled(doc);
    }
    for (q, a) in &fixture.episodes {
        refit_core.learn_episode(q, a);
    }
    let mut refit_ok = 0usize;
    let mut refit_fails = Vec::new();
    for (i, c) in fixture.cases.iter().enumerate() {
        refit_core.apoptosis();
        for t in &c.turns {
            refit_core.observe_turn(t);
        }
        let pred = refit_core.answer(&c.prompt);
        if pred == c.gold {
            refit_ok += 1;
        } else {
            refit_fails.push(json!({"idx": i, "prompt": c.prompt, "gold": c.gold, "pred": pred}));
        }
    }

    // Held-out v240 unseen paraphrase probes
    let probes: Vec<UnseenProbe> =
        serde_json::from_slice(&fs::read(corpus_dir.join("unseen_probes.json"))?)?;
    let mut probes_ok = 0usize;
    let mut probe_rows = Vec::new();
    for p in &probes {
        let plan = loaded.reason_plan(&p.prompt);
        let ans = String::from_utf8_lossy(&plan.answer_bytes).into_owned();
        let ok = plan.mode == p.expected_mode && ans == p.expected_answer;
        if ok {
            probes_ok += 1;
        }
        probe_rows.push(json!({
            "prompt": p.prompt,
            "expected_mode": p.expected_mode,
            "pred_mode": plan.mode,
            "expected_answer": p.expected_answer,
            "pred_answer": ans,
            "ok": ok
        }));
    }

    // Held-out v245 unseen syntax probes (comma/multi-clause/alien)
    let sx_probe_path = root.join("v245_syntax_core/unseen_syntax_probes.json");
    let sx_probes: Vec<UnseenProbe> = if sx_probe_path.exists() {
        serde_json::from_slice(&fs::read(&sx_probe_path)?)?
    } else {
        vec![]
    };
    let mut sx_ok = 0usize;
    let mut sx_rows = Vec::new();
    for p in &sx_probes {
        let plan = if let Some(ul) = &p.unlabeled {
            let mut cloned = ReasonerCore::load(ckpt_path)?;
            cloned.observe_unlabeled(ul);
            cloned.reason_plan(&p.prompt)
        } else {
            loaded.reason_plan(&p.prompt)
        };
        let ans = String::from_utf8_lossy(&plan.answer_bytes).into_owned();
        let ok = plan.mode == p.expected_mode && ans == p.expected_answer;
        if ok {
            sx_ok += 1;
        }
        sx_rows.push(json!({
            "prompt": p.prompt,
            "expected_mode": p.expected_mode,
            "pred_mode": plan.mode,
            "expected_answer": p.expected_answer,
            "pred_answer": ans,
            "ok": ok
        }));
    }

    let avg_reason_us =
        reason_times_us.iter().sum::<f64>() / reason_times_us.len().max(1) as f64;
    let (rss, peak) = crate::memory::working_set();
    let report = json!({
        "ckpt_mib": ckpt_mib,
        "load_ms": load_ms,
        "train_elapsed_s": train_elapsed_s,
        "avg_reason_us": avg_reason_us,
        "rss_mib": rss,
        "peak_rss_mib": peak,
        "n_preds": loaded.preds.len(),
        "n_anti_edges": loaded.anti_edges.len(),
        "val_quant_acc": val_quant_acc,
        "val_order_acc": val_order_acc,
        "val_prem_acc": val_prem_acc,
        "bench_320_ok": format!("{bench_ok}/{}", fixture.cases.len()),
        "refit_320_ok": format!("{refit_ok}/{}", fixture.cases.len()),
        "unseen_probes_ok": format!("{probes_ok}/{}", probes.len()),
        "unseen_syntax_probes_ok": format!("{sx_ok}/{}", sx_probes.len()),
        "bench_fails": bench_fails,
        "refit_fails": refit_fails,
        "probes": probe_rows,
        "syntax_probes": sx_rows,
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
    use crate::syntax_core::SyntaxRow;

    fn test_syntax_core() -> SyntaxCore {
        let rows: Vec<SyntaxRow> = [
            ("P1|P2:relA;P2|P1:relB;", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,2,3,1,8,9,9,9,0], "train"),
            ("P1|P2:relA;P2|P3:relA;revQ?", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,2,3,1,8,9,9,9,0,4,5,5,5,1], "train"),
            ("E1|E2:relA;E3|E2:relB;relBQ?", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,2,3,1,8,9,9,9,0,8,9,9,9,9,1], "train"),
            ("E1|E2:relA;E2|E3:relA;revQ?", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,2,3,1,8,9,9,9,0,4,5,5,5,1], "train"),
            ("A1|A2:relA;A2|A3:relB;relBQ?", vec![2,3,1,2,3,1,8,9,9,9,0,2,3,1,2,3,1,8,9,9,9,0,8,9,9,9,9,1], "val"),
        ]
        .into_iter()
        .map(|(text, tags, split)| SyntaxRow {
            text: text.to_owned(),
            tags,
            split: split.to_owned(),
        })
        .collect();
        let mut sx = SyntaxCore::blank();
        sx.train_from_rows(&rows, 10);
        sx
    }

    #[test]
    fn ttfs_graph_contrastive_antonym_and_roundtrip_work() {
        let mut core = ReasonerCore::blank();
        core.syntax = test_syntax_core();
        core.observe_unlabeled("P1|P2:relA;P2|P1:relB;");
        core.learn_episode("P1|P2:relA;P2|P3:relA;revQ?", "P3");
        let ans = core.answer("E1|E2:relA;E3|E2:relB;relBQ?");
        assert_eq!(ans, "E3");
        let ans2 = core.answer("E1|E2:relA;E2|E3:relA;revQ?");
        assert_eq!(ans2, "E3");

        let mut cyc = NeuralWorkingGraph::new();
        cyc.add_edge("N1", "N2");
        cyc.add_edge("N2", "N3");
        cyc.add_edge("N3", "N1");
        cyc.propagate();
        assert!(cyc.has_cycle_at(0));
        assert_eq!(cyc.trace_ttfs("N1", "N1"), vec!["N1", "N2", "N3", "N1"]);

        let tmp = std::env::temp_dir().join(format!("ng-rc245-{}.bin", std::process::id()));
        core.save(&tmp).unwrap();
        let reloaded = ReasonerCore::load(&tmp).unwrap();
        std::fs::remove_file(tmp).unwrap();
        assert_eq!(
            reloaded.answer("E1|E2:relA;E3|E2:relB;relBQ?"),
            "E3"
        );
    }
}
