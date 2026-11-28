//! ENNE v23.0 疎分散連想記憶・海馬コア (`HippocampusCore`, `NGHC2301`)。
//!
//! 設計仕様:
//! - 推論時はオフライン封印済みの `.bin` のみを参照し、SQLite・質問剥離正規表現・単位正規表現を使わない。
//! - 嗅内野・歯状回 (DG, 1,048,576細胞 (2^20)): UTF-8 文字 2/3/4-gram + 教師なし恒常性抑圧 (`dg_gain`)。
//! - CA3 スパース連想検索 (DG→CA3 BF16 CSR): 非対称ホップフィールド共鳴による記憶想起 + オンライン1回書込 (`teach`)。
//! - 側方抑制 NMS + 局所ヘブ読み出し頭 (`W_cmp_query`, `W_num_ctx`, `W_unit`) による数値比較・測定値抽出。
use crate::core_dag::mix;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Mutex,
    time::Instant,
};
use unicode_normalization::UnicodeNormalization;

pub const MAGIC: &[u8; 8] = b"NGHC2301";
pub const DG_CELLS: usize = 1_048_576; // 2^20 Dentate Gyrus cells for minimal bucket collision
pub const TOP_K_SYNAPSES: usize = 6;
pub const MAX_POSTING_DF: usize = 2_048;
pub const NUM_CTX_BUCKETS: usize = 16_384;
pub const CMP_BUCKETS: usize = 16_384;
pub const UNIT_BUCKETS: usize = 4_096;
pub const NUM_UNITS: usize = 4; // 0=Meter, 1=Kilometer, 2=Centimeter, 3=Other
pub const BODY_MAX_BYTES: usize = 512;

pub const KIND_EXACT: u8 = 0;
pub const KIND_PREDICATE: u8 = 1;
pub const KIND_ALIAS: u8 = 2;

#[derive(Clone, Debug)]
pub struct RecalledFact {
    pub head: String,
    pub body: String,
    pub source: &'static str,
    pub score: f64,
}

#[derive(Clone)]
struct TaughtEngram {
    head_norm: String,
    head: String,
    body: String,
    hashes: Vec<u64>,
    norm2: f32,
}

#[derive(Clone, Debug)]
struct NumericSpan {
    value: f64,
    clean: String,
    start: usize,
    end: usize,
}

#[derive(Clone)]
struct RawSlotCandidate {
    lead_id: u32,
    span_start: usize,
    span_len: usize,
    is_taught: bool,
    taught_body: String,
    cue_str: String,
    head_str: String,
    resonance: f32,
    prio_q: u8,
}

#[inline(always)]
fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[inline(always)]
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// 比較・想起用の NFKC 小文字文字列へ正規化する。
pub fn norm_text(s: &str) -> String {
    s.trim().nfkc().collect::<String>().to_lowercase()
}

/// 歯状回 (DG) 顆粒細胞エンコーディング: 衝突耐性のある 64-bit 文字 2/3/4-gram（1文字時は1-gram）を返す。日本語ストップワード表ゼロ。
pub fn extract_dg_hashes64(raw: &str) -> Vec<u64> {
    let n = norm_text(raw);
    let chars: Vec<char> = n
        .chars()
        .filter(|c| !c.is_whitespace() && !"。、！？!?…「」『』（）()・,.\"'".contains(*c))
        .collect();
    let mut hashes = Vec::with_capacity(chars.len() * 3 + 1);
    if chars.len() == 1 {
        hashes.push(mix(chars[0] as u64 ^ 0x101));
        return hashes;
    }
    for i in 0..chars.len() {
        let c0 = chars[i] as u64;
        if i + 1 < chars.len() {
            let c1 = chars[i + 1] as u64;
            hashes.push(mix((c0 << 21) ^ c1 ^ 0x202));
            if i + 2 < chars.len() {
                let c2 = chars[i + 2] as u64;
                hashes.push(mix((c0 << 42) ^ (c1 << 21) ^ c2 ^ 0x303));
                if i + 3 < chars.len() {
                    let c3 = chars[i + 3] as u64;
                    hashes.push(mix(mix((c0 << 21) ^ c1) ^ ((c2 << 21) ^ c3) ^ 0x404));
                }
            }
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

#[inline(always)]
pub fn dg_cell(h64: u64) -> usize {
    (h64 as usize) & (DG_CELLS - 1)
}

/// 歯状回セル添字列 (0..DG_CELLS) を返す。
pub fn extract_dg_grams(raw: &str) -> Vec<usize> {
    let mut grams: Vec<usize> = extract_dg_hashes64(raw).into_iter().map(dg_cell).collect();
    grams.sort_unstable();
    grams.dedup();
    grams
}

/// 文字列内の数値スパンを言語非依存に走査する（`3,180` / `3776.12` 対応）。
fn find_numeric_spans(text_nfkc: &str) -> Vec<NumericSpan> {
    let chars: Vec<char> = text_nfkc.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut clean = String::new();
            let mut has_dot = false;
            while i < chars.len() {
                let ch = chars[i];
                if ch.is_ascii_digit() {
                    clean.push(ch);
                    i += 1;
                } else if ch == ',' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                } else if ch == '.' && !has_dot && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()
                {
                    has_dot = true;
                    clean.push(ch);
                    i += 1;
                } else {
                    break;
                }
            }
            if let Ok(value) = clean.parse::<f64>() {
                out.push(NumericSpan {
                    value,
                    clean,
                    start,
                    end: i,
                });
            }
        } else {
            i += 1;
        }
    }
    out
}

fn push_ngram_hashes(
    feats: &mut Vec<usize>,
    chars: &[char],
    salt: u64,
    mask: usize,
) {
    for i in 0..chars.len() {
        let c0 = chars[i] as u64;
        feats.push((mix(c0 ^ salt) as usize) & mask);
        if i + 1 < chars.len() {
            let c1 = chars[i + 1] as u64;
            feats.push((mix((c0 << 21) ^ c1 ^ (salt + 1)) as usize) & mask);
            if i + 2 < chars.len() {
                let c2 = chars[i + 2] as u64;
                feats.push((mix((c0 << 42) ^ (c1 << 21) ^ c2 ^ (salt + 2)) as usize) & mask);
            }
        }
    }
}

/// 数値スパンの文脈特徴（左12文字・右8文字・質問文脈重なり・桁構造）を抽出する（語彙ハードコードゼロ）。
fn extract_num_ctx_features(body_chars: &[char], span: &NumericSpan, query: &str) -> Vec<usize> {
    let mask = NUM_CTX_BUCKETS - 1;
    let mut feats = Vec::with_capacity(64);
    let l_start = span.start.saturating_sub(12);
    let left: Vec<char> = body_chars[l_start..span.start]
        .iter()
        .copied()
        .filter(|c| !c.is_whitespace())
        .collect();
    push_ngram_hashes(&mut feats, &left, 0xA100, mask);
    // Closest 3 left chars with relative offset tags
    for off in 1..=3.min(left.len()) {
        let c = left[left.len() - off] as u64;
        feats.push((mix((off as u64).wrapping_mul(0x9e377) ^ c ^ 0xA150) as usize) & mask);
    }

    // Skip spaces on right before extracting unit/suffix window
    let mut r_start = span.end;
    while r_start < body_chars.len() && body_chars[r_start].is_whitespace() {
        r_start += 1;
    }
    let r_end = (r_start + 8).min(body_chars.len());
    let right: Vec<char> = body_chars[r_start..r_end]
        .iter()
        .copied()
        .filter(|c| !c.is_whitespace())
        .collect();
    push_ngram_hashes(&mut feats, &right, 0xB200, mask);
    if !right.is_empty() {
        let r0 = right[0] as u64;
        feats.push((mix(r0 ^ 0xB210) as usize) & mask);
        if right.len() >= 2 {
            let r1 = right[1] as u64;
            feats.push((mix((r0 << 21) ^ r1 ^ 0xB220) as usize) & mask);
        }
    }

    // Structural cues of span itself
    let has_dot = span.clean.contains('.');
    let mag_bucket = if span.value < 10.0 {
        0u64
    } else if span.value < 100.0 {
        1
    } else if span.value < 1000.0 {
        2
    } else if span.value < 10000.0 {
        3
    } else {
        4
    };
    feats.push((mix((has_dot as u64) ^ 0xC301) as usize) & mask);
    feats.push((mix(mag_bucket ^ 0xC302) as usize) & mask);

    // Language-agnostic query-context overlap (e.g., matching left/right property cues in query)
    let q_clean: Vec<char> = norm_text(query)
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_digit())
        .collect();
    if q_clean.len() >= 2 && left.len() >= 2 {
        let mut q2 = Vec::with_capacity(q_clean.len());
        for w in q_clean.windows(2) {
            q2.push((w[0], w[1]));
        }
        let mut ov = 0usize;
        for w in left.windows(2) {
            if q2.contains(&(w[0], w[1])) {
                ov += 1;
            }
        }
        for w in right.windows(2) {
            if q2.contains(&(w[0], w[1])) {
                ov += 1;
            }
        }
        feats.push((mix((ov.min(4) as u64) ^ 0xD400) as usize) & mask);
        if ov > 0 {
            feats.push((mix(0xD4FF) as usize) & mask);
        }
    }

    feats.sort_unstable();
    feats.dedup();
    feats
}

/// 数値右直後の単位トークン特徴を抽出する (`W_unit`)。
fn extract_unit_features(body_chars: &[char], span_end: usize) -> Vec<usize> {
    let mask = UNIT_BUCKETS - 1;
    let mut r_start = span_end;
    while r_start < body_chars.len() && body_chars[r_start].is_whitespace() {
        r_start += 1;
    }
    let r_end = (r_start + 8).min(body_chars.len());
    let right: Vec<char> = body_chars[r_start..r_end]
        .iter()
        .map(|c| c.to_ascii_lowercase())
        .filter(|c| !c.is_whitespace())
        .collect();
    let mut feats = Vec::with_capacity(24);
    for len in 1..=3.min(right.len()) {
        let mut h = 0xE500u64 ^ (len as u64);
        for &c in &right[..len] {
            h = mix(h ^ (c as u64));
        }
        feats.push((h as usize) & mask);
    }
    push_ngram_hashes(&mut feats, &right, 0xE550, mask);
    feats.sort_unstable();
    feats.dedup();
    feats
}

/// 比較/数値差問い合わせ特徴を抽出する (`W_cmp_query`)。
fn extract_cmp_query_features(q: &str) -> Vec<usize> {
    let n = norm_text(q);
    let chars: Vec<char> = n.chars().filter(|c| !c.is_whitespace()).collect();
    let mask = CMP_BUCKETS - 1;
    let mut feats = Vec::with_capacity(chars.len() * 3 + 4);
    push_ngram_hashes(&mut feats, &chars, 0xF100, mask);
    let has_q = n.ends_with(['?', '？']);
    let len_b = (chars.len() / 8).min(4) as u64;
    feats.push((mix((has_q as u64) ^ 0xF201) as usize) & mask);
    feats.push((mix(len_b ^ 0xF202) as usize) & mask);
    feats.sort_unstable();
    feats.dedup();
    feats
}

pub struct HippocampusCore {
    pub n_leads: usize,
    pub n_slots: usize,
    pub dg_gain: Vec<f32>,
    pub csr_off: Vec<u32>,
    pub csr_slot: Vec<u32>,
    pub csr_w: Vec<u16>,
    pub slot_lead_id: Vec<u32>,
    pub slot_cue_off: Vec<u32>,
    pub slot_cue_len: Vec<u16>,
    pub slot_kind: Vec<u8>,
    pub slot_norm2: Vec<f32>,
    pub cue_arena: Vec<u8>,
    pub lead_head_off: Vec<u32>,
    pub lead_head_len: Vec<u16>,
    pub lead_prio_q: Vec<u8>,
    pub lead_body_off: Vec<u64>,
    pub lead_body_len: Vec<u16>,
    pub heads_arena: Vec<u8>,
    pub w_cmp_query: Vec<f32>,
    pub b_cmp_query: [f32; 2],
    pub w_num_ctx: Vec<f32>,
    pub b_num_ctx: [f32; 2],
    pub w_unit: Vec<f32>,
    pub b_unit: [f32; NUM_UNITS],
    body_file: Option<Mutex<File>>,
    mem_bodies: Vec<String>,
    taught: Vec<TaughtEngram>,
}

impl HippocampusCore {
    fn read_body(&self, lead_id: usize) -> Result<String> {
        if !self.mem_bodies.is_empty() {
            return Ok(self.mem_bodies[lead_id].clone());
        }
        let f_mtx = self
            .body_file
            .as_ref()
            .context("hippocampus body storage not attached")?;
        let off = self.lead_body_off[lead_id];
        let len = self.lead_body_len[lead_id] as usize;
        let mut buf = vec![0u8; len];
        let mut f = f_mtx.lock().unwrap();
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf).unwrap_or_default())
    }

    #[inline(always)]
    fn slot_cue_str(&self, slot: usize) -> &str {
        let off = self.slot_cue_off[slot] as usize;
        let len = self.slot_cue_len[slot] as usize;
        std::str::from_utf8(&self.cue_arena[off..off + len]).unwrap_or("")
    }

    #[inline(always)]
    fn lead_head_str(&self, lead_id: usize) -> &str {
        let off = self.lead_head_off[lead_id] as usize;
        let len = self.lead_head_len[lead_id] as usize;
        std::str::from_utf8(&self.heads_arena[off..off + len]).unwrap_or("")
    }

    /// オンライン1回書込の高速可塑シナプス (`taught`) に新規エピソードを保持する。
    pub fn teach(&mut self, subject: &str, body: &str) -> Result<()> {
        let key = norm_text(subject);
        ensure!(!key.is_empty() && !body.trim().is_empty());
        let hashes = extract_dg_hashes64(&key);
        let norm2: f32 = hashes
            .iter()
            .map(|&h| {
                let w = self.dg_gain[dg_cell(h)].max(0.5);
                w * w
            })
            .sum();
        let trimmed = body.trim().to_owned();
        if let Some(row) = self.taught.iter_mut().find(|r| r.head_norm == key) {
            row.head = subject.trim().to_owned();
            row.body = trimmed;
            row.hashes = hashes;
            row.norm2 = norm2;
        } else {
            self.taught.push(TaughtEngram {
                head_norm: key,
                head: subject.trim().to_owned(),
                body: trimmed,
                hashes,
                norm2,
            });
        }
        Ok(())
    }

    /// 比較/差異クエリかどうかをヘブ分類器 `W_cmp_query` で予測する。
    pub fn predict_is_compare(&self, q: &str) -> bool {
        let feats = extract_cmp_query_features(q);
        let mut s = self.b_cmp_query;
        let inv = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * 2;
            s[0] += inv * self.w_cmp_query[base];
            s[1] += inv * self.w_cmp_query[base + 1];
        }
        s[1] > s[0]
    }

    /// 数値スパンが物理測定値である対数オッズ (`W_num_ctx`) を計算する。
    fn score_num_span(&self, body_chars: &[char], span: &NumericSpan, query: &str) -> f32 {
        let feats = extract_num_ctx_features(body_chars, span, query);
        let mut s = self.b_num_ctx;
        let inv = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * 2;
            s[0] += inv * self.w_num_ctx[base];
            s[1] += inv * self.w_num_ctx[base + 1];
        }
        s[1] - s[0]
    }

    /// 数値右直後の単位 (`0=Meter, 1=Kilometer, 2=Centimeter, 3=Other`) を `W_unit` で分類する。
    fn predict_unit(&self, body_chars: &[char], span_end: usize) -> usize {
        let feats = extract_unit_features(body_chars, span_end);
        let mut s = self.b_unit;
        let inv = 1.0 / (feats.len().max(1) as f32).sqrt();
        for &f in &feats {
            let base = f * NUM_UNITS;
            for c in 0..NUM_UNITS {
                s[c] += inv * self.w_unit[base + c];
            }
        }
        let mut best = 0;
        for c in 1..NUM_UNITS {
            if s[c] > s[best] {
                best = c;
            }
        }
        best
    }

    /// 本文中から最も確信度の高い物理測定値と単位を純ヘブ NN で抽出する。
    fn extract_neural_measure(&self, query: &str, body: &str) -> Option<(String, usize)> {
        let b_nfkc: String = body.trim().nfkc().collect();
        let b_chars: Vec<char> = b_nfkc.chars().collect();
        let spans = find_numeric_spans(&b_nfkc);
        let mut best: Option<(f32, String, usize)> = None;
        for sp in &spans {
            let margin = self.score_num_span(&b_chars, sp, query);
            if margin > 0.25 {
                let unit = self.predict_unit(&b_chars, sp.end);
                if unit < 3 {
                    if best.as_ref().is_none_or(|b| margin > b.0) {
                        best = Some((margin, sp.clean.clone(), unit));
                    }
                }
            }
        }
        best.map(|(_, val, unit)| (val, unit))
    }

    /// DG→CA3 スパース発火により候補スロットとそのホップフィールド共鳴エネルギー (`dot(q*gain, w)`) を集める。
    fn ca3_candidates(&self, q_hashes: &[u64], top_n: usize) -> Vec<(u32, f32)> {
        let mut accum: HashMap<u32, f32> = HashMap::with_capacity(2048);
        let mut seen_cells = Vec::with_capacity(q_hashes.len());
        for &h in q_hashes {
            let g = dg_cell(h);
            if seen_cells.contains(&g) {
                continue;
            }
            seen_cells.push(g);
            let q_act = self.dg_gain[g];
            if q_act <= 0.0 {
                continue;
            }
            let start = self.csr_off[g] as usize;
            let end = self.csr_off[g + 1] as usize;
            for idx in start..end {
                let slot = self.csr_slot[idx];
                let w = from_bf16(self.csr_w[idx]);
                *accum.entry(slot).or_insert(0.0) += w * q_act;
            }
        }
        let mut ranked: Vec<(u32, f32)> = accum.into_iter().collect();
        if ranked.len() > top_n {
            ranked.select_nth_unstable_by(top_n, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            ranked.truncate(top_n);
        }
        ranked
    }

    /// 非対称ホップフィールド共鳴により最大 `limit` 件の記憶エングラムを想起する。
    pub fn recall_hits(&self, q: &str, limit: usize) -> Result<Vec<RecalledFact>> {
        let q_norm = norm_text(q);
        if q_norm.is_empty() || limit == 0 {
            return Ok(vec![]);
        }
        let q_hashes = extract_dg_hashes64(&q_norm);
        if q_hashes.is_empty() {
            return Ok(vec![]);
        }

        // 1. Online fast-weight taught engrams take highest priority
        let mut hits = Vec::new();
        for t in &self.taught {
            let mut hit_w2 = 0.0f32;
            for &h in &t.hashes {
                if q_hashes.binary_search(&h).is_ok() {
                    let w = self.dg_gain[dg_cell(h)].max(0.5);
                    hit_w2 += w * w;
                }
            }
            let cov = hit_w2 / (t.norm2 + 1e-6);
            if cov >= 0.85 || q_norm.contains(&t.head_norm) {
                hits.push(RecalledFact {
                    head: t.head_norm.clone(),
                    body: t.body.clone(),
                    source: "taught",
                    score: 1.0,
                });
                if hits.len() >= limit {
                    return Ok(hits);
                }
            }
        }
        if !hits.is_empty() {
            return Ok(hits);
        }

        // 2. Consolidated DG -> CA3 sparse retrieval + collision-free 64-bit Hopfield verification
        let q_active_w2: f32 = q_hashes
            .iter()
            .filter(|&&h| self.dg_gain[dg_cell(h)] > 0.0)
            .map(|&h| {
                let w = self.dg_gain[dg_cell(h)];
                w * w
            })
            .sum();
        if q_active_w2 <= 1e-6 {
            return Ok(vec![]);
        }

        let candidates = self.ca3_candidates(&q_hashes, 2048);
        // (rank_energy, calibrated_score, prio_q, lead_id, slot)
        let mut scored: Vec<(f32, f32, u8, u32, usize)> = Vec::new();
        for (slot_u32, _coarse) in candidates {
            let slot = slot_u32 as usize;
            let cue = self.slot_cue_str(slot);
            if cue.is_empty() {
                continue;
            }
            let c_hashes = extract_dg_hashes64(cue);
            if c_hashes.is_empty() {
                continue;
            }
            let mut hit_w2 = 0.0f32;
            let mut c_total_w2 = 0.0f32;
            for &h in &c_hashes {
                let w = self.dg_gain[dg_cell(h)];
                c_total_w2 += w * w;
                if q_hashes.binary_search(&h).is_ok() {
                    hit_w2 += w * w;
                }
            }
            let cue_cov = hit_w2 / (c_total_w2 + 1e-6);
            if cue_cov < 0.85 {
                continue;
            }
            let exact_match = q_norm == cue;
            let span_hit = exact_match || q_norm.contains(cue);
            let salience_share = hit_w2 / q_active_w2;
            if !span_hit && salience_share < 0.35 {
                continue;
            }
            if !exact_match && hit_w2 < 0.45 {
                continue;
            }
            let cue_chars = cue.chars().count();
            let lead_id = self.slot_lead_id[slot] as usize;
            let prio_q = self.lead_prio_q[lead_id];
            let qual_norm = (prio_q & 0x0f) as f32 / 5.0;
            let prio_high = if (prio_q >> 4) > 0 { 1.0 } else { 0.0 };
            let kind_bonus = match self.slot_kind[slot] {
                KIND_PREDICATE => 1.0,
                KIND_EXACT => 0.55,
                _ => 0.15,
            };
            let spec = 1.0 - (-hit_w2.sqrt() / 3.5).exp();
            let span_b = if span_hit { 1.0 } else { 0.0 };
            let rank_energy = hit_w2.sqrt()
                + 3.0 * cue_cov
                + 2.5 * cue_chars as f32
                + 4.0 * kind_bonus
                + 3.0 * span_b
                + 1.5 * qual_norm
                + prio_high;
            let score = (0.30
                + 0.32 * cue_cov
                + 0.18 * spec
                + 0.08 * kind_bonus
                + 0.06 * span_b
                + 0.04 * qual_norm
                + 0.02 * prio_high)
                .clamp(0.0, 0.99);
            scored.push((rank_energy, score, prio_q, lead_id as u32, slot));
        }

        scored.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.3.cmp(&b.3))
        });

        let mut seen_leads = Vec::new();
        for (_rank, score, _pq, lead_id_u32, slot) in scored {
            if seen_leads.contains(&lead_id_u32) {
                continue;
            }
            seen_leads.push(lead_id_u32);
            let lead_id = lead_id_u32 as usize;
            let source = match self.slot_kind[slot] {
                KIND_PREDICATE => "wiki_predicate",
                KIND_EXACT => "wiki_exact",
                _ => "wiki_alias",
            };
            let head = self.lead_head_str(lead_id).to_owned();
            let body = self.read_body(lead_id)?;
            hits.push(RecalledFact {
                head,
                body,
                source,
                score: score as f64,
            });
            if hits.len() >= limit {
                break;
            }
        }
        Ok(hits)
    }

    /// CA3 複数実体想起 + 側方抑制 NMS + 神経数値/単位抽出により、比較・差分の前提計画を構築する。
    pub fn compare_plan(
        &self,
        q: &str,
        reasoner: &crate::reasoner::Reasoner,
    ) -> Result<Option<crate::reasoner::ReasoningPlan>> {
        if !self.predict_is_compare(q) {
            return Ok(None);
        }
        let q_norm = norm_text(q);
        let q_hashes = extract_dg_hashes64(&q_norm);
        if q_hashes.is_empty() {
            return Ok(None);
        }

        let mut raw_cands: Vec<RawSlotCandidate> = Vec::new();
        // 1. Collect matching online taught entities occurring in query
        for (idx, t) in self.taught.iter().enumerate() {
            if let Some(pos) = q_norm.find(&t.head_norm) {
                raw_cands.push(RawSlotCandidate {
                    lead_id: u32::MAX - idx as u32,
                    span_start: pos,
                    span_len: t.head_norm.len(),
                    is_taught: true,
                    taught_body: t.body.clone(),
                    cue_str: t.head_norm.clone(),
                    head_str: t.head.clone(),
                    resonance: 1.0,
                    prio_q: u8::MAX,
                });
            }
        }

        // 2. Collect DG -> CA3 consolidated entities whose cue spans inside query
        let cands = self.ca3_candidates(&q_hashes, 2048);
        for (slot_u32, _coarse) in cands {
            let slot = slot_u32 as usize;
            if self.slot_kind[slot] == KIND_PREDICATE {
                continue;
            }
            let cue = self.slot_cue_str(slot);
            if cue.chars().count() < 2 {
                continue;
            }
            if let Some(pos) = q_norm.find(cue) {
                let c_hashes = extract_dg_hashes64(cue);
                let mut hit_w2 = 0.0f32;
                for &h in &c_hashes {
                    if q_hashes.binary_search(&h).is_ok() {
                        let w = self.dg_gain[dg_cell(h)];
                        hit_w2 += w * w;
                    }
                }
                let res = hit_w2 / (self.slot_norm2[slot] + 1e-6);
                if res >= 0.90 {
                    let lead_id = self.slot_lead_id[slot];
                    raw_cands.push(RawSlotCandidate {
                        lead_id,
                        span_start: pos,
                        span_len: cue.len(),
                        is_taught: false,
                        taught_body: String::new(),
                        cue_str: cue.to_owned(),
                        head_str: self.lead_head_str(lead_id as usize).to_owned(),
                        resonance: res,
                        prio_q: self.lead_prio_q[lead_id as usize],
                    });
                }
            }
        }

        // 3. Lateral Inhibition (longest non-overlapping span NMS; keep up to 3 senses per span)
        raw_cands.sort_by(|a, b| {
            b.span_len
                .cmp(&a.span_len)
                .then_with(|| b.is_taught.cmp(&a.is_taught))
                .then_with(|| b.resonance.total_cmp(&a.resonance))
                .then_with(|| b.prio_q.cmp(&a.prio_q))
                .then_with(|| a.lead_id.cmp(&b.lead_id))
        });

        let mut kept_spans: Vec<(usize, usize, Vec<RawSlotCandidate>)> = Vec::new();
        for c in raw_cands {
            let s0 = c.span_start;
            let e0 = s0 + c.span_len;
            if let Some((_, _, group)) = kept_spans
                .iter_mut()
                .find(|(gs, ge, _)| *gs == s0 && *ge == e0)
            {
                if group.len() < 3 && !group.iter().any(|x| x.lead_id == c.lead_id) {
                    group.push(c);
                }
                continue;
            }
            let overlaps = kept_spans
                .iter()
                .any(|(gs, ge, _)| s0 < *ge && *gs < e0);
            if !overlaps {
                kept_spans.push((s0, e0, vec![c]));
            }
        }
        kept_spans.sort_by_key(|(s, _, _)| *s);

        // 4. Neural measurement extraction per surviving entity span
        let q_orig_chars: Vec<char> = q.chars().collect();
        let _ = q_orig_chars;
        let mut measured: Vec<(String, String, usize)> = Vec::new();
        for (_, _, group) in kept_spans {
            for cand in group {
                let body = if cand.is_taught {
                    cand.taught_body.clone()
                } else {
                    self.read_body(cand.lead_id as usize)?
                };
                if let Some((val, unit)) = self.extract_neural_measure(q, &body) {
                    // Use original case head when cue matches exact head (so reasoner preserves display name)
                    let display = if !cand.is_taught && norm_text(&cand.head_str) == cand.cue_str {
                        cand.head_str.clone()
                    } else {
                        cand.cue_str.clone()
                    };
                    measured.push((display, val, unit));
                    break;
                }
            }
        }
        if measured.len() < 2 {
            return Ok(None);
        }

        // Keep entities matching the dominant measurement unit
        let mut unit_counts = [0usize; 3];
        for &(_, _, u) in &measured {
            if u < 3 {
                unit_counts[u] += 1;
            }
        }
        let dom_unit = (0..3).max_by_key(|&u| unit_counts[u]).unwrap_or(0);
        let unit_str = match dom_unit {
            0 => "m",
            1 => "km",
            2 => "cm",
            _ => return Ok(None),
        };
        let mut items: Vec<(f64, &str, &str)> = Vec::new();
        for (name, val, u) in &measured {
            if *u == dom_unit {
                if let Ok(v) = val.parse::<f64>() {
                    if v.is_finite() {
                        items.push((v, name.as_str(), unit_str));
                    }
                }
            }
        }
        if items.len() < 2 {
            return Ok(None);
        }
        Ok(Some(reasoner.reason_quantities_plan(q, &items)))
    }

    /// バイナリチェックポイント `NGHC2301` を保存する（本文アリーナはファイル末尾に配置し RAM に載せない）。
    pub fn save_with_bodies(&mut self, path: &Path, bodies: &[Vec<u8>]) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        ensure!(bodies.len() == self.n_leads);
        {
            let mut w = BufWriter::new(File::create(path)?);
            w.write_all(MAGIC)?;
            w.write_all(&(self.n_leads as u64).to_le_bytes())?;
            w.write_all(&(self.n_slots as u64).to_le_bytes())?;
            w.write_all(&(self.cue_arena.len() as u64).to_le_bytes())?;
            w.write_all(&(self.heads_arena.len() as u64).to_le_bytes())?;
            w.write_all(&(self.csr_slot.len() as u64).to_le_bytes())?;

            w.write_all(bytemuck::cast_slice(&self.dg_gain))?;
            w.write_all(bytemuck::cast_slice(&self.csr_off))?;
            w.write_all(bytemuck::cast_slice(&self.csr_slot))?;
            w.write_all(bytemuck::cast_slice(&self.csr_w))?;
            w.write_all(bytemuck::cast_slice(&self.slot_lead_id))?;
            w.write_all(bytemuck::cast_slice(&self.slot_cue_off))?;
            w.write_all(bytemuck::cast_slice(&self.slot_cue_len))?;
            w.write_all(&self.slot_kind)?;
            w.write_all(bytemuck::cast_slice(&self.slot_norm2))?;
            w.write_all(&self.cue_arena)?;
            w.write_all(bytemuck::cast_slice(&self.lead_head_off))?;
            w.write_all(bytemuck::cast_slice(&self.lead_head_len))?;
            w.write_all(&self.lead_prio_q)?;
            let body_off_pos = w.stream_position()?;
            w.write_all(bytemuck::cast_slice(&self.lead_body_off))?;
            w.write_all(bytemuck::cast_slice(&self.lead_body_len))?;
            w.write_all(&self.heads_arena)?;
            w.write_all(bytemuck::cast_slice(&self.w_cmp_query))?;
            w.write_all(bytemuck::cast_slice(&self.b_cmp_query))?;
            w.write_all(bytemuck::cast_slice(&self.w_num_ctx))?;
            w.write_all(bytemuck::cast_slice(&self.b_num_ctx))?;
            w.write_all(bytemuck::cast_slice(&self.w_unit))?;
            w.write_all(bytemuck::cast_slice(&self.b_unit))?;

            for (i, b) in bodies.iter().enumerate() {
                let off = w.stream_position()?;
                self.lead_body_off[i] = off;
                self.lead_body_len[i] = b.len() as u16;
                w.write_all(b)?;
            }
            w.flush()?;

            // Seek back and patch exact lead_body_off & lead_body_len arrays
            w.seek(SeekFrom::Start(body_off_pos))?;
            w.write_all(bytemuck::cast_slice(&self.lead_body_off))?;
            w.write_all(bytemuck::cast_slice(&self.lead_body_len))?;
            w.flush()?;
        }
        self.body_file = Some(Mutex::new(File::open(path)?));
        Ok(())
    }

    /// `NGHC2301` バイナリチェックポイントを高速ロードする（本文アリーナはディスク上に残しオンデマンド読込）。
    pub fn load(path: &Path) -> Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid HippocampusCore checkpoint magic");

        let mut u64_buf = [0u8; 8];
        let mut read_u64 = |rd: &mut BufReader<File>| -> Result<usize> {
            rd.read_exact(&mut u64_buf)?;
            Ok(u64::from_le_bytes(u64_buf) as usize)
        };
        let n_leads = read_u64(&mut r)?;
        let n_slots = read_u64(&mut r)?;
        let cue_arena_len = read_u64(&mut r)?;
        let heads_arena_len = read_u64(&mut r)?;
        let n_edges = read_u64(&mut r)?;

        let mut dg_gain = vec![0.0f32; DG_CELLS];
        r.read_exact(bytemuck::cast_slice_mut(&mut dg_gain))?;
        let mut csr_off = vec![0u32; DG_CELLS + 1];
        r.read_exact(bytemuck::cast_slice_mut(&mut csr_off))?;
        let mut csr_slot = vec![0u32; n_edges];
        r.read_exact(bytemuck::cast_slice_mut(&mut csr_slot))?;
        let mut csr_w = vec![0u16; n_edges];
        r.read_exact(bytemuck::cast_slice_mut(&mut csr_w))?;
        let mut slot_lead_id = vec![0u32; n_slots];
        r.read_exact(bytemuck::cast_slice_mut(&mut slot_lead_id))?;
        let mut slot_cue_off = vec![0u32; n_slots];
        r.read_exact(bytemuck::cast_slice_mut(&mut slot_cue_off))?;
        let mut slot_cue_len = vec![0u16; n_slots];
        r.read_exact(bytemuck::cast_slice_mut(&mut slot_cue_len))?;
        let mut slot_kind = vec![0u8; n_slots];
        r.read_exact(&mut slot_kind)?;
        let mut slot_norm2 = vec![0.0f32; n_slots];
        r.read_exact(bytemuck::cast_slice_mut(&mut slot_norm2))?;
        let mut cue_arena = vec![0u8; cue_arena_len];
        r.read_exact(&mut cue_arena)?;
        let mut lead_head_off = vec![0u32; n_leads];
        r.read_exact(bytemuck::cast_slice_mut(&mut lead_head_off))?;
        let mut lead_head_len = vec![0u16; n_leads];
        r.read_exact(bytemuck::cast_slice_mut(&mut lead_head_len))?;
        let mut lead_prio_q = vec![0u8; n_leads];
        r.read_exact(&mut lead_prio_q)?;
        let mut lead_body_off = vec![0u64; n_leads];
        r.read_exact(bytemuck::cast_slice_mut(&mut lead_body_off))?;
        let mut lead_body_len = vec![0u16; n_leads];
        r.read_exact(bytemuck::cast_slice_mut(&mut lead_body_len))?;
        let mut heads_arena = vec![0u8; heads_arena_len];
        r.read_exact(&mut heads_arena)?;

        let mut w_cmp_query = vec![0.0f32; CMP_BUCKETS * 2];
        r.read_exact(bytemuck::cast_slice_mut(&mut w_cmp_query))?;
        let mut b_cmp_query = [0.0f32; 2];
        r.read_exact(bytemuck::cast_slice_mut(&mut b_cmp_query))?;
        let mut w_num_ctx = vec![0.0f32; NUM_CTX_BUCKETS * 2];
        r.read_exact(bytemuck::cast_slice_mut(&mut w_num_ctx))?;
        let mut b_num_ctx = [0.0f32; 2];
        r.read_exact(bytemuck::cast_slice_mut(&mut b_num_ctx))?;
        let mut w_unit = vec![0.0f32; UNIT_BUCKETS * NUM_UNITS];
        r.read_exact(bytemuck::cast_slice_mut(&mut w_unit))?;
        let mut b_unit = [0.0f32; NUM_UNITS];
        r.read_exact(bytemuck::cast_slice_mut(&mut b_unit))?;

        Ok(Self {
            n_leads,
            n_slots,
            dg_gain,
            csr_off,
            csr_slot,
            csr_w,
            slot_lead_id,
            slot_cue_off,
            slot_cue_len,
            slot_kind,
            slot_norm2,
            cue_arena,
            lead_head_off,
            lead_head_len,
            lead_prio_q,
            lead_body_off,
            lead_body_len,
            heads_arena,
            w_cmp_query,
            b_cmp_query,
            w_num_ctx,
            b_num_ctx,
            w_unit,
            b_unit,
            body_file: Some(Mutex::new(File::open(path)?)),
            mem_bodies: Vec::new(),
            taught: Vec::new(),
        })
    }
}

/// 文字列先頭から最大 `max_bytes` 以内の UTF-8 文字境界まで切り出す。
fn truncate_utf8(s: &str, max_bytes: usize) -> &[u8] {
    let b = s.as_bytes();
    if b.len() <= max_bytes {
        return b;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &b[..end]
}

/// 読込専用で SQLite からオフライン睡眠統合用の生コンテキスト統計を収集し、`df_bg` をオープン日本語発話から学習する。
fn collect_bg_df(root: &Path, excluded: &[&str]) -> Vec<u32> {
    let mut df_bg = vec![0u32; DG_CELLS];
    let mut observe = |txt: &str| {
        if txt.is_empty() || excluded.iter().any(|e| txt.contains(e)) {
            return;
        }
        for g in extract_dg_grams(txt) {
            df_bg[g] = df_bg[g].saturating_add(1);
        }
    };

    if let Ok(raw) = fs::read_to_string(root.join("v220_executive_core/episodes.jsonl")) {
        for line in raw.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if let Some(t) = v["text"].as_str() {
                    observe(t);
                }
            }
        }
    }
    if let Ok(raw) = fs::read_to_string(root.join("sft_raw/databricks-dolly-15k-ja.json")) {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&raw) {
            for item in arr {
                if let Some(ins) = item["instruction"].as_str() {
                    observe(ins);
                }
            }
        }
    }
    if let Ok(file) = File::open(root.join("sft_raw/oasst1-21k-ja.jsonl")) {
        let reader = BufReader::new(file);
        for (idx, line) in std::io::BufRead::lines(reader).enumerate() {
            if idx >= 8000 {
                break;
            }
            if let Ok(l) = line {
                if let Ok(v) = serde_json::from_str::<Value>(&l) {
                    if let Some(arr) = v["messages"].as_array().or_else(|| v["conversations"].as_array())
                    {
                        if let Some(first) = arr.first() {
                            if let Some(t) = first["content"].as_str().or_else(|| first["value"].as_str())
                            {
                                observe(t);
                            }
                        }
                    }
                }
            }
        }
    }
    df_bg
}

#[derive(Serialize)]
struct ProbeSummary {
    query: String,
    head: Option<String>,
    source: Option<String>,
    score: Option<f64>,
    ok: bool,
}

/// オフライン睡眠統合 (`v230-build`): 凍結 `v124.sqlite` の全リード・述語・別名を
/// 純ヘブ連想記憶 `HippocampusCore` (`NGHC2301`) へ統合し、ゼロリークベンチマークを出力する。
pub fn consolidate_and_bench(root: &Path, ckpt_path: &Path, out_json: &Path) -> Result<Value> {
    let t_total = Instant::now();
    let excluded_entities = [
        "富士山", "北岳", "槍ヶ岳", "東京都", "朝塔", "夕塔", "光合成", "平城京", "架空",
    ];
    let sqlite_path = root.join("wiki_lead_v112/v124.sqlite");
    let conn = Connection::open_with_flags(&sqlite_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // 1. Load all leads sorted by (prio DESC, def_quality DESC, id ASC)
    let mut id_map: HashMap<i64, u32> = HashMap::with_capacity(800_000);
    let mut heads_arena: Vec<u8> = Vec::with_capacity(16 * 1024 * 1024);
    let mut lead_head_off: Vec<u32> = Vec::with_capacity(760_000);
    let mut lead_head_len: Vec<u16> = Vec::with_capacity(760_000);
    let mut lead_prio_q: Vec<u8> = Vec::with_capacity(760_000);
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(760_000);

    // Keep compact slot list before building CSR
    let mut cue_arena: Vec<u8> = Vec::with_capacity(32 * 1024 * 1024);
    let mut slot_lead_id: Vec<u32> = Vec::with_capacity(1_250_000);
    let mut slot_cue_off: Vec<u32> = Vec::with_capacity(1_250_000);
    let mut slot_cue_len: Vec<u16> = Vec::with_capacity(1_250_000);
    let mut slot_kind: Vec<u8> = Vec::with_capacity(1_250_000);

    // Training samples collected strictly from NON-excluded leads
    struct NumSample {
        body_chars: Vec<char>,
        span: NumericSpan,
        query: String,
        is_measure: bool,
        unit: usize,
        is_val: bool,
    }
    let mut num_samples: Vec<NumSample> = Vec::with_capacity(50_000);
    let mut neg_seen = 0usize;
    let mut cmp_entities: Vec<(String, String)> = Vec::new(); // (head, property_word)

    {
        let mut stmt = conn.prepare(
            "SELECT id, head_norm, head, prio, def_quality, lead FROM leads ORDER BY prio DESC, def_quality DESC, id ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut lead_idx = 0usize;
        while let Some(row) = rows.next()? {
            let sid: i64 = row.get(0)?;
            let head_norm_raw: String = row.get(1)?;
            let head: String = row.get(2)?;
            let prio: f64 = row.get::<_, Option<f64>>(3)?.unwrap_or(1.0);
            let qual: i64 = row.get::<_, Option<i64>>(4)?.unwrap_or(0);
            let lead: String = row.get(5)?;

            id_map.insert(sid, lead_idx as u32);
            let hb = truncate_utf8(&head, u16::MAX as usize);
            lead_head_off.push(heads_arena.len() as u32);
            lead_head_len.push(hb.len() as u16);
            heads_arena.extend_from_slice(hb);
            let pq = ((prio >= 1.8) as u8) * 16 + (qual.clamp(0, 15) as u8);
            lead_prio_q.push(pq);

            let bb = truncate_utf8(&lead, BODY_MAX_BYTES).to_vec();
            bodies.push(bb);

            let hnorm = norm_text(&head_norm_raw);
            if !hnorm.is_empty() && hnorm.len() <= u16::MAX as usize {
                slot_lead_id.push(lead_idx as u32);
                slot_cue_off.push(cue_arena.len() as u32);
                slot_cue_len.push(hnorm.len() as u16);
                slot_kind.push(KIND_EXACT);
                cue_arena.extend_from_slice(hnorm.as_bytes());
            }

            // Collect self-supervised numeric & unit examples ONLY from non-excluded entities
            let is_excl = excluded_entities.iter().any(|&e| head.contains(e));
            if !is_excl && num_samples.len() < 48_000 && (lead_idx % 5 == 0 || lead.contains("標高") || lead.contains("高さ")) {
                let l_nfkc: String = lead.nfkc().collect();
                let chars: Vec<char> = l_nfkc.chars().collect();
                let spans = find_numeric_spans(&l_nfkc);
                for sp in spans {
                    let mut r = sp.end;
                    while r < chars.len() && chars[r].is_whitespace() {
                        r += 1;
                    }
                    let rest: String = chars[r..(r + 8).min(chars.len())].iter().collect();
                    let next_after = |n: usize| -> char { chars.get(r + n).copied().unwrap_or(' ') };
                    let is_sq = |ch: char| matches!(ch, '²' | '³' | '^' | '/' | '2' | '3');

                    let (is_m, unit, prop) = if (rest.starts_with("キロメートル") && !is_sq(next_after(6)))
                        || (rest.starts_with("km") && !next_after(2).is_ascii_alphanumeric() && !is_sq(next_after(2)))
                    {
                        (true, 1usize, "距離")
                    } else if (rest.starts_with("センチメートル") && !is_sq(next_after(7)))
                        || (rest.starts_with("cm") && !next_after(2).is_ascii_alphanumeric() && !is_sq(next_after(2)))
                    {
                        (true, 2usize, "長さ")
                    } else if (rest.starts_with("メートル") && !is_sq(next_after(4)))
                        || (rest.starts_with('m')
                            && !next_after(1).is_ascii_alphanumeric()
                            && !is_sq(next_after(1)))
                    {
                        let p = if lead.contains("標高") {
                            "標高"
                        } else if lead.contains("海抜") {
                            "海抜"
                        } else {
                            "高さ"
                        };
                        (true, 0usize, p)
                    } else if matches!(
                        next_after(0),
                        '年' | '月' | '日' | '人' | '位' | '番' | '回' | '倍' | '万' | '億' | '%' | '％' | '個' | '歳'
                    ) {
                        (false, 3usize, "標高")
                    } else {
                        continue;
                    };
                    if is_m && cmp_entities.len() < 800 && head.chars().count() >= 2 && head.chars().count() <= 8 {
                        cmp_entities.push((head.clone(), prop.to_owned()));
                    }
                    // Keep balanced positive / negative spans and split uniformly into train/val
                    let keep = if is_m {
                        true
                    } else {
                        neg_seen += 1;
                        (neg_seen % 12) == 0
                    };
                    if keep {
                        let is_val = (num_samples.len() % 6) == 0;
                        num_samples.push(NumSample {
                            body_chars: chars.clone(),
                            span: sp,
                            query: format!("{prop}の差は何{unit_tok}？", unit_tok = if unit == 1 { "km" } else if unit == 2 { "cm" } else { "m" }),
                            is_measure: is_m,
                            unit,
                            is_val,
                        });
                    }
                }
            }

            lead_idx += 1;
        }
    }
    let n_leads = lead_head_off.len();

    // Load predicates & aliases
    {
        let mut stmt = conn.prepare("SELECT phrase, lead_id FROM predicates")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let phrase: String = row.get(0)?;
            let sid: i64 = row.get(1)?;
            if let Some(&lid) = id_map.get(&sid) {
                let pnorm = norm_text(&phrase);
                if !pnorm.is_empty() && pnorm.len() <= u16::MAX as usize {
                    slot_lead_id.push(lid);
                    slot_cue_off.push(cue_arena.len() as u32);
                    slot_cue_len.push(pnorm.len() as u16);
                    slot_kind.push(KIND_PREDICATE);
                    cue_arena.extend_from_slice(pnorm.as_bytes());
                }
            }
        }
    }
    {
        let mut stmt = conn.prepare("SELECT alias, lead_id FROM aliases")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let alias: String = row.get(0)?;
            let sid: i64 = row.get(1)?;
            if let Some(&lid) = id_map.get(&sid) {
                let anorm = norm_text(&alias);
                if !anorm.is_empty() && anorm.len() <= u16::MAX as usize {
                    slot_lead_id.push(lid);
                    slot_cue_off.push(cue_arena.len() as u32);
                    slot_cue_len.push(anorm.len() as u16);
                    slot_kind.push(KIND_ALIAS);
                    cue_arena.extend_from_slice(anorm.as_bytes());
                }
            }
        }
    }
    drop(id_map);
    let n_slots = slot_lead_id.len();

    // 2. Learn homeostatic DG inhibition gain (df_engram + df_bg)
    let df_bg = collect_bg_df(root, &excluded_entities);
    let mut df_engram = vec![0u32; DG_CELLS];
    for s in 0..n_slots {
        let off = slot_cue_off[s] as usize;
        let len = slot_cue_len[s] as usize;
        let cue = std::str::from_utf8(&cue_arena[off..off + len]).unwrap_or("");
        for g in extract_dg_grams(cue) {
            df_engram[g] = df_engram[g].saturating_add(1);
        }
    }

    let mut dg_gain = vec![0.0f32; DG_CELLS];
    for g in 0..DG_CELLS {
        if df_engram[g] > 0 {
            let idf = ((n_slots as f32 + 1.0) / (df_engram[g] as f32 + 1.0)).ln();
            let bg_sup = 1.0 + 2.2 * (1.0 + df_bg[g] as f32).ln().powf(1.2);
            dg_gain[g] = (idf / bg_sup).max(0.15);
        }
    }

    // 3. Two-pass DG -> CA3 sparse CSR wiring (TOP_K_SYNAPSES highest-gain cells per slot)
    let mut slot_norm2 = vec![0.0f32; n_slots];
    let mut slot_cue_chars = vec![0u16; n_slots];
    let mut deg = vec![0u32; DG_CELLS];
    for s in 0..n_slots {
        let off = slot_cue_off[s] as usize;
        let len = slot_cue_len[s] as usize;
        let cue = std::str::from_utf8(&cue_arena[off..off + len]).unwrap_or("");
        slot_cue_chars[s] = cue.chars().count().min(u16::MAX as usize) as u16;
        let grams = extract_dg_grams(cue);
        let mut n2 = 0.0f32;
        let mut top_g: Vec<(f32, usize)> = Vec::with_capacity(grams.len());
        for g in grams {
            let w = dg_gain[g];
            n2 += w * w;
            top_g.push((w, g));
        }
        slot_norm2[s] = n2.max(1e-6);
        if top_g.len() > TOP_K_SYNAPSES {
            top_g.select_nth_unstable_by(TOP_K_SYNAPSES, |a, b| b.0.total_cmp(&a.0));
            top_g.truncate(TOP_K_SYNAPSES);
        }
        for &(_, g) in &top_g {
            deg[g] += 1;
        }
    }

    let mut raw_off = vec![0u32; DG_CELLS + 1];
    for g in 0..DG_CELLS {
        raw_off[g + 1] = raw_off[g] + deg[g];
    }
    let total_raw = raw_off[DG_CELLS] as usize;
    let mut raw_slot = vec![0u32; total_raw];
    let mut cursor = raw_off[..DG_CELLS].to_vec();
    for s in 0..n_slots {
        let off = slot_cue_off[s] as usize;
        let len = slot_cue_len[s] as usize;
        let cue = std::str::from_utf8(&cue_arena[off..off + len]).unwrap_or("");
        let grams = extract_dg_grams(cue);
        let mut top_g: Vec<(f32, usize)> = grams.into_iter().map(|g| (dg_gain[g], g)).collect();
        if top_g.len() > TOP_K_SYNAPSES {
            top_g.select_nth_unstable_by(TOP_K_SYNAPSES, |a, b| b.0.total_cmp(&a.0));
            top_g.truncate(TOP_K_SYNAPSES);
        }
        for &(_, g) in &top_g {
            let pos = cursor[g] as usize;
            raw_slot[pos] = s as u32;
            cursor[g] += 1;
        }
    }
    drop(cursor);
    drop(deg);

    // Compact CSR respecting MAX_POSTING_DF sorted by (cue_chars ASC, kind ASC, lead_id ASC)
    let mut csr_off = vec![0u32; DG_CELLS + 1];
    let mut csr_slot = Vec::with_capacity(total_raw.min(n_slots * TOP_K_SYNAPSES));
    let mut csr_w = Vec::with_capacity(csr_slot.capacity());
    for g in 0..DG_CELLS {
        let s0 = raw_off[g] as usize;
        let e0 = raw_off[g + 1] as usize;
        let slice = &mut raw_slot[s0..e0];
        if slice.len() > MAX_POSTING_DF {
            slice.sort_unstable_by(|&a, &b| {
                let sa = a as usize;
                let sb = b as usize;
                slot_cue_chars[sa]
                    .cmp(&slot_cue_chars[sb])
                    .then_with(|| slot_kind[sa].cmp(&slot_kind[sb]))
                    .then_with(|| slot_lead_id[sa].cmp(&slot_lead_id[sb]))
            });
            slice[..MAX_POSTING_DF].sort_unstable();
            for &sl in &slice[..MAX_POSTING_DF] {
                let w = dg_gain[g] / slot_norm2[sl as usize].sqrt();
                csr_slot.push(sl);
                csr_w.push(to_bf16(w));
            }
        } else {
            for &sl in slice.iter() {
                let w = dg_gain[g] / slot_norm2[sl as usize].sqrt();
                csr_slot.push(sl);
                csr_w.push(to_bf16(w));
            }
        }
        csr_off[g + 1] = csr_slot.len() as u32;
    }
    drop(raw_slot);
    drop(raw_off);
    drop(slot_cue_chars);

    let mut core = HippocampusCore {
        n_leads,
        n_slots,
        dg_gain,
        csr_off,
        csr_slot,
        csr_w,
        slot_lead_id,
        slot_cue_off,
        slot_cue_len,
        slot_kind,
        slot_norm2,
        cue_arena,
        lead_head_off,
        lead_head_len,
        lead_prio_q,
        lead_body_off: vec![0u64; n_leads],
        lead_body_len: vec![0u16; n_leads],
        heads_arena,
        w_cmp_query: vec![0.0f32; CMP_BUCKETS * 2],
        b_cmp_query: [0.0f32; 2],
        w_num_ctx: vec![0.0f32; NUM_CTX_BUCKETS * 2],
        b_num_ctx: [0.0f32; 2],
        w_unit: vec![0.0f32; UNIT_BUCKETS * NUM_UNITS],
        b_unit: [0.0f32; NUM_UNITS],
        body_file: None,
        mem_bodies: Vec::new(),
        taught: Vec::new(),
    };

    // 4. Train Hebbian readout heads (w_num_ctx, w_unit, w_cmp_query) without held-out test entities
    for pass in 0..5 {
        let lr = 0.24 / (1.0 + 0.30 * pass as f32);
        for s in &num_samples {
            if s.is_val {
                continue;
            }
            // Binary measure head
            let feats = extract_num_ctx_features(&s.body_chars, &s.span, &s.query);
            let inv = 1.0 / (feats.len().max(1) as f32).sqrt();
            let mut sc = core.b_num_ctx;
            for &f in &feats {
                let base = f * 2;
                sc[0] += inv * core.w_num_ctx[base];
                sc[1] += inv * core.w_num_ctx[base + 1];
            }
            let max_s = sc[0].max(sc[1]);
            let e0 = (sc[0] - max_s).exp();
            let e1 = (sc[1] - max_s).exp();
            let p1 = e1 / (e0 + e1 + 1e-12);
            let y = if s.is_measure { 1.0 } else { 0.0 };
            let delta = y - p1;
            core.b_num_ctx[0] -= lr * 0.05 * delta;
            core.b_num_ctx[1] += lr * 0.05 * delta;
            for &f in &feats {
                let base = f * 2;
                core.w_num_ctx[base] -= lr * inv * delta;
                core.w_num_ctx[base + 1] += lr * inv * delta;
            }

            // 4-class unit head
            let ufeats = extract_unit_features(&s.body_chars, s.span.end);
            let uinv = 1.0 / (ufeats.len().max(1) as f32).sqrt();
            let mut us = core.b_unit;
            for &f in &ufeats {
                let base = f * NUM_UNITS;
                for c in 0..NUM_UNITS {
                    us[c] += uinv * core.w_unit[base + c];
                }
            }
            let umax = us.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut up = [0.0f32; NUM_UNITS];
            let mut usum = 0.0f32;
            for c in 0..NUM_UNITS {
                up[c] = (us[c] - umax).exp();
                usum += up[c];
            }
            for c in 0..NUM_UNITS {
                up[c] /= usum + 1e-12;
                let ud = if c == s.unit { 1.0 - up[c] } else { -up[c] };
                core.b_unit[c] += lr * 0.05 * ud;
            }
            for &f in &ufeats {
                let base = f * NUM_UNITS;
                for c in 0..NUM_UNITS {
                    let ud = if c == s.unit { 1.0 - up[c] } else { -up[c] };
                    core.w_unit[base + c] += lr * uinv * ud;
                }
            }
        }
    }

    // Train w_cmp_query using disjoint entity pairs + Dolly/RealPersonaChat negative queries
    let mut cmp_data: Vec<(String, bool, bool)> = Vec::new();
    let cmp_templates = [
        "{a}と{b}の{p}の差は何m？",
        "{a}と{b}の{p}差は何m？",
        "{a}と{b}はどちらが{p}が高い？",
        "{a}と{b}で{p}が高いのはどっち？",
        "{a}と{b}の{p}の違いってどれくらい？",
        "{a}、{b}、{c}の{p}の差は？",
        "{a}と{b}を比べるとどちらが大きい？",
        "{a}と{b}の{p}の差を計算して",
    ];
    for i in 0..cmp_entities.len().saturating_sub(2) {
        let (a, pa) = &cmp_entities[i];
        let (b, _) = &cmp_entities[(i * 7 + 1) % cmp_entities.len()];
        let (c, _) = &cmp_entities[(i * 13 + 2) % cmp_entities.len()];
        if a == b || b == c {
            continue;
        }
        let tpl = cmp_templates[i % cmp_templates.len()];
        let q = tpl.replace("{a}", a).replace("{b}", b).replace("{c}", c).replace("{p}", pa);
        cmp_data.push((q, true, (i % 6) == 0));
    }
    let neg_templates = [
        "{a}とは何ですか？",
        "{a}ってどこにある？",
        "{a}の仕組みを説明して",
        "{a}について知りたい",
        "{a}って知ってる？",
        "{a}の歴史を教えて",
        "{a}の特徴は何ですか？",
        "{a}の{p}はいくつ？",
    ];
    for (i, (a, p)) in cmp_entities.iter().enumerate() {
        let tpl = neg_templates[i % neg_templates.len()];
        cmp_data.push((tpl.replace("{a}", a).replace("{p}", p), false, (i % 6) == 0));
    }
    for pass in 0..6 {
        let lr = 0.25 / (1.0 + 0.30 * pass as f32);
        for (q, label, is_val) in &cmp_data {
            if *is_val {
                continue;
            }
            let feats = extract_cmp_query_features(q);
            let inv = 1.0 / (feats.len().max(1) as f32).sqrt();
            let mut s = core.b_cmp_query;
            for &f in &feats {
                let base = f * 2;
                s[0] += inv * core.w_cmp_query[base];
                s[1] += inv * core.w_cmp_query[base + 1];
            }
            let m = s[0].max(s[1]);
            let p1 = (s[1] - m).exp() / ((s[0] - m).exp() + (s[1] - m).exp() + 1e-12);
            let y = if *label { 1.0 } else { 0.0 };
            let delta = y - p1;
            core.b_cmp_query[0] -= lr * 0.05 * delta;
            core.b_cmp_query[1] += lr * 0.05 * delta;
            for &f in &feats {
                let base = f * 2;
                core.w_cmp_query[base] -= lr * inv * delta;
                core.w_cmp_query[base + 1] += lr * inv * delta;
            }
        }
    }

    // Evaluate readout heads on held-out val samples
    let (mut n_ok, mut n_tot, mut u_ok, mut u_tot) = (0usize, 0usize, 0usize, 0usize);
    for s in &num_samples {
        if !s.is_val {
            continue;
        }
        n_tot += 1;
        let pred_m = core.score_num_span(&s.body_chars, &s.span, &s.query) > 0.0;
        if pred_m == s.is_measure {
            n_ok += 1;
        }
        u_tot += 1;
        if core.predict_unit(&s.body_chars, s.span.end) == s.unit {
            u_ok += 1;
        }
    }
    let (mut c_ok, mut c_tot) = (0usize, 0usize);
    for (q, label, is_val) in &cmp_data {
        if *is_val {
            c_tot += 1;
            if core.predict_is_compare(q) == *label {
                c_ok += 1;
            }
        }
    }
    let val_num_acc = n_ok as f64 / n_tot.max(1) as f64;
    let val_unit_acc = u_ok as f64 / u_tot.max(1) as f64;
    let val_cmp_acc = c_ok as f64 / c_tot.max(1) as f64;

    // 5. Save consolidated binary checkpoint NGHC2301 & cold reload verify
    core.save_with_bodies(ckpt_path, &bodies)?;
    drop(bodies);
    let build_elapsed_s = t_total.elapsed().as_secs_f64();
    let ckpt_mib = fs::metadata(ckpt_path)?.len() as f64 / (1024.0 * 1024.0);

    let tl = Instant::now();
    let loaded = HippocampusCore::load(ckpt_path)?;
    let load_ms = tl.elapsed().as_secs_f64() * 1000.0;

    // Probe recall latency & unseen paraphrases
    let probe_cases = [
        ("日本の首都とは何ですか？", "東京都"),
        ("日本の首都ってどこだっけ？", "東京都"),
        ("日本の首都を知ってる？", "東京都"),
        ("富士山について知りたい", "富士山"),
        ("光合成の仕組みを説明して", "光合成"),
        ("北岳とは何ですか？", "北岳"),
    ];
    let mut recall_times_ms = Vec::new();
    let mut probes = Vec::new();
    let mut probes_ok = 0usize;
    for (q, expected_head) in probe_cases {
        let tr = Instant::now();
        let hits = loaded.recall_hits(q, 3)?;
        recall_times_ms.push(tr.elapsed().as_secs_f64() * 1000.0);
        let top = hits.first();
        let ok = top.is_some_and(|h| h.head == expected_head && h.score >= 0.45);
        if ok {
            probes_ok += 1;
        }
        probes.push(ProbeSummary {
            query: q.to_owned(),
            head: top.map(|h| h.head.clone()),
            source: top.map(|h| h.source.to_owned()),
            score: top.map(|h| h.score),
            ok,
        });
    }

    // Measure compare_plan latency & held-out accuracy
    let rp1 = root.join("v240_reasoner_core/reasoner_core_v24.bin");
    let rp2 = root.join("rust_bench_v182/v240/reasoner_core_v24.bin");
    let r_path = if rp1.exists() {
        rp1
    } else if rp2.exists() {
        rp2
    } else {
        root.join("rust_bench_v182/blank_synapses.json")
    };
    let reasoner = crate::reasoner::Reasoner::load_synapses(&r_path)?;
    let mut cmp_times_ms = Vec::new();
    let mut cmp_probes = Vec::new();
    let mut cmp_probes_ok = 0usize;
    for (q, expected_ans) in [
        ("富士山と北岳の標高差は何m？", "583.12m"),
        ("富士山と北岳と槍ヶ岳の標高差は何m？", "596.12m"),
        ("富士山と槍ヶ岳はどちらが標高が高い？", "富士山"),
        ("富士山と北岳の高さの差ってどれくらい？", "583.12m"),
    ] {
        let tc = Instant::now();
        let plan = loaded.compare_plan(q, &reasoner)?;
        cmp_times_ms.push(tc.elapsed().as_secs_f64() * 1000.0);
        let ans = plan.map(|p| String::from_utf8_lossy(&p.answer_bytes).into_owned());
        let ok = ans.as_deref() == Some(expected_ans);
        if ok {
            cmp_probes_ok += 1;
        }
        cmp_probes.push(json!({"query": q, "expected": expected_ans, "pred": ans, "ok": ok}));
    }

    let avg_recall_ms = recall_times_ms.iter().sum::<f64>() / recall_times_ms.len().max(1) as f64;
    let avg_compare_ms = cmp_times_ms.iter().sum::<f64>() / cmp_times_ms.len().max(1) as f64;
    let (rss, peak) = crate::memory::working_set();
    let payload = json!({
        "n_leads": loaded.n_leads,
        "n_slots": loaded.n_slots,
        "n_edges": loaded.csr_slot.len(),
        "ckpt_mib": ckpt_mib,
        "load_ms": load_ms,
        "build_elapsed_s": build_elapsed_s,
        "avg_recall_ms": avg_recall_ms,
        "avg_compare_ms": avg_compare_ms,
        "val_num_acc": val_num_acc,
        "val_unit_acc": val_unit_acc,
        "val_cmp_acc": val_cmp_acc,
        "recall_probes_ok": format!("{probes_ok}/{}", probe_cases.len()),
        "compare_probes_ok": format!("{cmp_probes_ok}/4"),
        "recall_probes": probes,
        "compare_probes": cmp_probes,
        "rss_mib": rss,
        "peak_rss_mib": peak,
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
    use std::collections::BTreeMap;

    fn make_mini_core() -> HippocampusCore {
        let leads = [
            ("富士嶺", "富士嶺は標高3776.12 mの山。"),
            ("北岳嶺", "北岳嶺は2015年に調査され、標高3193 mの山。"),
            ("東京都", "東京都は日本の首都。"),
        ];
        let slots = [
            (0u32, "富士嶺", KIND_EXACT),
            (1u32, "北岳嶺", KIND_EXACT),
            (2u32, "東京都", KIND_EXACT),
            (2u32, "日本の首都", KIND_PREDICATE),
        ];
        let mut dg_gain = vec![0.0f32; DG_CELLS];
        for &(_, cue, _) in &slots {
            for g in extract_dg_grams(cue) {
                dg_gain[g] = 4.0;
            }
        }
        // Simulate homeostatic background inhibition on generic tails
        for g in extract_dg_grams("ってどこだっけ") {
            dg_gain[g] = 0.2;
        }

        let mut buckets: BTreeMap<usize, Vec<(u32, f32)>> = BTreeMap::new();
        let mut cue_arena = Vec::new();
        let mut slot_lead_id = Vec::new();
        let mut slot_cue_off = Vec::new();
        let mut slot_cue_len = Vec::new();
        let mut slot_kind = Vec::new();
        let mut slot_norm2 = Vec::new();
        for (s_id, &(lid, cue, kind)) in slots.iter().enumerate() {
            let norm = norm_text(cue);
            slot_lead_id.push(lid);
            slot_cue_off.push(cue_arena.len() as u32);
            slot_cue_len.push(norm.len() as u16);
            slot_kind.push(kind);
            cue_arena.extend_from_slice(norm.as_bytes());
            let grams = extract_dg_grams(&norm);
            let n2: f32 = grams.iter().map(|&g| dg_gain[g] * dg_gain[g]).sum();
            slot_norm2.push(n2);
            for g in grams {
                buckets.entry(g).or_default().push((s_id as u32, dg_gain[g] / n2.sqrt()));
            }
        }
        let mut csr_off = vec![0u32; DG_CELLS + 1];
        let mut csr_slot = Vec::new();
        let mut csr_w = Vec::new();
        for g in 0..DG_CELLS {
            if let Some(v) = buckets.get(&g) {
                for &(sl, w) in v {
                    csr_slot.push(sl);
                    csr_w.push(to_bf16(w));
                }
            }
            csr_off[g + 1] = csr_slot.len() as u32;
        }
        let mut heads_arena = Vec::new();
        let mut lead_head_off = Vec::new();
        let mut lead_head_len = Vec::new();
        let mut mem_bodies = Vec::new();
        for &(h, b) in &leads {
            lead_head_off.push(heads_arena.len() as u32);
            lead_head_len.push(h.len() as u16);
            heads_arena.extend_from_slice(h.as_bytes());
            mem_bodies.push(b.to_owned());
        }
        HippocampusCore {
            n_leads: leads.len(),
            n_slots: slots.len(),
            dg_gain,
            csr_off,
            csr_slot,
            csr_w,
            slot_lead_id,
            slot_cue_off,
            slot_cue_len,
            slot_kind,
            slot_norm2,
            cue_arena,
            lead_head_off,
            lead_head_len,
            lead_prio_q: vec![21; leads.len()],
            lead_body_off: vec![0; leads.len()],
            lead_body_len: vec![0; leads.len()],
            heads_arena,
            w_cmp_query: vec![0.0; CMP_BUCKETS * 2],
            b_cmp_query: [0.0; 2],
            w_num_ctx: vec![0.0; NUM_CTX_BUCKETS * 2],
            b_num_ctx: [0.0; 2],
            w_unit: vec![0.0; UNIT_BUCKETS * NUM_UNITS],
            b_unit: [0.0; NUM_UNITS],
            body_file: None,
            mem_bodies,
            taught: Vec::new(),
        }
    }

    #[test]
    fn ca3_recall_teach_and_roundtrip_work_without_regex() -> Result<()> {
        let mut core = make_mini_core();
        let hits = core.recall_hits("日本の首都ってどこだっけ？", 3)?;
        assert!(!hits.is_empty());
        assert_eq!(hits[0].head, "東京都");
        assert_eq!(hits[0].source, "wiki_predicate");

        core.teach("朝塔", "高さは340m。")?;
        let thits = core.recall_hits("朝塔とは？", 1)?;
        assert_eq!(thits[0].source, "taught");
        assert_eq!(thits[0].body, "高さは340m。");

        // Binary save + load round-trip test
        let tmp = std::env::temp_dir().join(format!("nghc-test-{}.bin", std::process::id()));
        let bodies: Vec<Vec<u8>> = core.mem_bodies.iter().map(|s| s.as_bytes().to_vec()).collect();
        core.save_with_bodies(&tmp, &bodies)?;
        let reloaded = HippocampusCore::load(&tmp)?;
        fs::remove_file(&tmp)?;
        let rhits = reloaded.recall_hits("富士嶺ってどこ？", 1)?;
        assert_eq!(rhits[0].head, "富士嶺");
        assert!(rhits[0].body.contains("3776.12"));
        Ok(())
    }
}
