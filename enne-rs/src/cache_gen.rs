//! バイト遷移の局所共起と凍結コアの読み出しを統合する。
use crate::{
    inference::{CoreState, Inference},
    reasoner::{ReasonMode, ReasoningPlan},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const BYTE_COUNT: usize = 256;
const NEXT_P: usize = 256;
const ANS: usize = 257;
const NEXT_V: usize = 258;
const END: usize = 259;
const SYMBOLS: usize = 260;
const BOS: u16 = 260;
const ORDER: usize = 18;
const PLACEHOLDER_P: [&str; 4] = ["藍晶", "砂丘", "水紋", "白帆"];
const PLACEHOLDER_V: [&str; 2] = ["42", "17"];
const PLACEHOLDER_ANS: &str = "月影";

#[derive(Deserialize)]
pub struct Example {
    pub mode: ReasonMode,
    pub text: String,
    pub split: String,
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct QueueState {
    p_more: bool,
    p_started: bool,
    v_more: bool,
    v_pending: bool,
    ans_rem: bool,
}

impl QueueState {
    fn new(p_idx: usize, p_len: usize, v_idx: usize, v_len: usize, ans_rem: bool) -> Self {
        let v_more = v_idx < v_len;
        Self {
            p_more: p_idx < p_len,
            p_started: p_idx > 0,
            v_more,
            v_pending: v_more && p_idx > v_idx,
            ans_rem,
        }
    }
}

#[derive(Hash, Eq, PartialEq)]
struct Context {
    mode: ReasonMode,
    state: QueueState,
    opening: [u16; 3],
    suffix: Vec<u16>,
}

/// 発話の先頭1文字（または先頭制御記号）を取り出し、長距離の反復を抑える。
fn opening_sig(h: &[u16]) -> [u16; 3] {
    if h.len() < 2 {
        return [0; 3];
    }
    if h[1] >= BYTE_COUNT as u16 {
        return [h[1], 0, 0];
    }
    if h.len() < 4 {
        return [0; 3];
    }
    [h[1], h[2], h[3]]
}

pub struct Bridge {
    counts: HashMap<Context, Vec<(usize, u32)>>,
    sums: Vec<[f32; 256]>,
    freq: [usize; SYMBOLS],
    pub weight: f32,
    pub dev: Vec<(f32, f64)>,
}

/// 動的キュー記号 `<P*>`, `<V*>`, `<ANS>` を内部シンボル一個に畳み、他は UTF-8 バイトへ展開する。
fn parse_marker(s: &str) -> Option<(usize, usize)> {
    if s.starts_with("<ANS>") {
        return Some((ANS, 5));
    }
    let bytes = s.as_bytes();
    if bytes.len() >= 4 && bytes[0] == b'<' && (bytes[1] == b'P' || bytes[1] == b'V') {
        let mut i = 2;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > 2 && i < bytes.len() && bytes[i] == b'>' {
            let sym = if bytes[1] == b'P' { NEXT_P } else { NEXT_V };
            return Some((sym, i + 1));
        }
    }
    None
}

fn tokenize(mut s: &str) -> Vec<usize> {
    let mut out = vec![];
    while !s.is_empty() {
        if let Some((sym, len)) = parse_marker(s) {
            out.push(sym);
            s = &s[len..];
        } else {
            let c = s.chars().next().unwrap();
            out.extend(s.as_bytes()[..c.len_utf8()].iter().map(|&b| b as usize));
            s = &s[c.len_utf8()..];
        }
    }
    out.push(END);
    out
}

/// 入力済みバイトの末尾について RFC 3629 の部分列条件を検査する。
fn legal(bytes: &[u8], next: u8, remaining: usize) -> bool {
    let offset = bytes.len().saturating_sub(3);
    let start = (offset..bytes.len())
        .rev()
        .find(|&i| bytes[i] & 0xc0 != 0x80);
    let mut tail = [0; 4];
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

/// 対数確率をビットに変換する。
pub fn bits(logits: &[f32], index: usize) -> f64 {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ((peak - logits[index]) as f64
        + logits
            .iter()
            .map(|&x| ((x - peak) as f64).exp())
            .sum::<f64>()
            .ln())
        / std::f64::consts::LN_2
}

/// ターン内で最後に観測したバイトの次記号分布を保持する。
struct Readout<'a> {
    core: Option<&'a Inference>,
    state: Option<CoreState>,
    logits: [f32; 256],
    h: [f32; 256],
}

impl<'a> Readout<'a> {
    /// 例として空の質問には改行一個で初期状態を与える。
    fn new(core: Option<&'a Inference>, prompt: &[u8]) -> Self {
        let mut r = Self {
            core,
            state: core.map(Inference::init_state),
            logits: [0.; 256],
            h: [0.; 256],
        };
        if core.is_some() {
            for &b in prompt.iter().chain([&b'\n']) {
                r.feed(b);
            }
        }
        r
    }
    /// キャッシュ由来のバイトにも同じ状態遷移を適用する。
    fn feed(&mut self, b: u8) {
        if let (Some(c), Some(st)) = (self.core, self.state.as_mut()) {
            (self.logits, self.h) = c.step_byte(st, b);
        }
    }
}

impl Bridge {
    /// 直前の記号列、発話先頭の1文字、動的キュー状態をキーにする。
    fn key(mode: ReasonMode, state: QueueState, opening: [u16; 3], h: &[u16], n: usize) -> Context {
        Context {
            mode,
            state,
            opening,
            suffix: h[h.len() - n..].to_vec(),
        }
    }

    /// 文単位の訓練・開発分割を保ち、BOS 付き局所共起と活性の外積を加算する。
    pub fn train(core: &Inference, examples: &[Example]) -> Result<Self> {
        let mut b = Self {
            counts: HashMap::new(),
            sums: vec![[0.; 256]; SYMBOLS],
            freq: [0; SYMBOLS],
            weight: 0.,
            dev: vec![],
        };
        let candidates = [0.12, 0.3, 0.6, 1.0];
        let mut losses = [0.; 4];
        let mut total = 0;
        for split in ["train", "dev"] {
            for ex in examples.iter().filter(|x| x.split == split) {
                let symbols = tokenize(&ex.text);
                let p_len = symbols.iter().filter(|&&s| s == NEXT_P).count();
                let v_len = symbols.iter().filter(|&&s| s == NEXT_V).count();
                let has_ans = symbols.contains(&ANS);
                let (mut p_idx, mut v_idx, mut ans_used) = (0usize, 0usize, false);
                let mut history = vec![BOS];
                let mut r = Readout::new(Some(core), &[]);
                for s in symbols {
                    let state = QueueState::new(p_idx, p_len, v_idx, v_len, has_ans && !ans_used);
                    if split == "train" {
                        let op = opening_sig(&history);
                        for n in 0..=history.len().min(ORDER) {
                            for key_op in [[0; 3], op] {
                                let counts = b
                                    .counts
                                    .entry(Self::key(ex.mode, state, key_op, &history, n))
                                    .or_default();
                                if let Some(pair) = counts.iter_mut().find(|p| p.0 == s) {
                                    pair.1 += 1;
                                } else {
                                    counts.push((s, 1));
                                }
                                if op == [0; 3] {
                                    break;
                                }
                            }
                        }
                        b.freq[s] += 1;
                        for (sum, &v) in b.sums[s].iter_mut().zip(&r.h) {
                            *sum += v;
                        }
                    } else {
                        let (scores, _) = b.scores(ex.mode, state, &history, &r);
                        let next_bytes = [
                            (p_idx < p_len)
                                .then(|| PLACEHOLDER_P[p_idx % PLACEHOLDER_P.len()].as_bytes()[0]),
                            (has_ans && !ans_used).then(|| PLACEHOLDER_ANS.as_bytes()[0]),
                            (v_idx < v_len)
                                .then(|| PLACEHOLDER_V[v_idx % PLACEHOLDER_V.len()].as_bytes()[0]),
                            Some(b'\n'),
                        ];
                        for (i, &w) in candidates.iter().enumerate() {
                            let mut v = scores;
                            b.mix(&mut v, &r, w, next_bytes);
                            losses[i] += bits(&v, s);
                        }
                        total += 1;
                    }
                    history.push(s as u16);
                    if s < BYTE_COUNT {
                        r.feed(s as u8);
                    } else if s == NEXT_P {
                        let span = PLACEHOLDER_P[p_idx % PLACEHOLDER_P.len()].as_bytes();
                        p_idx += 1;
                        for &x in span {
                            r.feed(x);
                        }
                    } else if s == ANS {
                        ans_used = true;
                        for &x in PLACEHOLDER_ANS.as_bytes() {
                            r.feed(x);
                        }
                    } else if s == NEXT_V {
                        let span = PLACEHOLDER_V[v_idx % PLACEHOLDER_V.len()].as_bytes();
                        v_idx += 1;
                        for &x in span {
                            r.feed(x);
                        }
                    }
                }
            }
        }
        ensure!(total > 0);
        b.dev = candidates
            .into_iter()
            .zip(losses.map(|x| x / total as f64))
            .collect();
        b.weight = b.dev.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        Ok(b)
    }

    /// 非ゼロ文脈の一致がある間は文中ユニグラムを混ぜず、最長一致支持集合も返す。
    fn scores(
        &self,
        mode: ReasonMode,
        state: QueueState,
        h: &[u16],
        r: &Readout,
    ) -> ([f32; SYMBOLS], [bool; SYMBOLS]) {
        let mut v = [1e-6_f32; SYMBOLS];
        let mut supported = [false; SYMBOLS];
        let mut matched_positive = false;
        let max_n = h.len().min(ORDER);
        let op = opening_sig(h);
        for n in 1..=max_n {
            let entry = if op != [0; 3] {
                self.counts
                    .get(&Self::key(mode, state, op, h, n))
                    .or_else(|| self.counts.get(&Self::key(mode, state, [0; 3], h, n)))
            } else {
                self.counts.get(&Self::key(mode, state, [0; 3], h, n))
            };
            if let Some(counts) = entry {
                let sum = counts.iter().map(|x| x.1).sum::<u32>() as f32;
                let keep = if matched_positive { 0.08 } else { 0.0 };
                for x in &mut v {
                    *x = *x * keep + if matched_positive { 0.0 } else { 1e-6 };
                }
                supported.fill(false);
                for &(s, count) in counts {
                    v[s] += (1.0 - keep) * count as f32 / sum;
                    supported[s] = true;
                }
                matched_positive = true;
            }
        }
        if !matched_positive {
            let entry = if op != [0; 3] {
                self.counts
                    .get(&Self::key(mode, state, op, h, 0))
                    .or_else(|| self.counts.get(&Self::key(mode, state, [0; 3], h, 0)))
            } else {
                self.counts.get(&Self::key(mode, state, [0; 3], h, 0))
            };
            if let Some(counts) = entry {
                let sum = counts.iter().map(|x| x.1).sum::<u32>() as f32;
                for &(s, count) in counts {
                    v[s] += count as f32 / sum;
                    supported[s] = true;
                }
            }
        }
        for (s, x) in v.iter_mut().enumerate() {
            *x = x.ln();
            if r.core.is_some() && self.freq[s] > 0 {
                *x += self.sums[s]
                    .iter()
                    .zip(r.h.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (256 * self.freq[s]) as f32;
            }
        }
        (v, supported)
    }

    /// バイト記号と制御記号の先頭バイトを同じ対数尺度でコアロジットへ統合する。
    fn mix(&self, v: &mut [f32; SYMBOLS], r: &Readout, w: f32, next_bytes: [Option<u8>; 4]) {
        let peak = r.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for i in 0..BYTE_COUNT {
            v[i] += w * (r.logits[i] - peak);
        }
        for (k, byte) in next_bytes.into_iter().enumerate() {
            if let Some(b) = byte {
                v[BYTE_COUNT + k] += w * (r.logits[b as usize] - peak);
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    pub core: bool,
    pub bridge: bool,
    pub cache: bool,
    pub short: bool,
    pub temperature: f32,
    pub seed: u64,
}

#[derive(Serialize)]
pub struct Output {
    pub text: String,
    pub pointer_ranges: Vec<(usize, usize)>,
    pub forced_budget_gates: usize,
    pub pointer_bytes: usize,
    pub stopped: bool,
    pub generated_core_steps: usize,
}

/// 温度ゼロでは最初の最大値、それ以外では再現可能な温度サンプリングを用いる。
fn choose(v: &[f32], t: f32, seed: &mut u64) -> usize {
    let best = (0..v.len())
        .reduce(|a, b| if v[b] > v[a] { b } else { a })
        .unwrap();
    if t == 0. {
        return best;
    }
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut residual = ((*seed >> 11) as f64 / (1u64 << 53) as f64)
        * v.iter()
            .map(|x| (((x - v[best]) / t) as f64).exp())
            .sum::<f64>();
    for (i, x) in v.iter().enumerate() {
        residual -= (((x - v[best]) / t) as f64).exp();
        if residual < 0. {
            return i;
        }
    }
    best
}

/// 動的キューの寿命はこの呼び出しに限定し、出力は全て一バイトずつコアへ戻す。
pub fn generate(
    core: &Inference,
    bridge: &Bridge,
    plan: &ReasoningPlan,
    prompt: &[u8],
    cfg: Config,
) -> Output {
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
    let (mut p_idx, mut v_idx, mut ans_used) = (0usize, 0usize, false);
    let mut bytes = vec![];
    let mut history = vec![BOS];
    let mut seed = cfg.seed;
    let mut pointer_ranges = vec![];
    let mut forced_budget_gates = 0;
    let mut pointer_bytes = 0;
    let mut stopped = false;
    let mut r = Readout::new(cfg.core.then_some(core), prompt);
    let total_payload: usize = path_items.iter().map(|s| s.len()).sum::<usize>()
        + val_items.iter().map(|s| s.len()).sum::<usize>()
        + ans_bytes.len();
    // 非キャッシュ条件は評価計画の文字列長も受け取らず、固定予算で停止する。
    let limit = match (cfg.short, cfg.cache) {
        (true, false) => 48,
        (true, true) => 48.max(ans_bytes.len()),
        (false, false) => 320,
        (false, true) => 320.max(total_payload),
    };
    while bytes.len() < limit {
        let state = QueueState::new(
            p_idx,
            path_items.len(),
            v_idx,
            val_items.len(),
            has_ans && !ans_used,
        );
        let (mut scores, supported) = if cfg.bridge {
            bridge.scores(plan.mode, state, &history, &r)
        } else {
            ([0.; SYMBOLS], [true; SYMBOLS])
        };
        if cfg.core {
            let next_bytes = if cfg.cache {
                [
                    path_items.get(p_idx).and_then(|s| s.first().copied()),
                    (has_ans && !ans_used).then(|| ans_bytes[0]),
                    val_items.get(v_idx).and_then(|s| s.first().copied()),
                    Some(b'\n'),
                ]
            } else {
                [None, None, None, Some(b'\n')]
            };
            bridge.mix(
                &mut scores,
                &r,
                if cfg.bridge { bridge.weight } else { 1. },
                next_bytes,
            );
        }
        let boundary = std::str::from_utf8(&bytes).is_ok();
        // 未発火キューの長さを残す。上限で発火を強制した回数は監査用に記録する。
        let reserve: usize = if cfg.cache && !cfg.short {
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
        let available = limit - bytes.len() - reserve;
        for s in 0..BYTE_COUNT {
            if available == 0 || !legal(&bytes, s as u8, available) {
                scores[s] = f32::NEG_INFINITY;
            }
        }
        let allow_p = cfg.bridge
            && !(cfg.short && !cfg.cache)
            && boundary
            && p_idx < path_items.len()
            && (!cfg.cache || path_items[p_idx].len() <= limit - bytes.len());
        let allow_ans = cfg.bridge
            && !(cfg.short && !cfg.cache)
            && boundary
            && has_ans
            && !ans_used
            && (!cfg.cache || ans_bytes.len() <= limit - bytes.len());
        let allow_v = cfg.bridge
            && !(cfg.short && !cfg.cache)
            && boundary
            && v_idx < val_items.len()
            && (!cfg.cache || val_items[v_idx].len() <= limit - bytes.len());
        if !allow_p {
            scores[NEXT_P] = f32::NEG_INFINITY;
        }
        if !allow_ans {
            scores[ANS] = f32::NEG_INFINITY;
        }
        if !allow_v {
            scores[NEXT_V] = f32::NEG_INFINITY;
        }
        let all_emitted =
            p_idx == path_items.len() && v_idx == val_items.len() && (!has_ans || ans_used);
        if !boundary || !cfg.bridge || (!cfg.short && !all_emitted) {
            scores[END] = f32::NEG_INFINITY;
        }
        // 最長一致文脈で支持された合法記号が存在する場合はそちらを優先する。
        if cfg.bridge && !(cfg.short && cfg.cache) {
            let has_supported = (0..SYMBOLS).any(|s| supported[s] && scores[s].is_finite());
            if has_supported {
                for s in 0..SYMBOLS {
                    if !supported[s] {
                        scores[s] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        // 短答モードの許容言語は回答ゲート一個のみ。
        if cfg.short && cfg.cache {
            scores.fill(f32::NEG_INFINITY);
            scores[if has_ans && !ans_used { ANS } else { END }] = 0.;
        }
        let s = choose(&scores, cfg.temperature, &mut seed);
        history.push(s as u16);
        if s == END {
            stopped = true;
            break;
        }
        if s < BYTE_COUNT {
            bytes.push(s as u8);
            r.feed(s as u8);
            if !cfg.bridge && s == 10 && bytes.len() > 1 {
                stopped = true;
                break;
            }
        } else {
            let span: &[u8] = if s == NEXT_P {
                let item = path_items[p_idx];
                p_idx += 1;
                item
            } else if s == ANS {
                ans_used = true;
                ans_bytes
            } else {
                let item = val_items[v_idx];
                v_idx += 1;
                item
            };
            if !cfg.cache {
                continue;
            }
            if available == 0 {
                forced_budget_gates += 1;
            }
            pointer_ranges.push((bytes.len(), bytes.len() + span.len()));
            for &b in span {
                bytes.push(b);
                r.feed(b);
                pointer_bytes += 1;
            }
        }
    }
    Output {
        text: String::from_utf8(bytes).expect("UTF-8"),
        forced_budget_gates,
        pointer_ranges,
        pointer_bytes,
        stopped,
        generated_core_steps: if cfg.core {
            r.state.as_ref().unwrap().len_after(prompt.len() + 1)
        } else {
            0
        },
    }
}

/// 独立したコアで出力を teacher forcing し、自由生成部分の尤度評価にも使える。
pub fn score_text(core: &Inference, prompt: &[u8], out: &Output) -> (f64, f64, usize) {
    let mut r = Readout::new(Some(core), prompt);
    let mut n = 0.;
    let mut expression = 0.;
    let mut count = 0;
    for (i, &b) in out.text.as_bytes().iter().enumerate() {
        let loss = bits(&r.logits, b as usize);
        n += loss;
        if !out.pointer_ranges.iter().any(|&(a, z)| i >= a && i < z) {
            expression += loss;
            count += 1;
        }
        r.feed(b);
    }
    (n, expression, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// サロゲート、過長形式、最大符号位置、残り長の境界を検査する。
    #[test]
    fn unicode_boundaries() {
        assert!(!legal(&[0xed], 0xa0, 5));
        assert!(!legal(&[0xf4], 0x90, 5));
        assert!(!legal(&[], 0xc0, 5));
        assert!(!legal(&[], 0xf0, 3));
        assert!(legal(&[0xf4, 0x8f, 0xbf], 0xbf, 1));
    }
    /// 実体名を埋め込まず動的ゲートへ畳むことを確認する。
    #[test]
    fn control_symbols() {
        assert_eq!(tokenize("<ANS>!"), [ANS, 33, END]);
        assert_eq!(
            tokenize("<P0><P1><P4><V0><V2>"),
            [NEXT_P, NEXT_P, NEXT_P, NEXT_V, NEXT_V, END]
        );
    }
}
