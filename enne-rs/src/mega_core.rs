//! ENNE CPU 専用大規模神経新生エンジン（v18.5: D=256・BF16 SoA・UTF-8文字単位長視野ホップ・ユニコード2文字トピック記憶・AVX2 SGEMM）。
//! CPU メモリ上でヘブ学習・有糸分裂・アポトーシスを実行する。
use crate::core_dag::mix;
use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

pub const DIM: usize = 256;
pub const VOCAB: usize = 256;
const HOPS: usize = 16;
const CHAR_HIST: usize = 18;
const TOPIC_SLOTS: usize = 4;
const SHARP_K: usize = 8;
const MAX_BIRTHS_PER_STEP: usize = 96;
const CHAR_SAL_SIZE: usize = 65536;

#[inline(always)]
fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// 不偏確率丸めにより、小刻みなヘブ更新でも BF16 重みが停滞しないようにする。
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

/// 親ID・条件種別（0=局所バイト/文字, 1=長距離2文字トピック）・64bit条件値から非ゼロ署名を作る。
#[inline(always)]
fn sig64(p: u32, k: usize, c: u64) -> u64 {
    mix(mix(mix(p as u64 + 11) ^ ((k as u64 + 1) * 1000003)) ^ mix(c.wrapping_add(2000029))).max(1)
}

#[inline(always)]
fn gain(col: usize) -> f32 {
    1.0 / (1.0 + 0.12 * col.div_ceil(4) as f32)
}

/// 8レーン並列の xorshift32 による AVX2 自動ベクトル化対応 BF16 確率丸めヘブ更新。
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

/// スレッド間で互いに素なノードID (`nid % n_threads == tid`) のみを書き換えるための生ポインタラッパ。
#[derive(Clone, Copy)]
struct DisjointArrays {
    w: *mut u16,
    sharp_dst: *mut i16,
    sharp_w: *mut u16,
    updates: *mut u32,
    recent: *mut u16,
    utility: *mut f32,
}
unsafe impl Send for DisjointArrays {}
unsafe impl Sync for DisjointArrays {}

#[inline(always)]
fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Clone)]
struct StreamCtx {
    /// 直近18文字の完了済み Unicode スカラー値リングバッファ（日本語約54バイト相当）。
    char_hist: [u32; CHAR_HIST],
    char_len: usize,
    cur_buf: [u8; 4],
    cur_len: u8,
    cur_expected: u8,
    /// 4個の自己組織化トピック記憶スロット（Unicode 2文字バイグラム `(prev_cp << 21) | cp`）。
    topic_slots: [u64; TOPIC_SLOTS],
    topic_score: [f32; TOPIC_SLOTS],
}

impl Default for StreamCtx {
    fn default() -> Self {
        Self {
            char_hist: [0; CHAR_HIST],
            char_len: 0,
            cur_buf: [0; 4],
            cur_len: 0,
            cur_expected: 0,
            topic_slots: [0; TOPIC_SLOTS],
            topic_score: [0.0; TOPIC_SLOTS],
        }
    }
}

impl StreamCtx {
    /// 1バイトを観測し、UTF-8文字が完了したら文字履歴と2文字トピック記憶を更新する。
    #[inline]
    fn observe_byte(&mut self, b: u8, char_sal: &[f32]) {
        for s in &mut self.topic_score {
            *s *= 0.992;
        }
        if b < 0x80 {
            self.cur_len = 0;
            self.cur_expected = 0;
            self.push_char(b as u32, char_sal);
        } else if b >= 0xc0 {
            self.cur_buf[0] = b;
            self.cur_len = 1;
            self.cur_expected = if b < 0xe0 {
                2
            } else if b < 0xf0 {
                3
            } else {
                4
            };
        } else if self.cur_len > 0 && self.cur_len < self.cur_expected {
            self.cur_buf[self.cur_len as usize] = b;
            self.cur_len += 1;
            if self.cur_len == self.cur_expected {
                if let Ok(s) = std::str::from_utf8(&self.cur_buf[..self.cur_len as usize]) {
                    if let Some(ch) = s.chars().next() {
                        self.push_char(ch as u32, char_sal);
                    }
                }
                self.cur_len = 0;
                self.cur_expected = 0;
            }
        } else {
            self.cur_len = 0;
            self.cur_expected = 0;
        }
    }

    #[inline]
    fn push_char(&mut self, cp: u32, char_sal: &[f32]) {
        let prev = if self.char_len > 0 {
            self.char_hist[(self.char_len - 1) % CHAR_HIST]
        } else {
            0
        };
        self.char_hist[self.char_len % CHAR_HIST] = cp;
        self.char_len += 1;
        if prev != 0 && cp > 0x20 && prev > 0x20 {
            let s1 = char_sal[(prev as usize) & (CHAR_SAL_SIZE - 1)];
            let s2 = char_sal[(cp as usize) & (CHAR_SAL_SIZE - 1)];
            let score = 0.5 * (s1 + s2);
            if score > 0.90 {
                let bg = ((prev as u64) << 21) | (cp as u64);
                if let Some(pos) = self.topic_slots.iter().position(|&x| x == bg) {
                    self.topic_score[pos] = self.topic_score[pos].max(score);
                } else {
                    let mut min_i = 0;
                    for k in 1..TOPIC_SLOTS {
                        if self.topic_score[k] < self.topic_score[min_i] {
                            min_i = k;
                        }
                    }
                    if score > self.topic_score[min_i] {
                        self.topic_slots[min_i] = bg;
                        self.topic_score[min_i] = score;
                    }
                }
            }
        }
    }

    /// 完了済み文字履歴から k 番目前の文字（k=1 が直前の完了文字、現在文字が丁度完了した直後は k=2 が1文字前）を返す。
    #[inline]
    fn prev_char(&self, back: usize) -> Option<u32> {
        // 現在観測中のバイトでちょうど文字が完了した直後(cur_len == 0)は、char_hist[char_len-1]が今見ている文字自身。
        // 従って1文字前は skip=1 (cur_len==0) または skip=0 (cur_len>0)。
        let skip = usize::from(self.cur_len == 0);
        let offset = back + skip;
        if offset == 0 || offset > CHAR_HIST || offset > self.char_len {
            return None;
        }
        let cp = self.char_hist[(self.char_len - offset) % CHAR_HIST];
        (cp != 0).then_some(cp)
    }

    /// スコア最大のトピック2文字バイグラムを返す。
    #[inline]
    fn top_topic(&self, rank: usize) -> Option<(u64, f32)> {
        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&a, &b| self.topic_score[b].total_cmp(&self.topic_score[a]));
        let idx = order[rank % TOPIC_SLOTS];
        let bg = self.topic_slots[idx];
        let sc = self.topic_score[idx];
        (bg != 0 && sc > 0.35).then_some((bg, sc))
    }
}

/// D=256・BF16 SoA・文字単位長視野ホップ対応の超大規模 Neurogenesis グラフ。
pub struct MegaCore {
    pub cap: usize,
    pub buckets: usize,
    pub len: usize,
    pub alive_count: usize,
    w: Vec<u16>,
    sharp_dst: Vec<i16>,
    sharp_w: Vec<u16>,
    parent: Vec<u32>,
    span: Vec<u8>,
    depth: Vec<u8>,
    alive: Vec<u8>,
    recent: Vec<u16>,
    updates: Vec<u32>,
    utility: Vec<f32>,
    pain: Vec<f32>,
    free: Vec<u32>,
    sigs: Vec<u64>,
    children: Vec<u32>,
    basis: Vec<f32>,
    bias: [f32; VOCAB],
    char_sal: Vec<f32>,
    rng: u64,
}

#[derive(Serialize)]
pub struct Milestone {
    pub bytes_seen: usize,
    pub elapsed_s: f64,
    pub chunk_bytes_per_s: f64,
    pub alive_neurons: usize,
    pub allocated_nodes: usize,
    pub synaptic_params: usize,
    pub chunk_train_bpc: f64,
    pub val_bpc: f64,
    pub rss_mib: f64,
    pub peak_rss_mib: f64,
    pub apoptosis_events: usize,
}

#[derive(Serialize)]
pub struct GenSample {
    pub prompt: String,
    pub greedy: String,
    pub sampled: String,
}

#[derive(Serialize)]
pub struct MegaReport {
    pub cap: usize,
    pub buckets: usize,
    pub dim: usize,
    pub hops: usize,
    pub batch: usize,
    pub threads: usize,
    pub train_bytes: usize,
    pub val_bytes: usize,
    pub total_elapsed_s: f64,
    pub avg_bytes_per_s: f64,
    pub final_alive_neurons: usize,
    pub final_allocated_nodes: usize,
    pub final_synaptic_params: usize,
    pub final_val_bpc: f64,
    pub infer_b1_median_ms: f64,
    pub infer_b1_ms_per_byte: f64,
    pub rss_mib: f64,
    pub peak_rss_mib: f64,
    pub milestones: Vec<Milestone>,
    pub stage_samples: Vec<(String, Vec<GenSample>)>,
    pub checkpoint_path: Option<String>,
    pub checkpoint_bytes: u64,
}

impl MegaCore {
    /// コーパス先頭 4MB の PPMI-SVD から D=256 感覚基底と自己組織化文字希少度を構築する。
    pub fn from_corpus(bytes: &[u8], cap: usize) -> Result<Self> {
        ensure!(bytes.len() >= 1024 && cap >= VOCAB);
        let sample = &bytes[..bytes.len().min(4_000_000)];
        let mut counts = [0f32; VOCAB];
        for &b in sample {
            counts[b as usize] += 1.0;
        }
        let mut cooc = nalgebra::DMatrix::<f64>::zeros(VOCAB, VOCAB);
        for (dist, weight) in [(1usize, 1.0f64), (2, 0.5)] {
            for (&a, &b) in sample.iter().zip(sample.iter().skip(dist)) {
                let (i, j) = (a as usize, b as usize);
                if i != 0 && j != 0 {
                    cooc[(i, j)] += weight;
                    cooc[(j, i)] += weight;
                }
            }
        }
        let totals: Vec<f64> = (0..VOCAB).map(|i| cooc.row(i).sum().max(1.0)).collect();
        let total = cooc.sum().max(1.0);
        for i in 0..VOCAB {
            for j in 0..VOCAB {
                cooc[(i, j)] = (cooc[(i, j)] * total / (totals[i] * totals[j]) + 1e-8)
                    .ln()
                    .max(0.0);
            }
            cooc[(i, i)] += 1e-4;
        }
        // f64 対称固有値分解（ゼロ行正則化により NaN・発散を回避）
        let eig = cooc.symmetric_eigen();
        let mut order: Vec<usize> = (0..VOCAB).collect();
        order.sort_by(|&a, &b| {
            eig.eigenvalues[b]
                .abs()
                .total_cmp(&eig.eigenvalues[a].abs())
        });
        let mut basis = vec![0.0f32; VOCAB * DIM];
        for i in 1..VOCAB {
            let mut norm2 = 0.0f64;
            for (j, &col) in order.iter().take(DIM).enumerate() {
                let s = eig.eigenvalues[col].abs().max(1e-6).sqrt();
                let v = eig.eigenvectors[(i, col)] * s;
                let vf = if v.is_finite() { v } else { 0.0 };
                basis[i * DIM + j] = vf as f32;
                norm2 += vf * vf;
            }
            let scale = (1.3 / norm2.sqrt().max(1e-5)) as f32;
            for j in 0..DIM {
                basis[i * DIM + j] *= scale;
            }
        }
        ensure!(basis.iter().all(|x| x.is_finite()), "basis non-finite");
        let count = sample.len() as f32;
        let bias = std::array::from_fn(|i| ((counts[i] + 1.0) / (count + VOCAB as f32)).ln());

        // Frequency-based Unicode character salience (without fixed word lists)
        let mut cfreq = vec![0u32; CHAR_SAL_SIZE];
        let mut total_chars = 0u64;
        let text_sample = String::from_utf8_lossy(sample);
        for ch in text_sample.chars() {
            let slot = (ch as usize) & (CHAR_SAL_SIZE - 1);
            cfreq[slot] = cfreq[slot].saturating_add(1);
            total_chars += 1;
        }
        let distinct = cfreq.iter().filter(|&&c| c > 0).count().max(1) as f32;
        let avg_freq = (total_chars as f32 / distinct).max(1.0);
        let mut char_sal = vec![0.0f32; CHAR_SAL_SIZE];
        for (i, &c) in cfreq.iter().enumerate() {
            char_sal[i] = (avg_freq / (c as f32 + 5.0)).powf(0.50).clamp(0.05, 3.0);
        }
        char_sal[0] = 0.0;
        char_sal[b'\n' as usize] = 0.0;
        char_sal[b' ' as usize] = 0.0;
        char_sal[b':' as usize] = 0.0;

        let buckets = ((cap / 2).max(262_144)).next_power_of_two();
        let init_nodes = VOCAB;
        let mut g = Self {
            cap,
            buckets,
            len: init_nodes,
            alive_count: 0,
            w: vec![0u16; init_nodes * DIM],
            sharp_dst: vec![-1i16; init_nodes * SHARP_K],
            sharp_w: vec![0u16; init_nodes * SHARP_K],
            parent: vec![0u32; init_nodes],
            span: vec![0u8; init_nodes],
            depth: vec![0u8; init_nodes],
            alive: vec![1u8; init_nodes],
            recent: vec![0u16; init_nodes],
            updates: vec![0u32; init_nodes],
            utility: vec![0.0f32; init_nodes],
            pain: vec![0.0f32; init_nodes],
            free: Vec::new(),
            sigs: vec![0u64; buckets * 4],
            children: vec![0u32; buckets * 4],
            basis,
            bias,
            char_sal,
            rng: 0x9e37_79b9_7f4a_7c15,
        };
        g.reserve_nodes(cap.min(1_000_000));
        Ok(g)
    }

    fn reserve_nodes(&mut self, target: usize) {
        if target <= self.len {
            return;
        }
        let add = target - self.len;
        self.w.reserve(add * DIM);
        self.sharp_dst.reserve(add * SHARP_K);
        self.sharp_w.reserve(add * SHARP_K);
        self.parent.reserve(add);
        self.span.reserve(add);
        self.depth.reserve(add);
        self.alive.reserve(add);
        self.recent.reserve(add);
        self.updates.reserve(add);
        self.utility.reserve(add);
        self.pain.reserve(add);
    }

    #[inline]
    fn alloc_node(&mut self, parent: u32, span: u8, depth: u8) -> Option<u32> {
        if let Some(id) = self.free.pop() {
            let idx = id as usize;
            self.w[idx * DIM..(idx + 1) * DIM].fill(0);
            self.sharp_dst[idx * SHARP_K..(idx + 1) * SHARP_K].fill(-1);
            self.sharp_w[idx * SHARP_K..(idx + 1) * SHARP_K].fill(0);
            self.parent[idx] = parent;
            self.span[idx] = span;
            self.depth[idx] = depth;
            self.alive[idx] = 1;
            self.recent[idx] = 1;
            self.updates[idx] = 0;
            self.utility[idx] = 0.05;
            self.pain[idx] = 0.0;
            self.alive_count += 1;
            return Some(id);
        }
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
        self.recent.push(1);
        self.updates.push(0);
        self.utility.push(0.05);
        self.pain.push(0.0);
        self.alive_count += 1;
        Some(id)
    }

    /// 局所条件（span<2は文字内バイト、span>=2は完了済みUnicode文字）を返す。
    #[inline]
    fn local_cond(stream: &[u8], t: usize, span: usize, ctx: &StreamCtx) -> Option<u64> {
        if span < 2 {
            t.checked_sub(span + 1)
                .map(|i| stream[i] as u64)
                .filter(|&c| c > 0)
        } else {
            ctx.prev_char(span - 1).map(|cp| 0x100 + cp as u64)
        }
    }

    /// 1ストリームの1時刻についてマルチスケール DAG 探索を行い、活性 h と sharp ロジットを返す。
    #[inline]
    fn forward_one(
        &self,
        stream: &[u8],
        t: usize,
        ctx: &mut StreamCtx,
        h_out: &mut [f32],
        logit_out: &mut [f32],
        vis_out: &mut Vec<(u8, u32)>,
    ) -> u32 {
        let cur_byte = stream[t];
        ctx.observe_byte(cur_byte, &self.char_sal);
        let root = cur_byte as u32;
        vis_out.clear();
        vis_out.push((0, root));
        let mut front = [Some(root), None, None, None];
        let mut leaf = root;
        let mask = self.buckets - 1;
        let top0 = ctx.top_topic(0).map(|(bg, _)| bg);
        let top1 = ctx.top_topic(1).map(|(bg, _)| bg);

        for hop in 1..=HOPS {
            let mut next = [None; 4];
            let mut any = false;
            let topic_c = if hop <= 6 {
                if hop % 2 == 1 { top0 } else { top1.or(top0) }
            } else {
                None
            };
            for (slot, parent) in front.into_iter().enumerate() {
                let Some(p) = parent else { continue };
                let loc = Self::local_cond(stream, t, self.span[p as usize] as usize, ctx);
                for (k, cond) in [loc, topic_c].into_iter().enumerate() {
                    if let Some(c) = cond {
                        let s = sig64(p, k, c);
                        let base = ((s as usize) & mask) * 4;
                        let row = &self.sigs[base..base + 4];
                        for m in 0..4 {
                            if row[m] == s {
                                let id = self.children[base + m];
                                next[(2 * slot + k) % 4] = Some(id);
                                any = true;
                                if k == 0 && self.span[id as usize] > self.span[leaf as usize] {
                                    leaf = id;
                                }
                                break;
                            }
                        }
                    }
                }
            }
            if !any {
                break;
            }
            for (j, id) in next.into_iter().enumerate() {
                if let Some(id) = id {
                    vis_out.push(((1 + (hop - 1) * 4 + j) as u8, id));
                }
            }
            front = next;
        }
        h_out.fill(0.0);
        logit_out.copy_from_slice(&self.bias);
        for &(col, n) in vis_out.iter() {
            let g = gain(col as usize);
            let wbase = n as usize * DIM;
            let wslice = &self.w[wbase..wbase + DIM];
            for j in 0..DIM {
                h_out[j] += from_bf16(wslice[j]) * g;
            }
            let sbase = n as usize * SHARP_K;
            let dst = &self.sharp_dst[sbase..sbase + SHARP_K];
            let sw = &self.sharp_w[sbase..sbase + SHARP_K];
            let g_sharp = g * 0.65;
            for j in 0..SHARP_K {
                let d = dst[j];
                if d >= 0 {
                    logit_out[d as usize] += from_bf16(sw[j]) * g_sharp;
                }
            }
        }
        for v in h_out.iter_mut() {
            *v = (*v * 0.5).tanh() * 2.0;
        }
        leaf
    }

    /// 複数ストリームを並列処理し、SGEMM・適応減衰付きヘブ学習・有糸分裂を実行する。
    pub fn run_chunk(&mut self, bytes: &[u8], batch: usize, train: bool) -> Result<(usize, f64)> {
        ensure!(bytes.len() >= 64 && batch > 0);
        let b = batch.min((bytes.len() / 32).max(1));
        let len = bytes.len() / b;
        let mut contexts = vec![StreamCtx::default(); b];
        let mut h_mat = vec![0.0f32; b * DIM];
        let mut logits = vec![0.0f32; b * VOCAB];
        let mut diff = vec![0.0f32; b * VOCAB];
        let mut err_mat = vec![0.0f32; b * DIM];
        let mut vis_all: Vec<Vec<(u8, u32)>> = (0..b).map(|_| Vec::with_capacity(80)).collect();
        let mut leaves = vec![0u32; b];
        let mut targets = vec![0usize; b];
        let mut bads = vec![0usize; b];
        let mut surprises = vec![0.0f32; b];
        let mut nll = 0.0f64;

        let n_threads = rayon::current_num_threads().max(1);
        let row_chunk = b.div_ceil(n_threads).max(16);
        let max_births = (b / 4).max(MAX_BIRTHS_PER_STEP);
        let mut buckets_by_tid: Vec<Vec<(u32, u16, u8)>> = (0..n_threads)
            .map(|_| Vec::with_capacity((b * 28) / n_threads + 64))
            .collect();
        let mut births: Vec<(u32, usize, u64)> = Vec::with_capacity(max_births);

        for t in 0..len - 1 {
            // 1) Parallel DAG traversal + BF16 accumulate
            let graph_ref: &Self = &*self;
            contexts
                .par_iter_mut()
                .zip(h_mat.par_chunks_mut(DIM))
                .zip(logits.par_chunks_mut(VOCAB))
                .zip(vis_all.par_iter_mut())
                .zip(leaves.par_iter_mut())
                .zip(targets.par_iter_mut())
                .enumerate()
                .for_each(|(i, (((((ctx, h_row), log_row), vis), leaf_slot), tgt))| {
                    let stream = &bytes[i * len..(i + 1) * len];
                    *leaf_slot = graph_ref.forward_one(stream, t, ctx, h_row, log_row, vis);
                    *tgt = stream[t + 1] as usize;
                });

            // 2) Parallel batched SGEMM: logits += H @ basis^T
            let basis_ref = &self.basis;
            h_mat
                .par_chunks(row_chunk * DIM)
                .zip(logits.par_chunks_mut(row_chunk * VOCAB))
                .for_each(|(hc, lc)| {
                    let m = hc.len() / DIM;
                    unsafe {
                        matrixmultiply::sgemm(
                            m,
                            DIM,
                            VOCAB,
                            1.0,
                            hc.as_ptr(),
                            DIM as isize,
                            1,
                            basis_ref.as_ptr(),
                            1,
                            DIM as isize,
                            1.0,
                            lc.as_mut_ptr(),
                            VOCAB as isize,
                            1,
                        );
                    }
                });

            // 3) Parallel softmax + NLL + diff = onehot - prob
            let step_nll: f64 = logits
                .par_chunks_mut(VOCAB)
                .zip(diff.par_chunks_mut(VOCAB))
                .zip(targets.par_iter())
                .zip(bads.par_iter_mut())
                .zip(surprises.par_iter_mut())
                .map(|((((prob, drow), &target), bad_out), surp_out)| {
                    let top = prob.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for p in prob.iter_mut() {
                        *p = (*p - top).exp();
                        sum += *p;
                    }
                    let inv = 1.0 / sum;
                    let mut bad = usize::from(target == 0);
                    let mut bad_p = -1.0f32;
                    for (j, p) in prob.iter_mut().enumerate() {
                        *p *= inv;
                        drow[j] = -*p;
                        if j != target && *p > bad_p {
                            bad_p = *p;
                            bad = j;
                        }
                    }
                    drow[target] += 1.0;
                    *bad_out = bad;
                    let pt = prob[target];
                    *surp_out = 1.0 - pt;
                    -(pt.max(1e-30) as f64).log2()
                })
                .sum();
            nll += step_nll;

            if !train {
                continue;
            }

            // 4) Parallel batched SGEMM: err = diff @ basis, scaled by tanh'
            diff.par_chunks(row_chunk * VOCAB)
                .zip(err_mat.par_chunks_mut(row_chunk * DIM))
                .zip(h_mat.par_chunks(row_chunk * DIM))
                .for_each(|((dc, ec), hc)| {
                    let m = dc.len() / VOCAB;
                    unsafe {
                        matrixmultiply::sgemm(
                            m,
                            VOCAB,
                            DIM,
                            1.0,
                            dc.as_ptr(),
                            VOCAB as isize,
                            1,
                            basis_ref.as_ptr(),
                            DIM as isize,
                            1,
                            0.0,
                            ec.as_mut_ptr(),
                            DIM as isize,
                            1,
                        );
                    }
                    for k in 0..m * DIM {
                        let hv = hc[k] * 0.5;
                        ec[k] *= (1.0 - hv * hv).max(0.15);
                    }
                });

            // 5) Parallel Hebbian update partitioned by disjoint node ID (nid % n_threads == tid)
            let inv_sqrt_d = 1.0 / (DIM as f32).sqrt();
            for v in &mut buckets_by_tid {
                v.clear();
            }
            for i in 0..b {
                for &(col, nid) in &vis_all[i] {
                    buckets_by_tid[(nid as usize) % n_threads].push((nid, i as u16, col));
                }
                if surprises[i] > 0.48 {
                    self.pain[leaves[i] as usize] += surprises[i];
                }
            }
            let ptrs = DisjointArrays {
                w: self.w.as_mut_ptr(),
                sharp_dst: self.sharp_dst.as_mut_ptr(),
                sharp_w: self.sharp_w.as_mut_ptr(),
                updates: self.updates.as_mut_ptr(),
                recent: self.recent.as_mut_ptr(),
                utility: self.utility.as_mut_ptr(),
            };
            let base_seed = self.rng.wrapping_add(t as u64 * 0x9e37_79b9_7f4a_7c15);
            self.rng = base_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let err_ref = &err_mat;
            let logits_ref = &logits;
            let targets_ref = &targets;
            let bads_ref = &bads;
            let surprises_ref = &surprises;
            buckets_by_tid
                .par_iter()
                .enumerate()
                .for_each(|(tid, jobs)| {
                    let ptrs = ptrs;
                    let mut rng8 = std::array::from_fn(|lane| {
                        let z = mix(base_seed ^ ((tid as u64) << 8) ^ (lane as u64 + 1));
                        (z as u32).max(1)
                    });
                    let mut rng_s = mix(base_seed ^ (tid as u64 + 101)).max(1);
                    for &(nid, bi, col) in jobs {
                        let i = bi as usize;
                        let idx = nid as usize;
                        let err_row = &err_ref[i * DIM..(i + 1) * DIM];
                        let target = targets_ref[i] as i16;
                        let bad = bads_ref[i] as i16;
                        let bad_prob = logits_ref[i * VOCAB + bad as usize];
                        let surp = surprises_ref[i];
                        let g = gain(col as usize);
                        unsafe {
                            let up_ptr = ptrs.updates.add(idx);
                            let up = (*up_ptr).saturating_add(1);
                            *up_ptr = up;
                            let rec_ptr = ptrs.recent.add(idx);
                            *rec_ptr = (*rec_ptr).saturating_add(1);
                            let lr = 0.24 * g / (1.0 + 0.02 * up as f32).sqrt();
                            let wslice = std::slice::from_raw_parts_mut(ptrs.w.add(idx * DIM), DIM);
                            let align = hebb_update_bf16_256(wslice, err_row, lr, &mut rng8);
                            *ptrs.utility.add(idx) += (align * inv_sqrt_d + 0.05).max(0.0) * 0.25;

                            // Adaptive root-protected sharp learning rate (reduces recency overwrite)
                            let slr = 0.30 * g / (1.0 + 0.005 * up as f32).sqrt();
                            let dst = std::slice::from_raw_parts_mut(
                                ptrs.sharp_dst.add(idx * SHARP_K),
                                SHARP_K,
                            );
                            let sw = std::slice::from_raw_parts_mut(
                                ptrs.sharp_w.add(idx * SHARP_K),
                                SHARP_K,
                            );
                            if let Some(pos) = dst.iter().position(|&v| v == target) {
                                let nv = (from_bf16(sw[pos]) + slr * surp).clamp(-3.5, 4.0);
                                sw[pos] = to_bf16_sr(nv, &mut rng_s);
                            } else if target > 0 && surp > 0.30 {
                                let pos = dst.iter().position(|&v| v < 0).unwrap_or_else(|| {
                                    (0..SHARP_K)
                                        .min_by(|&a, &c| {
                                            from_bf16(sw[a])
                                                .abs()
                                                .total_cmp(&from_bf16(sw[c]).abs())
                                        })
                                        .unwrap()
                                });
                                dst[pos] = target;
                                sw[pos] = to_bf16_sr((slr * surp).clamp(0.05, 1.5), &mut rng_s);
                            }
                            if bad_prob > 0.08 {
                                if let Some(pos) = dst.iter().position(|&v| v == bad) {
                                    let nv = (from_bf16(sw[pos]) - slr * 0.8 * bad_prob)
                                        .clamp(-3.5, 4.0);
                                    sw[pos] = to_bf16_sr(nv, &mut rng_s);
                                }
                            }
                        }
                    }
                });

            // 6) Parallel batched SGEMM basis update
            let rate = 0.04 / b as f32;
            let mut bias_delta = [0.0f32; VOCAB];
            for row in diff.chunks_exact(VOCAB) {
                for j in 0..VOCAB {
                    bias_delta[j] += row[j];
                }
            }
            for j in 0..VOCAB {
                self.bias[j] += rate * 0.4 * bias_delta[j];
            }
            let h_ref = &h_mat;
            let diff_ref = &diff;
            self.basis
                .par_chunks_mut(32 * DIM)
                .enumerate()
                .for_each(|(blk, bc)| {
                    let row0 = blk * 32;
                    let m = bc.len() / DIM;
                    unsafe {
                        matrixmultiply::sgemm(
                            m,
                            b,
                            DIM,
                            rate,
                            diff_ref.as_ptr().add(row0),
                            1,
                            VOCAB as isize,
                            h_ref.as_ptr(),
                            DIM as isize,
                            1,
                            1.0,
                            bc.as_mut_ptr(),
                            DIM as isize,
                            1,
                        );
                    }
                });
            if t % 16 == 0 {
                for x in &mut self.basis {
                    *x *= 0.9995;
                }
                self.basis[0..DIM].fill(0.0);
            }

            // 7) Mitosis (Neurogenesis): local byte/char hops + salient bigram topic hops
            births.clear();
            for i in 0..b {
                let lf = leaves[i];
                let li = lf as usize;
                if surprises[i] <= 0.48 || self.pain[li] < 2.2 || self.depth[li] as usize >= HOPS {
                    continue;
                }
                let ctx = &contexts[i];
                let long_cand = if self.span[li] >= 2 && t % 3 == 0 {
                    ctx.top_topic(t % 2).filter(|&(_, sc)| sc > 1.0).map(|(bg, _)| bg)
                } else {
                    None
                };
                if let Some(bg) = long_cand {
                    births.push((lf, 1, bg));
                } else if let Some(c) =
                    Self::local_cond(&bytes[i * len..(i + 1) * len], t, self.span[li] as usize, ctx)
                {
                    births.push((lf, 0, c));
                }
                if births.len() == max_births {
                    break;
                }
            }
            let mask = self.buckets - 1;
            for &(p, k, c) in &births {
                let s = sig64(p, k, c);
                let base = ((s as usize) & mask) * 4;
                if self.sigs[base..base + 4].contains(&s) {
                    continue;
                }
                let Some(slot_off) = self.sigs[base..base + 4].iter().position(|&e| e == 0) else {
                    continue;
                };
                let new_span = self.span[p as usize].saturating_add(u8::from(k == 0));
                let new_depth = self.depth[p as usize].saturating_add(1);
                let Some(id) = self.alloc_node(p, new_span, new_depth) else {
                    break;
                };
                self.sigs[base + slot_off] = s;
                self.children[base + slot_off] = id;
                self.pain[p as usize] *= 0.35;
            }
        }
        let count = b * (len - 1);
        Ok((count, nll / count as f64))
    }

    /// 頭蓋容量に近づいた際の代謝的アポトーシス（発火不足の雑音葉を剪定して再利用）。
    pub fn gentle_apoptosis(&mut self, min_recent: u16) -> usize {
        let mut killed = 0usize;
        for idx in VOCAB..self.len {
            if self.alive[idx] != 0 && self.recent[idx] < min_recent && self.utility[idx] < 0.08 {
                self.alive[idx] = 0;
                killed += 1;
            }
        }
        for _ in 0..HOPS {
            let mut changed = false;
            for idx in VOCAB..self.len {
                if self.alive[idx] != 0 && self.alive[self.parent[idx] as usize] == 0 {
                    self.alive[idx] = 0;
                    killed += 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if killed > 0 {
            self.free.clear();
            let mut alive_now = 0usize;
            for idx in VOCAB..self.len {
                if self.alive[idx] == 0 {
                    self.free.push(idx as u32);
                } else {
                    alive_now += 1;
                }
            }
            self.alive_count = alive_now;
            for m in 0..self.sigs.len() {
                if self.sigs[m] != 0 && self.alive[self.children[m] as usize] == 0 {
                    self.sigs[m] = 0;
                    self.children[m] = 0;
                }
            }
        }
        for idx in VOCAB..self.len {
            self.recent[idx] /= 2;
            self.utility[idx] *= 0.90;
        }
        killed
    }

    /// B=1 推論（自由生成）。UTF-8 境界を保ちつつ 1 バイトずつ自己回帰生成する。
    pub fn generate(
        &self,
        prompt: &[u8],
        max_bytes: usize,
        temperature: f32,
        mut seed: u64,
    ) -> Vec<u8> {
        let mut seq = prompt.to_vec();
        let mut ctx = StreamCtx::default();
        let mut h = [0.0f32; DIM];
        let mut logits = [0.0f32; VOCAB];
        let mut vis = Vec::with_capacity(80);
        for t in 0..seq.len() {
            let _ = self.forward_one(&seq, t, &mut ctx, &mut h, &mut logits, &mut vis);
        }
        let mut out = Vec::with_capacity(max_bytes);
        for _ in 0..max_bytes {
            for j in 0..VOCAB {
                let row: &[f32; DIM] = self.basis[j * DIM..(j + 1) * DIM].try_into().unwrap();
                logits[j] += dot(&h, row);
            }
            let remaining = max_bytes - out.len();
            for b in 0..VOCAB {
                if b == 0 || !legal_utf8(&out, b as u8, remaining) {
                    logits[b] = f32::NEG_INFINITY;
                }
            }
            let next = if temperature <= 0.0 {
                (0..VOCAB)
                    .reduce(|a, b| if logits[b] > logits[a] { b } else { a })
                    .unwrap() as u8
            } else {
                let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = [0.0f64; VOCAB];
                let mut sum = 0.0f64;
                for j in 0..VOCAB {
                    if logits[j].is_finite() {
                        let p = (((logits[j] - peak) / temperature) as f64).exp();
                        probs[j] = p;
                        sum += p;
                    }
                }
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut r = ((seed >> 11) as f64 / (1u64 << 53) as f64) * sum;
                let mut pick = 0u8;
                for j in 0..VOCAB {
                    r -= probs[j];
                    if r <= 0.0 {
                        pick = j as u8;
                        break;
                    }
                }
                pick
            };
            out.push(next);
            seq.push(next);
            let _ = self.forward_one(&seq, seq.len() - 1, &mut ctx, &mut h, &mut logits, &mut vis);
            if out.ends_with(b"\n\n") || (out.ends_with("\n人:".as_bytes()) && out.len() > 24) {
                break;
            }
        }
        out
    }

    /// バイナリチェックポイントを書き出す（v18.5: NGMEGA02）。
    pub fn save_checkpoint(&self, path: &Path) -> Result<u64> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let f = File::create(path)?;
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, f);
        w.write_all(b"NGMEGA02")?;
        w.write_all(&(DIM as u32).to_le_bytes())?;
        w.write_all(&(HOPS as u32).to_le_bytes())?;
        w.write_all(&(self.len as u32).to_le_bytes())?;
        w.write_all(&(self.alive_count as u32).to_le_bytes())?;
        w.write_all(&(self.buckets as u32).to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(&self.basis))?;
        w.write_all(bytemuck::cast_slice(&self.bias))?;
        w.write_all(bytemuck::cast_slice(&self.char_sal))?;
        w.write_all(&self.alive)?;
        w.write_all(&self.span)?;
        w.write_all(&self.depth)?;
        w.write_all(bytemuck::cast_slice(&self.parent))?;
        w.write_all(bytemuck::cast_slice(&self.w))?;
        w.write_all(bytemuck::cast_slice(&self.sharp_dst))?;
        w.write_all(bytemuck::cast_slice(&self.sharp_w))?;
        let mut edge_count = 0u64;
        for &s in &self.sigs {
            if s != 0 {
                edge_count += 1;
            }
        }
        w.write_all(&edge_count.to_le_bytes())?;
        for m in 0..self.sigs.len() {
            if self.sigs[m] != 0 {
                w.write_all(&self.sigs[m].to_le_bytes())?;
                w.write_all(&self.children[m].to_le_bytes())?;
            }
        }
        w.flush()?;
        Ok(fs::metadata(path)?.len())
    }
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

/// メガ規模の成長実験を実行し、スケーリング曲線・対話生成サンプル・推論速度・チェックポイントを出力する。
pub fn run_mega_experiment(
    train_path: &Path,
    val_path: &Path,
    out_json: &Path,
    ckpt_path: Option<&Path>,
    cap: usize,
    batch: usize,
    max_train_bytes: usize,
) -> Result<MegaReport> {
    let train_all = fs::read(train_path)?;
    let train_bytes = if max_train_bytes > 0 {
        train_all.len().min(max_train_bytes)
    } else {
        train_all.len()
    };
    let train = &train_all[..train_bytes];
    let val = fs::read(val_path)?;

    let prompts = [
        "人: こんにちは！今日はちょっと疲れたよ。\n答: ",
        "人: 日本の首都はどこですか？\n答: ",
        "人: 富士山について簡単に教えてください。\n答: ",
        "人: 人工知能とは何ですか？分かりやすく説明して。\n答: ",
        "人: 集中力を上げるコツを3つ教えてください。\n答: ",
        "人: カボチャを使ったおすすめ料理はありますか？\n答: ",
    ];
    let sample_gen = |g: &MegaCore| -> Vec<GenSample> {
        prompts
            .iter()
            .map(|&p| {
                let g_out = g.generate(p.as_bytes(), 180, 0.0, 42);
                let s_out = g.generate(p.as_bytes(), 180, 0.65, 7);
                GenSample {
                    prompt: p.to_owned(),
                    greedy: String::from_utf8_lossy(&g_out).into_owned(),
                    sampled: String::from_utf8_lossy(&s_out).into_owned(),
                }
            })
            .collect()
    };

    let t0 = Instant::now();
    let mut g = MegaCore::from_corpus(train, cap)?;
    let val_mid = &val[..val.len().min(524_288)];
    let (_, init_val_bpc) = g.run_chunk(val_mid, batch.min(64), false)?;
    let (rss0, peak0) = crate::memory::working_set();
    eprintln!(
        "[mega-v18.5] init done in {:.2}s | cap={} buckets={} hops={} val_bpc={:.4} rss={:.1} MiB",
        t0.elapsed().as_secs_f64(),
        cap,
        g.buckets,
        HOPS,
        init_val_bpc,
        rss0
    );

    let mut milestones = vec![Milestone {
        bytes_seen: 0,
        elapsed_s: t0.elapsed().as_secs_f64(),
        chunk_bytes_per_s: 0.0,
        alive_neurons: 0,
        allocated_nodes: g.len,
        synaptic_params: g.len * (DIM + SHARP_K),
        chunk_train_bpc: init_val_bpc,
        val_bpc: init_val_bpc,
        rss_mib: rss0,
        peak_rss_mib: peak0,
        apoptosis_events: 0,
    }];
    let mut stage_samples = Vec::new();
    let mut apoptosis_events = 0usize;

    let mut boundaries: Vec<usize> = vec![
        5_000_000,
        15_000_000,
        40_000_000,
        80_000_000,
        130_000_000,
        190_000_000,
        260_000_000,
        340_000_000,
        train_bytes,
    ]
    .into_iter()
    .filter(|&x| x <= train_bytes)
    .collect();
    if boundaries.last().copied() != Some(train_bytes) {
        boundaries.push(train_bytes);
    }
    boundaries.dedup();

    let mut pos = 0usize;
    let mut next_sample_stage = 0usize;
    let sample_thresholds = [15_000_000usize, 130_000_000, train_bytes];

    for end in boundaries {
        let slice = &train[pos..end];
        if slice.len() < 1024 {
            continue;
        }
        let ct = Instant::now();
        let (_, train_bpc) = g.run_chunk(slice, batch, true)?;
        let csec = ct.elapsed().as_secs_f64();
        if g.alive_count > (cap * 90) / 100 && end < train_bytes {
            let killed = g.gentle_apoptosis(2);
            apoptosis_events += 1;
            eprintln!(
                "[mega-v18.5] metabolic apoptosis pruned {killed} weak nodes -> alive={}",
                g.alive_count
            );
        }
        let val_slice = if end == train_bytes { &val[..] } else { val_mid };
        let (_, val_bpc) = g.run_chunk(val_slice, batch.min(64), false)?;
        let (rss, peak) = crate::memory::working_set();
        let bps = slice.len() as f64 / csec;
        eprintln!(
            "[mega-v18.5] {:6.1} MB | {:.1}s ({:.0} B/s) | alive={:10} | train_bpc={:.4} | val_bpc={:.4} | rss={:.0} MiB",
            end as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
            bps,
            g.alive_count,
            train_bpc,
            val_bpc,
            rss
        );
        milestones.push(Milestone {
            bytes_seen: end,
            elapsed_s: t0.elapsed().as_secs_f64(),
            chunk_bytes_per_s: bps,
            alive_neurons: g.alive_count,
            allocated_nodes: g.len,
            synaptic_params: g.alive_count * (DIM + SHARP_K),
            chunk_train_bpc: train_bpc,
            val_bpc,
            rss_mib: rss,
            peak_rss_mib: peak,
            apoptosis_events,
        });
        while next_sample_stage < sample_thresholds.len()
            && end >= sample_thresholds[next_sample_stage]
        {
            let label = format!(
                "{:.0}MB / {:.2}M neurons",
                end as f64 / 1e6,
                g.alive_count as f64 / 1e6
            );
            stage_samples.push((label, sample_gen(&g)));
            next_sample_stage += 1;
        }
        pos = end;
    }

    // Measure B=1 inference speed (75B prefill + 60B generation, 9 runs median)
    let pref = &val[..val.len().min(75)];
    let _ = g.generate(pref, 60, 0.0, 1);
    let mut lats = Vec::with_capacity(9);
    for _ in 0..9 {
        let st = Instant::now();
        let _ = std::hint::black_box(g.generate(pref, 60, 0.0, 1));
        lats.push(st.elapsed().as_secs_f64() * 1000.0);
    }
    lats.sort_by(f64::total_cmp);
    let infer_ms = lats[lats.len() / 2];

    let mut ckpt_bytes = 0u64;
    let mut ckpt_str = None;
    if let Some(cp) = ckpt_path {
        let cs = Instant::now();
        ckpt_bytes = g.save_checkpoint(cp)?;
        ckpt_str = Some(cp.display().to_string());
        eprintln!(
            "[mega-v18.5] saved checkpoint {} ({:.2} GiB) in {:.1}s",
            cp.display(),
            ckpt_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            cs.elapsed().as_secs_f64()
        );
    }

    let total_elapsed_s = t0.elapsed().as_secs_f64();
    let (rss_mib, peak_rss_mib) = crate::memory::working_set();
    let final_val_bpc = milestones.last().map(|m| m.val_bpc).unwrap_or(init_val_bpc);
    let report = MegaReport {
        cap,
        buckets: g.buckets,
        dim: DIM,
        hops: HOPS,
        batch,
        threads: rayon::current_num_threads(),
        train_bytes,
        val_bytes: val.len(),
        total_elapsed_s,
        avg_bytes_per_s: train_bytes as f64 / total_elapsed_s,
        final_alive_neurons: g.alive_count,
        final_allocated_nodes: g.len,
        final_synaptic_params: g.alive_count * (DIM + SHARP_K),
        final_val_bpc,
        infer_b1_median_ms: infer_ms,
        infer_b1_ms_per_byte: infer_ms / 60.0,
        rss_mib,
        peak_rss_mib,
        milestones,
        stage_samples,
        checkpoint_path: ckpt_str,
        checkpoint_bytes: ckpt_bytes,
    };
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
    fn bf16_sr_unbiased_and_mega_grows() -> Result<()> {
        let mut rng = 123456789u64;
        let mut acc = 0.0f64;
        for _ in 0..4000 {
            acc += from_bf16(to_bf16_sr(1.001, &mut rng)) as f64;
        }
        let mean = acc / 4000.0;
        assert!((mean - 1.001).abs() < 0.002, "mean={mean}");

        let text = "人: 日本の首都はどこですか？\n答: 日本の首都は東京です。\n\n人: 富士山はどこですか？\n答: 富士山は日本一高い山です。\n\n".repeat(120);
        let bytes = text.as_bytes();
        let mut g = MegaCore::from_corpus(bytes, 8192)?;
        let (_, bpc1) = g.run_chunk(bytes, 16, true)?;
        let (_, bpc2) = g.run_chunk(bytes, 16, true)?;
        assert!(g.alive_count > 30, "alive={}", g.alive_count);
        assert!(bpc2 < bpc1, "bpc did not drop: {bpc1} -> {bpc2}");
        let generated = g.generate("人: 日本の首都はどこですか？\n答: ".as_bytes(), 45, 0.0, 1);
        assert!(std::str::from_utf8(&generated).is_ok());
        Ok(())
    }
}
