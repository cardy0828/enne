//! v0.8 の複数ストリーム学習。読み出しを並列化し、更新順を固定する。
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::Path,
};
const VOCAB: usize = 256;

const HOPS: usize = 12;
const BUCKETS: usize = 262144;
/// 符号なしの折り返し乗算で PyTorch の 64 bit ハッシュを再現する。
pub fn mix(mut h: u64) -> u64 {
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}
/// 親・条件種別・バイト値から非ゼロ署名を作る。
fn sig(p: usize, k: usize, c: u8) -> u64 {
    mix(mix(mix(p as u64 + 11) ^ ((k as u64 + 1) * 1000003)) ^ (c as u64 + 2000029)).max(1)
}
#[derive(Clone)]
struct Node<const DIM: usize> {
    w: [f32; DIM],
    dst: [i16; 8],
    sharp: [f32; 8],
    parent: usize,
    span: usize,
    depth: usize,
    fires: u32,
    recent: u32,
    updates: u32,
    utility: f32,
    pain: f32,
    alive: bool,
}
impl<const DIM: usize> Default for Node<DIM> {
    /// 新規感覚ノードの可塑・代謝状態を初期化する。
    fn default() -> Self {
        Self {
            w: [0.; DIM],
            dst: [-1; 8],
            sharp: [0.; 8],
            parent: 0,
            span: 0,
            depth: 0,
            fires: 0,
            recent: 0,
            updates: 0,
            utility: 0.,
            pain: 0.,
            alive: true,
        }
    }
}
pub struct NeurogenesisGraph<const DIM: usize = 128> {
    nodes: Vec<Node<DIM>>,
    free: BTreeSet<usize>,
    cap: usize,
    table: Vec<[(u64, usize); 4]>,
    basis: Vec<[f32; DIM]>,
    bias: [f32; VOCAB],
    sal: [f32; VOCAB],
}
#[derive(Default, Clone)]
struct Context {
    slots: [Option<u8>; 2],
    score: [f32; 2],
}
struct Step<const DIM: usize> {
    vis: Vec<(usize, usize)>,
    h: [f32; DIM],
    err: [f32; DIM],
    prob: [f32; VOCAB],
    target: usize,
    bad: usize,
    surprise: f32,
    leaf: usize,
}
impl<const DIM: usize> NeurogenesisGraph<DIM> {
    /// Python と共通の SVD 初期基底・バイアス・希少度から、隠れ細胞ゼロで開始する。
    pub fn load(path: &Path, cap: usize) -> Result<Self> {
        let b = fs::read(path)?;
        ensure!(b.len() == (VOCAB * DIM + 2 * VOCAB) * 4 && cap >= VOCAB);
        let v: Vec<_> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let basis = v[..VOCAB * DIM]
            .chunks_exact(DIM)
            .map(|row| row.try_into().unwrap())
            .collect();
        Ok(Self::new(
            cap,
            basis,
            v[VOCAB * DIM..VOCAB * (DIM + 1)].try_into().unwrap(),
            v[VOCAB * (DIM + 1)..].try_into().unwrap(),
        ))
    }
    /// 共起窓1,2からPPMIを求め、SVDにより感覚基底を構築する。例 128次元・150000ノード。
    pub fn from_corpus(bytes: &[u8], cap: usize) -> Result<Self> {
        ensure!(!bytes.is_empty() && cap >= VOCAB && DIM > 0 && DIM <= VOCAB);
        let bytes = &bytes[..bytes.len().min(2_000_000)];
        let mut counts = [0f32; VOCAB];
        for &b in bytes {
            counts[b as usize] += 1.;
        }
        let mut cooc = nalgebra::DMatrix::<f32>::zeros(VOCAB, VOCAB);
        for (dist, weight) in [(1, 1.), (2, 0.5)] {
            for (&a, &b) in bytes.iter().zip(bytes.iter().skip(dist)) {
                let (i, j) = (a as usize, b as usize);
                if i != 0 && j != 0 {
                    cooc[(i, j)] += weight;
                    cooc[(j, i)] += weight;
                }
            }
        }
        let totals: Vec<f32> = (0..VOCAB).map(|i| cooc.row(i).sum().max(1.)).collect();
        let total = cooc.sum().max(1.);
        for i in 0..VOCAB {
            for j in 0..VOCAB {
                cooc[(i, j)] = (cooc[(i, j)] * total / (totals[i] * totals[j]) + 1e-8)
                    .ln()
                    .max(0.)
            }
        }
        let svd = cooc.svd(true, false);
        let u = svd.u.unwrap();
        let mut basis = vec![[0.; DIM]; VOCAB];
        for i in 1..VOCAB {
            for j in 0..DIM {
                basis[i][j] = u[(i, j)] * svd.singular_values[j].max(1e-6).sqrt()
            }
            let scale = 1.3 / dot(&basis[i], &basis[i]).sqrt().max(1e-5);
            for v in &mut basis[i] {
                *v *= scale
            }
        }
        let count = bytes.len() as f32;
        let bias = std::array::from_fn(|i| ((counts[i] + 1.) / (count + VOCAB as f32)).ln());
        let mut sal = std::array::from_fn(|i| {
            ((count / VOCAB as f32) / (counts[i] + 5.))
                .powf(0.55)
                .clamp(0.05, 3.)
        });
        sal[0] = 0.;
        Ok(Self::new(cap, basis, bias, sal))
    }
    /// 基底と出力バイアスから共通の初期構造を構築する。
    fn new(cap: usize, basis: Vec<[f32; DIM]>, bias: [f32; VOCAB], sal: [f32; VOCAB]) -> Self {
        Self {
            nodes: vec![Node::default(); VOCAB],
            free: BTreeSet::new(),
            cap,
            table: vec![[(0, 0); 4]; BUCKETS],
            basis,
            bias,
            sal,
        }
    }
    /// 遅延した文脈バイトと直近の履歴から最大4枝ずつ辿り、予測分布と局所誤差を作る。
    fn step(&self, stream: &[u8], t: usize, ctx: &mut Context) -> Step<DIM> {
        for s in &mut ctx.score {
            *s *= 0.965
        }
        if t >= 6 {
            let c = stream[t - 6];
            let i = usize::from(ctx.score[1] < ctx.score[0]);
            let sal = self.sal[c as usize];
            if c > 0 && !ctx.slots.contains(&Some(c)) && sal > ctx.score[i] && sal > 0.75 {
                ctx.slots[i] = Some(c);
                ctx.score[i] = sal;
            }
        }
        let context = ctx.slots[usize::from(ctx.score[1] > ctx.score[0])];
        let root = stream[t] as usize;
        let mut vis = vec![(0, root)];
        let mut front = [Some(root), None, None, None];
        let mut leaf = root;
        for hop in 1..=HOPS {
            let mut next = [None; 4];
            for (slot, parent) in front.into_iter().enumerate() {
                let Some(p) = parent else { continue };
                let dist = self.nodes[p].span + 1;
                let loc = t.checked_sub(dist).map(|i| stream[i]);
                for (k, cond) in [loc, context.filter(|_| hop <= 4)].into_iter().enumerate() {
                    if let Some(c) = cond.filter(|c| *c > 0) {
                        let s = sig(p, k, c);
                        if let Some(&(_, id)) = self.table[s as usize & (BUCKETS - 1)]
                            .iter()
                            .find(|e| e.0 == s)
                        {
                            next[(2 * slot + k) % 4] = Some(id);
                            if k == 0 && self.nodes[id].span > self.nodes[leaf].span {
                                leaf = id;
                            }
                        }
                    }
                }
            }
            if next.iter().all(Option::is_none) {
                break;
            }
            for (j, id) in next.into_iter().enumerate() {
                if let Some(id) = id {
                    vis.push((1 + (hop - 1) * 4 + j, id))
                }
            }
            front = next;
        }
        let mut h = [0.; DIM];
        let mut prob = self.bias;
        for &(col, n) in &vis {
            let g = gain(col);
            let node = &self.nodes[n];
            for (i, w) in node.w.iter().enumerate() {
                h[i] += g * w
            }
            for j in 0..8 {
                if node.dst[j] >= 0 {
                    prob[node.dst[j] as usize] += node.sharp[j] * g * 0.65
                }
            }
        }
        for v in &mut h {
            *v = (*v * 0.5).tanh() * 2.
        }
        for (i, p) in prob.iter_mut().enumerate() {
            *p += dot(&h, &self.basis[i]);
        }
        let top = prob.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.;
        for p in &mut prob {
            *p = (*p - top).exp();
            total += *p;
        }
        for p in &mut prob {
            *p /= total
        }
        let target = stream[t + 1] as usize;
        let mut err = self.basis[target];
        let mut bad = usize::from(target == 0);
        for (i, row) in self.basis.iter().enumerate() {
            if i != target && prob[i] > prob[bad] {
                bad = i
            }
            for j in 0..DIM {
                err[j] -= prob[i] * row[j];
            }
        }
        for j in 0..DIM {
            err[j] *= (1. - (h[j] * 0.5).powi(2)).max(0.15)
        }
        Step {
            vis,
            h,
            err,
            prob,
            target,
            bad,
            surprise: 1. - prob[target],
            leaf,
        }
    }
    /// 各時刻の全ストリームを同じ状態で評価し、列ごとに Hebbian 更新と分裂を行う。
    pub fn run_stream(
        &mut self,
        bytes: &[u8],
        batch: usize,
        train: bool,
    ) -> Result<(usize, f64, usize)> {
        ensure!(bytes.len() >= 2 && batch > 0);
        let b = batch.min((bytes.len() / 32).max(1));
        let len = bytes.len() / b;
        let mut contexts = vec![Context::default(); b];
        let mut nll = 0f64;
        for t in 0..len - 1 {
            let calc =
                |(i, c): (usize, &mut Context)| self.step(&bytes[i * len..(i + 1) * len], t, c);
            let steps: Vec<_> = if b < 16 {
                contexts.iter_mut().enumerate().map(calc).collect()
            } else {
                contexts.par_iter_mut().enumerate().map(calc).collect()
            };
            nll += steps
                .iter()
                .map(|s| -(s.prob[s.target] as f64).log2())
                .sum::<f64>();
            if !train {
                continue;
            }
            let mut columns: Vec<Vec<(usize, usize)>> = vec![vec![]; 1 + 4 * HOPS];
            for (i, s) in steps.iter().enumerate() {
                for &(col, n) in &s.vis {
                    columns[col].push((n, i));
                }
            }
            for (col, list) in columns.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
                let g = gain(col);
                for &(n, _) in list {
                    self.nodes[n].updates += 1
                }
                for &(id, i) in list {
                    let node = &mut self.nodes[id];
                    node.utility +=
                        (dot(&node.w, &steps[i].err) / (DIM as f32).sqrt() + 0.05).max(0.) * 0.25;
                }
                let mut first = HashSet::new();
                for &(id, i) in list {
                    let s = &steps[i];
                    let n = &mut self.nodes[id];
                    n.fires += 1;
                    n.recent += 1;

                    let lr = 0.24 * g / (1. + 0.02 * n.updates as f32).sqrt();
                    for j in 0..DIM {
                        n.w[j] += lr * s.err[j]
                    }
                    if first.insert(id) {
                        update_sharp(n, s, 0.30 * g);
                    }
                }
            }
            let rate = 0.04 / b as f32;
            let mut counts = [0f32; VOCAB];
            let mut probability_sum = [0f32; VOCAB];
            for step in &steps {
                counts[step.target] += 1.;
                for j in 0..VOCAB {
                    probability_sum[j] += step.prob[j];
                }
                for k in 0..DIM {
                    self.basis[step.target][k] += rate * step.h[k];
                }
                if step.surprise > 0.48 {
                    self.nodes[step.leaf].pain += step.surprise;
                }
            }
            for j in 0..VOCAB {
                self.bias[j] += rate * 0.4 * (counts[j] - probability_sum[j]);
                let mut delta = [0.; DIM];
                for step in &steps {
                    for k in 0..DIM {
                        delta[k] += step.prob[j] * step.h[k];
                    }
                }
                for k in 0..DIM {
                    self.basis[j][k] -= rate * delta[k];
                }
            }
            if t % 16 == 0 {
                for row in &mut self.basis {
                    for x in row {
                        *x *= 0.9995
                    }
                }
                self.basis[0] = [0.; DIM]
            }
            let mut births = vec![];
            for (i, s) in steps.iter().enumerate() {
                let p = &self.nodes[s.leaf];
                if s.surprise <= 0.48 || p.pain < 2.2 || p.depth >= HOPS {
                    continue;
                }
                let ctx = &contexts[i];
                let top = usize::from(ctx.score[1] > ctx.score[0]);
                let long = p.span >= 2 && ctx.score[top] > 1.1 && t % 3 == 0;
                let cond = if long {
                    ctx.slots[top]
                } else {
                    t.checked_sub(p.span + 1).map(|j| bytes[i * len + j])
                };
                if let Some(c) = cond.filter(|x| *x > 0) {
                    births.push((s.leaf, usize::from(long), c));
                    if births.len() == 64 {
                        break;
                    }
                }
            }
            let mut used = HashSet::new();
            let mut relieved = HashSet::new();
            let available: Vec<_> = self
                .free
                .iter()
                .copied()
                .chain(self.nodes.len()..self.cap)
                .take(births.len())
                .collect();
            for ((p, k, c), id) in births.into_iter().zip(available) {
                let s = sig(p, k, c);
                let bucket = s as usize & (BUCKETS - 1);
                if !used.insert(bucket) || self.table[bucket].iter().any(|e| e.0 == s) {
                    continue;
                }
                if let Some(slot) = self.table[bucket].iter_mut().find(|e| e.0 == 0) {
                    *slot = (s, id);
                    let node = Node {
                        parent: p,
                        span: self.nodes[p].span + usize::from(k == 0),
                        depth: self.nodes[p].depth + 1,
                        fires: 1,
                        recent: 1,
                        utility: 0.05,
                        ..Node::default()
                    };
                    while self.nodes.len() <= id {
                        self.free.insert(self.nodes.len());
                        self.nodes.push(Node {
                            alive: false,
                            ..Node::default()
                        })
                    }
                    self.nodes[id] = node;
                    self.free.remove(&id);
                    if relieved.insert(p) {
                        self.nodes[p].pain *= 0.35;
                    }
                }
            }
        }
        let count = b * (len - 1);
        Ok((
            count,
            nll / count as f64,
            self.nodes.len() - self.free.len() - VOCAB,
        ))
    }
    /// 発火不足・低効用ノードとその子孫を除去し、IDを再利用可能にする。
    pub fn apoptosis(&mut self) {
        for n in self.nodes.iter_mut().skip(VOCAB) {
            if n.recent < 3
                || (n.utility < 0.02
                    && dot(&n.w, &n.w) < 0.15 * 0.15
                    && n.sharp.iter().map(|w| w.abs()).sum::<f32>() < 0.15)
            {
                n.alive = false
            }
        }
        for _ in 0..HOPS {
            let mut changed = false;
            for id in VOCAB..self.nodes.len() {
                let p = self.nodes[id].parent;
                if self.nodes[id].alive && !self.nodes[p].alive {
                    self.nodes[id].alive = false;
                    changed = true
                }
            }
            if !changed {
                break;
            }
        }
        for (i, n) in self.nodes.iter_mut().enumerate() {
            if !n.alive {
                self.free.insert(i);
            }
            n.recent = 0;
            n.utility *= 0.85
        }
        for row in &mut self.table {
            for slot in row {
                if slot.0 != 0 && !self.nodes[slot.1].alive {
                    *slot = (0, 0)
                }
            }
        }
    }
}
/// 列番号から深さを復元して残差ゲインを返す。
fn gain(col: usize) -> f32 {
    1. / (1. + 0.15 * col.div_ceil(4) as f32)
}
/// 共有特徴ベクトル同士の内積を求める。
fn dot<const DIM: usize>(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// 正解を強化し、誤った上位候補の既存接続を抑制する。
fn update_sharp<const DIM: usize>(n: &mut Node<DIM>, s: &Step<DIM>, lr: f32) {
    let target = s.target as i16;
    if let Some(i) = n.dst.iter().position(|v| *v == target) {
        n.sharp[i] = (n.sharp[i] + lr * s.surprise).clamp(-3.5, 4.)
    } else if target > 0 && s.surprise > 0.30 {
        let i = n.dst.iter().position(|v| *v < 0).unwrap_or_else(|| {
            (0..8)
                .min_by(|&a, &b| n.sharp[a].abs().total_cmp(&n.sharp[b].abs()))
                .unwrap()
        });
        n.dst[i] = target;
        n.sharp[i] = (lr * s.surprise).clamp(0.05, 1.5)
    }
    if s.prob[s.bad] > 0.08 {
        if let Some(i) = n.dst.iter().position(|v| *v == s.bad as i16) {
            n.sharp[i] = (n.sharp[i] - lr * 0.8 * s.prob[s.bad]).clamp(-3.5, 4.)
        }
    }
}
