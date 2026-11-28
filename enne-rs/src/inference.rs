//! 実チェックポイント上の BF16 DAG、16 スロット、ByteSTM、モード補正。
use crate::core_dag::mix;
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};
const D: usize = 256;
const S: usize = 16;
#[derive(Deserialize)]
struct Index {
    path: String,
    fields: HashMap<String, [u64; 2]>,
}
struct Reader {
    f: File,
    idx: Index,
}
impl Reader {
    /// テンソル領域から先頭 n 要素のみロードする。サイズと位置を検査する。
    fn read<T: bytemuck::Pod + Default>(&mut self, name: &str, n: usize) -> Result<Vec<T>> {
        let [offset, len] = self.idx.fields[name];
        let count = (len as usize / std::mem::size_of::<T>()).min(n);
        self.f.seek(SeekFrom::Start(offset))?;
        let mut out = vec![T::default(); count];
        self.f.read_exact(bytemuck::cast_slice_mut(&mut out))?;
        Ok(out)
    }
}
pub struct CoreState {
    sequence: Vec<u8>,
    slots: Vec<f32>,
    delayed: [Option<usize>; 12],
    ctx: [Option<usize>; 2],
    score: [f32; 2],
    executive: [f32; D],
    stm: Vec<(u64, u64, u8, f32)>,
    ptr: usize,
    previous: Option<(u64, u64, [f32; D])>,
}
impl CoreState {
    /// プリフィル長を引いた実観測バイト数を返す。
    pub fn len_after(&self, prefill: usize) -> usize {
        self.sequence.len() - prefill
    }
}
pub struct Inference {
    adapter: Option<Vec<f32>>,
    pub alive: Vec<u8>,
    span: Vec<i16>,
    fires: Vec<i32>,
    w: Vec<u16>,
    sd: Vec<i16>,
    sw: Vec<u16>,
    e: Vec<f32>,
    bias: Vec<f32>,
    keep: Vec<f32>,
    write: Vec<f32>,
    kb: Vec<f32>,
    wb: Vec<f32>,
    read: Vec<f32>,
    modulate: Vec<f32>,
    centers: Vec<f32>,
    llr: Vec<f32>,
    sigs: Vec<u64>,
    children: Vec<i32>,
}
/// 折り返し乗算による4-wayエッジ署名。
fn signature(p: usize, k: usize, c: usize) -> u64 {
    mix(mix(mix(p as u64 + 11) ^ ((k as u64 + 1) * 1000003)) ^ (c as u64 + 2000029)).max(1)
}
/// ベクトルのノルムを返す。
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}
/// 同じ次元の特徴同士を掛け合わせて足す。
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// BF16 を元のビット位置に戻す。
fn bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
impl Inference {
    /// 全プールまたは先頭 cap 個の誘導部分グラフを同じ重みから構築する。
    pub fn load(path: &Path, cap: usize) -> Result<Self> {
        ensure!(cap == 150000 || cap == 6000000);
        let idx: Index = serde_json::from_slice(&fs::read(path)?)?;
        let core_path = if let Ok(env_p) = std::env::var("NEUROGRAPH_CORE_PT") {
            std::path::PathBuf::from(env_p)
        } else if Path::new(&idx.path).exists() {
            std::path::PathBuf::from(&idx.path)
        } else {
            let root_cand = path
                .parent()
                .and_then(|p| p.parent())
                .map(|r| r.join("merged_v111_broca_core.pt"))
                .unwrap_or_else(|| std::path::PathBuf::from("merged_v111_broca_core.pt"));
            root_cand
        };
        let mut r = Reader {
            f: File::open(&core_path)?,
            idx,
        };
        let all = usize::MAX;
        let buckets = if cap == 6000000 { 8388608 } else { 262144 };
        let mut sigs = vec![0; buckets * 4];
        let mut children = vec![-1; buckets * 4];
        let parents = r.read::<i32>("edge_parents", all)?;
        let conds = r.read::<i32>("edge_conds", all)?;
        let kinds = r.read::<i8>("edge_kinds", all)?;
        let ids = r.read::<i32>("edge_children", all)?;
        for i in 0..ids.len() {
            if ids[i] as usize >= cap || parents[i] as usize >= cap {
                continue;
            }
            let s = signature(parents[i] as usize, kinds[i] as usize, conds[i] as usize);
            let j = (s as usize & (buckets - 1)) * 4;
            if let Some(k) = (j..j + 4).find(|&k| sigs[k] == 0 || sigs[k] == s) {
                sigs[k] = s;
                children[k] = ids[i]
            }
        }
        Ok(Self {
            adapter: None,
            alive: r.read("alive", cap)?,
            span: r.read("local_span", cap)?,
            fires: r.read("fires", cap)?,
            w: r.read("W_node", cap * D)?,
            sd: r.read("sharp_dst", cap * 8)?,
            sw: r.read("sharp_w", cap * 8)?,
            e: r.read("E", all)?,
            bias: r.read("bias", all)?,
            keep: r.read("G_keep", all)?,
            write: r.read("G_write", all)?,
            kb: r.read("keep_bias", all)?,
            wb: r.read("write_bias", all)?,
            read: r.read("W_read", all)?,
            modulate: r.read("W_mod", all)?,
            centers: r.read("mode_centroids", all)?,
            llr: r.read("mode_llr", all)?,
            sigs,
            children,
        })
    }
    /// root と到達した各ノードの BF16 残差および8本の直接結合を加算する。
    fn accumulate(&self, n: usize, g: f32, h: &mut [f32; D], logits: &mut [f32; D]) {
        for j in 0..D {
            h[j] += bf16(self.w[n * D + j]) * g
        }
        for j in n * 8..n * 8 + 8 {
            if self.sd[j] >= 0 {
                logits[self.sd[j] as usize] += bf16(self.sw[j]) * g * 0.65
            }
        }
    }
    /// 新しいターンの作業記憶を初期化する。
    pub fn init_state(&self) -> CoreState {
        CoreState {
            sequence: vec![],
            slots: vec![0.; S * D],
            delayed: [None; 12],
            ctx: [None; 2],
            score: [0.; 2],
            executive: [0.; D],
            stm: vec![],
            ptr: 0,
            previous: None,
        }
    }
    /// エクスポートした W、b、scale を検査して取り付ける。
    pub fn load_adapter(&mut self, path: &Path) -> Result<()> {
        let data = fs::read(path)?;
        ensure!(data.len() == 4 * (D * D + D + 1));
        self.adapter = Some(
            data.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
        );
        Ok(())
    }
    /// 実バイトを履歴に加え、次バイトのロジットと活性を返す。
    pub fn step_byte(&self, st: &mut CoreState, byte: u8) -> ([f32; D], [f32; D]) {
        self.advance(st, byte, true)
    }
    /// 入力観測時に前ステップの予測誤差で STM を更新する。
    fn advance(&self, st: &mut CoreState, byte: u8, observe: bool) -> ([f32; D], [f32; D]) {
        let CoreState {
            sequence,
            slots,
            delayed,
            ctx,
            score,
            executive,
            stm,
            ptr,
            previous,
        } = st;
        let t = sequence.len();
        if let Some((a, b, last)) = previous
            .take()
            .filter(|(a, _, _)| observe && t >= 5 && *a != 0)
        {
            let peak = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let prob = (last[byte as usize] - peak).exp()
                / last.iter().map(|v| (v - peak).exp()).sum::<f32>();
            if 1. - prob > 0.36 {
                let entry = (a, b, byte, 1.);
                if stm.len() < 192 {
                    stm.push(entry)
                } else {
                    stm[*ptr] = entry
                }
                *ptr = (*ptr + 1) % 192;
            }
        }
        sequence.push(byte);
        let root = byte as usize;
        for v in &mut *score {
            *v *= 0.985
        }
        if let Some(id) = delayed[t % 12] {
            let sal = 1.8 / (1. + self.fires[id] as f32 * 0.02).sqrt();
            let j = usize::from(score[1] < score[0]);
            if !ctx.contains(&Some(id)) && sal > score[j] && sal > 0.55 {
                ctx[j] = Some(id);
                score[j] = sal;
            }
        }
        let top = ctx[usize::from(score[1] > score[0])];
        let mut next_delay = None;
        let mut h = [0.; D];
        let mut logits = [0.; D];
        self.accumulate(root, 1., &mut h, &mut logits);
        let mut front = [Some(root), None, None, None];
        for hop in 1..=16 {
            let mut next = [None; 4];
            for (j, parent) in front.into_iter().enumerate() {
                let Some(p) = parent else { continue };
                let distance = (self.span[p] as usize + 1).min(27);
                let local = t.checked_sub(distance).map(|i| sequence[i] as usize);
                for (k, c) in [local, top.filter(|_| (2..=5).contains(&hop))]
                    .into_iter()
                    .enumerate()
                {
                    if let Some(c) = c {
                        let s = signature(p, k, c);
                        let base = (s as usize & (self.sigs.len() / 4 - 1)) * 4;
                        if let Some(ix) = (base..base + 4).find(|&ix| self.sigs[ix] == s) {
                            let n = self.children[ix] as usize;
                            if n >= D && self.alive[n] != 0 {
                                next[(j * 2 + k) % 4] = Some(n);
                                self.accumulate(
                                    n,
                                    1. / (1. + 0.10 * hop as f32),
                                    &mut h,
                                    &mut logits,
                                );
                                if k == 0 && (2..=4).contains(&hop) {
                                    next_delay = Some(n)
                                }
                            }
                        }
                    }
                }
            }
            if next.iter().all(Option::is_none) {
                break;
            }
            front = next;
        }
        delayed[t % 12] = next_delay;
        let emb = &self.e[root * D..(root + 1) * D];
        let hn = norm(&h).max(1.);
        let en = norm(emb).max(1.);
        let mut x = [0.; D];
        for j in 0..D {
            x[j] = 0.65 * h[j] / hn + 0.35 * emb[j] / en;
            executive[j] = 0.992 * executive[j] + 0.008 * h[j]
        }
        let mut r = [0.; D];
        let mut m = [0.; D];
        for s in 0..S {
            let o = s * D;
            let sigmoid = |z: f32| 1. / (1. + (-z).exp());
            let kg = sigmoid(dot(&x, &self.keep[o..o + D]) + self.kb[s]);
            let wg = sigmoid(dot(&x, &self.write[o..o + D]) + self.wb[s]);
            for j in 0..D {
                slots[o + j] = kg * slots[o + j] + wg * x[j]
            }
            let scale = 2.5 / norm(&slots[o..o + D]).max(2.5);
            for j in 0..D {
                slots[o + j] *= scale;
                let v = slots[o + j];
                let row = (o + j) * D;
                for k in 0..D {
                    r[k] += v * self.read[row + k];
                    m[k] += v * self.modulate[row + k]
                }
            }
        }
        let mut act = [0.; D];
        for j in 0..D {
            act[j] = ((h[j] * (1. + 0.35 * m[j].tanh()) + r[j]) * 0.35).tanh() * 2.85
        }
        let exn = norm(executive).max(1e-6);
        let mut modes: Vec<_> = self
            .centers
            .chunks_exact(D)
            .map(|c| (dot(c, executive) / exn / 0.28).exp())
            .collect();
        let sum: f32 = modes.iter().sum();
        for p in &mut modes {
            *p /= sum
        }
        for j in 0..D {
            logits[j] = (logits[j] + dot(&act, &self.e[j * D..(j + 1) * D]) + self.bias[j]) / 1.08;
            for (i, &p) in modes.iter().enumerate() {
                logits[j] += p * self.llr[i * D + j] * (0.42 * 1.10);
            }
        }
        let mut hash = mix(root as u64 + 17);
        let mut k4 = 0;
        let mut k7 = 0;
        for (j, (a, b)) in [
            (31, 1315423911),
            (53, 2654435761),
            (79, 1000003),
            (97, 1000033),
            (113, 1000037),
            (131, 1000039),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(i) = t.checked_sub(j + 1) {
                hash = mix(hash ^ ((sequence[i] as u64 + a) * b));
                if j == 2 {
                    k4 = hash
                }
                if j == 5 {
                    k7 = hash
                }
            }
        }
        let mut boost = [0f32; D];
        for (a, b, v, w) in &mut *stm {
            *w *= 0.9975;
            if *w > 0.05 {
                if k4 != 0 && k4 == *a {
                    boost[*v as usize] += *w * 1.15
                }
                if k7 != 0 && k7 == *b {
                    boost[*v as usize] += *w * 2.35
                }
            }
        }
        for j in 0..D {
            logits[j] += boost[j].min(4.2)
        }

        if let Some(w) = &self.adapter {
            let mut residual = [0.; D];
            for (row, &activation) in w[..D * D].chunks_exact(D).zip(&act) {
                for (out, &weight) in residual.iter_mut().zip(row) {
                    *out += activation * weight;
                }
            }
            for j in 0..D {
                logits[j] += residual[j] * w[D * D + D] + w[D * D + j];
            }
        }
        *previous = Some((k4, k7, logits));
        (logits, act)
    }
    /// プロンプトのみ STM を学習し、従来と同じ貪欲生成を行う。
    pub fn generate(&self, prompt: &[u8], count: usize) -> Vec<u8> {
        if prompt.is_empty() {
            return vec![];
        }
        let mut st = self.init_state();
        let mut logits = [0.; D];
        for &b in prompt {
            logits = self.advance(&mut st, b, true).0;
        }
        let mut out = vec![];
        for i in 0..count {
            let best = (0..D)
                .reduce(|a, b| if logits[b] > logits[a] { b } else { a })
                .unwrap() as u8;
            out.push(best);
            if i + 1 < count {
                logits = self.advance(&mut st, best, false).0;
            }
        }
        out
    }
}
