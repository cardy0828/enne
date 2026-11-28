//! ENNE v24.0 関係推論野ファサード (`ReasonerCore`, `NGRC2451` + `SyntaxCore`)。
//!
//! 文字単位構文・節境界コアと関係推論コアに委譲し、実行時の固定辞書やグラフ探索を持たず動作する。

#[path = "learning.rs"]
mod learning;

use crate::{reasoner_core::ReasonerCore, syntax_core::SyntaxCore};
use anyhow::Result;
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReasonMode {
    Copy,
    OrderExtreme,
    OrderRank,
    OrderPairwise,
    Transfer,
    Elimination,
    ContradictionCycle,
    ContradictionConflict,
    QuantDiff,
    QuantCompare,
    Syllogism,
    Abstain,
}

#[derive(Debug, serde::Serialize)]
pub struct ReasoningPlan {
    pub mode: ReasonMode,
    pub answer_bytes: Vec<u8>,
    pub path_nodes: Vec<Vec<u8>>,
    pub auxiliary_values: Vec<Vec<u8>>,
}

pub struct Reasoner {
    core: ReasonerCore,
}

impl Reasoner {
    /// v24.5 統合済みバイナリ重み (`NGRC2451`) または空初期シードを開く。
    pub fn open(path: &Path) -> Result<Self> {
        Self::load_synapses(path)
    }

    /// バイナリ重み (`NGRC2451`) または白紙シードJSONを読み、作業記憶を初期化する。
    pub fn load_synapses(path: &Path) -> Result<Self> {
        let head = fs::read(path)?;
        let core = if head.starts_with(b"NGRC2451") {
            ReasonerCore::load(path)?
        } else {
            let s_cand1 = path.parent().map(|p| p.join("v245/syntax_core_v245.bin"));
            let s_cand2 = path
                .parent()
                .and_then(|p| p.parent())
                .map(|r| r.join("v245_syntax_core/syntax_core_v245.bin"));
            let syntax = if let Some(p) = s_cand1.filter(|p| p.exists()) {
                SyntaxCore::load(&p)?
            } else if let Some(p) = s_cand2.filter(|p| p.exists()) {
                SyntaxCore::load(&p)?
            } else {
                SyntaxCore::blank()
            };
            ReasonerCore::from_blank_seed_json_with_syntax(&head, syntax)?
        };
        Ok(Self { core })
    }

    /// 一時作業記憶を除いたシナプス重みを `NGRC2451` 形式で保存する。
    pub fn save_synapses(&self, path: &Path) -> Result<()> {
        self.core.save(path)
    }

    /// セッションの一時前提を追加する。
    pub fn observe_turn(&mut self, s: &str) {
        self.core.observe_turn(s);
    }

    /// セッション作業記憶に保持中の前提が存在するか返す。
    pub fn has_memory(&self) -> bool {
        self.core.has_memory()
    }

    /// 文字単位構文パーサで入力に複数の節が含まれるか判定する。
    pub fn has_multi_clauses(&self, text: &str) -> bool {
        self.core.has_multi_clauses(text)
    }

    /// 神経前題識別ヘッドで記憶へ保持可能な前提か判定する。
    pub fn is_premise(&self, text: &str) -> bool {
        self.core.is_premise(text)
    }

    /// 学習済みの棄権ラベルと一致するか判定する。
    pub fn is_unknown(&self, answer: &str) -> bool {
        self.core.is_unknown(answer)
    }

    /// セッション一時作業記憶を消去する。
    pub fn apoptosis(&mut self) {
        self.core.apoptosis();
    }

    /// 無教師ペア文から同義・反意シナプスを1回で獲得する。
    pub fn observe_unlabeled(&mut self, text: &str) {
        self.core.observe_unlabeled(text);
    }

    /// 教師付きエピソードから関係役割・極性を1回で獲得する。
    pub fn learn_episode(&mut self, prompt: &str, answer: &str) {
        self.core.learn_episode(prompt, answer);
    }

    /// 関係推論を実行し、結論文字列を返す。
    pub fn answer(&self, prompt: &str) -> String {
        self.core.answer(prompt)
    }

    /// 文字構文スパン解析・再帰 TTFS 作業記憶グラフ・神経ヘッドによる推論経路プランを返す。
    pub fn reason_plan(&self, prompt: &str) -> ReasoningPlan {
        self.core.reason_plan(prompt)
    }

    /// 数値付き実体タプル列から直接定量推論プランを構築する。
    pub fn reason_quantities_plan(&self, prompt: &str, items: &[(f64, &str, &str)]) -> ReasoningPlan {
        self.core.reason_quantities_plan(prompt, items)
    }

    /// 内部 `ReasonerCore` への参照を返す。
    pub fn core(&self) -> &ReasonerCore {
        &self.core
    }
}
