//! ENNE v23.0 連想海馬ファサード (`HippocampusCore`, `NGHC2301`)。
//!
//! 推論時の参照とオンライン書込を疎分散連想記憶 `HippocampusCore` に委譲する。
use crate::hippocampus_core::HippocampusCore;
use anyhow::Result;
use std::path::Path;

pub struct Fact {
    pub head: String,
    pub body: String,
    pub source: &'static str,
    pub score: f64,
}

pub struct Hippocampus {
    core: HippocampusCore,
}

impl Hippocampus {
    /// 統合済み連想記憶チェックポイント (`NGHC2301`) をロードする。ランタイム DB 接続は行わない。
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            core: HippocampusCore::load(path)?,
        })
    }

    /// オンライン1回書込の高速可塑シナプスに新規知識を記憶させる。
    pub fn teach(&mut self, subject: &str, body: &str) -> Result<()> {
        self.core.teach(subject, body)
    }

    /// DG→CA3 非対称ホップフィールド共鳴により最優先のエングラム本文を想起する。
    pub fn lookup(&self, q: &str) -> Result<Option<String>> {
        Ok(self
            .lookup_hits(q, 1)?
            .into_iter()
            .next()
            .map(|hit| hit.body))
    }

    /// 同一優先順位内で最大 `limit` 件の想起エングラムと確信度を返す。
    pub fn lookup_hits(&self, q: &str, limit: usize) -> Result<Vec<Fact>> {
        Ok(self
            .core
            .recall_hits(q, limit)?
            .into_iter()
            .map(|h| Fact {
                head: h.head,
                body: h.body,
                source: h.source,
                score: h.score,
            })
            .collect())
    }

    /// 複数実体想起と神経数値/単位抽出を用いて比較・差異クエリの結論を返す。
    pub fn compare(&self, q: &str, reasoner: &crate::reasoner::Reasoner) -> Result<Option<String>> {
        Ok(self
            .compare_plan(q, reasoner)?
            .map(|p| String::from_utf8(p.answer_bytes).unwrap()))
    }

    /// 側方抑制 NMS と局所ヘブ読み出し (`W_cmp_query`, `W_num_ctx`, `W_unit`) で比較経路を構築する。
    pub fn compare_plan(
        &self,
        q: &str,
        reasoner: &crate::reasoner::Reasoner,
    ) -> Result<Option<crate::reasoner::ReasoningPlan>> {
        self.core.compare_plan(q, reasoner)
    }

    /// 内部の連想海馬コアへの参照を返す。
    pub fn core(&self) -> &HippocampusCore {
        &self.core
    }
}
