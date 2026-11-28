//! ENNE v25.0 司令塔ルータ (`ExecutiveCore`, `NGEC2501`)。
//!
//! モード選択・数値と単位の抽出 (`CalcSpanTagger`)・四則演算分類を `ExecutiveCore` に委譲し、
//! `Planner` (`BrocaCore`) 向けの構造化アクション `ExecAction` を返す。
use crate::executive_core::ExecutiveCore;
use chrono::Local;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Calc,
    Clock,
    Fact,
    Dialogue,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecAction {
    Calc { value: String, unit: String },
    Clock { hour: String, minute: String },
    Fact,
    Dialogue,
    Abstain,
}

fn format_number(v: f64) -> String {
    let exponent = if v.abs() > 0.0 {
        v.abs().log10().floor() as i32
    } else {
        0
    };
    if (v - v.round()).abs() < 1e-9 {
        format!("{v:.0}")
    } else if !(-4..4).contains(&exponent) {
        let scaled = v / 10f64.powi(exponent);
        let mantissa = format!("{scaled:.3}");
        format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        )
    } else {
        let digits = (3 - exponent).max(0) as usize;
        let formatted = format!("{v:.digits$}");
        if digits > 0 {
            formatted.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            formatted
        }
    }
}

/// Routes input via `ExecutiveCore` into 5 cognitive modes and structured `ExecAction` payloads.
pub fn route_with_core(core: &ExecutiveCore, s: &str) -> (Mode, ExecAction) {
    let q = s.trim().nfkc().collect::<String>();
    if q.is_empty() {
        return (Mode::Abstain, ExecAction::Abstain);
    }
    let (mode_id, _) = core.predict_mode(&q);
    let mode = match mode_id {
        0 => Mode::Calc,
        1 => Mode::Clock,
        2 => Mode::Fact,
        3 => Mode::Dialogue,
        _ => Mode::Abstain,
    };

    match mode {
        Mode::Clock => {
            let now = Local::now();
            (
                Mode::Clock,
                ExecAction::Clock {
                    hour: now.format("%H").to_string(),
                    minute: now.format("%M").to_string(),
                },
            )
        }
        Mode::Calc => {
            let (ops, unit) = core.extract_calc_operands_and_unit(&q);
            if ops.len() >= 2 {
                if let (Ok(a), Ok(b)) = (ops[0].parse::<f64>(), ops[1].parse::<f64>()) {
                    let (op_id, _) = core.predict_calc_op(&q);
                    let val = match op_id {
                        0 => Some(a + b),
                        1 => Some(a - b),
                        2 => Some(a * b),
                        3 if b.abs() > 1e-12 => Some(a / b),
                        _ => None,
                    };
                    if let Some(v) = val.filter(|x| x.is_finite()) {
                        return (
                            Mode::Calc,
                            ExecAction::Calc {
                                value: format_number(v),
                                unit,
                            },
                        );
                    }
                }
            }
            (Mode::Fact, ExecAction::Fact)
        }
        Mode::Dialogue => (Mode::Dialogue, ExecAction::Dialogue),
        Mode::Abstain => (Mode::Abstain, ExecAction::Abstain),
        Mode::Fact => (Mode::Fact, ExecAction::Fact),
    }
}
