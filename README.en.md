# ENNE — Engineering Neural Network Engine

[日本語](README.md) | **English**

ENNE is an attempt to build intelligence not as "a monolithic model trained once and treated as finished," but as "a system designed, grown, and extended incrementally." It has been developed for Japanese from v0.1 through v25.0.

This README documents the latest **v25.0 pure-CPU Rust text pipeline (`enne-rs/`)**. Background notes, figures, and experimental logs from the earlier Python / GPU research versions (v0.1–v15.2, `src/neurograph/`) are separated into [docs/archive-v15.md](docs/archive-v15.md) (Japanese).

1. **Grow components on CPU alone.** Trains and runs custom components on a local CPU without GPU clusters, pretrained large language models, or external tokenizers.
2. **Design intelligence, not only train it.** Splits executive routing, memory, character syntax parsing, relational reasoning, metacognition, and utterance generation into independent organs with distinct learning rules so individual components can be swapped or retrained independently.
3. **Treat intelligence as a growing system rather than a finished artifact.** Dynamically constructs a working-memory graph per inference for transient conversation premises and unseen entities, clears premise history at session reset, and supports online teaching and relation learning.

---

## 1. Train and Run on CPU Alone (GPU 0 bytes)

**Runs on CPU alone with no external pretrained models or backpropagation.** Sensory inputs are derived from UTF-8 byte streams, character n-grams, and character-class transitions, and trained with competitive Oja-Hebbian learning, local delta rules, contrastive Hebbian memory, and BF16 growing DAGs (without backpropagation or pretrained tokenizers).

**Each component trains in seconds to minutes and stays compact in memory.**

| Component | Version | CPU Train / Build Time | Checkpoint Size | Validation Metrics |
|---|---|---|---|---|
| Executive (`ExecutiveCore`) | v22.0 / v25.0 | 2.53 s | 1.58 MiB | 5-mode `99.28%`, 4-op `100%`, Calc Span F1 `1.00` |
| Hippocampus (`HippocampusCore`) | v23.0 | 9.23 s | 366.87 MiB | `756,941` leads, comparison acc `100%`, recall `6/6` |
| Relational Reasoner (`ReasonerCore`) | v24.0 | 1.09 s | 1.95 MiB | Relational bench `320/320`, cold-retrain `320/320` |
| Syntax Parser (`SyntaxCore`) | v24.5 | 5.76 s | 1.28 MiB | BIO char acc `99.99%`, Span F1 `99.99%`, unseen syntax `12/12` |
| Metacog & Speech (`MetacogCore` / `BrocaCore`) | v25.0 | 25.99 s | 0.16 MiB + 23.40 MiB | Store gate `99.93%`, definition BIO `99.91%`, dialogue intent `91.13%` |
| Trace Generator (`ReasoningCore`, optional) | v19.0 | 920.33 s (96,000 cases) | — | `11,135` alive nodes, trace contains answer `320/320`, Peak RSS `165.26 MiB` |

Even with all text organs loaded together, startup takes **140.90 ms** at **270.99 MiB** Peak RSS. When fixing a specific behavior, only the responsible organ needs retraining in seconds instead of rebuilding the whole system.

---

## 2. Modular Design and Swappable Organs (v25.0 Text Pipeline)

ENNE's text pipeline consists of seven independent organs with separated responsibilities. Dimensions, learning rates, and thresholds in code are explicit design parameters.

| Role | Organ (`enne-rs/src/`) | Mechanism |
|---|---|---|
| Executive & Span Extraction | `ExecutiveCore` (`executive_core.rs`, `executive.rs`) | Two-layer Oja-Hebbian k-WTA (128 units) over char 1..4-grams and structural features + local delta classifier, plus 5-tag BIO Viterbi extractor (`CalcSpanTagger`) |
| Associative Hippocampus | `HippocampusCore` (`hippocampus_core.rs`, `hippocampus.rs`) | Offline index sealed into DG (`1,048,576` cells) -> CA3 BF16 CSR (`6,753,200` edges), NMS + Hebbian numeric/comparison readout, and online `teach` writes |
| Syntax & Clause Boundaries | `SyntaxCore` (`syntax_core.rs`) | Oja-Hebbian k-WTA (64 units) over character windows and script transitions + 10-tag BIO transition-matrix Viterbi decoder |
| Relational Reasoning | `ReasonerCore` (`reasoner_core.rs`, `reasoner.rs`) | Recursive spiking TTFS working-memory graph ($\gamma = 0.75$) and contrastive Hebbian predicate memory with numeric, rank, and premise readout heads |
| Metacognitive Control | `MetacogCore` (`metacog.rs`) | Single-clause premise retention gate (64 units, top-8) and grounding adoption perceptron over 7-D recall telemetry |
| Fact Distillation & Broca Output | `BrocaCore` (`broca_core.rs`, `planner.rs`) | 4-tag BIO definition extractor (`FactDistiller`), 7-intent dialogue classifier (`DialogueIntentHead`), and 14-action BF16 `BrocaDAG` over a 260-symbol native pointer vocabulary |
| Internal Reasoning Trace | `ReasoningCore` (`reasoning_core.rs`) | UTF-8 14-hop BF16 byte DAG (260-symbol vocabulary) for `explain` trace rendering (optional mount) |

Because intermediate states and reasoning traces are decoupled, misclassifications (such as declarative sentences without periods or syllogism phrasing) can be isolated and repaired by updating only the affected organ and non-overlapping corpus. See [docs/architecture.md](docs/architecture.md) (Japanese) for architectural details.

---

## 3. Growing System and Ephemeral Cache Nodes

**Dynamically constructs a working-memory graph per inference and clears premise history afterward (ephemeral cache nodes).** `ReasonerCore` holds in-session transient premises as `turns: Vec<String>`, and on each inference (`reason_plan`), `SyntaxCore` re-parses `turns` together with the query. A local working-memory graph (`NeuralWorkingGraph`) sized to the entities and relations in context is then instantiated, linked with temporary synapses (directed order, equivalence, membership, inhibition, numeric values), evaluated with recursive TTFS propagation (cycles, reachability, ranking, numeric differences), and discarded at the end of the query.

Resolved entity names and numbers are not stored in network weights; they are passed via `ReasoningPlan` using `<P>`, `<V>`, and `<ANS>` pointers to `BrocaCore` and `ReasoningCore`. In [results/results_v190.json](results/results_v190.json), keeping the same precomputed `plan` while disabling only the generator-side (`ReasoningCore`) pointer-cache lookup (`use_cache`) yielded `320/320` outputs containing the gold answer when enabled versus `nocache_contains: 0` (0 of 320 outputs containing the gold answer string) when disabled. Because session resets (`reset_history` / `apoptosis`) clear `turns`, transient premises do not carry over across sessions. In `v250-verify` ([results/results_v250_integration.json](results/results_v250_integration.json)), asserting `富士山の標高は3776m` yields `583m` and reverts to the hippocampal value `583.12m` after reset; non-retention of `追跡キー＝砂時計印` across resets is verified by `bleed: false` in [results/results_v190.json](results/results_v190.json) and in the E2E suite ([enne-rs/tests/e2e.rs](enne-rs/tests/e2e.rs)).

**Training-time neurogenesis and online updates operate alongside this.** `BrocaDAG` (`1,714` alive nodes) and `ReasoningCore` (`11,135` alive nodes) grow and prune character-hop BF16 DAG nodes during training, while hippocampal `teach` (single-episode write) and `ReasonerCore::learn_relation` (online acquisition of unseen antonym pairs and endpoints) add knowledge and relations without retraining the entire system.

---

## Current Status (v25.0): Integrated CPU-Only Text Pipeline

In v25.0 (`enne-rs/`), the text pipeline runs end-to-end on CPU alone across command routing, associative memory, character syntax parsing, relational reasoning, metacognitive gating, fact distillation, and utterance generation. In the primary integration log [results/results_v250_integration.json](results/results_v250_integration.json) from `v250-verify`, **31/31** checks pass (`dialogue_unique: 4`), with **140.90 ms** load time, **270.99 MiB** Peak RSS, and **0.27–6.19 ms** per query. Separately, `enne-rs/` includes 10 unit tests (no external assets needed) and 12 E2E/source-audit tests (requiring external assets).

```text
Q: 1袋4個入りのパンを5袋買うと全部で何個？                        [0.29 ms]  (word-problem multiplication)
A: 20個です。

Q: 今の時刻を教えて                                                  [0.50 ms]  (system clock)
A: 現在は20時18分です。

Q: 光合成の仕組みを説明して                                          [5.06 ms]  (hippocampal fact distillation)
A: 光合成は、光エネルギーを化学エネルギーに変換して生体に必要な有機物質を作り出す反応過程です。

Q: 蒼塔と朱塔の高さの差は何m？（after online teach）                 [5.47 ms]  (numeric diff after teach)
A: 240m   (explain: 根拠となる値は朱塔（180m）、蒼塔（420m）です。二つの数値の差を算出すると、答えは240mになります。)

Q: 鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？ [6.19 ms]  (syllogism check)
A: はい   (explain: 三段論法で所属関係を確認します。鯱、鯨、哺乳類という連鎖が成り立つことから、結論ははいとなります。)

Q: 追跡キー＝砂時計印 → 追跡キー＝                                   [1.06 ms / 4.25 ms]  (ephemeral copy)
A: 了解しました。一時記憶に保持しました。 → 砂時計印

Q: 架空の測定値は？ / 明日の株価はどうなりますか？                    [0.47 ms / 0.47 ms]  (abstention)
A: 情報がありません。

Q: 今日はちょっと疲れたよ。                                          [1.35 ms]  (polite dialogue intent)
A: お疲れさまです。無理をなさらず、少し休んでくださいね。
```

### Metrics Summary and Result JSONs

| Target | Key Results | Primary Record |
|---|---|---|
| Integrated Text Pipeline (31 cases) | `31/31` (Calc 5/5, Clock 1/1, Wiki 5/5, TeachCompare 2/2, Reasoning 6/6, MultiTurn 3/3, Copy 2/2, PostReset 1/1, Abstain 2/2, Dialogue 4/4) | [results/results_v250_integration.json](results/results_v250_integration.json) |
| `ExecutiveCore` (v22/v25) | 1,538 val cases: `val_mode_acc: 0.9928` / `val_op_acc: 1.0` / `val_calc_span_f1: 1.0`; 26 unseen probes: `25/26` (op `7/7`, span `7/7`) | [results/results_v220.json](results/results_v220.json) |
| `HippocampusCore` (v23) | `756,941` leads, `val_num_acc: 0.99975` / `val_unit_acc: 0.999375` / `val_cmp_acc: 1.0`; recall `6/6`, compare `4/4` | [results/results_v230.json](results/results_v230.json) |
| `ReasonerCore` (v24) | Relational bench `320/320`, cold-retrain `320/320`, unseen reasoning `16/16`, unseen syntax `12/12` | [results/results_v240.json](results/results_v240.json) |
| `SyntaxCore` (v24.5) | 2,402 val sequences: `val_char_acc: 0.9999` / `val_sep_f1: 0.9999` / `val_span_f1: 0.9999` | [results/results_v245.json](results/results_v245.json) |
| `MetacogCore` + `BrocaCore` (v25) | Store `0.9993` (unseen `7/7`), adopt `1.0`, def BIO `0.9991`, dialogue intent `0.9113` (unseen `5/5`), `BrocaDAG val_bpt: 0.0127` | [results/results_v250.json](results/results_v250.json) |
| `ReasoningCore` (v19) | `bench_320`: answer & trace containment `320/320`, `val_mode_acc: 0.9944`, `val_bpc: 0.0340`, unseen paraphrases `5/8` | [results/results_v190.json](results/results_v190.json) |
| CPU Scale Reference (v18.4) | 35M-node pool (333M bytes streamed): `val_bpc: 1.6011`, Peak RSS `20,168.08 MiB` | [results/results_v184_mega_35m.json](results/results_v184_mega_35m.json) |

Full metrics and discussion are documented in [docs/results.md](docs/results.md) (Japanese).

---

## Try It (Without External Data or Trained Weights)

Because trained checkpoints and corpora are not bundled in this repository, the interactive CLI and full E2E suite require external assets (see [docs/source.md](docs/source.md) (Japanese) for asset placement and CLI training/inference subcommands). Two self-contained checks run without external assets:

1. **Rust component unit tests (10 cases)** — verifies small-scale learning, inference, and serialization round-trips across neural cores:

```sh
cd enne-rs
cargo test --release --lib
```

2. **Toy example (~200-line conceptual implementation)** — demonstrates byte-level growth, pluggable external memory, and DAG merging on CPU in ~1 second:

```sh
python examples/toy_enne.py
```

---

## Limitations

- **Author-written and small-scale evaluations.** The 31 integration cases, 320 relational cases, and unseen probes (26 / 16 / 12 / 7 / 5 items) are all developer-authored; public benchmarks and third-party evaluations have not been conducted.
- **Constrained unseen-input routing and generation.** `ExecutiveCore` misclassified 1 of 26 unseen probes (`"次の試合でどのチームが優勝する？"`) as `Fact` instead of `Abstain`. Everyday dialogue responses from `BrocaCore` are constrained to polite templates across 7 intent classes, and `ReasoningCore` traces span 12 modes by 6 styles; open-ended generation and general conversation are not supported.
- **Audio and persona modules are omitted.** Work-in-progress speech modules (`ear` / `repair` / `audio`) and separated persona modules (`persona*`) are excluded from `enne-rs/`, which covers only the text pipeline.
- **External assets are not included.** `.bin` checkpoints, SQLite indexes, and training corpora are omitted, so the CLI and `tests/e2e.rs` cannot run from this repository alone.

Caveats on interpreting metrics and organ names (e.g., `Cortex`, `Hippocampus`, and `Broca` are architectural metaphors rather than claims of biological fidelity) are noted in [docs/results.md](docs/results.md) (Japanese). Records and limitations from v0.1–v15.2 are archived in [docs/archive-v15.md](docs/archive-v15.md) (Japanese).

---

## Repository Layout and Documentation

```text
enne-rs/         v25.0 pure-CPU Rust text pipeline (22 source files + tests/e2e.rs)
scripts/corpus/  v19–v25 training & audit corpus builders (5 scripts)
docs/            Design, results, history, source guide, and pre-v15.2 archive (Japanese)
results/         Primary result JSONs per version
examples/        Simplified conceptual sample (not the research core)
figures/         Figures for v0.1–v15.2 (generated by scripts/make_figures.py)
src/neurograph/  Archived v15.2 Python/GPU research code
```

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Text-pipeline structure, per-organ learning rules/hyperparameters, and ephemeral cache-node lifecycle (Japanese) |
| [docs/results.md](docs/results.md) | Full numerical results, supported claims, and caveats (Japanese) |
| [docs/source.md](docs/source.md) | `enne-rs/` build, training/verification CLI subcommands, environment variables, and `src/neurograph/` overview (Japanese) |
| [docs/history.md](docs/history.md) | Development changelog from v0.1 to v25.0 (Japanese) |
| [docs/archive-v15.md](docs/archive-v15.md) | Archived v0.1–v15.2 Python/GPU design notes, figures, detailed results, and limitations (Japanese) |

## About the Source Code

This repository focuses on the pure-CPU Rust code running the text pipeline as of v25.0 (`enne-rs/` 22 modules, E2E test suite, and 5 corpus builder scripts), while retaining the earlier Python/GPU version (`src/neurograph/`) for reference. Machine-specific paths and GPU selectors were replaced with environment variables (`ENNE_ROOT`, `ENNE_DATA_DIR`, etc.), and inline comments were cleaned up.

The README files (`README.md` and `README.en.md`), documentation, result JSONs, and version metadata are synchronized as of v25.0. When updating source code, update the documentation together.

## Data and License

The documentation, figures, scripts, and source code in this repository are released under the [MIT License](LICENSE).

Japanese Wikipedia, Aozora Bunko, Easy Japanese corpora, and public Japanese dialogue datasets (`magpie_ja`, `wiki_conv_ja`, `oasst1_ja`, `dolly_ja`, `RealPersonaChat`, `PersonaCast-JA`) were used for training and evaluation. Example outputs under `results/` contain sentences derived from Japanese Wikipedia text, which are governed by [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and excluded from the MIT License.
