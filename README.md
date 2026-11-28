# ENNE — Engineering Neural Network Engine

ENNEは、知能を「巨大なモデルを一度学習して完成させるもの」ではなく、「設計し、育て、後から機能を足していくシステム」として作る試みです。日本語を対象にv0.1からv25.0まで開発してきました。

この README は最新の **v25.0 純 Rust CPU テキスト経路（`enne-rs/`）** について説明しています。初期の Python / GPU 研究版（v0.1〜v15.2、`src/neurograph/`）の経緯・図表・実験記録は [docs/archive-v15.md](docs/archive-v15.md) に分離しています。

1. **手元の CPU だけで部品を育てられる。** 高価なGPUクラスタも、大規模な事前学習済みモデルも外部トークナイザーも使わず、CPU だけで独自の部品群を学習・実行します。
2. **知能を学習するだけでなく、設計できる。** 司令塔・記憶・文字構文解析・関係推論・メタ認知・発話生成を独立した機構に分け、それぞれに適した学習則を採用します。個別の部品だけを差し替えたり再学習したりできます。
3. **知能を完成品ではなく、成長するシステムとして扱う。** 会話中の仮前提や未知実体に対して推論ごとに作業記憶グラフを動的に構築し、セッション終了時に前提履歴を消去するほか、オンライン教示や学習による更新ができます。

> **English abstract.** ENNE treats an AI system not as a monolithic pretrained network, but as modular components designed, grown, and updated independently without backprop, pretrained LLMs, or external tokenizers. This README documents the latest **v25.0 pure-CPU Rust text pipeline (`enne-rs/`, 0 bytes GPU)**; historical Python/GPU work from v0.1 to v15.2 (`src/neurograph/`) is archived in [docs/archive-v15.md](docs/archive-v15.md). In v25.0, seven CPU organs — two-layer Oja-Hebbian executive router + `CalcSpanTagger`, sealed DG-to-CA3 sparse associative hippocampus (`756,941` leads), character-level BIO Viterbi syntax parser, recursive spiking TTFS working-memory reasoner (`320/320`), metacognitive store/grounding gates, Broca distiller/utterance DAG, and pointer-based trace generator — are each trained on CPU in seconds to minutes. During multi-turn reasoning, `ReasonerCore` retains transient premises in `turns` and dynamically constructs a local `NeuralWorkingGraph` per query via `SyntaxCore` and recursive TTFS propagation, emitting outputs via native `<P>`/`<V>`/`<ANS>` pointers while clearing `turns` on reset (`apoptosis`) with zero cross-session bleed. The `v250-verify` integration check passes 31/31 cases in 140.9 ms load time and 271.0 MiB peak RSS (`0.27–6.19 ms/turn`, `dialogue_unique: 4`; cargo unit/E2E suites are defined in `enne-rs/` separately). Evaluation sets are author-written and outputs are limited to studied task modes and 7-intent polite dialogue rather than open-ended generation. Source code (`enne-rs/`, corpus builders in `scripts/corpus/`, archived `src/neurograph/`) and result JSONs are included; trained weights, corpora, work-in-progress audio modules, and detached persona modules are omitted.

---

## ① 手元の CPU だけで学習・実行できる（GPU 0 bytes）

**外部モデル・逆伝播を使わず、CPU だけで動作します。** 感覚入力は UTF-8 バイト列や文字 n-gram・文字種遷移の特徴から構成し、競合オヤ・ヘブ学習・局所デルタ則・対照ヘブ記憶・BF16 成長グラフで学習します（誤差逆伝播や事前学習済みトークナイザー不使用）。

**部品ごとの学習が短時間で済み、軽量に常駐できます。**

| 部品 | 版 | CPU 学習・構築時間 | チェックポイント容量 | 検証精度 / 指標 |
|---|---|---|---|---|
| 司令塔（`ExecutiveCore`） | v22.0 / v25.0 | 2.53 秒 | 1.58 MiB | 5モード `99.28%`・4演算 `100%`・計算 Span F1 `1.00` |
| 連想海馬（`HippocampusCore`） | v23.0 | 9.23 秒 | 366.87 MiB | 756,941 見出し・比較精度 `100%`・想起 `6/6` |
| 関係推論野（`ReasonerCore`） | v24.0 | 1.09 秒 | 1.95 MiB | 関係推論ベンチ `320/320`・白紙再学習 `320/320` |
| 文字構文解析（`SyntaxCore`） | v24.5 | 5.76 秒 | 1.28 MiB | BIO 文字精度 `99.99%`・Span F1 `99.99%`・未知構文 `12/12` |
| メタ認知・発話野（`MetacogCore` / `BrocaCore`） | v25.0 | 25.99 秒 | 0.16 MiB + 23.40 MiB | 保持判定 `99.93%`・定義 BIO `99.91%`・対話意図 `91.13%` |
| 推論言語コア（`ReasoningCore`、任意） | v19.0 | 920.33 秒（96,000件） | — | 11,135 生存ノード・根拠トレース包含 `320/320`・Peak RSS `165.26 MiB` |

全テキスト器官を同時に読み込んだ結合構成でも、起動時間 **140.90 ms**、Peak RSS **270.99 MiB** で動作します。特定の機構を直したい場合は該当部品だけを数秒で再学習でき、全体の作り直しはいりません。

---

## ② 機能ごとに分けて設計・交換できる（v25.0 テキスト経路）

ENNE のテキスト経路は、役割の異なる 7 つの独立器官に分かれています。コード内の次元数・学習率・閾値はすべて明示的な設計値です。

| 機能 | 機関（`enne-rs/src/`） | 採用した技術 |
|---|---|---|
| 司令塔・Span 抽出 | `ExecutiveCore`（`executive_core.rs`, `executive.rs`） | 文字 1..4-gram＋構造特徴の 2層オヤ・ヘブ k-WTA（128素子）と局所デルタ分類、5タグ BIO Viterbi 抽出（`CalcSpanTagger`） |
| 連想海馬 | `HippocampusCore`（`hippocampus_core.rs`, `hippocampus.rs`） | オフライン索引を DG（1,048,576細胞）→ CA3 BF16 CSR（6,753,200辺）へ封印、NMS とヘブ数値/比較読み出し、`teach` オンライン書込 |
| 文字構文・節境界 | `SyntaxCore`（`syntax_core.rs`） | 文字窓・文字種遷移上のオヤ・ヘブ k-WTA（64素子）＋ 10種 BIO 遷移行列 Viterbi 復号 |
| 関係推論 | `ReasonerCore`（`reasoner_core.rs`, `reasoner.rs`） | 再帰スパイキング TTFS 作業記憶グラフ（$\gamma = 0.75$）と対照ヘブ述語記憶、数値/順位/前提読み出し頭 |
| メタ認知制御 | `MetacogCore`（`metacog.rs`） | 単一節前提の保持判定（64素子・top-8）と 7次元想起テレメトリ上の根拠採択パーセプトロン |
| 定義蒸留・発話生成 | `BrocaCore`（`broca_core.rs`, `planner.rs`） | 4タグ BIO 定義核抽出（`FactDistiller`）、7意図対話分類（`DialogueIntentHead`）、14行動・ネイティブポインター語彙 260 の BF16 `BrocaDAG` |
| 内部推論トレース | `ReasoningCore`（`reasoning_core.rs`） | UTF-8 文字 14ホップ BF16 DAG（語彙 260）による `explain` 根拠過程生成（任意マウント） |

各段階の中間結果と根拠トレースを分離しているため、誤分類が起きた箇所（たとえば句点なし宣言文の司令塔ルーティングや三段論法パターン）を特定し、該当モジュールと非重複コーパスだけを調整して直せます。詳細は [docs/architecture.md](docs/architecture.md) を参照してください。

---

## ③ 成長するシステムと一時キャッシュノード

**推論ごとに作業記憶グラフを動的に構築し、終了後に前提履歴を消去します（一時キャッシュノード）。** `ReasonerCore` は、会話中の仮前提を `turns: Vec<String>` としてセッション内に保持し、推論（`reason_plan`）のたびに `SyntaxCore` が `turns` と質問を再解析します。そして文脈に現れた実体や関係数に応じたローカルな作業記憶グラフ（`NeuralWorkingGraph`）をその場で生成し、一時シナプス（有向順序・同値・所属・抑制・数値など）で結んで再帰 TTFS 伝播（閉路・到達性・順位・数値差）を実行したあと、ローカルグラフを破棄します。

特定した実体名や数値は重みに書き込まず、`ReasoningPlan` を介して `<P>`・`<V>`・`<ANS>` のポインターとして `BrocaCore` や `ReasoningCore` へ渡します。なお、[results/results_v190.json](results/results_v190.json) では同じ計算済み `plan` を与えたまま生成器（`ReasoningCore`）側のポインターキャッシュ参照（`use_cache`）だけを無効化した比較を行っており、有効時は `320/320` が正答文字列を含んだのに対し、無効時は `nocache_contains: 0`（320件の出力中、正答文字列を含むものが 0 件）でした。セッション区切り（`reset_history` / `apoptosis`）で `turns` を消去するため、一時的な仮前提は次セッションへ引き継がれません。`v250-verify`（[results/results_v250_integration.json](results/results_v250_integration.json)）では仮前提 `富士山の標高は3776m` で `583m` と答えた後、リセット後に海馬の値に基づく `583.12m` へ戻ることを確認しており、`追跡キー＝砂時計印` のリセット後非残留は `bleed: false`（[results/results_v190.json](results/results_v190.json)）および E2E テスト（[enne-rs/tests/e2e.rs](enne-rs/tests/e2e.rs)）で検証しています。

**学習時の神経新生とオンライン更新も併存します。** 発話野（`BrocaDAG`：1,714 生存ノード）や推論言語コア（`ReasoningCore`：11,135 生存ノード）は文字ホップ型の BF16 DAG として学習時にノードを増減させ、海馬 `teach`（単回エピソード書込）や推論野 `learn_relation`（未知反意対・端点のオンライン学習）により、全体を再学習せず新しい知識・関係をその場で追加できます。

---

## 現在の到達点（v25.0）：CPU のみのテキスト経路で統合動作

v25.0（`enne-rs/`）では、GPU を使わず CPU のみで、コマンドルーティング・連想記憶・文字構文解析・関係推論・メタ認知制御・定義文蒸留・発話生成までのテキスト経路を一体として動かしています。統合検証コマンド `v250-verify` の一次記録 [results/results_v250_integration.json](results/results_v250_integration.json) では 31 項目中 **31/31**（`dialogue_unique: 4`）を通過しています（読込 **140.90 ms**、Peak RSS **270.99 MiB**、応答 **0.27〜6.19 ms/件**）。別途、`enne-rs/` には単体テスト 10 件（外部資産不要）と E2E・ソース監査テスト 12 件（要外部資産）を収録しています。

```text
Q: 1袋4個入りのパンを5袋買うと全部で何個？                        [0.29 ms]
A: 20個です。

Q: 今の時刻を教えて                                                  [0.50 ms]
A: 現在は20時18分です。

Q: 光合成の仕組みを説明して                                          [5.06 ms]
A: 光合成は、光エネルギーを化学エネルギーに変換して生体に必要な有機物質を作り出す反応過程です。

Q: 蒼塔と朱塔の高さの差は何m？（オンライン教示後）                   [5.47 ms]
A: 240m   (explain: 根拠となる値は朱塔（180m）、蒼塔（420m）です。二つの数値の差を算出すると、答えは240mになります。)

Q: 鯨の全員が哺乳類に属し、鯱の所属は鯨。鯱を哺乳類の一員と言い切れる？ [6.19 ms]
A: はい   (explain: 三段論法で所属関係を確認します。鯱、鯨、哺乳類という連鎖が成り立つことから、結論ははいとなります。)

Q: 追跡キー＝砂時計印 → 追跡キー＝                                   [1.06 ms / 4.25 ms]
A: 了解しました。一時記憶に保持しました。 → 砂時計印

Q: 架空の測定値は？ / 明日の株価はどうなりますか？                    [0.47 ms / 0.47 ms]
A: 情報がありません。

Q: 今日はちょっと疲れたよ。                                          [1.35 ms]
A: お疲れさまです。無理をなさらず、少し休んでくださいね。
```

### 各指標の詳細・結果 JSON

| 対象 | 主な結果 | 一次資料 |
|---|---|---|
| テキスト通貫結合（31項目） | `31/31`（Calc 5/5, Clock 1/1, Wiki 5/5, TeachCompare 2/2, Reasoning 6/6, MultiTurn 3/3, Copy 2/2, PostReset 1/1, Abstain 2/2, Dialogue 4/4） | [results/results_v250_integration.json](results/results_v250_integration.json) |
| `ExecutiveCore`（v22/v25） | 検証 1,538件 `val_mode_acc: 0.9928` / `val_op_acc: 1.0` / `val_calc_span_f1: 1.0`、未知 26問 `25/26`（演算 `7/7`, 抽出 `7/7`） | [results/results_v220.json](results/results_v220.json) |
| `HippocampusCore`（v23） | 756,941 見出し、`val_num_acc: 0.99975` / `val_unit_acc: 0.999375` / `val_cmp_acc: 1.0`、想起 `6/6`・比較 `4/4` | [results/results_v230.json](results/results_v230.json) |
| `ReasonerCore`（v24） | 関係推論 `320/320`、白紙再学習 `320/320`、未知推論 `16/16`、未知構文 `12/12` | [results/results_v240.json](results/results_v240.json) |
| `SyntaxCore`（v24.5） | 検証 2,402系列 `val_char_acc: 0.9999` / `val_sep_f1: 0.9999` / `val_span_f1: 0.9999` | [results/results_v245.json](results/results_v245.json) |
| `MetacogCore` + `BrocaCore`（v25） | 保持判定 `0.9993`（未知 `7/7`）・採択 `1.0`、定義 BIO `0.9991`、対話意図 `0.9113`（未知 `5/5`）、`BrocaDAG val_bpt: 0.0127` | [results/results_v250.json](results/results_v250.json) |
| `ReasoningCore`（v19） | `bench_320`: 短答・根拠包含 `320/320`、`val_mode_acc: 0.9944`、`val_bpc: 0.0340`、未知言い換え `5/8` | [results/results_v190.json](results/results_v190.json) |
| CPU 大規模実験（v18.4・対照） | 3,500万ノード枠（3.33億バイト学習）`val_bpc: 1.6011`、Peak RSS `20,168.08 MiB` | [results/results_v184_mega_35m.json](results/results_v184_mega_35m.json) |

全数値と考察は [docs/results.md](docs/results.md) を参照してください。

---

## 試す（外部データ・学習済み重みなしで動く範囲）

学習済みモデルやコーパスは収録していないため、対話 CLI や全量 E2E テストには外部資産が必要です（配置方法は [docs/source.md](docs/source.md) を参照）。外部資産なしで動作を確認できる範囲は次の 2 つです。

1. **Rust 部品別ユニットテスト（10 件）** — 各ニューラルコアの小規模学習・推論・シリアライズ往復を検証

```sh
cd enne-rs
cargo test --release --lib
```

2. **簡易サンプルコード（約 200 行の概念実装）** — バイト単位成長・外部記憶着脱・DAG 統合の基本原理を CPU で約 1 秒で実行

```sh
python examples/toy_enne.py
```

---

## 限界

- **評価集合が内部作成かつ小規模です。** 結合検証 31 項目・関係推論ベンチマーク 320 問・未知プローブ群（26問・16問・12問・7問・5問）はいずれも開発者自身の作問であり、外部公開ベンチマークや第三者による採点は未実施です。
- **未知入力の分類や発話には制約があります。** `ExecutiveCore` の未知プローブ 26 問のうち 1 問（「次の試合でどのチームが優勝する？」）は `Abstain` ではなく `Fact` に誤分類しました。また、`BrocaCore` の日常対話出力は 7 意図分類に基づく定型・丁寧応答、`ReasoningCore` の思考トレースは 12 モード×6 文体の範囲に収束した生成であり、自由な長文生成・雑談には対応していません。
- **音声と人格はこの版に収録していません。** 開発途中の音声モジュール（`ear` / `repair` / `audio`）および分離中の人格モジュール（`persona*`）は `enne-rs/` から除外し、テキスト経路のみをまとめています。
- **外部資産は同梱していません。** `.bin` チェックポイント、SQLite 索引、学習コーパスは含まれないため、このリポジトリ単体では対話 CLI や `tests/e2e.rs` は実行できません。

結果の解釈上の注意点やモジュール名の扱い（「Cortex」「Hippocampus」「Broca」等は設計上の比喩であり生体脳の再現主張ではないこと）は [docs/results.md](docs/results.md) を参照してください。v0.1〜v15.2 時点の記録と限界は [docs/archive-v15.md](docs/archive-v15.md) にまとめています。

---

## リポジトリ構成・関連資料

```
enne-rs/         v25.0 時点のテキスト経路・純 Rust CPU 実装（22 ファイル + tests/e2e.rs）
scripts/corpus/  v19〜v25 学習・検証コーパス構築スクリプト（5 本）
docs/            設計・結果・履歴・ソース案内・v15.2 以前アーカイブ
results/         各版の結果JSON（評価の一次資料）
examples/        考え方を示す簡略サンプル（研究本体の実装ではない）
figures/         v0.1〜v15.2 の記録図（scripts/make_figures.py で生成）
src/neurograph/  v15.2 時点の Python/GPU 研究コード（アーカイブ）
```

| 資料 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | テキスト経路の全体構造、各コアの学習則・ハイパーパラメータ、一時キャッシュノードの仕組み |
| [docs/results.md](docs/results.md) | 全実験の結果数値、言えること・言えないこと |
| [docs/source.md](docs/source.md) | `enne-rs/` のビルド・環境変数・CLI 入口、アーカイブされた `src/neurograph/` の構成 |
| [docs/history.md](docs/history.md) | v0.1〜v25.0 の開発履歴 |
| [docs/archive-v15.md](docs/archive-v15.md) | 過去の Python / GPU 版（v0.1〜v15.2）の設計説明・図表・詳細結果と限界 |

## ソースコードについて

v25.0 時点のテキスト経路を動かす Rust CPU コード（`enne-rs/` 22 ファイル＋結合テスト＋コーパス構築スクリプト 5 本）を中心に収めています（過去の Python/GPU 版 `src/neurograph/` も参照用に保持）。実験機のパスと GPU の指定は環境変数（`ENNE_ROOT` / `ENNE_DATA_DIR` 等）を参照する形にし、コメントを整理しています。

README・docs・結果JSON・バージョン表記は、v25.0 時点の内容にそろえています。ソースを更新するときは、README と docs も同時に更新します。

## データとライセンス

このリポジトリの文書・図・スクリプト・ソースコードは [MIT License](LICENSE) です。

学習・評価にはWikipedia日本語版、青空文庫、やさしい日本語コーパス、公開の日本語対話データ（magpie_ja、wiki_conv_ja、oasst1_ja、dolly_ja、RealPersonaChat、PersonaCast-JA）を使いました。`results/` の出力例にはWikipedia本文に由来する文が含まれます。これらは [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) の下にあり、MIT Licenseの対象外です。
