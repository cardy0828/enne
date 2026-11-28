# ソースコード

このリポジトリには 2 系統の実装を収録しています。

1. **`enne-rs/`（v25.0（テキスト経路・純 Rust CPU 版）時点）**: 本リポジトリ現在の主力実装です。CPU のみで動作し、外部トークナイザー・深層学習フレームワーク・GPU を使わず、命令ルーティング・連想記憶・文字構文解析・関係推論・メタ認知・発話生成までをテキスト経路として統合しています。
2. **`src/neurograph/`（v15.2 時点の Python/GPU 研究版）**: バイト成長 DAG 統合および GPU 学習の経緯を含む従来版です。

テキスト経路への一本化と構造検証のため、現在開発途中の音声モジュール（`audio.rs` / `ear.rs` / `repair.rs`）および分離した人格モジュール（`persona.rs` / `persona_core.rs`）は `enne-rs/` には含めていません。

---

## Rust 版（`enne-rs/`、v25.0 時点）

### 構成・ファイル対応（22 ファイル + 結合テスト + コーパス構築 5 本）

| 分類 | ファイル | 内容 |
|---|---|---|
| エントリ・統合 | `enne-rs/src/lib.rs`, `main.rs`, `brain.rs` | クレート公開、CLI サブコマンド、`Brain` 統合経路 |
| 司令塔（v22.0 / v25.0） | `executive.rs`, `executive_core.rs` | 5モード分類・4演算分類・文字単位 `CalcSpanTagger`（`NGEC2501`） |
| 連想海馬（v23.0） | `hippocampus.rs`, `hippocampus_core.rs` | オフライン封印型 DG→CA3 疎分散連想記憶と量的比較（`NGHC2301`） |
| 構文解析（v24.5） | `syntax_core.rs` | 10種 BIO 文字構文・節境界知覚（`NGSX2451`） |
| 関係推論野（v24.0） | `reasoner.rs`, `reasoner_core.rs`, `learning.rs` | TTFS 再帰作業記憶グラフと対照ヘブ述語記憶（`NGRC2451`） |
| メタ認知・発話（v25.0） | `metacog.rs`, `planner.rs`, `broca_core.rs` | 記憶保持・根拠採択ゲート（`NGMC2501`）、`FactDistiller` / `DialogueIntentHead` / `BrocaDAG`（`NGBC2501`） |
| 推論トレース・DAG 基盤 | `reasoning_core.rs`, `core_dag.rs`, `inference.rs`, `cache_gen.rs`, `cache_bench.rs`, `mega_core.rs` | v19.0 推論言語コア、BF16 Byte-DAG 学習・推論基盤 |
| 計測・結合テスト | `memory.rs`, `suite.rs`, `tests/e2e.rs` | RSS 計測、320問再学習検証、テキスト通貫 E2E・ソース監査テスト |
| クレート固定情報 | `enne-rs/Cargo.toml`, `enne-rs/Cargo.lock` | クレート定義および依存バージョン固定用ロックファイル（再現ビルド用に収録） |
| コーパス構築（オフライン学習用） | `scripts/corpus/build_v{190,220,240,245,250}_*.py` | リーク検査付きの訓練・検証コーパス生成スクリプト |

※ チェックポイント形式の互換性を保つため、バイナリファイルのマジックヘッダ（`NGEC2501`, `NGHC2301`, `NGRC2451`, `NGSX2451`, `NGMC2501`, `NGBC2501` 等）は従来名称を維持しています。各コード内の閾値・次元数・学習率は明示的な設計値です。

### ビルドとテスト（Rust 2024 / CPU 専用）

```sh
# 外部資産なしで実行できるビルドおよび単体テスト（10件）
cd enne-rs
cargo build --release
cargo test --release --lib

# 学習済み .bin 等を配置したディレクトリを指定して全 E2E・検証を実行
ENNE_ROOT=/path/to/assets cargo test --release
ENNE_ROOT=/path/to/assets cargo run --release -- v250-verify
ENNE_ROOT=/path/to/assets cargo run --release -- status
ENNE_ROOT=/path/to/assets cargo run --release -- think "1袋4個入りのパンを5袋買うと全部で何個？"
ENNE_ROOT=/path/to/assets cargo run --release -- explain "富士山と北岳の標高差は何m？"
```

### 学習・構築サブコマンド（Rust 版）

各器官の学習用コーパスは `scripts/corpus/build_v{190,220,240,245,250}_*.py` で生成し、`ENNE_ROOT` 配下に配置した状態で `enne-rs/src/main.rs` の学習・構築サブコマンドを実行します。

| サブコマンド | 実装モジュール | 用途・学習内容 |
|---|---|---|
| `train <size> <batch> [pool_cap] [data_path] [basis_path]` | `core_dag.rs` | `NeurogenesisGraph` によるバイト成長 DAG のストリーム学習（同クラスに `apoptosis` 剪定を実装） |
| `mega-train <train_bin> <val_bin> <out_json> [cap] [batch] [max_bytes] [ckpt_path]` | `mega_core.rs` | BF16 ストリーム成長 DAG（ノード成長・`gentle_apoptosis` 剪定）の学習・計測 |
| `v190-save [corpus_dir] [ckpt_path] [epochs]` / `v190-bench [corpus_dir] [out_json] [cap]` | `reasoning_core.rs` | 推論言語コア DAG の学習・保存およびベンチマーク |
| `v220-train [corpus_dir] [ckpt_path] [out_json] [epochs]` | `executive_core.rs` | 司令塔（5モード分類・4演算分類・計算 BIO タガー）の学習・評価 |
| `v230-build [ckpt_path] [out_json]` | `hippocampus_core.rs` | 連想海馬（DG→CA3）の構築・評価 |
| `v240-train [corpus_dir] [ckpt_path] [out_json] [epochs]` | `reasoner_core.rs` | 関係推論野（述語記憶・多問評価）の学習・評価 |
| `v245-train [corpus_dir] [ckpt_path] [out_json] [epochs]` | `syntax_core.rs` | 10種 BIO 文字構文・節境界知覚モデルの学習・評価 |
| `v250-train [corpus_dir] [out_json]` | `broca_core.rs` / `metacog.rs` | メタ認知・`FactDistiller`・`DialogueIntentHead`・`BrocaDAG` の学習・評価 |

```sh
# 例: v25.0 発話・メタ認知の学習（v250_broca_metacog 配下のコーパスを使用）
python -X utf8 scripts/corpus/build_v250_broca_corpus.py /path/to/assets
cd enne-rs
ENNE_ROOT=/path/to/assets cargo run --release -- v250-train
```

### 主な環境変数（Rust 版 / コーパススクリプト）

| 変数 | 既定値 | 内容 |
|---|---|---|
| `ENNE_ROOT`（Rust 版） | `NEUROGRAPH_ROOT` → コンパイル時 `CARGO_MANIFEST_DIR` の親 | `rust_bench_v182/` や `v220_executive_core/` 等を置く基点ディレクトリ |
| `ENNE_ROOT`（Python コーパス 5 本） | 第1位置引数 → `ENNE_ROOT` → カレントディレクトリ `.` | 同上（位置引数が最優先、未指定時に環境変数 `ENNE_ROOT` を参照） |

学習済み `.bin` 重み、`v124.sqlite`（記事冒頭文索引）、`sft_raw/` や生コーパス等のデータ資産はリポジトリに含めていません。

---

## Python / GPU 版（`src/neurograph/`、v15.2 時点）

`src/neurograph/` は、v15.2（進化世代 gen_002）時点の研究コードのうち、初期の対話システムと DAG 統合を動かすのに必要なファイルだけを抜き出したものです。研究用リポジトリには試行錯誤した版を含めて135ファイルありましたが、対話端末・睡眠統合・メタ認知の学習と評価から import をたどって到達する45ファイル、DAG 統合の3ファイル、C++ カーネル3つ、設定用の `settings.py` を収録しました。

パッケージ名は `neurograph` のままです。学習済みの pickle がこの名前でクラスを参照しているためです。

## 研究時からの変更点

挙動は研究時のコードと同じです。変えたのは次の点だけです。

- 実験機のデータパス（コーパス・対話データ）の直書きを、`settings.py` の環境変数に置き換えた
- GPU の UUID の直書きを削除した。研究時と同じ検査が必要な場合は `ENNE_GPU_UUID` と `ENNE_STRICT_DEVICE=1` を指定する
- 改行コードを LF にそろえた
- コメントと docstring を整理した。各ファイル冒頭の説明を書き直し、事実だけを書いた日本語の説明は残して、開発中に書かれた英語の説明や誇張した表現（「Zero Hardcoded Rules!」など）は削除した。ログ出力の誇張表現も直した。整理の前後で、docstring を除いた構文木が一致することを確認している

コードには閾値や混合係数などの設計値が含まれます。方式の説明は [architecture.md](architecture.md) を参照してください。

## 動作環境

- Linux（`resource`・`fcntl` を使うため）
- CUDA の使える GPU。研究では RTX 3060 12GB を使い、対話時の PyTorch ピーク割当は約7.15GiB
- Python 3.10 以上、PyTorch、Triton、NumPy
- g++（OpenMP）。`engram_kernel.cpp` と `episodic_kernel.cpp` は初回実行時に自動でビルドされる。`motor_kernel_v50.cpp` は手動でビルドする

```sh
pip install -e .
g++ -O3 -march=native -fopenmp -shared -fPIC src/neurograph/motor_kernel_v50.cpp -o src/neurograph/libmotor_v50.so
```

## 含まれていないもの

学習済みモデル、記憶カートリッジ、検索索引、学習データは含みません。コードは作業ディレクトリからの相対パスでこれらを読みます。

| 資産 | 内容 |
|---|---|
| `merged_v111_broca_core.pt`、`merged_v111_adapter.pkl` | 統合後の本体DAGと読み出し |
| `engrams_v20/`、`engrams_v22/`、`engrams_v80/` | 記憶カートリッジ（分野別・対話） |
| `macro_v70/`、`instruct_v80/`、`reasoning_v90/`、`broca_v100/` | モーター単位・役割表現・対話適応などの学習済み部品 |
| `wiki_lead_v112/` | 記事冒頭文・本文の SQLite 索引 |
| `memory_gate_v120/` | ゲート・仲裁・ツール・発話・メタ認知の学習済みノード |
| `evolution_v140/` | 進化世代（教示・睡眠の記録と各世代のスナップショット） |
| `ENNE_DATA_DIR` | 事前学習コーパス（`easy_ja.train`・`aozora.train`・`wikipedia.train`） |
| `ENNE_SFT_DIR` | 整形済み対話データ（`train.jsonl` など） |

## 環境変数

| 変数 | 既定値 | 内容 |
|---|---|---|
| `ENNE_DATA_DIR` | `data` | 事前学習コーパスのディレクトリ |
| `ENNE_SFT_RAW_DIR` | `sft_raw` | 対話データの元ファイル |
| `ENNE_SFT_DIR` | `sft_data_v80` | 整形済み対話データ |
| `ENNE_GPU_UUID` | なし | 指定すると `CUDA_VISIBLE_DEVICES` との一致を検査する |
| `ENNE_STRICT_DEVICE` | なし | `1` で RTX 3060・`expandable_segments:True` の固定まで検査する |
| `NEUROGRAPH_EVOLUTION_DIR` | `evolution_v140` | 進化世代の保存先 |

## 入口

資産を置いたディレクトリで実行します。

```sh
python -m neurograph.chat_cli --trace            # 対話。/teach /correct /sleep /rollback /tools /metacog などを使える
python -m neurograph.chat_cli --once "128 * 37"  # 1回だけ応答する
python -m neurograph.evolve_v140                 # 睡眠統合を1回実行する
python -m neurograph.train_v152_metacog          # メタ認知の校正
python -m neurograph.run_v152_metacog_eval       # v15.2 の評価
python -m neurograph.run_v14_scale90m            # 本体DAGの継続学習（v1.4）
python -m neurograph.run_v110_merge              # 対話用DAGと基盤DAGの統合（v11.0）
python -m neurograph.run_v111_merge_polish       # 統合後の読み出し調整（v11.1）
```

## 版名の読み方

ファイル名の版番号は、その部品を作った版です。古い版の部品も現行版から使っているため、版名つきのファイルが並んでいます。

`lateral_brain_v13.py`、`run_v13_unified.py`、`run_v14_scale90m.py` の `v13`・`v14` は v1.3・v1.4 のことで、v13.x・v14.0 とは別です。

## 部品とファイル

| 部品 | ファイル |
|---|---|
| 本体：成長するバイトDAG | `model.py`、`model_v2.py`、`growing_graph_v8.py`、`utf8_dag_v88.py`、`triton_dag_kernel.py`、`executive_v12.py`、`lateral_brain_v13.py`、`lateral_v13b.py`、`run_v13_unified.py`、`run_v14_scale90m.py` |
| 作業記憶・長期記憶 | `memory_v3a.py`、`memory_v3b.py`、`unified_brain_v62.py` |
| 外部記憶（バイト記憶・文章記憶） | `engram_v20.py`、`engram_kernel.cpp`、`episodic_v21.py`、`episodic_kernel.cpp` |
| デコーダ | `reason_decode_v30.py`、`speculative_decode_v31.py` |
| モーター単位・役割表現 | `thought_cortex_v50.py`、`motor_kernel_v50.cpp`、`role_cortex_v60.py` |
| 対話適応 | `sft_data_v80.py`、`instruct_cortex_v80.py`、`reasoning_graph_v90.py`、`broca_cortex_v100.py` |
| DAG統合 | `cortical_merge_v110.py`、`run_v110_merge.py`、`run_v111_merge_polish.py` |
| 記事冒頭文検索と統合脳 | `wiki_lead_hippocampus_v112.py`、`run_v112_lead_qa.py` |
| 記憶ゲート・合議・仲裁 | `memory_gate_v120.py`、`sampling_v123.py`、`panel_v124.py`、`consensus_arbiter_v125.py` |
| ツール・発話 | `tool_cortex_v130.py`、`build_v130_body_index.py`、`build_v131_body_index.py`、`train_v130.py`、`run_v130_eval.py`、`utterance_cortex_v131.py`、`action_axis_v132.py`、`operators_v132.py` |
| 教示・睡眠・世代管理 | `evolution_v140.py`、`gate_v140.py`、`curiosity_v140.py`、`evolve_v140.py` |
| メタ認知 | `metacog_v152.py`、`train_v152_metacog.py`、`run_v152_metacog_eval.py` |
| 対話端末・設定 | `chat_cli.py`、`settings.py` |

## 更新するとき

ソースを更新するときは、`pyproject.toml` の version、README、docs、結果JSONを同時に更新します。公開前に次の検査を通します。版のずれと、実験機のパス・GPU の UUID などの残りを調べます。

```sh
python -X utf8 scripts/check_release.py
```
