# ソースコード

`src/neurograph/` は、v15.2（進化世代 gen_002）時点の研究コードのうち、現行の対話システムと DAG 統合を動かすのに必要なファイルだけを抜き出したものです。研究用リポジトリには試行錯誤した版を含めて135ファイルありますが、現行版の入口（対話端末・睡眠統合・メタ認知の学習と評価）から import をたどって到達する45ファイル、DAG 統合の3ファイル、C++ カーネル3つ、設定用の `settings.py` を収録しました。

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
