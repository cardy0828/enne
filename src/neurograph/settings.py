"""実行環境に依存する設定値。研究時は実験機のパスと GPU を直書きしていたものを、環境変数で与える形にした。"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ENNE_DATA_DIR", "data"))
SFT_RAW_DIR = Path(os.environ.get("ENNE_SFT_RAW_DIR", "sft_raw"))
SFT_DIR = Path(os.environ.get("ENNE_SFT_DIR", "sft_data_v80"))
GPU = os.environ.get("ENNE_GPU_UUID") or None
STRICT_DEVICE = os.environ.get("ENNE_STRICT_DEVICE") == "1"


def require_gpu() -> None:
    """ENNE_GPU_UUID が指定されていれば、その GPU だけが見えていることを確かめる。"""
    if GPU is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("CUDA_VISIBLE_DEVICES に ENNE_GPU_UUID と同じ UUID を指定してください")
