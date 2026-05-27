"""
GAFDataset – nạp dữ liệu IMU từ manifest, áp dụng pipeline GADF và trả về
tensor ảnh (C, H, W) cho GAFormer.

Pipeline mỗi mẫu:
  CSV  →  smooth (5 điểm)  →  min-max [0,1]  →  add mag channels
       →  GADF (N×N×8)    →  resize (224×224×8)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess, add_magnitude_channels
from gadf import signal_to_gadf_tensor, signal_to_gasf_tensor

DEFAULT_COLS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
IMG_SIZE = 224


def label_sort_key(label_id: str) -> tuple[int, int]:
    label_id = label_id.upper()
    prefix = label_id[:1]
    try:
        number = int(label_id[1:])
    except ValueError:
        number = 999
    prefix_order = {"G": 0, "N": 1}.get(prefix, 2)
    return prefix_order, number


def build_label_list(manifest_df: pd.DataFrame) -> list[str]:
    return sorted(manifest_df["label_id"].astype(str).unique(), key=label_sort_key)


def parse_subjects(subjects: str | None, fallback: Iterable[str]) -> list[str]:
    if subjects is None:
        return list(fallback)
    subjects = subjects.strip()
    if not subjects:
        return []
    return [s.strip() for s in subjects.split(",") if s.strip()]


@dataclass
class GAFDatasetConfig:
    manifest_path: Path
    root_dir: Path
    subjects: list[str]
    label_to_index: dict[str, int]
    img_size: int = IMG_SIZE
    smooth_window: int = 5
    transform: str = "gadf"   # "gadf" hoặc "gasf"


class GAFDataset(Dataset):
    """Dataset trả về (image_tensor, label) trong đó image_tensor có shape (8, img_size, img_size)."""

    def __init__(self, cfg: GAFDatasetConfig) -> None:
        manifest = pd.read_csv(cfg.manifest_path).reset_index(drop=True)
        subject_ids = manifest["subject_id"].astype(str)

        if cfg.subjects:
            mask = subject_ids.isin(set(cfg.subjects))
            manifest = manifest[mask].reset_index(drop=True)

        self.manifest = manifest
        self.root_dir = cfg.root_dir
        self.label_to_index = cfg.label_to_index
        self.img_size = cfg.img_size
        self.smooth_window = cfg.smooth_window

        # Tiền tính toán tất cả ảnh GADF và cache trong RAM
        self.transform = cfg.transform
        self.images: list[torch.Tensor] = []
        self.targets: list[int] = []

        transform_fn = signal_to_gasf_tensor if self.transform == "gasf" else signal_to_gadf_tensor
        print(f"  Caching {len(manifest)} mẫu {self.transform.upper()} (img_size={cfg.img_size})...")
        for _, row in manifest.iterrows():
            csv_path = cfg.root_dir / Path(str(row["path"]).replace("\\", "/"))
            raw = pd.read_csv(csv_path, usecols=DEFAULT_COLS)[DEFAULT_COLS].to_numpy(dtype=np.float32)

            x = preprocess(raw, window=self.smooth_window)   # (N,6) in [0,1]
            x = add_magnitude_channels(x)                     # (N,8)
            img = transform_fn(x, self.img_size)              # (8,H,W)

            self.images.append(img)
            self.targets.append(self.label_to_index[str(row["label_id"])])

        self.targets_tensor = torch.tensor(self.targets, dtype=torch.long)
        print(f"  Cache xong: {len(self.images)} ảnh {self.transform.upper()}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.targets_tensor[idx]


# ---------------------------------------------------------------------------
# RawDataset – tín hiệu 1D thô cho CNN-biLSTM
# ---------------------------------------------------------------------------

class RawDataset(Dataset):
    """
    Dataset trả về (signal_tensor, label) với tín hiệu IMU 1D đã smooth+normalize.
    Tín hiệu được cắt hoặc đệm về độ dài cố định seq_len.
    Đầu ra shape: (seq_len, 6)
    """

    def __init__(self, cfg: GAFDatasetConfig, seq_len: int = 150) -> None:
        manifest = pd.read_csv(cfg.manifest_path).reset_index(drop=True)
        if cfg.subjects:
            manifest = manifest[
                manifest["subject_id"].astype(str).isin(set(cfg.subjects))
            ].reset_index(drop=True)

        self.seq_len = seq_len
        self.signals: list[torch.Tensor] = []
        self.targets: list[int] = []

        print(f"  Caching {len(manifest)} mẫu RAW (seq_len={seq_len})...")
        for _, row in manifest.iterrows():
            csv_path = cfg.root_dir / Path(str(row["path"]).replace("\\", "/"))
            raw = pd.read_csv(csv_path, usecols=DEFAULT_COLS)[DEFAULT_COLS].to_numpy(dtype=np.float32)
            x = preprocess(raw, window=cfg.smooth_window)   # (N, 6) in [0,1]
            x = self._pad_or_truncate(x, seq_len)            # (seq_len, 6)
            self.signals.append(torch.from_numpy(x))
            self.targets.append(cfg.label_to_index[str(row["label_id"])])

        self.targets_tensor = torch.tensor(self.targets, dtype=torch.long)
        print(f"  Cache xong: {len(self.signals)} tín hiệu thô")

    def _pad_or_truncate(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        n = x.shape[0]
        if n >= seq_len:
            return x[:seq_len]
        pad = np.zeros((seq_len - n, x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.targets_tensor[idx]
