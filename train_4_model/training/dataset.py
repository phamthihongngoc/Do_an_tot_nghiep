"""Dataset, manifest loading và subject-based split cho IMU gesture data.

Luồng:
    processed_50hz/
    └── manifest_50hz.csv  (sinh bởi resample_to_50hz.py)
        path, subject_id, label_id, ...

    → IMUDataset (torch.Dataset)  hoặc
    → load_numpy_splits()  (cho sklearn RF)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích load dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def load_labels_json(labels_json_path: Path) -> List[str]:
    """Trả về danh sách label IDs theo thứ tự (G1, G2, ..., N1, ...)."""
    data = json.loads(labels_json_path.read_text(encoding="utf-8"))
    return [item["id"] for item in data.get("labels", [])]


def load_manifest(processed_dir: Path, hz: int = 50) -> pd.DataFrame:
    """Load manifest CSV từ thư mục processed."""
    candidates = [
        processed_dir / f"manifest_{hz}hz.csv",
        processed_dir / "manifest_50hz.csv",
        processed_dir / "manifest.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Không tìm thấy manifest trong {processed_dir}.\n"
        "Chạy: python data_collection/resample_to_50hz.py --output <processed_dir>"
    )


def _read_csv_sequence(path: Path, base_dir: Path) -> np.ndarray | None:
    """Đọc 1 file CSV → (T, 6) float32. Trả None nếu lỗi."""
    # Normalize path separators (Windows backslash → forward slash on Linux/Mac)
    norm_path = Path(str(path).replace("\\", "/"))
    full = base_dir / norm_path if not norm_path.is_absolute() else norm_path
    if not full.exists():
        return None
    try:
        df = pd.read_csv(full, usecols=CHANNELS)
        missing = [c for c in CHANNELS if c not in df.columns]
        if missing:
            return None
        return df[CHANNELS].to_numpy(dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Split theo subject
# ─────────────────────────────────────────────────────────────────────────────
def split_manifest(
    manifest: pd.DataFrame,
    test_subjects: List[str],
    val_subjects: List[str],
    label_filter: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Phân chia manifest thành train/val/test theo subject ID."""
    if label_filter is not None:
        manifest = manifest[manifest["label_id"].isin(label_filter)]

    test_mask = manifest["subject_id"].isin(test_subjects)
    val_mask  = manifest["subject_id"].isin(val_subjects)
    return {
        "test":  manifest[test_mask].reset_index(drop=True),
        "val":   manifest[val_mask].reset_index(drop=True),
        "train": manifest[~test_mask & ~val_mask].reset_index(drop=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chuẩn hóa (fit trên train, apply trên val/test)
# ─────────────────────────────────────────────────────────────────────────────
def compute_normalization(sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """mean, std theo chiều channel từ danh sách chuỗi (T, 6)."""
    all_data = np.concatenate([s for s in sequences if s is not None], axis=0)
    mean = all_data.mean(axis=0).astype(np.float32)
    std  = all_data.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}


def normalize(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((seq - mean) / std).astype(np.float32)


def resample_1d(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation (T, C) → (target_len, C)."""
    if seq.shape[0] == target_len:
        return seq
    src = np.linspace(0, 1, seq.shape[0])
    dst = np.linspace(0, 1, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class IMUDataset(Dataset):
    """Dataset trả về (tensor[T,6], label_int) cho DataLoader.

    Args:
        df          : manifest split (train/val/test)
        base_dir    : thư mục gốc để resolve đường dẫn tương đối trong manifest
        label_ids   : danh sách label theo thứ tự (dùng để map ID → int)
        target_len  : độ dài chuỗi đồng nhất (resample nếu cần)
        mean, std   : tham số chuẩn hóa (None = không chuẩn hóa)
        augment_fn  : hàm augmentation (sequence: ndarray) → ndarray | None
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: Path,
        label_ids: List[str],
        target_len: int = 201,
        mean: np.ndarray | None = None,
        std:  np.ndarray | None = None,
        augment_fn: Any = None,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.base_dir  = Path(base_dir)
        self.label2idx = {lbl: i for i, lbl in enumerate(label_ids)}
        self.target_len = target_len
        self.mean      = mean
        self.std       = std
        self.augment   = augment_fn

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row  = self.df.iloc[idx]
        seq  = _read_csv_sequence(row["path"], self.base_dir)
        if seq is None or len(seq) == 0:
            seq = np.zeros((self.target_len, len(CHANNELS)), dtype=np.float32)

        seq = resample_1d(seq, self.target_len)

        if self.augment is not None:
            seq = self.augment(seq)

        if self.mean is not None and self.std is not None:
            seq = normalize(seq, self.mean, self.std)

        label = self.label2idx.get(str(row["label_id"]), 0)
        return torch.from_numpy(seq), label


# ─────────────────────────────────────────────────────────────────────────────
# Numpy arrays cho sklearn (RF)
# ─────────────────────────────────────────────────────────────────────────────
def load_numpy_splits(
    cfg: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load dữ liệu thành numpy arrays (X, y) cho RF.

    Trả về dict với keys 'train', 'val', 'test'.
    X shape: (N, target_len, 6);  y shape: (N,)
    """
    from training import REPO_ROOT

    paths      = cfg["paths"]
    split_cfg  = cfg["split"]
    prep_cfg   = cfg["preprocessing"]
    labels_all = load_labels_json(Path(paths["labels_json"]))
    label_filter = split_cfg.get("label_filter")
    if label_filter:
        labels_all = [l for l in labels_all if l in label_filter]
    label2idx = {l: i for i, l in enumerate(labels_all)}

    processed_dir = Path(paths["processed"])
    manifest      = load_manifest(processed_dir)
    splits        = split_manifest(
        manifest,
        test_subjects=split_cfg["test_subjects"],
        val_subjects=split_cfg["val_subjects"],
        label_filter=label_filter,
    )

    target_len = prep_cfg["target_len"]
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, df in splits.items():
        seqs, labels = [], []
        for _, row in df.iterrows():
            seq = _read_csv_sequence(row["path"], processed_dir.parent)
            if seq is None:
                continue
            seq = resample_1d(seq, target_len)
            seqs.append(seq)
            labels.append(label2idx.get(str(row["label_id"]), 0))
        if seqs:
            result[split_name] = (np.stack(seqs), np.array(labels, dtype=np.int64))
    return result
