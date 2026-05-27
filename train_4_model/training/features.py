"""Feature engineering cho Random Forest.

extract_features_rf(sequences)  →  (N, num_features)
Dùng bởi scripts/train_rf_latefusion.py và evaluate.py.

Các đặc trưng per kênh (10 thống kê):
    mean, std, min, max, rms, kurtosis, skewness, energy,
    zero_crossing_rate, iqr

Đặc trưng tương quan giữa các cặp kênh: corr(ax,ay), corr(ax,az), ...

Tổng: 6 kênh × 10 = 60 + 15 cặp = 75 đặc trưng.

Lưu ý:
    Demo/predict.py dùng extract_rf_features() riêng (24-d đơn giản).
    LateFusionRF export ra demo nhận 24-d → tương thích.
    Nhưng khi train bằng script này, dùng 75-d để học tốt hơn.
    models_rf.py biết cách chia 24-d cho demo; script train tự xử lý 75-d.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
NUM_CHANNELS = 6

# Chỉ số kênh accel / gyro
ACCEL_IDX = [0, 1, 2]
GYRO_IDX  = [3, 4, 5]


def _per_channel_features(data: np.ndarray) -> np.ndarray:
    """data: (T, C) → (10*C,) features."""
    T, C = data.shape
    feats = []
    for c in range(C):
        x = data[:, c]
        feats.append(x.mean())
        feats.append(x.std())
        feats.append(x.min())
        feats.append(x.max())
        feats.append(float(np.sqrt(np.mean(x ** 2))))          # RMS
        feats.append(float(kurtosis(x, fisher=True)))           # Kurtosis
        feats.append(float(skew(x)))                            # Skewness
        feats.append(float(np.sum(x ** 2) / T))                 # Energy
        zc = float(np.sum(np.diff(np.sign(x)) != 0)) / T       # ZCR
        feats.append(zc)
        feats.append(float(np.percentile(x, 75) - np.percentile(x, 25)))  # IQR
    return np.array(feats, dtype=np.float32)


def _correlation_features(data: np.ndarray) -> np.ndarray:
    """Tất cả cặp tương quan Pearson trong C kênh: C*(C-1)/2 features."""
    C = data.shape[1]
    feats = []
    for i in range(C):
        for j in range(i + 1, C):
            corr = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
            feats.append(0.0 if np.isnan(corr) else corr)
    return np.array(feats, dtype=np.float32)


def extract_features_single(seq: np.ndarray) -> np.ndarray:
    """(T, 6) → (75,) feature vector."""
    pc = _per_channel_features(seq)      # (60,)
    cc = _correlation_features(seq)      # (15,)
    return np.concatenate([pc, cc])


def extract_features_rf(sequences: np.ndarray) -> np.ndarray:
    """(N, T, 6) → (N, 75) feature matrix dùng cho train RF."""
    return np.stack([extract_features_single(s) for s in sequences])


def extract_features_branch(
    sequences: np.ndarray,
    branch: str = "all",  # "all" | "accel" | "gyro"
) -> np.ndarray:
    """Trích xuất features cho một nhánh cụ thể."""
    if branch == "accel":
        seqs = sequences[:, :, ACCEL_IDX]
    elif branch == "gyro":
        seqs = sequences[:, :, GYRO_IDX]
    else:
        seqs = sequences
    return np.stack([
        np.concatenate([_per_channel_features(s), _correlation_features(s)])
        for s in seqs
    ])


# ── Đặc trưng đơn giản 24-d (tương thích demo) ────────────────────────────
def extract_features_demo(sequences: np.ndarray) -> np.ndarray:
    """(N, T, 6) → (N, 24) — giống demo/predict.py.extract_rf_features()."""
    mean  = sequences.mean(axis=1)
    std   = sequences.std(axis=1)
    min_v = sequences.min(axis=1)
    max_v = sequences.max(axis=1)
    return np.concatenate([mean, std, min_v, max_v], axis=1).astype(np.float32)
