"""
Bước 1 – Tiền xử lý tín hiệu IMU.

Hai phép biến đổi theo luận án:
  (3.11) Làm mịn 5 điểm (sliding-window mean)
  (3.12) Chuẩn hóa Min–Max về [0, 1]
"""
from __future__ import annotations

import numpy as np


def smooth_signal(x: np.ndarray, window: int = 5) -> np.ndarray:
    """5-point uniform moving-average filter (công thức 3.11).

    x shape: (N, C)  hoặc (N,)
    Các mẫu ở rìa được xử lý bằng edge-padding để không thay đổi độ dài.
    """
    if x.ndim == 1:
        return _smooth_1d(x, window)
    return np.stack([_smooth_1d(x[:, c], window) for c in range(x.shape[1])], axis=1)


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-Max chuẩn hóa về [0, 1] trên từng kênh (công thức 3.12).

    x shape: (N, C)
    """
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    return (x - x_min) / np.maximum(x_max - x_min, eps)


def preprocess(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Áp dụng làm mịn → chuẩn hóa Min-Max.

    Args:
        x: (N, 6) – [ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps]
    Returns:
        (N, 6) float32 trong [0, 1]
    """
    x = smooth_signal(x, window)
    x = minmax_normalize(x)
    return x.astype(np.float32)


def add_magnitude_channels(x: np.ndarray) -> np.ndarray:
    """Thêm kênh biên độ tổng hợp ma và mg (công thức trong luận án).

    Args:
        x: (N, 6) đã chuẩn hóa – [ax, ay, az, gx, gy, gz]
    Returns:
        (N, 8) – [ax, ay, az, gx, gy, gz, ma, mg]
    """
    ma = np.linalg.norm(x[:, :3], axis=1, keepdims=True)   # |acc|
    mg = np.linalg.norm(x[:, 3:], axis=1, keepdims=True)   # |gyro|
    # chuẩn hóa biên độ về [0, 1]
    ma = minmax_normalize(ma)
    mg = minmax_normalize(mg)
    return np.concatenate([x, ma, mg], axis=1).astype(np.float32)
