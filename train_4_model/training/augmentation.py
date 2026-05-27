"""Data Augmentation cho chuỗi IMU 1D — áp dụng ngẫu nhiên khi train.

Tất cả hàm nhận (T, C) ndarray → (T, C) ndarray.

Các kỹ thuật:
    1. Jitter          — thêm nhiễu Gauss nhỏ
    2. Scale           — nhân ngẫu nhiên biên độ
    3. TimeWarp        — giãn/nén trục thời gian cục bộ
    4. MagnitudeWarp   — biến dạng trơn biên độ theo thời gian
    5. ChannelDropout  — zero một kênh ngẫu nhiên
    6. Compose         — kết hợp các augmentation theo xác suất
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Các phép augmentation đơn lẻ
# ─────────────────────────────────────────────────────────────────────────────
def jitter(seq: np.ndarray, sigma: float = 0.015) -> np.ndarray:
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def scale(seq: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    factor = np.random.uniform(lo, hi)
    return (seq * factor).astype(np.float32)


def time_warp(seq: np.ndarray, max_ratio: float = 0.10) -> np.ndarray:
    """Giãn/nén trục thời gian bằng cách warp điểm neo ngẫu nhiên."""
    T, C = seq.shape
    # Chọn 1 điểm neo và warp ±max_ratio
    pivot = np.random.randint(T // 4, 3 * T // 4)
    warp  = int(T * np.random.uniform(-max_ratio, max_ratio))
    new_pivot = min(max(pivot + warp, 1), T - 2)
    src = np.array([0, pivot,     T - 1], dtype=np.float64)
    dst = np.array([0, new_pivot, T - 1], dtype=np.float64)
    t_old = np.linspace(0, T - 1, T)
    t_new = np.interp(t_old, dst, src)   # Map new-time → old-time
    out = np.empty_like(seq)
    for c in range(C):
        out[:, c] = np.interp(t_new, t_old, seq[:, c])
    return out.astype(np.float32)


def magnitude_warp(seq: np.ndarray, sigma: float = 0.15, knots: int = 4) -> np.ndarray:
    """Nhân chuỗi với đường cong warp mượt ngẫu nhiên."""
    T = seq.shape[0]
    # Tạo knots ngẫu nhiên rồi cubic-spline interpolate
    from scipy.interpolate import CubicSpline
    knot_x = np.linspace(0, T - 1, knots + 2)
    knot_y = np.random.normal(1.0, sigma, size=knots + 2)
    warp_curve = CubicSpline(knot_x, knot_y)(np.arange(T))
    warp_curve = np.clip(warp_curve, 0.5, 1.5).reshape(-1, 1).astype(np.float32)
    return (seq * warp_curve).astype(np.float32)


def channel_dropout(seq: np.ndarray, p: float = 0.10) -> np.ndarray:
    """Zero một kênh ngẫu nhiên với xác suất p."""
    if np.random.rand() < p:
        ch = np.random.randint(seq.shape[1])
        seq = seq.copy()
        seq[:, ch] = 0.0
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Compose — đường ống augmentation chính
# ─────────────────────────────────────────────────────────────────────────────
class AugmentPipeline:
    """Áp dụng ngẫu nhiên các augmentation theo xác suất."""

    def __init__(self, cfg: dict) -> None:
        self.enabled           = cfg.get("enabled", True)
        self.jitter_sigma      = cfg.get("jitter_sigma", 0.015)
        self.scale_lo, self.scale_hi = cfg.get("scale_range", [0.9, 1.1])
        self.time_warp_max     = cfg.get("time_warp_max", 0.10)
        self.magnitude_warp    = cfg.get("magnitude_warp", True)
        self.channel_dropout_p = cfg.get("channel_dropout_p", 0.10)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return seq
        seq = jitter(seq, self.jitter_sigma)
        seq = scale(seq, self.scale_lo, self.scale_hi)
        if np.random.rand() < 0.5:
            seq = time_warp(seq, self.time_warp_max)
        if self.magnitude_warp and np.random.rand() < 0.5:
            try:
                seq = magnitude_warp(seq)
            except ImportError:
                pass  # scipy không có → bỏ qua magnitude warp
        seq = channel_dropout(seq, self.channel_dropout_p)
        return seq


def build_augment(cfg: dict) -> AugmentPipeline | None:
    """Trả về AugmentPipeline nếu enabled, ngược lại trả None."""
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return None
    return AugmentPipeline(aug_cfg)
