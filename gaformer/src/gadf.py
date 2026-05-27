"""
Bước 2 – Biến đổi Gramian Angular Difference Field (GADF).

Thuật toán theo luận án (công thức 3.13 – 3.15):
  1. Mã hóa tọa độ cực:  phi_i = arccos(x_scaled_i)
  2. GADF(i,j) = sin(phi_i - phi_j)

Đầu vào  : chuỗi 1D đã chuẩn hóa về [0,1], shape (N,)
Đầu ra   : ma trận (N, N) GADF
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gadf_1d(x: np.ndarray) -> np.ndarray:
    """Tính GADF cho một kênh tín hiệu 1D.

    Args:
        x: (N,) float trong [0, 1]
    Returns:
        (N, N) float32  – ma trận GADF
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float64)
    phi = np.arccos(x)                        # (N,)
    phi_i = phi[:, None]                      # (N, 1)
    phi_j = phi[None, :]                      # (1, N)
    return np.sin(phi_i - phi_j).astype(np.float32)


def gadf_multichannel(x: np.ndarray) -> np.ndarray:
    """Tính GADF cho tất cả kênh và xếp chồng thành tensor đa kênh.

    Args:
        x: (N, C) float trong [0, 1]   C = 8 kênh [ax,ay,az,gx,gy,gz,ma,mg]
    Returns:
        (C, N, N) float32
    """
    n_channels = x.shape[1]
    maps = [gadf_1d(x[:, c]) for c in range(n_channels)]
    return np.stack(maps, axis=0)   # (C, N, N)


def resize_gadf(gadf: np.ndarray, size: int = 224) -> torch.Tensor:
    """Resize (C, N, N) → (C, size, size) bằng bilinear interpolation.

    Args:
        gadf: numpy (C, N, N)
        size: kích thước ảnh đầu ra
    Returns:
        torch.Tensor (C, size, size) float32
    """
    t = torch.from_numpy(gadf).unsqueeze(0)          # (1, C, N, N)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)                               # (C, size, size)


def signal_to_gadf_tensor(x: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Pipeline hoàn chỉnh GADF: (N, 8) → tensor (8, img_size, img_size).

    Gọi sau khi đã qua preprocess() + add_magnitude_channels().
    """
    gadf = gadf_multichannel(x)                  # (8, N, N)
    tensor = resize_gadf(gadf, img_size)          # (8, img_size, img_size)
    return tensor


# ---------------------------------------------------------------------------
# GASF – Gramian Angular Summation Field
# GASF(i,j) = cos(phi_i + phi_j)  (đối xứng, thông tin tương quan tổng hợp)
# ---------------------------------------------------------------------------

def gasf_1d(x: np.ndarray) -> np.ndarray:
    """Tính GASF cho một kênh tín hiệu 1D.

    Args:
        x: (N,) float trong [0, 1]
    Returns:
        (N, N) float32  – ma trận GASF
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float64)
    phi = np.arccos(x)                        # (N,)
    phi_i = phi[:, None]                      # (N, 1)
    phi_j = phi[None, :]                      # (1, N)
    return np.cos(phi_i + phi_j).astype(np.float32)


def gasf_multichannel(x: np.ndarray) -> np.ndarray:
    """Tính GASF cho tất cả kênh.

    Args:
        x: (N, C) float trong [0, 1]
    Returns:
        (C, N, N) float32
    """
    return np.stack([gasf_1d(x[:, c]) for c in range(x.shape[1])], axis=0)


def signal_to_gasf_tensor(x: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Pipeline hoàn chỉnh GASF: (N, 8) → tensor (8, img_size, img_size)."""
    gasf = gasf_multichannel(x)
    return resize_gadf(gasf, img_size)
