"""Bảng màu và theme thống nhất cho toàn bộ báo cáo.

Tông màu: Xanh dương Navy / Cam / Trắng — chuyên nghiệp, hiện đại.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Màu sắc chính ─────────────────────────────────────────────────────────
BLUE_DARK   = "#1565C0"   # Primary
BLUE_MID    = "#1976D2"
BLUE_LIGHT  = "#42A5F5"
ORANGE      = "#F57C00"   # Secondary / Accent
ORANGE_LITE = "#FFB74D"
TEAL        = "#00897B"
RED         = "#D32F2F"
GRAY        = "#607D8B"
GRAY_LITE   = "#ECEFF1"
WHITE       = "#FFFFFF"
BLACK       = "#212121"

# Palette cho multi-class (15 gesture + 5 noise)
PALETTE_20 = [
    "#1565C0", "#F57C00", "#00897B", "#D32F2F", "#7B1FA2",
    "#0288D1", "#FFB300", "#2E7D32", "#C62828", "#4527A0",
    "#00838F", "#EF6C00", "#1B5E20", "#880E4F", "#37474F",
    "#0D47A1", "#E65100", "#004D40", "#B71C1C", "#311B92",
]

# 6 kênh IMU
CHANNEL_COLORS = {
    "ax_g":   "#1565C0",
    "ay_g":   "#00897B",
    "az_g":   "#D32F2F",
    "gx_dps": "#F57C00",
    "gy_dps": "#7B1FA2",
    "gz_dps": "#0288D1",
}


def apply_theme() -> None:
    """Áp dụng rcParams toàn cục — gọi 1 lần ở đầu script."""
    mpl.rcParams.update({
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
        "axes.edgecolor":    GRAY,
        "axes.labelcolor":   BLACK,
        "axes.grid":         True,
        "grid.color":        GRAY_LITE,
        "grid.linewidth":    0.6,
        "xtick.color":       BLACK,
        "ytick.color":       BLACK,
        "text.color":        BLACK,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "figure.dpi":        120,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.facecolor": WHITE,
    })


def savefig(fig: plt.Figure, path: str, **kwargs) -> None:
    """Lưu figure với cài đặt chuẩn."""
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)
