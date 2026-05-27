"""Sơ đồ minh họa kiến trúc các mô hình Deep Learning."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from visualization.style import BLUE_DARK, BLUE_MID, BLUE_LIGHT, ORANGE, WHITE, GRAY, apply_theme, savefig


def _box(ax, x, y, w, h, text, facecolor=None, fontsize=9, bold=False):
    fc = facecolor or BLUE_DARK
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=WHITE, linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=WHITE,
            fontweight="bold" if bold else "normal", wrap=True,
            multialignment="center")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))


def _label(ax, x, y, text, fontsize=8):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=GRAY, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# CNN 1D
# ─────────────────────────────────────────────────────────────────────────────
def plot_cnn_architecture(
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(-1, 3); ax.axis("off")

    layers = [
        (0.8,  "Input\n(B, 201, 6)",           GRAY,      True),
        (2.3,  "Conv1D 6→32\nk=7, BN, ReLU\nMaxPool",  BLUE_DARK, False),
        (4.0,  "Conv1D 32→64\nk=5, BN, ReLU\nMaxPool",  BLUE_DARK, False),
        (5.7,  "Conv1D 64→128\nk=3, BN, ReLU\nMaxPool", BLUE_DARK, False),
        (7.4,  "AdaptiveAvgPool\n→ Flatten",    BLUE_MID,  False),
        (9.1,  "FC 128→128\nReLU, Dropout",     BLUE_MID,  False),
        (10.8, "FC 128→N\n(N classes)",         ORANGE,    False),
        (12.5, "Output\n(B, N)",                GRAY,      True),
    ]

    bw, bh = 1.35, 0.9
    for i, (x, txt, color, bold) in enumerate(layers):
        _box(ax, x, 1.0, bw, bh, txt, facecolor=color, fontsize=8, bold=bold)
        if i > 0:
            _arrow(ax, layers[i-1][0] + bw/2, 1.0, x - bw/2, 1.0)

    ax.set_title("Sơ đồ kiến trúc CNN 1D cho chuỗi tín hiệu IMU",
                 fontsize=13, fontweight="bold", color=BLUE_DARK, pad=10)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CNN + BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
def plot_lstm_architecture(
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 5); ax.axis("off")

    # Top row: CNN encoder
    cnn_layers = [
        (1.2,  3.8, "Input\n(B, 201, 6)",          GRAY,      True),
        (3.0,  3.8, "Conv1D 6→32\nk=5, ReLU",      BLUE_DARK, False),
        (5.0,  3.8, "Conv1D 32→64\nk=5, ReLU\nDropout", BLUE_DARK, False),
    ]
    # Middle: reshape
    reshape = (7.0, 3.8, "Reshape\n(B, T', 64)", BLUE_MID, False)

    # Bottom row: BiLSTM + FC
    lstm_layers = [
        (7.0,  1.8, "BiLSTM\nhidden=128\n2 layers",       "#7B1FA2", False),
        (9.2,  1.8, "Last hidden\n(B, 256)",               BLUE_MID,  False),
        (11.1, 1.8, "FC 256→128\nReLU, Dropout",          BLUE_MID,  False),
        (12.9, 1.8, "FC 128→N\n→ Output",                 ORANGE,    False),
    ]

    bw, bh = 1.5, 0.85
    for i, (x, y, txt, color, bold) in enumerate(cnn_layers):
        _box(ax, x, y, bw, bh, txt, facecolor=color, fontsize=8, bold=bold)
        if i > 0:
            _arrow(ax, cnn_layers[i-1][0] + bw/2, cnn_layers[i-1][1],
                   x - bw/2, y)

    _box(ax, *reshape[:2], bw, bh, reshape[2], facecolor=reshape[3], fontsize=8)
    _arrow(ax, cnn_layers[-1][0] + bw/2, cnn_layers[-1][1],
           reshape[0] - bw/2, reshape[1])
    # vertical arrow reshape → BiLSTM
    _arrow(ax, reshape[0], reshape[1] - bh/2, lstm_layers[0][0], lstm_layers[0][1] + bh/2)

    for i, (x, y, txt, color, bold) in enumerate(lstm_layers):
        _box(ax, x, y, bw, bh, txt, facecolor=color, fontsize=8, bold=bold)
        if i > 0:
            _arrow(ax, lstm_layers[i-1][0] + bw/2, lstm_layers[i-1][1], x - bw/2, y)

    ax.text(3.0, 4.85, "CNN Encoder (Trích xuất đặc trưng cục bộ)",
            ha="center", va="center", fontsize=9, color=BLUE_DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BLUE_LIGHT, alpha=0.25))
    ax.text(10.0, 0.75, "Bidirectional LSTM (Mô hình hóa thời gian)",
            ha="center", va="center", fontsize=9, color="#7B1FA2",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDE7F6", alpha=0.5))

    ax.set_title("Sơ đồ kiến trúc CNN + BiLSTM cho chuỗi tín hiệu IMU",
                 fontsize=13, fontweight="bold", color=BLUE_DARK, pad=10)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Transformer
# ─────────────────────────────────────────────────────────────────────────────
def plot_transformer_architecture(
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 5); ax.axis("off")

    bw, bh = 1.5, 0.85
    # Main pipeline
    main = [
        (0.9,  2.5, "Input\n(B, 201, 6)",              GRAY,      True),
        (2.7,  2.5, "Linear Proj.\n6 → d_model=64",    BLUE_DARK, False),
        (4.5,  2.5, "Positional\nEncoding",             BLUE_MID,  False),
    ]
    # Encoder block (repeated 3×)
    enc_x = 6.8
    enc_layers = [
        (enc_x, 4.0, "LayerNorm",                       BLUE_DARK, False),
        (enc_x, 2.9, "MultiHead\nAttention\n(4 heads)", "#7B1FA2", False),
        (enc_x, 1.7, "LayerNorm\n+ FFN\n64→256→64",    BLUE_DARK, False),
    ]
    after_enc = [
        (9.2,  2.5, "× 3 Encoder\nLayers",             BLUE_MID,  False),
        (11.0, 2.5, "Global\nAvgPool\n→ (B, 64)",       BLUE_MID,  False),
        (12.8, 2.5, "FC 64→N\nOutput",                  ORANGE,    False),
    ]

    for i, (x, y, txt, color, bold) in enumerate(main):
        _box(ax, x, y, bw, bh, txt, facecolor=color, fontsize=8, bold=bold)
        if i > 0:
            _arrow(ax, main[i-1][0] + bw/2, main[i-1][1], x - bw/2, y)

    # Draw encoder block box
    rect = mpatches.FancyBboxPatch(
        (enc_x - bw/2 - 0.15, 1.15), bw + 0.3, 3.4,
        boxstyle="round,pad=0.05",
        facecolor="#E3F2FD", edgecolor=BLUE_MID, linewidth=1.5, linestyle="--", zorder=0,
    )
    ax.add_patch(rect)
    ax.text(enc_x, 4.72, "Transformer Encoder Layer (×3)",
            ha="center", fontsize=8, color=BLUE_MID, fontweight="bold")

    for i, (x, y, txt, color, bold) in enumerate(enc_layers):
        _box(ax, x, y, bw - 0.1, bh - 0.05, txt, facecolor=color, fontsize=7.5)
        if i > 0:
            _arrow(ax, enc_layers[i-1][0], enc_layers[i-1][1] - bh/2 + 0.05,
                   x, y + bh/2 - 0.05)

    _arrow(ax, main[-1][0] + bw/2, main[-1][1], enc_x - bw/2 + 0.15, enc_layers[1][1])

    for i, (x, y, txt, color, bold) in enumerate(after_enc):
        _box(ax, x, y, bw, bh, txt, facecolor=color, fontsize=8, bold=bold)
        if i == 0:
            _arrow(ax, enc_x + bw/2 - 0.15, enc_layers[-1][1], x - bw/2, y)
        else:
            _arrow(ax, after_enc[i-1][0] + bw/2, after_enc[i-1][1], x - bw/2, y)

    ax.set_title("Sơ đồ kiến trúc Transformer Encoder cho nhận dạng cử chỉ",
                 fontsize=13, fontweight="bold", color=BLUE_DARK, pad=10)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
