"""Vẽ sơ đồ kiến trúc CNN1D và CNN-BiLSTM."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Màu sắc ───
C_INPUT   = "#AED6F1"   # xanh nhạt
C_CONV    = "#2E86C1"   # xanh đậm
C_BN      = "#85C1E9"   # xanh mid
C_RELU    = "#F0B27A"   # cam nhạt
C_POOL    = "#58D68D"   # xanh lá
C_AVGPOOL = "#A9DFBF"   # xanh lá nhạt
C_FC      = "#C39BD3"   # tím nhạt
C_OUTPUT  = "#F1948A"   # đỏ nhạt
C_LSTM    = "#F9E79F"   # vàng nhạt
C_LSTM2   = "#F4D03F"   # vàng đậm
C_CONCAT  = "#FAD7A0"   # cam đào

TEXT_COLOR = "white"
TEXT_DARK  = "#1a1a2e"

def draw_box(ax, x, y, w, h, color, text, fontsize=9, text_color=TEXT_COLOR, radius=0.015):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.005,rounding_size={radius}",
                         facecolor=color, edgecolor="#2c3e50", linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4,
            multialignment="center")

def draw_arrow(ax, x, y1, y2, color="#2c3e50"):
    ax.annotate("", xy=(x, y2 + 0.005), xytext=(x, y1 - 0.005),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)

def side_label(ax, x, y, text, fontsize=8):
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            color="#555555", style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# SƠ ĐỒ 1 — CNN1D
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6, 13))
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")
ax1.set_facecolor("#f8f9fa")
fig1.patch.set_facecolor("#f8f9fa")

ax1.text(0.5, 0.975, "Kiến trúc CNN1D", ha="center", va="top",
         fontsize=14, fontweight="bold", color="#1a1a2e")

cx = 0.5
bw = 0.58
bh = 0.042

# Các lớp từ trên xuống
layers = [
    # (y_center, color, text, side_text)
    (0.930, C_INPUT,   "Input\n(B, T=201, C=6)",          ""),
    (0.870, "#95A5A6", "Permute → (B, C=6, T=201)",       ""),
    # Block 1
    (0.800, C_CONV,    "Conv1d(6 → 32, kernel=7, pad=3)",  "(B, 32, 201)"),
    (0.758, C_BN,      "BatchNorm1d(32)",                  ""),
    (0.716, C_RELU,    "ReLU",                             ""),
    (0.674, C_POOL,    "MaxPool1d(2)",                     "(B, 32, 100)"),
    # Block 2
    (0.610, C_CONV,    "Conv1d(32 → 64, kernel=5, pad=2)", "(B, 64, 100)"),
    (0.568, C_BN,      "BatchNorm1d(64)",                  ""),
    (0.526, C_RELU,    "ReLU",                             ""),
    (0.484, C_POOL,    "MaxPool1d(2)",                     "(B, 64, 50)"),
    # Block 3
    (0.420, C_CONV,    "Conv1d(64 → 128, kernel=3, pad=1)","(B, 128, 50)"),
    (0.378, C_BN,      "BatchNorm1d(128)",                 ""),
    (0.336, C_RELU,    "ReLU",                             ""),
    (0.294, C_POOL,    "MaxPool1d(2)",                     "(B, 128, 25)"),
    # Pooling
    (0.230, C_AVGPOOL, "AdaptiveAvgPool1d(1)",             "(B, 128, 1)"),
    (0.188, "#95A5A6", "Flatten",                          "(B, 128)"),
    # FC Head
    (0.130, C_FC,      "Linear(128 → 128)",                ""),
    (0.092, C_RELU,    "ReLU  +  Dropout(0.4)",           ""),
    (0.050, C_FC,      "Linear(128 → 20)",                 ""),
    (0.010, C_OUTPUT,  "Output logits  (B, 20)",           ""),
]

for i, (y, color, text, side) in enumerate(layers):
    tc = TEXT_DARK if color in (C_BN, C_RELU, C_POOL, C_AVGPOOL, "#95A5A6", C_LSTM, C_CONCAT) else TEXT_COLOR
    draw_box(ax1, cx, y, bw, bh, color, text, fontsize=8.5, text_color=tc)
    if side:
        side_label(ax1, cx + bw/2 + 0.02, y, side, fontsize=7.5)
    if i < len(layers) - 1:
        draw_arrow(ax1, cx, y - bh/2, layers[i+1][0] + bh/2)

# Block labels (bên trái)
for label, y_top, y_bot in [
    ("Block 1", 0.820, 0.654),
    ("Block 2", 0.630, 0.464),
    ("Block 3", 0.440, 0.274),
    ("FC Head", 0.150, 0.030),
]:
    xb = cx - bw/2 - 0.08
    yc = (y_top + y_bot) / 2
    ax1.annotate("", xy=(xb + 0.02, y_bot), xytext=(xb + 0.02, y_top),
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=1))
    ax1.text(xb, yc, label, ha="center", va="center", fontsize=8,
             color="#666", rotation=90, fontweight="bold")

plt.tight_layout()
plt.savefig("g:/Do_an/diagram_CNN1D.png", dpi=180, bbox_inches="tight",
            facecolor=fig1.get_facecolor())
print("Saved: diagram_CNN1D.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SƠ ĐỒ 2 — CNN-BiLSTM
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(6.5, 15))
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis("off")
ax2.set_facecolor("#f8f9fa")
fig2.patch.set_facecolor("#f8f9fa")

ax2.text(0.5, 0.978, "Kiến trúc CNN-BiLSTM", ha="center", va="top",
         fontsize=14, fontweight="bold", color="#1a1a2e")

cx = 0.5
bw = 0.60
bh = 0.040

layers2 = [
    (0.940, C_INPUT,   "Input  (B, T=201, C=6)",              ""),
    (0.887, "#95A5A6", "Permute → (B, C=6, T=201)",           ""),
    # CNN Block 1
    (0.828, C_CONV,    "Conv1d(6 → 32, kernel=5, pad=2)",     "(B, 32, 201)"),
    (0.788, C_BN,      "BatchNorm1d(32)",                     ""),
    (0.748, C_RELU,    "ReLU",                                ""),
    (0.708, C_POOL,    "MaxPool1d(2)",                        "(B, 32, 100)"),
    # CNN Block 2
    (0.650, C_CONV,    "Conv1d(32 → 64, kernel=5, pad=2)",    "(B, 64, 100)"),
    (0.610, C_BN,      "BatchNorm1d(64)",                     ""),
    (0.570, C_RELU,    "ReLU",                                ""),
    (0.530, C_POOL,    "MaxPool1d(2)",                        "(B, 64, 50)"),
    # Permute back
    (0.475, "#95A5A6", "Permute → (B, T'=50, C'=64)",        ""),
    # BiLSTM
    (0.415, C_LSTM2,   "Bi-LSTM Layer 1\nhidden=128, bidirectional=True",   "(B, 50, 256)"),
    (0.345, C_LSTM2,   "Bi-LSTM Layer 2\nhidden=128, bidirectional=True",   "(B, 50, 256)"),
    # Concat
    (0.278, C_CONCAT,  "Lấy hidden state cuối\nconcat(forward_h, backward_h)",  "(B, 256)"),
    # FC head
    (0.212, C_FC,      "Dropout(0.3)  →  Linear(256 → 128)", ""),
    (0.165, C_RELU,    "ReLU",                                ""),
    (0.118, C_FC,      "Dropout(0.3)  →  Linear(128 → 20)",  ""),
    (0.060, C_OUTPUT,  "Output logits  (B, 20)",              ""),
]

for i, (y, color, text, side) in enumerate(layers2):
    tc = TEXT_DARK if color in (C_BN, C_RELU, C_POOL, C_AVGPOOL,
                                "#95A5A6", C_LSTM, C_LSTM2, C_CONCAT) else TEXT_COLOR
    fs = 8.2
    h = bh if "\n" not in text else bh * 1.5
    draw_box(ax2, cx, y, bw, h, color, text, fontsize=fs, text_color=tc)
    if side:
        side_label(ax2, cx + bw/2 + 0.02, y, side, fontsize=7.5)
    if i < len(layers2) - 1:
        next_y = layers2[i+1][0]
        next_h = bh if "\n" not in layers2[i+1][2] else bh * 1.5
        draw_arrow(ax2, cx, y - h/2, next_y + next_h/2)

# Block labels
for label, y_top, y_bot in [
    ("CNN\nEncoder", 0.848, 0.488),
    ("Bi-LSTM", 0.435, 0.258),
    ("FC Head", 0.232, 0.040),
]:
    xb = cx - bw/2 - 0.09
    yc = (y_top + y_bot) / 2
    ax2.annotate("", xy=(xb + 0.025, y_bot), xytext=(xb + 0.025, y_top),
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=1))
    ax2.text(xb - 0.01, yc, label, ha="center", va="center", fontsize=8.5,
             color="#555", rotation=90, fontweight="bold")

plt.tight_layout()
plt.savefig("g:/Do_an/diagram_CNN_BiLSTM.png", dpi=180, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print("Saved: diagram_CNN_BiLSTM.png")
plt.close()
