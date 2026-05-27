"""Vẽ sơ đồ kiến trúc GAFormer đề xuất (CoAtNet-light) chuẩn IEEE – dọc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
C_INPUT  = "#FFFFFF"   # trắng    – Đầu vào
C_PREP   = "#D1ECF1"   # cyan     – Tiền xử lý
C_GADF   = "#FFE0B2"   # cam nhạt – GADF
C_STEM   = "#CFD8DC"   # xám xanh – Conv Stem
C_MBCONV = "#D6E4F7"   # xanh     – MBConv
C_TRANS  = "#E8DAEF"   # tím nhạt – Transformer
C_POOL   = "#D5F5E3"   # xanh lá  – Pooling
C_FC     = "#FEF9E7"   # vàng     – FC
C_DROP   = "#FDEBD0"   # cam      – Dropout
C_OUT    = "#F2F3F4"   # xám      – Output
BORDER   = "#2C3E50"

# ── Layers ───────────────────────────────────────────────────────────────────
#  (label, color, note)
LAYERS = [
    ("Đầu vào IMU  (B, T, 6)",
     C_INPUT, "B mẫu × T bước × 6 kênh"),

    ("Smooth  +  Min-Max Normalize  (6 kênh)",
     C_PREP, ""),

    ("Tính magnitude  →  ghép kênh  (B, T, 8)",
     C_PREP, ""),

    ("GADF Transform  →  (B, 8, 224, 224)",
     C_GADF, "Ảnh 2D từ tín hiệu 1D"),

    ("S0 Stem — Conv2d(8→64, k=3, s=2) + BN + GELU\nConv2d(64→64) + BN + GELU  →  (B, 64, 112, 112)",
     C_STEM, ""),

    ("S1 — MBConv × 2\n64 → 96,  stride=2  →  (B, 96, 56, 56)",
     C_MBCONV, ""),

    ("S2 — MBConv × 2\n96 → 192,  stride=2  →  (B, 192, 28, 28)",
     C_MBCONV, ""),

    ("S3 — TransformerBlock × 3\n192 → 384,  stride=2,  6 heads  →  (B, 384, 14, 14)",
     C_TRANS, ""),

    ("S4 — TransformerBlock × 2\n384 → 768,  stride=2,  8 heads  →  (B, 768, 7, 7)",
     C_TRANS, ""),

    ("Global Average Pooling  →  (B, 768)",
     C_POOL, ""),

    ("Dropout(0.1)",
     C_DROP, ""),

    ("Linear  768 → 20",
     C_FC, ""),

    ("Đầu ra  (B, 20)  logits",
     C_OUT, "Softmax → nhãn cử chỉ"),
]

# ── Nhóm brace bên trái ──────────────────────────────────────────────────────
GROUPS = [
    (1,  3,  "Tiền\nxử lý"),
    (5,  6,  "MBConv\nBlocks"),
    (7,  8,  "Trans-\nformer"),
    (9,  11, "Classifier\nHead"),
]

# ── Layout ───────────────────────────────────────────────────────────────────
GAP = 0.18
W   = 0.65
X0  = (1 - W) / 2

def _bh(label: str) -> float:
    n = label.count("\n")
    return 0.44 if n == 0 else 0.58

heights = [_bh(lbl) for lbl, _, _ in LAYERS]
fig_h = sum(heights) + (len(LAYERS) - 1) * GAP + 1.1

fig, ax = plt.subplots(figsize=(7.2, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)
ax.axis("off")

y_positions = []
y = fig_h - 0.5

for i, (label, color, note) in enumerate(LAYERS):
    bh = heights[i]
    if i > 0:
        y -= GAP
    y -= bh / 2
    y_positions.append(y)

    rect = mpatches.FancyBboxPatch(
        (X0, y - bh / 2), W, bh,
        boxstyle="round,pad=0.012",
        linewidth=1.2, edgecolor=BORDER, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)

    n_lines = label.count("\n")
    fontsize = 7.5 if n_lines >= 1 else 8.5
    ax.text(0.5, y, label,
            ha="center", va="center", fontsize=fontsize,
            fontfamily="DejaVu Sans", zorder=4)

    group_notes = {g[2] for g in GROUPS}
    if note and note not in group_notes:
        ax.text(X0 + W + 0.025, y, note,
                ha="left", va="center", fontsize=7.0,
                color="#555555", style="italic", zorder=4)

    y -= bh / 2

# ── Mũi tên ──────────────────────────────────────────────────────────────────
for i in range(len(LAYERS) - 1):
    y_top    = y_positions[i]   - heights[i]   / 2
    y_bottom = y_positions[i+1] + heights[i+1] / 2
    ax.annotate("",
                xy=(0.5, y_bottom), xycoords="data",
                xytext=(0.5, y_top), textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=BORDER,
                                lw=1.1, mutation_scale=10),
                zorder=2)

# ── Brace + label bên trái ───────────────────────────────────────────────────
for (s, e, grp_label) in GROUPS:
    y_top_g = y_positions[s] + heights[s] / 2 + GAP * 0.4
    y_bot_g = y_positions[e] - heights[e] / 2 - GAP * 0.4
    y_mid   = (y_top_g + y_bot_g) / 2
    xb      = X0 - 0.065
    bw      = 0.018

    ax.plot([xb, xb - bw], [y_top_g, y_top_g], color="#7F8C8D", lw=1.2, zorder=5)
    ax.plot([xb, xb - bw], [y_bot_g, y_bot_g], color="#7F8C8D", lw=1.2, zorder=5)
    ax.plot([xb - bw, xb - bw], [y_top_g, y_bot_g], color="#7F8C8D", lw=1.2, zorder=5)

    ax.text(xb - bw - 0.01, y_mid, grp_label,
            ha="right", va="center", fontsize=7.5,
            color="#2C3E50", fontweight="bold", rotation=90)

# ── Brace bên phải: GAFormer Backbone (S0–S4) ────────────────────────────────
bb_s, bb_e = 4, 8
y_top_bb = y_positions[bb_s] + heights[bb_s] / 2 + GAP * 0.4
y_bot_bb = y_positions[bb_e] - heights[bb_e] / 2 - GAP * 0.4
y_mid_bb = (y_top_bb + y_bot_bb) / 2
xr  = X0 + W + 0.01
brw = 0.018

ax.plot([xr, xr + brw], [y_top_bb, y_top_bb], color="#117A65", lw=1.2, zorder=5)
ax.plot([xr, xr + brw], [y_bot_bb, y_bot_bb], color="#117A65", lw=1.2, zorder=5)
ax.plot([xr + brw, xr + brw], [y_top_bb, y_bot_bb], color="#117A65", lw=1.2, zorder=5)
ax.text(xr + brw + 0.01, y_mid_bb, "GAFormer\nBackbone\n(CoAtNet-\nlight)",
        ha="left", va="center", fontsize=7.2,
        color="#117A65", fontweight="bold")

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_PREP,   edgecolor=BORDER, label="Tiền xử lý tín hiệu"),
    mpatches.Patch(facecolor=C_GADF,   edgecolor=BORDER, label="GADF Transform"),
    mpatches.Patch(facecolor=C_STEM,   edgecolor=BORDER, label="Conv Stem (S0)"),
    mpatches.Patch(facecolor=C_MBCONV, edgecolor=BORDER, label="MBConv Block (S1-S2)"),
    mpatches.Patch(facecolor=C_TRANS,  edgecolor=BORDER, label="Transformer Block (S3-S4)"),
    mpatches.Patch(facecolor=C_POOL,   edgecolor=BORDER, label="Global Avg Pooling"),
    mpatches.Patch(facecolor=C_FC,     edgecolor=BORDER, label="Fully Connected"),
    mpatches.Patch(facecolor=C_DROP,   edgecolor=BORDER, label="Dropout"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=6.5,
          framealpha=0.9, bbox_to_anchor=(1.0, 0.0), ncol=2)

ax.set_title("Kiến trúc mô hình GAFormer đề xuất\ncho bài toán nhận dạng hoạt động từ cảm biến IMU",
             fontsize=10, fontweight="bold", pad=8, y=1.0, va="top")

plt.tight_layout(pad=0.3)
out = Path(__file__).parent / "gaformer_architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
