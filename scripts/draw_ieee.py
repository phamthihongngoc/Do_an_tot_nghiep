"""IEEE-style architecture diagrams — CNN1D & CNN-BiLSTM."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    "font.family":  "DejaVu Serif",
    "font.size":    9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ── Palette ──────────────────────────────────────────────────────────────────
WHITE    = "#FFFFFF"
GRAY_LT  = "#F5F5F5"
GRAY_MD  = "#CCCCCC"
GRAY_DK  = "#888888"
BLACK    = "#111111"

C_IN   = "#EAF4FB"   # input
C_CONV = "#D6EAF8"   # Conv1D
C_BN   = "#E8F8F5"   # BN
C_ACT  = "#FEF9E7"   # ReLU
C_POOL = "#F4ECF7"   # Pool
C_FC   = "#FDEDEC"   # Linear
C_LSTM = "#FEFDE7"   # LSTM
C_CAT  = "#FEF5E7"   # concat
C_OUT  = GRAY_LT     # output
EDGE   = "#555555"


def draw_box(ax, xc, yc, w, h, fc, label, sub=None, fs=8.2):
    patch = FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor=fc, edgecolor=EDGE, linewidth=0.85, zorder=3,
    )
    ax.add_patch(patch)
    label_y = yc + (0.012 if sub else 0)
    ax.text(xc, label_y, label,
            ha="center", va="center", fontsize=fs, color=BLACK,
            fontfamily="DejaVu Serif", zorder=4, multialignment="center")
    if sub:
        ax.text(xc, yc - 0.020, sub,
                ha="center", va="center", fontsize=6.4,
                color=GRAY_DK, fontstyle="italic",
                fontfamily="DejaVu Serif", zorder=4)


def draw_arrow(ax, xc, y_top, y_bot):
    ax.annotate("", xy=(xc, y_bot + 0.004), xytext=(xc, y_top - 0.004),
                arrowprops=dict(arrowstyle="-|>", color=BLACK,
                                lw=0.85, mutation_scale=9), zorder=2)


def draw_brace(ax, xc, w, y_top, y_bot, label, fs=7.2):
    """Left-side bracket + rotated label."""
    xb = xc - w / 2 - 0.035
    ax.plot([xb, xb], [y_bot, y_top], color=GRAY_MD, lw=0.9)
    ax.plot([xb, xb + 0.014], [y_top, y_top], color=GRAY_MD, lw=0.9)
    ax.plot([xb, xb + 0.014], [y_bot, y_bot], color=GRAY_MD, lw=0.9)
    ax.text(xb - 0.022, (y_top + y_bot) / 2, label,
            ha="center", va="center", fontsize=fs, color=BLACK,
            rotation=90, fontweight="bold", fontfamily="DejaVu Serif")


def draw_separator(ax, xc, w, y, label=""):
    ax.plot([xc - w / 2 - 0.01, xc + w / 2 + 0.01],
            [y, y], color=GRAY_MD, lw=0.55, ls="--", zorder=1)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CNN1D
# ═════════════════════════════════════════════════════════════════════════════
#
# Layout: liệt kê (y_raw, fc, label, sub) từ trên xuống dưới
# y_raw là khoảng cách tương đối — sẽ được remap vào [0.04, 0.94]
#
RAW1 = [
    # y   fc      label                              sub
    (19, C_IN,   "Input Signal",                  "B × 201 × 6"),
    # block 1
    (17, C_CONV, "Conv1D   k=7,  C: 6 → 32",      "B × 32 × 201"),
    (16, C_BN,   "Batch Normalization",            ""),
    (15, C_ACT,  "ReLU",                           ""),
    (14, C_POOL, "MaxPool1D   (stride = 2)",       "B × 32 × 100"),
    # block 2
    (12, C_CONV, "Conv1D   k=5,  C: 32 → 64",     "B × 64 × 100"),
    (11, C_BN,   "Batch Normalization",            ""),
    (10, C_ACT,  "ReLU",                           ""),
    ( 9, C_POOL, "MaxPool1D   (stride = 2)",       "B × 64 × 50"),
    # block 3
    ( 7, C_CONV, "Conv1D   k=3,  C: 64 → 128",    "B × 128 × 50"),
    ( 6, C_BN,   "Batch Normalization",            ""),
    ( 5, C_ACT,  "ReLU",                           ""),
    ( 4, C_POOL, "MaxPool1D   (stride = 2)",       "B × 128 × 25"),
    # global pool + flatten
    ( 2, C_POOL, "Adaptive Avg Pool1D  →  Flatten","B × 128"),
    # fc head
    ( 0.8, C_FC, "Linear (128 → 128)  +  ReLU  +  Dropout (p=0.4)", ""),
    ( 0.0, C_FC, "Linear (128 → 20)",              ""),
    (-0.9, C_OUT,"Output Logits",                  "B × 20"),
]

H1  = 0.048
GAP = 0.006
CX1 = 0.50
W1  = 0.70

fig1, ax1 = plt.subplots(figsize=(3.6, 10.5))
fig1.patch.set_facecolor(WHITE)
ax1.set_facecolor(WHITE)
ax1.set_xlim(0, 1);  ax1.set_ylim(0, 1);  ax1.axis("off")

# ─ remap y ─
raw_vals = [r[0] for r in RAW1]
lo, hi   = min(raw_vals), max(raw_vals)
def remap1(v): return 0.040 + (v - lo) / (hi - lo) * (0.920 - 0.040)

ax1.text(CX1, 0.980, "1D Convolutional Neural Network (CNN1D)",
         ha="center", va="top", fontsize=10, fontweight="bold",
         color=BLACK, fontfamily="DejaVu Serif")

mapped1 = [(remap1(r[0]), r[1], r[2], r[3]) for r in RAW1]

for i, (y, fc, lbl, sub) in enumerate(mapped1):
    draw_box(ax1, CX1, y, W1, H1, fc, lbl, sub or None)
    if i < len(mapped1) - 1:
        draw_arrow(ax1, CX1, y - H1/2, mapped1[i+1][0] + H1/2)

# separators between blocks
for sep_raw in [13, 8, 3, 1.4]:
    draw_separator(ax1, CX1, W1, remap1(sep_raw))

# braces
draw_brace(ax1, CX1, W1, remap1(17.4), remap1(13.6), "Block 1")
draw_brace(ax1, CX1, W1, remap1(12.4), remap1(8.6),  "Block 2")
draw_brace(ax1, CX1, W1, remap1(7.4),  remap1(3.6),  "Block 3")
draw_brace(ax1, CX1, W1, remap1(1.2),  remap1(-1.1), "FC Head")

plt.tight_layout(pad=0.4)
plt.savefig("g:/Do_an/ieee_CNN1D.png", dpi=300, bbox_inches="tight",
            facecolor=WHITE)
print("Saved: ieee_CNN1D.png")
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CNN-BiLSTM
# ═════════════════════════════════════════════════════════════════════════════
RAW2 = [
    # y    fc      label                                        sub
    (22,  C_IN,   "Input Signal",                             "B × 201 × 6"),
    # CNN block 1
    (20,  C_CONV, "Conv1D   k=5,  C: 6 → 32",               "B × 32 × 201"),
    (19,  C_BN,   "Batch Normalization",                      ""),
    (18,  C_ACT,  "ReLU",                                     ""),
    (17,  C_POOL, "MaxPool1D   (stride = 2)",                 "B × 32 × 100"),
    # CNN block 2
    (15,  C_CONV, "Conv1D   k=5,  C: 32 → 64",              "B × 64 × 100"),
    (14,  C_BN,   "Batch Normalization",                      ""),
    (13,  C_ACT,  "ReLU",                                     ""),
    (12,  C_POOL, "MaxPool1D   (stride = 2)",                 "B × 64 × 50"),
    # reshape
    (10.5, GRAY_LT,"Reshape  →  Time-series Sequence",        "B × 50 × 64"),
    # BiLSTM
    (8.8, C_LSTM, "Bidirectional LSTM  —  Layer 1\nhidden = 128,  bidirectional = True",
                                                              "B × 50 × 256"),
    (6.8, C_LSTM, "Bidirectional LSTM  —  Layer 2\nhidden = 128,  bidirectional = True",
                                                              "B × 50 × 256"),
    # concat
    (5.0, C_CAT,  "Last Hidden State\nconcat ( h_fwd , h_bwd )",
                                                              "B × 256"),
    # FC
    (3.2, C_FC,   "Dropout (p=0.3)  →  Linear (256 → 128)  →  ReLU", ""),
    (2.0, C_FC,   "Dropout (p=0.3)  →  Linear (128 → 20)",  ""),
    (0.5, C_OUT,  "Output Logits",                           "B × 20"),
]

H2  = 0.048
HH2 = 0.068   # tall rows (2-line text)
CX2 = 0.52
W2  = 0.68

fig2, ax2 = plt.subplots(figsize=(4.0, 12.5))
fig2.patch.set_facecolor(WHITE)
ax2.set_facecolor(WHITE)
ax2.set_xlim(0, 1);  ax2.set_ylim(0, 1);  ax2.axis("off")

raw2_vals = [r[0] for r in RAW2]
lo2, hi2  = min(raw2_vals), max(raw2_vals)
def remap2(v): return 0.038 + (v - lo2) / (hi2 - lo2) * (0.920 - 0.038)

ax2.text(CX2, 0.985, "CNN with Bidirectional LSTM (CNN-BiLSTM)",
         ha="center", va="top", fontsize=10, fontweight="bold",
         color=BLACK, fontfamily="DejaVu Serif")

def row_h(lbl): return HH2 if "\n" in lbl else H2

mapped2 = [(remap2(r[0]), r[1], r[2], r[3], row_h(r[2])) for r in RAW2]

for i, (y, fc, lbl, sub, h) in enumerate(mapped2):
    draw_box(ax2, CX2, y, W2, h, fc, lbl, sub or None)
    if i < len(mapped2) - 1:
        ny, _, _, _, nh = mapped2[i + 1]
        draw_arrow(ax2, CX2, y - h/2, ny + nh/2)

# separators
for sep_raw in [16, 11.4, 9.8, 4.0, 2.8]:
    draw_separator(ax2, CX2, W2, remap2(sep_raw))

# braces
draw_brace(ax2, CX2, W2, remap2(20.4), remap2(11.6), "CNN\nEncoder")
draw_brace(ax2, CX2, W2, remap2(9.4),  remap2(4.2),  "Bi-LSTM")
draw_brace(ax2, CX2, W2, remap2(3.8),  remap2(0.0),  "FC Head")

plt.tight_layout(pad=0.4)
plt.savefig("g:/Do_an/ieee_CNN_BiLSTM.png", dpi=300, bbox_inches="tight",
            facecolor=WHITE)
print("Saved: ieee_CNN_BiLSTM.png")
plt.close()
