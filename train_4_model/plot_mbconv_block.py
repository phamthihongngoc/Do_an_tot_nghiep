"""Vẽ sơ đồ chi tiết khối MBConv (Inverted Residual) trong GAFormer – chuẩn IEEE."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
C_INPUT  = "#FFFFFF"
C_EXPAND = "#D6E4F7"   # xanh  – Pointwise Expand
C_DW     = "#D5F5E3"   # lá    – Depthwise Conv
C_PROJ   = "#E8DAEF"   # tím   – Pointwise Project
C_ADD    = "#FEF9E7"   # vàng  – Residual Add
C_OUT    = "#F2F3F4"   # xám   – Output
C_SKIP   = "#FDEBD0"   # cam   – Shortcut box
BORDER   = "#2C3E50"
SKIP_CLR = "#C0392B"   # đỏ    – nét shortcut

# ── Các layer trong nhánh chính ──────────────────────────────────────────────
MAIN_LAYERS = [
    ("Đầu vào  (B, C_in, H, W)",                                C_INPUT),
    ("Conv2d 1×1:  C_in → 4·C_in\n+ BatchNorm2d  +  GELU",     C_EXPAND),
    ("Conv2d 3×3 DW:  4·C_in → 4·C_in\n+ BatchNorm2d  +  GELU  [stride = s]", C_DW),
    ("Conv2d 1×1:  4·C_in → C_out\n+ BatchNorm2d",             C_PROJ),
    ("⊕   Cộng phần dư  (Residual Add)",                        C_ADD),
    ("Đầu ra  (B, C_out, H/s, W/s)",                            C_OUT),
]

LABELS_RIGHT = [
    "",
    "Mở rộng kênh (×4)",
    "Tích chập theo chiều sâu",
    "Chiếu về kênh đầu ra",
    "",
    "",
]

# ── Layout ───────────────────────────────────────────────────────────────────
GAP  = 0.24
W    = 0.54        # chiều rộng box chính
X0   = 0.08        # lề trái
XCTR = X0 + W / 2  # tâm ngang box chính

def _bh(label: str) -> float:
    return 0.56 if "\n" in label else 0.43

heights = [_bh(lbl) for lbl, _ in MAIN_LAYERS]
fig_h   = sum(heights) + (len(MAIN_LAYERS) - 1) * GAP + 1.0

fig, ax = plt.subplots(figsize=(8.0, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)
ax.axis("off")

y_pos = []
y = fig_h - 0.5

for i, (label, color) in enumerate(MAIN_LAYERS):
    h = heights[i]
    if i > 0:
        y -= GAP
    y -= h / 2
    y_pos.append(y)

    rect = mpatches.FancyBboxPatch(
        (X0, y - h / 2), W, h,
        boxstyle="round,pad=0.012",
        linewidth=1.2, edgecolor=BORDER, facecolor=color, zorder=3,
    )
    ax.add_patch(rect)

    fs = 8.0 if "\n" in label else 8.8
    ax.text(XCTR, y, label,
            ha="center", va="center", fontsize=fs,
            fontfamily="DejaVu Sans", zorder=4)

    if LABELS_RIGHT[i]:
        ax.text(X0 + W + 0.018, y, LABELS_RIGHT[i],
                ha="left", va="center", fontsize=7.0,
                color="#555555", style="italic", zorder=4)

    y -= h / 2

# ── Mũi tên nhánh chính ──────────────────────────────────────────────────────
for i in range(len(MAIN_LAYERS) - 1):
    y_top = y_pos[i]   - heights[i]   / 2
    y_bot = y_pos[i+1] + heights[i+1] / 2
    ax.annotate("",
                xy=(XCTR, y_bot), xycoords="data",
                xytext=(XCTR, y_top), textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=BORDER,
                                lw=1.1, mutation_scale=10),
                zorder=2)

# ── Nhánh tắt (Shortcut) bên phải ────────────────────────────────────────────
xR  = X0 + W          # cạnh phải box chính = 0.62
xVL = 0.695            # đường dọc shortcut
xSL = 0.715            # lề trái box shortcut
xSR = 0.985            # lề phải box shortcut
xSC = (xSL + xSR) / 2 # tâm box shortcut

# y lấy từ tâm Input (idx 0) và Add (idx 4)
y_inp = y_pos[0]
y_add = y_pos[4]
y_mid = (y_inp + y_add) / 2

# ── Nét nằm ngang: Input → đường dọc ────
ax.annotate("",
            xy=(xVL, y_inp), xycoords="data",
            xytext=(xR, y_inp), textcoords="data",
            arrowprops=dict(arrowstyle="-", color=SKIP_CLR, lw=1.5, linestyle="dashed"),
            zorder=5)

# ── Nét đứng dọc ────
ax.plot([xVL, xVL], [y_add, y_inp], color=SKIP_CLR, lw=1.5,
        linestyle="dashed", zorder=5)

# ── Mũi tên đường ngang: đường dọc → Add ────
ax.annotate("",
            xy=(xR, y_add), xycoords="data",
            xytext=(xVL, y_add), textcoords="data",
            arrowprops=dict(arrowstyle="-|>", color=SKIP_CLR, lw=1.5,
                            linestyle="dashed", mutation_scale=10),
            zorder=5)

# ── Box shortcut ────
sh_h = 0.80
rect_skip = mpatches.FancyBboxPatch(
    (xSL, y_mid - sh_h / 2), xSR - xSL, sh_h,
    boxstyle="round,pad=0.012",
    linewidth=1.2, edgecolor=SKIP_CLR, facecolor=C_SKIP,
    linestyle="dashed", zorder=4,
)
ax.add_patch(rect_skip)

skip_txt = ("Shortcut\n"
            "Conv2d(1×1, stride=s)\n"
            "+ BatchNorm2d\n"
            "nếu C_in ≠ C_out hoặc s ≠ 1\n\n"
            "Identity\n"
            "nếu C_in = C_out và s = 1")
ax.text(xSC, y_mid, skip_txt,
        ha="center", va="center", fontsize=7.2,
        color=SKIP_CLR, fontfamily="DejaVu Sans",
        fontweight="bold", linespacing=1.4, zorder=5)

# ── Chú thích mũi tên shortcut ────
ax.text(xVL + 0.005, (y_inp + y_mid) / 2 + 0.06, "skip",
        ha="left", va="center", fontsize=7.0,
        color=SKIP_CLR, style="italic", zorder=5)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_EXPAND, edgecolor=BORDER, label="Pointwise Expand (1×1 Conv)"),
    mpatches.Patch(facecolor=C_DW,     edgecolor=BORDER, label="Depthwise Conv (3×3 DW)"),
    mpatches.Patch(facecolor=C_PROJ,   edgecolor=BORDER, label="Pointwise Project (1×1 Conv)"),
    mpatches.Patch(facecolor=C_ADD,    edgecolor=BORDER, label="Residual Add  ⊕"),
    mpatches.Patch(facecolor=C_SKIP,   edgecolor=SKIP_CLR, label="Shortcut (nhánh tắt)", linestyle="dashed"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.0,
          framealpha=0.9, bbox_to_anchor=(0.0, 0.0))

ax.set_title("Khối dư đảo ngược  MBConv\n(Inverted Residual Block trong GAFormer)",
             fontsize=10, fontweight="bold", pad=8, y=1.0, va="top")

plt.tight_layout(pad=0.3)
out = Path(__file__).parent / "mbconv_block.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
