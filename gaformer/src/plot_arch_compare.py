"""
Tạo hai hình so sánh kiến trúc và hiệu năng:

  arch_compare.png  – So sánh kiến trúc CoAtNet-light vs CoAtNet-0 (Bảng 3.6)
  perf_compare.png  – So sánh số tham số và thời gian suy luận (Bảng 3.7)

Số tham số được tính trực tiếp từ mô hình; latency cần truyền vào từ
kết quả comparison_results.json (nếu có) hoặc dùng giá trị thủ công.

Usage:
  python plot_arch_compare.py
  python plot_arch_compare.py --results G:/DOAN2/gaformer/checkpoints/comparison/comparison_results.json
  python plot_arch_compare.py --num-classes 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import build_coatnet0, build_gaformer
from utils import count_parameters


# ---------------------------------------------------------------------------
# Dữ liệu kiến trúc (cố định theo luận án)
# ---------------------------------------------------------------------------

ARCH_DATA = [
    # (giai_đoạn, loại,        CoAtNet-0  ,  CoAtNet-light, thay_đổi)
    ("S0-Conv",    "Conv stem", "L=2, D=64",  "L=2, D=64",   False),
    ("S1-MBConv",  "MBConv",   "L=2, D=96",  "L=2, D=96",   False),
    ("S2-MBConv",  "MBConv",   "L=3, D=192", "L=2, D=192",  True),
    ("S3-TFM",     "TFM",      "L=5, D=384", "L=3, D=384",  True),
    ("S4-TFM",     "TFM",      "L=2, D=768", "L=2, D=768",  False),
]


def plot_architecture_table(out_path: Path) -> None:
    """Vẽ bảng so sánh kiến trúc (phong cách Bảng 3.6)."""
    col_labels = ["Giai đoạn", "CoAtNet-0\n(gốc)", "CoAtNet-light\n(đề xuất)"]
    cell_data  = [[row[0], row[2], row[3]] for row in ARCH_DATA]

    # Màu hàng: vàng nhạt nếu có thay đổi, trắng nếu không
    row_colors = []
    for row in ARCH_DATA:
        changed = row[4]
        row_colors.append(
            ["#fff9c4", "#fff9c4", "#fff9c4"] if changed
            else ["white", "white", "white"]
        )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    # Đậm header
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#1a73e8")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # Đậm ô CoAtNet-light đã thay đổi
    for row_i, row in enumerate(ARCH_DATA, start=1):
        if row[4]:
            tbl[row_i, 2].set_text_props(fontweight="bold")

    legend = [
        mpatches.Patch(facecolor="#fff9c4", edgecolor="gray", label="Giảm số block so với CoAtNet-0"),
        mpatches.Patch(facecolor="white",   edgecolor="gray", label="Giữ nguyên"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=9, frameon=True)

    ax.set_title("So sánh kiến trúc CoAtNet-light và CoAtNet-0",
                 fontsize=12, fontweight="bold", pad=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")


# ---------------------------------------------------------------------------
# Bảng so sánh tham số và latency (Bảng 3.7)
# ---------------------------------------------------------------------------

def plot_perf_table(num_classes: int, out_path: Path, results_json: Path | None = None) -> None:
    """Vẽ bảng + biểu đồ so sánh tham số và latency."""

    # Tính tham số từ mô hình thực
    model_c0   = build_coatnet0(num_classes)
    model_gafo = build_gaformer(num_classes)
    params_c0   = count_parameters(model_c0)   / 1e6
    params_gafo = count_parameters(model_gafo) / 1e6

    # Latency: lấy từ kết quả thực nếu có, còn không dùng placeholder
    lat_c0 = lat_gafo = None
    if results_json and Path(results_json).exists():
        data = json.loads(Path(results_json).read_text(encoding="utf-8"))
        for r in data:
            if r["method"] == "gadf_coatnet0":
                lat_c0 = r["latency_ms"]
            elif r["method"] == "gaformer":
                lat_gafo = r["latency_ms"]

    labels    = ["GADF + CoAtNet-0", "GADF + CoAtNet-light\n(GAFormer)"]
    params    = [params_c0, params_gafo]
    latencies = [lat_c0 if lat_c0 else float("nan"),
                 lat_gafo if lat_gafo else float("nan")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("So sánh số tham số và thời gian suy luận", fontsize=12, fontweight="bold")

    x = np.arange(len(labels))
    colors = ["#e74c3c", "#2ecc71"]

    # Subplot (a): số tham số
    bars = axes[0].bar(x, params, color=colors, alpha=0.85, width=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Triệu tham số (M)", fontsize=10)
    axes[0].set_title("(a) Số lượng tham số (M)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot (b): latency
    if any(not np.isnan(v) for v in latencies):
        bars2 = axes[1].bar(x, latencies, color=colors, alpha=0.85, width=0.4)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
        axes[1].set_ylabel("ms / mẫu", fontsize=10)
        axes[1].set_title("(b) Thời gian nhận dạng (ms/mẫu)", fontsize=11)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars2, latencies):
            if not np.isnan(val):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                             f"{val:.2f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Chưa có dữ liệu latency\n(chạy compare_train.py trước)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=10,
                     color="gray", style="italic")
        axes[1].set_title("(b) Thời gian nhận dạng (ms/mẫu)", fontsize=11)
        axes[1].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")

    # In bảng ra terminal
    print(f"\n{'Phương pháp':<35} {'Tham số (M)':>12} {'Latency (ms)':>14}")
    print("-" * 64)
    for lbl, p, l in zip(["GADF + CoAtNet-0", "GADF + CoAtNet-light (GAFormer)"], params, latencies):
        lat_str = f"{l:.2f}" if not np.isnan(l) else "–"
        print(f"{lbl:<35} {p:>12.2f} {lat_str:>14}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def plot_perf_all(results_json: Path, out_path: Path) -> None:
    """Vẽ biểu đồ params + latency cho tất cả 4 model từ comparison_results.json."""
    data = json.loads(Path(results_json).read_text(encoding="utf-8"))

    # Thứ tự hiển thị cố định
    order = ["gadf_coatnet0", "gasf_coatnet0", "gadf_resnet50", "gaformer"]
    data_map = {r["method"]: r for r in data}
    rows = [data_map[m] for m in order if m in data_map]

    labels    = [r["label"].replace(" (GAFormer)", "\n(GAFormer)") for r in rows]
    params    = [r["params_M"]   for r in rows]
    latencies = [r["latency_ms"] for r in rows]

    # Tô màu GAFormer khác để nổi bật
    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#2ecc71"]

    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("So sánh số tham số và thời gian suy luận – 4 phương pháp",
                 fontsize=13, fontweight="bold")

    # --- (a) Params ---
    bars = axes[0].bar(x, params, color=colors, alpha=0.88, width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Số tham số (M)", fontsize=11)
    axes[0].set_title("(a) Số lượng tham số (M)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- (b) Latency ---
    bars2 = axes[1].bar(x, latencies, color=colors, alpha=0.88, width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("ms / mẫu", fontsize=11)
    axes[1].set_title("(b) Thời gian nhận dạng (ms/mẫu)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, latencies):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(facecolor="#95a5a6", label="Baseline"),
        mpatches.Patch(facecolor="#2ecc71", label="GAFormer (đề xuất)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")

    # In bảng ra terminal
    print(f"\n{'Phương pháp':<38} {'Tham số (M)':>12} {'Latency (ms)':>14}")
    print("-" * 67)
    for r in rows:
        print(f"{r['label']:<38} {r['params_M']:>12.2f} {r['latency_ms']:>14.2f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vẽ bảng so sánh kiến trúc và hiệu năng")
    p.add_argument("--out-dir",     default=r"G:\Do_an\gaformer_updated\results")
    p.add_argument("--num-classes", type=int, default=20,
                   help="Số lớp (dùng để tính tham số mô hình)")
    p.add_argument("--results",
                   default=r"G:\Do_an\gaformer_updated\checkpoints\comparison\comparison_results.json",
                   help="Đường dẫn comparison_results.json")
    return p


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    plot_architecture_table(out_dir / "arch_compare.png")
    plot_perf_table(
        num_classes=args.num_classes,
        out_path=out_dir / "perf_compare_2model.png",
        results_json=args.results,
    )
    if args.results and Path(args.results).exists():
        plot_perf_all(
            results_json=Path(args.results),
            out_path=out_dir / "perf_compare.png",
        )
