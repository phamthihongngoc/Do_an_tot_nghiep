"""
Vẽ tín hiệu IMU trước và sau bộ lọc làm mịn (sliding-window mean).

Layout:
  Hàng (a) – Tín hiệu gốc   : gia tốc (ax, ay, az) | vận tốc góc (gx, gy, gz)
  Hàng (b) – Tín hiệu đã lọc: gia tốc đã lọc       | vận tốc góc đã lọc

Usage:
  python plot_smoothing.py
  python plot_smoothing.py --csv path/to/file.csv --out out.png --window 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import smooth_signal

COLS       = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
ACC_LABELS = ["ax", "ay", "az"]
GYR_LABELS = ["gx", "gy", "gz"]
ACC_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]   # đỏ / xanh lá / xanh dương
GYR_COLORS = ["#e67e22", "#9b59b6", "#1abc9c"]   # cam / tím / ngọc


def _find_sample_csv(manifest_path: Path, root_dir: Path) -> Path:
    df = pd.read_csv(manifest_path)
    return root_dir / str(df["path"].iloc[0])


def _draw_group(ax, t, data, labels, colors, title, ylabel):
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax.plot(t, data[:, i], color=col, label=lbl, linewidth=1.1)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=8)


def plot_before_after(csv_path: Path, out_path: Path, window: int = 5) -> None:
    raw = pd.read_csv(csv_path, usecols=COLS)[COLS].to_numpy(dtype=np.float32)
    filtered = smooth_signal(raw, window=window)
    t = np.arange(len(raw))

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"So sánh tín hiệu IMU trước và sau bộ lọc làm mịn (cửa sổ w = {window})",
        fontsize=13, fontweight="bold",
    )

    # Hàng (a) – gốc
    _draw_group(axes[0, 0], t, raw[:, :3], ACC_LABELS, ACC_COLORS,
                "(a)  Gia tốc – Tín hiệu gốc", "Biên độ (g)")
    _draw_group(axes[0, 1], t, raw[:, 3:], GYR_LABELS, GYR_COLORS,
                "(a)  Vận tốc góc – Tín hiệu gốc", "Biên độ (°/s)")

    # Hàng (b) – đã lọc
    _draw_group(axes[1, 0], t, filtered[:, :3], ACC_LABELS, ACC_COLORS,
                "(b)  Gia tốc – Tín hiệu đã lọc", "Biên độ (g)")
    _draw_group(axes[1, 1], t, filtered[:, 3:], GYR_LABELS, GYR_COLORS,
                "(b)  Vận tốc góc – Tín hiệu đã lọc", "Biên độ (°/s)")

    for ax in axes[1]:
        ax.set_xlabel("Mẫu (số thứ tự)", fontsize=9)

    # Nhãn (a) / (b) bên trái
    for row, label in zip(axes, ["(a)", "(b)"]):
        row[0].annotate(
            label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.12, 0.5), textcoords="axes fraction",
            fontsize=13, fontweight="bold", va="center", ha="right",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vẽ tín hiệu IMU trước/sau làm mịn")
    p.add_argument("--csv",      default=None,
                   help="Đường dẫn file CSV mẫu (tự tìm từ manifest nếu bỏ trống)")
    p.add_argument("--manifest", default=r"G:\DOAN2\data_collection\data\SHG_Dataset\manifest_shg.csv")
    p.add_argument("--root",     default=r"G:\DOAN2\data_collection\data\SHG_Dataset")
    p.add_argument("--out",      default=r"G:\DOAN2\gaformer\results\smoothing_demo.png")
    p.add_argument("--window",   type=int, default=5)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _find_sample_csv(Path(args.manifest), Path(args.root))
    print(f"Dùng file mẫu: {csv_path}")
    plot_before_after(csv_path, Path(args.out), window=args.window)
