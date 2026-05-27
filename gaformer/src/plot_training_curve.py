"""
Vẽ đường cong huấn luyện (loss + accuracy) từ history.csv.

Usage:
  python plot_training_curve.py
  python plot_training_curve.py --run-dir G:/Do_an/gaformer_updated/checkpoints/gaformer_20260525_113523
  python plot_training_curve.py --run-dir G:/Do_an/gaformer_updated/checkpoints/gaformer_20260525_113523 --out training_curve.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(history_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Đường cong huấn luyện GAFormer", fontsize=14, fontweight="bold")

    # --- (a) Loss ---
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2, marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#3498db", linewidth=2, marker="s", markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("(a) Loss theo epoch", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, df["epoch"].max())

    # --- (b) Accuracy ---
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc", color="#e74c3c", linewidth=2, marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_acc"]   * 100, label="Val Acc",   color="#3498db", linewidth=2, marker="s", markersize=3)
    ax.plot(df["epoch"], df["val_f1"]    * 100, label="Val F1",    color="#2ecc71", linewidth=2, linestyle="--", marker="^", markersize=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("(%)", fontsize=12)
    ax.set_title("(b) Accuracy & F1 theo epoch", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, df["epoch"].max())
    ax.set_ylim(0, 105)

    # Ghi chú epoch tốt nhất (val_acc cao nhất)
    best_epoch = df.loc[df["val_acc"].idxmax(), "epoch"]
    best_val   = df["val_acc"].max() * 100
    axes[1].axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.5)
    axes[1].annotate(
        f"Best: epoch {int(best_epoch)}\n({best_val:.1f}%)",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 1, best_val - 10),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")

    # In tóm tắt
    best_row = df.loc[df["val_acc"].idxmax()]
    print(f"\nEpoch tốt nhất: {int(best_row['epoch'])}")
    print(f"  Val Accuracy : {best_row['val_acc']*100:.2f}%")
    print(f"  Val F1       : {best_row['val_f1']*100:.2f}%")
    print(f"  Train Loss   : {best_row['train_loss']:.4f}")
    print(f"  Val Loss     : {best_row['val_loss']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vẽ đường cong huấn luyện")
    p.add_argument(
        "--run-dir",
        default=r"G:\Do_an\gaformer_updated\checkpoints\gaformer_20260525_113523",
    )
    p.add_argument("--out", default=None, help="Đường dẫn ảnh đầu ra (mặc định: <run-dir>/training_curve.png)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    history_csv = run_dir / "history.csv"
    out_path = Path(args.out) if args.out else run_dir / "training_curve.png"

    if not history_csv.exists():
        print(f"Không tìm thấy: {history_csv}")
        raise SystemExit(1)

    plot_training_curve(history_csv, out_path)
