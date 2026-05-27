"""Bảng thống kê dạng hình (matplotlib table) cho báo cáo đồ án."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.style import BLUE_DARK, WHITE, GRAY_LITE, ORANGE, apply_theme, savefig


def _render_table(ax, rows, cols, title, col_widths=None):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(BLUE_DARK)
            cell.set_text_props(color=WHITE, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(GRAY_LITE)
        else:
            cell.set_facecolor(WHITE)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=BLUE_DARK, pad=14)


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: Chia tập dữ liệu theo subject
# ─────────────────────────────────────────────────────────────────────────────
def table_dataset_split(
    splits: Dict[str, pd.DataFrame],
    test_subjects: List[str],
    val_subjects: List[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    total = sum(len(v) for v in splits.values())

    train_subs = sorted(set(splits.get("train", pd.DataFrame()).get("subject_id", [])))
    rows = [
        ["Train",      ", ".join(train_subs), len(splits.get("train", [])),
         f"{len(splits.get('train', [])) / total * 100:.1f}%"],
        ["Validation", ", ".join(val_subjects), len(splits.get("val", [])),
         f"{len(splits.get('val', [])) / total * 100:.1f}%"],
        ["Test",       ", ".join(test_subjects), len(splits.get("test", [])),
         f"{len(splits.get('test', [])) / total * 100:.1f}%"],
        ["Tổng",       "—", total, "100%"],
    ]
    cols = ["Tập", "Subjects", "Số mẫu", "Tỉ lệ"]

    fig, ax = plt.subplots(figsize=(10, 2.2))
    _render_table(ax, rows, cols, "Bảng chia tập dữ liệu theo train / validation / test")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: Số mẫu và điểm IMU theo tập
# ─────────────────────────────────────────────────────────────────────────────
def table_samples_imu_per_split(
    splits: Dict[str, pd.DataFrame],
    target_len: int = 201,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    rows = []
    total_samples = sum(len(v) for v in splits.values())
    total_imu     = total_samples * target_len

    for split_name, label in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        df = splits.get(split_name, pd.DataFrame())
        n  = len(df)
        rows.append([label, str(n), f"{n * target_len:,}", f"{n / total_samples * 100:.1f}%"])
    rows.append(["Tổng", str(total_samples), f"{total_imu:,}", "100%"])

    cols = ["Tập", "Số mẫu (trial)", "Tổng điểm IMU", "Tỉ lệ"]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    _render_table(ax, rows, cols, "Bảng số lượng mẫu và điểm IMU theo tập dữ liệu")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: Cấu trúc dữ liệu cảm biến trong tệp CSV
# ─────────────────────────────────────────────────────────────────────────────
def table_csv_structure(
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    rows = [
        ["ax_g",   "float32", "g (m/s²÷9.81)", "Gia tốc trục X"],
        ["ay_g",   "float32", "g",              "Gia tốc trục Y"],
        ["az_g",   "float32", "g",              "Gia tốc trục Z"],
        ["gx_dps", "float32", "°/s",            "Vận tốc góc trục X"],
        ["gy_dps", "float32", "°/s",            "Vận tốc góc trục Y"],
        ["gz_dps", "float32", "°/s",            "Vận tốc góc trục Z"],
    ]
    cols = ["Tên cột", "Kiểu dữ liệu", "Đơn vị", "Mô tả"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    _render_table(ax, rows, cols, "Cấu trúc dữ liệu cảm biến trong tệp CSV")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: Thống kê theo từng người thực hiện
# ─────────────────────────────────────────────────────────────────────────────
def table_subjects_stats(
    manifest: pd.DataFrame,
    target_len: int = 201,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    grp  = manifest.groupby("subject_id")
    rows = []
    for sub, df in sorted(grp, key=lambda x: x[0]):
        n_ges   = df["label_id"].nunique()
        n_trial = len(df)
        imu_pts = n_trial * target_len
        rows.append([sub, str(n_ges), str(n_trial), f"{imu_pts:,}"])

    total_trials = len(manifest)
    rows.append(["Tổng",
                 str(manifest["label_id"].nunique()),
                 str(total_trials),
                 f"{total_trials * target_len:,}"])

    cols = ["Subject", "Số cử chỉ", "Số trial", "Tổng điểm IMU"]
    n    = len(rows)
    fig, ax = plt.subplots(figsize=(9, 0.5 + n * 0.42))
    _render_table(ax, rows, cols, "Bảng số điểm IMU theo người thực hiện")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: Thống kê nhanh đặc trưng tín hiệu G3 vs G4
# ─────────────────────────────────────────────────────────────────────────────
def table_feature_stats(
    seq_a: np.ndarray,   # (T, 6) — G3
    seq_b: np.ndarray,   # (T, 6) — G4
    label_a: str = "G3",
    label_b: str = "G4",
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
    rows = []
    for i, ch in enumerate(CHANNELS):
        a, b = seq_a[:, i], seq_b[:, i]
        rows.append([
            ch,
            f"{a.mean():.3f}", f"{a.std():.3f}", f"{a.min():.3f}", f"{a.max():.3f}",
            f"{b.mean():.3f}", f"{b.std():.3f}", f"{b.min():.3f}", f"{b.max():.3f}",
        ])

    cols = ["Kênh",
            f"{label_a} Mean", f"{label_a} Std", f"{label_a} Min", f"{label_a} Max",
            f"{label_b} Mean", f"{label_b} Std", f"{label_b} Min", f"{label_b} Max"]

    fig, ax = plt.subplots(figsize=(16, 3.2))
    _render_table(ax, rows, cols,
                  f"Thống kê nhanh đặc trưng tín hiệu mẫu {label_a} và {label_b}")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: So sánh mô hình theo Accuracy, F1, độ phức tạp
# ─────────────────────────────────────────────────────────────────────────────
def table_model_comparison(
    results: List[Dict[str, Any]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    results: list of dict với keys:
        model, top1_acc, macro_f1, params_m, infer_ms
    """
    apply_theme()
    rows = []
    for r in results:
        rows.append([
            r.get("model", ""),
            f"{r.get('val_acc',   0):.4f}",
            f"{r.get('top1_acc',  0):.4f}",
            f"{r.get('macro_f1',  r.get('top1_acc', 0)):.4f}",
            f"{r.get('params_m',  0):.3f}",
            f"{r.get('infer_ms',  0):.2f}",
        ])

    cols = ["Mô hình", "Val Acc", "Test Acc", "Macro F1", "Params (M)", "Infer (ms)"]
    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.55 * len(rows)))
    _render_table(ax, rows, cols,
                  "Bảng so sánh mô hình theo độ chính xác, F1 và độ phức tạp")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
