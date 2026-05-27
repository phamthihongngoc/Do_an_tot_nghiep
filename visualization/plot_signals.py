"""Biểu đồ tín hiệu IMU.

Hàm:
    plot_all_gestures(data_by_label, ...)  — Lưới G1–G15, mỗi ô 1 cử chỉ
    plot_gesture_detail(seq, label, ...)   — Chi tiết 1 cử chỉ (vd G3)
    plot_sequence_lengths(manifest, ...)   — Boxplot độ dài chuỗi theo label
    plot_samples_per_subject(manifest, ...) — Bar chart số mẫu theo người
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.style import (
    BLUE_DARK, BLUE_LIGHT, CHANNEL_COLORS, ORANGE, PALETTE_20,
    GRAY, WHITE, apply_theme, savefig,
)

CHANNELS    = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
ACCEL_CH    = ["ax_g", "ay_g", "az_g"]
GYRO_CH     = ["gx_dps", "gy_dps", "gz_dps"]


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Lưới G1–G15 — mỗi ô một cử chỉ điển hình
# ─────────────────────────────────────────────────────────────────────────────
def plot_all_gestures(
    data_by_label: Dict[str, np.ndarray],  # {label_id: (T, 6)}
    label_names: Dict[str, str],           # {label_id: name}
    sample_rate: float = 50.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    gesture_ids = sorted(
        [k for k in data_by_label if k.startswith("G")],
        key=lambda x: int(x[1:]),
    )
    n = len(gesture_ids)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8),
                              sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for idx, gid in enumerate(gesture_ids):
        ax  = axes[idx]
        seq = data_by_label[gid]        # (T, 6)
        T   = seq.shape[0]
        t   = np.arange(T) / sample_rate

        for ch in ACCEL_CH:
            i = CHANNELS.index(ch)
            ax.plot(t, seq[:, i], color=CHANNEL_COLORS[ch], lw=1.0, alpha=0.85)
        ax.set_title(f"{gid}\n{label_names.get(gid, '')}", fontsize=9, color=BLUE_DARK,
                     fontweight="bold")
        ax.set_xlabel("s", fontsize=8)
        ax.set_ylabel("g", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend chung
    handles = [plt.Line2D([0], [0], color=CHANNEL_COLORS[c], lw=1.5, label=c)
               for c in ACCEL_CH]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=9,
               framealpha=0.9, title="Accelerometer")
    fig.suptitle("Tín hiệu Gia tốc kế — G1 đến G15", fontsize=14,
                 fontweight="bold", color=BLUE_DARK, y=1.01)
    plt.tight_layout()

    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Chi tiết 1 cử chỉ (6 kênh, 2 subplot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_gesture_detail(
    seq: np.ndarray,            # (T, 6)
    label_id: str = "G3",
    label_name: str = "tv_favorite_chanel",
    sample_rate: float = 50.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    T = seq.shape[0]
    t = np.arange(T) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    for ch in ACCEL_CH:
        i = CHANNELS.index(ch)
        ax1.plot(t, seq[:, i], color=CHANNEL_COLORS[ch], lw=1.6, label=ch)
    ax1.set_ylabel("Acceleration (g)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"Cử chỉ {label_id} — {label_name}", fontsize=13,
                  fontweight="bold", color=BLUE_DARK)

    for ch in GYRO_CH:
        i = CHANNELS.index(ch)
        ax2.plot(t, seq[:, i], color=CHANNEL_COLORS[ch], lw=1.6, label=ch, linestyle="--")
    ax2.set_ylabel("Angular Velocity (°/s)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2):
        ax.axhline(0, color=ORANGE, lw=0.8, ls=":")

    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Phân bố số mẫu theo người dùng
# ─────────────────────────────────────────────────────────────────────────────
def plot_samples_per_subject(
    manifest: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    counts = manifest.groupby("subject_id").size().reset_index(name="count")
    counts = counts.sort_values("subject_id")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(counts["subject_id"], counts["count"],
                  color=BLUE_DARK, edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, str(int(h)),
                ha="center", va="bottom", fontsize=8, color=BLUE_DARK)

    ax.set_xlabel("Subject ID", fontsize=11)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_title("Phân bố tổng số mẫu IMU theo người dùng", fontsize=13,
                 fontweight="bold", color=BLUE_DARK)
    ax.set_ylim(0, counts["count"].max() * 1.15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Boxplot độ dài chuỗi theo lớp cử chỉ
# ─────────────────────────────────────────────────────────────────────────────
def plot_sequence_lengths(
    manifest: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    mdf = manifest.copy()
    mdf["samples"] = pd.to_numeric(mdf["samples"], errors="coerce")
    mdf = mdf.dropna(subset=["samples"])

    gesture_order = sorted(
        mdf["label_id"].unique(),
        key=lambda x: (0 if x.startswith("G") else 1, int(x[1:]) if x[1:].isdigit() else 0),
    )
    data = [mdf[mdf["label_id"] == lid]["samples"].values for lid in gesture_order]

    fig, ax = plt.subplots(figsize=(12, 4))
    bp = ax.boxplot(data, labels=gesture_order, patch_artist=True,
                    boxprops=dict(facecolor=BLUE_LIGHT, color=BLUE_DARK),
                    medianprops=dict(color=ORANGE, lw=2),
                    whiskerprops=dict(color=BLUE_DARK),
                    flierprops=dict(marker=".", color=GRAY, ms=3))

    ax.set_xlabel("Lớp cử chỉ", fontsize=11)
    ax.set_ylabel("Số mẫu (samples)", fontsize=11)
    ax.set_title("Độ dài chuỗi tín hiệu IMU theo lớp cử chỉ", fontsize=13,
                 fontweight="bold", color=BLUE_DARK)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Số mẫu theo từng cử chỉ
# ─────────────────────────────────────────────────────────────────────────────
def plot_samples_per_gesture(
    manifest: pd.DataFrame,
    label_names: Dict[str, str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    counts = manifest.groupby("label_id").size().reset_index(name="count")
    counts["_sort"] = counts["label_id"].apply(
        lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 999
    )
    counts = counts.sort_values("_sort")

    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(range(len(counts)), counts["count"],
                  color=BLUE_DARK, edgecolor="white", linewidth=0.5)
    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(row["count"])), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts["label_id"], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Cử chỉ", fontsize=11)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_title("Số lượng mẫu theo từng cử chỉ trong tập dữ liệu",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.set_ylim(0, counts["count"].max() * 1.18)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Số mẫu theo tập train / val / test
# ─────────────────────────────────────────────────────────────────────────────
def plot_samples_per_split(
    splits: Dict[str, pd.DataFrame],
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    split_names  = ["train", "val", "test"]
    split_colors = [BLUE_DARK, ORANGE, "#00897B"]
    split_labels = ["Train", "Validation", "Test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    totals = [len(splits.get(s, [])) for s in split_names]
    bars = ax1.bar(split_labels, totals, color=split_colors, edgecolor="white")
    for bar, total in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(total), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số mẫu", fontsize=11)
    ax1.set_title("Tổng số mẫu theo tập", fontsize=12, fontweight="bold", color=BLUE_DARK)
    ax1.set_ylim(0, max(totals) * 1.2)

    all_lids = sorted(
        set().union(*[set(splits[s]["label_id"]) for s in split_names if s in splits]),
        key=lambda x: (0 if x.startswith("G") else 1, int(x[1:]) if x[1:].isdigit() else 999),
    )
    x  = np.arange(len(all_lids))
    bw = 0.25
    for i, (s, color, lbl) in enumerate(zip(split_names, split_colors, split_labels)):
        if s not in splits:
            continue
        cnt  = splits[s].groupby("label_id").size()
        vals = [int(cnt.get(lid, 0)) for lid in all_lids]
        ax2.bar(x + i * bw, vals, bw, label=lbl, color=color, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x + bw)
    ax2.set_xticklabels(all_lids, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Số mẫu", fontsize=11)
    ax2.set_title("Phân bố theo cử chỉ và tập", fontsize=12, fontweight="bold", color=BLUE_DARK)
    ax2.legend(fontsize=9)

    fig.suptitle("Số lượng mẫu theo tập Train / Validation / Test",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Tổng điểm IMU theo người thực hiện
# ─────────────────────────────────────────────────────────────────────────────
def plot_imu_points_per_subject(
    manifest: pd.DataFrame,
    target_len: int = 201,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    counts = manifest.groupby("subject_id").size().reset_index(name="n_trials")
    counts["imu_points"] = counts["n_trials"] * target_len
    counts = counts.sort_values("subject_id")

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(counts["subject_id"], counts["imu_points"],
                  color=BLUE_DARK, edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50,
                f"{int(h):,}", ha="center", va="bottom", fontsize=8, color=BLUE_DARK)
    ax.set_xlabel("Subject ID", fontsize=11)
    ax.set_ylabel("Tổng số điểm IMU", fontsize=11)
    ax.set_title("Tổng số điểm lấy mẫu IMU theo từng người thực hiện",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, counts["imu_points"].max() * 1.18)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: So sánh tín hiệu gia tốc / con quay của 2 cử chỉ cạnh nhau
# ─────────────────────────────────────────────────────────────────────────────
def plot_gesture_comparison(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    label_a: str = "G3",
    label_b: str = "G4",
    channel_group: str = "accel",
    sample_rate: float = 50.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    chs      = ACCEL_CH if channel_group == "accel" else GYRO_CH
    unit     = "g" if channel_group == "accel" else "°/s"
    title_ch = ("Gia tốc kế (Accelerometer)"
                if channel_group == "accel" else "Con quay hồi chuyển (Gyroscope)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for ax, seq, label in zip(axes, [seq_a, seq_b], [label_a, label_b]):
        T = seq.shape[0]
        t = np.arange(T) / sample_rate
        for ch in chs:
            i = CHANNELS.index(ch)
            ax.plot(t, seq[:, i], color=CHANNEL_COLORS[ch], lw=1.6, label=ch)
        ax.axhline(0, color=GRAY, lw=0.6, ls=":")
        ax.set_title(f"Cử chỉ {label}", fontsize=12, fontweight="bold", color=BLUE_DARK)
        ax.set_xlabel("Thời gian (s)", fontsize=10)
        ax.set_ylabel(unit, fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle(f"So sánh tín hiệu {title_ch} — {label_a} vs {label_b}",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
