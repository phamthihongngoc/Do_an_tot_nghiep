"""Biểu đồ và bảng đánh giá hiệu năng mô hình.

Hàm:
    plot_loss_acc_curves(history, ...)    — Loss/Acc qua epochs (train vs val)
    plot_f1_per_class(y_true, y_pred, ...)— Grouped bar F1 các model vs lớp
    table_model_performance(results, ...) — Bảng so sánh Top-1/Top-5/Params
    table_architecture(models_info, ...)  — Bảng kiến trúc
    table_param_analysis(records, ...)    — Bảng phân tích tham số
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from visualization.style import (
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, ORANGE, ORANGE_LITE,
    GRAY, PALETTE_20, WHITE, apply_theme, savefig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Loss & Accuracy curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_acc_curves(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], color=BLUE_DARK,  lw=2.0, label="Train")
    ax1.plot(epochs, history["val_loss"],   color=ORANGE,     lw=2.0, label="Validation", ls="--")
    ax1.set_title("Loss", fontsize=12, fontweight="bold", color=BLUE_DARK)
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], color=BLUE_DARK, lw=2.0, label="Train")
    ax2.plot(epochs, history["val_acc"],   color=ORANGE,    lw=2.0, label="Validation", ls="--")
    best_val = max(history["val_acc"])
    best_ep  = history["val_acc"].index(best_val) + 1
    ax2.axvline(best_ep, color=ORANGE, lw=1.2, ls=":", alpha=0.8)
    ax2.annotate(f"Best={best_val:.3f}\n@ep{best_ep}",
                 xy=(best_ep, best_val), xytext=(best_ep + 1, best_val - 0.05),
                 fontsize=8, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax2.set_title("Accuracy", fontsize=12, fontweight="bold", color=BLUE_DARK)
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05);    ax2.legend()

    fig.suptitle(f"Training History — {model_name}", fontsize=13,
                 fontweight="bold", color=BLUE_DARK)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: F1-score per class — so sánh các model
# ─────────────────────────────────────────────────────────────────────────────
def plot_f1_per_class(
    preds_by_model: Dict[str, tuple],   # {model_name: (y_true, y_pred)}
    label_ids: List[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    models   = list(preds_by_model.keys())
    n_labels = len(label_ids)
    x        = np.arange(n_labels)
    bar_w    = 0.8 / len(models)
    colors   = [BLUE_DARK, ORANGE, "#00897B", "#7B1FA2"][:len(models)]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, (y_true, y_pred)) in enumerate(preds_by_model.items()):
        f1s = f1_score(y_true, y_pred, labels=list(range(n_labels)),
                       average=None, zero_division=0)
        f1s = f1s[:n_labels]
        offset = (i - len(models) / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w * 0.9, label=name,
               color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(label_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("So sánh F1-score theo từng lớp cử chỉ trên tập Test",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bảng: So sánh hiệu năng các model
# ─────────────────────────────────────────────────────────────────────────────
def table_model_performance(
    results: List[Dict[str, Any]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    results: list of dicts với keys:
        model, top1_acc, top5_acc, params_m, gflops, infer_ms, val_acc
    """
    apply_theme()
    cols = ["Model", "Val Acc", "Top-1 Acc", "Top-5 Acc",
            "Params (M)", "GFLOPs", "Infer (ms)"]
    rows = []
    for r in results:
        rows.append([
            r.get("model", ""),
            f"{r.get('val_acc', 0):.4f}",
            f"{r.get('top1_acc', 0):.4f}",
            f"{r.get('top5_acc', 0):.4f}",
            f"{r.get('params_m', 0):.3f}",
            f"{r.get('gflops', 0):.3f}",
            f"{r.get('infer_ms', 0):.2f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(BLUE_DARK)
            cell.set_text_props(color=WHITE, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
        else:
            cell.set_facecolor(WHITE)

    ax.set_title("Đánh giá hiệu năng các mô hình", fontsize=13,
                 fontweight="bold", color=BLUE_DARK, pad=20)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: So sánh Accuracy và Macro F1 theo mô hình (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_acc_f1_by_model(
    preds_by_model: Dict[str, tuple],
    save_path: str | Path | None = None,
) -> plt.Figure:
    from sklearn.metrics import accuracy_score, f1_score
    apply_theme()
    models = list(preds_by_model.keys())
    accs   = [accuracy_score(yt, yp) for yt, yp in preds_by_model.values()]
    f1s    = [f1_score(yt, yp, average="macro", zero_division=0)
              for yt, yp in preds_by_model.values()]

    x  = np.arange(len(models))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
    ax.bar(x - bw / 2, accs, bw, label="Accuracy", color=BLUE_DARK, alpha=0.9, edgecolor="white")
    ax.bar(x + bw / 2, f1s,  bw, label="Macro F1", color=ORANGE,    alpha=0.9, edgecolor="white")
    for i, (acc, f1) in enumerate(zip(accs, f1s)):
        ax.text(i - bw / 2, acc + 0.005, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + bw / 2, f1  + 0.005, f"{f1:.3f}",  ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("So sánh Accuracy và Macro F1 theo mô hình",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.legend(fontsize=10)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Accuracy theo mô hình — lần chạy tốt nhất
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_per_model(
    preds_by_model: Dict[str, tuple],
    save_path: str | Path | None = None,
) -> plt.Figure:
    from sklearn.metrics import accuracy_score
    apply_theme()
    models = list(preds_by_model.keys())
    accs   = [accuracy_score(yt, yp) for yt, yp in preds_by_model.values()]
    colors = [PALETTE_20[i % len(PALETTE_20)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
    bars = ax.bar(models, accs, color=colors, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy theo mô hình ở lần chạy tốt nhất",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: Macro F1 theo mô hình — lần chạy tốt nhất
# ─────────────────────────────────────────────────────────────────────────────
def plot_f1_per_model(
    preds_by_model: Dict[str, tuple],
    save_path: str | Path | None = None,
) -> plt.Figure:
    from sklearn.metrics import f1_score
    apply_theme()
    models = list(preds_by_model.keys())
    f1s    = [f1_score(yt, yp, average="macro", zero_division=0)
              for yt, yp in preds_by_model.values()]
    colors = [PALETTE_20[i % len(PALETTE_20)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
    bars = ax.bar(models, f1s, color=colors, edgecolor="white")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Macro F1-score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Macro F1 theo mô hình ở lần chạy tốt nhất",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hình: So sánh Accuracy nhận dạng theo từng cử chỉ
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_per_class(
    preds_by_model: Dict[str, tuple],
    label_ids: List[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    models   = list(preds_by_model.keys())
    n_labels = len(label_ids)
    x        = np.arange(n_labels)
    bar_w    = 0.8 / len(models)
    colors   = [BLUE_DARK, ORANGE, "#00897B", "#7B1FA2"][:len(models)]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, (y_true, y_pred)) in enumerate(preds_by_model.items()):
        per_class_acc = np.array([
            (y_true[y_true == cls] == y_pred[y_true == cls]).mean()
            if (y_true == cls).sum() > 0 else 0.0
            for cls in range(n_labels)
        ])
        offset = (i - len(models) / 2 + 0.5) * bar_w
        ax.bar(x + offset, per_class_acc, bar_w * 0.9,
               label=name, color=colors[i], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(label_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("So sánh Accuracy nhận dạng theo từng cử chỉ/hoạt động",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
