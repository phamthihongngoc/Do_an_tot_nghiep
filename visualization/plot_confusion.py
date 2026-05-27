"""Ma trận nhầm lẫn (Confusion Matrix) — tông màu xanh Navy."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from visualization.style import BLUE_DARK, WHITE, apply_theme, savefig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_ids: List[str],
    model_name: str = "Model",
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()
    n = len(label_ids)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(float) / np.maximum(row_sums, 1)
        fmt      = ".2f"
        title_sfx = "(Normalized)"
    else:
        cm_plot  = cm
        fmt      = "d"
        title_sfx = "(Count)"

    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(cm_plot, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            txt = f"{val:{fmt}}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color=WHITE if val > thresh else BLUE_DARK)

    ax.set_xticks(range(n));  ax.set_xticklabels(label_ids, rotation=90, fontsize=8)
    ax.set_yticks(range(n));  ax.set_yticklabels(label_ids, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name} {title_sfx}",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    plt.tight_layout()

    if save_path:
        savefig(fig, save_path)
    return fig


def plot_confusion_comparison(
    preds_by_model: dict,    # {name: (y_true, y_pred)}
    label_ids: List[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Subplot grid — confusion matrix của nhiều model cạnh nhau."""
    apply_theme()
    models  = list(preds_by_model.keys())
    n_mod   = len(models)
    n_cls   = len(label_ids)
    ncols   = min(n_mod, 2)
    nrows   = (n_mod + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * max(6, n_cls * 0.42),
                                       nrows * max(5, n_cls * 0.38)))
    axes = np.array(axes).flatten()

    for idx, name in enumerate(models):
        ax = axes[idx]
        y_true, y_pred = preds_by_model[name]
        n = len(label_ids)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n))).astype(float)
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)

        ax.imshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight="bold", color=BLUE_DARK)
        ax.set_xticks(range(n)); ax.set_xticklabels(label_ids, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(label_ids, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for ax in axes[n_mod:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — So sánh các mô hình",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig
