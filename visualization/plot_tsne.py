"""t-SNE visualization cho không gian đặc trưng 15 cử chỉ."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from visualization.style import PALETTE_20, BLUE_DARK, WHITE, apply_theme, savefig


def plot_tsne(
    features: np.ndarray,   # (N, D) — đặc trưng thô (trước classifier)
    labels: np.ndarray,     # (N,) integer labels
    label_ids: List[str],
    model_name: str = "Model",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    apply_theme()

    print(f"[t-SNE] Fitting ({features.shape[0]} samples, perplexity={perplexity})...")
    X = StandardScaler().fit_transform(features)
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)

    unique_labels = np.unique(labels)
    colors = [PALETTE_20[i % len(PALETTE_20)] for i in range(len(label_ids))]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lid in enumerate(unique_labels):
        mask = labels == lid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[int(lid)]], s=18, alpha=0.65, edgecolors="none",
                   label=label_ids[int(lid)] if int(lid) < len(label_ids) else str(lid))

    # Legend ngoài figure
    handles = [mpatches.Patch(facecolor=colors[int(l)],
                               label=label_ids[int(l)] if int(l) < len(label_ids) else str(l))
               for l in unique_labels]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.9, title="Gesture", title_fontsize=10)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title(f"t-SNE — Không gian đặc trưng — {model_name}",
                 fontsize=13, fontweight="bold", color=BLUE_DARK)
    ax.set_facecolor(WHITE)
    plt.tight_layout()

    if save_path:
        savefig(fig, save_path)
    return fig


def extract_dl_features(
    model,               # nn.Module đã load
    loader,              # DataLoader
    device: str = "cpu",
    layer_hook: str = "penultimate",  # Lấy feature trước lớp FC cuối
) -> tuple[np.ndarray, np.ndarray]:
    """Trích xuất đặc trưng từ penultimate layer của DL model.

    Trả về (features, labels) arrays.
    """
    import torch
    import torch.nn as nn

    features_list, labels_list = [], []
    activations: dict = {}

    def hook_fn(module, inp, out):
        activations["feat"] = out.detach().cpu()

    # Gắn hook vào layer kế cuối (trước Linear cuối)
    target_layer = None
    children = list(model.children())
    if hasattr(model, "head"):
        head_layers = list(model.head.children())
        for lyr in reversed(head_layers):
            if isinstance(lyr, nn.Linear):
                break
            if isinstance(lyr, (nn.ReLU, nn.GELU, nn.Dropout)):
                target_layer = lyr
                break
    if target_layer is None and len(children) > 0:
        target_layer = children[-2] if len(children) >= 2 else children[-1]

    handle = None
    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    with torch.no_grad():
        for x, y in loader:
            model(x.to(dev))
            feat = activations.get("feat")
            if feat is not None:
                features_list.append(feat.numpy())
            labels_list.extend(y.numpy())

    if handle:
        handle.remove()

    features = np.concatenate(features_list, axis=0) if features_list else np.array([])
    return features, np.array(labels_list)
