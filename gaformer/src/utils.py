"""Các hàm tiện ích dùng chung: metrics, lưu file, seed."""
from __future__ import annotations

import csv
import json
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, f1


def evaluate_with_loss(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            total += labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if total == 0:
        return 0.0, 0.0, 0.0
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return total_loss / total, float((y_true == y_pred).mean()), float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def inference_time_ms(model: torch.nn.Module, loader, device: torch.device, warmup: int = 3) -> float:
    model.eval()
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for step, (images, _) in enumerate(loader):
            images = images.to(device)
            if step < warmup:
                _ = model(images)
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_samples += images.size(0)
    return (total_time / max(total_samples, 1)) * 1000.0


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_summary_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    def col_name(index: int) -> str:
        name = ""
        index += 1
        while index:
            index, remainder = divmod(index - 1, 26)
            name = chr(65 + remainder) + name
        return name

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = f"{col_name(col_idx)}{row_idx}"
            text = escape(str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


# ---------------------------------------------------------------------------
# Full evaluation: collect predictions + features in one pass
# ---------------------------------------------------------------------------

def evaluate_and_collect(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (y_true, y_pred, features) từ một lần chạy qua loader."""
    model.eval()
    all_preds, all_labels, all_feats = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            try:
                logits, feats = model(images, return_features=True)
            except TypeError:
                logits = model(images)
                # Dùng tạm logits
                feats = logits
                
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_feats),
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Tính Accuracy, Precision, Recall, F1 (macro average). Trả về tỉ lệ [0, 1]."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    save_path: Path,
) -> None:
    """Vẽ ma trận nhầm lẫn dạng heatmap và lưu file ảnh."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(label_names)
    fig_size = max(8, n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Dự đoán", fontsize=12)
    ax.set_ylabel("Thực tế", fontsize=12)
    ax.set_title("Ma trận nhầm lẫn", fontsize=14)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def plot_tsne(
    features: np.ndarray,
    y_true: np.ndarray,
    label_names: list[str],
    save_path: Path,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Giảm chiều đặc trưng bằng t-SNE và lưu biểu đồ phân tán."""
    n_samples = features.shape[0]
    effective_perp = float(min(perplexity, max(5.0, n_samples - 1)))
    tsne = TSNE(n_components=2, perplexity=effective_perp, random_state=random_state, n_iter=1000)
    emb = tsne.fit_transform(features)

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(label_names):
        mask = y_true == cls_idx
        if mask.any():
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=cls_name,
                color=colors[cls_idx % len(colors)],
                s=25, alpha=0.75,
            )
    ax.legend(loc="best", fontsize=8, markerscale=1.5, title="Cử chỉ")
    ax.set_title("t-SNE – Phân bố đặc trưng học được", fontsize=14)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
