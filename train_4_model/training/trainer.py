"""Training loop cho Deep Learning models (CNN / CNN-BiLSTM / Transformer).

Lưu kết quả vào:
    training/results_{model}_100epoch/run_{timestamp}/
        best.pt            — trọng số tốt nhất
        normalization.json — mean, std
        labels.json        — danh sách label IDs
        meta.json          — siêu dữ liệu run
        history.json       — loss/acc qua epochs
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Một epoch train/eval
# ─────────────────────────────────────────────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    """Trả về (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / max(total, 1), correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Hàm train chính
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    run_dir: Path,
    label_ids: List[str],
    norm_stats: Dict[str, np.ndarray],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Huấn luyện model và lưu kết quả vào run_dir.

    Returns:
        history dict với train_loss, val_loss, train_acc, val_acc per epoch
    """
    t_cfg      = cfg["training"]
    dev        = torch.device(device)
    model      = model.to(dev)
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion  = nn.CrossEntropyLoss(
        label_smoothing=t_cfg.get("label_smoothing", 0.05)
    )
    optimizer  = AdamW(
        model.parameters(),
        lr=float(t_cfg["learning_rate"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    max_epochs = int(t_cfg["max_epochs"])
    scheduler_name = t_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    patience    = int(t_cfg.get("patience", 15))
    best_val    = -np.inf
    no_improve  = 0
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_ckpt   = run_dir / "best.pt"

    print(f"\n▶ Training — {max_epochs} epochs | device={device}")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, dev)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, dev)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val
        if improved:
            best_val   = va_acc
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": va_acc}, best_ckpt)
        else:
            no_improve += 1

        marker = "✓" if improved else f" ({no_improve}/{patience})"
        print(f"  [{epoch:3d}/{max_epochs}] {marker} "
              f"loss={tr_loss:.4f}/{va_loss:.4f}  "
              f"acc={tr_acc:.4f}/{va_acc:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if no_improve >= patience:
            print(f"  Early stopping tại epoch {epoch} "
                  f"(không cải thiện {patience} epochs liên tiếp).")
            break

    # ── Lưu artifacts ────────────────────────────────────────────────────────
    (run_dir / "normalization.json").write_text(
        json.dumps({"mean": norm_stats["mean"].tolist(),
                    "std":  norm_stats["std"].tolist()}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "labels.json").write_text(
        json.dumps(label_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    meta = {
        "val_accuracy":  best_val,
        "epochs_trained": len(history["train_loss"]),
        "created_at":    datetime.now().isoformat(timespec="seconds"),
        "run_dir":       str(run_dir),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  Best val_acc = {best_val:.4f}  →  {best_ckpt}")
    return {**history, "best_val_acc": best_val, "run_dir": str(run_dir)}


# ─────────────────────────────────────────────────────────────────────────────
# Đánh giá trên test set
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trả về (accuracy, y_true, y_pred)."""
    dev    = torch.device(device)
    model  = model.to(dev).eval()
    y_true, y_pred = [], []
    for x, y in loader:
        preds = model(x.to(dev)).argmax(1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true == y_pred).mean(), y_true, y_pred
