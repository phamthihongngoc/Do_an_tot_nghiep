"""
Huấn luyện GAFormer trên tập dữ liệu GesHome (IMU).

Cách dùng:
  cd gaformer/src
  python train.py --manifest G:/DOAN2/data_collection/data/manifest.csv ^
                  --root    G:/DOAN2/data_collection ^
                  --epochs  30 --batch-size 16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import GAFDataset, GAFDatasetConfig, build_label_list, parse_subjects
from model import build_gaformer
from utils import (
    compute_classification_metrics,
    count_parameters,
    evaluate_and_collect,
    evaluate_with_loss,
    inference_time_ms,
    plot_confusion_matrix,
    plot_tsne,
    save_json,
    seed_all,
    write_summary_xlsx,
)

# Subject splits (giống training_50hz_clean)
DEFAULT_TRAIN = [f"S{i:02d}" for i in range(1, 12)]   # S01–S11
DEFAULT_VAL   = ["S12", "S13"]
DEFAULT_TEST  = ["S14", "S15"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _esc(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _fmt_elapsed(s: float) -> str:
    if s >= 60:
        return f"{int(s // 60)}m {int(s % 60)}s"
    return f"{s:.0f}s" if s >= 1 else f"{s * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = total = correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        correct += int((torch.argmax(logits, 1) == labels).sum())
    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    label_list = build_label_list(manifest_df)
    label_to_index = {lbl: idx for idx, lbl in enumerate(label_list)}
    print(f"Nhãn ({len(label_list)}): {label_list}")

    train_subjects = parse_subjects(args.train_subjects, DEFAULT_TRAIN)
    val_subjects   = parse_subjects(args.val_subjects,   DEFAULT_VAL)
    test_subjects  = parse_subjects(args.test_subjects,  DEFAULT_TEST)

    run_id = f"gaformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)

    # --- Build datasets ---
    def make_ds(subjects: list[str]) -> GAFDataset:
        return GAFDataset(GAFDatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=subjects,
            label_to_index=label_to_index,
            img_size=args.img_size,
            smooth_window=args.smooth_window,
        ))

    print("\nNạp dữ liệu train...")
    train_ds = make_ds(train_subjects)
    print("Nạp dữ liệu val...")
    val_ds = make_ds(val_subjects)
    print("Nạp dữ liệu test...")
    test_ds = make_ds(test_subjects)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    model = build_gaformer(num_classes=len(label_list), in_channels=8, dropout=args.dropout).to(device)
    params = count_parameters(model)
    print(f"Tham số mô hình: {params:,} ({params / 1e6:.2f}M)")

    save_json(run_dir / "labels.json", {"labels": label_list})
    save_json(run_dir / "config.json", vars(args))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    best_val_acc = 0.0
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"

    with history_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "elapsed_s"])

    print(f"\nBắt đầu huấn luyện {args.epochs} epoch...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1 = evaluate_with_loss(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0

        with history_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                     f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{elapsed:.1f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | {_fmt_elapsed(elapsed)} "
            f"| loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            + (" *" if val_acc == best_val_acc else ""),
            flush=True,
        )

    # --- Test evaluation ---
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    y_true, y_pred, test_features = evaluate_and_collect(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred)
    test_acc = metrics["accuracy"]
    test_f1  = metrics["f1"]
    latency_ms = inference_time_ms(model, test_loader, device)

    print(
        f"\nTest  Accuracy={test_acc*100:.2f}%  Precision={metrics['precision']*100:.2f}%"
        f"  Recall={metrics['recall']*100:.2f}%  F1={test_f1*100:.2f}%"
        f"  latency={latency_ms:.2f}ms/sample"
    )
    save_json(run_dir / "test_metrics.json", {
        "accuracy_pct":  round(test_acc * 100, 4),
        "precision_pct": round(metrics["precision"] * 100, 4),
        "recall_pct":    round(metrics["recall"] * 100, 4),
        "f1_pct":        round(test_f1 * 100, 4),
    })

    cm_path = run_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, label_list, cm_path)
    print(f"Ma trận nhầm lẫn : {cm_path}")

    tsne_path = run_dir / "tsne.png"
    print("Đang tính t-SNE (có thể mất vài giây)...")
    plot_tsne(test_features, y_true, label_list, tsne_path)
    print(f"t-SNE             : {tsne_path}")

    # --- Summary ---
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_id", "params_M", "latency_ms", "test_acc", "test_f1"])
        w.writerow([run_id, f"{params / 1e6:.2f}", f"{latency_ms:.4f}", f"{test_acc:.4f}", f"{test_f1:.4f}"])
    write_summary_xlsx(summary_path, results_dir / "summary.xlsx")

    print(f"Kết quả: {run_dir}")
    return {"run_id": run_id, "test_acc": test_acc, "test_f1": test_f1, "params_M": params / 1e6}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Huấn luyện GAFormer")
    p.add_argument("--manifest", default=r"G:\DOAN2\data_collection\data\SHG_Dataset\manifest_shg.csv")
    p.add_argument("--root",     default=r"G:\DOAN2\data_collection\data\SHG_Dataset")
    p.add_argument("--results-dir", default=r"G:\DOAN2\gaformer\checkpoints")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--img-size",     type=int,   default=224)
    p.add_argument("--smooth-window", type=int,  default=5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--train-subjects", default=None)
    p.add_argument("--val-subjects",   default=None)
    p.add_argument("--test-subjects",  default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
