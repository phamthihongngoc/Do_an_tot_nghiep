"""
So sánh 5 phương pháp nhận dạng cử chỉ trên tập GesHome.

Phương pháp:
  1. cnn_bilstm      – 1D-CNN + Bi-LSTM
  2. gasf_coatnet0   – GASF + CoAtNet-0
  3. gadf_resnet50   – GADF + ResNet-50
  4. gadf_coatnet0   – GADF + CoAtNet-0
  5. gaformer        – GADF + CoAtNet-light (GAFormer)

Usage:
  # Chạy TẤT CẢ phương pháp
  python compare_train.py --method all

  # Chạy từng phương pháp
  python compare_train.py --method gaformer
  python compare_train.py --method cnn_bilstm

  # Chỉ vẽ bảng so sánh từ kết quả đã lưu
  python compare_train.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from baselines import build_cnn_bilstm, build_resnet50_8ch
from dataset import GAFDataset, GAFDatasetConfig, RawDataset, build_label_list, parse_subjects
from model import build_coatnet0, build_gaformer
from utils import compute_classification_metrics, count_parameters, evaluate_and_collect, seed_all

DEFAULT_TRAIN = [f"S{i:02d}" for i in range(1, 12)]
DEFAULT_VAL   = ["S12", "S13"]
DEFAULT_TEST  = ["S14", "S15"]

SEQ_LEN = 150   # độ dài cố định cho RawDataset (ms ở 50 Hz ≈ 3 giây)


# ---------------------------------------------------------------------------
# Cấu hình từng phương pháp
# ---------------------------------------------------------------------------

def get_method_cfg(method: str, num_classes: int, dropout: float) -> dict:
    cfgs = {
        "cnn_bilstm": {
            "label":        "1D-CNN-biLSTM",
            "model_fn":     lambda: build_cnn_bilstm(num_classes, dropout=dropout),
            "dataset_type": "raw",
        },
        "gasf_coatnet0": {
            "label":        "GASF + CoAtNet-0",
            "model_fn":     lambda: build_coatnet0(num_classes, dropout=dropout),
            "dataset_type": "gasf",
        },
        "gadf_resnet50": {
            "label":        "GADF + ResNet50",
            "model_fn":     lambda: build_resnet50_8ch(num_classes, dropout=dropout),
            "dataset_type": "gadf",
        },
        "gadf_coatnet0": {
            "label":        "GADF + CoAtNet-0",
            "model_fn":     lambda: build_coatnet0(num_classes, dropout=dropout),
            "dataset_type": "gadf",
        },
        "gaformer": {
            "label":        "GADF + CoAtNet-light (GAFormer)",
            "model_fn":     lambda: build_gaformer(num_classes, dropout=dropout),
            "dataset_type": "gadf",
        },
    }
    if method not in cfgs:
        raise ValueError(f"Phương pháp không hợp lệ: {method}. Chọn: {list(cfgs)}")
    return cfgs[method]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaf_ds(cfg_args, subjects, manifest_path, root_dir, transform, img_size, smooth_window, label_to_index):
    return GAFDataset(GAFDatasetConfig(
        manifest_path=manifest_path,
        root_dir=root_dir,
        subjects=subjects,
        label_to_index=label_to_index,
        img_size=img_size,
        smooth_window=smooth_window,
        transform=transform,
    ))


def _make_raw_ds(subjects, manifest_path, root_dir, smooth_window, label_to_index):
    return RawDataset(GAFDatasetConfig(
        manifest_path=manifest_path,
        root_dir=root_dir,
        subjects=subjects,
        label_to_index=label_to_index,
        smooth_window=smooth_window,
    ), seq_len=SEQ_LEN)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = total = correct = 0
    for batch, labels in loader:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
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


def eval_with_loss(model, loader, criterion, device):
    model.eval()
    total_loss = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            total += labels.size(0)
            all_preds.append(torch.argmax(logits, 1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    from sklearn.metrics import f1_score
    return total_loss / max(total, 1), float((y_true == y_pred).mean()), float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Huấn luyện một phương pháp
# ---------------------------------------------------------------------------

def run_method(method: str, args: argparse.Namespace, label_list: list[str], out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    num_classes = len(label_list)
    label_to_index = {lbl: idx for idx, lbl in enumerate(label_list)}
    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)

    mcfg = get_method_cfg(method, num_classes, args.dropout)
    print(f"\n{'='*60}")
    print(f"Phương pháp: {mcfg['label']}")
    print(f"{'='*60}")

    train_subjects = parse_subjects(args.train_subjects, DEFAULT_TRAIN)
    val_subjects   = parse_subjects(args.val_subjects,   DEFAULT_VAL)
    test_subjects  = parse_subjects(args.test_subjects,  DEFAULT_TEST)

    ds_type = mcfg["dataset_type"]
    if ds_type == "raw":
        train_ds = _make_raw_ds(train_subjects, manifest_path, root_dir, args.smooth_window, label_to_index)
        val_ds   = _make_raw_ds(val_subjects,   manifest_path, root_dir, args.smooth_window, label_to_index)
        test_ds  = _make_raw_ds(test_subjects,  manifest_path, root_dir, args.smooth_window, label_to_index)
    else:
        _ds = lambda subs: _make_gaf_ds(args, subs, manifest_path, root_dir, ds_type, args.img_size, args.smooth_window, label_to_index)
        train_ds, val_ds, test_ds = _ds(train_subjects), _ds(val_subjects), _ds(test_subjects)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = mcfg["model_fn"]().to(device)
    params_M = count_parameters(model) / 1e6
    print(f"Tham số: {params_M:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = out_dir / f"{method}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1 = eval_with_loss(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": val_acc,
                         "train_loss": tr_loss, "val_loss": val_loss, "val_f1": val_f1})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        print(f"  Ep {epoch:3d}/{args.epochs} | {elapsed:.0f}s"
              f" | tr_acc={tr_acc:.4f} val_acc={val_acc:.4f}{' *' if val_acc == best_val_acc else ''}")

    # Lưu lịch sử
    hist_path = out_dir / f"{method}_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    # Đánh giá trên tập test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    # CNN-biLSTM không có return_features nên dùng evaluate đơn giản
    if ds_type == "raw":
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch, labels in test_loader:
                logits = model(batch.to(device))
                all_preds.append(torch.argmax(logits, 1).cpu().numpy())
                all_labels.append(labels.numpy())
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        metrics = compute_classification_metrics(y_true, y_pred)
    else:
        y_true, y_pred, _ = evaluate_and_collect(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred)

    # Đo latency
    model.eval()
    warmup, total_t, total_n = 3, 0.0, 0
    with torch.no_grad():
        for step, (batch, _) in enumerate(test_loader):
            batch = batch.to(device)
            if step < warmup:
                model(batch); continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_t += time.perf_counter() - t0
            total_n += batch.size(0)
    latency_ms = (total_t / max(total_n, 1)) * 1000.0

    result = {
        "method":       method,
        "label":        mcfg["label"],
        "params_M":     round(params_M, 2),
        "latency_ms":   round(latency_ms, 2),
        "accuracy_pct": round(metrics["accuracy"]  * 100, 2),
        "precision_pct":round(metrics["precision"] * 100, 2),
        "recall_pct":   round(metrics["recall"]    * 100, 2),
        "f1_pct":       round(metrics["f1"]        * 100, 2),
    }
    print(f"\n  Test  Acc={result['accuracy_pct']}%  Prec={result['precision_pct']}%"
          f"  Rec={result['recall_pct']}%  F1={result['f1_pct']}%  lat={result['latency_ms']}ms")
    return result, history


# ---------------------------------------------------------------------------
# Vẽ đồ thị quá khớp (CoAtNet-0 vs CoAtNet-light)
# ---------------------------------------------------------------------------

def plot_overfitting(out_dir: Path) -> None:
    """So sánh đường cong train/val acc của gadf_coatnet0 và gaformer."""
    pairs = [
        ("gadf_coatnet0", "CoAtNet-0 (gốc)", "#e74c3c"),
        ("gaformer",      "CoAtNet-light (GAFormer)", "#2ecc71"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("So sánh hiện tượng quá khớp: CoAtNet-0 và CoAtNet-light", fontsize=13, fontweight="bold")

    for method, label, color in pairs:
        hist_path = out_dir / f"{method}_history.csv"
        if not hist_path.exists():
            print(f"  Chưa có lịch sử cho {method}, bỏ qua.")
            continue
        df = pd.read_csv(hist_path)
        epochs = df["epoch"].values
        for ax_i, (col, line_label) in enumerate([("train_acc", "Train"), ("val_acc", "Val")]):
            ls = "-" if line_label == "Train" else "--"
            axes[0].plot(epochs, df[col].values, color=color, linestyle=ls,
                         label=f"{label} – {line_label}", linewidth=1.5)

        axes[1].plot(epochs, df["train_acc"].values - df["val_acc"].values,
                     color=color, label=f"{label}", linewidth=1.5)

    axes[0].set_title("(a) Độ chính xác train / val", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8); axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title("(b) Khoảng cách train – val (chỉ số quá khớp)", fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("train_acc − val_acc")
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[1].legend(fontsize=8); axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = out_dir / "overfitting_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# Vẽ bảng so sánh kết quả (Bảng 3.10)
# ---------------------------------------------------------------------------

def plot_results_table(results: list[dict], out_dir: Path) -> None:
    if not results:
        return
    labels    = [r["label"] for r in results]
    acc       = [r["accuracy_pct"]  for r in results]
    precision = [r["precision_pct"] for r in results]
    recall    = [r["recall_pct"]    for r in results]
    f1        = [r["f1_pct"]        for r in results]

    x = np.arange(len(labels))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 6))
    ax.bar(x - 1.5*width, acc,       width, label="Accuracy",  color=colors[0], alpha=0.85)
    ax.bar(x - 0.5*width, precision, width, label="Precision", color=colors[1], alpha=0.85)
    ax.bar(x + 0.5*width, recall,    width, label="Recall",    color=colors[2], alpha=0.85)
    ax.bar(x + 1.5*width, f1,        width, label="F1-score",  color=colors[3], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("(%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("So sánh hiệu suất các phương pháp trên tập GesHome", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Thêm giá trị trên mỗi cột
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    plt.tight_layout()
    save_path = out_dir / "method_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_METHODS = ["cnn_bilstm", "gasf_coatnet0", "gadf_resnet50", "gadf_coatnet0", "gaformer"]


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "comparison_results.json"

    # Nếu chỉ vẽ từ kết quả đã lưu
    if args.plot_only:
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            plot_overfitting(out_dir)
            plot_results_table(results, out_dir)
        else:
            print(f"Chưa có kết quả tại {results_path}")
        return

    manifest_df = pd.read_csv(Path(args.manifest))
    label_list = build_label_list(manifest_df)
    print(f"Nhãn ({len(label_list)}): {label_list}")

    methods_to_run = ALL_METHODS if args.method == "all" else [args.method]

    # Nạp kết quả cũ nếu có
    existing = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text(encoding="utf-8")):
            existing[r["method"]] = r

    all_results = list(existing.values())

    for method in methods_to_run:
        if method in existing and not args.force:
            print(f"\nBỏ qua {method} (đã có kết quả). Dùng --force để chạy lại.")
            continue
        result, _ = run_method(method, args, label_list, out_dir)
        # Cập nhật hoặc thêm mới
        existing[method] = result
        all_results = list(existing.values())
        results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Vẽ đồ thị sau khi có đủ kết quả
    plot_overfitting(out_dir)
    # Sắp xếp theo thứ tự chuẩn
    ordered = [existing[m] for m in ALL_METHODS if m in existing]
    plot_results_table(ordered, out_dir)
    print(f"\nKết quả lưu tại: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="So sánh các phương pháp nhận dạng cử chỉ")
    p.add_argument("--method",   default="all",
                   help="Phương pháp cần chạy: all | cnn_bilstm | gasf_coatnet0 | gadf_resnet50 | gadf_coatnet0 | gaformer")
    p.add_argument("--plot-only", action="store_true", help="Chỉ vẽ từ kết quả đã lưu")
    p.add_argument("--force",    action="store_true",  help="Chạy lại dù đã có kết quả")
    p.add_argument("--manifest", default=r"G:\DOAN2\data_collection\data\SHG_Dataset\manifest_shg.csv")
    p.add_argument("--root",     default=r"G:\DOAN2\data_collection\data\SHG_Dataset")
    p.add_argument("--out-dir",  default=r"G:\DOAN2\gaformer\checkpoints\comparison")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--img-size",    type=int,   default=224)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--train-subjects", default=None)
    p.add_argument("--val-subjects",   default=None)
    p.add_argument("--test-subjects",  default=None)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
