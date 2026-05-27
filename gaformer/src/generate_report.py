"""
Sinh báo cáo phân loại chi tiết từng class từ model đã train.

Chạy trên Colab (nơi có data) sau khi upload script này:

  python generate_report.py \
    --run-dir  /content/drive/MyDrive/DOAN2/gaformer/checkpoints/gaformer_20260525_113523 \
    --manifest /content/drive/MyDrive/DOAN2/SHG_Dataset/manifest_shg.csv \
    --root     /content/drive/MyDrive/DOAN2/SHG_Dataset

Hoặc local nếu có data:
  python generate_report.py --run-dir G:/Do_an/gaformer_updated/checkpoints/gaformer_20260525_113523

Output:
  <run-dir>/per_class_report.csv       – bảng Precision/Recall/F1 từng class
  <run-dir>/per_class_report.json      – cùng dữ liệu dạng JSON
  <run-dir>/per_class_bar.png          – biểu đồ bar chart F1 từng class
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GAFDataset, GAFDatasetConfig, build_label_list, parse_subjects
from model import build_gaformer
from utils import seed_all

DEFAULT_TEST = ["S14", "S15"]


def generate_report(args: argparse.Namespace) -> None:
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(args.run_dir)
    manifest_path = Path(args.manifest)
    root_dir = Path(args.root)

    # Load labels & config
    labels_json = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))
    label_list = labels_json["labels"]
    label_to_index = {lbl: i for i, lbl in enumerate(label_list)}
    num_classes = len(label_list)
    print(f"Classes ({num_classes}): {label_list}")

    config_json = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    img_size     = config_json.get("img_size", 224)
    smooth_window = config_json.get("smooth_window", 5)
    batch_size   = config_json.get("batch_size", 16)

    test_subjects = parse_subjects(args.test_subjects, DEFAULT_TEST)

    # Build dataset
    test_ds = GAFDataset(GAFDatasetConfig(
        manifest_path=manifest_path,
        root_dir=root_dir,
        subjects=test_subjects,
        label_to_index=label_to_index,
        img_size=img_size,
        smooth_window=smooth_window,
    ))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Load model
    model = build_gaformer(num_classes=num_classes, in_channels=8, dropout=0.0).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=label_list,
        output_dict=True,
        zero_division=0,
    )
    print("\n" + classification_report(y_true, y_pred, target_names=label_list, zero_division=0))

    # Lưu CSV
    rows = []
    for cls_name in label_list:
        r = report_dict[cls_name]
        rows.append({
            "class":     cls_name,
            "precision": round(r["precision"] * 100, 2),
            "recall":    round(r["recall"]    * 100, 2),
            "f1":        round(r["f1-score"]  * 100, 2),
            "support":   int(r["support"]),
        })
    # Thêm dòng tổng
    for agg in ("macro avg", "weighted avg"):
        r = report_dict[agg]
        rows.append({
            "class":     agg,
            "precision": round(r["precision"] * 100, 2),
            "recall":    round(r["recall"]    * 100, 2),
            "f1":        round(r["f1-score"]  * 100, 2),
            "support":   int(r["support"]),
        })

    df = pd.DataFrame(rows)
    csv_path = run_dir / "per_class_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Đã lưu: {csv_path}")

    json_path = run_dir / "per_class_report.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu: {json_path}")

    # Bar chart F1 từng class
    class_rows = [r for r in rows if r["class"] in label_list]
    names = [r["class"] for r in class_rows]
    f1_vals = [r["f1"] for r in class_rows]

    # Phân màu: gesture (G) vs non-gesture (N)
    colors = ["#2ecc71" if n.startswith("G") else "#e74c3c" for n in names]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(names, f1_vals, color=colors, alpha=0.85, width=0.6)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1-score (%)", fontsize=12)
    ax.set_title("F1-score theo từng class – GAFormer", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(facecolor="#2ecc71", label="Cử chỉ (G1–G15)"),
        mpatches.Patch(facecolor="#e74c3c", label="Không cử chỉ (N1–N5)"),
    ]
    ax.legend(handles=legend, fontsize=10)

    plt.tight_layout()
    bar_path = run_dir / "per_class_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {bar_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sinh báo cáo phân loại chi tiết từng class")
    p.add_argument("--run-dir",  required=True,
                   help="Thư mục chứa best.pt, labels.json, config.json")
    p.add_argument("--manifest", required=True,
                   help="Đường dẫn manifest.csv")
    p.add_argument("--root",     required=True,
                   help="Thư mục gốc dataset")
    p.add_argument("--test-subjects", default=None,
                   help="VD: S14,S15 (mặc định: S14,S15)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_report(args)
