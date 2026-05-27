"""
Sinh confusion matrix + t-SNE cho 4 model trong comparison:
  - gadf_coatnet0  : GADF + CoAtNet-0
  - gasf_coatnet0  : GASF + CoAtNet-0
  - gadf_resnet50  : GADF + ResNet-50
  - gaformer       : GADF + CoAtNet-light (GAFormer)

Chạy trên Colab (nơi có data):

  python compare_confusion.py \
    --comp-dir /content/drive/MyDrive/DOAN2/gaformer/checkpoints/comparison \
    --labels   /content/drive/MyDrive/DOAN2/gaformer/checkpoints/gaformer_20260525_113523/labels.json \
    --manifest /content/drive/MyDrive/DOAN2/SHG_Dataset/manifest_shg.csv \
    --root     /content/drive/MyDrive/DOAN2/SHG_Dataset

Output (mỗi model sinh 2 file trong <comp-dir>):
  {method}_confusion.png
  {method}_tsne.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from baselines import build_resnet50_8ch
from dataset import GAFDataset, GAFDatasetConfig, parse_subjects
from model import build_coatnet0, build_gaformer
from utils import plot_confusion_matrix, plot_tsne, seed_all

DEFAULT_TEST = ["S14", "S15"]

METHOD_CFG = {
    "gadf_coatnet0": {
        "label":        "GADF + CoAtNet-0",
        "model_fn":     lambda n: build_coatnet0(n, dropout=0.0),
        "dataset_type": "gadf",
        "has_return_features": True,   # GAFormer class hỗ trợ return_features
    },
    "gasf_coatnet0": {
        "label":        "GASF + CoAtNet-0",
        "model_fn":     lambda n: build_coatnet0(n, dropout=0.0),
        "dataset_type": "gasf",
        "has_return_features": True,
    },
    "gadf_resnet50": {
        "label":        "GADF + ResNet-50",
        "model_fn":     lambda n: build_resnet50_8ch(n, dropout=0.0),
        "dataset_type": "gadf",
        "has_return_features": False,  # dùng forward hook lấy từ avgpool
    },
    "gaformer": {
        "label":        "GADF + CoAtNet-light (GAFormer)",
        "model_fn":     lambda n: build_gaformer(n, in_channels=8, dropout=0.0),
        "dataset_type": "gadf",
        "has_return_features": True,
    },
}


# ---------------------------------------------------------------------------
# Helper: inference một lần, trả về (y_true, y_pred, features)
# ---------------------------------------------------------------------------

def _collect_with_hook(model: torch.nn.Module, loader, device: torch.device):
    """Dùng forward hook trên avgpool của ResNet để lấy feature 2048-dim."""
    captured: list[np.ndarray] = []

    def _hook(module, input, output):
        # output shape: (B, 2048, 1, 1)  →  flatten → (B, 2048)
        captured.append(output.flatten(1).detach().cpu().numpy())

    handle = model.avgpool.register_forward_hook(_hook)
    try:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
    finally:
        handle.remove()

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(captured),
    )


def _collect_with_return_features(model: torch.nn.Module, loader, device: torch.device):
    """Dùng model(x, return_features=True) cho GAFormer / CoAtNet."""
    all_preds, all_labels, all_feats = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, feats = model(images, return_features=True)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_feats.append(feats.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_feats),
    )


def collect(model, loader, device, has_return_features: bool):
    if has_return_features:
        return _collect_with_return_features(model, loader, device)
    return _collect_with_hook(model, loader, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    comp_dir      = Path(args.comp_dir)
    manifest_path = Path(args.manifest)
    root_dir      = Path(args.root)

    labels_data    = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    label_list     = labels_data["labels"]
    label_to_index = {lbl: i for i, lbl in enumerate(label_list)}
    num_classes    = len(label_list)
    print(f"Classes ({num_classes}): {label_list}")

    test_subjects = parse_subjects(args.test_subjects, DEFAULT_TEST)
    methods = list(METHOD_CFG.keys()) if args.method == "all" else [args.method]

    for method in methods:
        cfg = METHOD_CFG[method]
        print(f"\n{'='*55}")
        print(f"Model: {cfg['label']}")

        ckpt_path = comp_dir / f"{method}_best.pt"
        if not ckpt_path.exists():
            print(f"  Không tìm thấy {ckpt_path}, bỏ qua.")
            continue

        # Dataset
        test_ds = GAFDataset(GAFDatasetConfig(
            manifest_path=manifest_path,
            root_dir=root_dir,
            subjects=test_subjects,
            label_to_index=label_to_index,
            img_size=args.img_size,
            smooth_window=args.smooth_window,
            transform=cfg["dataset_type"],
        ))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        print(f"  Test samples: {len(test_ds)}")

        # Model
        model = cfg["model_fn"](num_classes).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  Loaded weights (epoch {ckpt.get('epoch', '?')})")

        # Inference + feature extraction
        y_true, y_pred, features = collect(
            model, test_loader, device, cfg["has_return_features"]
        )
        acc = (y_true == y_pred).mean() * 100
        print(f"  Test Accuracy : {acc:.2f}%")
        print(f"  Feature shape : {features.shape}")

        # Confusion matrix
        cm_path = comp_dir / f"{method}_confusion.png"
        plot_confusion_matrix(y_true, y_pred, label_list, cm_path)
        print(f"  Đã lưu CM  : {cm_path}")

        # t-SNE
        tsne_path = comp_dir / f"{method}_tsne.png"
        print(f"  Đang tính t-SNE...")
        plot_tsne(features, y_true, label_list, tsne_path)
        print(f"  Đã lưu t-SNE: {tsne_path}")

    print("\nHoàn tất!")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sinh confusion matrix + t-SNE cho 4 model so sánh")
    p.add_argument("--comp-dir",  required=True,
                   help="Thư mục chứa *_best.pt (VD: checkpoints/comparison)")
    p.add_argument("--labels",    required=True,
                   help="Đường dẫn labels.json")
    p.add_argument("--manifest",  required=True,
                   help="Đường dẫn manifest.csv")
    p.add_argument("--root",      required=True,
                   help="Thư mục gốc dataset")
    p.add_argument("--method", default="all",
                   choices=["all", "gadf_coatnet0", "gasf_coatnet0", "gadf_resnet50", "gaformer"],
                   help="Chạy tất cả hoặc chỉ một model")
    p.add_argument("--test-subjects", default=None, help="VD: S14,S15")
    p.add_argument("--img-size",      type=int, default=224)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--batch-size",    type=int, default=32)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
