"""Bước 2: Huấn luyện Deep Learning — CNN / CNN-BiLSTM / Transformer.

Kết quả lưu vào:
    training/results_{model}_200epoch/run_{timestamp}/

Cách dùng:
    python scripts/train_dl.py --model transformer
    python scripts/train_dl.py --model cnn --epochs 50 --device cuda
    python scripts/train_dl.py --model lstm --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training import REPO_ROOT, load_config
from training.augmentation import build_augment
from training.dataset import (
    IMUDataset,
    compute_normalization,
    load_labels_json,
    load_manifest,
    split_manifest,
    _read_csv_sequence,
    resample_1d,
)
from training.models import build_model, count_parameters
from training.trainer import evaluate, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=["cnn", "lstm", "transformer"])
    parser.add_argument("--config",  default=None)
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--no-aug",  action="store_true", help="Tắt augmentation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    # Override từ CLI
    if args.epochs:  cfg["training"]["max_epochs"]    = args.epochs
    if args.lr:      cfg["training"]["learning_rate"] = args.lr
    if args.batch:   cfg["training"]["batch_size"]    = args.batch
    if args.no_aug:  cfg["augmentation"]["enabled"]   = False

    paths     = cfg["paths"]
    split_cfg = cfg["split"]
    prep_cfg  = cfg["preprocessing"]
    t_cfg     = cfg["training"]

    labels_all  = load_labels_json(Path(paths["labels_json"]))
    label_filter = split_cfg.get("label_filter")
    if label_filter:
        labels_all = [l for l in labels_all if l in label_filter]
    num_classes = len(labels_all)

    processed_dir = Path(paths["processed"])
    manifest      = load_manifest(processed_dir)
    splits        = split_manifest(
        manifest,
        test_subjects=split_cfg["test_subjects"],
        val_subjects=split_cfg["val_subjects"],
        label_filter=label_filter,
    )

    target_len = prep_cfg["target_len"]

    # Tính chuẩn hóa trên train set
    print("[train_dl] Tính chuẩn hóa trên train set...")
    train_seqs = []
    for _, row in splits["train"].iterrows():
        s = _read_csv_sequence(row["path"], processed_dir.parent)
        if s is not None:
            train_seqs.append(resample_1d(s, target_len))
    norm = compute_normalization(train_seqs)
    print(f"  mean={norm['mean'].round(4)}, std={norm['std'].round(4)}")

    augment = build_augment(cfg)
    print(f"[train_dl] Augmentation: {'ON' if augment else 'OFF'}")

    def make_ds(split_name, do_aug=False):
        return IMUDataset(
            splits[split_name], processed_dir.parent, labels_all, target_len,
            mean=norm["mean"], std=norm["std"],
            augment_fn=augment if do_aug else None,
        )

    nw = t_cfg.get("num_workers", 2)
    train_loader = DataLoader(make_ds("train", do_aug=True),
                              batch_size=t_cfg["batch_size"], shuffle=True, num_workers=nw)
    val_loader   = DataLoader(make_ds("val"),
                              batch_size=t_cfg["batch_size"], shuffle=False, num_workers=nw)
    test_loader  = DataLoader(make_ds("test"),
                              batch_size=t_cfg["batch_size"], shuffle=False, num_workers=nw)

    model = build_model(args.model, num_classes, cfg)
    print(f"[train_dl] Model: {args.model.upper()} | Params: {count_parameters(model):,}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    key_map = {"cnn": "results_cnn", "lstm": "results_lstm", "transformer": "results_transformer"}
    run_dir = Path(paths[key_map[args.model]]) / f"run_{ts}"

    history = train(model, train_loader, val_loader, cfg, run_dir,
                    labels_all, norm, device=args.device)

    # ── Đánh giá trên test ───────────────────────────────────────────────────
    ckpt = torch.load(run_dir / "best.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_acc, y_true, y_pred = evaluate(model, test_loader, args.device)
    print(f"\n[train_dl] Test accuracy = {test_acc:.4f}")

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["test_accuracy"] = float(test_acc)
    meta["model_name"]    = args.model
    meta["num_classes"]   = num_classes
    meta["target_len"]    = target_len
    meta["params"]        = count_parameters(model)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    np.save(run_dir / "test_preds.npy", np.stack([y_true, y_pred]))
    print(f"[train_dl] Run saved → {run_dir}")


if __name__ == "__main__":
    main()
