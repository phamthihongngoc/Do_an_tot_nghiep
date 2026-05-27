"""Bước 2b: Huấn luyện Late Fusion Random Forest.

Kiến trúc: 2 nhánh RF độc lập (Accel + Gyro) → Soft/Hard Vote.

Kết quả lưu vào:
    training/results_rf/run_{timestamp}/
        model.joblib   — LateFusionRF object
        labels.json    — danh sách label IDs
        report.json    — accuracy, classification report

Cách dùng:
    python scripts/train_rf_latefusion.py
    python scripts/train_rf_latefusion.py --fusion hard --n-estimators 500
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from training import load_config
from training.dataset import load_numpy_splits, load_labels_json
from training.features import extract_features_demo, extract_features_rf
from training.models_rf import LateFusionRF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default=None)
    parser.add_argument("--fusion",       default=None, choices=["soft", "hard"])
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--features",     default="demo", choices=["demo", "rich"],
                        help="demo=24-d (tương thích export); rich=75-d (train quality)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rf_cfg = cfg["models"]["random_forest"]

    if args.fusion:      rf_cfg["fusion_strategy"] = args.fusion
    if args.n_estimators: rf_cfg["n_estimators"]   = args.n_estimators

    # ── Load dữ liệu ─────────────────────────────────────────────────────────
    print("[train_rf] Loading data...")
    splits = load_numpy_splits(cfg)      # {"train": (X,y), "val": ..., "test": ...}

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # ── Trích xuất features ───────────────────────────────────────────────────
    feat_fn = extract_features_demo if args.features == "demo" else extract_features_rf
    print(f"[train_rf] Feature extraction ({args.features})...")
    F_train = feat_fn(X_train)
    F_val   = feat_fn(X_val)
    F_test  = feat_fn(X_test)
    print(f"  Feature shape: {F_train.shape}")

    # ── Huấn luyện ──────────────────────────────────────────────────────────
    model = LateFusionRF(
        n_estimators    = rf_cfg["n_estimators"],
        max_depth       = rf_cfg.get("max_depth"),
        min_samples_split = rf_cfg["min_samples_split"],
        class_weight    = rf_cfg["class_weight"],
        fusion_strategy = rf_cfg["fusion_strategy"],
        random_state    = cfg["project"]["seed"],
    )
    print(f"[train_rf] Training {model}...")
    model.fit(F_train, y_train)

    val_acc  = accuracy_score(y_val,  model.predict(F_val))
    test_acc = accuracy_score(y_test, model.predict(F_test))
    print(f"[train_rf] Val  accuracy = {val_acc:.4f}")
    print(f"[train_rf] Test accuracy = {test_acc:.4f}")

    labels_all = load_labels_json(Path(cfg["paths"]["labels_json"]))
    label_filter = cfg["split"].get("label_filter")
    if label_filter:
        labels_all = [l for l in labels_all if l in label_filter]

    # ── Lưu kết quả ─────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["paths"]["results_rf"]) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # model.joblib — dùng LateFusion với 24-d features để tương thích demo
    # Nếu train với rich features, lưu thêm version demo-compatible
    if args.features == "rich":
        print("[train_rf] Tạo bản demo-compatible (24-d)...")
        demo_model = LateFusionRF(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg.get("max_depth"),
            min_samples_split=rf_cfg["min_samples_split"],
            class_weight=rf_cfg["class_weight"],
            fusion_strategy=rf_cfg["fusion_strategy"],
            random_state=cfg["project"]["seed"],
        )
        demo_model.fit(extract_features_demo(X_train), y_train)
        demo_model.save(str(run_dir / "model.joblib"))
    else:
        model.save(str(run_dir / "model.joblib"))

    (run_dir / "labels.json").write_text(
        json.dumps(labels_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    y_pred_test = model.predict(F_test)
    report = {
        "val_accuracy":  float(val_acc),
        "test_accuracy": float(test_acc),
        "fusion_strategy": rf_cfg["fusion_strategy"],
        "feature_mode":  args.features,
        "n_estimators":  rf_cfg["n_estimators"],
        "classification_report": classification_report(
            y_test, y_pred_test,
            target_names=labels_all[:int(y_test.max()) + 1],
            output_dict=True,
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(run_dir / "test_preds.npy", np.stack([y_test, y_pred_test]))

    print(f"[train_rf] Run saved → {run_dir}")


if __name__ == "__main__":
    main()
