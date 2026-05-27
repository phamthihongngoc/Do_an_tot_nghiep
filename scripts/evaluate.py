"""Bước 3: Đánh giá và so sánh tất cả models trên test set.

Đọc best run của mỗi model, chạy inference, in bảng so sánh.

Cách dùng:
    python scripts/evaluate.py
    python scripts/evaluate.py --models cnn lstm transformer rf
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from training import load_config
from training.dataset import (
    IMUDataset,
    load_labels_json,
    load_manifest,
    split_manifest,
    compute_normalization,
    _read_csv_sequence,
    resample_1d,
)
from training.features import extract_features_demo
from training.models import build_model
from training.models_rf import LateFusionRF
from training.trainer import evaluate as dl_evaluate
from torch.utils.data import DataLoader


def _find_best_run(results_dir: Path, metric_key: str = "val_accuracy") -> Path | None:
    """Tìm run có metric cao nhất trong thư mục kết quả."""
    best_dir, best_val = None, -np.inf
    if not results_dir.exists():
        return None
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_f = run_dir / "meta.json"
        report_f = run_dir / "report.json"
        for f in (meta_f, report_f):
            if f.exists():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    v = float(data.get(metric_key, data.get("val_accuracy", -1)))
                    if v > best_val:
                        best_val = v
                        best_dir = run_dir
                except Exception:
                    pass
    return best_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default=None)
    parser.add_argument("--models",  nargs="+", default=["cnn", "lstm", "transformer", "rf"],
                        choices=["cnn", "lstm", "transformer", "rf"])
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    paths      = cfg["paths"]
    split_cfg  = cfg["split"]
    prep_cfg   = cfg["preprocessing"]
    target_len = prep_cfg["target_len"]

    labels_all = load_labels_json(Path(paths["labels_json"]))
    label_filter = split_cfg.get("label_filter")
    if label_filter:
        labels_all = [l for l in labels_all if l in label_filter]

    processed_dir = Path(paths["processed"])
    manifest      = load_manifest(processed_dir)
    splits        = split_manifest(
        manifest,
        test_subjects=split_cfg["test_subjects"],
        val_subjects=split_cfg["val_subjects"],
        label_filter=label_filter,
    )

    # Tính norm stats từ train
    train_seqs = []
    for _, row in splits["train"].iterrows():
        s = _read_csv_sequence(row["path"], processed_dir.parent)
        if s is not None:
            train_seqs.append(resample_1d(s, target_len))
    from training.dataset import compute_normalization
    norm = compute_normalization(train_seqs)

    results_key = {
        "cnn": "results_cnn", "lstm": "results_lstm",
        "transformer": "results_transformer", "rf": "results_rf",
    }

    print(f"\n{'Model':<15} {'Val Acc':>8} {'Test Acc':>9} {'F1-macro':>9} {'Infer(ms)':>10}")
    print("-" * 55)

    summary = {}
    for model_name in args.models:
        run_dir = _find_best_run(Path(paths[results_key[model_name]]))
        if run_dir is None:
            print(f"  {model_name:<13}  [Chưa có run — chạy train trước]")
            continue

        test_df = splits["test"]
        if model_name == "rf":
            # RF inference
            model = LateFusionRF.load(str(run_dir / "model.joblib"))
            seqs, y_true = [], []
            for _, row in test_df.iterrows():
                s = _read_csv_sequence(row["path"], processed_dir.parent)
                if s is None:
                    continue
                seqs.append(resample_1d(s, target_len))
                y_true.append(labels_all.index(str(row["label_id"])))
            X = np.stack(seqs)
            F = extract_features_demo(X)
            t0 = time.perf_counter()
            y_pred = model.predict(F)
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) / len(y_pred) * 1000
            val_acc  = json.loads((run_dir / "report.json").read_text(
                encoding="utf-8")).get("val_accuracy", 0)
        else:
            # DL inference
            ckpt = torch.load(run_dir / "best.pt", map_location=args.device)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            model = build_model(model_name, len(labels_all), cfg)
            model.load_state_dict(ckpt["model"])
            ds = IMUDataset(test_df, processed_dir.parent, labels_all, target_len,
                            mean=norm["mean"], std=norm["std"])
            loader = DataLoader(ds, batch_size=64, shuffle=False)
            t0 = time.perf_counter()
            _, y_true, y_pred = dl_evaluate(model, loader, args.device)
            t1 = time.perf_counter()
            infer_ms  = (t1 - t0) / len(y_pred) * 1000
            val_acc   = float(meta.get("val_accuracy", 0))

        test_acc = accuracy_score(y_true, y_pred)
        f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"  {model_name:<13} {val_acc:>8.4f} {test_acc:>9.4f} {f1:>9.4f} {infer_ms:>9.2f}")
        summary[model_name] = {
            "val_accuracy": val_acc, "test_accuracy": test_acc,
            "f1_macro": f1, "infer_ms_per_sample": infer_ms,
            "run_dir": str(run_dir),
        }

    out = Path(paths["figures"]).parent / "eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[evaluate] Tóm tắt lưu tại: {out}")


if __name__ == "__main__":
    main()
