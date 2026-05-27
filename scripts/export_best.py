"""Bước 4: Tự động chọn model tốt nhất và export sang demo/models/.

Pipeline:
    1. Scan tất cả run dirs của mỗi model type
    2. Chọn run có val_accuracy cao nhất (≥ min_threshold)
    3. Copy artifacts sang demo/models/{model}_{timestamp}/
    4. In tóm tắt

Cách dùng:
    python scripts/export_best.py
    python scripts/export_best.py --models transformer rf --min-acc 0.85
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from training import load_config


def _find_best_run(results_dir: Path) -> tuple[Path | None, float]:
    """Trả về (run_dir, best_val_acc)."""
    best_dir, best_val = None, -1.0
    if not results_dir.exists():
        return None, -1.0

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for fname in ("meta.json", "report.json"):
            f = run_dir / fname
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                v = float(data.get("val_accuracy", -1))
                if v > best_val:
                    best_val = v
                    best_dir = run_dir
            except Exception:
                pass
    return best_dir, best_val


def export_dl_run(run_dir: Path, dest: Path) -> None:
    """Copy best.pt + normalization.json + labels.json."""
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("best.pt", "normalization.json", "labels.json", "meta.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def export_rf_run(run_dir: Path, dest: Path) -> None:
    """Copy model.joblib + labels.json."""
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("model.joblib", "labels.json", "report.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=None)
    parser.add_argument("--models",   nargs="+",
                        default=["cnn", "lstm", "transformer", "rf"])
    parser.add_argument("--min-acc",  type=float, default=None)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Chỉ in kết quả, không copy file")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    paths      = cfg["paths"]
    exp_cfg    = cfg.get("export", {})
    min_thresh = args.min_acc or exp_cfg.get("min_threshold", 0.80)
    demo_dest  = Path(paths["demo_models"])

    results_key = {
        "cnn":         "results_cnn",
        "lstm":        "results_lstm",
        "transformer": "results_transformer",
        "rf":          "results_rf",
    }

    print(f"\n{'Model':<14} {'Best Val Acc':>13} {'Status':<30}")
    print("-" * 60)

    exported = []
    for model_name in args.models:
        results_dir = Path(paths[results_key[model_name]])
        best_dir, best_val = _find_best_run(results_dir)

        if best_dir is None:
            print(f"  {model_name:<12}  {'N/A':>13}  Chưa có run.")
            continue

        if best_val < min_thresh:
            print(f"  {model_name:<12}  {best_val:>12.4f}  Dưới ngưỡng {min_thresh:.2f} — bỏ qua.")
            continue

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = demo_dest / f"{model_name}_{ts}"

        if not args.dry_run:
            if model_name == "rf":
                export_rf_run(best_dir, dest)
            else:
                export_dl_run(best_dir, dest)

        status = f"→ {dest}" if not args.dry_run else f"[dry-run] → {dest}"
        print(f"  {model_name:<12}  {best_val:>12.4f}  {status}")
        exported.append({"model": model_name, "val_acc": best_val,
                         "source": str(best_dir), "dest": str(dest)})

    if not args.dry_run and exported:
        log_path = demo_dest / "export_log.json"
        demo_dest.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
        print(f"\n[export] ✓ {len(exported)} model(s) exported → {demo_dest}")
    elif not exported:
        print("\n[export] Không có model nào đủ điều kiện export.")


if __name__ == "__main__":
    main()
