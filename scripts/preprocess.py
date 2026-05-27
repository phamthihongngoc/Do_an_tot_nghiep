"""Bước 1: Tiền xử lý — resample 100Hz → 50Hz và tạo manifest.

Wrapper gọi lại resample_to_50hz.py của data_collection (không sửa code gốc).

Cách dùng:
    python scripts/preprocess.py
    python scripts/preprocess.py --config config.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from training import REPO_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample IMU data 100Hz → 50Hz")
    parser.add_argument("--config", default=None, help="Đường dẫn config.yaml")
    parser.add_argument("--target-duration", type=float, default=None,
                        help="Cắt/pad chuỗi về độ dài cố định (giây)")
    parser.add_argument("--pad-mode", choices=["edge", "zeros"], default=None)
    args = parser.parse_args()

    cfg      = load_config(args.config)
    paths    = cfg["paths"]
    prep     = cfg["preprocessing"]

    input_dir  = paths["raw_data"]
    output_dir = paths["processed"]
    target_hz  = prep["target_hz"]
    pad_mode   = args.pad_mode or prep.get("pad_mode", "edge")
    target_dur = args.target_duration or prep.get("window_seconds")

    resample_script = REPO_ROOT / "data_collection" / "resample_to_50hz.py"
    if not resample_script.exists():
        print(f"[ERROR] Không tìm thấy: {resample_script}")
        sys.exit(1)

    # Đường dẫn truyền vào resample_to_50hz.py là tương đối từ data_collection/
    dc_root  = REPO_ROOT / "data_collection"
    rel_in   = input_dir.relative_to(dc_root)
    rel_out  = output_dir.relative_to(dc_root)

    cmd = [
        sys.executable, str(resample_script),
        "--input",       str(rel_in),
        "--output",      str(rel_out),
        "--target-hz",   str(target_hz),
        "--pad-mode",    pad_mode,
    ]
    if target_dur is not None:
        cmd += ["--target-duration", str(target_dur)]

    print(f"[preprocess] Chạy: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(dc_root))
    if result.returncode != 0:
        print("[ERROR] Resample thất bại.")
        sys.exit(result.returncode)

    print(f"\n[preprocess] ✓ Xong. Dữ liệu 50Hz tại: {output_dir}")
    print(f"[preprocess] Manifest: {output_dir / 'manifest_50hz.csv'}")


if __name__ == "__main__":
    main()
