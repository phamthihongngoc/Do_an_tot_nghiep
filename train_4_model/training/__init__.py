"""Training package — gesture recognition IMU.

Usage:
    from training import load_config, REPO_ROOT
    cfg = load_config()
    raw_data_dir = cfg["paths"]["raw_data"]   # Path object
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load config.yaml và resolve tất cả paths thành Path objects tuyệt đối."""
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    for key, value in paths.items():
        if value is not None:
            paths[key] = REPO_ROOT / value
    return cfg
