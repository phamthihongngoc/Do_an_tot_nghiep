import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_label_meta(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {}
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    for item in data.get("labels", []):
        label_id = str(item.get("id", "")).strip()
        if not label_id:
            continue
        meta[label_id] = {
            "name": item.get("name", ""),
            "group": item.get("group", ""),
            "device": item.get("device", ""),
            "system_state": item.get("system_state", ""),
        }
    return meta


def build_label_series(df: pd.DataFrame, time_col: str) -> pd.DataFrame | None:
    if "label" not in df.columns:
        return None
    label_col = df["label"].astype(str)
    if label_col.nunique(dropna=False) == 1:
        return pd.DataFrame({time_col: [0.0], "label": [label_col.iloc[0]]})

    label_df = pd.DataFrame({time_col: df[time_col].values, "label": label_col.values})
    return label_df.sort_values(time_col)


def resample_csv(
    path: Path,
    output_path: Path,
    target_hz: float,
    target_duration: float | None,
    pad_mode: str,
) -> dict | None:
    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        return None

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    if df.empty:
        return None

    df = df.drop_duplicates(subset=["time"], keep="first")

    start = df["time"].iloc[0]
    df["time"] = df["time"] - start
    end = df["time"].iloc[-1]
    if end <= 0:
        return None

    dt = 1.0 / target_hz
    duration = target_duration if target_duration is not None else end
    new_time = np.arange(0.0, duration + 1e-9, dt)

    out = pd.DataFrame({"time": new_time})
    numeric_cols = [c for c in df.columns if c not in {"time", "label"}]
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        out[col] = np.interp(new_time, df["time"].values, series.values)
        if target_duration is not None and pad_mode == "zeros":
            out.loc[new_time > end, col] = 0.0

    label_info = build_label_series(df, "time")
    if label_info is not None:
        out = pd.merge_asof(
            out.sort_values("time"),
            label_info,
            on="time",
            direction="backward",
        )
        out["label"] = out["label"].bfill()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        "samples": len(out),
        "duration_s": f"{float(out['time'].iloc[-1] - out['time'].iloc[0]):.3f}",
    }


def parse_name_parts(file_name: str) -> dict:
    pattern = re.compile(
        r"(?P<session>\d{8}_\d{6})_(?P<subject>S\d+?)_(?P<label>[GN]\d+)_T(?P<trial>\d+)\.csv$",
        re.IGNORECASE,
    )
    match = pattern.search(file_name)
    if not match:
        return {"session_id": "", "subject_id": "", "label_id": "", "trial": ""}
    return {
        "session_id": match.group("session"),
        "subject_id": match.group("subject"),
        "label_id": match.group("label").upper(),
        "trial": match.group("trial"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="Input data folder")
    parser.add_argument("--output", default="data/processed_50hz", help="Output data folder")
    parser.add_argument("--target-hz", type=float, default=50.0, help="Target sampling rate")
    parser.add_argument("--target-duration", type=float, default=None, help="Target duration (s)")
    parser.add_argument("--pad-mode", choices=["edge", "zeros"], default="edge")
    parser.add_argument("--no-manifest", action="store_true", help="Skip manifest creation")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = root / args.input
    output_dir = root / args.output
    labels_path = root / "labels.json"
    label_meta = load_label_meta(labels_path)

    manifest_rows = []
    for path in sorted(input_dir.rglob("*.csv")):
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel
        stats = resample_csv(
            path,
            out_path,
            target_hz=args.target_hz,
            target_duration=args.target_duration,
            pad_mode=args.pad_mode,
        )
        if stats is None:
            continue

        name_parts = parse_name_parts(path.name)
        if not name_parts["subject_id"] or not name_parts["label_id"]:
            parts = path.parts
            try:
                name_parts["subject_id"] = parts[parts.index("raw") + 1]
                name_parts["label_id"] = parts[parts.index("raw") + 2]
            except Exception:
                pass

        recorded_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        meta = label_meta.get(name_parts["label_id"], {})
        manifest_rows.append(
            {
                "session_id": name_parts["session_id"],
                "subject_id": name_parts["subject_id"],
                "label_id": name_parts["label_id"],
                "label_name": meta.get("name", ""),
                "label_group": meta.get("group", ""),
                "label_device": meta.get("device", ""),
                "label_system_state": meta.get("system_state", ""),
                "scene": meta.get("device", ""),
                "trial": name_parts["trial"],
                "duration_s": stats["duration_s"],
                "samples": str(stats["samples"]),
                "path": str(out_path.relative_to(root)),
                "recorded_at": recorded_at,
            }
        )

    if args.no_manifest:
        return

    manifest_path = output_dir / f"manifest_{int(args.target_hz)}hz.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "subject_id",
                "label_id",
                "label_name",
                "label_group",
                "label_device",
                "label_system_state",
                "scene",
                "trial",
                "duration_s",
                "samples",
                "path",
                "recorded_at",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
