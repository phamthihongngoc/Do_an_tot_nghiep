"""Gesture prediction utilities — shared by CLI and backend.

Loads a trained checkpoint (CNN/LSTM/Transformer) or exported Random Forest
artifact plus labels, and exposes a single `predict(samples)` function that
takes a (T, 6) array of IMU samples (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
and returns the predicted label id, human-readable name, target device, and
confidence.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Make the training model definitions importable. Prefer the cleaned 50Hz tree
# when it exists, and fall back to the original training folder.
ROOT = Path(__file__).resolve().parents[1]
for model_src in (
    ROOT / "train_4_model" / "training",
    ROOT / "training",
    ROOT / "training_50hz_clean" / "src",
):
    if model_src.exists():
        sys.path.insert(0, str(model_src))
from models import CNN1D, LSTMModel, TransformerModel  # noqa: E402


CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
DEFAULT_TARGET_LEN = int(os.environ.get("DEMO_TARGET_LEN", "201"))
TORCH_MODELS = {"cnn", "lstm", "transformer"}
SKLEARN_MODEL_ALIASES = {"rf": "random_forest", "random_forest": "random_forest"}


def canonical_model_name(name: str) -> str:
    name = name.lower()
    return SKLEARN_MODEL_ALIASES.get(name, name)


def resample_sequence(samples: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, C) IMU sequence to the model input length."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if samples.shape[0] == target_len:
        return samples.astype(np.float32, copy=False)
    if samples.shape[0] == 0:
        raise ValueError("Cannot predict an empty IMU sequence")
    if samples.shape[0] == 1:
        return np.repeat(samples.astype(np.float32), target_len, axis=0)

    src_x = np.linspace(0.0, 1.0, samples.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        out[:, channel] = np.interp(dst_x, src_x, samples[:, channel]).astype(np.float32)
    return out


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    name = canonical_model_name(name)
    if name == "cnn":
        return CNN1D(input_channels=len(CHANNELS), num_classes=num_classes)
    if name == "lstm":
        return LSTMModel(input_channels=len(CHANNELS), num_classes=num_classes)
    if name == "transformer":
        return TransformerModel(input_channels=len(CHANNELS), num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def load_label_ids(run_dir: Path) -> List[str]:
    labels_path = run_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing labels.json or meta.json in {run_dir}")
        labels = json.loads(meta_path.read_text(encoding="utf-8")).get("labels")

    if isinstance(labels, dict) and "id_to_label" in labels:
        return list(labels["id_to_label"])
    if isinstance(labels, dict) and "labels" in labels:
        return list(labels["labels"])
    return list(labels)


def extract_rf_features(samples: np.ndarray) -> np.ndarray:
    # samples: (N, T, C)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1)
    min_v = samples.min(axis=1)
    max_v = samples.max(axis=1)
    return np.concatenate([mean, std, min_v, max_v], axis=1).astype(np.float32)


@dataclass
class GesturePredictor:
    model: Any
    model_name: str
    mean: np.ndarray  # (6,)
    std: np.ndarray   # (6,)
    id_to_label: List[str]
    label_meta: Dict[str, Dict[str, Any]]
    device: torch.device
    target_len: int

    @classmethod
    def load(cls, run_dir: str | Path, model_name: str,
             labels_json_path: str | Path | None = None,
             device: str = "cpu",
             target_len: int = DEFAULT_TARGET_LEN) -> "GesturePredictor":
        run_dir = Path(run_dir)
        model_name = canonical_model_name(model_name)
        id_to_label = load_label_ids(run_dir)

        if model_name == "random_forest":
            model_path = run_dir / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing Random Forest artifact: {model_path}")
            import joblib

            model = joblib.load(model_path)
            mean = np.zeros(len(CHANNELS), dtype=np.float32)
            std = np.ones(len(CHANNELS), dtype=np.float32)
        else:
            ckpt = torch.load(run_dir / "best.pt", map_location=device)
            norm = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            model = build_model(model_name, num_classes=len(id_to_label))
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            model.to(device)
            model.eval()

        # Optional richer per-label metadata (device/action) from labels.json.
        meta: Dict[str, Dict[str, Any]] = {}
        if labels_json_path is None:
            labels_json_path = ROOT / "data_collection" / "labels.json"
        labels_json_path = Path(labels_json_path)
        if labels_json_path.exists():
            data = json.loads(labels_json_path.read_text(encoding="utf-8"))
            for item in data.get("labels", []):
                meta[item["id"]] = item

        return cls(model=model, model_name=model_name, mean=mean, std=std,
                   id_to_label=id_to_label, label_meta=meta,
                   device=torch.device(device), target_len=target_len)

    def _prepare(self, samples: np.ndarray) -> torch.Tensor:
        if samples.ndim != 2 or samples.shape[1] != len(CHANNELS):
            raise ValueError(
                f"Expected (T, {len(CHANNELS)}) samples, got {samples.shape}")
        x = samples.astype(np.float32)
        x = resample_sequence(x, self.target_len)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, T, 6)

    def _predict_sklearn_proba(self, samples: np.ndarray) -> np.ndarray:
        x = samples.astype(np.float32)
        x = resample_sequence(x, self.target_len)
        features = extract_rf_features(x[None, :, :])
        probs = np.zeros(len(self.id_to_label), dtype=np.float32)

        if hasattr(self.model, "predict_proba"):
            model_probs = self.model.predict_proba(features)[0]
            classes = getattr(self.model, "classes_", np.arange(len(model_probs)))
            for cls, prob in zip(classes, model_probs):
                idx = int(cls)
                if 0 <= idx < len(probs):
                    probs[idx] = float(prob)
        else:
            idx = int(self.model.predict(features)[0])
            if 0 <= idx < len(probs):
                probs[idx] = 1.0
        return probs

    @torch.no_grad()
    def predict(self, samples: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        input_samples = int(np.asarray(samples).shape[0])
        if self.model_name == "random_forest":
            probs = self._predict_sklearn_proba(np.asarray(samples))
        else:
            x = self._prepare(np.asarray(samples))
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        top = [
            {
                "label": self.id_to_label[i],
                "confidence": float(probs[i]),
                **{k: self.label_meta.get(self.id_to_label[i], {}).get(k)
                   for k in ("name", "device", "system_state")},
            }
            for i in top_idx
        ]
        best = top[0]
        return {
            "label": best["label"],
            "name": best.get("name"),
            "device": best.get("device"),
            "system_state": best.get("system_state"),
            "confidence": best["confidence"],
            "top_k": top,
            "input_samples": input_samples,
            "target_len": self.target_len,
        }


def load_csv(path: str | Path) -> np.ndarray:
    """Load a single trial CSV produced by the collector.

    Expects columns: time, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, label.
    Returns array of shape (T, 6) in CHANNELS order.
    """
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    return df[CHANNELS].to_numpy(dtype=np.float32)


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Predict gesture from a CSV trial.")
    ap.add_argument("--run-dir", required=True,
                    help="Path to results/<model>_<ts> containing best.pt etc.")
    ap.add_argument("--model", required=True, choices=["cnn", "lstm", "transformer", "random_forest", "rf"])
    ap.add_argument("--csv", required=True, help="Trial CSV to predict.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-len", type=int, default=DEFAULT_TARGET_LEN,
                    help="Model input length. Use 201 for the 50Hz dataset.")
    args = ap.parse_args()

    predictor = GesturePredictor.load(
        args.run_dir, args.model, device=args.device, target_len=args.target_len)
    samples = load_csv(args.csv)
    result = predictor.predict(samples)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
