"""Late Fusion Random Forest — 2 nhánh độc lập: Accel + Gyro.

Kiến trúc:
    Input (N, 24) ← extract_rf_features() của demo/predict.py
    ┌─────────────────────┐  ┌────────────────────┐
    │  RF Nhánh Accel     │  │  RF Nhánh Gyro     │
    │  (12 đặc trưng)     │  │  (12 đặc trưng)    │
    └──────────┬──────────┘  └─────────┬──────────┘
               │      Late Fusion      │
               └──────────┬────────────┘
                     Soft/Hard Vote
                    (num_classes,)

Tương thích demo/predict.py:
    model.predict_proba(features_24)  →  (N, num_classes)
    model.classes_                    →  [0, 1, ..., num_classes-1]
"""
from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Chỉ số đặc trưng trong vector 24-d [mean(6), std(6), min(6), max(6)]
# Với channels = [ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps]
_ACCEL_IDX = [0, 1, 2,  6, 7, 8,  12, 13, 14,  18, 19, 20]  # ax/ay/az × 4 stats
_GYRO_IDX  = [3, 4, 5,  9, 10, 11, 15, 16, 17,  21, 22, 23]  # gx/gy/gz × 4 stats


class LateFusionRF:
    """Random Forest Late Fusion.

    Interface giống sklearn: fit / predict / predict_proba / classes_.
    Được serialize bởi joblib và load bởi GesturePredictor trong demo.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        class_weight: str = "balanced",
        fusion_strategy: Literal["soft", "hard"] = "soft",
        random_state: int = 42,
    ) -> None:
        rf_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        self.accel_rf = RandomForestClassifier(**rf_kwargs)
        self.gyro_rf  = RandomForestClassifier(**rf_kwargs)
        self.fusion_strategy: Literal["soft", "hard"] = fusion_strategy
        self.classes_: np.ndarray | None = None
        self._fitted = False

    # ── Trích xuất đặc trưng per-branch ────────────────────────────────────
    @staticmethod
    def split_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tách (N, 24) → (N, 12) accel + (N, 12) gyro."""
        return features[:, _ACCEL_IDX], features[:, _GYRO_IDX]

    # ── Huấn luyện ─────────────────────────────────────────────────────────
    def fit(self, features: np.ndarray, y: np.ndarray) -> "LateFusionRF":
        """
        features : (N, 24)  — [mean(6), std(6), min(6), max(6)]
        y        : (N,)     — integer labels 0..num_classes-1
        """
        f_accel, f_gyro = self.split_features(features)
        self.accel_rf.fit(f_accel, y)
        self.gyro_rf.fit(f_gyro, y)
        self.classes_ = self.accel_rf.classes_
        self._fitted = True
        return self

    # ── Dự đoán ────────────────────────────────────────────────────────────
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Trả về (N, num_classes) xác suất sau late fusion."""
        if not self._fitted:
            raise RuntimeError("Model chưa được huấn luyện. Gọi .fit() trước.")
        f_accel, f_gyro = self.split_features(features)
        p_accel = self.accel_rf.predict_proba(f_accel)   # (N, C)
        p_gyro  = self.gyro_rf.predict_proba(f_gyro)     # (N, C)

        if self.fusion_strategy == "soft":
            return (p_accel + p_gyro) / 2.0
        # Hard fusion: majority vote per sample
        pred_a = self.accel_rf.predict(f_accel)
        pred_g = self.gyro_rf.predict(f_gyro)
        n, c   = p_accel.shape
        fused  = np.zeros((n, c), dtype=np.float32)
        for i in range(n):
            if pred_a[i] == pred_g[i]:
                fused[i, pred_a[i]] = 1.0
            else:
                fused[i] = (p_accel[i] + p_gyro[i]) / 2.0  # tie-break → soft
        return fused

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    # ── Serialize ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LateFusionRF":
        return joblib.load(path)

    # ── Thông tin ──────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"LateFusionRF(n_estimators={self.accel_rf.n_estimators}, "
                f"fusion={self.fusion_strategy}, {status})")
