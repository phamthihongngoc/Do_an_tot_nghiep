"""ReportGenerator — tự động sinh toàn bộ biểu đồ và bảng báo cáo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from training import REPO_ROOT, load_config
from training.dataset import load_labels_json, load_manifest, _read_csv_sequence, resample_1d, split_manifest
from visualization.plot_signals import (
    plot_all_gestures, plot_gesture_detail,
    plot_samples_per_subject, plot_sequence_lengths,
    plot_samples_per_gesture, plot_samples_per_split,
    plot_imu_points_per_subject, plot_gesture_comparison,
)
from visualization.plot_metrics import (
    plot_loss_acc_curves, plot_f1_per_class, table_model_performance,
    plot_acc_f1_by_model, plot_accuracy_per_model, plot_f1_per_model,
    plot_accuracy_per_class,
)
from visualization.plot_confusion import plot_confusion_matrix, plot_confusion_comparison
from visualization.plot_tsne import plot_tsne
from visualization.plot_tables import (
    table_dataset_split, table_samples_imu_per_split,
    table_csv_structure, table_subjects_stats,
    table_feature_stats, table_model_comparison,
)
from visualization.plot_architecture import (
    plot_cnn_architecture, plot_lstm_architecture, plot_transformer_architecture,
)

CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]


class ReportGenerator:
    """Sinh tất cả figures và tables cho báo cáo đồ án."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg       = cfg or load_config()
        self.paths     = self.cfg["paths"]
        self.fig_dir   = Path(self.paths["figures"])
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.labels_all  = load_labels_json(Path(self.paths["labels_json"]))
        self.label_names = self._load_label_names()
        self.processed_dir = Path(self.paths["processed"])
        self._manifest: pd.DataFrame | None = None

    def _load_label_names(self) -> Dict[str, str]:
        data = json.loads(Path(self.paths["labels_json"]).read_text(encoding="utf-8"))
        return {item["id"]: item["name"] for item in data.get("labels", [])}

    @property
    def manifest(self) -> pd.DataFrame:
        if self._manifest is None:
            self._manifest = load_manifest(self.processed_dir)
        return self._manifest

    @property
    def splits(self) -> Dict[str, pd.DataFrame]:
        if not hasattr(self, "_splits") or self._splits is None:
            split_cfg = self.cfg["split"]
            self._splits = split_manifest(
                self.manifest,
                test_subjects=split_cfg["test_subjects"],
                val_subjects=split_cfg["val_subjects"],
            )
        return self._splits

    def _load_representative_sequence(self, label_id: str) -> np.ndarray | None:
        subset = self.manifest[self.manifest["label_id"] == label_id]
        if subset.empty:
            return None
        row = subset.iloc[len(subset) // 2]
        seq = _read_csv_sequence(row["path"], self.processed_dir.parent)
        if seq is None:
            return None
        return resample_1d(seq, self.cfg["preprocessing"]["target_len"])

    def _ok(self, name: str) -> Path:
        path = self.fig_dir / name
        print(f"[report] OK  {name}")
        return path

    def _skip(self, name: str, reason: str) -> None:
        print(f"[report] SKIP {name}: {reason}")

    # ── EDA — tín hiệu ──────────────────────────────────────────────────────
    def fig_all_gestures(self) -> Path:
        gesture_ids = [l for l in self.labels_all if l.startswith("G")]
        data = {gid: seq for gid in gesture_ids
                if (seq := self._load_representative_sequence(gid)) is not None}
        path = self.fig_dir / "fig_all_gestures.png"
        plot_all_gestures(data, self.label_names, save_path=path)
        return self._ok("fig_all_gestures.png")

    def fig_gesture_detail(self, label_id: str = "G3") -> Path:
        seq = self._load_representative_sequence(label_id)
        if seq is None:
            self._skip(f"fig_gesture_detail_{label_id}.png", "khong co du lieu")
            return self.fig_dir / "missing.png"
        path = self.fig_dir / f"fig_gesture_detail_{label_id}.png"
        plot_gesture_detail(seq, label_id, self.label_names.get(label_id, ""), save_path=path)
        return self._ok(path.name)

    def fig_gesture_comparison(self, label_a: str = "G3", label_b: str = "G4") -> List[Path]:
        seq_a = self._load_representative_sequence(label_a)
        seq_b = self._load_representative_sequence(label_b)
        if seq_a is None or seq_b is None:
            self._skip(f"fig_gesture_comparison_{label_a}_{label_b}", "khong co du lieu")
            return []
        paths = []
        for ch_grp in ("accel", "gyro"):
            path = self.fig_dir / f"fig_gesture_comparison_{ch_grp}_{label_a}_{label_b}.png"
            plot_gesture_comparison(seq_a, seq_b, label_a, label_b,
                                    channel_group=ch_grp, save_path=path)
            paths.append(self._ok(path.name))
        return paths

    # ── EDA — thống kê ──────────────────────────────────────────────────────
    def fig_samples_per_gesture(self) -> Path:
        path = self.fig_dir / "fig_samples_per_gesture.png"
        plot_samples_per_gesture(self.manifest, self.label_names, save_path=path)
        return self._ok(path.name)

    def fig_samples_per_split(self) -> Path:
        path = self.fig_dir / "fig_samples_per_split.png"
        plot_samples_per_split(self.splits, save_path=path)
        return self._ok(path.name)

    def fig_samples_per_subject(self) -> Path:
        path = self.fig_dir / "fig_samples_per_subject.png"
        plot_samples_per_subject(self.manifest, save_path=path)
        return self._ok(path.name)

    def fig_imu_points_per_subject(self) -> Path:
        target_len = self.cfg["preprocessing"]["target_len"]
        path = self.fig_dir / "fig_imu_points_per_subject.png"
        plot_imu_points_per_subject(self.manifest, target_len=target_len, save_path=path)
        return self._ok(path.name)

    def fig_sequence_lengths(self) -> Path:
        path = self.fig_dir / "fig_sequence_lengths.png"
        plot_sequence_lengths(self.manifest, save_path=path)
        return self._ok(path.name)

    # ── Bảng EDA ────────────────────────────────────────────────────────────
    def tbl_csv_structure(self) -> Path:
        path = self.fig_dir / "tbl_csv_structure.png"
        table_csv_structure(save_path=path)
        return self._ok(path.name)

    def tbl_dataset_split(self) -> Path:
        split_cfg = self.cfg["split"]
        path = self.fig_dir / "tbl_dataset_split.png"
        table_dataset_split(
            self.splits,
            test_subjects=split_cfg["test_subjects"],
            val_subjects=split_cfg["val_subjects"],
            save_path=path,
        )
        return self._ok(path.name)

    def tbl_samples_imu_per_split(self) -> Path:
        target_len = self.cfg["preprocessing"]["target_len"]
        path = self.fig_dir / "tbl_samples_imu_per_split.png"
        table_samples_imu_per_split(self.splits, target_len=target_len, save_path=path)
        return self._ok(path.name)

    def tbl_subjects_stats(self) -> Path:
        target_len = self.cfg["preprocessing"]["target_len"]
        path = self.fig_dir / "tbl_subjects_stats.png"
        table_subjects_stats(self.manifest, target_len=target_len, save_path=path)
        return self._ok(path.name)

    def tbl_feature_stats(self, label_a: str = "G3", label_b: str = "G4") -> Path:
        seq_a = self._load_representative_sequence(label_a)
        seq_b = self._load_representative_sequence(label_b)
        if seq_a is None or seq_b is None:
            self._skip("tbl_feature_stats.png", "khong co du lieu")
            return self.fig_dir / "missing.png"
        path = self.fig_dir / f"tbl_feature_stats_{label_a}_{label_b}.png"
        table_feature_stats(seq_a, seq_b, label_a, label_b, save_path=path)
        return self._ok(path.name)

    # ── Kiến trúc ───────────────────────────────────────────────────────────
    def fig_architecture_cnn(self) -> Path:
        path = self.fig_dir / "fig_arch_cnn.png"
        plot_cnn_architecture(save_path=path)
        return self._ok(path.name)

    def fig_architecture_lstm(self) -> Path:
        path = self.fig_dir / "fig_arch_lstm.png"
        plot_lstm_architecture(save_path=path)
        return self._ok(path.name)

    def fig_architecture_transformer(self) -> Path:
        path = self.fig_dir / "fig_arch_transformer.png"
        plot_transformer_architecture(save_path=path)
        return self._ok(path.name)

    # ── Training figures ─────────────────────────────────────────────────────
    def fig_loss_acc_curves(self, model_name: str, history: Dict) -> Path:
        path = self.fig_dir / f"fig_loss_acc_{model_name}.png"
        plot_loss_acc_curves(history, model_name, save_path=path)
        return self._ok(path.name)

    def fig_loss_acc_from_file(self, model_name: str, history_json: Path) -> Path:
        history = json.loads(Path(history_json).read_text(encoding="utf-8"))
        return self.fig_loss_acc_curves(model_name, history)

    # ── Evaluation figures ───────────────────────────────────────────────────
    def fig_confusion_matrix(self, model_name: str, y_true, y_pred) -> Path:
        path = self.fig_dir / f"fig_confusion_{model_name}.png"
        plot_confusion_matrix(y_true, y_pred, self.labels_all,
                               model_name=model_name, save_path=path)
        return self._ok(path.name)

    def fig_confusion_comparison(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_confusion_comparison.png"
        plot_confusion_comparison(preds_by_model, self.labels_all, save_path=path)
        return self._ok(path.name)

    def fig_f1_comparison(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_f1_comparison.png"
        plot_f1_per_class(preds_by_model, self.labels_all, save_path=path)
        return self._ok(path.name)

    def fig_acc_f1_by_model(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_acc_f1_by_model.png"
        plot_acc_f1_by_model(preds_by_model, save_path=path)
        return self._ok(path.name)

    def fig_accuracy_per_model(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_accuracy_per_model.png"
        plot_accuracy_per_model(preds_by_model, save_path=path)
        return self._ok(path.name)

    def fig_f1_per_model(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_f1_per_model.png"
        plot_f1_per_model(preds_by_model, save_path=path)
        return self._ok(path.name)

    def fig_accuracy_per_class(self, preds_by_model) -> Path:
        path = self.fig_dir / "fig_accuracy_per_class.png"
        plot_accuracy_per_class(preds_by_model, self.labels_all, save_path=path)
        return self._ok(path.name)

    def fig_tsne(self, model_name: str, features, labels) -> Path:
        path = self.fig_dir / f"fig_tsne_{model_name}.png"
        plot_tsne(features, labels, self.labels_all, model_name=model_name, save_path=path)
        return self._ok(path.name)

    def fig_performance_table(self, results) -> Path:
        path = self.fig_dir / "tbl_model_performance.png"
        table_model_performance(results, save_path=path)
        return self._ok(path.name)

    def tbl_model_comparison(self, results) -> Path:
        path = self.fig_dir / "tbl_model_comparison.png"
        table_model_comparison(results, save_path=path)
        return self._ok(path.name)

    # ── Tập hợp ─────────────────────────────────────────────────────────────
    def generate_eda(self) -> List[Path]:
        """Sinh tất cả EDA figures và bảng (không cần kết quả training)."""
        paths = []
        _run = lambda fn, *a, **kw: self._try(fn, paths, *a, **kw)

        # Hình tín hiệu
        _run(self.fig_all_gestures)
        for gid in ["G3", "G4", "G8", "G14"]:
            _run(self.fig_gesture_detail, gid)
        _run(self.fig_gesture_comparison, "G3", "G4")

        # Hình thống kê
        _run(self.fig_samples_per_gesture)
        _run(self.fig_samples_per_split)
        _run(self.fig_samples_per_subject)
        _run(self.fig_imu_points_per_subject)
        _run(self.fig_sequence_lengths)

        # Bảng
        _run(self.tbl_csv_structure)
        _run(self.tbl_dataset_split)
        _run(self.tbl_samples_imu_per_split)
        _run(self.tbl_subjects_stats)
        _run(self.tbl_feature_stats, "G3", "G4")

        # Sơ đồ kiến trúc
        _run(self.fig_architecture_cnn)
        _run(self.fig_architecture_lstm)
        _run(self.fig_architecture_transformer)

        print(f"\n[report] EDA xong — {len(paths)} figures/tables → {self.fig_dir}")
        return paths

    def generate_eval(
        self,
        preds_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
        histories: Dict[str, Dict] | None = None,
        features_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
        performance_rows: List[Dict[str, Any]] | None = None,
    ) -> List[Path]:
        """Sinh tất cả evaluation figures (cần preds từ kết quả training)."""
        paths = []
        _run = lambda fn, *a, **kw: self._try(fn, paths, *a, **kw)

        if histories:
            for name, hist in histories.items():
                _run(self.fig_loss_acc_curves, name, hist)

        if preds_by_model:
            _run(self.fig_acc_f1_by_model, preds_by_model)
            _run(self.fig_accuracy_per_model, preds_by_model)
            _run(self.fig_f1_per_model, preds_by_model)
            _run(self.fig_accuracy_per_class, preds_by_model)
            _run(self.fig_f1_comparison, preds_by_model)
            _run(self.fig_confusion_comparison, preds_by_model)
            for name, (yt, yp) in preds_by_model.items():
                _run(self.fig_confusion_matrix, name, yt, yp)

        if features_by_model:
            for name, (feats, lbls) in features_by_model.items():
                _run(self.fig_tsne, name, feats, lbls)

        if performance_rows:
            _run(self.fig_performance_table, performance_rows)
            _run(self.tbl_model_comparison, performance_rows)

        print(f"\n[report] Eval xong — {len(paths)} figures/tables → {self.fig_dir}")
        return paths

    def generate_all(self, preds_by_model=None, histories=None,
                     features_by_model=None, performance_rows=None) -> List[Path]:
        paths = self.generate_eda()
        paths += self.generate_eval(
            preds_by_model or {},
            histories=histories,
            features_by_model=features_by_model,
            performance_rows=performance_rows,
        )
        print(f"\n[report] Tong cong {len(paths)} figures/tables → {self.fig_dir}")
        return paths

    def _try(self, fn, paths, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, list):
                paths.extend(result)
            elif result is not None:
                paths.append(result)
        except Exception as e:
            print(f"[report] ERR  {fn.__name__}: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = ReportGenerator()
    gen.generate_eda()
