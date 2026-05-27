"""FastAPI backend for the smart-home gesture demo.

Endpoints:
    GET  /health                  -> liveness
    GET  /labels                  -> full label metadata (from data_collection/labels.json)
    GET  /devices                 -> simulated device state
    GET  /serial/ports            -> available Serial ports
    POST /predict                 -> JSON { "samples": [[ax,ay,az,gx,gy,gz], ...] }
    POST /predict/csv             -> multipart upload of a trial CSV
    POST /predict/serial          -> collect one live Serial window and predict
    WS   /ws/serial               -> stream live Serial IMU samples and predictions
    POST /devices/{device}/{op}   -> manual device toggle (for UI testing)

Run:
    uvicorn demo.backend.app:app --reload --host 0.0.0.0 --port 8000

Configure via env vars:
    DEMO_RUN_DIR          -> path to trained run folder
    DEMO_MODEL            -> cnn | lstm | transformer | random_forest  (default: transformer)
    DEMO_LABELS           -> path to data_collection/labels.json (auto-detected)
    DEMO_TARGET_LEN       -> model input length (default: 201 for 50Hz)
    DEMO_SAMPLE_RATE_HZ   -> display/config metadata (default: 50)
    DEMO_WINDOW_SECONDS   -> live Serial capture window (default: 4.0)
    DEMO_CONFIDENCE_THRESHOLD -> minimum confidence for live device actions
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - pyserial is optional until live demo
    serial = None
    list_ports = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from predict import CHANNELS, DEFAULT_TARGET_LEN, GesturePredictor, canonical_model_name  # noqa: E402


SUPPORTED_MODELS = ("cnn", "lstm", "transformer", "random_forest")


def _find_latest_run(model: str) -> Optional[Path]:
    model = canonical_model_name(model)
    artifact = "model.joblib" if model == "random_forest" else "best.pt"
    if model == "random_forest":
        bases = [
            ROOT / "training_50hz_clean" / "baselines" / "random_forest",
            ROOT / "training" / "results_rf",
        ]
    else:
        bases = [
            ROOT / "training_50hz_clean" / "checkpoints" / f"{model}_100epoch",
            ROOT / "training_50hz_clean" / "checkpoints" / f"{model}_curve_100epoch",
            ROOT / "training" / f"results_{model}_100epoch",
            ROOT / "training_50hz_clean" / "checkpoints" / f"{model}_50epoch",
            ROOT / "training" / f"results_{model}_50epoch",
            ROOT / "training" / "results_full",
        ]
    runs = []
    for base in bases:
        if not base.exists():
            continue
        runs.extend(p for p in base.iterdir() if p.is_dir() and (p / artifact).exists())
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


MODEL_NAME = canonical_model_name(os.environ.get("DEMO_MODEL", "transformer"))
if MODEL_NAME not in SUPPORTED_MODELS:
    raise RuntimeError(f"Unknown DEMO_MODEL={MODEL_NAME}. Use one of: {', '.join(SUPPORTED_MODELS)}")
RUN_DIR_OVERRIDE = os.environ.get("DEMO_RUN_DIR")
RUN_DIR = ""
LABELS_FILE = os.environ.get(
    "DEMO_LABELS", str(ROOT / "data_collection" / "labels.json"))
TARGET_LEN = int(os.environ.get("DEMO_TARGET_LEN", str(DEFAULT_TARGET_LEN)))
SAMPLE_RATE_HZ = int(os.environ.get("DEMO_SAMPLE_RATE_HZ", "50"))
WINDOW_SECONDS = float(os.environ.get("DEMO_WINDOW_SECONDS", "4.0"))
PREDICT_INTERVAL_SECONDS = float(os.environ.get("DEMO_PREDICT_INTERVAL_SECONDS", "3.5"))
ACTION_COOLDOWN_SECONDS = float(os.environ.get("DEMO_ACTION_COOLDOWN_SECONDS", "3.5"))
CONFIDENCE_THRESHOLD = float(os.environ.get("DEMO_CONFIDENCE_THRESHOLD", "0.50"))
MIN_GYRO_RMS_DPS = float(os.environ.get("DEMO_MIN_GYRO_RMS_DPS", "6.0"))
MIN_ACCEL_DYNAMIC_G = float(os.environ.get("DEMO_MIN_ACCEL_DYNAMIC_G", "0.035"))
LIVE_CONTEXT_MARGIN = float(os.environ.get("DEMO_LIVE_CONTEXT_MARGIN", "0.62"))
CONTEXT_G2_MIN_CONFIDENCE = float(os.environ.get("DEMO_CONTEXT_G2_MIN_CONFIDENCE", "0.08"))
CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE = float(os.environ.get("DEMO_CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE", "0.005"))
GESTURE_CAPTURE_SECONDS = float(os.environ.get("DEMO_GESTURE_CAPTURE_SECONDS", "3.5"))
GESTURE_RECOVERY_SECONDS = float(os.environ.get("DEMO_GESTURE_RECOVERY_SECONDS", "0.8"))
STOP_CONFIRM_SECONDS = float(os.environ.get("DEMO_STOP_CONFIRM_SECONDS", "6.0"))
STOP_CONFIDENCE_THRESHOLD = float(os.environ.get("DEMO_STOP_CONFIDENCE_THRESHOLD", "0.995"))
TARGET_DEVICES = ["tv", "speaker", "light", "curtain"]
CONTROL_SEQUENCE = [
    {"label": "G1", "device": "system", "op": "system_start"},
    {"label": "G2", "device": "system", "op": "select_next_target"},
    {"label": "G3", "device": "tv", "op": "tv_channel_swing"},
    {"label": "G4", "device": "tv", "op": "tv_switch_source"},
    {"label": "G5", "device": "tv", "op": "tv_channel_up"},
    {"label": "G6", "device": "tv", "op": "tv_channel_down"},
    {"label": "G7", "device": "tv", "op": "tv_voice_search"},
    {"label": "G8", "device": "speaker", "op": "speaker_volume_up"},
    {"label": "G9", "device": "speaker", "op": "speaker_volume_down"},
    {"label": "G10", "device": "light", "op": "light_on"},
    {"label": "G11", "device": "light", "op": "light_off"},
    {"label": "G12", "device": "curtain", "op": "curtain_close"},
    {"label": "G13", "device": "curtain", "op": "curtain_open"},
    {"label": "G14", "device": "system", "op": "system_stop"},
    {"label": "G15", "device": "system", "op": "emergency_reset"},
]
DEVICE_SELECT_OPS = {
    "tv": "select_tv",
    "speaker": "select_speaker",
    "light": "select_light",
    "curtain": "select_curtain",
}
DEVICE_COMMANDS = {
    "tv": [
        {"label": "G3", "op": "tv_channel_swing"},
        {"label": "G4", "op": "tv_switch_source"},
        {"label": "G5", "op": "tv_channel_up"},
        {"label": "G6", "op": "tv_channel_down"},
        {"label": "G7", "op": "tv_voice_search"},
    ],
    "speaker": [
        {"label": "G8", "op": "speaker_volume_up"},
        {"label": "G9", "op": "speaker_volume_down"},
    ],
    "light": [
        {"label": "G10", "op": "light_on"},
        {"label": "G11", "op": "light_off"},
    ],
    "curtain": [
        {"label": "G12", "op": "curtain_close"},
        {"label": "G13", "op": "curtain_open"},
    ],
}
GESTURE_COMMANDS = {
    item["label"]: {"device": device, "op": item["op"]}
    for device, items in DEVICE_COMMANDS.items()
    for item in items
}
NOISE_LABELS = {"G16", "G17", "G18", "N1", "N2", "N3", "N4", "N5"}
DEVICE_DISPLAY_NAMES = {
    "tv": "Tivi",
    "light": "Đèn",
    "speaker": "Loa",
    "curtain": "Rèm",
    "fan": "Quạt",
    "ac": "Điều hòa",
    "system": "Hệ thống",
    "none": "Nhiễu",
}

def _load_predictor(model_name: str, run_dir: Optional[str] = None) -> tuple[str, GesturePredictor]:
    model_name = canonical_model_name(model_name)
    if model_name not in SUPPORTED_MODELS:
        raise RuntimeError(f"Unknown model: {model_name}")
    selected_run_dir = run_dir
    if selected_run_dir is None:
        found = _find_latest_run(model_name)
        if found is None:
            artifact = "model.joblib" if model_name == "random_forest" else "best.pt"
            raise RuntimeError(
                f"No trained {model_name} run with {artifact} found. "
                "Set DEMO_RUN_DIR explicitly or retrain/export that model.")
        selected_run_dir = str(found)
    loaded = GesturePredictor.load(
        selected_run_dir, model_name, labels_json_path=LABELS_FILE, target_len=TARGET_LEN)
    return str(selected_run_dir), loaded


RUN_DIR, predictor = _load_predictor(MODEL_NAME, RUN_DIR_OVERRIDE)
PREDICTOR_LOCK = threading.RLock()


# ---------- Simulated smart-home state ----------
DEVICES: Dict[str, Dict[str, Any]] = {
    "tv":      {"power": False, "channel": 1, "source": "HDMI1", "voice_search": False},
    "speaker": {"power": True, "volume": 30},
    "light":   {"power": False, "brightness": 80},
    "fan":     {"power": False, "speed": 1},
    "ac":      {"power": False, "temp": 26},
    "curtain": {"open": True},
    "system":  {
        "state": "Idle",
        "active": False,
        "selected_device": None,
        "sequence_index": 0,
        "expected_label": "G1",
        "select_count": 0,
        "device_index": -1,
        "current_command_index": 0,
        "device_command_count": 0,
    },
}
HISTORY: List[Dict[str, Any]] = []  # last N predictions/actions
MAX_HISTORY = 50
LAST_ACTION: Dict[str, Any] = {"label": None, "ts": 0.0}
STOP_PENDING: Dict[str, Any] = {"ts": 0.0, "label": None}


def device_display_name(device: Any) -> str:
    return DEVICE_DISPLAY_NAMES.get(str(device), str(device))


def build_action_message(action: Dict[str, Any]) -> Optional[str]:
    """Build the short user-facing notification read by the web TTS layer."""
    if not action.get("accepted"):
        return None

    op = str(action.get("op") or "")
    if op == "system_start":
        return "HỆ THỐNG ĐÃ ĐƯỢC MỞ"
    if op == "system_stop":
        return "ĐÃ TẮT HỆ THỐNG"
    if op == "emergency_reset":
        return "RESET LẠI HỆ THỐNG"
    if op == "select_tv":
        return "ĐÃ CHỌN TV"
    if op == "select_speaker":
        return "ĐÃ CHỌN LOA"
    if op == "select_light":
        return "CHỌN ĐÈN"
    if op == "select_curtain":
        return "CHỌN RÈM"
    if op == "sequence_complete":
        return "Đã hoàn thành toàn bộ kịch bản điều khiển."

    if op == "light_on":
        return "ĐÃ BẬT ĐÈN"
    if op == "light_off":
        return "ĐÃ TẮT ĐÈN"
    if op == "curtain_open":
        return "RÈM ĐÃ MỞ"
    if op == "curtain_close":
        return "RÈM ĐÃ ĐÓNG"

    if op == "speaker_volume_up":
        return "TĂNG ÂM LƯỢNG LOA"
    if op == "speaker_volume_down":
        return "GIẢM ÂM LƯỢNG LOA"

    if op == "tv_favorite":
        return "CHỌN KÊNH YÊU THÍCH"
    if op == "tv_channel_swing":
        return "ĐỔI KÊNH"
    if op == "tv_switch_source":
        return "CHỌN NGUỒN HDMI/TV"
    if op == "tv_channel_up":
        return "TĂNG KÊNH"
    if op == "tv_channel_down":
        return "GIẢM KÊNH"
    if op == "tv_voice_search":
        return "MỞ TÌM KIẾM BẰNG GIỌNG NÓI TRÊN YOUTUBE"

    return f"Đã thực hiện lệnh {op}."


def reset_home_state() -> None:
    STOP_PENDING.update({"ts": 0.0, "label": None})
    DEVICES["tv"].update({"power": False, "channel": 1, "source": "HDMI1", "voice_search": False})
    DEVICES["speaker"].update({"power": True, "volume": 30})
    DEVICES["light"].update({"power": False, "brightness": 80})
    DEVICES["fan"].update({"power": False, "speed": 1})
    DEVICES["ac"].update({"power": False, "temp": 26})
    DEVICES["curtain"].update({"open": True})
    DEVICES["system"].update({
        "state": "Idle",
        "active": False,
        "selected_device": None,
        "sequence_index": 0,
        "expected_label": "G1",
        "select_count": 0,
        "device_index": -1,
        "current_command_index": 0,
        "device_command_count": 0,
    })


def _current_device() -> Optional[str]:
    raw_index = DEVICES["system"].get("device_index")
    index = int(raw_index) if raw_index is not None else -1
    if index < 0 or index >= len(TARGET_DEVICES):
        return None
    return TARGET_DEVICES[index]


def _next_device_index() -> Optional[int]:
    raw_index = DEVICES["system"].get("device_index")
    index = (int(raw_index) if raw_index is not None else -1) + 1
    if index >= len(TARGET_DEVICES):
        return None
    return index


def _expected_step() -> Optional[Dict[str, str]]:
    system = DEVICES["system"]
    if not system.get("active"):
        return {"label": "G1", "device": "system", "op": "system_start"}

    device = _current_device()
    if device is None:
        first_device = TARGET_DEVICES[0]
        return {"label": "G2", "device": first_device, "op": DEVICE_SELECT_OPS[first_device]}

    command_index = int(system.get("current_command_index") or 0)
    commands = DEVICE_COMMANDS[device]
    if command_index < len(commands):
        command = commands[command_index]
        return {"label": command["label"], "device": device, "op": command["op"]}

    next_index = _next_device_index()
    if next_index is None:
        return None
    next_device = TARGET_DEVICES[next_index]
    return {"label": "G2", "device": next_device, "op": DEVICE_SELECT_OPS[next_device]}


def _set_expected_from_state() -> None:
    system = DEVICES["system"]
    step = _expected_step()
    system["expected_label"] = step["label"] if step else "DONE"


def _select_device(device_index: int) -> Dict[str, str]:
    system = DEVICES["system"]
    device = TARGET_DEVICES[device_index]
    op = DEVICE_SELECT_OPS[device]
    system.update({
        "state": "Selection Point",
        "selected_device": device,
        "device_index": device_index,
        "current_command_index": 0,
        "device_command_count": 0,
        "select_count": int(system.get("select_count") or 0) + 1,
        "sequence_index": int(system.get("sequence_index") or 0) + 1,
    })
    _apply_sequence_op(op)
    _set_expected_from_state()
    return {"label": "G2", "device": device, "op": op}


def _advance_current_command() -> None:
    system = DEVICES["system"]
    system["current_command_index"] = int(system.get("current_command_index") or 0) + 1
    system["device_command_count"] = int(system.get("device_command_count") or 0) + 1
    system["sequence_index"] = int(system.get("sequence_index") or 0) + 1
    _set_expected_from_state()


def _turn_off_device(device: str) -> None:
    if device == "tv":
        DEVICES["tv"].update({"power": False, "voice_search": False})
    elif device == "speaker":
        DEVICES["speaker"]["power"] = False
    elif device == "light":
        DEVICES["light"]["power"] = False
    elif device == "curtain":
        DEVICES["curtain"]["open"] = False


def _skip_to_next_device_selection() -> None:
    system = DEVICES["system"]
    device = _current_device()
    if device is None:
        _set_expected_from_state()
        return

    system["current_command_index"] = len(DEVICE_COMMANDS[device])
    system["device_command_count"] = max(1, int(system.get("device_command_count") or 0))
    system["sequence_index"] = int(system.get("sequence_index") or 0) + 1
    next_index = _next_device_index()
    if next_index is None:
        system.update({"state": "Completed", "expected_label": "DONE"})
        return

    system["state"] = "Selection Point"
    system["expected_label"] = "G2"


def _apply_sequence_op(op: str) -> None:
    if op == "select_tv":
        DEVICES["tv"].update({"voice_search": False})
    elif op == "tv_favorite":
        DEVICES["tv"].update({"power": True, "channel": 99, "voice_search": False})
    elif op == "tv_channel_swing":
        d = DEVICES["tv"]
        d["power"] = True
        d["channel"] = int(d.get("channel", 1)) + 1
        d["voice_search"] = False
    elif op == "tv_switch_source":
        d = DEVICES["tv"]
        d["power"] = True
        order = ["HDMI1", "HDMI2", "TV", "YouTube"]
        d["source"] = order[(order.index(d.get("source", "HDMI1")) + 1) % len(order)]
        d["voice_search"] = False
    elif op == "tv_channel_up":
        d = DEVICES["tv"]
        d["power"] = True
        d["channel"] = int(d.get("channel", 1)) + 1
        d["voice_search"] = False
    elif op == "tv_channel_down":
        d = DEVICES["tv"]
        d["power"] = True
        d["channel"] = max(1, int(d.get("channel", 1)) - 1)
        d["voice_search"] = False
    elif op == "tv_voice_search":
        DEVICES["tv"].update({"power": True, "source": "YouTube", "voice_search": True})
    elif op == "select_speaker":
        DEVICES["speaker"]["power"] = True
    elif op == "speaker_volume_up":
        d = DEVICES["speaker"]
        d["power"] = True
        d["volume"] = min(100, int(d.get("volume", 0)) + 5)
    elif op == "speaker_volume_down":
        d = DEVICES["speaker"]
        d["power"] = True
        d["volume"] = max(0, int(d.get("volume", 0)) - 5)
    elif op == "light_on":
        DEVICES["light"]["power"] = True
    elif op == "light_off":
        DEVICES["light"]["power"] = False
    elif op == "curtain_close":
        DEVICES["curtain"]["open"] = False
    elif op == "curtain_open":
        DEVICES["curtain"]["open"] = True


def apply_gesture(label_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a gesture to the simulated device state. Returns action summary."""
    global STOP_PENDING
    name = (meta.get("name") or "").lower()
    system = DEVICES["system"]
    selected = system.get("selected_device")
    is_stop_label = label_id == "G14" or "stop_palm" in name
    action: Dict[str, Any] = {
        "label": label_id,
        "device": meta.get("device") or "system",
        "name": meta.get("name"),
        "op": "noop",
        "accepted": False,
        "active": bool(system.get("active")),
        "selected_device": selected,
        "expected_label": "G2" if system.get("active") else "G1",
        "sequence_index": int(system.get("sequence_index") or 0),
    }

    if not is_stop_label:
        STOP_PENDING.update({"ts": 0.0, "label": None})

    if label_id in NOISE_LABELS or label_id.startswith("N"):
        action["device"] = "none"
        action["op"] = "ignored_idle" if not system.get("active") else "noise_rejected"
        return action

    if label_id == "G1" or "star_first" in name:
        if system.get("active"):
            action.update({
                "op": "active_start_rejected",
                "accepted": False,
                "active": True,
                "selected_device": selected,
                "expected_label": "G2",
                "sequence_index": int(system.get("sequence_index") or 0),
            })
            return action

        system.update({
            "state": "Active Mode",
            "active": True,
            "selected_device": "tv",
            "sequence_index": 0,
            "expected_label": "G2",
            "select_count": 0,
            "device_index": TARGET_DEVICES.index("tv"),
            "current_command_index": 0,
            "device_command_count": 0,
            "emergency": False,
        })
        action.update({
            "op": "system_start",
            "accepted": True,
            "active": True,
            "selected_device": system["selected_device"],
            "expected_label": system["expected_label"],
            "sequence_index": system["sequence_index"],
        })
        return action

    if is_stop_label:
        if not system.get("active"):
            STOP_PENDING.update({"ts": 0.0, "label": None})
            action.update({
                "op": "ignored_idle",
                "accepted": False,
                "active": False,
                "selected_device": None,
                "expected_label": "G1",
                "sequence_index": 0,
            })
            return action

        now = time.time()
        pending_age = now - float(STOP_PENDING.get("ts") or 0.0)
        if STOP_PENDING.get("label") != "G14" or pending_age > STOP_CONFIRM_SECONDS:
            STOP_PENDING.update({"ts": now, "label": "G14"})
            action.update({
                "op": "system_stop_pending",
                "accepted": False,
                "active": True,
                "selected_device": selected,
                "expected_label": "G14",
                "sequence_index": int(system.get("sequence_index") or 0),
                "confirm_within_seconds": STOP_CONFIRM_SECONDS,
            })
            return action

        STOP_PENDING.update({"ts": 0.0, "label": None})
        system.update({
            "state": "Exit Point",
            "active": False,
            "selected_device": None,
            "sequence_index": 0,
            "expected_label": "G1",
            "select_count": 0,
            "device_index": -1,
            "current_command_index": 0,
            "device_command_count": 0,
            "emergency": False,
        })
        action.update({
            "op": "system_stop",
            "accepted": True,
            "active": False,
            "selected_device": system.get("selected_device"),
            "expected_label": system.get("expected_label"),
            "sequence_index": system.get("sequence_index"),
        })
        return action

    if label_id == "G15" or "emergency" in name or "reset" in name:
        if not system.get("active"):
            action["op"] = "ignored_idle"
            action["expected_label"] = "G1"
            return action
        DEVICES["tv"]["voice_search"] = False
        system.update({
            "state": "Emergency",
            "active": True,
            "selected_device": selected or "tv",
            "expected_label": "G2",
            "emergency": True,
        })
        action.update({
            "op": "emergency_reset",
            "accepted": True,
            "active": True,
            "selected_device": system.get("selected_device"),
            "expected_label": "G2",
            "sequence_index": int(system.get("sequence_index") or 0),
            "clear_buffer": True,
            "stop_effects": True,
        })
        return action

    if not system.get("active"):
        action["op"] = "ignored_idle"
        action["expected_label"] = "G1"
        return action

    if label_id == "G2":
        current = system.get("selected_device")
        if current in TARGET_DEVICES:
            next_device = TARGET_DEVICES[(TARGET_DEVICES.index(current) + 1) % len(TARGET_DEVICES)]
        else:
            next_device = TARGET_DEVICES[0]
        system.update({
            "state": "Active Mode",
            "selected_device": next_device,
            "device_index": TARGET_DEVICES.index(next_device),
            "select_count": int(system.get("select_count") or 0) + 1,
            "sequence_index": int(system.get("sequence_index") or 0) + 1,
            "current_command_index": 0,
            "device_command_count": 0,
            "expected_label": "G2",
            "emergency": False,
        })
        op = DEVICE_SELECT_OPS[next_device]
        action.update({
            "op": op,
            "device": next_device,
            "accepted": True,
            "active": True,
            "selected_device": next_device,
            "expected_label": "G2",
            "select_count": system.get("select_count"),
            "sequence_index": int(system.get("sequence_index") or 0),
        })
        return action

    command = GESTURE_COMMANDS.get(label_id)
    if command is None:
        action.update({
            "op": "unexpected_sequence",
            "selected_device": system.get("selected_device"),
            "expected_label": "G2",
            "sequence_index": int(system.get("sequence_index") or 0),
        })
        return action

    selected = system.get("selected_device")
    if command["device"] != selected:
        action.update({
            "op": "unexpected_sequence",
            "device": command["device"],
            "expected_device": selected,
            "expected_label": "G2",
            "selected_device": selected,
            "sequence_index": int(system.get("sequence_index") or 0),
        })
        return action

    _apply_sequence_op(command["op"])
    system.update({
        "state": "Active Mode",
        "current_command_index": int(system.get("current_command_index") or 0) + 1,
        "device_command_count": int(system.get("device_command_count") or 0) + 1,
        "sequence_index": int(system.get("sequence_index") or 0) + 1,
        "expected_label": "G2",
        "emergency": False,
    })

    action.update({
        "op": command["op"],
        "device": command["device"],
        "accepted": True,
        "active": True,
        "selected_device": system.get("selected_device"),
        "select_count": system.get("select_count"),
        "expected_label": "G2",
        "sequence_index": int(system.get("sequence_index") or 0),
        "device_command_count": system.get("device_command_count"),
    })
    return action


# ---------- FastAPI app ----------
app = FastAPI(title="Smart Home Gesture Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    samples: List[List[float]] = Field(
        ..., description=f"Sequence of (T, 6) samples: {CHANNELS}")
    top_k: int = 3


class SerialPredictRequest(BaseModel):
    port: str = Field(..., description="Serial port, for example COM5")
    baud: int = 115200
    duration_s: float = WINDOW_SECONDS
    top_k: int = 3
    calibrate: bool = False


def _parse_serial_data_line(line: str) -> Optional[List[float]]:
    parts = [p.strip() for p in line.strip().split(",")]
    if not parts:
        return None

    values: List[str]
    if parts[0].upper() == "DATA":
        # Canonical firmware format:
        # DATA,millis_ms,sample_index,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c
        if len(parts) < 9:
            return None
        values = parts[3:9]
    else:
        # Fallback format from lightweight streamers:
        # ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
        if len(parts) != 6:
            return None
        values = parts

    try:
        return [float(v) for v in values]
    except ValueError:
        return None


def _require_pyserial() -> None:
    if serial is None:
        raise HTTPException(
            500,
            "pyserial is not installed. Run: "
            r"G:\DOAN2\.venv\Scripts\python.exe -m pip install -r G:\DOAN2\demo\requirements.txt",
        )


def _send_serial_command(ser: Any, command: str) -> None:
    ser.write((command.strip() + "\n").encode("ascii"))
    ser.flush()


def _read_serial_line(ser: Any) -> str:
    return ser.readline().decode("utf-8", errors="replace").strip()


def _collect_serial_samples(
    port: str,
    baud: int,
    duration_s: float,
    calibrate: bool = False,
) -> np.ndarray:
    _require_pyserial()
    samples: List[List[float]] = []
    try:
        with serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.2,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        ) as ser:
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            if calibrate:
                _send_serial_command(ser, "CALIBRATE")
                deadline = time.time() + 6.0
                while time.time() < deadline:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line.startswith("CALIBRATED"):
                        break

            _send_serial_command(ser, "START")
            deadline = time.time() + max(0.2, duration_s)
            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                parsed = _parse_serial_data_line(line)
                if parsed is not None:
                    samples.append(parsed)

            try:
                _send_serial_command(ser, "STOP")
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            500,
            f"Serial error on {port}: {exc}. Close Arduino Serial Monitor or other apps using this COM port.",
        ) from exc

    if not samples:
        raise HTTPException(400, f"No DATA samples received from {port}.")
    return np.asarray(samples, dtype=np.float32)


@app.get("/health")
def health() -> Dict[str, Any]:
    with PREDICTOR_LOCK:
        model_name = MODEL_NAME
        run_dir = RUN_DIR
        num_classes = len(predictor.id_to_label)
        target_len = predictor.target_len
    return {
        "ok": True,
        "model": model_name,
        "run_dir": run_dir,
        "num_classes": num_classes,
        "channels": CHANNELS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "predict_interval_seconds": PREDICT_INTERVAL_SECONDS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_gyro_rms_dps": MIN_GYRO_RMS_DPS,
        "min_accel_dynamic_g": MIN_ACCEL_DYNAMIC_G,
        "live_context_margin": LIVE_CONTEXT_MARGIN,
        "context_g2_min_confidence": CONTEXT_G2_MIN_CONFIDENCE,
        "context_g2_over_stop_min_confidence": CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE,
        "gesture_capture_seconds": GESTURE_CAPTURE_SECONDS,
        "gesture_recovery_seconds": GESTURE_RECOVERY_SECONDS,
        "stop_confirm_seconds": STOP_CONFIRM_SECONDS,
        "stop_confidence_threshold": STOP_CONFIDENCE_THRESHOLD,
        "control_flow": "target_state_machine_g2_cycle_g14_idle",
        "action_cooldown_seconds": ACTION_COOLDOWN_SECONDS,
        "target_len": target_len,
        "target_devices": TARGET_DEVICES,
        "control_sequence": CONTROL_SEQUENCE,
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    with PREDICTOR_LOCK:
        current_model = MODEL_NAME
        current_run_dir = RUN_DIR
        current_target_len = predictor.target_len
    items = []
    for name in SUPPORTED_MODELS:
        found = _find_latest_run(name)
        run_dir = str(found) if found else None
        if name == current_model:
            run_dir = current_run_dir
        items.append({
            "name": name,
            "available": bool(run_dir),
            "run_dir": run_dir,
            "active": name == current_model,
        })
    return {
        "current": current_model,
        "target_len": current_target_len,
        "models": items,
    }


@app.post("/models/{model_name}")
def select_model(model_name: str) -> Dict[str, Any]:
    global MODEL_NAME, RUN_DIR, predictor, LAST_ACTION
    model_name = canonical_model_name(model_name)
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")
    try:
        new_run_dir, new_predictor = _load_predictor(model_name)
    except RuntimeError as exc:
        raise HTTPException(404, str(exc)) from exc

    with PREDICTOR_LOCK:
        MODEL_NAME = model_name
        RUN_DIR = new_run_dir
        predictor = new_predictor
        LAST_ACTION = {"label": None, "ts": 0.0}
        STOP_PENDING.update({"ts": 0.0, "label": None})
        num_classes = len(predictor.id_to_label)
        target_len = predictor.target_len

    return {
        "ok": True,
        "model": model_name,
        "run_dir": new_run_dir,
        "num_classes": num_classes,
        "target_len": target_len,
    }


@app.get("/labels")
def labels() -> Dict[str, Any]:
    with PREDICTOR_LOCK:
        return {
            "id_to_label": predictor.id_to_label,
            "meta": predictor.label_meta,
        }


@app.get("/devices")
def get_devices() -> Dict[str, Any]:
    return {"devices": DEVICES, "history": HISTORY[-20:]}


@app.get("/serial/ports")
def serial_ports() -> Dict[str, Any]:
    if list_ports is None:
        return {"pyserial_available": False, "ports": []}
    ports = [
        {"device": p.device, "description": p.description}
        for p in list_ports.comports()
    ]
    return {"pyserial_available": True, "ports": ports}


@app.post("/devices/{device}/{op}")
def manual_op(device: str, op: str) -> Dict[str, Any]:
    if device not in DEVICES:
        raise HTTPException(404, f"Unknown device: {device}")

    op_key = f"{device}_{op}"
    aliases = {
        "tv_on": "select_tv",
        "tv_favorite": "tv_favorite",
        "tv_source": "tv_switch_source",
        "tv_channel_up": "tv_channel_up",
        "tv_channel_down": "tv_channel_down",
        "tv_voice_search": "tv_voice_search",
        "speaker_volume_up": "speaker_volume_up",
        "speaker_volume_down": "speaker_volume_down",
        "light_on": "light_on",
        "light_off": "light_off",
        "curtain_open": "curtain_open",
        "curtain_close": "curtain_close",
    }
    mapped_op = aliases.get(op_key)
    if mapped_op is None:
        raise HTTPException(400, f"Unsupported manual op: {device}/{op}")

    DEVICES["system"].update({"state": "Manual", "active": True, "selected_device": device})
    _apply_sequence_op(mapped_op)
    action = {
        "label": "manual",
        "device": device,
        "name": op_key,
        "op": mapped_op,
        "accepted": True,
        "active": True,
        "selected_device": device,
        "expected_label": DEVICES["system"].get("expected_label"),
        "sequence_index": DEVICES["system"].get("sequence_index"),
    }
    action["message"] = build_action_message(action)
    entry = {"ts": time.time(), "source": "manual", **action}
    HISTORY.append(entry); HISTORY[:] = HISTORY[-MAX_HISTORY:]
    return {"devices": DEVICES, "action": entry}


def _allowed_labels_for_state() -> set[str]:
    system = DEVICES["system"]
    if not system.get("active"):
        return {"G1", "G14", "G15"}

    allowed = {"G2", "G14", "G15"}
    selected = system.get("selected_device")
    if selected in DEVICE_COMMANDS:
        allowed.update(command["label"] for command in DEVICE_COMMANDS[str(selected)])
    return allowed


def _adjust_prediction_for_context(result: Dict[str, Any], confidence_threshold: float) -> Dict[str, Any]:
    """Prefer a plausible command in top-k when the raw top label is out of context."""
    adjusted = dict(result)
    top_k = list(result.get("top_k") or [])
    if not top_k:
        return adjusted

    raw_label = str(result.get("label") or "")
    raw_confidence = float(result.get("confidence") or 0.0)
    allowed = _allowed_labels_for_state()

    allowed_candidates = [item for item in top_k if str(item.get("label") or "") in allowed]
    if not allowed_candidates:
        adjusted["decision_rule"] = "raw_out_of_context"
        adjusted["allowed_labels"] = sorted(allowed)
        return adjusted

    g2_candidate = next((item for item in allowed_candidates if str(item.get("label") or "") == "G2"), None)
    if DEVICES["system"].get("active") and g2_candidate is not None:
        g2_confidence = float(g2_candidate.get("confidence") or 0.0)
        if raw_label == "G1" and g2_confidence >= CONTEXT_G2_MIN_CONFIDENCE:
            adjusted.update(g2_candidate)
            adjusted["raw_label"] = raw_label
            adjusted["raw_confidence"] = raw_confidence
            adjusted["decision_rule"] = "context_g2_over_active_g1"
            adjusted["allowed_labels"] = sorted(allowed)
            return adjusted
        if raw_label == "G14" and g2_confidence >= CONTEXT_G2_OVER_STOP_MIN_CONFIDENCE:
            adjusted.update(g2_candidate)
            adjusted["raw_label"] = raw_label
            adjusted["raw_confidence"] = raw_confidence
            adjusted["decision_rule"] = "context_g2_over_stop"
            adjusted["allowed_labels"] = sorted(allowed)
            return adjusted

    if raw_label in allowed or raw_label in {"G14", "G15"}:
        adjusted["decision_rule"] = "raw_top1"
        return adjusted

    best_allowed = max(allowed_candidates, key=lambda item: float(item.get("confidence") or 0.0))
    allowed_confidence = float(best_allowed.get("confidence") or 0.0)
    if str(best_allowed.get("label") or "") == "G2":
        min_context_confidence = CONTEXT_G2_MIN_CONFIDENCE
    else:
        min_context_confidence = max(0.22, min(confidence_threshold, raw_confidence * LIVE_CONTEXT_MARGIN))
    if allowed_confidence >= min_context_confidence:
        adjusted.update(best_allowed)
        adjusted["raw_label"] = raw_label
        adjusted["raw_confidence"] = raw_confidence
        adjusted["decision_rule"] = "context_topk"
        adjusted["allowed_labels"] = sorted(allowed)
    else:
        adjusted["decision_rule"] = "raw_low_context_confidence"
        adjusted["allowed_labels"] = sorted(allowed)
    return adjusted


def _motion_metrics(samples: np.ndarray) -> Dict[str, float]:
    if samples.size == 0:
        return {
            "gyro_rms_dps": 0.0,
            "gyro_peak_dps": 0.0,
            "accel_dynamic_g": 0.0,
            "motion_score": 0.0,
        }

    accel = samples[:, :3].astype(np.float32)
    gyro = samples[:, 3:].astype(np.float32)
    accel_dynamic = accel - np.median(accel, axis=0, keepdims=True)
    accel_norm = np.linalg.norm(accel_dynamic, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    gyro_rms = float(np.sqrt(np.mean(np.square(gyro_norm)))) if len(gyro_norm) else 0.0
    gyro_peak = float(np.max(gyro_norm)) if len(gyro_norm) else 0.0
    accel_rms = float(np.sqrt(np.mean(np.square(accel_norm)))) if len(accel_norm) else 0.0
    motion_score = max(
        gyro_rms / max(MIN_GYRO_RMS_DPS, 1e-6),
        accel_rms / max(MIN_ACCEL_DYNAMIC_G, 1e-6),
    )
    return {
        "gyro_rms_dps": gyro_rms,
        "gyro_peak_dps": gyro_peak,
        "accel_dynamic_g": accel_rms,
        "motion_score": float(motion_score),
    }


def _has_live_motion(metrics: Dict[str, float]) -> bool:
    return (
        metrics.get("gyro_rms_dps", 0.0) >= MIN_GYRO_RMS_DPS
        or metrics.get("accel_dynamic_g", 0.0) >= MIN_ACCEL_DYNAMIC_G
    )


def _slice_rows_by_time(
    rows: List[tuple[float, List[float]]],
    start_ts: float,
    end_ts: float,
) -> np.ndarray:
    selected = [row for ts, row in rows if start_ts <= ts <= end_ts]
    return np.asarray(selected, dtype=np.float32)


def _rows_since(
    rows: List[tuple[float, List[float]]],
    start_ts: float,
    end_ts: Optional[float] = None,
) -> List[tuple[float, List[float]]]:
    if end_ts is None:
        end_ts = rows[-1][0] if rows else start_ts
    return [(ts, row) for ts, row in rows if start_ts <= ts <= end_ts]


def _build_live_candidates(
    rows: List[tuple[float, List[float]]],
    window_s: float,
    min_samples: int,
) -> List[tuple[str, np.ndarray, Dict[str, Any]]]:
    if not rows:
        return []

    times = np.asarray([ts for ts, _ in rows], dtype=np.float64)
    samples = np.asarray([row for _, row in rows], dtype=np.float32)
    candidates: List[tuple[str, np.ndarray, Dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()

    def add_candidate(name: str, start_ts: float, end_ts: float) -> None:
        mask = (times >= start_ts) & (times <= end_ts)
        idx = np.flatnonzero(mask)
        if len(idx) < min_samples:
            return
        key = (int(idx[0]), int(idx[-1]))
        if key in seen:
            return
        seen.add(key)
        window = samples[idx[0]:idx[-1] + 1]
        metrics = _motion_metrics(window)
        candidates.append((
            name,
            window,
            {
                "start_offset_s": float(start_ts - times[0]),
                "end_offset_s": float(end_ts - times[0]),
                "samples": int(window.shape[0]),
                **metrics,
            },
        ))

    first_ts = float(times[0])
    last_ts = float(times[-1])
    add_candidate("full", first_ts, last_ts)

    for duration in (2.0, 3.0, window_s):
        duration = min(float(duration), max(0.5, last_ts - first_ts))
        add_candidate(f"recent_{duration:.1f}s", max(first_ts, last_ts - duration), last_ts)

    accel = samples[:, :3]
    gyro = samples[:, 3:]
    accel_dynamic = accel - np.median(accel, axis=0, keepdims=True)
    energy = np.linalg.norm(gyro, axis=1) / max(MIN_GYRO_RMS_DPS, 1e-6)
    energy += np.linalg.norm(accel_dynamic, axis=1) / max(MIN_ACCEL_DYNAMIC_G, 1e-6)
    peak_ts = float(times[int(np.argmax(energy))])
    for duration in (2.0, 3.0):
        half = duration / 2.0
        start_ts = max(first_ts, min(peak_ts - half, last_ts - duration))
        end_ts = min(last_ts, start_ts + duration)
        add_candidate(f"active_{duration:.1f}s", start_ts, end_ts)

    return candidates


def _score_live_result(result: Dict[str, Any], metrics: Dict[str, Any], confidence_threshold: float) -> float:
    adjusted = _adjust_prediction_for_context(result, confidence_threshold)
    label = str(adjusted.get("label") or "")
    confidence = float(adjusted.get("confidence") or 0.0)
    score = confidence
    if label in _allowed_labels_for_state():
        score += 0.08
    if label in NOISE_LABELS or label.startswith("N"):
        score -= 0.08
    score += min(float(metrics.get("motion_score") or 0.0), 3.0) * 0.015
    return score


def _run_live_predict(
    candidates: List[tuple[str, np.ndarray, Dict[str, Any]]],
    top_k: int,
    source: str,
    confidence_threshold: float,
    use_cooldown: bool,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("No live prediction candidates")

    with PREDICTOR_LOCK:
        active_predictor = predictor

    ranked = []
    for name, window, metrics in candidates:
        result = active_predictor.predict(window, top_k=top_k)
        score = _score_live_result(result, metrics, confidence_threshold)
        ranked.append((score, name, window, metrics, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, selected_name, selected_window, selected_metrics, selected_result = ranked[0]
    diagnostics = {
        "selected_candidate": selected_name,
        "selected_metrics": selected_metrics,
        "candidates": [
            {
                "name": name,
                "score": float(score),
                "label": result.get("label"),
                "confidence": result.get("confidence"),
                "samples": metrics.get("samples"),
                "gyro_rms_dps": metrics.get("gyro_rms_dps"),
                "accel_dynamic_g": metrics.get("accel_dynamic_g"),
            }
            for score, name, _, metrics, result in ranked[:5]
        ],
    }
    return _run_predict(
        selected_window,
        top_k=top_k,
        source=source,
        confidence_threshold=confidence_threshold,
        use_cooldown=use_cooldown,
        precomputed_result=selected_result,
        diagnostics=diagnostics,
    )


def _run_predict(
    samples: np.ndarray,
    top_k: int,
    source: str,
    confidence_threshold: float = 0.0,
    use_cooldown: bool = False,
    precomputed_result: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global LAST_ACTION
    with PREDICTOR_LOCK:
        active_model = MODEL_NAME
        active_predictor = predictor
        active_target_len = predictor.target_len

    result = precomputed_result if precomputed_result is not None else active_predictor.predict(samples, top_k=top_k)
    result = _adjust_prediction_for_context(result, confidence_threshold)
    meta = active_predictor.label_meta.get(result["label"], {})
    now = time.time()
    is_force_stop = result["label"] == "G15"

    if result["label"] == "G14" and result["confidence"] < STOP_CONFIDENCE_THRESHOLD:
        action = {
            "label": result["label"],
            "device": result.get("device"),
            "name": result.get("name"),
            "op": "system_stop_low_confidence",
            "accepted": False,
            "active": bool(DEVICES["system"].get("active")),
            "selected_device": DEVICES["system"].get("selected_device"),
            "expected_label": "G2" if DEVICES["system"].get("active") else "G1",
            "stop_confidence_threshold": STOP_CONFIDENCE_THRESHOLD,
        }
    elif result["confidence"] < confidence_threshold and not is_force_stop:
        action = {
            "label": result["label"],
            "device": result.get("device"),
            "name": result.get("name"),
            "op": "low_confidence",
            "accepted": False,
            "active": bool(DEVICES["system"].get("active")),
            "selected_device": DEVICES["system"].get("selected_device"),
        }
    elif not is_force_stop and use_cooldown and LAST_ACTION["label"] == result["label"] and now - LAST_ACTION["ts"] < ACTION_COOLDOWN_SECONDS:
        action = {
            "label": result["label"],
            "device": result.get("device"),
            "name": result.get("name"),
            "op": "debounced",
            "accepted": False,
            "active": bool(DEVICES["system"].get("active")),
            "selected_device": DEVICES["system"].get("selected_device"),
        }
    else:
        action = apply_gesture(result["label"], meta or {})
        if action.get("accepted"):
            LAST_ACTION = {"label": result["label"], "ts": now}

    action["message"] = build_action_message(action)
    history_top_k = [
        {
            "label": item.get("label"),
            "confidence": item.get("confidence"),
        }
        for item in result.get("top_k", [])
    ]
    diagnostic_entry: Dict[str, Any] = {}
    if diagnostics:
        selected_metrics = diagnostics.get("selected_metrics") or {}
        diagnostic_entry = {
            "candidate": diagnostics.get("selected_candidate"),
            "motion_score": selected_metrics.get("motion_score"),
            "gyro_rms_dps": selected_metrics.get("gyro_rms_dps"),
            "accel_dynamic_g": selected_metrics.get("accel_dynamic_g"),
        }

    entry = {
        "ts": now,
        "source": source,
        "confidence": result["confidence"],
        "top_k": history_top_k,
        "decision_rule": result.get("decision_rule"),
        "raw_label": result.get("raw_label"),
        "raw_confidence": result.get("raw_confidence"),
        **diagnostic_entry,
        **action,
    }
    HISTORY.append(entry); HISTORY[:] = HISTORY[-MAX_HISTORY:]
    payload = {
        "prediction": result,
        "action": entry,
        "devices": DEVICES,
        "received_samples": int(samples.shape[0]),
        "target_len": active_target_len,
        "model": active_model,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


@app.post("/predict")
def predict_json(req: PredictRequest) -> Dict[str, Any]:
    samples = np.asarray(req.samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != len(CHANNELS):
        raise HTTPException(400, f"samples must be (T, {len(CHANNELS)})")
    return _run_predict(samples, req.top_k, source="json")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data))
        missing = [c for c in CHANNELS if c not in df.columns]
        if missing:
            raise HTTPException(400, f"CSV missing columns: {missing}")
        samples = df[CHANNELS].to_numpy(dtype=np.float32)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Bad CSV: {exc}") from exc
    return _run_predict(samples, top_k=3, source=f"csv:{file.filename}")


@app.post("/predict/serial")
def predict_serial(req: SerialPredictRequest) -> Dict[str, Any]:
    samples = _collect_serial_samples(
        port=req.port,
        baud=req.baud,
        duration_s=req.duration_s,
        calibrate=req.calibrate,
    )
    return _run_predict(samples, req.top_k, source=f"serial:{req.port}")


@app.websocket("/ws/serial")
async def ws_serial(websocket: WebSocket) -> None:
    await websocket.accept()
    params = websocket.query_params
    port = (params.get("port") or "").strip()
    baud = int(params.get("baud") or "115200")
    window_s = float(params.get("window_s") or str(WINDOW_SECONDS))
    capture_s = float(params.get("capture_s") or params.get("predict_interval_s") or str(GESTURE_CAPTURE_SECONDS))
    capture_s = min(4.0, max(3.0, capture_s))
    retention_s = max(window_s, capture_s + 0.5)
    calibrate = (params.get("calibrate") or "false").lower() in {"1", "true", "yes", "on"}
    confidence = float(params.get("confidence") or str(CONFIDENCE_THRESHOLD))
    top_k = int(params.get("top_k") or "5")

    if not port:
        await websocket.send_json({"type": "error", "message": "Missing Serial port, for example COM5."})
        await websocket.close(code=1008)
        return
    if serial is None:
        await websocket.send_json({"type": "error", "message": "pyserial is not installed."})
        await websocket.close(code=1011)
        return

    ser = None
    buffer: Deque[tuple[float, List[float]]] = deque()
    sample_index = 0
    last_idle_status_at = 0.0
    last_collect_status_at = 0.0
    gesture_started_at: Optional[float] = None
    suppress_until = 0.0
    try:
        await websocket.send_json({"type": "status", "message": f"Opening {port}..."})
        ser = await asyncio.to_thread(
            serial.Serial,
            port=port,
            baudrate=baud,
            timeout=0.05,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        )
        await asyncio.sleep(2.0)
        await asyncio.to_thread(ser.reset_input_buffer)
        await asyncio.to_thread(ser.reset_output_buffer)

        if calibrate:
            await websocket.send_json({"type": "status", "message": "Calibrating gyro offset..."})
            await asyncio.to_thread(_send_serial_command, ser, "CALIBRATE")
            deadline = time.time() + 6.0
            while time.time() < deadline:
                line = await asyncio.to_thread(_read_serial_line, ser)
                if line.startswith("CALIBRATED"):
                    await websocket.send_json({"type": "status", "message": line})
                    break
                await asyncio.sleep(0)

        await asyncio.to_thread(_send_serial_command, ser, "START")
        with PREDICTOR_LOCK:
            active_model = MODEL_NAME
            active_run_dir = RUN_DIR
            active_target_len = predictor.target_len
        await websocket.send_json({
            "type": "status",
            "message": "Streaming",
            "channels": CHANNELS,
            "window_s": window_s,
            "capture_s": capture_s,
            "target_len": active_target_len,
            "model": active_model,
            "run_dir": active_run_dir,
        })

        while True:
            line = await asyncio.to_thread(_read_serial_line, ser)
            parsed = _parse_serial_data_line(line)
            now = time.time()
            if parsed is None:
                await asyncio.sleep(0.005)
                continue

            sample_index += 1
            buffer.append((now, parsed))
            while buffer and now - buffer[0][0] > retention_s:
                buffer.popleft()

            sample = dict(zip(CHANNELS, parsed))
            await websocket.send_json({
                "type": "sample",
                "ts": now,
                "index": sample_index,
                "sample": sample,
                "buffer_samples": len(buffer),
            })

            rows = list(buffer)
            recent_rows = _rows_since(rows, now - 0.8, now)
            recent_values = recent_rows if len(recent_rows) >= 5 else rows
            recent_window = np.asarray([row for _, row in recent_values], dtype=np.float32)
            recent_motion = _motion_metrics(recent_window)
            is_moving = _has_live_motion(recent_motion)

            if gesture_started_at is None:
                if now < suppress_until:
                    continue
                if is_moving:
                    gesture_started_at = max(rows[0][0], now - 0.35)
                    last_collect_status_at = now
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Detected motion - collecting {capture_s:.1f}s",
                        "motion": recent_motion,
                        "capture_s": capture_s,
                    })
                elif now - last_idle_status_at >= 1.0:
                    last_idle_status_at = now
                    await websocket.send_json({
                        "type": "status",
                        "message": "Streaming - waiting for gesture",
                        "motion": recent_motion,
                    })
                continue

            elapsed_s = now - gesture_started_at
            if now - last_collect_status_at >= 0.8:
                last_collect_status_at = now
                await websocket.send_json({
                    "type": "status",
                    "message": f"Collecting gesture {min(elapsed_s, capture_s):.1f}/{capture_s:.1f}s",
                    "motion": recent_motion,
                    "capture_s": capture_s,
                })
            if elapsed_s < capture_s:
                continue

            prediction_rows = _rows_since(rows, gesture_started_at, now)
            with PREDICTOR_LOCK:
                active_target_len = predictor.target_len
            if len(prediction_rows) < max(25, active_target_len // 4):
                gesture_started_at = None
                continue

            window = np.asarray([row for _, row in prediction_rows], dtype=np.float32)
            motion = _motion_metrics(window)
            if not _has_live_motion(motion):
                gesture_started_at = None
                suppress_until = now + 0.4
                continue

            candidates = _build_live_candidates(
                prediction_rows,
                window_s=min(retention_s, capture_s + 0.5),
                min_samples=max(25, active_target_len // 4),
            )
            gesture_started_at = None
            suppress_until = now + GESTURE_RECOVERY_SECONDS
            buffer.clear()
            if not candidates:
                continue
            payload = await asyncio.to_thread(
                _run_live_predict,
                candidates,
                top_k,
                f"ws:{port}",
                confidence,
                True,
            )
            payload["type"] = "prediction"
            payload["capture_seconds"] = capture_s
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if ser is not None:
            try:
                await asyncio.to_thread(_send_serial_command, ser, "STOP")
            except Exception:
                pass
            try:
                await asyncio.to_thread(ser.close)
            except Exception:
                pass


# ---------- Static web demo ----------
WEB_DIR = ROOT / "demo" / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
