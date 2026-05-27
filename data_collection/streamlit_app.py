import csv
import io
import json
import math
import random
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "labels.json"
RAW_DIR = ROOT / "data" / "raw"
BACKUP_DIR = ROOT / "data" / "backup"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
SAMPLE_RATE_HZ = 100

CSV_COLUMNS = [
    "time",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
    "label",
]

MANIFEST_COLUMNS = [
    "session_id",
    "subject_id",
    "label_id",
    "label_name",
    "trial",
    "duration_s",
    "samples",
    "path",
    "recorded_at",
]

SIGNAL_COLORS = {
    "ax_g": "#ff1744",
    "ay_g": "#0b8d31",
    "az_g": "#4f21ff",
    "gx_dps": "#ff9f1c",
    "gy_dps": "#8a13ba",
    "gz_dps": "#b12a34",
}

SIGNAL_LABELS = {
    "ax_g": "Acc X",
    "ay_g": "Acc Y",
    "az_g": "Acc Z",
    "gx_dps": "Gyro X",
    "gy_dps": "Gyro Y",
    "gz_dps": "Gyro Z",
}


PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "signal_chart",
        "height": 900,
        "width": 1600,
        "scale": 2,
    },
}


@st.cache_data
def load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["labels"]


def load_existing_dataset_files():
    files = []
    seen = set()

    if MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size > 0:
        try:
            manifest = pd.read_csv(MANIFEST_PATH)
        except pd.errors.EmptyDataError:
            manifest = pd.DataFrame()

        for _, row in manifest.tail(200).iloc[::-1].iterrows():
            raw_path = ROOT / str(row.get("path", ""))
            if not raw_path.exists():
                continue
            file_name = raw_path.name
            seen.add(file_name)
            files.append(
                {
                    "file": file_name,
                    "path": str(raw_path.relative_to(ROOT)),
                    "label": str(row.get("label_id", "")),
                    "size_kb": round(raw_path.stat().st_size / 1024, 2),
                    "samples": int(row.get("samples", 0) or 0),
                    "collected_at": str(row.get("recorded_at", "")),
                }
            )

    if RAW_DIR.exists():
        for raw_path in sorted(RAW_DIR.rglob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True):
            if raw_path.name in seen:
                continue
            label = ""
            samples = 0
            try:
                frame = pd.read_csv(raw_path, usecols=["label"])
                samples = len(frame)
                if samples:
                    label = str(frame["label"].iloc[0])
            except Exception:
                parts = raw_path.stem.split("_")
                label = next((part for part in parts if part.startswith(("G", "N"))), "")

            files.append(
                {
                    "file": raw_path.name,
                    "path": str(raw_path.relative_to(ROOT)),
                    "label": label,
                    "size_kb": round(raw_path.stat().st_size / 1024, 2),
                    "samples": samples,
                    "collected_at": datetime.fromtimestamp(raw_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
            if len(files) >= 200:
                break

    return files[:200]


def init_state():
    defaults = {
        "trial_counters": {},
        "current_rows": [],
        "latest_csv": "",
        "latest_file_name": "None",
        "status": "Gesture Data: waiting",
        "collecting": False,
        "worker_state": None,
        "worker_thread": None,
        "worker_stop": None,
        "worker_lock": None,
        "serial_conn": None,
        "serial_port_name": "",
        "serial_connected": False,
        "serial_message": "",
        "serial_debug_lines": [],
        "latest_backup": "",
    }
    if "files" not in st.session_state:
        st.session_state.files = load_existing_dataset_files()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_style():
    st.markdown(
        """
        <style>
          :root {
            --bg: #f7f3ec;
            --panel: #ffffff;
            --ink: #17313a;
            --muted: #69757d;
            --line: #d8dde3;
            --accent: #ff4758;
            --accent-strong: #e51c33;
            --blue: #2f67c7;
            --shadow: 0 12px 28px rgba(31, 42, 55, 0.08);
          }

          .stApp {
            color: var(--ink);
            background:
              radial-gradient(circle at 15% 10%, rgba(255, 241, 215, 0.85), transparent 34%),
              radial-gradient(circle at 84% 5%, rgba(220, 242, 238, 0.8), transparent 32%),
              var(--bg);
          }

          .block-container,
          main .block-container,
          div[data-testid="stAppViewContainer"] .block-container {
            width: 100%;
            max-width: none;
            padding: 24px 18px 18px;
          }

          section.main > div {
            max-width: none;
          }

          .title {
            display: block;
            width: 100%;
            margin: 0 0 18px;
            padding: 4px 0 2px;
            text-align: center;
            color: var(--ink);
            font-family: Arial, Helvetica, sans-serif;
            font-size: clamp(26px, 2.7vw, 38px);
            line-height: 1.15;
            letter-spacing: 0;
            font-weight: 800;
            overflow: visible;
          }

          .panel-title {
            margin: 0 0 10px;
            color: #1f2937;
            font-size: 21px;
            line-height: 1.15;
            font-weight: 700;
          }

          .section-title {
            margin: 0 0 8px;
            color: #202b34;
            font-size: 22px;
            line-height: 1.15;
            font-weight: 700;
          }

          .main-meta {
            display: grid;
            gap: 4px;
            margin: 0 0 10px;
            color: #17252e;
            font-size: 14px;
            font-weight: 700;
          }

          .status {
            min-height: 20px;
            margin: 0 0 6px 10px;
            color: var(--accent-strong);
            font-size: 16px;
            font-weight: 800;
          }

          .gesture-title {
            margin: 0 0 8px;
            color: var(--accent-strong);
            font-size: 22px;
            line-height: 1.15;
            font-weight: 900;
            letter-spacing: 0;
          }

          .chart-health {
            display: flex;
            gap: 14px;
            margin: -2px 0 6px 10px;
            color: #53606a;
            font-size: 12px;
            font-weight: 700;
          }

          .chart-health strong {
            color: #17252e;
          }

          .muted {
            color: #8a949c;
            font-size: 13px;
            margin: 4px 0 8px;
          }

          .serial-help {
            margin: -2px 0 8px;
            padding: 8px 10px;
            border: 1px solid #d8dde3;
            border-radius: 8px;
            color: #33424b;
            background: rgba(255, 255, 255, 0.72);
            font-size: 12px;
            line-height: 1.45;
          }

          .divider {
            height: 1px;
            margin: 14px 0;
            background: #d8d2c8;
          }

          .env p {
            margin: 4px 0 0;
            color: #17313a;
            font-size: 14px;
          }

          .summary-card {
            display: grid;
            gap: 10px;
            padding: 14px 14px;
            border: 1px solid #d9e0e6;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: var(--shadow);
          }

          .metric {
            display: grid;
            grid-template-columns: 28px 1fr auto;
            gap: 8px;
            align-items: center;
            color: #17252e;
            font-size: 13px;
            font-weight: 700;
          }

          .metric-icon {
            display: grid;
            width: 22px;
            height: 22px;
            place-items: center;
            border-radius: 6px;
            color: #fff;
            background: var(--blue);
            font-size: 12px;
          }

          .metric-value {
            color: #17252e;
            font-weight: 800;
          }

          .stButton > button,
          .stDownloadButton > button {
            width: 100%;
            min-height: 36px;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: #273640;
            background: #fff;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          .stButton > button[kind="primary"] {
            border-color: var(--accent);
            color: #fff;
            background: var(--accent);
            font-weight: 700;
          }

          .stButton > button[kind="primary"]:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
          }

          div[data-testid="stMetricValue"],
          div[data-testid="stMetricLabel"] {
            color: #17252e;
          }

          div[data-testid="stDataFrame"] {
            border: 1px solid #d9e0e6;
            border-radius: 8px;
            background: #fff;
          }

          .stCodeBlock {
            border-radius: 8px;
          }

          div[data-testid="stVerticalBlock"] {
            gap: 0.45rem;
          }

          div[data-testid="stHorizontalBlock"] {
            gap: 0.6rem;
          }

          div[data-testid="stWidgetLabel"] > label {
            min-height: 1rem;
            font-size: 0.82rem;
          }

          div[data-baseweb="select"] > div,
          input {
            min-height: 34px;
          }

          .stSlider {
            padding-top: 0;
            padding-bottom: 0;
          }

          div[data-testid="stCheckbox"] {
            margin-top: -4px;
          }

          @media (max-width: 780px) {
            .block-container {
              padding-left: 12px;
              padding-right: 12px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def source_display(source_key):
    return "ESP32 Serial" if source_key == "serial" else "Computer (mock)"


def label_display(label):
    return f"{label['id']} - {label['name']}"


def format_timestamp(date_time):
    return date_time.strftime("%Y%m%d_%H%M%S")


def next_trial(subject_id, label_id):
    key = f"{subject_id}_{label_id}"
    st.session_state.trial_counters[key] = st.session_state.trial_counters.get(key, 0) + 1
    return st.session_state.trial_counters[key]


def reserve_trials(subject_id, label_id, repeats):
    key = f"{subject_id}_{label_id}"
    first_trial = st.session_state.trial_counters.get(key, 0) + 1
    st.session_state.trial_counters[key] = first_trial + repeats - 1
    return first_trial


def build_file_name(session_id, subject_id, label_id, trial):
    safe_subject = "_".join((subject_id or "S01").split())
    return f"{session_id}_{safe_subject}_{label_id}_T{trial:03d}.csv"


def build_output_path(session_id, subject_id, label_id, trial):
    safe_subject = "_".join((subject_id or "S01").split())
    safe_label = "_".join((label_id or "G1").split())
    label_dir = RAW_DIR / safe_subject / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / build_file_name(session_id, safe_subject, safe_label, trial)


def backup_file_name(subject_id, label_id, session_id):
    safe_subject = "_".join((subject_id or "S01").split())
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"backup_{created_at}_{safe_subject}_{label_id}_{session_id}.zip"


def parse_data_line(line):
    parts = line.strip().split(",")
    if len(parts) != 10 or parts[0] != "DATA":
        return None
    try:
        return {
            "millis_ms": parts[1],
            "sample_index": parts[2],
            "ax_g": float(parts[3]),
            "ay_g": float(parts[4]),
            "az_g": float(parts[5]),
            "gx_dps": float(parts[6]),
            "gy_dps": float(parts[7]),
            "gz_dps": float(parts[8]),
            "temp_c": float(parts[9]),
        }
    except ValueError:
        return None


def label_seed(label_id):
    prefix = label_id[:1].upper()
    try:
        number = int(label_id[1:])
    except ValueError:
        number = 1
    return number + 15 if prefix == "N" else number


def synthetic_sample(index, label_id):
    t = index / SAMPLE_RATE_HZ
    gesture_seed = label_seed(label_id)
    motion = math.sin(t * (2.8 + gesture_seed * 0.08)) * (0.65 if gesture_seed <= 15 else 0.25)
    noise = lambda: random.random() - 0.5
    return {
        "millis_ms": str(round(t * 1000)),
        "sample_index": str(index),
        "ax_g": motion + noise() * 0.22,
        "ay_g": math.cos(t * 2.2) * 0.35 + noise() * 0.22,
        "az_g": 9.8 + math.sin(t * 4.1) * 0.25 + noise() * 0.35,
        "gx_dps": math.sin(t * 7.1) * 12 + noise() * 22,
        "gy_dps": math.cos(t * 6.4) * 10 + noise() * 22,
        "gz_dps": math.sin(t * 5.7 + 1.2) * 8 + noise() * 22,
        "temp_c": 31.5 + noise() * 0.2,
    }


def make_row(sample, start_time, label, subject_id, trial, session_id):
    elapsed = max(0.0, time.perf_counter() - start_time)
    return {
        "time": f"{elapsed:.4f}",
        "pc_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": f"{elapsed:.4f}",
        "millis_ms": sample["millis_ms"],
        "sample_index": sample["sample_index"],
        "ax_g": f"{float(sample['ax_g']):.5f}",
        "ay_g": f"{float(sample['ay_g']):.5f}",
        "az_g": f"{float(sample['az_g']):.5f}",
        "gx_dps": f"{float(sample['gx_dps']):.5f}",
        "gy_dps": f"{float(sample['gy_dps']):.5f}",
        "gz_dps": f"{float(sample['gz_dps']):.5f}",
        "temp_c": f"{float(sample['temp_c']):.2f}",
        "label": label["id"],
        "label_id": label["id"],
        "label_name": label["name"],
        "label_group": label["group"],
        "device": label["device"],
        "system_state": label["system_state"],
        "subject_id": subject_id,
        "trial": trial,
        "session_id": session_id,
    }


def rows_to_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_csv_file(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(row):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size > 0
    with MANIFEST_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_trial_file(rows, label, subject_id, trial, session_id, duration_sec):
    output_path = build_output_path(session_id, subject_id, label["id"], trial)
    write_csv_file(output_path, rows)
    append_manifest(
        {
            "session_id": session_id,
            "subject_id": subject_id,
            "label_id": label["id"],
            "label_name": label["name"],
            "trial": trial,
            "duration_s": duration_sec,
            "samples": len(rows),
            "path": str(output_path.relative_to(ROOT)),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    csv_text = rows_to_csv(rows)
    return {
        "file": output_path.name,
        "path": str(output_path.relative_to(ROOT)),
        "label": label["id"],
        "size_kb": round(len(csv_text.encode("utf-8")) / 1024, 2),
        "samples": len(rows),
        "csv_text": csv_text,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }


def create_backup_zip(saved_files, subject_id, label_id, session_id):
    if not saved_files:
        return ""

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / backup_file_name(subject_id, label_id, session_id)
    safe_subject = "_".join((subject_id or "S01").split())
    safe_label = "_".join((label_id or "G1").split())

    with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if MANIFEST_PATH.exists():
            archive.write(MANIFEST_PATH, MANIFEST_PATH.relative_to(ROOT))

        for file_record in saved_files:
            relative_path = file_record.get("path", "")
            if not relative_path:
                csv_text = file_record.get("csv_text", "")
                if csv_text:
                    archive_name = Path("data") / "raw" / safe_subject / safe_label / file_record["file"]
                    archive.writestr(archive_name.as_posix(), csv_text)
                continue
            raw_path = ROOT / relative_path
            if raw_path.exists():
                archive.write(raw_path, raw_path.relative_to(ROOT))

    return str(backup_path.relative_to(ROOT))


def record_file(file_name, label_id, csv_text, path="", samples=0):
    size_kb = len(csv_text.encode("utf-8")) / 1024
    st.session_state.files.insert(
        0,
        {
            "file": file_name,
            "path": path,
            "label": label_id,
            "size_kb": round(size_kb, 2),
            "samples": samples,
            "collected_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    st.session_state.files = st.session_state.files[:200]


def record_saved_file(file_record):
    st.session_state.files.insert(
        0,
        {
            "file": file_record["file"],
            "path": file_record.get("path", ""),
            "label": file_record["label"],
            "size_kb": file_record["size_kb"],
            "samples": file_record.get("samples", 0),
            "collected_at": file_record.get("collected_at", datetime.now().isoformat(timespec="seconds")),
        },
    )
    st.session_state.files = st.session_state.files[:200]


def available_ports():
    if list_ports is None:
        return []
    return list(list_ports.comports())


def serial_error_types():
    if serial is None:
        return (OSError,)
    return (serial.SerialException, OSError)


def serial_port_name(conn=None, fallback=""):
    if fallback:
        return fallback
    if conn is not None:
        port = getattr(conn, "port", "")
        if port:
            return port
    return st.session_state.get("serial_port_name") or "COM"


def serial_failure_message(action, port_name, exc):
    return (
        f"Loi Serial khi {action} tren {port_name}: {exc}. "
        "Cach xu ly: bam Close, dong Arduino Serial Monitor/Serial Plotter hoac app khac dang dung COM, "
        "rut cam lai USB ESP32, chon lai dung cong COM, sau do Connect va Ping lai."
    )


def close_serial_connection(conn):
    if conn is None:
        return
    try:
        if getattr(conn, "is_open", False):
            conn.close()
    except serial_error_types():
        pass


def reset_serial_buffers(conn, action="reset bo dem"):
    port_name = serial_port_name(conn)
    try:
        conn.reset_input_buffer()
        conn.reset_output_buffer()
    except serial_error_types() as exc:
        close_serial_connection(conn)
        raise RuntimeError(serial_failure_message(action, port_name, exc)) from exc


def close_serial():
    conn = st.session_state.get("serial_conn")
    close_serial_connection(conn)
    st.session_state.serial_conn = None
    st.session_state.serial_connected = False
    st.session_state.serial_port_name = ""


def ensure_serial_connection(port_name, baud=115200):
    if serial is None:
        raise RuntimeError("Thieu thu vien pyserial. Cai bang lenh: python -m pip install -r requirements.txt")
    if not port_name:
        raise RuntimeError("Chua chon cong Serial.")

    conn = st.session_state.get("serial_conn")
    if conn is not None and conn.is_open and st.session_state.serial_port_name == port_name:
        return conn

    close_serial()
    try:
        conn = serial.Serial(
            port=port_name,
            baudrate=baud,
            timeout=0.25,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        )
    except serial_error_types() as exc:
        raise RuntimeError(serial_failure_message("mo cong", port_name, exc)) from exc
    time.sleep(2.0)
    reset_serial_buffers(conn)
    st.session_state.serial_conn = conn
    st.session_state.serial_port_name = port_name
    st.session_state.serial_connected = True
    return conn


def send_serial(conn, command):
    if conn is None or not getattr(conn, "is_open", False):
        raise RuntimeError("Chua ket noi ESP32 Serial.")

    port_name = serial_port_name(conn)
    try:
        conn.write((command.strip() + "\n").encode("ascii"))
        conn.flush()
    except serial_error_types() as exc:
        close_serial_connection(conn)
        raise RuntimeError(serial_failure_message(f"gui lenh {command.strip()}", port_name, exc)) from exc


def read_until_quiet(conn, seconds=1.0):
    end_at = time.time() + seconds
    lines = []
    while time.time() < end_at:
        try:
            raw = conn.readline()
        except serial_error_types() as exc:
            close_serial_connection(conn)
            raise RuntimeError(serial_failure_message("doc du lieu", serial_port_name(conn), exc)) from exc
        if raw:
            lines.append(raw.decode("utf-8", errors="replace").strip())
    return lines


def test_serial_stream(conn, seconds=2.0):
    reset_serial_buffers(conn)
    send_serial(conn, "START")
    lines = ["APP_SENT,START"]
    data_count = 0
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                raw = conn.readline()
            except serial_error_types() as exc:
                close_serial_connection(conn)
                raise RuntimeError(serial_failure_message("doc DATA", serial_port_name(conn), exc)) from exc
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
                if parse_data_line(line) is not None:
                    data_count += 1
    finally:
        try:
            send_serial(conn, "STOP")
            lines.append("APP_SENT,STOP")
        except Exception as exc:
            lines.append(f"APP_STOP_FAILED,{exc}")
            pass
    return data_count, lines[-20:]


def update_worker(worker_state, lock, **updates):
    with lock:
        worker_state.update(updates)


def remember_serial_line(worker_state, lock, line):
    if not line:
        return
    with lock:
        lines = worker_state.setdefault("serial_debug_lines", [])
        lines.append(line)
        del lines[:-12]


def append_worker_row(worker_state, lock, row):
    with lock:
        worker_state["rows"].append(row)


def collection_worker(
    worker_state,
    lock,
    stop_event,
    source_key,
    serial_conn,
    label,
    subject_id,
    duration_sec,
    baseline_sec,
    first_trial,
    repeats,
    rest_sec,
    save_to_disk,
    session_id,
):
    try:
        update_worker(
            worker_state,
            lock,
            rows=[],
            latest_csv="",
            latest_file_name="None",
            saved_files=[],
            serial_debug_lines=[],
            progress=0.0,
            status="Gesture Data: waiting",
            done=False,
            stopped=False,
            error=None,
        )

        for repeat_index in range(repeats):
            trial = first_trial + repeat_index
            file_name = build_file_name(session_id, subject_id, label["id"], trial)
            update_worker(
                worker_state,
                lock,
                rows=[],
                latest_csv="",
                latest_file_name=file_name,
                current_repeat=repeat_index + 1,
                total_repeats=repeats,
                status=f"Trial {repeat_index + 1}/{repeats}: prepare {label['id']} - {label['name']}",
            )

            for remaining in range(int(baseline_sec), 0, -1):
                if stop_event.is_set():
                    update_worker(worker_state, lock, status="Collection stopped.", stopped=True, done=True)
                    return
                overall = repeat_index / max(1, repeats)
                update_worker(
                    worker_state,
                    lock,
                    status=f"Trial {repeat_index + 1}/{repeats}: Baseline {remaining}s",
                    progress=overall,
                )
                time.sleep(1.0)

            if stop_event.is_set():
                update_worker(worker_state, lock, status="Collection stopped.", stopped=True, done=True)
                return

            start_time = time.perf_counter()
            update_worker(
                worker_state,
                lock,
                status=f"Trial {repeat_index + 1}/{repeats}: Gesture Data {label['id']} - {label['name']}",
            )

            if source_key == "serial":
                if serial_conn is None or not serial_conn.is_open:
                    raise RuntimeError("Chua ket noi ESP32 Serial.")
                reset_serial_buffers(serial_conn)
                send_serial(serial_conn, "START")
                remember_serial_line(worker_state, lock, "APP_SENT,START")
                deadline = time.perf_counter() + duration_sec
                try:
                    while not stop_event.is_set() and time.perf_counter() < deadline:
                        try:
                            line = serial_conn.readline().decode("utf-8", errors="replace").strip()
                        except serial_error_types() as exc:
                            close_serial_connection(serial_conn)
                            raise RuntimeError(
                                serial_failure_message("doc DATA", serial_port_name(serial_conn), exc)
                            ) from exc
                        remember_serial_line(worker_state, lock, line)
                        if line.startswith("ERROR"):
                            raise RuntimeError(f"ESP32 serial error: {line}")
                        parsed = parse_data_line(line)
                        if parsed is not None:
                            append_worker_row(
                                worker_state,
                                lock,
                                make_row(parsed, start_time, label, subject_id, trial, session_id),
                            )
                        trial_progress = 1 - max(0.0, deadline - time.perf_counter()) / duration_sec
                        overall = (repeat_index + max(0.0, min(1.0, trial_progress))) / max(1, repeats)
                        update_worker(worker_state, lock, progress=max(0.0, min(1.0, overall)))
                finally:
                    try:
                        send_serial(serial_conn, "STOP")
                        remember_serial_line(worker_state, lock, "APP_SENT,STOP")
                    except Exception as exc:
                        remember_serial_line(worker_state, lock, f"APP_STOP_FAILED,{exc}")
            else:
                total_samples = round(duration_sec * SAMPLE_RATE_HZ)
                for index in range(total_samples):
                    if stop_event.is_set():
                        break
                    row = make_row(synthetic_sample(index, label["id"]), start_time, label, subject_id, trial, session_id)
                    append_worker_row(worker_state, lock, row)
                    trial_progress = (index + 1) / max(1, total_samples)
                    overall = (repeat_index + trial_progress) / max(1, repeats)
                    update_worker(worker_state, lock, progress=max(0.0, min(1.0, overall)))
                    time.sleep(1 / SAMPLE_RATE_HZ)

            with lock:
                rows = list(worker_state["rows"])
                serial_debug_lines = list(worker_state.get("serial_debug_lines", []))

            if stop_event.is_set():
                update_worker(worker_state, lock, status="Collection stopped.", stopped=True, done=True)
                return

            if source_key == "serial" and not rows:
                debug_tail = " | ".join(serial_debug_lines[-6:]) if serial_debug_lines else "No serial lines received."
                raise RuntimeError(
                    "Khong nhan duoc dong DATA tu ESP32. "
                    f"Serial gan nhat: {debug_tail}"
                )

            csv_text = rows_to_csv(rows)
            file_record = {
                "file": file_name,
                "path": "",
                "label": label["id"],
                "size_kb": round(len(csv_text.encode("utf-8")) / 1024, 2),
                "samples": len(rows),
                "csv_text": csv_text,
                "collected_at": datetime.now().isoformat(timespec="seconds"),
            }
            if save_to_disk:
                file_record = save_trial_file(rows, label, subject_id, trial, session_id, duration_sec)

            with lock:
                worker_state["saved_files"].append(file_record)

            update_worker(
                worker_state,
                lock,
                latest_csv=file_record["csv_text"],
                latest_file_name=file_record["file"],
                status=f"Saved trial {repeat_index + 1}/{repeats}: {file_record['file']}",
            )

            if repeat_index < repeats - 1 and rest_sec > 0:
                rest_deadline = time.perf_counter() + rest_sec
                while not stop_event.is_set() and time.perf_counter() < rest_deadline:
                    remaining = max(0.0, rest_deadline - time.perf_counter())
                    update_worker(
                        worker_state,
                        lock,
                        status=f"Rest before next trial: {remaining:.1f}s",
                    )
                    time.sleep(min(0.2, remaining))

        with lock:
            rows = list(worker_state["rows"])
            latest_csv = worker_state.get("latest_csv", rows_to_csv(rows))
            latest_file_name = worker_state.get("latest_file_name", "None")

        if stop_event.is_set():
            update_worker(worker_state, lock, status="Collection stopped.", stopped=True, done=True)
            return

        update_worker(
            worker_state,
            lock,
            latest_csv=latest_csv,
            latest_file_name=latest_file_name,
            progress=1.0,
            status=f"Completed {repeats} trial(s): {label['id']} - {label['name']}",
            done=True,
            stopped=False,
        )
    except Exception as exc:
        updates = {"status": str(exc), "error": str(exc), "done": True}
        if source_key == "serial":
            updates["serial_disconnected"] = True
        update_worker(worker_state, lock, **updates)


def start_collection(source_key, port_name, label, subject_id, duration_sec, baseline_sec, repeats, rest_sec, save_to_disk):
    if st.session_state.collecting:
        return

    clean_subject = subject_id.strip() or "S01"
    serial_conn = None
    if source_key == "serial":
        serial_conn = ensure_serial_connection(port_name)

    repeats = max(1, int(repeats))
    first_trial = reserve_trials(clean_subject, label["id"], repeats)
    session_id = format_timestamp(datetime.now())
    worker_state = {
        "rows": [],
        "latest_csv": "",
        "latest_file_name": "None",
        "saved_files": [],
        "progress": 0.0,
        "status": "Gesture Data: waiting",
        "done": False,
        "stopped": False,
        "error": None,
        "label": label,
        "subject_id": clean_subject,
        "session_id": session_id,
        "first_trial": first_trial,
        "repeats": repeats,
    }
    lock = threading.Lock()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=collection_worker,
        args=(
            worker_state,
            lock,
            stop_event,
            source_key,
            serial_conn,
            label,
            clean_subject,
            float(duration_sec),
            float(baseline_sec),
            first_trial,
            repeats,
            float(rest_sec),
            bool(save_to_disk),
            session_id,
        ),
        daemon=True,
    )
    st.session_state.worker_state = worker_state
    st.session_state.worker_lock = lock
    st.session_state.worker_stop = stop_event
    st.session_state.worker_thread = thread
    st.session_state.current_rows = []
    st.session_state.latest_csv = ""
    st.session_state.latest_file_name = "None"
    st.session_state.status = "Gesture Data: waiting"
    st.session_state.collecting = True
    thread.start()


def snapshot_worker():
    worker_state = st.session_state.get("worker_state")
    lock = st.session_state.get("worker_lock")
    if worker_state is None or lock is None:
        return None
    with lock:
        snapshot = dict(worker_state)
        snapshot["rows"] = list(worker_state.get("rows", []))
        snapshot["serial_debug_lines"] = list(worker_state.get("serial_debug_lines", []))
    return snapshot


def finalize_finished_worker():
    snapshot = snapshot_worker()
    if not snapshot or not st.session_state.collecting or not snapshot.get("done"):
        return

    rows = snapshot.get("rows", [])
    latest_csv = snapshot.get("latest_csv") or rows_to_csv(rows)
    saved_files = snapshot.get("saved_files") or []
    has_finished_data = bool(saved_files) or bool(rows)
    st.session_state.current_rows = rows
    st.session_state.latest_csv = latest_csv if has_finished_data and not snapshot.get("error") else ""
    st.session_state.latest_file_name = snapshot.get("latest_file_name") if st.session_state.latest_csv else "None"
    st.session_state.status = snapshot.get("status", "Gesture Data: waiting")
    st.session_state.serial_debug_lines = snapshot.get("serial_debug_lines", [])
    st.session_state.collecting = False
    if snapshot.get("serial_disconnected"):
        close_serial()

    if saved_files:
        for file_record in reversed(saved_files):
            record_saved_file(file_record)
        label = snapshot.get("label") or {}
        backup_path = create_backup_zip(
            saved_files,
            snapshot.get("subject_id", ""),
            label.get("id", ""),
            snapshot.get("session_id", ""),
        )
        if backup_path:
            st.session_state.latest_backup = backup_path
    elif st.session_state.latest_csv:
        label = snapshot.get("label") or {}
        record_file(
            st.session_state.latest_file_name,
            label.get("id", ""),
            st.session_state.latest_csv,
            samples=len(rows),
        )

    st.session_state.worker_state = None
    st.session_state.worker_thread = None
    st.session_state.worker_stop = None
    st.session_state.worker_lock = None


def preserve_worker_rows(snapshot=None):
    snapshot = snapshot or snapshot_worker()
    if not snapshot:
        return

    rows = snapshot.get("rows", [])
    if not rows:
        return

    latest_csv = snapshot.get("latest_csv") or rows_to_csv(rows)
    latest_file_name = snapshot.get("latest_file_name") or "stopped_collection.csv"
    st.session_state.current_rows = rows
    st.session_state.latest_csv = latest_csv
    st.session_state.latest_file_name = latest_file_name
    st.session_state.serial_debug_lines = snapshot.get("serial_debug_lines", st.session_state.serial_debug_lines)


def chart_rows():
    if st.session_state.collecting:
        snapshot = snapshot_worker()
        if snapshot is not None:
            st.session_state.status = snapshot.get("status", st.session_state.status)
            st.session_state.serial_debug_lines = snapshot.get("serial_debug_lines", st.session_state.serial_debug_lines)
            rows = snapshot.get("rows", [])
            if rows:
                preserve_worker_rows(snapshot)
                return rows, snapshot.get("progress", 0.0)
    return st.session_state.current_rows, 1.0 if st.session_state.latest_csv else 0.0


def render_signal_chart(rows, duration_sec, label):
    elapsed = [float(row["elapsed_s"]) for row in rows] if rows else []
    max_t = max(float(duration_sec), elapsed[-1] if elapsed else 0.1)
    gesture_text = f"Gesture Data: {label['id']} - {label['name']}"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
    )

    for field in ["ax_g", "ay_g", "az_g"]:
        values = [float(row[field]) for row in rows] if rows else []
        fig.add_trace(
            go.Scattergl(
                x=elapsed,
                y=values,
                mode="lines",
                name=SIGNAL_LABELS[field],
                line={"color": SIGNAL_COLORS[field], "width": 1.7},
                hovertemplate="time=%{x:.4f}s<br>%{fullData.name}=%{y:.5f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for field in ["gx_dps", "gy_dps", "gz_dps"]:
        values = [float(row[field]) for row in rows] if rows else []
        fig.add_trace(
            go.Scattergl(
                x=elapsed,
                y=values,
                mode="lines",
                name=SIGNAL_LABELS[field],
                line={"color": SIGNAL_COLORS[field], "width": 1.7},
                hovertemplate="time=%{x:.4f}s<br>%{fullData.name}=%{y:.5f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=585,
        margin={"l": 74, "r": 118, "t": 92, "b": 58},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        dragmode="zoom",
        legend={
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.67,
            "xanchor": "left",
            "x": 1.025,
            "font": {"size": 12, "color": "#203447"},
            "itemsizing": "constant",
        },
        font={"family": "Arial, Helvetica, sans-serif", "color": "#203447", "size": 12},
        modebar={"orientation": "h"},
        annotations=[
            {
                "text": gesture_text,
                "xref": "paper",
                "yref": "paper",
                "x": -0.038,
                "y": 1.145,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {
                    "family": "Arial, Helvetica, sans-serif",
                    "size": 17,
                    "color": "#e51c33",
                },
            },
            {
                "text": "Accelerometer Data",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.025,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "bgcolor": "rgba(255,255,255,0.95)",
                "borderpad": 3,
                "font": {
                    "family": "Arial, Helvetica, sans-serif",
                    "size": 16,
                    "color": "#203447",
                },
            },
            {
                "text": "Gyroscope Data",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.505,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "bgcolor": "rgba(255,255,255,0.95)",
                "borderpad": 3,
                "font": {
                    "family": "Arial, Helvetica, sans-serif",
                    "size": 16,
                    "color": "#203447",
                },
            },
        ],
    )
    fig.update_xaxes(
        range=[0, max_t],
        showgrid=True,
        gridcolor="#e8edf2",
        linecolor="#e8edf2",
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text="Acceleration (g)",
        range=[-2, 2],
        showgrid=True,
        gridcolor="#e8edf2",
        linecolor="#e8edf2",
        zeroline=False,
        showspikes=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Angular Velocity (deg/s)",
        range=[-40, 40],
        showgrid=True,
        gridcolor="#e8edf2",
        linecolor="#e8edf2",
        zeroline=False,
        showspikes=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    return fig


def render_counts_chart(labels, files):
    counts = {label["id"]: 0 for label in labels}
    for file in files:
        counts[file.get("label", "")] = counts.get(file.get("label", ""), 0) + 1

    x_labels = [label["id"] for label in labels]
    values = [counts[label_id] for label_id in x_labels]
    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=values,
            marker_color="#2f67c7",
            hovertemplate="label=%{x}<br>files=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=190,
        margin={"l": 28, "r": 8, "t": 4, "b": 34},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#53606a", "size": 10},
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-90, showgrid=False)
    fig.update_yaxes(range=[0, max(1, max(values, default=0))], showgrid=True, gridcolor="#e8edf2", zeroline=False)
    return fig


def total_size_text(files):
    total = sum(float(file.get("size_kb", 0)) for file in files)
    return f"{total / 1024:.2f} MB" if total >= 1024 else f"{total:.2f} KB"


def latest_backup_download():
    relative_path = st.session_state.get("latest_backup", "")
    file_name = Path(relative_path).name if relative_path else "gesture_batch.zip"
    if not relative_path:
        return b"", file_name, True

    backup_path = ROOT / relative_path
    if not backup_path.exists():
        return b"", file_name, True

    return backup_path.read_bytes(), backup_path.name, False


def recent_files_table(files):
    rows = files[:8]
    if not rows:
        return pd.DataFrame(columns=["", "file", "label", "samples", "size_kb"])
    frame = pd.DataFrame(rows)
    if "samples" not in frame.columns:
        frame["samples"] = 0
    frame = frame[["file", "label", "samples", "size_kb"]]
    frame.insert(0, "", range(len(frame)))
    return frame


def numeric_count(rows, fields):
    count = 0
    for row in rows:
        try:
            if all(math.isfinite(float(row[field])) for field in fields):
                count += 1
        except (KeyError, TypeError, ValueError):
            continue
    return count


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


def main():
    st.set_page_config(page_title="Gesture Data Collection", layout="wide")
    init_state()
    finalize_finished_worker()
    apply_style()

    labels = load_labels()
    label_by_text = {label_display(label): label for label in labels}

    st.markdown('<h1 class="title">GESTURE DATA COLLECTION</h1>', unsafe_allow_html=True)

    has_dataset = bool(st.session_state.files)
    if has_dataset:
        left, center, right = st.columns([0.78, 4.6, 0.95], gap="small")
    else:
        left, center = st.columns([0.78, 5.25], gap="small")
        right = None

    with left:
        st.markdown('<h2 class="panel-title">Control Panel</h2>', unsafe_allow_html=True)
        source_text = st.selectbox(
            "Data source",
            ["Computer (mock)", "ESP32 Serial"],
            index=0,
            disabled=st.session_state.collecting,
        )
        source_key = "serial" if source_text == "ESP32 Serial" else "mock"

        selected_label_text = st.selectbox(
            "Select gesture",
            list(label_by_text.keys()),
            disabled=st.session_state.collecting,
        )
        selected_label = label_by_text[selected_label_text]
        subject_id = st.selectbox(
            "Subject ID",
            [f"S{index:02d}" for index in range(1, 16)],
            index=0,
            disabled=st.session_state.collecting,
        )

        duration_col, baseline_col = st.columns(2)
        with duration_col:
            duration_sec = st.number_input(
                "Duration (s)",
                min_value=1.0,
                max_value=10.0,
                value=4.0,
                step=0.5,
                disabled=st.session_state.collecting,
            )
        with baseline_col:
            baseline_sec = st.number_input(
                "Baseline (s)",
                min_value=0,
                max_value=5,
                value=2,
                step=1,
                disabled=st.session_state.collecting,
            )

        repeats_col, rest_col = st.columns(2)
        with repeats_col:
            repeats = st.number_input(
                "Repeats",
                min_value=1,
                max_value=500,
                value=30,
                step=1,
                disabled=st.session_state.collecting,
            )
        with rest_col:
            rest_sec = st.number_input(
                "Rest (s)",
                min_value=0.0,
                max_value=30.0,
                value=1.5,
                step=0.5,
                disabled=st.session_state.collecting,
            )

        save_to_disk = st.checkbox(
            "Save each trial to data/raw",
            value=True,
            disabled=st.session_state.collecting,
        )
        st.markdown('<p class="muted">Sampling rate: <strong>100Hz</strong></p>', unsafe_allow_html=True)

        port_name = ""
        if source_key == "serial":
            st.markdown(
                '<p class="serial-help">Chon cong COM cua CP2102/USB Serial. Neu khong thay cong, dong Arduino Serial Monitor/Serial Plotter roi bam refresh cua trinh duyet.</p>',
                unsafe_allow_html=True,
            )
            ports = available_ports()
            port_options = [port.device for port in ports]
            port_name = st.selectbox("Serial port", port_options, disabled=st.session_state.collecting) if port_options else ""
            if not port_options:
                st.caption("Khong tim thay cong Serial nao.")

        connect_col, ping_col, stream_col, disconnect_col = st.columns([1.25, 0.85, 1.05, 1.05])
        with connect_col:
            connect_clicked = st.button("OK" if st.session_state.serial_connected else "Connect", disabled=source_key != "serial")
        with ping_col:
            ping_clicked = st.button("Ping", disabled=source_key != "serial" or not st.session_state.serial_connected)
        with stream_col:
            stream_clicked = st.button("DATA", disabled=source_key != "serial" or not st.session_state.serial_connected)
        with disconnect_col:
            disconnect_clicked = st.button("Close", disabled=source_key != "serial" or not st.session_state.serial_connected)

        if connect_clicked:
            try:
                ensure_serial_connection(port_name)
                st.session_state.serial_message = f"Da ket noi ESP32 Serial tren {port_name}."
            except Exception as exc:
                close_serial()
                st.session_state.serial_message = str(exc)

        if ping_clicked:
            try:
                conn = ensure_serial_connection(st.session_state.serial_port_name or port_name)
                send_serial(conn, "PING")
                lines = read_until_quiet(conn, seconds=1.0)
                st.session_state.serial_debug_lines = lines[-12:]
                st.session_state.serial_message = "Da gui PING toi ESP32." + (f" Phan hoi: {' | '.join(lines[:3])}" if lines else "")
            except Exception as exc:
                close_serial()
                st.session_state.serial_message = str(exc)

        if stream_clicked:
            try:
                conn = ensure_serial_connection(st.session_state.serial_port_name or port_name)
                data_count, lines = test_serial_stream(conn, seconds=2.0)
                st.session_state.serial_debug_lines = lines[-12:]
                st.session_state.serial_message = f"Test DATA: nhan {data_count} dong DATA trong 2 giay."
            except Exception as exc:
                close_serial()
                st.session_state.serial_message = str(exc)

        if disconnect_clicked:
            close_serial()
            st.session_state.serial_message = "Da ngat ket noi ESP32 Serial."

        if st.session_state.serial_message and source_key == "serial":
            st.caption(st.session_state.serial_message)
        if source_key == "serial" and st.session_state.serial_debug_lines:
            with st.expander("Serial debug", expanded=False):
                st.code("\n".join(st.session_state.serial_debug_lines[-12:]), language="text")

        start_clicked = st.button("Start Collection", type="primary", disabled=st.session_state.collecting)
        stop_clicked = st.button("Stop Collection", disabled=not st.session_state.collecting)
        preview_enabled = st.checkbox("Show current dataset preview")

        if start_clicked:
            try:
                start_collection(
                    source_key,
                    port_name,
                    selected_label,
                    subject_id,
                    duration_sec,
                    baseline_sec,
                    repeats,
                    rest_sec,
                    save_to_disk,
                )
                rerun_app()
            except Exception as exc:
                st.session_state.status = str(exc)

        if stop_clicked and st.session_state.worker_stop is not None:
            preserve_worker_rows()
            st.session_state.worker_stop.set()
            st.session_state.status = "Stopping collection..."
            rerun_app()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="env"><p><strong>Environment:</strong> {source_display(source_key)}</p></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.latest_backup:
            st.caption(f"Latest backup: {st.session_state.latest_backup}")

    rows, progress_value = chart_rows()
    acc_samples = numeric_count(rows, ["ax_g", "ay_g", "az_g"])
    gyro_samples = numeric_count(rows, ["gx_dps", "gy_dps", "gz_dps"])

    with center:
        latest_file = st.session_state.latest_file_name or "None"
        st.markdown(
            f"""
            <div class="main-meta">
              <span>Current source: <strong>{source_display(source_key)}</strong></span>
              <span>Latest collected file: <strong>{latest_file}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<h2 class="section-title">Signal charts</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="status">{st.session_state.status}</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-health">
              <span>Rows: <strong>{len(rows)}</strong></span>
              <span>Acceleration samples: <strong>{acc_samples}</strong></span>
              <span>Gyroscope samples: <strong>{gyro_samples}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        signal_fig = render_signal_chart(rows, duration_sec, selected_label)
        st.plotly_chart(signal_fig, use_container_width=True, config=PLOTLY_CONFIG)

        backup_bytes, backup_name, backup_disabled = latest_backup_download()
        download_col, zip_col, progress_col = st.columns([0.85, 0.85, 3.0], vertical_alignment="center")
        with download_col:
            st.download_button(
                "Download CSV",
                data=st.session_state.latest_csv,
                file_name=st.session_state.latest_file_name if st.session_state.latest_csv else "gesture_data.csv",
                mime="text/csv",
                disabled=not st.session_state.latest_csv,
            )
        with zip_col:
            st.download_button(
                "Download ZIP",
                data=backup_bytes,
                file_name=backup_name,
                mime="application/zip",
                disabled=backup_disabled,
            )
        with progress_col:
            st.progress(max(0.0, min(1.0, float(progress_value))))

        if preview_enabled:
            st.code(rows_to_csv(rows[-6:]), language="csv")

    if right is not None:
        with right:
            st.markdown('<h2 class="panel-title">Dataset overview</h2>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="metric">
                    <span class="metric-icon">F</span>
                    <span>Total files</span>
                    <strong class="metric-value">{len(st.session_state.files)}</strong>
                  </div>
                  <div class="metric">
                    <span class="metric-icon" style="background: #7f3db7;">S</span>
                    <span>Total size</span>
                    <strong class="metric-value">{total_size_text(st.session_state.files)}</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.markdown('<h2 class="panel-title">Gesture counts</h2>', unsafe_allow_html=True)
            counts_fig = render_counts_chart(labels, st.session_state.files)
            st.plotly_chart(counts_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.markdown('<h2 class="panel-title">Recent files</h2>', unsafe_allow_html=True)
            st.dataframe(recent_files_table(st.session_state.files), hide_index=True, use_container_width=True, height=160)

    if st.session_state.collecting:
        time.sleep(0.2)
        rerun_app()


if __name__ == "__main__":
    main()
