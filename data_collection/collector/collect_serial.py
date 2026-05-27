import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "labels.json"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

DATA_COLUMNS = [
    "time",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
    "label",
]


def load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"].upper(): item for item in data["labels"]}


def require_pyserial():
    if serial is None:
        raise SystemExit(
            "Thieu thu vien pyserial. Cai bang lenh: python -m pip install -r requirements.txt"
        )


def print_ports():
    require_pyserial()
    ports = list(list_ports.comports())
    if not ports:
        print("Khong tim thay cong Serial nao.")
        return
    for port in ports:
        print(f"{port.device}\t{port.description}")


def serial_error_types():
    if serial is None:
        return (OSError,)
    return (serial.SerialException, OSError)


def serial_port_name(ser=None, fallback=""):
    if fallback:
        return fallback
    if ser is not None:
        port = getattr(ser, "port", "")
        if port:
            return port
    return "COM"


def serial_failure_message(action, port_name, exc):
    return (
        f"Loi Serial khi {action} tren {port_name}: {exc}\n"
        "Cach xu ly: dong Arduino Serial Monitor/Serial Plotter hoac app khac dang dung COM, "
        "rut cam lai USB ESP32, chon lai dung cong COM, roi chay lai."
    )


def close_serial_connection(ser):
    if ser is None:
        return
    try:
        if getattr(ser, "is_open", False):
            ser.close()
    except serial_error_types():
        pass


def reset_serial_buffers(ser, action="reset bo dem"):
    port_name = serial_port_name(ser)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except serial_error_types() as exc:
        close_serial_connection(ser)
        raise SystemExit(serial_failure_message(action, port_name, exc)) from exc


def open_serial(port, baud, timeout):
    require_pyserial()
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            write_timeout=1.0,
            rtscts=False,
            dsrdtr=False,
        )
    except serial_error_types() as exc:
        raise SystemExit(serial_failure_message("mo cong", port, exc)) from exc
    time.sleep(2.0)
    reset_serial_buffers(ser)
    return ser


def send_command(ser, command):
    if ser is None or not getattr(ser, "is_open", False):
        raise SystemExit("Chua ket noi ESP32 Serial.")

    port_name = serial_port_name(ser)
    try:
        ser.write((command.strip() + "\n").encode("ascii"))
        ser.flush()
    except serial_error_types() as exc:
        close_serial_connection(ser)
        raise SystemExit(serial_failure_message(f"gui lenh {command.strip()}", port_name, exc)) from exc


def read_until_quiet(ser, seconds=1.0):
    end_at = time.time() + seconds
    lines = []
    while time.time() < end_at:
        try:
            raw = ser.readline()
        except serial_error_types() as exc:
            close_serial_connection(ser)
            raise SystemExit(serial_failure_message("doc du lieu", serial_port_name(ser), exc)) from exc
        if raw:
            lines.append(raw.decode("utf-8", errors="replace").strip())
    return lines


def calibrate(ser):
    send_command(ser, "CALIBRATE")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except serial_error_types() as exc:
            close_serial_connection(ser)
            raise SystemExit(serial_failure_message("doc du lieu CALIBRATE", serial_port_name(ser), exc)) from exc
        if line:
            print(line)
        if line.startswith("CALIBRATED"):
            return
    print("Canh bao: chua nhan duoc dong CALIBRATED, tiep tuc thu.")


def countdown(seconds, message):
    for remaining in range(seconds, 0, -1):
        print(f"{message} {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")


def parse_data_line(line):
    parts = line.split(",")
    if len(parts) != 10 or parts[0] != "DATA":
        return None
    return {
        "millis_ms": parts[1],
        "sample_index": parts[2],
        "ax_g": parts[3],
        "ay_g": parts[4],
        "az_g": parts[5],
        "gx_dps": parts[6],
        "gy_dps": parts[7],
        "gz_dps": parts[8],
        "temp_c": parts[9],
    }


def build_output_path(label_id, subject_id, trial, session_id):
    safe_subject = subject_id.replace(" ", "_")
    safe_label = label_id.replace(" ", "_")
    label_dir = RAW_DIR / safe_subject / safe_label
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{session_id}_{safe_subject}_{safe_label}_T{trial:03d}.csv"


def append_manifest(row):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST_PATH.exists() and MANIFEST_PATH.stat().st_size > 0
    with MANIFEST_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "subject_id",
                "label_id",
                "label_name",
                "trial",
                "duration_s",
                "samples",
                "path",
                "recorded_at",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def collect_trial(ser, label, subject_id, trial, session_id, duration_s):
    output_path = build_output_path(label["id"], subject_id, trial, session_id)
    start_pc = time.time()
    samples = 0

    reset_serial_buffers(ser)
    send_command(ser, "START")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATA_COLUMNS)
        writer.writeheader()

        deadline = start_pc + duration_s
        while time.time() < deadline:
            try:
                line = ser.readline().decode("utf-8", errors="replace").strip()
            except serial_error_types() as exc:
                close_serial_connection(ser)
                raise SystemExit(serial_failure_message("doc DATA", serial_port_name(ser), exc)) from exc
            parsed = parse_data_line(line)
            if parsed is None:
                continue

            now = time.time()
            row = {
                "time": f"{now - start_pc:.4f}",
                "ax_g": parsed["ax_g"],
                "ay_g": parsed["ay_g"],
                "az_g": parsed["az_g"],
                "gx_dps": parsed["gx_dps"],
                "gy_dps": parsed["gy_dps"],
                "gz_dps": parsed["gz_dps"],
                "label": label["id"],
            }
            writer.writerow(row)
            samples += 1

    stop_sent = True
    try:
        send_command(ser, "STOP")
    except SystemExit as exc:
        stop_sent = False
        print(exc)
    if stop_sent and getattr(ser, "is_open", False):
        try:
            read_until_quiet(ser, seconds=0.2)
        except SystemExit as exc:
            print(exc)

    append_manifest(
        {
            "session_id": session_id,
            "subject_id": subject_id,
            "label_id": label["id"],
            "label_name": label["name"],
            "trial": trial,
            "duration_s": duration_s,
            "samples": samples,
            "path": str(output_path.relative_to(ROOT)),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    print(f"Da ghi {samples} mau -> {output_path}")


def label_sort_key(label_id):
    prefix = label_id[:1].upper()
    try:
        number = int(label_id[1:])
    except ValueError:
        number = 999
    prefix_order = {"G": 0, "N": 1}.get(prefix, 2)
    return prefix_order, number


def choose_labels(requested_label, labels):
    requested = requested_label.upper()
    if requested == "ALL":
        return [labels[key] for key in sorted(labels.keys(), key=label_sort_key)]
    if requested not in labels:
        valid = ", ".join(sorted(labels.keys(), key=label_sort_key))
        raise SystemExit(f"Nhãn không hợp lệ: {requested_label}. Chọn một trong: {valid}, ALL")
    return [labels[requested]]


def print_label_table(labels):
    print("Danh sach nhan:")
    for key in sorted(labels.keys(), key=label_sort_key):
        label = labels[key]
        print(
            f"{label['id']:>3} | {label['name']:<24} | {label['device']:<8} | {label['system_state']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Thu du lieu cu chi tay tu ESP32 + MPU6050 qua Serial."
    )
    parser.add_argument("--port", help="Cong Serial, vi du COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--subject", default="S01", help="Ma nguoi tham gia, vi du S01")
    parser.add_argument("--label", default="G1", help="G1..G15, N1..N5 hoac ALL")
    parser.add_argument("--repeats", type=int, default=10, help="So lan lap moi nhan")
    parser.add_argument("--duration", type=float, default=4.0, help="So giay ghi moi lan lap")
    parser.add_argument("--prepare", type=int, default=3, help="So giay dem nguoc truoc khi ghi")
    parser.add_argument("--rest", type=float, default=1.5, help="So giay nghi giua cac lan lap")
    parser.add_argument("--no-calibrate", action="store_true", help="Bo qua can bang gyro")
    parser.add_argument("--list-ports", action="store_true", help="Liet ke cong Serial")
    parser.add_argument("--list-labels", action="store_true", help="Liet ke nhan G1-G15 va N1-N5")
    args = parser.parse_args()

    labels = load_labels()

    if args.list_labels:
        print_label_table(labels)
        return

    if args.list_ports:
        print_ports()
        return

    if not args.port:
        raise SystemExit("Can truyen --port. Dung --list-ports de xem cong hien co.")

    selected_labels = choose_labels(args.label, labels)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Session: {session_id}")
    print(f"Subject: {args.subject}")
    print(f"Output: {RAW_DIR}")

    with open_serial(args.port, args.baud, timeout=0.25) as ser:
        send_command(ser, "PING")
        for line in read_until_quiet(ser, seconds=1.0):
            print(line)

        if not args.no_calibrate:
            print("Dat cam bien nam yen tren tay/ban, giu yen de can bang gyro.")
            countdown(3, "Bat dau can bang sau")
            calibrate(ser)

        for label in selected_labels:
            print()
            print(f"Nhan {label['id']}: {label['name']}")
            print(f"Start: {label['start']}")
            print(f"Stop : {label['stop']}")

            for trial in range(1, args.repeats + 1):
                input(f"Lan {trial}/{args.repeats}. Nhan Enter khi da san sang...")
                countdown(args.prepare, "Bat dau ghi sau")
                collect_trial(ser, label, args.subject, trial, session_id, args.duration)
                if args.rest > 0:
                    time.sleep(args.rest)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDa dung thu du lieu.")
        sys.exit(130)
