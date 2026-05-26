from __future__ import annotations

import csv
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


CAPTURE_BASE_DIR = env_path(
    "RASPI_CAPTURE_BASE_DIR",
    Path.home() / "dual_capture0327",
)

# By default this server exposes the latest image written into any cam1 folder
# under CAPTURE_BASE_DIR, for example ~/dual_capture0327/2026-04-28/cam1.
IMAGE_DIR = env_path("RASPI_IMAGE_DIR", CAPTURE_BASE_DIR)
SERVE_CAMERA_DIR = os.environ.get("RASPI_SERVE_CAMERA", "cam1").strip()
IMAGE_SEARCH_RECURSIVE = env_bool("RASPI_IMAGE_RECURSIVE", True)

HOST = os.environ.get("RASPI_IMAGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("RASPI_IMAGE_PORT", "8000"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CAPTURE_ENABLED = env_bool("RASPI_CAPTURE_ENABLED", True)
CAPTURE_PROMPT_START = env_bool("RASPI_CAPTURE_PROMPT_START", True)
CAPTURE_INTERVAL_SECONDS = env_float("RASPI_CAPTURE_INTERVAL", 2.0)
CAM0_ID = env_int("RASPI_CAM0_ID", 0)
CAM1_ID = env_int("RASPI_CAM1_ID", 1)
CAPTURE_IMAGE_EXT = os.environ.get("RASPI_CAPTURE_IMAGE_EXT", "jpg").strip(".")
CAPTURE_WIDTH = env_int("RASPI_CAPTURE_WIDTH", 2560)
CAPTURE_HEIGHT = env_int("RASPI_CAPTURE_HEIGHT", 1440)
CAPTURE_SHUTTER = env_int("RASPI_CAPTURE_SHUTTER", 5000)
CAPTURE_GAIN = env_float("RASPI_CAPTURE_GAIN", 3.0)
USE_AWB_GAINS = env_bool("RASPI_USE_AWB_GAINS", True)
AWB_RED = env_float("RASPI_AWB_RED", 1.8)
AWB_BLUE = env_float("RASPI_AWB_BLUE", 1.5)
USE_LENS_POSITION = env_bool("RASPI_USE_LENS_POSITION", True)
LENS_POSITION = env_float("RASPI_LENS_POSITION", 6.3)
RUN_FOREVER = env_bool("RASPI_RUN_FOREVER", True)
MAX_SHOTS = env_int("RASPI_MAX_SHOTS", 100)

LOG_FILE = CAPTURE_BASE_DIR / "run.log"
CSV_FILE = CAPTURE_BASE_DIR / "capture_timestamps.csv"
STOP_EVENT = threading.Event()
LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class CapturePaths:
    date_folder: str
    cam0_dir: Path
    cam1_dir: Path
    cam0_file: str
    cam1_file: str
    cam0_path: Path
    cam1_path: Path


def get_latest_image() -> Path | None:
    if not IMAGE_DIR.exists():
        return None

    if IMAGE_SEARCH_RECURSIVE:
        paths = IMAGE_DIR.rglob("*")
    else:
        paths = IMAGE_DIR.iterdir()

    image_files = [path for path in paths if should_serve_image(path)]
    if not image_files:
        return None

    return max(image_files, key=lambda path: (path.stat().st_mtime, path.name))


def should_serve_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False

    if path.name.startswith("."):
        return False

    if not SERVE_CAMERA_DIR:
        return True

    return path.parent.name == SERVE_CAMERA_DIR


def image_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        ),
    }


class LatestImageHandler(BaseHTTPRequestHandler):
    server_version = "RaspiLatestImage/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in {"/", "/latest-meta"}:
            self.send_latest_meta()
            return

        if path == "/latest-image":
            self.send_latest_image()
            return

        self.send_json({"error": "not_found"}, status=404)

    def send_latest_meta(self) -> None:
        latest = get_latest_image()
        if latest is None:
            self.send_json(
                {
                    "error": "no_image_found",
                    "image_dir": str(IMAGE_DIR),
                    "camera_dir": SERVE_CAMERA_DIR,
                    "recursive": IMAGE_SEARCH_RECURSIVE,
                },
                status=404,
            )
            return

        self.send_json(image_metadata(latest))

    def send_latest_image(self) -> None:
        latest = get_latest_image()
        if latest is None:
            self.send_json(
                {
                    "error": "no_image_found",
                    "image_dir": str(IMAGE_DIR),
                    "camera_dir": SERVE_CAMERA_DIR,
                    "recursive": IMAGE_SEARCH_RECURSIVE,
                },
                status=404,
            )
            return

        content_type = mimetypes.guess_type(latest.name)[0] or "image/jpeg"
        encoded_filename = quote(latest.name)
        stat = latest.stat()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("X-Image-Filename", encoded_filename)
        self.send_header("X-Image-Mtime", str(stat.st_mtime))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename*=UTF-8''{encoded_filename}",
        )
        self.end_headers()

        with latest.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def local_time_ms() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def epoch_ms() -> int:
    return int(time.time() * 1000)


def log_capture(message: str, echo: bool = True) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with LOG_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    if echo:
        print(line, flush=True)


def ensure_capture_files() -> None:
    CAPTURE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_FILE.exists():
        return

    header = [
        "shot_index",
        "date_folder",
        "trigger_time_local",
        "trigger_time_epoch_ms",
        "cam0_file",
        "cam0_start_local",
        "cam0_start_epoch_ms",
        "cam0_end_local",
        "cam0_end_epoch_ms",
        "cam0_status",
        "cam1_file",
        "cam1_start_local",
        "cam1_start_epoch_ms",
        "cam1_end_local",
        "cam1_end_epoch_ms",
        "cam1_status",
    ]
    with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(header)


def common_camera_args() -> list[str]:
    args = [
        "-n",
        "--width",
        str(CAPTURE_WIDTH),
        "--height",
        str(CAPTURE_HEIGHT),
        "--shutter",
        str(CAPTURE_SHUTTER),
        "--gain",
        str(CAPTURE_GAIN),
        "-t",
        "1",
    ]

    if USE_AWB_GAINS:
        args.extend(["--awbgains", f"{AWB_RED},{AWB_BLUE}"])

    if USE_LENS_POSITION:
        args.extend(["--lens-position", str(LENS_POSITION)])

    return args


def build_capture_paths() -> CapturePaths:
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp_name = now.strftime("%Y%m%d_%H%M%S")
    day_dir = CAPTURE_BASE_DIR / date_folder
    cam0_dir = day_dir / "cam0"
    cam1_dir = day_dir / "cam1"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)

    cam0_file = f"cam0_{timestamp_name}.{CAPTURE_IMAGE_EXT}"
    cam1_file = f"cam1_{timestamp_name}.{CAPTURE_IMAGE_EXT}"
    return CapturePaths(
        date_folder=date_folder,
        cam0_dir=cam0_dir,
        cam1_dir=cam1_dir,
        cam0_file=cam0_file,
        cam1_file=cam1_file,
        cam0_path=cam0_dir / cam0_file,
        cam1_path=cam1_dir / cam1_file,
    )


def temp_capture_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def finalize_capture_file(temp_path: Path, final_path: Path, status: int) -> None:
    if status == 0 and temp_path.exists():
        temp_path.replace(final_path)
        return

    temp_path.unlink(missing_ok=True)


def start_camera_process(
    camera_id: int,
    output_path: Path,
    log_output,
) -> subprocess.Popen[bytes] | None:
    command = [
        "rpicam-still",
        "--camera",
        str(camera_id),
        *common_camera_args(),
        "-o",
        str(output_path),
    ]
    try:
        return subprocess.Popen(
            command,
            stdout=log_output,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        log_output.write("rpicam-still command was not found.\n")
        log_output.flush()
        return None


def wait_for_process(process: subprocess.Popen[bytes] | None) -> int:
    if process is None:
        return 127
    return process.wait()


def append_capture_row(row: list[object]) -> None:
    with CSV_FILE.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def capture_once(shot_index: int) -> None:
    paths = build_capture_paths()
    cam0_temp_path = temp_capture_path(paths.cam0_path)
    cam1_temp_path = temp_capture_path(paths.cam1_path)
    trigger_local = local_time_ms()
    trigger_ms = epoch_ms()

    log_capture(
        f"[shot {shot_index}] capturing {paths.cam0_file} / {paths.cam1_file}"
    )

    with LOG_FILE.open("a", encoding="utf-8") as log_output:
        cam0_start_local = local_time_ms()
        cam0_start_ms = epoch_ms()
        process0 = start_camera_process(CAM0_ID, cam0_temp_path, log_output)

        cam1_start_local = local_time_ms()
        cam1_start_ms = epoch_ms()
        process1 = start_camera_process(CAM1_ID, cam1_temp_path, log_output)

        cam0_status = wait_for_process(process0)
        cam0_end_local = local_time_ms()
        cam0_end_ms = epoch_ms()

        cam1_status = wait_for_process(process1)
        cam1_end_local = local_time_ms()
        cam1_end_ms = epoch_ms()

    finalize_capture_file(cam0_temp_path, paths.cam0_path, cam0_status)
    finalize_capture_file(cam1_temp_path, paths.cam1_path, cam1_status)

    append_capture_row(
        [
            shot_index,
            paths.date_folder,
            trigger_local,
            trigger_ms,
            paths.cam0_file,
            cam0_start_local,
            cam0_start_ms,
            cam0_end_local,
            cam0_end_ms,
            cam0_status,
            paths.cam1_file,
            cam1_start_local,
            cam1_start_ms,
            cam1_end_local,
            cam1_end_ms,
            cam1_status,
        ]
    )

    log_capture(
        f"[shot {shot_index}] done | cam0_status={cam0_status} "
        f"cam1_status={cam1_status}"
    )


def parse_start_time(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def read_start_time() -> datetime | None:
    env_start_time = os.environ.get("RASPI_CAPTURE_START_TIME")
    if env_start_time is not None:
        return parse_start_time(env_start_time)

    if not CAPTURE_PROMPT_START:
        return None

    try:
        value = input(
            "Enter start time (YYYY-MM-DD HH:MM:SS), or press Enter to start now: "
        )
    except EOFError:
        return None

    return parse_start_time(value)


def wait_until_start(start_time: datetime | None) -> None:
    if start_time is None:
        log_capture("No start time was provided; capture starts immediately.")
        return

    now = datetime.now()
    if now >= start_time:
        log_capture("Configured start time has already passed; capture starts now.")
        return

    wait_seconds = (start_time - now).total_seconds()
    log_capture(
        f"Waiting {wait_seconds:.0f} seconds until start time "
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}."
    )
    while not STOP_EVENT.is_set():
        remaining = (start_time - datetime.now()).total_seconds()
        if remaining <= 0:
            break
        STOP_EVENT.wait(min(remaining, 1.0))

    log_capture("Start time reached; capture loop begins.")


def capture_loop(start_time: datetime | None) -> None:
    ensure_capture_files()
    log_capture("==============================", echo=False)
    log_capture("Capture service started.")
    log_capture(f"CAPTURE_BASE_DIR={CAPTURE_BASE_DIR}", echo=False)
    log_capture(f"INTERVAL={CAPTURE_INTERVAL_SECONDS}", echo=False)
    log_capture(f"CAM0_ID={CAM0_ID} CAM1_ID={CAM1_ID}", echo=False)
    log_capture(f"WIDTH={CAPTURE_WIDTH} HEIGHT={CAPTURE_HEIGHT}", echo=False)
    log_capture(f"SHUTTER={CAPTURE_SHUTTER} GAIN={CAPTURE_GAIN}", echo=False)

    wait_until_start(start_time)

    shot_index = 0
    while not STOP_EVENT.is_set():
        if not RUN_FOREVER and shot_index >= MAX_SHOTS:
            log_capture(f"Reached max shots ({MAX_SHOTS}); capture loop exits.")
            return

        loop_started_at = time.monotonic()
        try:
            capture_once(shot_index)
        except Exception as exc:
            log_capture(f"[shot {shot_index}] capture failed: {exc}")

        shot_index += 1
        elapsed = time.monotonic() - loop_started_at
        sleep_seconds = CAPTURE_INTERVAL_SECONDS - elapsed
        if sleep_seconds > 0:
            STOP_EVENT.wait(sleep_seconds)

    log_capture("Capture service stopped.")


def main() -> None:
    print(f"Serving latest image from: {IMAGE_DIR}")
    if SERVE_CAMERA_DIR:
        print(f"Serving only images from folders named: {SERVE_CAMERA_DIR}")
    print(f"Recursive image search: {IMAGE_SEARCH_RECURSIVE}")
    print(f"HTTP endpoint: http://{HOST}:{PORT}/latest-image")
    print("Set RASPI_IMAGE_DIR to change the watched Raspberry Pi folder.")

    capture_thread: threading.Thread | None = None
    if CAPTURE_ENABLED:
        try:
            start_time = read_start_time()
        except ValueError:
            print("Invalid start time. Use format YYYY-MM-DD HH:MM:SS.")
            return

        capture_thread = threading.Thread(
            target=capture_loop,
            args=(start_time,),
            name="dual-csi-capture",
            daemon=True,
        )
        capture_thread.start()
    else:
        print("Camera capture is disabled; server will only expose existing files.")

    server = ThreadingHTTPServer((HOST, PORT), LatestImageHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRaspberry Pi image server stopped.")
    finally:
        STOP_EVENT.set()
        server.server_close()
        if capture_thread is not None:
            capture_thread.join(timeout=5)


if __name__ == "__main__":
    main()
