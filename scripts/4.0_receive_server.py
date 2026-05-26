from __future__ import annotations

import json
import mimetypes
import os
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

UPLOAD_DIR = Path(
    os.environ.get(
        "LAPTOP_IMAGE_DIR",
        str(PROJECT_ROOT / "data" / "raspberry_latest_images"),
    )
)
HOST = os.environ.get("LAPTOP_RECEIVE_HOST", "0.0.0.0")
PORT = int(os.environ.get("LAPTOP_RECEIVE_PORT", "8001"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def sanitize_filename(filename: str) -> str:
    name = filename.replace("\\", "/")
    name = Path(name).name.strip()
    if not name:
        name = datetime.now().strftime("raspi_%Y%m%d_%H%M%S.jpg")

    invalid_chars = '<>:"/\\|?*'
    return "".join("_" if char in invalid_chars else char for char in name)


def get_latest_image() -> Path | None:
    if not UPLOAD_DIR.exists():
        return None

    image_files = [
        path
        for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        return None

    return max(image_files, key=lambda path: (path.stat().st_mtime, path.name))


class LaptopReceiveHandler(BaseHTTPRequestHandler):
    server_version = "LaptopImageReceiver/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/latest-meta"}:
            latest = get_latest_image()
            if latest is None:
                self.send_json({"error": "no_image_found"}, status=404)
                return

            stat = latest.stat()
            self.send_json(
                {
                    "filename": latest.name,
                    "path": str(latest),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
            return

        if path == "/latest-image":
            self.send_latest_image()
            return

        self.send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self.send_json({"error": "not_found"}, status=404)
            return

        query = parse_qs(parsed.query)
        filename = query.get("filename", [None])[0]
        filename = filename or self.headers.get("X-Image-Filename")
        filename = sanitize_filename(unquote(filename or ""))

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_json({"error": "empty_upload"}, status=400)
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOAD_DIR / filename
        temp_path = UPLOAD_DIR / f".{save_path.name}.tmp"

        remaining = content_length
        with temp_path.open("wb") as file:
            while remaining > 0:
                chunk = self.rfile.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                file.write(chunk)
                remaining -= len(chunk)

        if remaining != 0:
            temp_path.unlink(missing_ok=True)
            self.send_json({"error": "incomplete_upload"}, status=400)
            return

        temp_path.replace(save_path)
        self.send_json(
            {
                "saved": True,
                "filename": save_path.name,
                "path": str(save_path),
                "size": save_path.stat().st_size,
            }
        )

    def send_latest_image(self) -> None:
        latest = get_latest_image()
        if latest is None:
            self.send_json({"error": "no_image_found"}, status=404)
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


def main() -> None:
    print(f"Receiving images into: {UPLOAD_DIR}")
    print(f"Upload endpoint: http://{HOST}:{PORT}/upload?filename=image.jpg")
    print(f"Latest-image endpoint: http://{HOST}:{PORT}/latest-image")

    server = ThreadingHTTPServer((HOST, PORT), LaptopReceiveHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLaptop receive server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
