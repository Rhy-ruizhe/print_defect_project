from __future__ import annotations

import argparse
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "defect_classifier.pth"
YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "results"
    / "yolo_print_object3"
    / "weights"
    / "best.pt"
)

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raspberry_latest_images"
CROPPED_IMAGE_DIR = PROJECT_ROOT / "data" / "cropped_latest_images"

# Change this IP/port to the Raspberry Pi running scripts/4.1_rasp_server.py.
RASPBERRY_PI_IMAGE_URL = os.environ.get(
    "RASPBERRY_PI_IMAGE_URL",
    "http://192.168.0.102:8000/latest-image",  # Rasp IP address
)

POLL_INTERVAL_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 10
YOLO_CONF_THRES = 0.3
YOLO_PADDING = 20
IMAGE_SIZE = 320

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["defect", "no_defect"]


@dataclass
class DetectionResult:
    source_image: Path
    cropped_image: Path | None
    prediction: str
    confidence: float
    yolo_confidence: float | None
    status: str = "ok"


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def sanitize_filename(filename: str, fallback: str) -> str:
    name = filename.replace("\\", "/")
    name = Path(name).name.strip()
    if not name:
        name = fallback

    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid_chars else char for char in name)
    return cleaned or fallback


def filename_from_response(response: requests.Response) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content_type = (response.headers.get("Content-Type") or "").split(";")[0]
    extension = mimetypes.guess_extension(content_type.strip()) or ".jpg"
    fallback = f"raspi_{timestamp}{extension}"

    header_filename = response.headers.get("X-Image-Filename")
    if header_filename:
        return sanitize_filename(unquote(header_filename), fallback)

    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(
        r"filename\*?=(?:UTF-8''|\"?)([^\";]+)\"?",
        content_disposition,
        flags=re.IGNORECASE,
    )
    if match:
        return sanitize_filename(unquote(match.group(1)), fallback)

    return fallback


def download_latest_image(
    image_url: str,
    output_dir: Path,
    timeout_seconds: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(image_url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        filename = filename_from_response(response)
        save_path = output_dir / filename
        temp_path = output_dir / f".{save_path.name}.tmp"

        with temp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    file.write(chunk)

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("The Raspberry Pi returned an empty image file.")

    temp_path.replace(save_path)
    return save_path


def read_cv_image(image_path: Path) -> np.ndarray | None:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_cv_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    extension = image_path.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"Failed to encode cropped image: {image_path}")
    encoded.tofile(str(image_path))


def crop_print_object(
    yolo_model: YOLO,
    image_path: Path,
    output_dir: Path,
) -> tuple[Path | None, float | None]:
    image = read_cv_image(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    results = yolo_model.predict(
        source=str(image_path),
        conf=YOLO_CONF_THRES,
        verbose=False,
    )

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None

    best_box = None
    best_confidence = -1.0
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence > best_confidence:
            best_confidence = confidence
            best_box = box

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
    x1 = max(0, x1 - YOLO_PADDING)
    y1 = max(0, y1 - YOLO_PADDING)
    x2 = min(width, x2 + YOLO_PADDING)
    y2 = min(height, y2 + YOLO_PADDING)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, best_confidence

    output_name = f"{image_path.stem}_crop{image_path.suffix or '.jpg'}"
    cropped_path = output_dir / output_name
    write_cv_image(cropped_path, cropped)
    return cropped_path, best_confidence


def load_classifier_model(model_path: Path) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def classify_image(model: nn.Module, image_path: Path) -> tuple[str, float]:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_index = torch.argmax(probabilities, dim=1).item()

    return CLASS_NAMES[pred_index], float(probabilities[0][pred_index].item())


def run_detection_once(
    classifier_model: nn.Module,
    yolo_model: YOLO,
    image_url: str,
    raw_dir: Path,
    cropped_dir: Path,
    local_image: Path | None,
    request_timeout_seconds: float,
) -> DetectionResult:
    if local_image is None:
        source_image = download_latest_image(
            image_url=image_url,
            output_dir=raw_dir,
            timeout_seconds=request_timeout_seconds,
        )
    else:
        source_image = resolve_project_path(local_image)
        if not source_image.exists():
            raise FileNotFoundError(f"Local image not found: {source_image}")

    cropped_image, yolo_confidence = crop_print_object(
        yolo_model=yolo_model,
        image_path=source_image,
        output_dir=cropped_dir,
    )

    if cropped_image is None:
        return DetectionResult(
            source_image=source_image,
            cropped_image=None,
            prediction="no_defect",
            confidence=0.0,
            yolo_confidence=yolo_confidence,
            status="no_print_object_detected",
        )

    prediction, confidence = classify_image(classifier_model, cropped_image)
    return DetectionResult(
        source_image=source_image,
        cropped_image=cropped_image,
        prediction=prediction,
        confidence=confidence,
        yolo_confidence=yolo_confidence,
    )


def print_detection_result(result: DetectionResult) -> None:
    print(f"SourceImage: {result.source_image}")
    if result.cropped_image is not None:
        print(f"CroppedImage: {result.cropped_image}")
    else:
        print("CroppedImage: none")

    if result.yolo_confidence is not None:
        print(f"YoloConfidence: {result.yolo_confidence:.6f}")
    else:
        print("YoloConfidence: none")

    print(f"Status: {result.status}")
    print("Prediction:", result.prediction)
    print(f"Confidence: {result.confidence:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the latest Raspberry Pi image, crop it, and classify defects."
    )
    parser.add_argument("--watch", action="store_true",
                        help="Run every 10 seconds.")
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help="Seconds between checks in --watch mode.",
    )
    parser.add_argument(
        "--image-url",
        default=RASPBERRY_PI_IMAGE_URL,
        help="URL of the Raspberry Pi latest-image endpoint.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_IMAGE_DIR,
        help="Folder where downloaded raw images are saved on the laptop.",
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=CROPPED_IMAGE_DIR,
        help="Folder where YOLO crops are saved on the laptop.",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=CLASSIFIER_MODEL_PATH,
        help="Path to the trained defect classifier .pth file.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=YOLO_MODEL_PATH,
        help="Path to the trained YOLO model .pt file.",
    )
    parser.add_argument(
        "--local-image",
        type=Path,
        default=None,
        help="Classify a local image instead of downloading from the Raspberry Pi.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds when downloading from the Raspberry Pi.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    classifier_model_path = resolve_project_path(args.classifier_model)
    yolo_model_path = resolve_project_path(args.yolo_model)
    raw_dir = resolve_project_path(args.raw_dir)
    cropped_dir = resolve_project_path(args.cropped_dir)

    if not classifier_model_path.exists():
        print(f"Classifier model file not found: {classifier_model_path}")
        return 1

    if not yolo_model_path.exists():
        print(f"YOLO model file not found: {yolo_model_path}")
        return 1

    classifier_model = load_classifier_model(classifier_model_path)
    yolo_model = YOLO(str(yolo_model_path))

    while True:
        try:
            result = run_detection_once(
                classifier_model=classifier_model,
                yolo_model=yolo_model,
                image_url=args.image_url,
                raw_dir=raw_dir,
                cropped_dir=cropped_dir,
                local_image=args.local_image,
                request_timeout_seconds=args.timeout,
            )
            print_detection_result(result)
        except requests.RequestException as exc:
            print(f"Image download failed: {exc}")
            if not args.watch:
                return 2
        except Exception as exc:
            print(f"Detection failed: {exc}")
            if not args.watch:
                return 3

        if not args.watch:
            return 0

        print(f"Waiting {args.interval:g} seconds before next defect check...")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
