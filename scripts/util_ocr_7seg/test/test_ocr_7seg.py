import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    from ..ocr_test_7seg import ROI_FILE, WindSpeedOCR, select_and_save_roi
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.util_ocr_7seg.ocr_test_7seg import ROI_FILE, WindSpeedOCR, select_and_save_roi


def make_debug_dir(debug_dir):
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_debug_row(log_file, row):
    file_exists = log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample",
                "timestamp",
                "mode",
                "wind",
                "stable",
                "raw",
                "text",
                "stable_text",
                "raw_text",
                "confidence",
                "elapsed_ms",
                "failure_reason",
                "retry_attempts",
                "retry_selected_attempt",
                "retry_candidates",
                "digit_box_summary",
                "digit_summary",
                "frame_path",
                "preview_path",
                "rectified_path",
                "binary_path",
                "digit_paths",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_debug_sample(debug_dir, sample_index, info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{sample_index:04d}_{timestamp}"
    frame_path = debug_dir / f"{prefix}_frame.png"
    preview_path = debug_dir / f"{prefix}_preview.png"
    rectified_path = debug_dir / f"{prefix}_rectified.png"
    enhanced_path = debug_dir / f"{prefix}_enhanced.png"
    binary_path = debug_dir / f"{prefix}_binary.png"
    digit_paths = []

    cv2.imwrite(str(frame_path), info["frame"])
    if info.get("preview") is not None:
        cv2.imwrite(str(preview_path), info["preview"])
    else:
        preview_path = None

    if info.get("rectified") is not None:
        cv2.imwrite(str(rectified_path), info["rectified"])
    else:
        rectified_path = None

    if info.get("rectified_enhanced") is not None:
        cv2.imwrite(str(enhanced_path), info["rectified_enhanced"])

    if info.get("rectified_binary") is not None:
        cv2.imwrite(str(binary_path), info["rectified_binary"])
    else:
        binary_path = None

    for digit_index, digit_image in enumerate(info.get("digit_images", []), start=1):
        digit_path = debug_dir / f"{prefix}_D{digit_index}.png"
        cv2.imwrite(str(digit_path), digit_image)
        digit_paths.append(str(digit_path))

    return frame_path, preview_path, rectified_path, binary_path, digit_paths, timestamp


def run_test_project(
    camera_index,
    roi_file,
    interval_seconds,
    count,
    show_preview,
    use_raw,
    debug_dir,
    retry_attempts,
    retry_delay_seconds,
):
    debug_path = make_debug_dir(debug_dir) if debug_dir else None
    log_file = debug_path / "ocr_debug_log.csv" if debug_path else None

    with WindSpeedOCR(
        camera_index=camera_index,
        roi_file=roi_file,
        warmup_frames=30,
    ) as wind_reader:
        print("Wind speed OCR is running. Press Ctrl+C to stop.")

        sample_index = 0
        while count <= 0 or sample_index < count:
            sample_index += 1
            started_at = time.perf_counter()

            needs_debug_info = show_preview or debug_path is not None
            if needs_debug_info:
                if use_raw:
                    if retry_attempts > 1:
                        value, info = wind_reader.recognize_with_retry(
                            attempts=retry_attempts,
                            delay_seconds=retry_delay_seconds,
                            return_debug=True,
                        )
                    else:
                        value, info = wind_reader.recognize(return_debug=True)
                    raw_value = value
                    raw_text = info["text"] if info["text"] else "--"
                    stable_text = raw_text
                    stable_value = value
                else:
                    if retry_attempts > 1:
                        value, info = wind_reader.recognize_stable_with_retry(
                            attempts=retry_attempts,
                            delay_seconds=retry_delay_seconds,
                            return_debug=True,
                        )
                    else:
                        value, info = wind_reader.recognize_stable(
                            return_debug=True)
                    stable_value = value
                    raw_value = info["raw_value"]
                    raw_text = info["raw_text"] if info["raw_text"] else "--"
                    stable_text = info["text"] if info["text"] else "--"
                if show_preview and info["preview"] is not None:
                    cv2.imshow("Wind Speed OCR Debug", info["preview"])
                    cv2.waitKey(1)
                confidence = info["confidence"]
            else:
                if use_raw:
                    if retry_attempts > 1:
                        value = wind_reader.recognize_with_retry(
                            attempts=retry_attempts,
                            delay_seconds=retry_delay_seconds,
                        )
                    else:
                        value = wind_reader.recognize()
                else:
                    if retry_attempts > 1:
                        value = wind_reader.recognize_stable_with_retry(
                            attempts=retry_attempts,
                            delay_seconds=retry_delay_seconds,
                        )
                    else:
                        value = wind_reader.recognize_stable()
                stable_value = value
                raw_value = value
                raw_text = str(value) if value is not None else "--"
                stable_text = raw_text
                confidence = None

            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            value_text = f"{value:.2f}" if value is not None else "(empty)"
            stable_value_text = f"{stable_value:.2f}" if stable_value is not None else "(empty)"
            raw_value_text = f"{raw_value:.2f}" if raw_value is not None else "(empty)"
            frame_path = None
            preview_path = None
            rectified_path = None
            binary_path = None
            digit_paths = []
            timestamp = datetime.now().isoformat(timespec="milliseconds")

            if debug_path is not None:
                frame_path, preview_path, rectified_path, binary_path, digit_paths, timestamp = save_debug_sample(
                    debug_path,
                    sample_index,
                    info,
                )
                write_debug_row(
                    log_file,
                    {
                        "sample": sample_index,
                        "timestamp": timestamp,
                        "mode": "raw" if use_raw else "stable",
                        "wind": value_text,
                        "stable": stable_value_text,
                        "raw": raw_value_text,
                        "text": stable_text,
                        "stable_text": stable_text,
                        "raw_text": raw_text,
                        "confidence": "" if confidence is None else f"{confidence:.4f}",
                        "elapsed_ms": f"{elapsed_ms:.1f}",
                        "failure_reason": info.get("failure_reason", ""),
                        "retry_attempts": info.get("retry_attempts", 1),
                        "retry_selected_attempt": info.get("retry_selected_attempt", 1),
                        "retry_candidates": info.get("retry_candidates", ""),
                        "digit_box_summary": info.get("digit_box_summary", ""),
                        "digit_summary": info.get("digit_summary", ""),
                        "frame_path": str(frame_path),
                        "preview_path": "" if preview_path is None else str(preview_path),
                        "rectified_path": "" if rectified_path is None else str(rectified_path),
                        "binary_path": "" if binary_path is None else str(binary_path),
                        "digit_paths": " | ".join(digit_paths),
                    },
                )

            if confidence is None:
                print(
                    f"[{sample_index}] mode={'raw' if use_raw else 'stable'} "
                    f"wind={value_text} stable={stable_value_text} raw={raw_value_text} "
                    f"stable_text={stable_text} time={elapsed_ms:.1f}ms"
                )
            else:
                print(
                    f"[{sample_index}] mode={'raw' if use_raw else 'stable'} "
                    f"wind={value_text} stable={stable_value_text} raw={raw_value_text} "
                    f"stable_text={stable_text} raw_text={raw_text} "
                    f"conf={confidence:.2f} reason={info.get('failure_reason', '')} "
                    f"retry={info.get('retry_selected_attempt', 1)}/{info.get('retry_attempts', 1)} "
                    f"time={elapsed_ms:.1f}ms"
                )

            sleep_seconds = interval_seconds - \
                (time.perf_counter() - started_at)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Test main project for the 7-segment wind speed OCR module.")
    parser.add_argument("--camera", type=int, default=1,
                        help="OpenCV camera index.")
    parser.add_argument("--roi-file", default=ROI_FILE,
                        help="Path to ROI JSON.")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between OCR reads.")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of reads. 0 means forever.")
    parser.add_argument("--preview", action="store_true",
                        help="Show debug preview window.")
    parser.add_argument("--select-roi", action="store_true",
                        help="Select ROI before running.")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw single-frame OCR instead of stable value.")
    parser.add_argument("--retry", type=int, default=1,
                        help="Fresh-frame OCR attempts per read.")
    parser.add_argument("--retry-delay", type=float,
                        default=0.10, help="Seconds between retry frames.")
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Save each captured frame, preview, and CSV log to this directory.",
    )
    args = parser.parse_args()

    if args.select_roi:
        select_and_save_roi(camera_index=args.camera, roi_file=args.roi_file)

    try:
        run_test_project(
            camera_index=args.camera,
            roi_file=args.roi_file,
            interval_seconds=args.interval,
            count=args.count,
            show_preview=args.preview,
            use_raw=args.raw,
            debug_dir=args.debug_dir,
            retry_attempts=args.retry,
            retry_delay_seconds=args.retry_delay,
        )
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
