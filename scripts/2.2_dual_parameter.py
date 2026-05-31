import csv
import time
from datetime import datetime
from pathlib import Path

import requests
from util_ocr_7seg.ocr_test_7seg import CAMERA_INDEX, ROI_FILE, WindSpeedOCR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://192.168.0.101"
API_KEY = "E9FBFC206F274E6A8927EAD651D06753"
READ_INTERVAL_SECONDS = 5
OCR_RETRY_ATTEMPTS = 5
OCR_RETRY_DELAY_SECONDS = 0.10
CSV_LOG_PATH = PROJECT_ROOT / "data" / "wind_speed_dual_parameter_log.csv"
CSV_FIELDS = [
    "timestamp",
    "ocr_status",
    "wind_speed_m_s",
    "raw_text",
    "confidence",
    "failure_reason",
    "previous_feedrate",
    "target_feedrate",
    "final_feedrate",
    "feedrate_status",
    "feedrate_error",
    "previous_flowrate",
    "target_flowrate",
    "final_flowrate",
    "flowrate_status",
    "flowrate_error",
]

HEADERS = {
    "X-Api-Key": API_KEY,
    "Content-Type": "application/json",
}


def set_feedrate(factor: int) -> None:
    """Send a feedrate override command to OctoPrint."""
    payload = {
        "command": "feedrate",
        "factor": factor,
    }

    response = requests.post(
        f"{BASE_URL}/api/printer/printhead",
        headers=HEADERS,
        json=payload,
        timeout=5,
    )
    response.raise_for_status()
    print(f"Feedrate set to {factor}%. HTTP status: {response.status_code}")


def set_flowrate(factor: int) -> None:
    """Send a flowrate override command to OctoPrint."""
    payload = {
        "command": "flowrate",
        "factor": factor,
    }

    response = requests.post(
        f"{BASE_URL}/api/printer/tool",
        headers=HEADERS,
        json=payload,
        timeout=5,
    )
    response.raise_for_status()
    print(f"Flowrate set to {factor}%. HTTP status: {response.status_code}")


def append_csv_log(
    ocr_status: str,
    wind_speed: float | None,
    info: dict,
    previous_feedrate: int,
    target_feedrate: int | None,
    final_feedrate: int,
    feedrate_status: str,
    feedrate_error: str,
    previous_flowrate: int,
    target_flowrate: int | None,
    final_flowrate: int,
    flowrate_status: str,
    flowrate_error: str,
) -> None:
    """Append one OCR/control result to the dual-parameter CSV log."""
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = (
        not CSV_LOG_PATH.exists() or CSV_LOG_PATH.stat().st_size == 0
    )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ocr_status": ocr_status,
        "wind_speed_m_s": "" if wind_speed is None else f"{wind_speed:.2f}",
        "raw_text": info["text"] or "",
        "confidence": f"{info['confidence']:.4f}",
        "failure_reason": info["failure_reason"] or "",
        "previous_feedrate": previous_feedrate,
        "target_feedrate": "" if target_feedrate is None else target_feedrate,
        "final_feedrate": final_feedrate,
        "feedrate_status": feedrate_status,
        "feedrate_error": feedrate_error,
        "previous_flowrate": previous_flowrate,
        "target_flowrate": "" if target_flowrate is None else target_flowrate,
        "final_flowrate": final_flowrate,
        "flowrate_status": flowrate_status,
        "flowrate_error": flowrate_error,
    }

    with CSV_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def get_target_parameters(
    wind_speed: float,
    current_flowrate: int,
) -> tuple[int, int]:
    if wind_speed <= 5.4:
        target_feedrate = 100
        minimum_flowrate = 100
    elif wind_speed > 10.8:
        target_feedrate = 30
        minimum_flowrate = 200
    else:
        target_feedrate = 60
        minimum_flowrate = 150

    return target_feedrate, max(current_flowrate, minimum_flowrate)


def describe_parameter_decision(
    wind_speed: float,
    target_feedrate: int,
    target_flowrate: int,
) -> None:
    if wind_speed <= 5.4:
        print(
            f"Wind speed {wind_speed:.2f} <= 5.4, setting feedrate to "
            f"{target_feedrate}% and keeping flowrate at {target_flowrate}%."
        )
    elif 5.4 < wind_speed <= 10.8:
        print(
            f"Wind speed {wind_speed:.2f} is between 5.4 and 10.8, "
            f"setting feedrate to {target_feedrate}% and keeping flowrate "
            f"at least 150%."
        )
    else:
        print(
            f"Wind speed {wind_speed:.2f} > 10.8, setting feedrate to "
            f"{target_feedrate}% and raising flowrate to {target_flowrate}%."
        )


def update_feedrate_if_needed(
    target_feedrate: int,
    current_feedrate: int,
) -> tuple[int, str, str]:
    if target_feedrate == current_feedrate:
        print(f"Feedrate already at {current_feedrate}%.")
        return current_feedrate, "unchanged", ""

    try:
        set_feedrate(target_feedrate)
        return target_feedrate, "updated", ""
    except requests.RequestException as exc:
        print(f"Failed to update feedrate: {exc}")
        return current_feedrate, "update_failed", str(exc)


def update_flowrate_if_needed(
    target_flowrate: int,
    current_flowrate: int,
) -> tuple[int, str, str]:
    if target_flowrate == current_flowrate:
        print(f"Flowrate already at {current_flowrate}%.")
        return current_flowrate, "unchanged", ""

    try:
        set_flowrate(target_flowrate)
        return target_flowrate, "updated", ""
    except requests.RequestException as exc:
        print(f"Failed to update flowrate: {exc}")
        return current_flowrate, "update_failed", str(exc)


def restore_parameters_and_stop() -> None:
    print(
        "\nCtrl+C detected. Restoring feedrate and flowrate to 100% "
        "before shutdown..."
    )
    try:
        set_feedrate(100)
    except requests.RequestException as exc:
        print(f"Failed to restore feedrate to 100%: {exc}")

    try:
        set_flowrate(100)
    except requests.RequestException as exc:
        print(f"Failed to restore flowrate to 100%: {exc}")

    print("Waiting 5 seconds before fully stopping...")
    time.sleep(5)
    print("Program stopped by user.")


def main() -> None:
    current_feedrate = 100
    current_flowrate = 100
    print("Wind-speed OCR feedrate/flowrate control started.")
    print(
        f"Wind speed will be read from OCR every {READ_INTERVAL_SECONDS} seconds."
    )
    print(f"CSV log will be saved to: {CSV_LOG_PATH}")
    print("Press Ctrl+C to stop the program.")

    try:
        with WindSpeedOCR(
            camera_index=CAMERA_INDEX,
            roi_file=ROI_FILE,
            warmup_frames=30,
        ) as reader:
            while True:
                wind_speed, info = reader.recognize_with_retry(
                    attempts=OCR_RETRY_ATTEMPTS,
                    delay_seconds=OCR_RETRY_DELAY_SECONDS,
                    return_debug=True,
                )

                previous_feedrate = current_feedrate
                previous_flowrate = current_flowrate
                target_feedrate = None
                target_flowrate = None
                feedrate_status = "not_applicable"
                flowrate_status = "not_applicable"
                feedrate_error = ""
                flowrate_error = ""

                if wind_speed is None:
                    print(
                        "OCR failed to read wind speed. "
                        f"raw={info['text'] or '--'} "
                        f"conf={info['confidence']:.2f} "
                        f"reason={info['failure_reason'] or 'unknown'}"
                    )
                else:
                    print(
                        f"\nOCR wind speed: {wind_speed:.2f} m/s "
                        f"(raw={info['text']}, conf={info['confidence']:.2f})"
                    )
                    target_feedrate, target_flowrate = get_target_parameters(
                        wind_speed,
                        current_flowrate,
                    )
                    describe_parameter_decision(
                        wind_speed,
                        target_feedrate,
                        target_flowrate,
                    )

                    (
                        current_feedrate,
                        feedrate_status,
                        feedrate_error,
                    ) = update_feedrate_if_needed(
                        target_feedrate,
                        current_feedrate,
                    )
                    (
                        current_flowrate,
                        flowrate_status,
                        flowrate_error,
                    ) = update_flowrate_if_needed(
                        target_flowrate,
                        current_flowrate,
                    )

                append_csv_log(
                    ocr_status="failed" if wind_speed is None else "ok",
                    wind_speed=wind_speed,
                    info=info,
                    previous_feedrate=previous_feedrate,
                    target_feedrate=target_feedrate,
                    final_feedrate=current_feedrate,
                    feedrate_status=feedrate_status,
                    feedrate_error=feedrate_error,
                    previous_flowrate=previous_flowrate,
                    target_flowrate=target_flowrate,
                    final_flowrate=current_flowrate,
                    flowrate_status=flowrate_status,
                    flowrate_error=flowrate_error,
                )

                print(
                    f"Waiting {READ_INTERVAL_SECONDS} seconds before next OCR read..."
                )
                time.sleep(READ_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        restore_parameters_and_stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Unhandled error in 2.2_dual_parameter.py: {exc}")
        input("Press Enter to exit...")
