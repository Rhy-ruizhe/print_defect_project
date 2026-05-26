import subprocess
import sys
import time
from pathlib import Path

DEFECT_CLASS = "defect"
CHECK_INTERVAL_SECONDS = 5
CONFIG_ERROR_RETURN_CODE = 1


def parse_prediction(output: str) -> str | None:
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("prediction:"):
            return line.split(":", 1)[1].strip().lower()
    return None


def run_classifier(trigger_script: Path, script_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(trigger_script)],
        capture_output=True,
        text=True,
        cwd=script_dir,
    )


def run_dual_parameter_control(
    dual_parameter_script: Path,
    script_dir: Path,
) -> None:
    print("Starting OCR wind-speed feedrate/flowrate control...")
    dual_parameter_result = subprocess.run(
        [sys.executable, str(dual_parameter_script)],
        cwd=script_dir,
    )
    if dual_parameter_result.returncode != 0:
        print(
            "Dual-parameter control script exited with code "
            f"{dual_parameter_result.returncode}."
        )


def main() -> None:
    try:
        script_dir = Path(__file__).resolve().parent
        trigger_script = script_dir / "3.2_test_classifier.py"
        dual_parameter_script = script_dir / "3.4_dual_parameter.py"

        if not trigger_script.exists():
            print(f"Classifier script not found: {trigger_script}")
            return

        if not dual_parameter_script.exists():
            print(f"Dual-parameter script not found: {dual_parameter_script}")
            return

        print("Defect dual-parameter controller started.")
        print(
            f"Latest Raspberry Pi image will be checked every "
            f"{CHECK_INTERVAL_SECONDS} seconds."
        )
        print(
            "Press Ctrl+C to stop before defect-triggered dual-parameter "
            "control starts."
        )

        while True:
            print("\nRunning defect check...")
            trigger_result = run_classifier(trigger_script, script_dir)

            if trigger_result.stdout:
                print(trigger_result.stdout, end="")

            if trigger_result.stderr:
                print("Classifier script error output:")
                print(trigger_result.stderr, end="")

            if trigger_result.returncode == CONFIG_ERROR_RETURN_CODE:
                print(
                    "Classifier configuration error. "
                    "Controller will stop instead of retrying."
                )
                return

            if trigger_result.returncode != 0:
                print(
                    f"Classifier script exited with code "
                    f"{trigger_result.returncode}. Retrying after "
                    f"{CHECK_INTERVAL_SECONDS} seconds."
                )
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            prediction = parse_prediction(trigger_result.stdout)

            if prediction is None:
                print(
                    "No prediction result found in classifier output. "
                    f"Retrying after {CHECK_INTERVAL_SECONDS} seconds."
                )
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if prediction == DEFECT_CLASS:
                print("Trigger detected: defect")
                run_dual_parameter_control(dual_parameter_script, script_dir)
                return

            print(
                "No defect trigger detected. "
                f"Waiting {CHECK_INTERVAL_SECONDS} seconds before next check."
            )
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nController stopped by user.")


if __name__ == "__main__":
    main()
