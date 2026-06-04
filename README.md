# Print Defect Detection and Parameter Control

This repository contains the code, trained models, YOLO run outputs, and datasets used for print defect detection and parameter-control experiments.

Single-Parameter Adjustment: https://youtu.be/-bbHQhmS76I
The video shows how the travel speed is changed during printing

Dual-Parameter Adjustment: https://youtu.be/FCghLTBxQnE
The video shows how the travel speed and extrusion rate are changed during printing

The project combines:

- YOLO-based object detection
- Image classification for defect / no-defect prediction
- OCR-based wind-speed reading
- Single-parameter feedrate control
- Dual-parameter feedrate / flowrate control
- Raspberry Pi image receiving workflows

## Repository Structure
<img width="7144" height="6858" alt="graduation  thesis - Frame 27" src="https://github.com/user-attachments/assets/2c80a95e-f78b-48da-86f6-5229f0ca69a4" />


```text
scripts/
  0.0_controller.py                 Main controller for single-parameter control
  0.1_dual_parameter_controller.py  Main controller for dual-parameter control
  1.1_classifier.py                 Defect detection pipeline
  2.1_single_parameter.py           Single-parameter control logic
  2.2_dual_parameter.py             Dual-parameter control logic
  4.1_rasp_server.py                Raspberry Pi image receiving server
  util_ocr_7seg/                    Seven-segment OCR utilities

data/
  cls_dataset/                      Classification dataset
  yolo_dataset/                     YOLO dataset

models/
  defect_classifier.pth             Trained defect classifier
  defect_dinov3_classifier.pth      Trained DINOv3 classifier

runs/
  detect/results/yolo_print_object3/
    weights/best.pt                 Trained YOLO model used by 1.1_classifier.py
    weights/last.pt                 Last YOLO training checkpoint
```

Large binary files such as images and model weights are tracked with Git LFS.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you clone this repository on a new machine, make sure Git LFS is installed before cloning or pulling the dataset/model files:

```powershell
git lfs install
git lfs pull
```

## Defect Detection Pipeline

The main detection script is:

```powershell
python scripts\1.1_classifier.py
```

It uses two trained models:

```text
runs/detect/results/yolo_print_object3/weights/best.pt
models/defect_classifier.pth
```

The YOLO model first localizes the printed object region. The classifier then predicts whether the cropped region is `defect` or `no_defect`.

The file `yolov8n.pt` is only the original YOLOv8 nano pretrained starting point. It is useful for training YOLO, but `1.1_classifier.py` uses the trained `best.pt` checkpoint instead.

## Controller Scripts

There are two top-level controller scripts. Both repeatedly run `1.1_classifier.py` to check the latest Raspberry Pi image. When the classifier predicts `defect`, the controller starts a parameter-control script.

### Single-Parameter Controller

Run:

```powershell
python scripts\0.0_controller.py
```

This controller:

- checks the latest image every 10 seconds
- runs `scripts/1.1_classifier.py`
- waits if no defect is detected
- starts `scripts/2.1_single_parameter.py` when a defect is detected

Use this mode when the experiment only adjusts one process parameter, currently the feedrate-related control path.

### Dual-Parameter Controller

Run:

```powershell
python scripts\0.1_dual_parameter_controller.py
```

This controller:

- checks the latest image every 5 seconds
- runs `scripts/1.1_classifier.py`
- waits if no defect is detected
- starts `scripts/2.2_dual_parameter.py` when a defect is detected

Use this mode when the experiment adjusts two process parameters together, currently the feedrate / flowrate control path.

### Main Difference

```text
0.0_controller.py
  defect detected -> 2.1_single_parameter.py
  one controlled parameter
  10-second check interval

0.1_dual_parameter_controller.py
  defect detected -> 2.2_dual_parameter.py
  two controlled parameters
  5-second check interval
```

## Training and Testing

Train YOLO:

```powershell
python scripts\train_yolo.py
```

Test YOLO:

```powershell
python scripts\test_yolo.py
```

Train the classifier:

```powershell
python scripts\train_classifier.py
```

Test the classifier:

```powershell
python scripts\test_classifier.py
```

## Raspberry Pi Image Server

The Raspberry Pi receiving workflow is implemented in:

```powershell
python scripts\4.1_rasp_server.py
```

This script receives image captured, stores the latest image, and provides the image source used by the detection/controller workflow.

## Notes

- `venv/`, cache files, and temporary files are ignored by Git.
- `data/`, `models/`, and `runs/` are included because they are required for reproducibility.
- Git LFS is required for large images and model weights.
