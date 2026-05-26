# Print Defect Project

This repository contains Python scripts for print defect detection, YOLO training/inference, classifier training/testing, OCR-based wind-speed reading, and Raspberry Pi image receiving/control workflows.

## Repository Contents

```text
scripts/                  Source scripts
scripts/util_ocr_7seg/     7-segment OCR utilities
.gitignore                 Keeps local data, models, outputs, and virtualenv files out of Git
requirements.txt           Python dependencies
```

Large local assets are intentionally not committed to Git:

```text
data/                      Training and validation datasets
models/                    Trained classifier/model weights
runs/                      YOLO run outputs
results/                   Training results and metrics
*.pt, *.pth, *.onnx         Model weight/export files
```

## External Assets

Download the dataset and trained models from the project storage location, then place them in the project root using this structure:

```text
data/
  yolo_dataset/
    data.yaml
    images/
    labels/
  cls_dataset/
    train/
    val/

models/
  defect_classifier.pth
  defect_dinov3_classifier.pth

yolov8n.pt
```

Asset download links:

```text
Dataset: TODO
Model weights: TODO
```

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

## Common Commands

Train YOLO:

```powershell
python scripts\train_yolo.py
```

Test YOLO:

```powershell
python scripts\test_yolo.py
```

Train classifier:

```powershell
python scripts\train_classifier.py
```

Run defect controller:

```powershell
python scripts\0.0_controller.py
```

Run dual-parameter controller:

```powershell
python scripts\0.1_dual_parameter_controller.py
```

## Notes

Keep large datasets, generated outputs, and model weights in external storage such as Hugging Face, Google Drive, OneDrive, Kaggle, or a lab server. This keeps the GitHub repository small and easy to clone.
