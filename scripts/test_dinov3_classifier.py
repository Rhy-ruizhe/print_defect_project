from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from train_dinov3_classifier import DinoV3Classifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "defect_dinov3_classifier.pth"
IMAGE_PATH = PROJECT_ROOT / "data" / "cropped_images" / "cam1_20260327_111848.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a single image with the trained DINOv3 defect classifier."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--image-path", type=Path, default=IMAGE_PATH)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def class_names_from_checkpoint(checkpoint: dict) -> list[str]:
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict):
        return ["defect", "no_defect"]

    idx_to_class = {class_index: class_name for class_name, class_index in class_to_idx.items()}
    return [idx_to_class[index] for index in range(len(idx_to_class))]


def load_model(checkpoint: dict, class_names: list[str]) -> nn.Module:
    model_name = checkpoint.get("model_name")
    if not isinstance(model_name, str):
        raise RuntimeError("Checkpoint is missing model_name.")

    freeze_backbone = bool(checkpoint.get("freeze_backbone", True))
    model = DinoV3Classifier(
        model_name=model_name,
        num_classes=len(class_names),
        freeze_backbone=freeze_backbone,
    )

    if freeze_backbone:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    model_path = resolve_project_path(args.model_path)
    image_path = resolve_project_path(args.image_path)

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 1

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return 1

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = class_names_from_checkpoint(checkpoint)
    image_size = int(checkpoint.get("image_size", 320))

    model = load_model(checkpoint, class_names)
    transform = make_transform(image_size)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_index = torch.argmax(probabilities, dim=1).item()

    print("Image:", image_path)
    print("Prediction:", class_names[pred_index])
    print("Confidence:", probabilities[0][pred_index].item())
    print("Probabilities:")
    for class_index, class_name in enumerate(class_names):
        print(f"  {class_name}: {probabilities[0][class_index].item():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
