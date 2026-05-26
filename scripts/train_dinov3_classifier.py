from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "cls_dataset"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "defect_dinov3_classifier.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "dinov3_classifier"

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BATCH_SIZE = 8
EPOCHS = 20
IMAGE_SIZE = 320
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True
RANDOM_SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoV3Classifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = self._load_backbone(model_name)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = self._hidden_size()
        self.classifier = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def _load_backbone(model_name: str) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "This script requires Hugging Face Transformers. Install it with:\n"
                '  .\\venv\\Scripts\\python.exe -m pip install "transformers>=4.56.0"'
            ) from exc

        try:
            return AutoModel.from_pretrained(model_name)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load the DINOv3 backbone. If this is your first time "
                "using Meta's DINOv3 weights, open the model page on Hugging Face, "
                "accept the license/access terms, then run `huggingface-cli login`."
            ) from exc

    def _hidden_size(self) -> int:
        config = self.backbone.config

        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size

        embed_dim = getattr(config, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim

        hidden_sizes = getattr(config, "hidden_sizes", None)
        if hidden_sizes:
            return int(hidden_sizes[-1])

        raise RuntimeError("Could not infer DINOv3 feature dimension from config.")

    @staticmethod
    def _pool_outputs(outputs: object) -> torch.Tensor:
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)):
            hidden = outputs[0]

        if hidden is None:
            raise RuntimeError("DINOv3 output does not include features to pool.")

        if hidden.ndim == 3:
            return hidden[:, 0]

        if hidden.ndim == 4:
            return hidden.mean(dim=(-2, -1))

        if hidden.ndim == 2:
            return hidden

        raise RuntimeError(f"Unsupported DINOv3 feature shape: {tuple(hidden.shape)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)

        features = self._pool_outputs(outputs)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary defect classifier with a DINOv3 backbone."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--head-lr", type=float, default=HEAD_LEARNING_RATE)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune the DINOv3 backbone instead of training only the classifier head.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def make_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_transform, val_transform = make_transforms(image_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    print("Class to index mapping:", train_dataset.class_to_idx)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def make_optimizer(
    model: DinoV3Classifier,
    freeze_backbone: bool,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    if freeze_backbone:
        return optim.AdamW(
            model.classifier.parameters(),
            lr=head_lr,
            weight_decay=weight_decay,
        )

    return optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    collect_predictions: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    is_training = optimizer is not None
    model.train(is_training)

    loss_sum = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if collect_predictions:
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    return loss_sum / total, correct / total, all_labels, all_preds


def write_training_log(history: list[dict[str, float]], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    output_path: Path,
) -> np.ndarray:
    matrix = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return matrix


def write_metrics_report(
    output_path: Path,
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    best_val_acc: float,
    final_val_loss: float,
    final_val_acc: float,
    matrix: np.ndarray,
) -> None:
    report = classification_report(
        true_labels,
        pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        file.write(f"Final validation loss: {final_val_loss:.4f}\n")
        file.write(f"Final validation accuracy: {final_val_acc:.4f}\n\n")
        file.write("Confusion matrix rows=true, columns=predicted:\n")
        file.write(str(matrix))
        file.write("\n\nClassification report:\n")
        file.write(report)


def load_best_checkpoint(
    model: DinoV3Classifier,
    checkpoint_path: Path,
    freeze_backbone: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if freeze_backbone:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])


def save_checkpoint(
    model: DinoV3Classifier,
    save_path: Path,
    model_name: str,
    class_to_idx: dict[str, int],
    image_size: int,
    best_val_acc: float,
    freeze_backbone: bool,
) -> None:
    checkpoint = {
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "best_val_acc": best_val_acc,
        "freeze_backbone": freeze_backbone,
    }

    if freeze_backbone:
        checkpoint["classifier_state_dict"] = model.classifier.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()

    torch.save(checkpoint, save_path)


def main() -> int:
    args = parse_args()
    set_seed(RANDOM_SEED)

    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    save_path = args.save_path if args.save_path.is_absolute() else PROJECT_ROOT / args.save_path
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else PROJECT_ROOT / args.results_dir
    )
    freeze_backbone = not args.unfreeze_backbone

    os.makedirs(save_path.parent, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, class_to_idx = make_loaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DinoV3Classifier(
        model_name=args.model_name,
        num_classes=len(class_to_idx),
        freeze_backbone=freeze_backbone,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        model=model,
        freeze_backbone=freeze_backbone,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )

    print(f"Device: {DEVICE}")
    print(f"DINOv3 backbone: {args.model_name}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Results dir: {results_dir}")

    best_val_acc = 0.0
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        write_training_log(history, results_dir / "training_log.csv")

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                save_path=save_path,
                model_name=args.model_name,
                class_to_idx=class_to_idx,
                image_size=args.image_size,
                best_val_acc=best_val_acc,
                freeze_backbone=freeze_backbone,
            )
            print(f"Best model saved to {save_path}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    if history:
        plot_training_curves(history, results_dir / "training_curves.png")

    if save_path.exists():
        load_best_checkpoint(model, save_path, freeze_backbone)
        final_val_loss, final_val_acc, true_labels, pred_labels = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            collect_predictions=True,
        )

        idx_to_class = {
            class_index: class_name for class_name, class_index in class_to_idx.items()
        }
        class_names = [idx_to_class[index] for index in range(len(idx_to_class))]

        matrix = plot_confusion_matrix(
            true_labels=true_labels,
            pred_labels=pred_labels,
            class_names=class_names,
            output_path=results_dir / "confusion_matrix.png",
        )
        write_metrics_report(
            output_path=results_dir / "metrics.txt",
            true_labels=true_labels,
            pred_labels=pred_labels,
            class_names=class_names,
            best_val_acc=best_val_acc,
            final_val_loss=final_val_loss,
            final_val_acc=final_val_acc,
            matrix=matrix,
        )

        print(f"Training log saved to {results_dir / 'training_log.csv'}")
        print(f"Training curves saved to {results_dir / 'training_curves.png'}")
        print(f"Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")
        print(f"Metrics report saved to {results_dir / 'metrics.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
