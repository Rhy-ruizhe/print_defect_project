import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/defect_classifier.pth"
# the real image in your folder
IMAGE_PATH = "data/cropped_images/cam1_20260327_111848.jpg"
IMAGE_SIZE = 320

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["defect", "no_defect"]


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found: {IMAGE_PATH}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print("Prediction:", CLASS_NAMES[pred])
    print("Confidence:", probs[0][pred].item())


if __name__ == "__main__":
    main()
