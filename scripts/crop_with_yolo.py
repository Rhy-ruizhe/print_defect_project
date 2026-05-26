from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "runs/detect/results/yolo_print_object3/weights/best.pt"
INPUT_DIR = "data/raw_images_for_crop"
OUTPUT_DIR = "data/cropped_images"

CONF_THRES = 0.3
PADDING = 20  # 框四周额外留一点，避免裁太紧
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    if not os.path.exists(INPUT_DIR):
        print(f"Input folder not found: {INPUT_DIR}")
        return

    model = YOLO(MODEL_PATH)

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        print("No images found in input folder.")
        return

    print(f"Found {len(image_files)} images.")

    for image_name in image_files:
        image_path = os.path.join(INPUT_DIR, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_name}")
            continue

        h, w = image.shape[:2]

        results = model.predict(
            source=image_path,
            conf=CONF_THRES,
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            print(f"No object detected: {image_name}")
            continue

        # 取置信度最高的那个框
        best_box = None
        best_conf = -1.0

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)

        # 加 padding
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"Empty crop: {image_name}")
            continue

        save_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(save_path, cropped)

        print(f"Cropped: {image_name} | conf={best_conf:.3f}")

    print("All done.")


if __name__ == "__main__":
    main()
