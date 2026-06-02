from ultralytics import YOLO

MODEL_PATH = "runs/detect/results/yolo_print_object3/weights/best.pt"
IMAGE_PATH = "data/validate/cam1_20260519_113058.jpg"   # 自己的某张图


def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=IMAGE_PATH,
        conf=0.3,
        save=True,
        project="results",
        name="yolo_test"
    )
    print("Prediction finished.")


if __name__ == "__main__":
    main()
