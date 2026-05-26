import os

IMAGE_DIR = "data/raw_images"
LABEL_DIR = "data/yolo_dataset/labels_raw"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def main():
    image_basenames = set()
    label_basenames = set()

    # 收集所有图片文件名（不带扩展名）
    for filename in os.listdir(IMAGE_DIR):
        name, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            image_basenames.add(name)

    # 收集所有标签文件名（不带扩展名），跳过 classes.txt
    for filename in os.listdir(LABEL_DIR):
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".txt" and filename.lower() != "classes.txt":
            label_basenames.add(name)

    images_without_labels = sorted(image_basenames - label_basenames)
    labels_without_images = sorted(label_basenames - image_basenames)

    print(f"Total images: {len(image_basenames)}")
    print(f"Total labels (excluding classes.txt): {len(label_basenames)}")
    print("-" * 50)

    print(f"Images without labels: {len(images_without_labels)}")
    for name in images_without_labels[:20]:
        print("  ", name)

    print("-" * 50)

    print(f"Labels without images: {len(labels_without_images)}")
    for name in labels_without_images[:20]:
        print("  ", name)

    print("-" * 50)

    # 检查空标签文件，跳过 classes.txt
    empty_labels = []
    for filename in os.listdir(LABEL_DIR):
        if filename.lower().endswith(".txt") and filename.lower() != "classes.txt":
            path = os.path.join(LABEL_DIR, filename)
            if os.path.getsize(path) == 0:
                empty_labels.append(filename)

    print(f"Empty label files: {len(empty_labels)}")
    for name in empty_labels[:20]:
        print("  ", name)

    print("-" * 50)

    # 检查 classes.txt 是否存在
    classes_path = os.path.join(LABEL_DIR, "classes.txt")
    if os.path.exists(classes_path):
        print("classes.txt found.")
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        print("Classes in classes.txt:", classes)
    else:
        print("classes.txt not found.")

    print("-" * 50)
    print("Check finished.")


if __name__ == "__main__":
    main()
