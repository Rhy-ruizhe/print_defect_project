import os
import random
import shutil

IMAGE_DIR = "data/raw_images"
LABEL_DIR = "data/yolo_dataset/labels_raw"
OUTPUT_BASE = "data/yolo_dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42


def find_image_file(base_name):
    """
    根据不带扩展名的文件名，在 IMAGE_DIR 中查找对应图片。
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def make_dirs():
    """
    创建 YOLO 所需的目录结构。
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_BASE, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, "labels", split), exist_ok=True)


def clear_old_split_files():
    """
    可选：清空旧的拆分结果，避免重复文件残留。
    """
    for split in ["train", "val", "test"]:
        image_split_dir = os.path.join(OUTPUT_BASE, "images", split)
        label_split_dir = os.path.join(OUTPUT_BASE, "labels", split)

        for folder in [image_split_dir, label_split_dir]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)


def main():
    random.seed(RANDOM_SEED)
    make_dirs()
    clear_old_split_files()

    # 只读取真正的标签文件，跳过 classes.txt
    label_files = [
        f for f in os.listdir(LABEL_DIR)
        if f.lower().endswith(".txt") and f.lower() != "classes.txt"
    ]

    valid_samples = []

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        image_file = find_image_file(base_name)

        if image_file is None:
            print(f"Warning: no image found for label {label_file}")
            continue

        label_path = os.path.join(LABEL_DIR, label_file)
        valid_samples.append((image_file, label_path, base_name))

    random.shuffle(valid_samples)

    total = len(valid_samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_samples = valid_samples[:train_end]
    val_samples = valid_samples[train_end:val_end]
    test_samples = valid_samples[val_end:]

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split_name, samples in splits.items():
        for image_path, label_path, base_name in samples:
            image_filename = os.path.basename(image_path)
            label_filename = os.path.basename(label_path)

            dst_image = os.path.join(
                OUTPUT_BASE, "images", split_name, image_filename)
            dst_label = os.path.join(
                OUTPUT_BASE, "labels", split_name, label_filename)

            shutil.copy2(image_path, dst_image)
            shutil.copy2(label_path, dst_label)

    print(f"Total valid samples: {total}")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")

    # 检查比例总和
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    print(f"Ratio sum: {ratio_sum}")

    if abs(ratio_sum - 1.0) > 1e-6:
        print("Warning: TRAIN_RATIO + VAL_RATIO + TEST_RATIO is not equal to 1.0")


if __name__ == "__main__":
    main()
