import os
import random
import shutil

SOURCE_DIR = "data/class_pool"
OUTPUT_DIR = "data/cls_dataset"

CLASSES = ["defect", "no_defect"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42


def make_output_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


def get_image_files(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def clear_output_dirs():
    """
    Optional:
    Clear old files in output folders before copying new ones.
    """
    if not os.path.exists(OUTPUT_DIR):
        return

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)


def split_one_class(class_name):
    class_input_dir = os.path.join(SOURCE_DIR, class_name)
    files = get_image_files(class_input_dir)

    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, split_files in split_map.items():
        for filename in split_files:
            src_path = os.path.join(class_input_dir, filename)
            dst_path = os.path.join(
                OUTPUT_DIR, split_name, class_name, filename)
            shutil.copy2(src_path, dst_path)

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")


def main():
    random.seed(RANDOM_SEED)

    # 检查类别文件夹是否存在
    for cls in CLASSES:
        class_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(class_dir):
            print(f"Class folder not found: {class_dir}")
            return

    make_output_dirs()
    clear_output_dirs()

    for cls in CLASSES:
        split_one_class(cls)

    print("\nDataset split finished.")


if __name__ == "__main__":
    main()
