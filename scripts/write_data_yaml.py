import os

yaml_content = """path: data/yolo_dataset
train: images/train
val: images/val
test: images/test

names:
  0: print_object
"""

output_path = "data/yolo_dataset/data.yaml"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"data.yaml has been written to: {output_path}")
