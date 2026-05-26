from ultralytics import YOLO


def main():
    # 加载预训练的 YOLOv8n 检测模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    model.train(
        data="data/yolo_dataset/data.yaml",   # 数据集配置文件
        epochs=100,                           # 训练轮数
        imgsz=640,                            # 输入图像尺寸
        batch=8,                              # batch size，显存不够可改小
        project="results",                    # 训练结果保存目录
        name="yolo_print_object",             # 本次训练的子文件夹名
        pretrained=True                       # 使用预训练权重
    )


if __name__ == "__main__":
    main()
