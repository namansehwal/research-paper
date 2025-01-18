# scripts/yolov8_train.py
from ultralytics import YOLO
import os

def train_yolov8(data_yaml, epochs=50, batch=16, img_size=640, model_save_path='models/yolov8_best.pt'):
    """
    Train YOLOv8 model.

    Args:
        data_yaml (str): Path to the YAML file containing dataset paths and class names.
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        img_size (int): Image size for training.
        model_save_path (str): Path to save the best model.
    """
    # Initialize YOLOv8 model (yolov8s is the small version)
    model = YOLO('yolov8s.pt')  # You can choose other versions like yolov8m.pt, yolov8l.pt

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        cache=True
    )

    # Save the best model
    best_model = results.best_model
    os.rename(best_model, model_save_path)
    print(f"Best YOLOv8 model saved to {model_save_path}")

if __name__ == "__main__":
    # Create the YAML configuration for YOLOv8
    data_yaml_content = """
    train: dataset/yolov8/images/train_augmented
    val: dataset/yolov8/images/val
    # test: dataset/yolov8/images/test  # Optional

    nc: 3
    names: ['unripe', 'ripe', 'overripe']
    """

    # Save the YAML configuration
    os.makedirs('dataset/yolov8', exist_ok=True)
    with open('dataset/yolov8/banana.yaml', 'w') as f:
        f.write(data_yaml_content)

    train_yolov8(
        data_yaml='dataset/yolov8/banana.yaml',
        epochs=50,
        batch=16,
        img_size=640,
        model_save_path='models/yolov8_best.pt'
    )
