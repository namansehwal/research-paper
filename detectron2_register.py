# scripts/detectron2_register.py
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def register_datasets():
    """
    Register training and validation datasets.
    """
    register_coco_instances("banana_train", {}, "dataset/detectron2/annotations_train.json", "dataset/yolov8/images/train_augmented")
    register_coco_instances("banana_val", {}, "dataset/detectron2/annotations_val.json", "dataset/yolov8/images/val")

    # Verify registration
    banana_train_metadata = MetadataCatalog.get("banana_train")
    banana_val_metadata = MetadataCatalog.get("banana_val")
    print("Datasets registered successfully.")

if __name__ == "__main__":
    register_datasets()
