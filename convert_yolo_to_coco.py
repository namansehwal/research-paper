# scripts/convert_yolo_to_coco.py
import os
import json
from PIL import Image
from tqdm import tqdm

def convert_yolo_to_coco(images_dir, labels_dir, output_json, class_names):
    """
    Convert YOLO annotations to COCO format.

    Args:
        images_dir (str): Directory containing images.
        labels_dir (str): Directory containing YOLO label files.
        output_json (str): Path to save the COCO JSON file.
        class_names (list): List of class names.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for idx, class_name in enumerate(class_names):
        coco['categories'].append({
            "id": idx,
            "name": class_name,
            "supercategory": "none"
        })

    annotation_id = 1
    for img_id, img_name in enumerate(tqdm(os.listdir(images_dir), desc="Converting Images")):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(images_dir, img_name)
        try:
            image = Image.open(img_path)
            width, height = image.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        coco['images'].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue  # No objects in this image

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_path}: {line}")
                continue
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])

            # Convert to absolute coordinates
            x_center_abs = x_center * width
            y_center_abs = y_center * height
            bbox_width_abs = bbox_width * width
            bbox_height_abs = bbox_height * height
            x_min = x_center_abs - (bbox_width_abs / 2)
            y_min = y_center_abs - (bbox_height_abs / 2)

            # Ensure bbox is within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            bbox_width_abs = min(bbox_width_abs, width - x_min)
            bbox_height_abs = min(bbox_height_abs, height - y_min)

            coco['annotations'].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, bbox_width_abs, bbox_height_abs],
                "area": bbox_width_abs * bbox_height_abs,
                "iscrowd": 0
            })
            annotation_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"COCO annotations saved to {output_json}")

if __name__ == "__main__":
    images_directory_train = 'dataset/yolov8/images/train_augmented'
    labels_directory_train = 'dataset/yolov8/labels/train_augmented'
    output_json_train = 'dataset/detectron2/annotations_train.json'
    class_names = ['unripe', 'ripe', 'overripe']

    images_directory_val = 'dataset/yolov8/images/val'
    labels_directory_val = 'dataset/yolov8/labels/val'
    output_json_val = 'dataset/detectron2/annotations_val.json'

    os.makedirs('dataset/detectron2', exist_ok=True)

    convert_yolo_to_coco(
        images_dir=images_directory_train,
        labels_dir=labels_directory_train,
        output_json=output_json_train,
        class_names=class_names
    )

    convert_yolo_to_coco(
        images_dir=images_directory_val,
        labels_dir=labels_directory_val,
        output_json=output_json_val,
        class_names=class_names
    )
