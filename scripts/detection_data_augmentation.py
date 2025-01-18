# scripts/detection_data_augmentation.py
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    ShiftScaleRotate, Blur, GaussNoise, HueSaturationValue
)
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from tqdm import tqdm

def augment_detection_images(images_dir, labels_dir, output_images_dir, output_labels_dir, augmentations, augment_factor=2):
    """
    Augment images and corresponding YOLO format labels.

    Args:
        images_dir (str): Path to input images.
        labels_dir (str): Path to input label files.
        output_images_dir (str): Path to save augmented images.
        output_labels_dir (str): Path to save augmented label files.
        augmentations (albumentations.Compose): Augmentation pipeline.
        augment_factor (int): Number of augmented images per original image.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = os.listdir(images_dir)
    for img_name in tqdm(image_files, desc="Augmenting Detection Images"):
        img_path = os.path.join(images_dir, img_name)
        base_name, ext = os.path.splitext(img_name)
        label_path = os.path.join(labels_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            continue  # Skip images without labels

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Read labels
        with open(label_path, 'r') as f:
            labels = f.readlines()

        bboxes = []
        category_ids = []
        for label in labels:
            parts = label.strip().split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_path}: {label}")
                continue
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))  # x_center, y_center, width, height
            bboxes.append(bbox)
            category_ids.append(class_id)

        for i in range(augment_factor):
            try:
                augmented = augmentations(image=image, bboxes=bboxes, class_labels=category_ids)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_class_ids = augmented['class_labels']
            except Exception as e:
                print(f"Error augmenting image {img_path}: {e}")
                continue

            augmented_pil = Image.fromarray(augmented_image)
            new_img_name = f"{base_name}_aug_{i}{ext}"
            augmented_pil.save(os.path.join(output_images_dir, new_img_name))

            # Save new labels
            with open(os.path.join(output_labels_dir, f"{base_name}_aug_{i}.txt"), 'w') as f:
                for cls_id, bbox in zip(augmented_class_ids, augmented_bboxes):
                    bbox_str = ' '.join([f"{coord:.6f}" for coord in bbox])
                    f.write(f"{cls_id} {bbox_str}\n")

if __name__ == "__main__":
    images_directory = 'dataset/yolov8/images/train'
    labels_directory = 'dataset/yolov8/labels/train'
    augmented_images_directory = 'dataset/yolov8/images/train_augmented'
    augmented_labels_directory = 'dataset/yolov8/labels/train_augmented'

    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=40, p=0.7),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        Blur(p=0.3),
        GaussNoise(p=0.3),
    ], bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})

    augment_detection_images(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_images_dir=augmented_images_directory,
        output_labels_dir=augmented_labels_directory,
        augmentations=augmentation_pipeline,
        augment_factor=2
    )
