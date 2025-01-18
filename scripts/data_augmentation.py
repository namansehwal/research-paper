# scripts/data_augmentation.py
import os
import shutil
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    ShiftScaleRotate, Blur, GaussNoise, HueSaturationValue
)
from PIL import Image
import numpy as np
from tqdm import tqdm

def augment_images(input_dir, output_dir, augmentations, augment_factor=2):
    """
    Augment images and save to the output directory.

    Args:
        input_dir (str): Path to the input directory containing class subdirectories.
        output_dir (str): Path to save augmented images.
        augmentations (albumentations.Compose): Augmentation pipeline.
        augment_factor (int): Number of augmented images per original image.
    """
    classes = os.listdir(input_dir)
    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)
        images = os.listdir(cls_input)
        for img_name in tqdm(images, desc=f"Augmenting {cls}"):
            img_path = os.path.join(cls_input, img_name)
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            for i in range(augment_factor):
                augmented = augmentations(image=image)
                augmented_image = augmented['image']
                augmented_pil = Image.fromarray(augmented_image)
                base_name, ext = os.path.splitext(img_name)
                new_name = f"{base_name}_aug_{i}{ext}"
                augmented_pil.save(os.path.join(cls_output, new_name))

if __name__ == "__main__":
    input_directory = 'dataset/original'
    augmented_directory = 'dataset/augmented'

    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=40, p=0.7),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        Blur(p=0.3),
        GaussNoise(p=0.3),
    ])

    augment_images(input_directory, augmented_directory, augmentation_pipeline, augment_factor=2)
