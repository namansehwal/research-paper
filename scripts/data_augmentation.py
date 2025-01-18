# scripts/data_augmentation.py
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate,
    ShiftScaleRotate, Blur, GaussNoise, ElasticTransform,
    GridDistortion, OpticalDistortion, RandomCrop, Affine,
    CLAHE, Emboss, Cutout, CoarseDropout, PadIfNeeded
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
    augmented_directory = 'dataset/augmented/images'

    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=40, p=0.7),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.3),
        PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=(0, 0, 0), p=1.0),
        RandomCrop(height=256, width=256, p=0.5),
        Affine(scale=(0.9, 1.1), shear=(10, 10), p=0.3),
        CLAHE(clip_limit=4.0, p=0.3),
        Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.2)
    ])

    augment_images(input_directory, augmented_directory, augmentation_pipeline, augment_factor=20)
