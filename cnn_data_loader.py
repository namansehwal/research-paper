# scripts/cnn_data_loader.py
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_data_loaders(data_dir, batch_size=32, img_size=224, split_ratio=0.8):
    """
    Create training and validation data loaders.

    Args:
        data_dir (str): Path to the dataset directory with class subdirectories.
        batch_size (int): Batch size.
        img_size (int): Image size for resizing.
        split_ratio (float): Ratio of training data.

    Returns:
        train_loader, val_loader: Data loaders for training and validation.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
