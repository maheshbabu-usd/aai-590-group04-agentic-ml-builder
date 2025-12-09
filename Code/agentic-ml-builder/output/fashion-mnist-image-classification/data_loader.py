import os
import numpy as np
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import requests
from io import BytesIO

class CustomImageDataset(Dataset):
    """Custom Dataset for loading and preprocessing image data for classification tasks."""
    
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            file_paths (List[str]): List of file paths or URLs to the images.
            labels (List[int]): List of labels corresponding to each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, int]: Tuple containing the image tensor and its label.
        """
        try:
            # Load image from file or URL
            if self.file_paths[idx].startswith('http'):
                response = requests.get(self.file_paths[idx])
                img = Image.open(BytesIO(response.content)).convert('L')
            else:
                img = Image.open(self.file_paths[idx]).convert('L')
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
            
            label = self.labels[idx]
            return img, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None

def get_transforms(augment: bool = False) -> transforms.Compose:
    """Returns the transformation pipeline for the dataset."""
    transform_list = [
        transforms.ToTensor(),  # Converts to (C, H, W) and normalizes to [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1.0, 1.0]
    ]
    
    if augment:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ] + transform_list
    
    return transforms.Compose(transform_list)

def create_dataloaders(file_paths: List[str], labels: List[int], batch_size: int = 32, 
                       train_split: float = 0.8, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoader for training and validation datasets.

    Args:
        file_paths (List[str]): List of file paths or URLs to the images.
        labels (List[int]): List of labels corresponding to each image.
        batch_size (int): Number of samples per batch.
        train_split (float): Proportion of the dataset to include in the train split.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoader for training and validation datasets.
    """
    # Define transformations
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # Create dataset
    dataset = CustomImageDataset(file_paths, labels, transform=None)
    
    # Split dataset into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Assign transformations to datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Example file paths and labels
    file_paths = ['path/to/image1.png', 'path/to/image2.png']  # Replace with actual paths or URLs
    labels = [0, 1]  # Example labels
    
    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(file_paths, labels, batch_size=32, train_split=0.8, num_workers=4)
    
    # Iterate over DataLoader
    for images, labels in train_loader:
        print(images.shape, labels)