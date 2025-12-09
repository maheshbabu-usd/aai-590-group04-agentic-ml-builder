import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

class ImageClassificationDataset(Dataset):
    """
    A custom Dataset class for loading image data for classification tasks.

    Attributes:
        file_paths (List[str]): List of file paths or URLs to the images.
        labels (List[int]): List of labels corresponding to the images.
        transform (Optional[transforms.Compose]): Transformations to apply to the images.
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        """
        Initializes the dataset with file paths, labels, and transformations.

        Args:
            file_paths (List[str]): List of file paths or URLs to the images.
            labels (List[int]): List of labels corresponding to the images.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        assert len(file_paths) == len(labels), "The number of file paths must match the number of labels."
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the image tensor and its label.
        """
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                response = requests.get(img_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), label  # Return a dummy tensor and the label

def create_dataloaders(file_paths: List[str], labels: List[int], batch_size: int, train_split: float = 0.8, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders.

    Args:
        file_paths (List[str]): List of file paths or URLs to the images.
        labels (List[int]): List of labels corresponding to the images.
        batch_size (int): Number of samples per batch.
        train_split (float): Proportion of the dataset to include in the train split.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    dataset = ImageClassificationDataset(file_paths, labels, transform=None)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Assign transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Example file paths and labels
    file_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'http://example.com/image3.jpg']
    labels = [0, 1, 0]

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(file_paths, labels, batch_size=32)

    # Iterate through the DataLoader
    for images, labels in train_loader:
        print(images.shape, labels)