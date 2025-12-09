import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
import numpy as np

def load_audio_file(file_path: str) -> torch.Tensor:
    """Load an audio file and return the waveform."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return torch.zeros(1, 16000)  # Return a zero tensor if loading fails

def extract_features(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Extract audio features using Mel Spectrogram."""
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    return mel_spectrogram

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have zero mean and unit variance."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def augment_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Apply random audio augmentations."""
    # Example augmentation: Add Gaussian noise
    noise = torch.randn_like(waveform) * 0.005
    return waveform + noise

class AudioDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[callable] = None):
        """
        Args:
            file_paths (List[str]): List of file paths to audio files.
            labels (List[int]): List of labels corresponding to each audio file.
            transform (Optional[callable]): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform = load_audio_file(file_path)
        sample_rate = 16000  # Assuming a fixed sample rate for simplicity

        features = extract_features(waveform, sample_rate)
        features = normalize(features)

        if self.transform:
            features = self.transform(features)

        return features, label

def create_dataloaders(file_paths: List[str], labels: List[int], batch_size: int, num_workers: int = 4, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        file_paths (List[str]): List of file paths to audio files.
        labels (List[int]): List of labels corresponding to each audio file.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        validation_split (float): Fraction of data to use for validation.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    dataset = AudioDataset(file_paths, labels, transform=augment_audio)

    # Split dataset into train and validation
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# Example usage:
# file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
# labels = [0, 1, ...]  # Corresponding labels
# train_loader, val_loader = create_dataloaders(file_paths, labels, batch_size=32)