import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TextRegressionDataset(Dataset):
    def __init__(self, data_file: str, vocab: Optional[torchtext.vocab.Vocab] = None, 
                 max_length: int = 100, train: bool = True):
        """
        Initialize the dataset.

        :param data_file: Path to the CSV file containing the data.
        :param vocab: Pre-built vocabulary. If None, it will be built from the data.
        :param max_length: Maximum length for padding sequences.
        :param train: Flag to indicate if the dataset is for training or validation.
        """
        self.data = pd.read_csv(data_file)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_length = max_length
        self.train = train

        # Error handling for missing data
        if self.data.isnull().values.any():
            raise ValueError("Data contains missing values. Please clean the data before proceeding.")

        # Lowercasing
        self.data['text'] = self.data['text'].str.lower()

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(self.data['text'])
        else:
            self.vocab = vocab

        # Normalize targets
        self.scaler = MinMaxScaler()
        self.data['target'] = self.scaler.fit_transform(self.data[['target']])

    def build_vocab(self, texts: pd.Series) -> torchtext.vocab.Vocab:
        """
        Build vocabulary from a series of texts.

        :param texts: A pandas series containing text data.
        :return: A torchtext vocabulary object.
        """
        def yield_tokens(data_iter):
            for text in data_iter:
                yield self.tokenizer(text)

        return build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>", "<pad>"])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Retrieve a sample from the dataset.

        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the tokenized text tensor and the target value.
        """
        text = self.data.iloc[idx]['text']
        target = self.data.iloc[idx]['target']

        # Tokenization
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]

        # Padding
        if len(token_ids) < self.max_length:
            token_ids.extend([self.vocab["<pad>"]] * (self.max_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_length]

        # Convert to tensor
        text_tensor = torch.tensor(token_ids, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float)

        # Data augmentation for training set
        if self.train:
            text_tensor = self.augment_text(text_tensor)

        return text_tensor, target_tensor

    def augment_text(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform data augmentation on the text tensor.

        :param text_tensor: The input text tensor.
        :return: Augmented text tensor.
        """
        # Example augmentation: Randomly swap two tokens
        if len(text_tensor) > 1 and torch.rand(1).item() > 0.5:
            idx1, idx2 = np.random.choice(len(text_tensor), 2, replace=False)
            text_tensor[idx1], text_tensor[idx2] = text_tensor[idx2], text_tensor[idx1]
        return text_tensor

def create_dataloaders(data_file: str, batch_size: int = 32, max_length: int = 100, 
                       train_split: float = 0.8, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    :param data_file: Path to the CSV file containing the data.
    :param batch_size: Batch size for the dataloaders.
    :param max_length: Maximum length for padding sequences.
    :param train_split: Proportion of data to use for training.
    :param num_workers: Number of workers for data loading.
    :return: A tuple containing the train and validation dataloaders.
    """
    # Load the full dataset
    full_dataset = TextRegressionDataset(data_file, max_length=max_length, train=True)

    # Split into train and validation sets
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader

def collate_fn(batch: List[Tuple[torch.Tensor, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle padding of sequences in a batch.

    :param batch: A list of tuples containing text tensors and target values.
    :return: A tuple containing a batch of text tensors and a batch of target values.
    """
    texts, targets = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    targets_tensor = torch.tensor(targets, dtype=torch.float)
    return texts_padded, targets_tensor

# Example usage
if __name__ == "__main__":
    data_file_path = "path/to/your/data.csv"
    train_loader, val_loader = create_dataloaders(data_file_path, batch_size=32, max_length=100)

    for batch in train_loader:
        text_batch, target_batch = batch
        print(text_batch.shape, target_batch.shape)
        break