"""Configuration for bird_classifier"""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Training and model configuration"""
    
    # Model
    model_type: str = "cnn"
    input_dim: int = 128
    hidden_dims: List[int] = None
    output_dim: int = 10
    dropout: float = 0.5
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.2
    num_workers: int = 4
    
    # Metrics
    metrics: List[str] = None
    
    # Paths
    data_path: str = "./data"
    model_save_path: str = "./models"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Device
    device: str = "cuda"  # Will auto-detect
    seed: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall']

config = Config()
