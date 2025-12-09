Below is a complete `conftest.py` file for a classification project using Pytest. This file includes fixtures for device selection, sample data generation, model instantiation, data loader creation, optimizer setup, loss function, and a temporary directory for model artifacts.

```python
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import shutil

# Fixture to determine the device (CPU or GPU)
@pytest.fixture(scope='session')
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixture to create mock training data
@pytest.fixture(scope='session')
def sample_data():
    # Generate random data for a binary classification problem
    num_samples = 100
    num_features = 20
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, 2, (num_samples,))
    return X, y

# Fixture to create a simple model instance
@pytest.fixture(scope='session')
def model():
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleClassifier, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    return SimpleClassifier(input_dim=20, output_dim=2)

# Fixture to create a DataLoader
@pytest.fixture(scope='session')
def dataloader(sample_data):
    X, y = sample_data
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)

# Fixture to create an optimizer
@pytest.fixture(scope='session')
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)

# Fixture to create a loss function
@pytest.fixture(scope='session')
def criterion():
    return nn.CrossEntropyLoss()

# Fixture to create a temporary directory for model artifacts
@pytest.fixture(scope='function')
def tmp_model_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

# Example of a test function using the fixtures
def test_training_step(device, model, dataloader, optimizer, criterion):
    model.to(device)
    model.train()
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0, "Loss should be non-negative"
```

### Explanation:

1. **Device Fixture**: Determines if a GPU is available and returns the appropriate device.
2. **Sample Data Fixture**: Generates random data for a binary classification task with 100 samples and 20 features.
3. **Model Fixture**: Defines a simple linear classifier model.
4. **Dataloader Fixture**: Creates a DataLoader from the sample data with a batch size of 10.
5. **Optimizer Fixture**: Sets up an SGD optimizer for the model.
6. **Criterion Fixture**: Uses CrossEntropyLoss for classification tasks.
7. **Temporary Model Directory Fixture**: Creates and cleans up a temporary directory for storing model artifacts.

This setup allows you to write tests that can easily utilize these fixtures to test different aspects of your machine learning pipeline.