Below is a complete `conftest.py` file with reusable fixtures for testing a machine learning classification project using Pytest. This setup includes fixtures for device selection, sample data creation, model instantiation, data loading, optimizer setup, loss function, and temporary directory creation for model artifacts.

```python
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

# Fixture to determine the device (CPU or GPU)
@pytest.fixture(scope='session')
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixture to create mock training data
@pytest.fixture(scope='session')
def sample_data():
    # Create random data for a binary classification problem
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randint(0, 2, (100,))  # Binary targets
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

    return SimpleClassifier(input_dim=10, output_dim=2)

# Fixture to create a sample DataLoader
@pytest.fixture(scope='session')
def dataloader(sample_data):
    X, y = sample_data
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)

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
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        # Cleanup is handled automatically

# Example usage of the fixtures in a test function
def test_training_step(device, model, dataloader, optimizer, criterion):
    model.to(device)
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    assert True  # Add meaningful assertions based on your test requirements
```

### Explanation:
- **Device Fixture**: Determines whether to use a GPU or CPU for computations.
- **Sample Data Fixture**: Generates random data for a binary classification task.
- **Model Fixture**: Defines a simple linear classifier model.
- **DataLoader Fixture**: Creates a DataLoader for the sample data.
- **Optimizer Fixture**: Sets up a stochastic gradient descent optimizer.
- **Criterion Fixture**: Uses cross-entropy loss, suitable for classification tasks.
- **Temporary Model Directory Fixture**: Provides a temporary directory for storing model artifacts during tests, automatically cleaned up after use.

This setup provides a robust foundation for testing various components of a machine learning pipeline in a classification project.