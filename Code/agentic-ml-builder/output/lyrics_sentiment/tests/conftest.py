Below is a complete `conftest.py` file for a regression machine learning project using PyTorch. This file includes reusable fixtures for testing purposes, such as device selection, sample data creation, model instantiation, data loading, optimizer setup, loss function selection, and temporary directory creation for model artifacts.

```python
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

# Fixture to determine the appropriate device (CPU or GPU)
@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixture to create mock training data
@pytest.fixture(scope='function')
def sample_data():
    # Generate random data for regression
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randn(100, 1)   # 100 samples, 1 target
    return X, y

# Fixture to create an instance of the model
@pytest.fixture(scope='function')
def model():
    # Simple linear regression model
    class SimpleRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    return SimpleRegressionModel(input_dim=10, output_dim=1)

# Fixture to create a sample dataloader
@pytest.fixture(scope='function')
def dataloader(sample_data):
    X, y = sample_data
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)

# Fixture to create an optimizer
@pytest.fixture(scope='function')
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)

# Fixture to create a loss function
@pytest.fixture(scope='function')
def criterion():
    return nn.MSELoss()

# Fixture to create a temporary directory for model artifacts
@pytest.fixture(scope='function')
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
        # Clean up is handled automatically

# Example test function using the fixtures
def test_training_step(device, model, dataloader, optimizer, criterion):
    model.to(device)
    model.train()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0  # Basic check to ensure loss is non-negative
```

### Explanation:
- **device**: Determines whether to use a GPU or CPU for computations.
- **sample_data**: Generates mock data for testing, simulating a regression problem.
- **model**: Defines a simple linear regression model using PyTorch's `nn.Module`.
- **dataloader**: Creates a `DataLoader` for batching the sample data.
- **optimizer**: Sets up a stochastic gradient descent optimizer for the model.
- **criterion**: Uses mean squared error as the loss function, suitable for regression tasks.
- **tmp_model_dir**: Provides a temporary directory for storing model artifacts during tests, automatically cleaned up after use.

This setup allows for easy and consistent testing of various components of an ML pipeline in a regression context.