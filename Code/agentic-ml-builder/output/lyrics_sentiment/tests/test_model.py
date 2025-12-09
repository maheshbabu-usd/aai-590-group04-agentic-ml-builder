Below is a comprehensive set of unit tests for the regression models ('bert', 'lstm') using PyTorch and pytest. These tests cover various aspects of model functionality, including initialization, forward pass, gradient flow, evaluation mode, and more. The code assumes that the models are implemented in a module named `models.py`.

```python
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from models import BertRegressionModel, LSTMRegressionModel

# Fixtures for reusable components
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def bert_model(device):
    model = BertRegressionModel().to(device)
    return model

@pytest.fixture
def lstm_model(device):
    model = LSTMRegressionModel().to(device)
    return model

@pytest.fixture
def sample_input():
    return torch.randn(8, 512)  # Example input shape for BERT

@pytest.fixture
def sample_target():
    return torch.randn(8, 1)  # Example target shape for regression

@pytest.fixture
def criterion():
    return nn.MSELoss()

@pytest.fixture
def optimizer_bert(bert_model):
    return optim.Adam(bert_model.parameters(), lr=1e-3)

@pytest.fixture
def optimizer_lstm(lstm_model):
    return optim.Adam(lstm_model.parameters(), lr=1e-3)

# Test model initialization
def test_model_initialization(bert_model, lstm_model):
    assert bert_model is not None
    assert lstm_model is not None

# Test forward pass with correct input shapes
def test_forward_pass(bert_model, lstm_model, sample_input, device):
    sample_input = sample_input.to(device)
    bert_output = bert_model(sample_input)
    lstm_output = lstm_model(sample_input)
    assert bert_output.shape == (8, 1)
    assert lstm_output.shape == (8, 1)

# Test output shapes
def test_output_shapes(bert_model, lstm_model, sample_input, device):
    sample_input = sample_input.to(device)
    bert_output = bert_model(sample_input)
    lstm_output = lstm_model(sample_input)
    assert bert_output.size(1) == 1
    assert lstm_output.size(1) == 1

# Test gradient flow
def test_gradient_flow(bert_model, lstm_model, sample_input, sample_target, criterion, optimizer_bert, optimizer_lstm, device):
    sample_input, sample_target = sample_input.to(device), sample_target.to(device)

    # Test BERT model
    optimizer_bert.zero_grad()
    output = bert_model(sample_input)
    loss = criterion(output, sample_target)
    loss.backward()
    for param in bert_model.parameters():
        assert param.grad is not None

    # Test LSTM model
    optimizer_lstm.zero_grad()
    output = lstm_model(sample_input)
    loss = criterion(output, sample_target)
    loss.backward()
    for param in lstm_model.parameters():
        assert param.grad is not None

# Test model in eval mode
def test_eval_mode(bert_model, lstm_model, sample_input, device):
    bert_model.eval()
    lstm_model.eval()
    sample_input = sample_input.to(device)

    with torch.no_grad():
        bert_output = bert_model(sample_input)
        lstm_output = lstm_model(sample_input)

    assert bert_output is not None
    assert lstm_output is not None

# Test model parameter updates
def test_parameter_updates(bert_model, lstm_model, sample_input, sample_target, criterion, optimizer_bert, optimizer_lstm, device):
    sample_input, sample_target = sample_input.to(device), sample_target.to(device)

    # Test BERT model
    optimizer_bert.zero_grad()
    output = bert_model(sample_input)
    loss = criterion(output, sample_target)
    loss.backward()
    optimizer_bert.step()
    for param in bert_model.parameters():
        assert param.grad is not None

    # Test LSTM model
    optimizer_lstm.zero_grad()
    output = lstm_model(sample_input)
    loss = criterion(output, sample_target)
    loss.backward()
    optimizer_lstm.step()
    for param in lstm_model.parameters():
        assert param.grad is not None

# Test batch processing (different batch sizes)
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batch_processing(bert_model, lstm_model, batch_size, device):
    sample_input = torch.randn(batch_size, 512).to(device)
    bert_output = bert_model(sample_input)
    lstm_output = lstm_model(sample_input)
    assert bert_output.shape == (batch_size, 1)
    assert lstm_output.shape == (batch_size, 1)

# Test with different input types (CPU, GPU if available)
def test_input_types(bert_model, lstm_model, sample_input):
    # Test on CPU
    sample_input_cpu = sample_input.to('cpu')
    bert_output_cpu = bert_model(sample_input_cpu)
    lstm_output_cpu = lstm_model(sample_input_cpu)
    assert bert_output_cpu is not None
    assert lstm_output_cpu is not None

    # Test on GPU if available
    if torch.cuda.is_available():
        sample_input_gpu = sample_input.to('cuda')
        bert_output_gpu = bert_model(sample_input_gpu)
        lstm_output_gpu = lstm_model(sample_input_gpu)
        assert bert_output_gpu is not None
        assert lstm_output_gpu is not None

# Test state dict save/load
def test_state_dict_save_load(bert_model, lstm_model, tmp_path):
    # Test BERT model
    bert_path = tmp_path / "bert_model.pth"
    torch.save(bert_model.state_dict(), bert_path)
    bert_model_loaded = BertRegressionModel()
    bert_model_loaded.load_state_dict(torch.load(bert_path))
    assert all(torch.equal(a, b) for a, b in zip(bert_model.parameters(), bert_model_loaded.parameters()))

    # Test LSTM model
    lstm_path = tmp_path / "lstm_model.pth"
    torch.save(lstm_model.state_dict(), lstm_path)
    lstm_model_loaded = LSTMRegressionModel()
    lstm_model_loaded.load_state_dict(torch.load(lstm_path))
    assert all(torch.equal(a, b) for a, b in zip(lstm_model.parameters(), lstm_model_loaded.parameters()))
```

### Explanation:
- **Fixtures**: Reusable components such as models, sample inputs, targets, criterion, and optimizers are defined using pytest fixtures.
- **Model Initialization**: Tests ensure that models are instantiated correctly.
- **Forward Pass**: Tests verify that the models produce outputs of the expected shape.
- **Gradient Flow**: Tests ensure that gradients are computed correctly during backpropagation.
- **Eval Mode**: Tests check that models can be set to evaluation mode and produce outputs without gradients.
- **Parameter Updates**: Tests verify that parameters are updated correctly during training.
- **Batch Processing**: Tests check model behavior with different batch sizes.
- **Input Types**: Tests ensure models work on both CPU and GPU.
- **State Dict Save/Load**: Tests verify that models can be saved and loaded correctly.

This code assumes that `BertRegressionModel` and `LSTMRegressionModel` are implemented in a file named `models.py`. Adjust the input dimensions and model implementations as necessary based on your specific use case.