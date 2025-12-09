import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification tasks.
    This network consists of two convolutional layers followed by max pooling,
    batch normalization, ReLU activation, and dropout for regularization.
    Finally, it has two fully connected layers for classification.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the SimpleCNN model.

        Args:
            num_classes (int): Number of classes for the classification task.
        """
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.25)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Assuming input images are 32x32
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the network using Xavier initialization for
        convolutional layers and He initialization for fully connected layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Example usage:
# model = SimpleCNN(num_classes=10)
# print(model)