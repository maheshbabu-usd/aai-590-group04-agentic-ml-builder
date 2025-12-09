import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for audio classification tasks.
    
    This model is designed to process audio data and output logits for classification.
    It consists of several convolutional layers followed by fully connected layers.
    Batch normalization and dropout are used for regularization.
    """
    def __init__(self, num_classes: int):
        """
        Initializes the AudioCNN model.
        
        Args:
            num_classes (int): The number of output classes for classification.
        """
        super(AudioCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming input size is (1, 64, 64)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, 1, height, width).
        
        Returns:
            torch.Tensor: The output logits with shape (batch_size, num_classes).
        """
        # Convolutional layers with ReLU, batch normalization, and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def _initialize_weights(self):
        """
        Initializes the weights of the model using Xavier uniform initialization for
        convolutional and fully connected layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Example usage:
# model = AudioCNN(num_classes=10)
# print(model)