import torch
import torch.nn as nn
from transformers import BertModel

class BertRegressionModel(nn.Module):
    """
    A PyTorch model for text-based regression tasks using BERT as the backbone.

    This model utilizes a pre-trained BERT model to extract features from text data.
    These features are then passed through a fully connected layer to produce a single
    output value suitable for regression tasks.

    Attributes:
        bert (BertModel): The pre-trained BERT model for feature extraction.
        dropout (nn.Dropout): Dropout layer for regularization.
        fc (nn.Linear): Fully connected layer for producing the regression output.
    """

    def __init__(self, bert_model_name: str = 'bert-base-uncased', dropout_prob: float = 0.3):
        """
        Initializes the BertRegressionModel.

        Args:
            bert_model_name (str): The name of the pre-trained BERT model to use.
            dropout_prob (float): The probability of an element to be zeroed in the dropout layer.
        """
        super(BertRegressionModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer for regression output
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
        # Initialize weights of the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks to avoid attention on padding tokens.

        Returns:
            torch.Tensor: The regression output as a single value.
        """
        # Get the last hidden state from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Pass through the fully connected layer to get the regression output
        regression_output = self.fc(pooled_output)
        
        return regression_output.squeeze(-1)

# Example usage:
# model = BertRegressionModel()
# input_ids = torch.tensor([[101, 2054, 2003, 1996, 3185, 102]])
# attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
# output = model(input_ids, attention_mask)
# print(output)