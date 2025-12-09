import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import os

# Define Dataset
class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

# Define LSTM Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Define BERT Model
class BERTRegressor(nn.Module):
    def __init__(self, bert_model_name):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.pooler_output)

# Training function
def train_epoch(model, data_loader, criterion, optimizer, device, scheduler=None, accumulation_steps=1):
    model.train()
    losses = []
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()

        if (len(losses) + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        losses.append(loss.item())
    return np.mean(losses)

# Validation function
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), targets)
            losses.append(loss.item())

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return np.mean(losses), mse, r2

# Main function
def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    # Placeholder for data loading
    # texts, targets = load_your_data_function()

    # Tokenizer and Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(texts, targets, tokenizer, max_length=128)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model selection
    if args.model == 'bert':
        model = BERTRegressor('bert-base-uncased').to(device)
    elif args.model == 'lstm':
        model = LSTMRegressor(input_dim=768, hidden_dim=256, output_dim=1, num_layers=2).to(device)
    else:
        raise ValueError("Model not supported. Choose 'bert' or 'lstm'.")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # TensorBoard
    writer = SummaryWriter()

    # Training loop
    best_mse = float('inf')
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, args.accumulation_steps)
        val_loss, val_mse, val_r2 = eval_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('R2/val', val_r2, epoch)

        # Checkpointing
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= args.patience:
            print("Early stopping triggered")
            break

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'lstm'], help="Model type to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--accumulation_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")

    args = parser.parse_args()
    main(args)