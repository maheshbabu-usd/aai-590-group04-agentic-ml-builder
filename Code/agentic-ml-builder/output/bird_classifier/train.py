import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import os

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Transformer model (simplified for demonstration)
class TransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(128, 512)
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))
        return x

# Custom Dataset class for audio data
class AudioDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Function to calculate metrics
def calculate_metrics(outputs, targets):
    _, preds = torch.max(outputs, 1)
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    return accuracy, precision, recall

# Training function
def train(model, criterion, optimizer, train_loader, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy, precision, recall = calculate_metrics(np.array(all_preds), np.array(all_targets))
    return avg_loss, accuracy, precision, recall

# Validation function
def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy, precision, recall = calculate_metrics(np.array(all_preds), np.array(all_targets))
    return avg_loss, accuracy, precision, recall

# Main function
def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your data here
    # For demonstration, using random data
    num_samples = 1000
    num_classes = 10
    data = np.random.rand(num_samples, 1, 64, 64).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Split data into train and validation
    split = int(0.8 * num_samples)
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_data, train_labels, transform=transform)
    val_dataset = AudioDataset(val_data, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    if args.model == 'cnn':
        model = CNNModel(num_classes=num_classes)
    elif args.model == 'transformer':
        model = TransformerModel(num_classes=num_classes)
    else:
        raise ValueError("Model type not supported. Choose 'cnn' or 'transformer'.")

    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # TensorBoard
    writer = SummaryWriter()

    # Early stopping and checkpointing
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Training
        train_loss, train_acc, train_prec, train_rec = train(model, criterion, optimizer, train_loader, device, args.accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")

        # Validation
        val_loss, val_acc, val_prec, val_rec = validate(model, criterion, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Precision/train', train_prec, epoch)
        writer.add_scalar('Precision/val', val_prec, epoch)
        writer.add_scalar('Recall/train', train_rec, epoch)
        writer.add_scalar('Recall/val', val_rec, epoch)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Training Script')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'transformer'], help='Model type to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")