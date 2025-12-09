import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import argparse
from tqdm import tqdm
import os

# Define models
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SmallResNetVariant(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNetVariant, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DeeperCNNWithBatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(DeeperCNNWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset (example with CIFAR-10)
class CustomImageDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, writer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1, keepdim=True)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    writer.add_scalar('Precision/train', precision, epoch)
    writer.add_scalar('Recall/train', recall, epoch)
    writer.add_scalar('F1/train', f1, epoch)

    return avg_loss, accuracy, precision, recall, f1

# Validation function
def validate(model, device, val_loader, criterion, epoch, writer):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            preds = output.argmax(dim=1, keepdim=True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('Precision/val', precision, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    writer.add_scalar('F1/val', f1, epoch)

    return avg_loss, accuracy, precision, recall, f1

# Main function
def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'small_resnet_variant', 'deeper_cnn_with_batchnorm'], help='model to use for training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CustomImageDataset(root='./data', train=True, transform=transform)
    val_dataset = CustomImageDataset(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == 'simple_cnn':
        model = SimpleCNN().to(device)
    elif args.model == 'small_resnet_variant':
        model = SmallResNetVariant().to(device)
    elif args.model == 'deeper_cnn_with_batchnorm':
        model = DeeperCNNWithBatchNorm().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    best_val_f1 = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 5

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(model, device, train_loader, optimizer, criterion, epoch, writer)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, device, val_loader, criterion, epoch, writer)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            if args.save_model:
                torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    writer.close()

if __name__ == '__main__':
    main()