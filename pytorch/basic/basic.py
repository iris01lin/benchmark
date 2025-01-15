import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import habana_frameworks.torch.hpu as htp
from habana_frameworks.torch.utils.library_loader import load_habana_module
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
import torch.distributed as dist
from datetime import timedelta
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN model on MNIST.")
    parser.add_argument('--device', type=str, choices=['cpu', 'hpu'], default='hpu', help="Device to use for training ('cpu' or 'hpu').")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--model_save_path', type=str, default='./mnist_model.pth', help="Path to save the trained model.")
    parser.add_argument('--num_hpus', type=int, default=1, help="Number of HPUs to use.")
    parser.add_argument('--timeout', type=int, default=90, help='Timeout for distributed connection in seconds')
    return parser.parse_args()

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened size after pooling
        self.fc2 = nn.Linear(128, 10)  # Output 10 classes (for MNIST)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def setup_device(device_choice):
    if device_choice == "hpu":
        load_habana_module()
        device = torch.device("hpu")
    else:
        device = torch.device("cpu")
    return device

def init_distributed_training(timeout):
    if not dist.is_initialized():
        world_size, rank, local_rank = initialize_distributed_hpu()
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        dist.init_process_group(backend="hccl", world_size=world_size, rank=rank, timeout=timedelta(seconds=timeout))
        print("Distributed training initialized with Habana.")
        htp.set_device(dist.get_rank())
        return rank, world_size
        
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    start_time = time.time()
    args = parse_args()
    
    device = setup_device(args.device)
    print(f"Using device: {device}")

    if args.device == "hpu" and args.num_hpus > 1:
        init_distributed_training(args.timeout)

    train_loader, test_loader = load_data(args.batch_size)

    model = SimpleCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    test_model(model, test_loader, criterion, device)

    save_model(model, args.model_save_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
