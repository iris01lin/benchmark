import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
import habana_frameworks.torch.hpu as htp
from habana_frameworks.torch.utils.library_loader import load_habana_module
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
import torch.distributed as dist
from datetime import timedelta
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MNIST or Food101 model using ResNet101.")
    parser.add_argument('--device', type=str, choices=['cpu', 'hpu'], default='hpu', help="Device to use for training ('cpu' or 'hpu').")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs for training.")
    parser.add_argument('--image_size', type=int, default=224, help="Input image size for ResNet101.")
    parser.add_argument('--model_save_path', type=str, default='./restnet.pth', help="Path to save the trained model.")
    parser.add_argument('--num_hpus', type=int, default=1, help="Number of HPUs to use.")
    parser.add_argument('--timeout', type=int, default=90, help='Timeout for distributed connection in seconds')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'food101'], default='mnist', help="Dataset to use for training ('mnist' or 'food101').")
    return parser.parse_args()

def load_data(batch_size, image_size, dataset):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.Food101(root='./data', split='test', download=True, transform=transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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

    train_loader, test_loader = load_data(args.batch_size, args.image_size, args.dataset)

    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 101)
    model = model.to(device)

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
