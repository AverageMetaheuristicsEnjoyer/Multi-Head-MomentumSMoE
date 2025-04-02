import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

from models import MHMomentumSMoE

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc = f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({
            "loss": train_loss / (batch_idx + 1),
            "acc": 100. * correct / total
        })
    
    return train_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST("./data", train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST("./data", train = False, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    model = MHMomentumSMoE(
        input_size = 28 * 28,
        hidden_size = args.hidden_size,
        num_classes = 10,
        num_experts = args.num_experts,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        k = args.k,
        dropout_rate = args.dropout_rate,
        mu = args.mu,
        gamma = args.gamma
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
    
    print(f"Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type = int, default = 128)
    parser.add_argument("--hidden-size", type = int, default = 256)
    parser.add_argument("--num-experts", type = int, default = 8)
    parser.add_argument("--num-heads", type = int, default = 4)
    parser.add_argument("--num-layers", type = int, default = 2)
    parser.add_argument("--k", type = int, default = 2)
    parser.add_argument("--dropout-rate", type = float, default = 0.1)
    parser.add_argument("--mu", type = float, default = 0.7)
    parser.add_argument("--gamma", type = float, default = 1.0)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--epochs", type = int, default = 10)
    
    args = parser.parse_args()
    main(args)