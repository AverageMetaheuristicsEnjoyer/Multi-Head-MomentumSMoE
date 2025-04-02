import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from models import MHMomentumSMoE
from train import train, validate

def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST("./data", train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST("./data", train = False, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    
    models = [
        {
            "name": "momentum_only",
            "num_heads": 1,                 # One-head
            "mu": args.mu,                  # With momentum
        },
        {
            "name": "multi_head_only",
            "num_heads": args.num_heads,    # Multi-head
            "mu": 0.0,                      # No momentum
        },
        {
            "name": "multi_head_momentum",
            "num_heads": args.num_heads,    # Multi-head
            "mu": args.mu,                  # With momentum
        },
    ]
    
    results = {}
    
    for param in models:
        print(f"Running model: {param["name"]}")
        
        model = MHMomentumSMoE(
            input_size = 28 * 28,
            hidden_size = args.hidden_size,
            num_classes = 10,
            num_experts = args.num_experts,
            num_heads = param["num_heads"],
            num_layers = args.num_layers,
            k = args.k,
            dropout_rate = args.dropout_rate,
            mu = param["mu"],
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
                torch.save(model.state_dict(), os.path.join(args.output_dir, "models", f"best_{param["name"]}_model.pt"))
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        results[param["name"]] = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs,
        }
    
    plt.figure(figsize = (12, 10))
    
    plt.subplot(2, 2, 1)
    for param in results:
        plt.plot(results[param]["train_losses"], label = param)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(args.epochs))
    plt.legend()
    plt.title("Training Loss")
    
    plt.subplot(2, 2, 2)
    for param in results:
        plt.plot(results[param]["test_losses"], label = param)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(args.epochs))
    plt.legend()
    plt.title("Test Loss")
    
    plt.subplot(2, 2, 3)
    for param in results:
        plt.plot(results[param]["train_accs"], label = param)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(args.epochs))
    plt.legend()
    plt.title("Training Accuracy")
    
    plt.subplot(2, 2, 4)
    for param in results:
        plt.plot(results[param]["test_accs"], label = param)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(args.epochs))
    plt.legend()
    plt.title("Test Accuracy")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    
    return results

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
    parser.add_argument("--output-dir", type = str, default = "output/", help = "Path to the output directory")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok = True)
    results = run_experiments(args)