import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

from utils import get_data, make_wandb_table, EarlyStopping
from models import MHMomentumSMoE

def iter_train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc = f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, lb_loss = model(data)
        
        loss = criterion(output, target)
        loss += lb_loss
        
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

def iter_validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def train(args):
    if args.wandb_key:
        wandb.login(key = args.wandb_key)
        wandb.init(
            project = args.project_name,
            name = args.run_name
        )
        
        config = {
            "dataset": args.data,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "inner_hidden_size": args.inner_hidden_size,
            "num_experts": args.num_experts,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "moe_top_k": args.moe_top_k,
            "dropout": args.dropout,
            "mu": args.mu,
            "gamma1": args.gamma1,
            "gamma2": args.gamma2,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "alpha": args.alpha
        }
        wandb.config.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, input_size, num_classes = get_data(ds_name = args.data, batch_size = args.batch_size)
    
    model = MHMomentumSMoE(
        input_size = input_size,
        hidden_size = args.hidden_size,
        inner_hidden_size = args.inner_hidden_size,
        num_classes = num_classes,
        num_experts = args.num_experts,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        moe_top_k = args.moe_top_k,
        dropout = args.dropout,
        mu = args.mu,
        gamma1 = args.gamma1,
        gamma2 = args.gamma2,
        beta1 = args.beta1,
        beta2 = args.beta2,
        mom_type = args.mom_type,
        alpha = args.alpha
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience = 10)
    
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = iter_train(model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = iter_validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if args.wandb_key:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })

        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(args.output_dir, "models", f"best_{args.run_name}_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")            
            
            if args.wandb_key:
                model_artifact = wandb.Artifact(
                    name = args.run_name,
                    type = "model"
                )

                model_artifact.add_file(model_path)

                wandb.log_artifact(
                    model_artifact,
                    aliases=[f"epoch - {epoch}", f"test_accuracy - {best_acc}"]
                )

                wandb.run.summary[f"{args.run_name}_best_acc"] = best_acc

        early_stopping.check(test_loss)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}\n")
            break

    make_wandb_table(
        train_losses,
        train_accs,
        test_losses,
        test_accs,
        args.run_name
    )
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default = "mnist", help = "Dataset used in training: [mnist, cifar10]")
    parser.add_argument("--batch-size", type = int, default = 128)
    parser.add_argument("--hidden-size", type = int, default = 256)
    parser.add_argument("--inner-hidden-size", type = int, default = 256)
    parser.add_argument("--num-experts", type = int, default = 8)
    parser.add_argument("--num-heads", type = int, default = 4)
    parser.add_argument("--num-layers", type = int, default = 2)
    parser.add_argument("--moe_top_k", type = int, default = 2)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--mu", type = float, default = 0.7)
    parser.add_argument("--gamma1", type=float, default = 1.0)
    parser.add_argument("--gamma2", type=float, default = 1.0)
    parser.add_argument("--beta1", type=float, default = 0.9)
    parser.add_argument("--beta2", type=float, default = 0.999)
    parser.add_argument("--mom-type", type=str, default = "heavy-ball")
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--output-dir", type = str, default = "output/", help = "Path to the output directory")
    parser.add_argument("--alpha", type = float, default = 0.01, help = "Coefficient for the load balancing loss from Multi-head MoE paper")
    
    # wandb arguments
    parser.add_argument("--wandb-key", type = str, default = None)
    parser.add_argument("--project-name", type = str, default = "project_name")
    parser.add_argument("--run-name", type = str, default = "run_name")
    
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok = True)
    train(args)