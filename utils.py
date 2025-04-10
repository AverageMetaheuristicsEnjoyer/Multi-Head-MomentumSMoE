import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_data(ds_name, batch_size):
    if ds_name == "mnist":
        num_channel, img_size = 1, 28
        num_classes = 10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST("./data", train = True, download = True, transform = transform)
        test_dataset = datasets.MNIST("./data", train = False, transform = transform)
    elif ds_name == "cifar10":
        num_channel, img_size = 3, 32
        num_classes = 10
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Test transforms without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")
    
    input_size = num_channel * img_size * img_size
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader, input_size, num_classes


class EarlyStopping:
    def __init__(self, patience = 5, delta = 0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement = 0
        self.stop_training = False
    
    def check(self, new_loss):
        if self.best_loss is None or new_loss < self.best_loss - self.delta:
            self.best_loss = new_loss
            self.no_improvement = 0
        else:
            self.no_improvement +=1
            if self.no_improvement > self.patience:
                self.stop_training = True

def make_wandb_table(
    train_losses,
    train_accs,
    test_losses,
    test_accs,
    run_name
):  
    columns = ["Model", "Best Test Accuracy", "Final Train Accuracy", "Final Train Loss", "Final Test Loss"]
    data = [
        [
            run_name,
            max(test_accs),
            train_accs[-1],
            train_losses[-1], 
            test_losses[-1]
        ]
    ]
    results_table = wandb.Table(columns = columns, data = data)
    wandb.log({"results_summary": results_table})