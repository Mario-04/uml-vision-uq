import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from .data import load_cifar10
from .models import CNN
from .artifacts import make_run_dir, save_run


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_default_device():
    """Pick GPU or MPS if available; else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches."""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        # training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        # validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def plot_accuracies(history, save_path="reports/figures/accuracy_plot.png"):
    accuracies = [x['val_acc'] for x in history]
    plt.figure()
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(save_path)
    plt.close()

def plot_losses(history, save_path="reports/figures/loss_plot.png"):
    train_losses = [x.get('train_loss', 0) for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(save_path)
    plt.close()

def train_mc_dropout_cifar10CNN(dropout_p: float = 0.2, seed: int = 42):
    set_seed(seed)
    device = get_default_device()
    print(f"Using device: {device}")

    batch_size = 256
    train_loader, val_loader, test_loader = load_cifar10(batch_size=batch_size, seed=seed)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader   = DeviceDataLoader(val_loader,   device)
    test_loader  = DeviceDataLoader(test_loader,  device)

    model = CNN(dropout_p).to(device)

    num_epochs = 20
    opt_func = torch.optim.Adam
    lr = 0.001

    run_dir = make_run_dir("MCDropout_CNN")
    history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)

    plot_accuracies(history, save_path=run_dir / "accuracy_plot.png")
    plot_losses(history,     save_path=run_dir / "loss_plot.png")

    config = {
        "model_class":  "MCDropout_CNN",
        "model_kwargs": {
            "dropout_p": dropout_p
        },
        "num_epochs":   num_epochs,
        "lr":           lr,
        "optimizer":    opt_func.__name__,
        "scheduler":    "CosineAnnealingLR",
        "batch_size":   batch_size,
        "seed":         seed,
        "device":       str(device),
        "timestamp":    datetime.now().isoformat(),
    }
    save_run(run_dir, model, history, config)
    print(f"Run saved to: {run_dir}")
    return model


def train_baseline_cifar10CNN(seed: int = 42):
    set_seed(seed)
    device = get_default_device()
    print(f"Using device: {device}")

    batch_size = 128
    train_loader, val_loader, test_loader = load_cifar10(batch_size=batch_size, seed=seed)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader   = DeviceDataLoader(val_loader,   device)
    test_loader  = DeviceDataLoader(test_loader,  device)

    model = CNN(0.0).to(device)

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001

    run_dir = make_run_dir("baseline_CNN")
    history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)

    plot_accuracies(history, save_path=run_dir / "accuracy_plot.png")
    plot_losses(history,     save_path=run_dir / "loss_plot.png")

    config = {
        "model_class":  "baseline_CNN",
        "model_kwargs": {},
        "num_epochs":   num_epochs,
        "lr":           lr,
        "optimizer":    opt_func.__name__,
        "scheduler":    "CosineAnnealingLR",
        "batch_size":   batch_size,
        "seed":         seed,
        "device":       str(device),
        "timestamp":    datetime.now().isoformat(),
    }
    save_run(run_dir, model, history, config)
    print(f"Run saved to: {run_dir}")
    return model