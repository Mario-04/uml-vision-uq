import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

IMAGE_SHAPE = (32, 32)
CIFAR_10_DIR  = 'datasets/cifar10'
CIFAR_100_DIR = 'datasets/cifar100'

# Statistics computed over each training set
# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py lines 30-40
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)
# Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py lines 30-40
cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

cifar100_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

cifar100_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

class CIFARDataset(Dataset):
    """Dataset backed by pre-saved .npy arrays; applies transforms via PIL."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x = x          # (N, H, W, C) uint8
        self.y = y.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.fromarray(self.x[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]


def _download_cifar(dataset_cls, data_dir: str):
    print(f"Downloading {dataset_cls.__name__} dataset")
    train = dataset_cls(root=data_dir, train=True,  download=True)
    test  = dataset_cls(root=data_dir, train=False, download=True)
    x_train, y_train = np.array(train.data), np.array(train.targets)
    x_test,  y_test  = np.array(test.data),  np.array(test.targets)
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(data_dir, "x_train.npy"), x_train)
    np.save(Path(data_dir, "y_train.npy"), y_train)
    np.save(Path(data_dir, "x_test.npy"),  x_test)
    np.save(Path(data_dir, "y_test.npy"),  y_test)


def download_cifar10():
    _download_cifar(CIFAR10, CIFAR_10_DIR)

def download_cifar100():
    _download_cifar(CIFAR100, CIFAR_100_DIR)

def download_all():
    download_cifar10()
    download_cifar100()

def _load_cifar(data_dir: str, train_tf, test_tf, batch_size: int = 128, seed: int = 42, pin_memory: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    p = Path(data_dir)

    if not p.exists():
        _download_cifar(CIFAR10 if "cifar10" in data_dir else CIFAR100, data_dir)

    x_train = np.load(p / "x_train.npy")
    y_train = np.load(p / "y_train.npy")
    x_test  = np.load(p / "x_test.npy")
    y_test  = np.load(p / "y_test.npy")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    val_size = len(x_train) // 10
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = CIFARDataset(x_train[train_idx], y_train[train_idx], transform=train_tf)
    val_ds   = CIFARDataset(x_train[val_idx],   y_train[val_idx],   transform=test_tf)
    test_ds  = CIFARDataset(x_test,             y_test,             transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=pin_memory, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=pin_memory, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=pin_memory, persistent_workers=True)
    return train_loader, val_loader, test_loader

def _pin_memory() -> bool:
    return torch.cuda.is_available()

def load_cifar10(batch_size: int = 128, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for CIFAR-10."""
    return _load_cifar(CIFAR_10_DIR, cifar10_transform_train, cifar10_transform_test, batch_size, seed, pin_memory=_pin_memory())

def load_cifar100(batch_size: int = 128, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for CIFAR-100 (near-OOD set)."""
    return _load_cifar(CIFAR_100_DIR, cifar100_transform_train, cifar100_transform_test, batch_size, seed, pin_memory=_pin_memory())
