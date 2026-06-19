import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, SVHN
from torchvision import transforms

IMAGE_SHAPE = (32, 32)
CIFAR_10_DIR  = 'datasets/cifar10'
SVHN_DIR = 'datasets/svhn'

# Statistics computed over each training set
# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py lines 30-40
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)
# Source: https://www.ricardodecal.com/guides/use-these-normalization-values-for-torchvision-datasets/
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1200, 0.1231, 0.1052)

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

svhn_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])

svhn_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
])

class CustomDataset(Dataset):
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


def download_cifar10():
    print(f"Downloading {CIFAR10.__name__} dataset")
    train = CIFAR10(root=CIFAR_10_DIR, train=True,  download=True)
    test  = CIFAR10(root=CIFAR_10_DIR, train=False, download=True)
    x_train, y_train = np.array(train.data), np.array(train.targets)
    x_test,  y_test  = np.array(test.data),  np.array(test.targets)
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")
    Path(CIFAR_10_DIR).mkdir(parents=True, exist_ok=True)
    np.save(Path(CIFAR_10_DIR, "x_train.npy"), x_train)
    np.save(Path(CIFAR_10_DIR, "y_train.npy"), y_train)
    np.save(Path(CIFAR_10_DIR, "x_test.npy"),  x_test)
    np.save(Path(CIFAR_10_DIR, "y_test.npy"),  y_test)

def download_svhn():
    print("Downloading SVHN dataset")
    train = SVHN(root=SVHN_DIR, split='train', download=True)
    test  = SVHN(root=SVHN_DIR, split='test', download=True)

    x_train = np.transpose(train.data, (0, 2, 3, 1))
    y_train = np.array(train.labels)
    x_test = np.transpose(test.data, (0, 2, 3, 1))
    y_test = np.array(test.labels)

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")

    Path(SVHN_DIR).mkdir(parents=True, exist_ok=True)
    np.save(Path(SVHN_DIR) / "x_train.npy", x_train)
    np.save(Path(SVHN_DIR) / "y_train.npy", y_train)
    np.save(Path(SVHN_DIR) / "x_test.npy", x_test)
    np.save(Path(SVHN_DIR) / "y_test.npy", y_test)

def download_all():
    download_cifar10()
    download_svhn()

def _load_trainers(data_dir: str, train_tf, test_tf, batch_size: int = 128, seed: int = 42, pin_memory: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    p = Path(data_dir)

    if not p.exists():
        download_cifar10()

    x_train = np.load(p / "x_train.npy")
    y_train = np.load(p / "y_train.npy")
    x_test  = np.load(p / "x_test.npy")
    y_test  = np.load(p / "y_test.npy")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    val_size = len(x_train) // 10
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = CustomDataset(x_train[train_idx], y_train[train_idx], transform=train_tf)
    val_ds   = CustomDataset(x_train[val_idx],   y_train[val_idx],   transform=test_tf)
    test_ds  = CustomDataset(x_test,             y_test,             transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin_memory, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory, persistent_workers=True)
    return train_loader, val_loader, test_loader

def load_cifar10(batch_size: int = 128, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for CIFAR-10."""
    return _load_trainers(CIFAR_10_DIR, cifar10_transform_train, cifar10_transform_test, batch_size, seed, pin_memory=torch.cuda.is_available())

def load_svhn(batch_size: int = 128, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for SVHN."""
    return _load_trainers(SVHN_DIR, svhn_transform_train, svhn_transform_test, batch_size, seed, pin_memory=torch.cuda.is_available())
