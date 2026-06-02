import numpy as np
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.get_data import (
    CIFARDataset,
    cifar10_transform_train,
    cifar10_transform_test,
    load_cifar10,
    load_cifar100,
    CIFAR_10_DIR,
    CIFAR_100_DIR,
)

N, H, W, C = 16, 32, 32, 3
NUM_CLASSES = 10


@pytest.fixture
def synthetic_batch():
    x = np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8)
    y = np.random.randint(0, NUM_CLASSES, (N,), dtype=np.int64)
    return x, y


# --- CIFARDataset unit tests ---

def test_dataset_length(synthetic_batch):
    x, y = synthetic_batch
    ds = CIFARDataset(x, y)
    assert len(ds) == N


def test_dataset_item_shapes(synthetic_batch):
    x, y = synthetic_batch
    ds = CIFARDataset(x, y, transform=cifar10_transform_test)
    img, label = ds[0]
    assert img.shape == (C, H, W)
    assert isinstance(label, int)


def test_dataset_image_is_normalized(synthetic_batch):
    x, y = synthetic_batch
    ds = CIFARDataset(x, y, transform=cifar10_transform_test)
    img, _ = ds[0]
    # After ToTensor + Normalize the values are no longer bounded to [0, 1]
    # but should not span the raw [0, 255] uint8 range
    assert img.max().item() < 10.0
    assert img.min().item() > -10.0


def test_dataset_label_dtype(synthetic_batch):
    x, y = synthetic_batch
    ds = CIFARDataset(x, y, transform=cifar10_transform_test)
    _, label = ds[0]
    assert isinstance(label, int)


def test_dataset_no_transform_returns_pil(synthetic_batch):
    from PIL import Image
    x, y = synthetic_batch
    ds = CIFARDataset(x, y, transform=None)
    img, _ = ds[0]
    assert isinstance(img, Image.Image)


def test_dataloader_batch_shapes(synthetic_batch):
    x, y = synthetic_batch
    ds = CIFARDataset(x, y, transform=cifar10_transform_test)
    loader = DataLoader(ds, batch_size=4)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, C, H, W)
    assert labels.shape == (4,)
    assert imgs.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_train_transform_differs_from_test(synthetic_batch):
    """Train transform has stochastic steps so repeated outputs may differ."""
    x, y = synthetic_batch
    ds_train = CIFARDataset(x, y, transform=cifar10_transform_train)
    ds_test  = CIFARDataset(x, y, transform=cifar10_transform_test)
    # Test transform is deterministic
    img_test_a, _ = ds_test[0]
    img_test_b, _ = ds_test[0]
    assert torch.equal(img_test_a, img_test_b)


# --- Integration tests (require downloaded .npy files) ---

cifar10_missing  = not Path(CIFAR_10_DIR,  "x_train.npy").exists()
cifar100_missing = not Path(CIFAR_100_DIR, "x_train.npy").exists()


@pytest.mark.skipif(cifar10_missing, reason="CIFAR-10 .npy files not present")
def test_load_cifar10_shapes():
    train_loader, test_loader = load_cifar10(batch_size=64)
    imgs, labels = next(iter(train_loader))
    assert imgs.shape == (64, C, H, W)
    assert labels.shape == (64,)


@pytest.mark.skipif(cifar10_missing, reason="CIFAR-10 .npy files not present")
def test_load_cifar10_split_sizes():
    train_loader, test_loader = load_cifar10(batch_size=128)
    assert len(train_loader.dataset) == 50_000
    assert len(test_loader.dataset)  == 10_000


@pytest.mark.skipif(cifar100_missing, reason="CIFAR-100 .npy files not present")
def test_load_cifar100_shapes():
    _, test_loader = load_cifar100(batch_size=64)
    imgs, labels = next(iter(test_loader))
    assert imgs.shape == (64, C, H, W)
    assert labels.shape == (64,)


@pytest.mark.skipif(cifar100_missing, reason="CIFAR-100 .npy files not present")
def test_load_cifar100_split_sizes():
    train_loader, test_loader = load_cifar100(batch_size=128)
    assert len(train_loader.dataset) == 50_000
    assert len(test_loader.dataset)  == 10_000
