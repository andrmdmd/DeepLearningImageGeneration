import ssl
from typing import Tuple

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from configs import Config

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

class CustomDataset(Dataset):
    """Please define your own `Dataset` here. We provide an example for CIFAR-10 dataset."""

    pass


def get_loader(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.root, train=True, download=True, transform=train_transform
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.root, train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
