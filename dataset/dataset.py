from typing import Tuple

from torchvision.transforms import transforms
import torchvision

import librosa
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import random
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter

class CustomDataset(Dataset):
    """Please define your own `Dataset` here. We provide an example for CIFAR-10 dataset."""

    pass

class DynamicUnderSampler(Sampler): 
    def __init__(self, dataset, num_samples_per_class=None):
        """
        Custom sampler for dynamic undersampling.
        Args:
            dataset: The dataset to sample from.
            num_samples_per_class: Number of samples per class (optional).
        """
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class
        self.class_indices = self._get_class_indices()
        self.min_count = min(len(indices) for indices in self.class_indices.values())
        if self.num_samples_per_class:
            self.min_count = min(self.min_count, self.num_samples_per_class)


    def _get_class_indices(self):
        """
        Group dataset indices by class.
        Returns:
            A dictionary mapping class labels to a list of indices.
        """
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __iter__(self):
        """
        Generate a new set of indices for undersampling at the start of each epoch.
        """
        sampled_indices = []
        for indices in self.class_indices.values():
            sampled_indices.extend(random.sample(indices, self.min_count))
        random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.min_count * len(self.class_indices)

class DynamicOverSampler(Sampler):
    def __init__(self, dataset):
        """
        Custom sampler for dynamic oversampling.
        Args:
            dataset: The dataset to sample from.
        """
        self.dataset = dataset

    def _get_sample_weights(self):
        """
        Calculate sample weights based on class frequencies.
        Returns:
            A list of weights for each sample in the dataset.
        """
        label_counts = Counter([self.dataset[i][1] for i in range(len(self.dataset))])
        total_samples = sum(label_counts.values())
        class_weights = {label: total_samples / count for label, count in label_counts.items()}
        sample_weights = [class_weights[label] for _, label in self.dataset]
        return sample_weights

    def __iter__(self):
        """
        Generate a new set of indices for oversampling at the start of each epoch.
        """
        sample_weights = self._get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return iter(sampler)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.dataset)

def get_loader(
        cfg
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.root, train=True, download=True, transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.root, train=False, download=True, transform=val_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.root, train=False, download=True, transform=val_transform 
    )

    if cfg.training.sampling_strategy == 'undersampling':
        sampler = DynamicUnderSampler(train_dataset)
        shuffle = False
    elif cfg.training.sampling_strategy == 'oversampling':
        sampler = DynamicOverSampler(train_dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader