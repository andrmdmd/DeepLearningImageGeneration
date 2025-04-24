import os
import ssl
from typing import Tuple, Optional, Union, List, Dict, Literal

import librosa
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio
import numpy as np
import random
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter


class SpeechCommandsDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            cfg,
            mode: Literal['training', 'validation', 'testing'] = 'training'
    ):
        self.root_dir = root_dir
        self.cfg = cfg
        self.mode = mode
        self.unknown_commands_included = cfg.data.unknown_commands_included

        if cfg.data.yes_no_binary:
            self.target_commands = ['yes', 'no']
        else:
            self.target_commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

        self.label_mapping = {label: idx for idx, label in enumerate(self.target_commands)}

        if self.cfg.data.silence_included:
            self.target_commands.append('_silence_')
            self.label_mapping['_silence_'] = len(self.label_mapping)

        if self.unknown_commands_included:
            self.label_mapping['_unknown_'] = len(self.label_mapping)

        if self.cfg.data.unknown_binary_classification:
            self.num_classes = 2
        else:
            self.num_classes = len(self.label_mapping)
        self._init_audio_transforms()
        self._load_dataset()

    def _init_audio_transforms(self):
        if self.cfg.data.representation == 'spectrogram':
            def spectrogram_transform(waveform):
                stft = librosa.stft(waveform, n_fft=self.cfg.data.n_fft, hop_length=self.cfg.data.hop_length)
                return np.expand_dims(np.abs(stft), axis=0) 

            self.audio_transform = spectrogram_transform

        elif self.cfg.data.representation == 'melspectrogram':
            def melspectrogram_transform(waveform):
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=waveform.squeeze(),  # Ensure waveform is 1D
                    sr=self.cfg.data.sample_rate,
                    n_fft=self.cfg.data.n_fft,
                    hop_length=self.cfg.data.hop_length,
                    n_mels=self.cfg.data.n_mels,
                )
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                return np.expand_dims(mel_spectrogram_db, axis=0) 

            self.audio_transform = melspectrogram_transform

        elif self.cfg.data.representation == 'mfcc':
            def mfcc_transform(waveform):
 
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=waveform,
                    sr=self.cfg.data.sample_rate,
                    n_fft=self.cfg.data.n_fft,
                    hop_length=self.cfg.data.hop_length,
                    n_mels=self.cfg.data.n_mels
                )
                return librosa.feature.mfcc(
                    S=librosa.power_to_db(mel_spectrogram),
                    sr=self.cfg.data.sample_rate,
                    n_mfcc=self.cfg.data.n_mfcc
                )

            self.audio_transform = mfcc_transform

        else:  # waveform
            self.audio_transform = None

    def _load_split_files(self, split_file: str) -> List[str]:
        filepaths = []
        with open(os.path.join(self.root_dir, split_file), 'r') as f:
            for line in f:
                rel_path = line.strip()
                label = rel_path.split('/')[0]
                if label in self.target_commands or (self.unknown_commands_included and label != '_background_noise_'):
                    filepaths.append(os.path.join(self.root_dir, rel_path))
        return filepaths

    def _load_dataset(self):
        self.filepaths = []
        self.labels = []

        if self.mode == 'training':
            validation_files = set(self._load_split_files('validation_list.txt'))
            testing_files = set(self._load_split_files('testing_list.txt'))

            if self.unknown_commands_included and not self.cfg.data.yes_no_binary:
                all_commands = [
                    d for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d)) and d != '_background_noise_'
                ]

                for label in all_commands:
                    label_dir = os.path.join(self.root_dir, label)
                    for fname in os.listdir(label_dir):
                        full_path = os.path.join(label_dir, fname)
                        if full_path not in validation_files | testing_files:
                            self.filepaths.append(full_path)
                            self.labels.append(label)
            else:
                for label in self.target_commands:
                    label_dir = os.path.join(self.root_dir, label)
                    if os.path.exists(label_dir):
                        for fname in os.listdir(label_dir):
                            full_path = os.path.join(label_dir, fname)
                            if full_path not in validation_files | testing_files:
                                self.filepaths.append(full_path)
                                self.labels.append(label)
        elif self.mode == 'validation':
            self.filepaths = self._load_split_files('validation_list.txt')
            self.labels = [os.path.relpath(p, self.root_dir).split('/')[0] for p in self.filepaths]
        elif self.mode == 'testing':
            self.filepaths = self._load_split_files('testing_list.txt')
            self.labels = [os.path.relpath(p, self.root_dir).split('/')[0] for p in self.filepaths]

        class_counts = Counter(self.labels)
        print(f"Class balance in {self.mode} data:")
        for label, count in class_counts.items():
            if label in self.target_commands:
                print(f"  {label}: {count}")
        unknown_count = sum(count for label, count in class_counts.items() if label not in self.target_commands)
        if unknown_count > 0:
            print(f"  _unknown_: {unknown_count}")
            print(f"  unknown percentage: {unknown_count / len(self.filepaths) * 100:.2f}%")
        
        # set weights of labels on the basis of class_counts and unknows_count (unknown is last label in label_mapping)
        if self.unknown_commands_included and self.cfg.training.sampling_strategy == 'weights':
            self.class_weights = [1.0 / class_counts[label] for label in list(self.label_mapping.keys())[:-1]]
            self.class_weights.append(1.0 / unknown_count)
            total_weight = sum(self.class_weights)
            self.class_weights = [w / total_weight for w in self.class_weights]

    def _load_audio(self, filepath):
        waveform, sample_rate = librosa.load(filepath, sr=None)

        # Convert stereo to mono if stereo
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0, keepdims=True)

        # Ensure waveform is 2D (1, N)
        if len(waveform.shape) == 1:
            waveform = np.expand_dims(waveform, axis=0)

        if sample_rate != self.cfg.data.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.cfg.data.sample_rate)
            waveform = torch.tensor(waveform)
            waveform = resampler(waveform).numpy()

        if waveform.shape[1] > self.cfg.data.sample_rate:
            waveform = waveform[:, -self.cfg.data.sample_rate:]
        elif waveform.shape[1] < self.cfg.data.sample_rate:
            padding = self.cfg.data.sample_rate - waveform.shape[1]
            waveform = np.pad(waveform, ((0, 0), (0, padding)), mode='constant')

        return waveform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform = self._load_audio(filepath)
        label = self.labels[idx]

        if self.audio_transform is not None:
            data = self.audio_transform(waveform)
        else:
            data = waveform

        if hasattr(self.cfg.data, 'transform') and self.cfg.data.transform:
            data = self.cfg.data.transform(data)

        if self.cfg.data.unknown_binary_classification:
            label = 1 if label not in [*self.target_commands, "_silence_"] else 0
        elif self.cfg.data.yes_no_binary:
            label = 1 if label == 'yes' else 0
        else:
            label = self.label_mapping.get(label, self.label_mapping['_unknown_'] if self.unknown_commands_included else -1)

        return data, label

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
    train_dataset = SpeechCommandsDataset(
        root_dir=cfg.data.root,
        cfg=cfg,
        mode='training'
    )

    val_dataset = SpeechCommandsDataset(
        root_dir=cfg.data.root,
        cfg=cfg,
        mode='validation'
    )

    test_dataset = SpeechCommandsDataset(
        root_dir=cfg.data.root,
        cfg=cfg,
        mode='testing'
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