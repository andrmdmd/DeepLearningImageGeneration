import os
import ssl
from typing import Tuple, Optional, Union, List, Dict, Literal

import librosa
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio
import numpy as np


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

        if cfg.data.yes_no_binary:
            self.target_commands = ['yes', 'no']
            self.unknown_commands_included = False
        else:
            self.target_commands = cfg.data.target_commands if hasattr(cfg.data, 'target_commands') else \
                ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
            self.unknown_commands_included = True

        self._init_audio_transforms()
        self._load_dataset()

    def _init_audio_transforms(self):
        if self.cfg.data.representation == 'spectrogram':
            def spectrogram_transform(waveform):
                n_fft = self.cfg.data.n_fft if hasattr(self.cfg.data, 'n_fft') else 400
                hop_length = self.cfg.data.hop_length if hasattr(self.cfg.data, 'hop_length') else 160
                stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
                return np.abs(stft)

            self.audio_transform = spectrogram_transform

        elif self.cfg.data.representation == 'melspectrogram':
            def melspectrogram_transform(waveform):
                sample_rate = self.cfg.data.sample_rate
                n_fft = self.cfg.data.n_fft if hasattr(self.cfg.data, 'n_fft') else 400
                hop_length = self.cfg.data.hop_length if hasattr(self.cfg.data, 'hop_length') else 160
                n_mels = self.cfg.data.n_mels if hasattr(self.cfg.data, 'n_mels') else 40
                return librosa.feature.melspectrogram(
                    y=waveform,
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )

            self.audio_transform = melspectrogram_transform

        elif self.cfg.data.representation == 'mfcc':
            def mfcc_transform(waveform):
                sample_rate = self.cfg.data.sample_rate
                n_fft = self.cfg.data.n_fft if hasattr(self.cfg.data, 'n_fft') else 400
                hop_length = self.cfg.data.hop_length if hasattr(self.cfg.data, 'hop_length') else 160
                n_mels = self.cfg.data.n_mels if hasattr(self.cfg.data, 'n_mels') else 40
                n_mfcc = self.cfg.data.n_mfcc if hasattr(self.cfg.data, 'n_mfcc') else 40
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=waveform,
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )
                return librosa.feature.mfcc(
                    S=librosa.power_to_db(mel_spectrogram),
                    sr=sample_rate,
                    n_mfcc=n_mfcc
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
                if label in self.target_commands or (self.unknown_commands_included and label == '_background_noise_'):
                    filepaths.append(os.path.join(self.root_dir, rel_path))
        return filepaths

    def _load_dataset(self):
        self.filepaths = []
        self.labels = []

        if self.mode == 'training':
            validation_files = set(self._load_split_files('validation_list.txt'))
            testing_files = set(self._load_split_files('testing_list.txt'))

            for label in self.target_commands:
                label_dir = os.path.join(self.root_dir, label)
                if os.path.exists(label_dir):
                    for fname in os.listdir(label_dir):
                        full_path = os.path.join(label_dir, fname)
                        if full_path not in validation_files and full_path not in testing_files:
                            self.filepaths.append(full_path)
                            self.labels.append(label)
        elif self.mode == 'validation':
            self.filepaths = self._load_split_files('validation_list.txt')
            self.labels = [os.path.relpath(p, self.root_dir).split('/')[0] for p in self.filepaths]
        elif self.mode == 'testing':
            self.filepaths = self._load_split_files('testing_list.txt')
            self.labels = [os.path.relpath(p, self.root_dir).split('/')[0] for p in self.filepaths]

        self.unknown_files = []
        if self.unknown_commands_included and not self.cfg.data.yes_no_binary:
            all_commands = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            unknown_commands = [cmd for cmd in all_commands if
                                cmd not in self.target_commands and cmd != '_background_noise_']

            for label in unknown_commands:
                label_dir = os.path.join(self.root_dir, label)
                if os.path.exists(label_dir):
                    for fname in os.listdir(label_dir):
                        full_path = os.path.join(label_dir, fname)
                        if self.mode == 'training':
                            validation_files = set(self._load_split_files('validation_list.txt'))
                            testing_files = set(self._load_split_files('testing_list.txt'))
                            if full_path not in validation_files and full_path not in testing_files:
                                self.unknown_files.append(full_path)
                        elif self.mode == 'validation':
                            if full_path in set(self._load_split_files('validation_list.txt')):
                                self.unknown_files.append(full_path)
                        elif self.mode == 'testing':
                            if full_path in set(self._load_split_files('testing_list.txt')):
                                self.unknown_files.append(full_path)

        self.noise_files = []
        if self.cfg.data.background_noise and self.unknown_commands_included:
            noise_dir = os.path.join(self.root_dir, '_background_noise_')
            if os.path.exists(noise_dir):
                for fname in os.listdir(noise_dir):
                    if fname.endswith('.wav'):
                        full_path = os.path.join(noise_dir, fname)
                        if self.mode == 'training':
                            self.noise_files.append(full_path)

    def _load_audio(self, filepath):
        waveform, sample_rate = librosa.load(filepath, sr=None)

        if sample_rate != self.cfg.data.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.cfg.data.sample_rate)
            waveform = resampler(waveform)

        # Ensure audio is 1 second long (sample_rate samples)
        # if waveform.shape[1] > self.cfg.data.sample_rate:
        #     waveform = waveform[:, :self.cfg.data.sample_rate]
        # elif waveform.shape[1] < self.cfg.data.sample_rate:
        #     padding = self.cfg.data.sample_rate - waveform.shape[1]
        #     waveform = torch.nn.functional.pad(waveform, (0, padding))

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

        if self.cfg.data.yes_no_binary:
            label = 1 if label == 'yes' else 0

        return data, label


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

    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
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