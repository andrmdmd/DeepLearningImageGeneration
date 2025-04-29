import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from configs import Config
from torchaudio.models.conformer import Conformer


class ClassicModel(nn.Module):
    def __init__(self, in_channels: int, base_dim: int, num_classes: int):
        super(ClassicModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_dim, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 5)
        self.fc1 = nn.Linear(base_dim * 2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(1)


# parameters based on best performing setup from Appendix A of https://arxiv.org/abs/2210.07240
def ViT3M(in_channels: int, num_classes: int) -> nn.Module:
    return timm.models.VisionTransformer(
        num_classes=num_classes,
        in_chans=in_channels,
        img_size=80,
        patch_size=4,
        embed_dim=192,
        depth=9,
        num_heads=12,
        mlp_ratio=2,
    )


class ConformerClassifier(nn.Module):
    def __init__(self,
                 input_dim=80,
                 n_heads=4,
                 num_layers=4,
                 num_classes=11,
                 dropout=0.1,
                 ff_expansion=4,
                 conv_kernel_size=31,
                 input_transform=None):
        super().__init__()
        self.encoder = Conformer(
            input_dim=input_dim,
            num_heads=n_heads,
            ffn_dim=ff_expansion * input_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.input_transform = input_transform

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        x, _ = self.encoder(x, lengths)  # shape: (batch, time, encoder_dim)
        x = x.transpose(1, 2)  # (batch, encoder_dim, time)
        x = self.pooling(x).squeeze(-1)  # (batch, encoder_dim)
        return self.classifier(x)


class EnsembleStrategy(nn.Module):
    def __init__(self, unknown_model, desired_model):
        super().__init__()
        self.unknown_model = unknown_model
        self.desired_model = desired_model

    def forward(self, x):
        unknown_output = self.unknown_model(x)
        desired_output = self.desired_model(x)
        return torch.cat(
            (desired_output * (1. - unknown_output[0]), unknown_output), dim=1
        )


def _build_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.model.architecture == "Conformer":
        # x shape: (batch, time, features)
        if cfg.data.representation == "waveform":
            input_transform = lambda x: x.view(
                x.shape[0],
                x.shape[2] // cfg.model.conformer.input_dim,
                cfg.model.conformer.input_dim,
            )
        elif cfg.data.representation == "mfcc":
            input_transform = lambda x: x.view(
                x.shape[0],
                x.shape[3],
                x.shape[2],
            )
        elif cfg.data.representation == "melspectrogram":
            input_transform = lambda x: x.view(
                x.shape[0],
                x.shape[3],
                x.shape[2],
            )
        elif cfg.data.representation == "spectrogram":
            input_transform = lambda x: x.view(
                x.shape[0],
                x.shape[3],
                x.shape[2],
            )
        else:
            raise ValueError(f"Unknown representation: {cfg.data.representation}")

        return ConformerClassifier(
            input_dim=cfg.model.conformer.input_dim,
            input_transform=input_transform,
            num_classes=num_classes,
            n_heads=cfg.model.conformer.num_heads,
            num_layers=cfg.model.conformer.num_layers,
            dropout=cfg.model.conformer.dropout,
            ff_expansion=4,
            conv_kernel_size=cfg.model.conformer.depthwise_conv_kernel_size,
        )
    elif cfg.model.architecture == "M5":
        return M5(n_input=1, n_output=num_classes, stride=16, n_channel=32)
    elif cfg.model.architecture == "ViT":
        return ViT3M(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    if (
            cfg.data.unknown_commands_included
            and cfg.training.sampling_strategy == "ensemble"
    ):
        return EnsembleStrategy(
            unknown_model=_build_model(cfg, 1),
            desired_model=_build_model(cfg, num_classes - 1),
        )
    else:
        return _build_model(cfg, num_classes)
