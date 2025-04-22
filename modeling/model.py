import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from configs import Config


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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SpeechTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(101, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer with relative positional encoding (Shaw et al., 2018)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x: [B, T, F]
        x = self.embed(x)                  # -> [B, T, d_model]
        x = x.transpose(0, 1)             # -> [T, B, d_model]
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.transpose(0, 1).transpose(1, 2)  # -> [B, d_model, T]
        x = self.pooling(x).squeeze(2)         # -> [B, d_model]
        return self.classifier(x)


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


def build_model(cfg: Config) -> nn.Module:
    if cfg.data.yes_no_binary:
        cfg.model.num_classes = 2
    else:
        cfg.model.num_classes = len(cfg.data.target_commands)
        if cfg.data.unknown_commands:
           cfg.model.num_classes += 1
        
    if cfg.model.architecture == 'M5':
        return M5(
        n_input=1,
        n_output=cfg.model.num_classes,
        stride=16,
        n_channel=32
    )
    elif cfg.model.architecture == 'ViT':
        return ViT3M(
            in_channels=1,
            num_classes=cfg.model.num_classes
        )
    elif cfg.model.architecture == 'transformer':
        return SpeechTransformer(
            num_classes=cfg.model.num_classes,
        )
    
   
