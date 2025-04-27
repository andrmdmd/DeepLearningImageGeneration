import torch
import torch.nn as nn

from configs import Config


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weights: torch.Tensor = None):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets)


def build_loss(cfg: Config, weights = None) -> ClassificationLoss:
    if cfg.training.sampling_strategy == "weights":
        return ClassificationLoss(label_smoothing=cfg.training.label_smoothing, weights=weights)
    return ClassificationLoss(label_smoothing=cfg.training.label_smoothing)
