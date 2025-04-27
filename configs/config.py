from __future__ import annotations

import dataclasses
from typing import List, Literal, Optional

from dataclass_wizard import JSONPyWizard


@dataclasses.dataclass
class TrainingConfig:
    engine: str = "engine"
    early_stopping_patience: int = 5
    label_smoothing: float = 0.0
    batch_size: int = 32
    val_freq: int = 1
    epochs: int = 50
    num_workers: int = 4
    accum_iter: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    lr: float = 0.0003
    weight_decay: float = 0.0001
    sampling_strategy: Literal["undersampling", "oversampling", "ensemble"] | None = None
    scheduler: Literal["cosine"] | None = "cosine"


@dataclasses.dataclass
class EvalConfig:
    num_workers: int = 4
    batch_size: int = 32


@dataclasses.dataclass
class DataConfig:
    root: str = "data"
    sample_rate: int = 16000
    representation: Literal["waveform", "spectrogram", "melspectrogram", "mfcc"] = "waveform"
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    n_mfcc: int = 80
    yes_no_binary: bool = False
    unknown_commands_included: bool = True
    silence_included: bool = True
    unknown_binary_classification: bool = False


@dataclasses.dataclass
class SweepConfig:
    name: Optional[str] = None
    config: str = ""
    project_name: str = ""

@dataclasses.dataclass
class WandbConfig:
    name: str = None
    tags: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class ConformerConfig:
    input_dim: int = 80
    num_heads: int = 4
    num_layers: int = 16
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.1

@dataclasses.dataclass
class ModelConfig:
    base_dim: int = 16
    resume_path: Optional[str] = None
    architecture: Literal["M5", "Transformer", "ViT", "Conformer"] = "M5"
    conformer: ConformerConfig = dataclasses.field(default_factory=ConformerConfig)

@dataclasses.dataclass
class Config(JSONPyWizard):
    training: TrainingConfig
    model: ModelConfig

    # Config for data option
    data: DataConfig

    # Config for evaluation option
    evaluation: EvalConfig

    # Wandb
    wandb: WandbConfig
    sweep: SweepConfig
    project_tracker: List[str] = dataclasses.field(default_factory=lambda: ["wandb"])

    project_dir: str = "project"
    log_dir: str = "logs"
    mixed_precision: str = "no"
    seed: int = 0
    config: Optional[str] = None