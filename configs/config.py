from __future__ import annotations
from dataclass_wizard import JSONPyWizard
import dataclasses
from typing import List, Literal, Optional
from dataclass_wizard import JSONPyWizard

@dataclasses.dataclass
class UNet2DTrainingConfig:
    warmup_ratio = 0.2
    timesteps: int = 1000

@dataclasses.dataclass
class TrainingConfig:
    unet2d: UNet2DTrainingConfig
    engine: str = "unet2d_engine"
    early_stopping_patience: int = 5
    label_smoothing: float = 0.0
    batch_size: int = 32
    val_freq: int = 1
    epochs: int = 5
    num_workers: int = 4
    accum_iter: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    lr: float = 0.0003
    weight_decay: float = 0.001
    # after how many epochs to sample some generated images
    save_image_epochs: int = 1
    # how many images to sample, dimension of a square grid (e.g. 4 means 4x4=16 images)
    sample_grid_dimension: int = 4


@dataclasses.dataclass
class EvalConfig:
    num_workers: int = 4
    batch_size: int = 32


@dataclasses.dataclass
class DataConfig:
    root: str = "data"
    in_channels: int = 3
    image_size: int = 64


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
class ModelConfig:
    base_dim: int = 64
    out_channels: int = 3
    resume_path: Optional[str] = None


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
    project_name: Optional[str] = None
    log_dir: str = "logs"
    mixed_precision: str = "no"
    seed: int = 0
    config: Optional[str] = None
