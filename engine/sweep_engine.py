import wandb
import accelerate

from engine.train_engine import Engine
from utils.config_merge import merge_configs
from configs import Config

class SweepEngine(Engine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)
        merge_configs(self.cfg, wandb.config["sweep"])
