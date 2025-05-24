import wandb
import accelerate

from engine.base_engine import BaseEngine
from utils.config_merge import merge_configs
from configs import Config

class SweepEngine:
    def __init__(self, engine_class: type, accelerator: accelerate.Accelerator, cfg: Config):
        if not issubclass(engine_class, BaseEngine):
            raise TypeError("engine_class must be a subclass of Engine")
        self.engine = engine_class(accelerator, cfg)
        merge_configs(self.engine.cfg, wandb.config["sweep"])

    def __getattr__(self, name):
        return getattr(self.engine, name)
