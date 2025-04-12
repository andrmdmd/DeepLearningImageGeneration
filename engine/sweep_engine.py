import wandb

from engine.train_engine import Engine
from utils.config_merge import merge_configs

class SweepEngine(Engine):
    def setup_training(self):
        self.accelerator.wait_for_everyone()

        merge_configs(self.cfg, wandb.config["sweep"])

        super().setup_training()