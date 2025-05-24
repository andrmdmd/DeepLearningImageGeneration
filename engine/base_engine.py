"""
BaseTrainer is used to show the training details without making our final trainer too complicated.
Users can extend this class to add more functionalities.
"""

import os
import csv
import json
import dataclasses

import accelerate
import torch
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from configs import Config, show_config
from utils.meter import AverageMeter
from utils.clipmmd import CLIPMMD
from PIL import Image
import wandb


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.3f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class BaseEngine:
    def __init__(
        self, accelerator: accelerate.Accelerator, cfg: Config, is_training_engine: bool = True
    ):
        # Setup accelerator for distributed training (or single GPU) automatically
        run_number = 1
        while os.path.exists(os.path.join(cfg.log_dir, cfg.project_dir, f"run_{run_number}")):
            run_number += 1
        self.base_dir = os.path.join(cfg.log_dir, cfg.project_dir, f"run_{run_number}")
        self.accelerator = accelerator
        os.makedirs(self.base_dir)

        self.accelerator.wait_for_everyone()

        self.cfg = cfg
        self.device = self.accelerator.device
        self.dtype = self.get_dtype()

        self.sub_task_progress = Progress(
            TextColumn("{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=True,
            disable=not self.accelerator.is_main_process,
        )
        self.epoch_progress = Progress(
            *self.sub_task_progress.columns,
            TextColumn("| [bold blue]min cmmd: {task.fields[cmmd]:.3f}"),
            transient=True,
            disable=not self.accelerator.is_main_process,
        )
        self.live_process = Live(Group(self.epoch_progress, self.sub_task_progress))
        self.live_process.start(refresh=self.live_process._renderable is not None)

        # Monitor for the time
        self.iter_time = AverageMeter()
        self.data_time = AverageMeter()

    def get_dtype(self):
        if self.cfg.mixed_precision == "no":
            return torch.float32
        elif self.cfg.mixed_precision == "fp16":
            return torch.float16
        elif self.cfg.mixed_precision == "bf16":
            return torch.bfloat16

    def print_dataset_details(self):
        self.accelerator.print(
            "📁 \033[1mLength of dataset\033[0m:\n"
            f" - 💪 Train: {len(self.train_loader.dataset)}\n"
            f" - 📝 Validation: {len(self.val_loader.dataset)}"
        )

    def print_model_details(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params
        self.accelerator.print(
            "🤖 \033[1mModel Parameters:\033[0m\n"
            f" - 🔥 Trainable: {trainable_params}\n"
            f" - 🧊 Non-trainable: {non_trainable_params}\n"
            f" - 🤯 Total: {total_params}"
        )
        self.accelerator.log(
            {
                "trainable_params": trainable_params,
            }
        )

    def print_training_details(self):
        try:
            show_config(self.cfg)
            config_path = os.path.join(self.base_dir, "config.json")
            with open(config_path, "w") as config_file:
                config_dict = dataclasses.asdict(self.cfg)
                json.dump(config_dict, config_file, indent=4)
            self.print_dataset_details()
        except Exception:
            pass
        try:
            self.print_model_details()
        except Exception:
            pass

    def reset(self):
        self.data_time.reset()
        self.iter_time.reset()

    def close(self):
        self.live_process.stop()
        self.accelerator.end_training()

    def log_results(self, metrics: dict, step: int | None = None, csv_name: str = "metrics.csv"):
        """
        Logs metrics and saves them to a CSV file.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Current validation step.
            csv_path (str): Path to the CSV file for saving metrics.
        """
        # Log metrics using the accelerator
        if step is not None:
            self.accelerator.log(metrics, step=step)
        else:
            self.accelerator.log(metrics)

        # Save metrics to CSV
        csv_path = os.path.join(self.base_dir, csv_name)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["step"] + list(metrics.keys()) if step else list(metrics.keys()),
            )
            if not file_exists:
                writer.writeheader()  # Write header if file doesn't exist
            writer.writerow({"step": step, **metrics} if step else metrics)


class ImageGenerationEngine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)
        self.clip_mmd = CLIPMMD(self.accelerator.device, os.path.join(self.cfg.data.root, "cats"), cfg)
        self.min_cmmd = float("inf")
        self.early_stopping_patience = self.cfg.training.early_stopping_patience
        self.early_stopping_counter = 0
        self.stop_training = False

    def make_grid(self, images, rows, cols):
        """
        Create a grid of images.
        """
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(self, epoch, pipeline):
        """
        Generate and save demo images, upload them to wandb, and evaluate with CLIP-MMD.
        """
        generator = torch.manual_seed(self.cfg.seed)
        images = pipeline(batch_size=self.cfg.training.sample_grid_dimension**2, generator=generator).images

        # Create a grid of images
        image_grid = self.make_grid(images, rows=self.cfg.training.sample_grid_dimension, cols=self.cfg.training.sample_grid_dimension)

        # Save the grid locally
        test_dir = os.path.join(self.base_dir, "checkpoint", "samples")
        os.makedirs(test_dir, exist_ok=True)
        grid_path = os.path.join(test_dir, f"{epoch:04d}.png")
        image_grid.save(grid_path)

        # Upload the grid to wandb
        if self.accelerator.is_main_process:
            wandb.log(
                {"Generated Images": wandb.Image(image_grid)},
                step=(epoch) * len(self.train_loader),
            )

        # Evaluate with CLIP-MMD
        if self.accelerator.is_main_process:
            clip_mmd_score = self.clip_mmd.compute_mmd(images)
            self.log_results(
                {"val/cmmd": clip_mmd_score},
                step=(epoch) * len(self.train_loader),
                csv_name="cmmd.csv",
            )
            if clip_mmd_score < self.min_cmmd:
                self.min_cmmd = clip_mmd_score
                self.accelerator.print(f"New best CLIP-MMD score: {clip_mmd_score:.8f}")
                save_path = os.path.join(self.base_dir, "checkpoint", f"epoch_{epoch}")
                pipeline.save_pretrained(save_path)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.stop_training = True
                    self.accelerator.print(
                        f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement."
                    )

    def sample_demo_images(self, epoch, pipeline):
        """
        Sample demo images after each epoch and save them.
        """
        if (
            (epoch + 1) % self.cfg.training.save_image_epochs == 0
            or epoch == self.cfg.training.epochs
        ):
            self.evaluate(epoch, pipeline)
        if epoch == self.cfg.training.epochs:
            save_path = os.path.join(self.base_dir, "checkpoint", f"epoch_{epoch}")
            pipeline.save_pretrained(save_path)