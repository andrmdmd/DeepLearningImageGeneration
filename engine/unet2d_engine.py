import os

import torch
import accelerate

from configs import Config
from dataset import get_loader
from modeling import build_unet2d_model

from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from engine.base_engine import ImageGenerationEngine

class UNet2DEngine(ImageGenerationEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        self.min_loss = float("inf")
        self.current_epoch = 1

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.training.unet2d.timesteps
        )

        self.accelerator.init_trackers(
            (
                self.cfg.project_name
                if self.cfg.project_name is not None
                else self.accelerator.project_configuration.project_dir
            ),
            config=self.cfg.to_dict(),
            init_kwargs={"wandb": self.cfg.to_dict()["wandb"]},
        )
        self.accelerator.log(
            {
                "base_dir": self.base_dir,
            }
        )

    def _train_one_epoch(self):
        epoch_progress = self.sub_task_progress.add_task(
            "loader", total=len(self.train_loader)
        )
        self.model.train()

        for loader_idx, batch in enumerate(self.train_loader, 1):
            images = batch[0]
            noise = torch.randn_like(images).to(self.accelerator.device)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.num_train_timesteps,
                (images.shape[0],),
                device=self.accelerator.device,
            ).long()

            noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

            with self.accelerator.accumulate(self.model):
                noise_pred = self.model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Step the scheduler
            self.scheduler.step()

            if self.accelerator.is_main_process:
                self.log_results(
                    {"loss/train": loss.item()},
                    step=(self.current_epoch - 1) * len(self.train_loader) + loader_idx,
                )
                self.min_loss = min(self.min_loss, loss.item())
            self.sub_task_progress.update(epoch_progress, advance=1)

        self.sub_task_progress.remove_task(epoch_progress)

        # Sample demo images after each epoch
        if self.accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=self.accelerator.unwrap_model(self.model),
                scheduler=self.noise_scheduler,
            )
            self.sample_demo_images(self.current_epoch, pipeline)

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)

        with self.accelerator.main_process_first():
            train_loader, val_loader, test_loader = get_loader(self.cfg)
        model = build_unet2d_model(self.cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        num_training_steps = len(train_loader) * self.cfg.training.epochs
        num_warmup_steps = int(
            self.cfg.training.unet2d.warmup_ratio * num_training_steps
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, scheduler
        )

    def train(self):
        train_progress = self.epoch_progress.add_task(
            "Epoch",
            total=self.cfg.training.epochs,
            completed=self.current_epoch - 1,
            cmmd=self.min_loss,
        )
        if self.accelerator.is_main_process:
            self.setup_training()
            self.print_training_details()
        self.accelerator.wait_for_everyone()
        for epoch in range(self.current_epoch, self.cfg.training.epochs + 1):
            self.current_epoch = epoch
            self._train_one_epoch()
            if epoch % self.cfg.training.val_freq == 0:
                self.accelerator.wait_for_everyone()
            if self.stop_training:
                break
            self.epoch_progress.update(train_progress, advance=1, cmmd=self.min_cmmd)
        self.epoch_progress.stop_task(train_progress)
        self.accelerator.wait_for_everyone()

    def reset(self):
        super().reset()
        self.min_loss = 0
