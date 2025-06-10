import os

import torch
import accelerate

from configs import Config
from dataset import get_loader
from modeling import build_unet2d_model

from diffusers import DDPMScheduler, DDPMPipeline, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from engine.base_engine import ImageGenerationEngine
from adaptive_augmentation import get_ada_aug
from PIL import Image


class UNet2DEngine(ImageGenerationEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        self.min_loss = float("inf")
        self.current_epoch = 1

        # based on config value set noise scheduler as DDPM or DDIM
        config = {
            "num_train_timesteps": cfg.training.unet2d.train_timesteps,
        }
        if cfg.training.unet2d.noise_scheduler == "ddim":
            self.noise_scheduler = DDIMScheduler.from_config(config)
        else:
            # default to DDPM
            self.noise_scheduler = DDPMScheduler.from_config(config)

        self.p = 0.0
        if self.cfg.training.aug_type == "linear":
            def linear_augmentation():
                p_step = 0.001
                if self.p < 0.5:
                    p = self.p + p_step

                return get_ada_aug(p=p)
            self.augmentation = linear_augmentation
        elif isinstance(self.cfg.training.aug_type,float):
            self.p = float(self.cfg.training.aug_type)

            def const_augmentation():
                return get_ada_aug(p=float(self.p))
            self.augmentation = const_augmentation
        else:
            raise ValueError(
                f"Unknown augmentation type: {self.cfg.training.aug_type}"
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

    def _sample_demo_images(self, epoch, pipeline: DDIMPipeline | DDPMPipeline):
        """
        Sample demo images after each epoch and save them.
        """

        def save_pipeline(epoch):
            save_path = os.path.join(self.base_dir, "checkpoint", f"epoch_{epoch}")
            pipeline.save_pretrained(save_path)

        if (
            (epoch + 1) % self.cfg.training.save_image_epochs == 0
            or epoch == self.cfg.training.epochs
        ):
            generator = torch.manual_seed(self.cfg.seed)
            # Generate images using the pipeline in batches
            images_count = self.cfg.training.metric_calculation_img_count
            all_images = []

            for i in range(0, images_count, self.cfg.training.batch_size):
                batch_size = min(
                    self.cfg.training.batch_size, images_count - i
                )
                batch_images = pipeline(
                    batch_size=batch_size,
                    generator=generator,
                    num_inference_steps=self.cfg.training.unet2d.inference_timesteps,
                ).images
                all_images.append(batch_images)

            all_images = [
                img for batch in all_images for img in batch
            ]
            self.evaluate(epoch, all_images, save_pipeline)
        if epoch == self.cfg.training.epochs:
            save_pipeline(epoch)

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

            aug = self.augmentation()
            for i in range(images.shape[0]):
                images[i] = aug(images[i])
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
            if self.cfg.training.unet2d.noise_scheduler == "ddim":
                pipeline = DDIMPipeline(
                    unet=self.accelerator.unwrap_model(self.model),
                    scheduler=self.noise_scheduler,
                )
            else:
                pipeline = DDPMPipeline(
                    unet=self.accelerator.unwrap_model(self.model),
                    scheduler=self.noise_scheduler,
                )

            self._sample_demo_images(self.current_epoch, pipeline)

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
