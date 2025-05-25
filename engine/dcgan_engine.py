import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from engine.base_engine import ImageGenerationEngine
from modeling.model import DCGANGenerator, DCGANDiscriminator
from dataset import get_loader
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image


class DCGANEngine(ImageGenerationEngine):
    def __init__(self, accelerator, cfg):
        super().__init__(accelerator, cfg)

        self.min_loss = float("inf")
        self.current_epoch = 1

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

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)

        self.netG = DCGANGenerator().to(self.device)
        self.netD = DCGANDiscriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
            betas=(self.cfg.training.dcgan.beta1, self.cfg.training.dcgan.beta2),
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
            betas=(self.cfg.training.dcgan.beta1, self.cfg.training.dcgan.beta2),
        )

        with self.accelerator.main_process_first():
            self.train_loader, _, _ = get_loader(self.cfg)

        (
            self.netG,
            self.netD,
            self.optimizerG,
            self.optimizerD,
            self.train_loader,
        ) = self.accelerator.prepare(
            self.netG, self.netD, self.optimizerG, self.optimizerD, self.train_loader
        )

    def _train_one_epoch(self, epoch):
        epoch_progress = self.sub_task_progress.add_task(
            "loader", total=len(self.train_loader)
        )
        self.netG.train()
        self.netD.train()

        for loader_idx, (real, _) in enumerate(self.train_loader):
            current_step = (self.current_epoch) * len(self.train_loader) + loader_idx
            b_size = real.size(0)
            real = real.to(self.device)
            label_real = torch.ones(b_size, device=self.device)
            label_fake = torch.zeros(b_size, device=self.device)

            self.optimizerD.zero_grad()
            output_real = self.netD(real).view(-1)
            lossD_real = self.criterion(output_real, label_real)

            noise = torch.randn(
                b_size, self.cfg.training.dcgan.nz, 1, 1, device=self.device
            )
            fake = self.netG(noise)
            output_fake = self.netD(fake.detach()).view(-1)
            lossD_fake = self.criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            self.accelerator.backward(lossD)
            self.optimizerD.step()

            self.optimizerG.zero_grad()
            output_fake = self.netD(fake).view(-1)
            lossG = self.criterion(output_fake, label_real)
            self.accelerator.backward(lossG)
            self.optimizerG.step()

            if self.accelerator.is_main_process and loader_idx % 100 == 0:
                self.accelerator.print(
                    f"Epoch [{epoch}/{self.cfg.training.epochs}] Batch [{loader_idx}/{len(self.train_loader)}] "
                    f"LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}"
                )

            combined_loss = lossD.item() + lossG.item()

            if combined_loss < self.min_loss:
                self.min_loss = combined_loss
                self.save_model(os.path.join(self.base_dir, "best_generator.pth"))

            if self.accelerator.is_main_process:
                self.log_results(
                    {
                        "lossD/train": lossD.item(),
                        "lossG/train": lossG.item(),
                        "combinedLoss/train": combined_loss,
                    },
                    step=current_step,
                )
            if self.accelerator.is_main_process and loader_idx % 100 == 0:
                self.sample_demo_images(
                    self.current_epoch, self.build_pipeline(), current_step
                )
            self.sub_task_progress.update(epoch_progress, advance=1)

        if self.accelerator.is_main_process:
            self.sample_demo_images(self.current_epoch, self.build_pipeline())

    def build_pipeline(self):
        class Pipeline:
            def __init__(self, generator, device, cfg):
                self.generator = generator
                self.device = device
                self.cfg = cfg

            def __call__(self, batch_size, generator):
                self.generator.eval()
                with torch.no_grad():
                    noise = torch.randn(
                        batch_size,
                        self.cfg.training.dcgan.nz,
                        1,
                        1,
                        device=self.device,
                        generator=generator,
                    )
                    generated_images = self.generator(noise)

                    low = float(generated_images.min())
                    high = float(generated_images.max())

                    img = generated_images.clone()
                    img = img.clamp(low, high)
                    img = img.sub(low).div(max(high - low, 1e-5))

                    arr = (
                        img.mul(255)
                        .add(0.5)
                        .clamp(0, 255)
                        .permute(0, 2, 3, 1)
                        .to(torch.uint8)
                        .to("cpu")
                    )
                    self.images = [
                        Image.fromarray(arr[i].numpy()) for i in range(arr.size(0))
                    ]
                return self

            def save_pretrained(self, _):
                # todo save the model
                pass

        return Pipeline(self.netG, self.device, self.cfg)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def save_model(self, best_model_path):
        if self.accelerator.is_main_process:
            torch.save(self.netG.state_dict(), best_model_path)

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
            # self.netG.apply(self.weights_init)
            # self.netD.apply(self.weights_init)

        self.accelerator.wait_for_everyone()
        self.netG.train()
        self.netD.train()
        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch
            self._train_one_epoch(epoch)
            if epoch % self.cfg.training.val_freq == 0:
                self.accelerator.wait_for_everyone()
            if self.stop_training:
                break
            self.epoch_progress.update(train_progress, advance=1, cmmd=self.min_cmmd)

    def print_model_details(self):
        def count_params(model):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable = sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            )
            total = trainable + non_trainable
            return trainable, non_trainable, total

        g_trainable, g_non_trainable, g_total = count_params(self.netG)
        d_trainable, d_non_trainable, d_total = count_params(self.netD)

        self.accelerator.print(
            "🤖 \033[1mGenerator Parameters:\033[0m\n"
            f" - 🔥 Trainable: {g_trainable}\n"
            f" - 🧊 Non-trainable: {g_non_trainable}\n"
            f" - 🤯 Total: {g_total}\n"
            "🦹 \033[1mDiscriminator Parameters:\033[0m\n"
            f" - 🔥 Trainable: {d_trainable}\n"
            f" - 🧊 Non-trainable: {d_non_trainable}\n"
            f" - 🤯 Total: {d_total}"
        )
        self.accelerator.log(
            {
                "generator_trainable_params": g_trainable,
                "discriminator_trainable_params": d_trainable,
            }
        )

    def reset(self):
        super().reset()
        self.min_loss = 0
