import os
import torch
import torch.nn as nn
import torch.optim as optim

from engine.base_engine import BaseEngine
from modeling.model import DCGANGenerator, DCGANDiscriminator
from dataset import get_loader

class DCGANEngine(BaseEngine):
    def __init__(self, accelerator, cfg):
        super().__init__(accelerator, cfg)
        self.device = self.accelerator.device

        nz = getattr(cfg.model, "nz", 100)
        ngf = getattr(cfg.model, "ngf", 64)
        ndf = getattr(cfg.model, "ndf", 64)
        nc = getattr(cfg.model, "nc", 3)

        self.netG = DCGANGenerator().to(self.device)
        self.netD = DCGANDiscriminator().to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999))

        # Data
        self.train_loader, _, _ = get_loader(cfg)

        (
            self.netG,
            self.netD,
            self.optimizerG,
            self.optimizerD,
            self.train_loader,
        ) = self.accelerator.prepare(
            self.netG, self.netD, self.optimizerG, self.optimizerD, self.train_loader
        )

        self.nz = nz

    def train(self):
        self.netG.train()
        self.netD.train()
        for epoch in range(self.cfg.training.epochs):
            for i, (real, _) in enumerate(self.train_loader):
                b_size = real.size(0)
                real = real.to(self.device)
                label_real = torch.ones(b_size, device=self.device)
                label_fake = torch.zeros(b_size, device=self.device)

                self.optimizerD.zero_grad()
                output_real = self.netD(real).view(-1)
                lossD_real = self.criterion(output_real, label_real)

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
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

                if self.accelerator.is_main_process and i % 100 == 0:
                    self.accelerator.print(
                        f"[{epoch}/{self.cfg.training.epochs}][{i}/{len(self.train_loader)}] "
                        f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}"
                    )

    def print_model_details(self):
        def count_params(model):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
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
        self.accelerator.log({
            "generator_trainable_params": g_trainable,
            "discriminator_trainable_params": d_trainable,
        })