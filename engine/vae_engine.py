import os
import torch
import torch.nn.functional as F
import accelerate
from torchvision.utils import save_image, make_grid

from configs import Config
from dataset import get_loader
from engine.base_engine import ImageGenerationEngine
from modeling.model import build_vae
from PIL import Image

class VAEEngine(ImageGenerationEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)
        self.current_epoch = 1
        self.min_loss = float("inf")
        self.accelerator.init_trackers(
            self.cfg.project_name if self.cfg.project_name else self.accelerator.project_configuration.project_dir,
            config=self.cfg.to_dict(),
            init_kwargs={"wandb": self.cfg.to_dict()["wandb"]},
        )
        self.accelerator.log({"base_dir": self.base_dir})

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)
        with self.accelerator.main_process_first():
            train_loader, val_loader, test_loader = get_loader(self.cfg)
        model = build_vae(self.cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def _train_one_epoch(self):
        epoch_progress = self.sub_task_progress.add_task(
            "loader", total=len(self.train_loader)
        )
        self.model.train()
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.accelerator.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            self.accelerator.backward(loss)
            self.optimizer.step()
            epoch_loss += loss.item()

            if self.accelerator.is_main_process:
                self.log_step = self.current_epoch * len(self.train_loader) + batch_idx
                if batch_idx % 100 == 0:
                    self.accelerator.print(f"Epoch [{self.current_epoch}/{self.cfg.training.epochs}] Batch [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
                self.log_results(
                    {"loss/train": loss.item()},
                    step= self.log_step,
                    csv_name="train_steps.csv"
                )
            self.sub_task_progress.update(epoch_progress, advance=1)

        avg_loss = epoch_loss / len(self.train_loader.dataset)
        if self.accelerator.is_main_process:
            epoch_step = (self.current_epoch+1) * len(self.train_loader)
            self.log_step = epoch_step
            self.log_results({"loss/train_epoch": avg_loss}, step=self.log_step , csv_name="train_metrics.csv")

        self.sub_task_progress.remove_task(epoch_progress)
        return avg_loss

    def laplacian_variance(self, img_tensor):
        # img_tensor: (B, C, H, W) or (C, H, W), values in [0, 1] or [0, 255]
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)  # to grayscale
        elif img_tensor.dim() == 3:
            img_tensor = img_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=img_tensor.dtype, device=img_tensor.device).unsqueeze(0).unsqueeze(0)
        lap = F.conv2d(img_tensor, laplacian_kernel, padding=1)
        var = lap.var().item()
        return var

    def sample_images(self, epoch, num_samples=16, save_path=None):
        self.model.eval()
        device = self.accelerator.device
        latent_dim = self.cfg.training.vae.latent_dim
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim, device=device)
            samples = self.model.decode(z)
            samples = samples.cpu()
        if save_path is not None:
            grid = make_grid(samples, nrow=int(num_samples ** 0.5), normalize=True)
            save_image(grid, save_path)

        low = float(samples.min())
        high = float(samples.max())

        img = samples.clone()
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
        all_images = [
            Image.fromarray(arr[i].numpy()) for i in range(arr.size(0))
                    ]
        
        lap_vars = []
        for i in range(samples.size(0)):
            lap_var = self.laplacian_variance(samples[i].reshape(3, samples.size(2), samples.size(3)))
            lap_vars.append(lap_var)
        avg_lap_var = sum(lap_vars) / len(lap_vars)
        if self.accelerator.is_main_process:
            self.log_results({"val/lapl_var": avg_lap_var}, step=self.log_step, csv_name="lapl_var.csv")
            self.accelerator.print(f"Average Laplacian Variance: {avg_lap_var:.4f}")
    
        self.evaluate(epoch, all_images)


    def train(self):
        if self.accelerator.is_main_process:
            self.setup_training()
            self.print_training_details()
        self.accelerator.wait_for_everyone()
        for epoch in range(self.current_epoch, self.cfg.training.epochs + 1):
            self.current_epoch = epoch
            self._train_one_epoch()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and epoch % self.cfg.training.save_image_epochs == 0:
                self.sample_images(
                    epoch=epoch,
                    num_samples=self.cfg.training.metric_calculation_img_count
                )
            if self.stop_training:
                break
        self.accelerator.wait_for_everyone()
    
    def reset(self):
        super().reset()
        self.min_loss = 0