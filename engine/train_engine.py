import json
import os
import time

import accelerate
import torch
import torch.nn.functional as F

from configs import Config
from dataset import get_loader
from engine.base_engine import BaseEngine
from modeling import build_loss, build_model
from utils.metrics import Metrics


class Engine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: Config):
        super().__init__(accelerator, cfg)

        self.min_loss = float("inf")
        self.current_epoch = 1
        self.max_acc = 0
        self.early_stopping_patience = self.cfg.training.early_stopping_patience
        self.early_stopping_counter = 0
        self.stop_training = False

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

        # Resume or not
        if self.cfg.model.resume_path is not None:
            with self.accelerator.main_process_first():
                self.load_from_checkpoint()

    def load_from_checkpoint(self):
        """
        Load model and optimizer from checkpoint for resuming training.
        Modify this for custom components if needed.
        """
        checkpoint = self.cfg.model.resume_path
        if not os.path.exists(checkpoint):
            self.accelerator.print(f"[WARN] Checkpoint {checkpoint} not found. Skipping...")
            return
        self.accelerator.load_state(checkpoint)

        if not os.path.exists(os.path.join(checkpoint, "meta_data.json")):
            self.accelerator.print(
                f"[WARN] meta data for resuming training is not found in {checkpoint}. Skipping..."
            )
            return

        with open(os.path.join(checkpoint, "meta_data.json"), "r") as f:
            meta_data = json.load(f)
        self.current_epoch = meta_data.get("epoch", 0) + 1
        self.max_acc = meta_data.get("max_acc", 0)
        self.accelerator.print(
            f"[WARN] Checkpoint loaded from {self.cfg.model.resume_path}, continue training or validate..."
        )
        del checkpoint

    def save_checkpoint(self, save_path: str):
        self.accelerator.save_state(save_path)
        unwrapped_model = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=True)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model.pth"))
        with open(os.path.join(save_path, "meta_data.json"), "w") as f:
            json.dump(
                {
                    "epoch": self.current_epoch,
                    "max_acc": self.max_acc,
                },
                f,
            )

    def _train_one_epoch(self):
        epoch_progress = self.sub_task_progress.add_task("loader", total=len(self.train_loader))
        self.model.train()
        step_loss = 0
        start = time.time()

        if self.scheduler is not None:
            self.accelerator.log(
                {
                    "scheduler_lr": self.scheduler.get_last_lr()[0],
                }
            )

        all_preds = []
        all_labels = []
        all_losses = []

        for loader_idx, (img, label) in enumerate(self.train_loader, 1):
            current_step = (self.current_epoch - 1) * len(self.train_loader) + loader_idx
            self.data_time.update(time.time() - start)
            with self.accelerator.accumulate(self.model):
                output = self.model(img)
                loss = self.loss_fn(output, label)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = self.accelerator.gather(loss.detach().cpu().clone()).mean()
                step_loss += loss.item() / self.cfg.training.accum_iter
                all_losses.append(loss.item())

                batch_pred, batch_label = self.accelerator.gather_for_metrics((output, label))
                all_preds.append(batch_pred)
                all_labels.append(batch_label)

            self.iter_time.update(time.time() - start)

            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                self.log_results(
                    {
                        "loss/train": step_loss,
                    },
                    step=current_step,
                    csv_name="train_steps.csv",
                )
                step_loss = 0

            self.accelerator.log(
                {
                    "time/iter": self.iter_time.val,
                    "time/data": self.data_time.val,
                },
                step=current_step,
            )
            self.sub_task_progress.update(epoch_progress, advance=1)

            start = time.time()

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metric_results = self.metrics.compute(all_preds, all_labels, all_losses)

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"train acc.: {metric_results['accuracy']:.3f}, loss: {metric_results['loss']:.3f}, "
                f"precision: {metric_results['precision']:.3f}, recall: {metric_results['recall']:.3f}, f1: {metric_results['f1']:.3f}"
            )
            self.log_results(
                {
                    "acc/train": metric_results["accuracy"],
                    "loss/train_epoch": metric_results["loss"],
                    "precision/train": metric_results["precision"],
                    "recall/train": metric_results["recall"],
                    "f1/train": metric_results["f1"],
                },
                step=self.current_epoch * len(self.train_loader),  # Use train steps
                csv_name="train_metrics.csv",
            )
        if self.scheduler is not None:
            self.scheduler.step()
        self.sub_task_progress.remove_task(epoch_progress)

    def validate(self):
        valid_progress = self.sub_task_progress.add_task("validate", total=len(self.val_loader))
        all_preds = []
        all_labels = []
        all_losses = []

        self.model.eval()
        with torch.no_grad():
            for img, label in self.val_loader:
                pred = self.model(img)
                loss = F.cross_entropy(pred, label)
                all_losses.append(loss.item())

                batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
                all_preds.append(batch_pred)
                all_labels.append(batch_label)

                self.sub_task_progress.update(valid_progress, advance=1)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metric_results = self.metrics.compute(all_preds, all_labels, all_losses)

        if self.accelerator.is_main_process and metric_results["accuracy"] > self.max_acc:
            save_path = os.path.join(self.base_dir, "checkpoint")
            self.accelerator.print(
                f"new best found with: {metric_results['accuracy']:.3f}, save to {save_path}"
            )
            self.max_acc = metric_results["accuracy"]
            self.save_checkpoint(os.path.join(save_path, f"epoch_{self.current_epoch}"))
            self.early_stopping_counter = 0  # Reset counter on improvement

            # Save best model
            self.accelerator.save_state(self.base_dir)
            unwrapped_model = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=True)
            torch.save(unwrapped_model.state_dict(), os.path.join(self.base_dir, "best.pth"))
        else:
            self.early_stopping_counter += 1  # Increment counter if no improvement

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"val. acc.: {metric_results['accuracy']:.3f}, loss: {metric_results['loss']:.3f}, "
                f"precision: {metric_results['precision']:.3f}, recall: {metric_results['recall']:.3f}, f1: {metric_results['f1']:.3f}"
            )
            self.log_results(
                {
                    "acc/val": metric_results["accuracy"],
                    "max_acc/val": self.max_acc,
                    "loss/val": metric_results["loss"],
                    "precision/val": metric_results["precision"],
                    "recall/val": metric_results["recall"],
                    "f1/val": metric_results["f1"],
                },
                step=self.current_epoch * len(self.train_loader),  # Use train steps
                csv_name="validation_metrics.csv",
            )

        # Check for early stopping
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.accelerator.print("Early stopping triggered. Stopping training.")
            self.stop_training = True  # Flag to stop training

        self.sub_task_progress.remove_task(valid_progress)

    def test(self):
        test_progress = self.sub_task_progress.add_task("test", total=len(self.test_loader))
        all_preds = []
        all_labels = []
        all_losses = []

        self.model.eval()
        with torch.no_grad():
            for img, label in self.test_loader:
                pred = self.model(img)
                loss = F.cross_entropy(pred, label)
                all_losses.append(loss.item())

                batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
                all_preds.append(batch_pred)
                all_labels.append(batch_label)

                self.sub_task_progress.update(test_progress, advance=1)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metric_results = self.metrics.compute(all_preds, all_labels, all_losses)

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"test acc.: {metric_results['accuracy']:.3f}, loss: {metric_results['loss']:.3f}, "
                f"precision: {metric_results['precision']:.3f}, recall: {metric_results['recall']:.3f}, f1: {metric_results['f1']:.3f}"
            )
            self.log_results(
                {
                    "acc/test": metric_results["accuracy"],
                    "loss/test": metric_results["loss"],
                    "precision/test": metric_results["precision"],
                    "recall/test": metric_results["recall"],
                    "f1/test": metric_results["f1"],
                },
                csv_name="test_metrics.csv",
            )

        self.sub_task_progress.remove_task(test_progress)

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)

        with self.accelerator.main_process_first():
            train_loader, val_loader, test_loader = get_loader(self.cfg)
        model = build_model(self.cfg, train_loader.dataset.num_classes)
        self.loss_fn = build_loss(
            self.cfg,
            (
                torch.tensor(train_loader.dataset.class_weights).to(self.accelerator.device)
                if self.cfg.training.sampling_strategy == "weights"
                else None
            ),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.lr * self.accelerator.num_processes,
            weight_decay=self.cfg.training.weight_decay,
        )
        if self.cfg.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.training.epochs
            )
        else:
            scheduler = None
        self.metrics = Metrics(train_loader.dataset.num_classes)

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
            acc=self.max_acc,
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
                self.validate()
            if self.stop_training:
                break
            self.epoch_progress.update(train_progress, advance=1, acc=self.max_acc)
        self.epoch_progress.stop_task(train_progress)
        self.accelerator.wait_for_everyone()
        self.accelerator.print("Loading best model for testing...")

        # Load the state dictionary from the .pth file
        best_model_path = os.path.join(self.base_dir, "best.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Model file not found at {best_model_path}")

        state_dict = torch.load(best_model_path, map_location=self.accelerator.device)

        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)

        self.accelerator.print("Testing...")
        self.test()
        self.accelerator.wait_for_everyone()

    def reset(self):
        super().reset()
        self.max_acc = 0
