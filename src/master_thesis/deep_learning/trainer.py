import json
import os
import torch
from tqdm import tqdm

from master_thesis.deep_learning.metrics import compute_metrics
from master_thesis.deep_learning.utils import (
    save_checkpoint,
    tensor_to_label_dict,
)
from master_thesis.deep_learning.losses import MultiTaskBCELoss


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        label_names,
        pos_weights=None,
    ):
        self.config = config

        self.device = torch.device(
            config["training"]["device"] if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        if config["training"].get("compile", False):
            self.model = torch.compile(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_names = label_names
        self.pretrain = config["data"].get("pretrain", False)
        self.model_use_tabular = config.get("tabular", {}).get("enabled", False)
        self.freeze_encoder_epochs = config["training"].get(
            "freeze_encoder_epochs",
            0,
        )
        self.early_stopping_patience = config["training"].get(
            "early_stopping_patience",
            0,
        )
        self.early_stopping_min_delta = config["training"].get(
            "early_stopping_min_delta",
            0.0,
        )
        self.label_indices = [
            train_loader.dataset.label_columns.index(name) for name in label_names
        ]

        self.history = {
            "experiment": config["experiment"]["name"],
            "config": config,
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "disease_macro_auroc": [],
            "disease_macro_auprc": [],
            "disease_macro_f1": [],
            "lr": [],
        }

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        scheduler_cfg = config["training"].get(
            "scheduler",
            {},
        )

        scheduler_type = scheduler_cfg.get(
            "type",
            "reduce_on_plateau",
        ).lower()

        if scheduler_type == "reduce_on_plateau":

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 5),
                min_lr=scheduler_cfg.get("min_lr", 1e-6),
            )

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        self.loss_fn = MultiTaskBCELoss(
            diseases=label_names,
            pretrain=self.pretrain,
            pos_weights=pos_weights,
        )

        self.use_amp = (
            config["training"].get("amp", False) and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.best_score = float("inf")
        self.early_stopping_counter = 0

    def _backbone_modules(self):
        model = getattr(self.model, "_orig_mod", self.model)
        modules = [model.encoder]

        if getattr(model, "cnn", None) is not None:
            modules.insert(0, model.cnn)

        return modules

    def _set_backbone_trainable(self, trainable):
        for module in self._backbone_modules():
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def _freeze_encoder(self):
        self._set_backbone_trainable(False)

    def _unfreeze_encoder(self):
        self._set_backbone_trainable(True)

    def save_history(self, path):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(path, "w") as f:
            json.dump(
                self.history,
                f,
                indent=4,
            )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(
            self.train_loader,
            desc="training",
        ):
            x = batch["waveform"].to(
                self.device,
                non_blocking=True,
            )

            tabular = None
            if self.model_use_tabular:
                tabular = batch["tabular"].to(
                    self.device,
                    non_blocking=True,
                )

            y = batch["labels"].to(
                self.device,
                non_blocking=True,
            )
            targets = tensor_to_label_dict(
                y,
                self.label_names,
                self.label_indices,
            )
            self.optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                logits, _ = self.model(x, tabular)
                loss = self.loss_fn(
                    logits,
                    targets,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_clip = self.config["training"].get("gradient_clip")

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=gradient_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        all_probs = []
        all_labels = []

        total_loss = 0.0

        for batch in tqdm(
            self.val_loader,
            desc="validation",
        ):
            x = batch["waveform"].to(
                self.device,
                non_blocking=True,
            )

            tabular = None
            if self.model_use_tabular:
                tabular = batch["tabular"].to(
                    self.device,
                    non_blocking=True,
                )

            y = batch["labels"].to(
                self.device,
                non_blocking=True,
            )
            targets = tensor_to_label_dict(
                y,
                self.label_names,
                self.label_indices,
            )

            logits, _ = self.model(x, tabular)

            loss = self.loss_fn(
                logits,
                targets,
            )
            total_loss += loss.item()

            batch_probs = []

            for disease in self.label_names:
                prob = torch.sigmoid(logits[disease])
                batch_probs.append(prob)

            batch_probs = torch.cat(
                batch_probs,
                dim=1,
            )
            all_probs.append(batch_probs.cpu())
            all_labels.append(y.cpu())

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()

        disease_labels = labels[
            :,
            self.label_indices,
        ]
        metrics = compute_metrics(
            disease_labels,
            probs,
            self.label_names,
        )

        metrics["val_loss"] = total_loss / len(self.val_loader)
        return metrics

    def fit(
        self,
        epochs,
        checkpoint_path,
    ):
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if checkpoint_dir:
            os.makedirs(
                checkpoint_dir,
                exist_ok=True,
            )

        history_path = os.path.join(
            checkpoint_dir,
            "best_history.json",
        )

        if not self.pretrain and self.freeze_encoder_epochs > 0:
            self._freeze_encoder()
            print(f"Frozen encoder for the first {self.freeze_encoder_epochs} epochs")

        for epoch in range(epochs):

            if (
                not self.pretrain
                and epoch == self.freeze_encoder_epochs
                and self.freeze_encoder_epochs > 0
            ):
                self._unfreeze_encoder()
                print("Unfroze encoder for finetuning")

            train_loss = self.train_epoch()
            metrics = self.validate()

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(metrics["val_loss"]))
            self.history["disease_macro_auroc"].append(
                float(metrics["disease_macro_auroc"])
            )
            self.history["disease_macro_auprc"].append(
                float(metrics["disease_macro_auprc"])
            )
            self.history["disease_macro_f1"].append(float(metrics["disease_macro_f1"]))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.save_history(history_path)

            self.scheduler.step(metrics["val_loss"])

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"train loss: " f"{train_loss:.4f}")
            print(f"val loss: " f"{metrics['val_loss']:.4f}")
            print(f"disease AUROC: " f"{metrics['disease_macro_auroc']:.4f}")
            print(f"disease AUPRC: " f"{metrics['disease_macro_auprc']:.4f}")
            print(f"disease F1: " f"{metrics['disease_macro_f1']:.4f}")
            print(f"learning rate: " f"{self.optimizer.param_groups[0]['lr']:.2e}")

            score = metrics["val_loss"]
            improved = score < (self.best_score - self.early_stopping_min_delta)
            if improved:
                self.best_score = score
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    score,
                    checkpoint_path,
                    self.config,
                )
                print(f"NEW BEST CHECKPOINT " f"(val_loss={score:.4f})")
                self.early_stopping_counter = 0
            else:
                print(f"best val_loss so far: " f"{self.best_score:.4f}")
                if self.early_stopping_patience > 0:
                    self.early_stopping_counter += 1
                    print(
                        f"early stopping counter: {self.early_stopping_counter}/"
                        f"{self.early_stopping_patience}"
                    )
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print("Early stopping triggered " f"after epoch {epoch + 1}")
                        break
