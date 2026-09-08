import argparse
import copy
import os
from unittest.mock import patch

import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from master_thesis.deep_learning.data import ECGDataset
from master_thesis.deep_learning.models.ecg_model import ECGModel
from master_thesis.deep_learning.trainer import Trainer
from master_thesis.deep_learning.utils import load_config, set_seed


def sample_hyperparameters(trial: optuna.Trial, base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    pretrain = cfg["data"].get("pretrain", False)
    use_cnn = cfg["model"].get("use_cnn", False)

    if pretrain:
        cfg["training"]["lr"] = trial.suggest_float("pretrain_lr", 1e-4, 5e-4, log=True)
    else:
        cfg["training"]["lr"] = trial.suggest_float("finetune_lr", 3e-5, 3e-4, log=True)
        cfg["training"]["freeze_encoder_epochs"] = trial.suggest_int(
            "freeze_encoder_epochs", 0, 10, step=2
        )
        cfg["loss"]["pos_weight_max"] = trial.suggest_float(
            "pos_weight_max", 5.0, 50.0, log=True
        )

    cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-5, 1e-3, log=True
    )
    cfg["training"]["scheduler"]["patience"] = trial.suggest_int(
        "scheduler_patience", 3, 8
    )
    cfg["training"]["scheduler"]["factor"] = trial.suggest_float(
        "scheduler_factor", 0.2, 0.5, step=0.1
    )

    if cfg.get("tabular", {}).get("enabled", False):
        cfg["tabular"]["hidden_dim"] = trial.suggest_categorical(
            "tabular_hidden_dim", [64, 128, 256]
        )
        cfg["tabular"]["fusion_dim"] = trial.suggest_categorical(
            "tabular_fusion_dim", [128, 256, 512]
        )
        cfg["tabular"]["dropout"] = trial.suggest_float(
            "tabular_dropout", 0.1, 0.4, step=0.1
        )

    cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256])
    cfg["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 2)
    cfg["model"]["dropout"] = trial.suggest_float("rnn_dropout", 0.1, 0.4, step=0.1)
    cfg["model"]["head_dropout"] = trial.suggest_float(
        "head_dropout", 0.1, 0.4, step=0.1
    )

    if use_cnn:
        channel_choice = trial.suggest_categorical("cnn_channels", ["32_64", "64_128"])
        cfg["model"]["cnn"]["channels"] = (
            [32, 64] if channel_choice == "32_64" else [64, 128]
        )

        kernel_choice = trial.suggest_categorical(
            "cnn_kernels", ["7_5", "11_7", "15_7"]
        )
        cfg["model"]["cnn"]["kernels"] = {
            "7_5": [7, 5],
            "11_7": [11, 7],
            "15_7": [15, 7],
        }[kernel_choice]
        cfg["model"]["cnn"]["dropout"] = trial.suggest_float(
            "cnn_dropout", 0.0, 0.3, step=0.1
        )

    return cfg


def create_objective(
    base_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_names: list,
    all_pos_weights: torch.Tensor,
    output_csv: str,
):
    pretrain = base_cfg["data"].get("pretrain", False)

    def objective(trial: optuna.Trial) -> float:
        config = sample_hyperparameters(trial, base_cfg)
        set_seed(config["seed"])

        if pretrain:
            pos_weights = all_pos_weights
        else:
            disease_indices = [
                train_loader.dataset.label_columns.index(name) for name in label_names
            ]
            pos_weight_max = config.get("loss", {}).get("pos_weight_max", 20.0)
            pos_weights = all_pos_weights[disease_indices].clamp(max=pos_weight_max)

        model = ECGModel(diseases=label_names, config=config)

        if not pretrain:
            pretrained_ckpt = config["training"].get("pretrained_checkpoint")
            if pretrained_ckpt and os.path.exists(pretrained_ckpt):
                checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            label_names=label_names,
            pos_weights=pos_weights,
        )

        epochs = config["training"]["epochs"]
        if not pretrain and trainer.freeze_encoder_epochs > 0:
            trainer._freeze_encoder()

        best_metric = -float("inf") if not pretrain else float("inf")
        best_val_loss = float("inf")
        best_auroc = 0.0

        early_stopping_counter = 0
        patience = config["training"].get("early_stopping_patience", 8)
        min_delta = config["training"].get("early_stopping_min_delta", 0.001)
        stopped_epoch = epochs

        for epoch in range(epochs):
            if (
                not pretrain
                and epoch == trainer.freeze_encoder_epochs
                and trainer.freeze_encoder_epochs > 0
            ):
                trainer._unfreeze_encoder()

            with patch(
                "master_thesis.deep_learning.trainer.tqdm",
                lambda iterable, **kwargs: iterable,
            ):
                _ = trainer.train_epoch()
                metrics = trainer.validate()

            trainer.scheduler.step(metrics["val_loss"])

            val_loss = metrics["val_loss"]
            current_metric = (
                metrics["disease_macro_auroc"] if not pretrain else metrics["val_loss"]
            )

            if not pretrain:
                best_auroc = max(best_auroc, metrics["disease_macro_auroc"])
                best_metric = max(best_metric, current_metric)
            else:
                best_metric = min(best_metric, current_metric)

            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                if patience > 0:
                    early_stopping_counter += 1

            trial.report(current_metric, step=epoch)
            if trial.should_prune():
                _log_trial_to_csv(
                    trial,
                    output_csv,
                    state="PRUNED",
                    best_metric=best_metric,
                    best_loss=best_val_loss,
                    best_auroc=best_auroc,
                    epoch=epoch,
                )
                raise optuna.TrialPruned()

            if patience > 0 and early_stopping_counter >= patience:
                stopped_epoch = epoch + 1
                break

        _log_trial_to_csv(
            trial,
            output_csv,
            state="COMPLETE",
            best_metric=best_metric,
            best_loss=best_val_loss,
            best_auroc=best_auroc,
            epoch=stopped_epoch,
        )
        return best_metric

    return objective


def _log_trial_to_csv(
    trial: optuna.Trial,
    output_csv: str,
    state: str,
    best_metric: float,
    best_loss: float,
    best_auroc: float,
    epoch: int,
):
    row = {
        "trial_number": trial.number,
        "state": state,
        "stopped_epoch": epoch,
        "target_metric": best_metric,
        "best_val_loss": best_loss,
        "best_macro_auroc": best_auroc,
        **trial.params,
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode="a", index=False, header=header)


def run_tuning(config_path: str, study_name: str, output_csv: str, n_trials: int = 40):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_config = load_config(config_path)
    pretrain = base_config["data"].get("pretrain", False)
    tabular_cfg = base_config.get("tabular", {})

    train_dataset = ECGDataset(
        waveforms_path=base_config["data"]["train_waveforms"],
        metadata_path=base_config["data"]["metadata"],
        split="train",
        pretrain=pretrain,
        tabular_path=(
            base_config["data"]["train_tabular"] if tabular_cfg.get("enabled") else None
        ),
        tabular_features=(
            tabular_cfg.get("features") if tabular_cfg.get("enabled") else None
        ),
    )
    val_dataset = ECGDataset(
        waveforms_path=base_config["data"]["val_waveforms"],
        metadata_path=base_config["data"]["metadata"],
        split="val",
        pretrain=pretrain,
        tabular_path=(
            base_config["data"]["val_tabular"] if tabular_cfg.get("enabled") else None
        ),
        tabular_features=(
            tabular_cfg.get("features") if tabular_cfg.get("enabled") else None
        ),
    )

    loss_cfg = base_config.get("loss", {}) or {}
    shd_name = loss_cfg.get("shd_name", "shd_moderate_or_greater_flag")
    label_names = (
        train_dataset.label_columns
        if pretrain
        else [n for n in train_dataset.label_columns if n != shd_name]
    )
    all_pos_weights = train_dataset.get_pos_weights()

    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config["data"]["batch_size"],
        shuffle=True,
        num_workers=base_config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=base_config["data"]["num_workers"] > 0,
        prefetch_factor=2 if base_config["data"]["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config["data"]["batch_size"],
        shuffle=False,
        num_workers=base_config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=base_config["data"]["num_workers"] > 0,
        prefetch_factor=2 if base_config["data"]["num_workers"] > 0 else None,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if not pretrain else "minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=123),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, max_resource=base_config["training"]["epochs"]
        ),
    )

    objective_fn = create_objective(
        base_config, train_loader, val_loader, label_names, all_pos_weights, output_csv
    )

    metric_name = "AUROC" if not pretrain else "ValLoss"
    pbar = tqdm(total=n_trials, desc=f"Optuna [{study_name}]", unit="trial")

    def tqdm_callback(study_instance: optuna.Study, trial: optuna.Trial):
        pbar.update(1)
        try:
            best_val = study_instance.best_value
            pbar.set_postfix({"Best " + metric_name: f"{best_val:.4f}"})
        except ValueError:
            pass

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[tqdm_callback],
        gc_after_trial=True,
    )
    pbar.close()

    print(f"\nOptimization Finished | Best Trial #{study.best_trial.number}")
    print(f"Best Target Metric: {study.best_value:.4f}")
    print(f"Results saved in real-time to: {output_csv}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--study_name", default="ecg_optuna_study")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Path to CSV file where all trial results will be saved.",
    )
    parser.add_argument("--n_trials", type=int, default=40)
    args = parser.parse_args()

    csv_path = args.output_csv or f"{args.study_name}_results.csv"
    run_tuning(args.config, args.study_name, csv_path, args.n_trials)
