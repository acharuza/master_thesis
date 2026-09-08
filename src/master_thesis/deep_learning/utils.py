import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config):
    required_sections = [
        "experiment",
        "data",
        "model",
        "training",
        "loss",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    model = config["model"]
    if model["rnn_type"].lower() not in ["gru", "lstm"]:
        raise ValueError("model.rnn_type must be gru or lstm")
    if model["hidden_dim"] <= 0:
        raise ValueError("hidden_dim must be positive")
    if model["num_layers"] <= 0:
        raise ValueError("num_layers must be positive")
    training = config["training"]

    if training["lr"] <= 0:
        raise ValueError("learning rate must be positive")
    if training["epochs"] <= 0:
        raise ValueError("epochs must be positive")

    freeze_encoder_epochs = training.get("freeze_encoder_epochs", 0)
    if freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs must be non-negative")

    early_stopping_patience = training.get("early_stopping_patience", 0)
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")

    early_stopping_min_delta = training.get("early_stopping_min_delta", 0.0)
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    loss = config.get("loss", {}) or {}
    pos_weight_max = loss.get("pos_weight_max", 20.0)
    if pos_weight_max <= 0:
        raise ValueError("pos_weight_max must be positive")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    score,
    path,
    config=None,
):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "score": score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        },
        path,
    )


def tensor_to_label_dict(labels, label_names, label_indices=None):
    if label_indices is None:
        label_indices = range(len(label_names))

    return {name: labels[:, index] for name, index in zip(label_names, label_indices)}
