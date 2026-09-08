import argparse
import os
import torch
from torch.utils.data import DataLoader
from master_thesis.deep_learning.utils import load_config, set_seed
from master_thesis.deep_learning.data import ECGDataset
from master_thesis.deep_learning.models.ecg_model import ECGModel
from master_thesis.deep_learning.trainer import Trainer


def main(config_path):
    config = load_config(config_path)
    set_seed(config["seed"])

    pretrain = config["data"].get("pretrain", False)
    pretrained_checkpoint = config["training"].get("pretrained_checkpoint")

    tabular_cfg = config["tabular"]

    train_dataset = ECGDataset(
        waveforms_path=config["data"]["train_waveforms"],
        metadata_path=config["data"]["metadata"],
        split="train",
        pretrain=config["data"]["pretrain"],
        tabular_path=(
            config["data"]["train_tabular"] if tabular_cfg["enabled"] else None
        ),
        tabular_features=(tabular_cfg["features"] if tabular_cfg["enabled"] else None),
    )

    val_dataset = ECGDataset(
        waveforms_path=config["data"]["val_waveforms"],
        metadata_path=config["data"]["metadata"],
        split="val",
        pretrain=config["data"]["pretrain"],
        tabular_path=(
            config["data"]["val_tabular"] if tabular_cfg["enabled"] else None
        ),
        tabular_features=(tabular_cfg["features"] if tabular_cfg["enabled"] else None),
    )

    loss_cfg = config.get("loss", {}) or {}

    shd_name = loss_cfg.get(
        "shd_name",
        "shd_moderate_or_greater_flag",
    )
    pos_weight_max = loss_cfg.get("pos_weight_max", 20.0)

    if pretrain:
        label_names = train_dataset.label_columns
    else:
        label_names = [name for name in train_dataset.label_columns if name != shd_name]

    if len(label_names) == 0:
        raise ValueError("No disease labels found.")

    if pretrain:
        print("Pretrain label:")
    else:
        print("Disease labels:")

    for name in label_names:
        print(f"  - {name}")

    if not pretrain:
        print(f"Excluded SHD label: {shd_name}")

    all_pos_weights = train_dataset.get_pos_weights()

    if pretrain:
        pos_weights = all_pos_weights
    else:
        disease_indices = [
            train_dataset.label_columns.index(name) for name in label_names
        ]
        pos_weights = all_pos_weights[disease_indices].clamp(max=pos_weight_max)

        print(f"Capped disease pos_weight at {pos_weight_max}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["data"]["num_workers"] > 0,
        prefetch_factor=(2 if config["data"]["num_workers"] > 0 else None),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["data"]["num_workers"] > 0,
        prefetch_factor=(2 if config["data"]["num_workers"] > 0 else None),
    )

    model = ECGModel(
        diseases=label_names,
        config=config,
    )

    if not pretrain:
        if not pretrained_checkpoint:
            raise ValueError(
                "training.pretrained_checkpoint is required for finetuning"
            )

        if not os.path.exists(pretrained_checkpoint):
            raise FileNotFoundError(pretrained_checkpoint)

        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        load_result = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded pretrained weights from {pretrained_checkpoint}")

        if load_result.missing_keys:
            print("Missing keys:")
            for key in load_result.missing_keys:
                print(f"  - {key}")

        if load_result.unexpected_keys:
            print("Unexpected keys:")
            for key in load_result.unexpected_keys:
                print(f"  - {key}")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config,
        label_names,
        pos_weights=pos_weights,
    )

    checkpoint_path = os.path.join(
        "checkpoints",
        config["experiment"]["name"],
        "best.pt",
    )

    trainer.fit(
        config["training"]["epochs"],
        checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
