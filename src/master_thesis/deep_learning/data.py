import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Dataset for ECG waveforms, labels, and optional tabular features."""

    SHD_LABEL = "shd_moderate_or_greater_flag"

    def __init__(
        self,
        waveforms_path: str,
        metadata_path: str,
        split: str = "train",
        pretrain: bool = False,
        tabular_path: str | None = None,
        tabular_features: list[str] | None = None,
    ):
        if not os.path.exists(waveforms_path):
            raise FileNotFoundError(waveforms_path)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)

        if tabular_path is not None and not os.path.exists(tabular_path):
            raise FileNotFoundError(tabular_path)

        meta = pd.read_csv(metadata_path)

        if "split" not in meta.columns:
            raise ValueError("Metadata must contain a 'split' column.")
        meta_split = meta[meta["split"] == split].reset_index(drop=True)

        if len(meta_split) == 0:
            raise ValueError(f"No samples found for split '{split}'")
        self.meta = meta_split

        flag_columns = [
            column for column in self.meta.columns if column.endswith("_flag")
        ]
        if pretrain:
            if self.SHD_LABEL not in self.meta.columns:
                raise ValueError(
                    f"Pretraining requires '{self.SHD_LABEL}' " "in metadata."
                )
            self.label_columns = [self.SHD_LABEL]
        else:
            self.label_columns = flag_columns

        if len(self.label_columns) == 0:
            raise ValueError(
                "No label columns ending with '_flag' " "found in metadata."
            )
        self.labels = self.meta[self.label_columns].to_numpy(dtype=np.float32)

        self.waveforms_path = waveforms_path
        self.waveforms = None
        waveforms = np.load(
            self.waveforms_path,
            mmap_mode="r",
        )
        if len(waveforms) != len(meta_split):
            raise ValueError(
                f"Waveforms length ({len(waveforms)}) != "
                f"metadata split count ({len(meta_split)}) "
                f"for split '{split}'."
            )
        del waveforms

        self.tabular = None
        self.tabular_features = (
            list(tabular_features) if tabular_features is not None else None
        )
        self.num_tabular_features = (
            len(self.tabular_features) if self.tabular_features is not None else 0
        )

        if tabular_path is not None:
            if self.num_tabular_features == 0:
                raise ValueError(
                    "tabular_features must be provided when "
                    "tabular_path is provided."
                )
            tabular = np.load(
                tabular_path,
                mmap_mode="r",
            )

            # [num_samples, num_features]
            if tabular.ndim != 2:
                raise ValueError(
                    f"Expected tabular array with shape " f"[N, F], got {tabular.shape}"
                )
            if len(tabular) != len(meta_split):
                raise ValueError(
                    f"Tabular length ({len(tabular)}) != "
                    f"metadata split count ({len(meta_split)}) "
                    f"for split '{split}'."
                )
            if tabular.shape[1] != self.num_tabular_features:
                raise ValueError(
                    f"Tabular feature count mismatch: "
                    f"config specifies {self.num_tabular_features} "
                    f"features {self.tabular_features}, "
                    f"but the .npy file contains "
                    f"{tabular.shape[1]} features."
                )
            self.tabular = tabular

    def _get_waveforms(self):
        """
        Lazily open the waveform memmap.

        Each DataLoader worker opens its own memmap.
        """
        if self.waveforms is None:
            self.waveforms = np.load(
                self.waveforms_path,
                mmap_mode="r",
            )
        return self.waveforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveforms = self._get_waveforms()
        wf = np.array(
            waveforms[idx],
            copy=True,
        )
        wf = np.squeeze(wf)
        if wf.shape != (2500, 12):
            raise ValueError(f"Expected ECG shape (2500, 12), " f"got {wf.shape}")
        waveform = torch.from_numpy(wf).float()

        labels = torch.from_numpy(self.labels[idx]).float()
        sample = {
            "waveform": waveform,
            "labels": labels,
        }

        if self.tabular is not None:
            tabular = np.array(
                self.tabular[idx],
                copy=True,
            )
            sample["tabular"] = torch.from_numpy(tabular).float()

        return sample

    def get_pos_weights(self):
        positives = self.labels.sum(axis=0)
        negatives = len(self.labels) - positives

        weights = negatives / (positives + 1e-6)

        return torch.tensor(
            weights,
            dtype=torch.float32,
        )
