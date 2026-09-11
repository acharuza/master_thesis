import pandas as pd
import numpy as np
from master_thesis.deep_learning.data import ECGDataset
import torch
from utils.config import (
    WAVEFORMS_PATH,
    METADATA_PATH,
    TABULAR_PATH,
    TABULAR_FEATURES,
)


def load_dataset(pretrain=False):
    dataset = ECGDataset(
        waveforms_path=str(WAVEFORMS_PATH),
        metadata_path=str(METADATA_PATH),
        split="test",
        pretrain=pretrain,
        tabular_path=str(TABULAR_PATH),
        tabular_features=TABULAR_FEATURES,
    )
    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)
    return dataset, metadata
