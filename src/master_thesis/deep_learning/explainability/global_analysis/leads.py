from __future__ import annotations
import numpy as np
import torch
from master_thesis.deep_learning.explainability.common.wrappers import ECGTargetWrapper
from master_thesis.deep_learning.explainability.common.integrated_gradients import (
    compute_integrated_gradients,
)

ECG_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def compute_global_lead_importance(
    model,
    dataset,
    target,
    device=None,
    n_steps=50,
    indices=None,
):
    """Compute disease-specific global ECG lead importance."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    wrapper = ECGTargetWrapper(
        model=model,
        target=target,
    ).to(device)

    if indices is None:
        indices = range(len(dataset))

    per_sample_importance = []
    valid_indices = []

    for index in indices:
        sample = dataset[index]
        waveform = sample["waveform"].unsqueeze(0).to(device)
        tabular = sample.get("tabular")

        if tabular is not None:
            tabular = tabular.unsqueeze(0).to(device)

        waveform_attribution, _ = compute_integrated_gradients(
            model=wrapper,
            waveform=waveform,
            tabular=tabular,
            n_steps=n_steps,
        )
        # [1, time, 12] -> [time, 12]
        attribution = waveform_attribution[0]

        # absolute because we want to measure contribution regardless of direction
        lead_importance = torch.mean(
            torch.abs(attribution),
            dim=0,
        )
        total = lead_importance.sum()

        if total <= 0:
            continue

        lead_importance = lead_importance / total
        per_sample_importance.append(lead_importance.detach().cpu().numpy())
        valid_indices.append(index)

    if not per_sample_importance:
        raise RuntimeError(f"No valid samples were available for target '{target}'.")

    per_sample_importance = np.stack(
        per_sample_importance,
        axis=0,
    )

    median = np.median(
        per_sample_importance,
        axis=0,
    )
    q25 = np.percentile(
        per_sample_importance,
        25,
        axis=0,
    )
    q75 = np.percentile(
        per_sample_importance,
        75,
        axis=0,
    )
    mean = np.mean(
        per_sample_importance,
        axis=0,
    )
    ranking = np.argsort(median)[::-1]
    return {
        "target": target,
        "lead_names": ECG_LEAD_NAMES,
        "per_sample_importance": per_sample_importance,
        "indices": np.asarray(valid_indices),
        "mean": mean,
        "median": median,
        "q25": q25,
        "q75": q75,
        "ranking": ranking,
    }
