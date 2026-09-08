from __future__ import annotations
import numpy as np
import torch
import shap
from master_thesis.deep_learning.explainability.common.wrappers import (
    TabularSHAPWrapper,
)


def compute_tabular_shap(
    model,
    dataset,
    index,
    target,
    device=None,
    background_indices=None,
):
    """Compute local SHAP values for the tabular features of one patient."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    sample = dataset[index]
    waveform = sample["waveform"].unsqueeze(0).to(device)
    tabular = sample["tabular"].unsqueeze(0).to(device)

    if background_indices is None:
        background_indices = range(min(100, len(dataset)))

    background = np.stack([dataset[i]["tabular"].numpy() for i in background_indices])
    background = torch.tensor(
        background,
        dtype=torch.float32,
        device=device,
    )

    wrapped_model = TabularSHAPWrapper(
        model=model,
        waveform=waveform,
        target=target,
    ).to(device)

    explainer = shap.GradientExplainer(
        wrapped_model,
        background,
    )
    shap_values = explainer.shap_values(tabular)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    # [1, features] -> [features]
    shap_values = np.squeeze(shap_values)

    with torch.no_grad():
        background_output = wrapped_model(background)

    expected_value = float(background_output.mean().detach().cpu().item())
    return {
        "index": index,
        "target": target,
        "shap_values": shap_values,
        "tabular_values": tabular[0].detach().cpu().numpy(),
        "expected_value": expected_value,
    }


def compute_global_tabular_shap(
    model,
    dataset,
    target,
    device=None,
    background_indices=None,
    explain_indices=None,
):
    """Compute SHAP values for tabular features across multiple patients."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if background_indices is None:
        background_indices = range(min(100, len(dataset)))
    if explain_indices is None:
        explain_indices = range(len(dataset))

    background = np.stack([dataset[i]["tabular"].numpy() for i in background_indices])
    background = torch.tensor(
        background,
        dtype=torch.float32,
        device=device,
    )

    all_shap_values = []
    all_tabular_values = []
    valid_indices = []

    for index in explain_indices:
        sample = dataset[index]
        waveform = sample["waveform"].unsqueeze(0).to(device)
        tabular = sample["tabular"].unsqueeze(0).to(device)

        wrapped_model = TabularSHAPWrapper(
            model=model,
            waveform=waveform,
            target=target,
        ).to(device)
        explainer = shap.GradientExplainer(
            wrapped_model,
            background,
        )

        shap_values = explainer.shap_values(tabular)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        # [1, features] -> [features]
        shap_values = np.squeeze(shap_values)
        all_shap_values.append(shap_values)
        all_tabular_values.append(sample["tabular"].numpy())
        valid_indices.append(index)

    if not all_shap_values:
        raise RuntimeError(f"No samples were explained for target '{target}'.")

    return {
        "target": target,
        "shap_values": np.stack(all_shap_values),
        "tabular_values": np.stack(all_tabular_values),
        "indices": np.asarray(valid_indices),
    }
