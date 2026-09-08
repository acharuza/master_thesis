from __future__ import annotations
import torch
from master_thesis.deep_learning.explainability.common.attention import (
    get_attention,
)
from master_thesis.deep_learning.explainability.common.integrated_gradients import (
    compute_integrated_gradients,
)
from master_thesis.deep_learning.explainability.common.wrappers import (
    ECGTargetWrapper,
)


def explain_patient(
    model,
    dataset,
    index,
    target,
    device=None,
    n_steps=50,
    threshold=0.5,
):
    """Generate a local explanation for one patient."""
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    if target not in dataset.label_columns:
        raise ValueError(
            f"Target '{target}' not found in dataset labels. "
            f"Available labels: {dataset.label_columns}"
        )

    sample = dataset[index]
    waveform = sample["waveform"].unsqueeze(0).to(device)
    tabular = sample.get("tabular")

    if tabular is not None:
        tabular = tabular.unsqueeze(0).to(device)

    target_model = ECGTargetWrapper(
        model=model,
        target=target,
    ).to(device)

    target_model.eval()
    with torch.no_grad():
        logit = target_model(
            waveform,
            tabular,
        )
        probability = torch.sigmoid(logit)

    logit_value = float(logit[0].cpu())
    probability_value = float(probability[0].cpu())
    prediction = int(probability_value >= threshold)

    waveform_attr, tabular_attr = compute_integrated_gradients(
        model=target_model,
        waveform=waveform,
        tabular=tabular,
        n_steps=n_steps,
    )

    # [1, 2500, 12] -> [2500, 12]
    waveform_attr = waveform_attr[0]

    lead_importance = waveform_attr.abs().sum(dim=0)
    lead_importance = lead_importance / (lead_importance.sum() + 1e-8)

    temporal_importance = waveform_attr.abs().sum(dim=1)
    temporal_importance = temporal_importance / (temporal_importance.sum() + 1e-8)

    attention = get_attention(
        model=model,
        waveform=waveform,
        tabular=tabular,
        target=target,
    )

    target_position = dataset.label_columns.index(target)
    true_label = float(dataset.labels[index][target_position])

    if tabular_attr is not None:
        tabular_attr = tabular_attr[0]
        tabular_attr_np = tabular_attr.detach().cpu().numpy()
    else:
        tabular_attr_np = None

    return {
        "index": index,
        "target": target,
        "true_label": true_label,
        "logit": logit_value,
        "probability": probability_value,
        "prediction": prediction,
        "prediction_threshold": threshold,
        # [2500, 12]
        "ecg_attribution": (waveform_attr.detach().cpu().numpy()),
        # [12]
        "lead_importance": (lead_importance.detach().cpu().numpy()),
        # [2500]
        "temporal_importance": (temporal_importance.detach().cpu().numpy()),
        # [2500]
        "attention": (attention.detach().cpu().numpy()),
        # [F] or None
        "tabular_attribution": tabular_attr_np,
    }
