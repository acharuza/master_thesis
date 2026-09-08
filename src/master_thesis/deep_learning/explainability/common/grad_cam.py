from __future__ import annotations
import numpy as np
import torch
from captum.attr import LayerGradCam
from master_thesis.deep_learning.explainability.common.wrappers import (
    ECGTargetWrapper,
)


def compute_cnn_grad_cam(
    model,
    waveform,
    target,
    tabular=None,
    device=None,
):
    """Compute 1D Grad-CAM attribution for the CNN component of the ECG model."""
    if device is None:
        device = next(model.parameters()).device
    if not hasattr(model, "cnn") or model.cnn is None:
        raise ValueError("CNN Grad-CAM requires model.cnn to be enabled.")
    if waveform.ndim != 3:
        raise ValueError(
            "Expected waveform shape [batch, time, leads], "
            f"got {tuple(waveform.shape)}."
        )
    if waveform.shape[0] != 1:
        raise ValueError(
            "compute_cnn_grad_cam currently expects " "a single ECG sample."
        )

    waveform = waveform.to(device)
    if tabular is not None:
        tabular = tabular.to(device)

    model.eval()
    wrapper = ECGTargetWrapper(
        model=model,
        target=target,
    ).to(device)

    original_training_states = {}

    for module in wrapper.modules():
        if isinstance(
            module,
            (torch.nn.GRU, torch.nn.LSTM),
        ):
            original_training_states[module] = module.training
            # because rnn cant be run backwards in eval mode
            module.train()

    cnn_layers = list(model.cnn.cnn.children())
    conv_layers = [layer for layer in cnn_layers if isinstance(layer, torch.nn.Conv1d)]
    if not conv_layers:
        raise ValueError("No Conv1d layer was found in model.cnn.")
    target_layer = conv_layers[-1]

    grad_cam = LayerGradCam(
        wrapper,
        target_layer,
    )
    attribution = grad_cam.attribute(
        inputs=(
            (
                waveform,
                tabular,
            )
            if tabular is not None
            else waveform
        ),
        relu_attributions=True,
    )

    for module, training_state in original_training_states.items():
        module.train(training_state)

    attribution = attribution.detach().cpu().numpy()
    attribution = np.squeeze(attribution)

    with torch.no_grad():
        logit = wrapper(
            waveform,
            tabular,
        )
        logit = float(logit[0].detach().cpu().item())
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

    model.eval()
    return {
        "target": target,
        "grad_cam": attribution,
        "logit": logit,
        "probability": probability,
        "target_layer": target_layer.__class__.__name__,
    }
