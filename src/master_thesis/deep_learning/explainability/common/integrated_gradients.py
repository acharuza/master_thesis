import torch
from captum.attr import IntegratedGradients


def compute_integrated_gradients(
    model,
    waveform,
    tabular=None,
    waveform_baseline=None,
    tabular_baseline=None,
    n_steps=50,
):
    was_training = model.training
    model.eval()

    # RNN backward requires the RNN to be in training mode
    for module in model.modules():
        if isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
            module.train()

    if waveform_baseline is None:
        waveform_baseline = torch.zeros_like(waveform)

    ig = IntegratedGradients(model)

    if tabular is not None:
        if tabular_baseline is None:
            tabular_baseline = torch.zeros_like(tabular)

        attributions = ig.attribute(
            inputs=(waveform, tabular),
            baselines=(waveform_baseline, tabular_baseline),
            n_steps=n_steps,
        )
        waveform_attribution = attributions[0]
        tabular_attribution = attributions[1]

    else:
        waveform_attribution = ig.attribute(
            waveform,
            baselines=waveform_baseline,
            n_steps=n_steps,
        )
        tabular_attribution = None

    model.train(was_training)
    return waveform_attribution, tabular_attribution
