import torch


@torch.no_grad()
def get_attention(model, waveform, tabular=None, target=None):
    """extract temporal attention from ECGModel"""

    model.eval()
    outputs, attentions = model(
        waveform,
        tabular,
    )

    if target is None:
        if len(attentions) != 1:
            raise ValueError(
                "target must be specified when the model has "
                "multiple attention targets."
            )
        target = next(iter(attentions))
    if target not in attentions:
        raise KeyError(
            f"Target '{target}' not found. "
            f"Available targets: {list(attentions.keys())}"
        )

    # [B, T, 1] -> [T]
    attention = attentions[target][0, :, 0]

    return attention
