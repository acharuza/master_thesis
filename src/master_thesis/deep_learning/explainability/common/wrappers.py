import torch.nn as nn


class ECGTargetWrapper(nn.Module):
    """Wrap ECGModel so that explainability methods see a single scalar target logit rather than the model's output dictionary."""

    def __init__(self, model: nn.Module, target: str):
        super().__init__()
        self.model = model
        self.target = target

    def forward(self, waveform, tabular=None):
        outputs = self.model(waveform, tabular)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if self.target not in outputs:
            raise KeyError(
                f"Target '{self.target}' not found in model outputs. "
                f"Available targets: {list(outputs.keys())}"
            )

        # [B, 1] -> [B]
        return outputs[self.target].squeeze(-1)


class TabularSHAPWrapper(nn.Module):
    """Wrapper for explaining tabular features while keeping the ECG waveform fixed."""

    def __init__(
        self,
        model,
        waveform,
        target,
    ):
        super().__init__()
        self.model = model
        self.target = target
        self.waveform = waveform

    def forward(self, tabular):
        waveform = self.waveform.expand(
            tabular.shape[0],
            -1,
            -1,
        )
        outputs = self.model(
            waveform,
            tabular,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if self.target not in outputs:
            raise KeyError(
                f"Target '{self.target}' not found. "
                f"Available outputs: {list(outputs.keys())}"
            )
        # [batch, 1]
        return outputs[self.target]
