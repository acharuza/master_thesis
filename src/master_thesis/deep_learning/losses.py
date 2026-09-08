import torch
import torch.nn as nn


class MultiTaskBCELoss(nn.Module):
    """Multi-task binary cross-entropy loss for multi-label classification."""

    def __init__(
        self,
        diseases,
        pretrain=False,
        pos_weights=None,
    ):
        super().__init__()

        self.diseases = list(diseases)
        self.pretrain = pretrain
        self.pos_weights = {}

        if pos_weights is not None:

            if len(pos_weights) != len(self.diseases):
                raise ValueError(
                    "Length of pos_weights must match number " "of diseases."
                )

            for i, disease in enumerate(self.diseases):
                self.pos_weights[disease] = pos_weights[i]

    def forward(self, outputs, targets):
        disease_losses = []

        for disease in self.diseases:
            logits = outputs[disease]
            if disease in self.pos_weights:
                loss_fn = nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weights[disease].to(logits.device)
                )
            else:
                loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(
                logits.squeeze(1),
                targets[disease],
            )
            disease_losses.append(loss)

        if len(disease_losses) == 0:
            raise ValueError("No disease losses were computed.")

        disease_loss = torch.stack(disease_losses).mean()

        return disease_loss
