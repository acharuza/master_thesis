import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Disease-specific temporal attention.

    Input:
        x: (batch, time, features)

    Output:
        context: (batch, features)
        weights: (batch, time, 1)
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):

        scores = self.score(x)

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * x, dim=1)

        return context, weights
