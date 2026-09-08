import torch.nn as nn


class BinaryHeads(nn.Module):
    """
    Independent binary classifier per disease.
    """

    def __init__(self, feature_dim, diseases):
        super().__init__()
        self.heads = nn.ModuleDict(
            {disease: nn.Linear(feature_dim, 1) for disease in diseases}
        )
