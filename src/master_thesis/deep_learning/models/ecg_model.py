import torch
import torch.nn as nn

from .attention import TemporalAttention
from .heads import BinaryHeads


class CNNFrontEnd(nn.Module):

    def __init__(
        self,
        input_channels=12,
        channels=(64, 128),
        kernels=(7, 5),
        dropout=0.0,
    ):
        super().__init__()

        if len(channels) != len(kernels):
            raise ValueError("channels and kernels must have the same length")

        layers = []
        in_channels = input_channels

        for out_channels, kernel_size in zip(channels, kernels):
            padding = kernel_size // 2

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )

            if dropout > 0:
                layers.append(nn.Dropout1d(dropout))

            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class ECGModel(nn.Module):
    """ECG classification model."""

    def __init__(self, diseases, config):
        super().__init__()

        model_cfg = config["model"]

        self.diseases = list(diseases)
        self.use_cnn = model_cfg["use_cnn"]
        tabular_cfg = config.get("tabular", {})
        self.use_tabular = tabular_cfg.get("enabled", False)

        rnn_type = model_cfg["rnn_type"].lower()
        hidden_dim = model_cfg["hidden_dim"]
        num_layers = model_cfg["num_layers"]

        head_dropout = model_cfg.get("head_dropout", 0.2)

        if self.use_cnn:
            cnn_cfg = model_cfg.get("cnn", {})
            self.cnn = CNNFrontEnd(
                input_channels=12,
                channels=cnn_cfg.get(
                    "channels",
                    [64, 128],
                ),
                kernels=cnn_cfg.get(
                    "kernels",
                    [7, 5],
                ),
                dropout=cnn_cfg.get(
                    "dropout",
                    0.0,
                ),
            )
            rnn_input_dim = self.cnn.output_dim
        else:
            self.cnn = None
            rnn_input_dim = 12

        rnn_dropout = model_cfg.get("dropout", 0.0) if num_layers > 1 else 0.0

        if rnn_type == "gru":
            self.encoder = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout,
            )
        elif rnn_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: {rnn_type}. " "Expected 'gru' or 'lstm'."
            )

        # bidirectional RNN doubles the feature dimension
        feature_dim = hidden_dim * 2
        self.feature_dim = feature_dim

        self.attention = nn.ModuleDict(
            {disease: TemporalAttention(feature_dim) for disease in self.diseases}
        )
        self.head_dropout = nn.Dropout(head_dropout)

        if self.use_tabular:
            tabular_dim = len(tabular_cfg["features"])
            tabular_hidden_dim = tabular_cfg.get(
                "hidden_dim",
                64,
            )
            tabular_dropout = tabular_cfg.get(
                "dropout",
                0.2,
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(
                    tabular_dim,
                    tabular_hidden_dim,
                ),
                nn.LayerNorm(tabular_hidden_dim),
                nn.ReLU(),
                nn.Dropout(tabular_dropout),
            )
            self.tabular_feature_dim = tabular_hidden_dim
            fusion_dim = tabular_cfg.get(
                "fusion_dim",
                feature_dim,
            )
            self.fusion_dim = fusion_dim
            self.fusion = nn.ModuleDict(
                {
                    disease: nn.Sequential(
                        nn.Linear(
                            feature_dim + tabular_hidden_dim,
                            fusion_dim,
                        ),
                        nn.ReLU(),
                        nn.Dropout(head_dropout),
                    )
                    for disease in self.diseases
                }
            )
        else:
            self.tabular_encoder = None
            self.tabular_feature_dim = 0
            self.fusion_dim = feature_dim
            self.fusion = None

        self.heads = BinaryHeads(
            feature_dim=self.fusion_dim,
            diseases=self.diseases,
        )

    def forward(self, x, tabular=None):
        if self.cnn is not None:
            x = self.cnn(x)
        sequence, _ = self.encoder(x)

        # [batch, time, hidden_dim * 2]

        if self.use_tabular:
            if tabular is None:
                raise ValueError(
                    "tabular input is required when " "tabular.enabled=True"
                )
            tabular_representation = self.tabular_encoder(tabular)

            # [batch, tabular_hidden_dim]

        outputs = {}
        attentions = {}

        for disease in self.diseases:
            attention = self.attention[disease]
            representation, weights = attention(sequence)
            # [batch, feature_dim]
            representation = self.head_dropout(representation)

            if self.use_tabular:
                fused = torch.cat(
                    [
                        representation,
                        tabular_representation,
                    ],
                    dim=-1,
                )
                # [batch, feature_dim + tabular_hidden_dim]
                representation = self.fusion[disease](fused)
                # [batch, fusion_dim]

            logits = self.heads.heads[disease](representation)
            outputs[disease] = logits
            attentions[disease] = weights
        return outputs, attentions
