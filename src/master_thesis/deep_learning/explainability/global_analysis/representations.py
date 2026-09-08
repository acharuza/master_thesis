from __future__ import annotations
import numpy as np
import torch
from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def extract_rnn_representations(
    model,
    dataset,
    target,
    device=None,
    indices=None,
    threshold=0.4,
):
    """Extract disease-specific ECG representations from the RNN + temporal attention pathway."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if indices is None:
        indices = range(len(dataset))

    representations = []
    labels = []
    probabilities = []
    predictions = []
    valid_indices = []

    for index in indices:
        sample = dataset[index]
        waveform = sample["waveform"].unsqueeze(0).to(device)
        tabular = sample.get("tabular")

        if tabular is not None:
            tabular = tabular.unsqueeze(0).to(device)

        x = waveform
        if model.cnn is not None:
            x = model.cnn(x)

        sequence, _ = model.encoder(x)
        attention = model.attention[target]
        representation, _ = attention(sequence)
        representations.append(representation[0].detach().cpu().numpy())
        true_label = float(sample["labels"][0].item())
        labels.append(true_label)

        outputs = model(
            waveform,
            tabular,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        logit = outputs[target]
        probability = torch.sigmoid(logit)[0].item()
        prediction = int(probability >= threshold)
        probabilities.append(probability)
        predictions.append(prediction)
        valid_indices.append(index)

    if not representations:
        raise RuntimeError(
            f"No representations were extracted " f"for target '{target}'."
        )

    return {
        "target": target,
        "representations": np.stack(representations),
        "labels": np.asarray(labels),
        "probabilities": np.asarray(probabilities),
        "predictions": np.asarray(predictions),
        "indices": np.asarray(valid_indices),
        "threshold": threshold,
    }


def compute_pca_representation(
    result,
    n_components=2,
):
    """Project learned RNN representations into PCA space."""
    representations = result["representations"]
    pca = PCA(
        n_components=n_components,
    )
    embedding = pca.fit_transform(representations)
    return {
        **result,
        "embedding": embedding,
        "pca": pca,
        "explained_variance_ratio": (pca.explained_variance_ratio_),
    }
