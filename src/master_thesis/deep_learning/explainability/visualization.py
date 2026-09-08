import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

TABULAR_FEATURE_NAMES = [
    "sex",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
]


def plot_local_explanation(
    result,
    waveform,
    figsize=(12, 16),
):
    """Plot one patient's local explanation."""
    waveform = np.asarray(waveform)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [T, 12]")

    n_time, n_leads = waveform.shape
    if n_leads != 12:
        raise ValueError(f"Expected 12 ECG leads, got {n_leads}")

    ecg_attr = np.asarray(result["ecg_attribution"])
    temporal_importance = np.asarray(result["temporal_importance"])
    attention = np.asarray(result["attention"])
    lead_importance = np.asarray(result["lead_importance"])
    tabular_attr = result.get("tabular_attribution")

    fig = plt.figure(figsize=figsize)
    left = 0.13
    right = 0.92
    width = right - left

    ax_ecg = fig.add_axes(
        [
            left,
            0.70,
            width,
            0.23,
        ]
    )
    ax_temporal = fig.add_axes(
        [
            left,
            0.54,
            width,
            0.10,
        ]
    )
    ax_attention = fig.add_axes(
        [
            left,
            0.40,
            width,
            0.10,
        ]
    )
    ax_leads = fig.add_axes(
        [
            left,
            0.23,
            width,
            0.10,
        ]
    )
    ax_tabular = fig.add_axes(
        [
            left,
            0.06,
            width,
            0.10,
        ]
    )
    cbar_ax = fig.add_axes(
        [
            0.935,
            0.70,
            0.015,
            0.23,
        ]
    )

    max_abs_attr = np.percentile(
        np.abs(ecg_attr),
        99,
    )
    if max_abs_attr <= 0:
        max_abs_attr = 1.0
    norm = Normalize(
        vmin=-max_abs_attr,
        vmax=max_abs_attr,
    )
    cmap = plt.get_cmap("coolwarm")
    x = np.arange(n_time)

    lead_scale = np.max(
        np.abs(waveform),
        axis=0,
    )
    lead_scale[lead_scale == 0] = 1.0
    normalized_waveform = waveform / lead_scale

    spacing = 4.0
    amplitude = 2.0

    for lead_idx in range(n_leads):
        lead_center = (n_leads - lead_idx - 1) * spacing
        y = normalized_waveform[:, lead_idx] * amplitude + lead_center
        points = np.array(
            [
                x,
                y,
            ]
        ).T.reshape(
            -1,
            1,
            2,
        )
        segments = np.concatenate(
            [
                points[:-1],
                points[1:],
            ],
            axis=1,
        )
        segment_attr = (ecg_attr[:-1, lead_idx] + ecg_attr[1:, lead_idx]) / 2.0
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=2.2,
        )
        lc.set_array(segment_attr)
        ax_ecg.add_collection(lc)
        ax_ecg.text(
            -0.02,
            lead_center,
            LEAD_NAMES[lead_idx],
            transform=ax_ecg.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    ax_ecg.set_xlim(
        0,
        n_time - 1,
    )
    ax_ecg.set_ylim(
        -amplitude - 0.5,
        (n_leads - 1) * spacing + amplitude + 0.5,
    )
    ax_ecg.set_xticks([])
    ax_ecg.set_yticks([])
    ax_ecg.set_ylabel("ECG leads")
    ax_ecg.set_title(
        "ECG with signed Integrated Gradients attribution",
        fontsize=11,
        pad=8,
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])
    fig.colorbar(
        sm,
        cax=cbar_ax,
        label="Signed attribution",
    )

    temporal = temporal_importance
    ax_temporal.plot(
        np.arange(len(temporal)),
        temporal,
        linewidth=1.8,
    )
    ax_temporal.set_xlim(
        0,
        len(temporal) - 1,
    )
    ax_temporal.set_ylabel("Importance")
    ax_temporal.set_title(
        "Temporal Integrated Gradients importance",
        fontsize=11,
        pad=6,
    )
    ax_temporal.grid(
        alpha=0.25,
    )

    attn = attention
    ax_attention.plot(
        np.arange(len(attn)),
        attn,
        linewidth=1.8,
    )
    ax_attention.set_xlim(
        0,
        len(attn) - 1,
    )
    ax_attention.set_ylabel("Attention")
    ax_attention.set_title(
        "Temporal attention",
        fontsize=11,
        pad=6,
    )
    ax_attention.grid(
        alpha=0.25,
    )

    ax_leads.bar(
        np.arange(12),
        lead_importance,
    )
    ax_leads.set_xticks(np.arange(12))
    ax_leads.set_xticklabels(LEAD_NAMES)
    ax_leads.set_ylabel("Importance")
    ax_leads.set_title(
        "Lead importance",
        fontsize=11,
        pad=6,
    )
    ax_leads.grid(
        axis="y",
        alpha=0.25,
    )

    if tabular_attr is not None:
        tabular_attr = np.asarray(tabular_attr)
        n_features = len(tabular_attr)
        names = (
            TABULAR_FEATURE_NAMES
            if len(TABULAR_FEATURE_NAMES) == n_features
            else [f"Feature {i}" for i in range(n_features)]
        )
        order = np.argsort(np.abs(tabular_attr))
        ax_tabular.barh(
            np.arange(n_features),
            tabular_attr[order],
        )
        ax_tabular.set_yticks(np.arange(n_features))
        ax_tabular.set_yticklabels([names[i] for i in order])
        ax_tabular.set_xlabel("Attribution")
        ax_tabular.set_title(
            "Tabular feature attribution",
            fontsize=11,
            pad=6,
        )
        ax_tabular.axvline(
            0,
            linewidth=0.8,
        )
        ax_tabular.grid(
            axis="x",
            alpha=0.25,
        )
    else:
        ax_tabular.text(
            0.5,
            0.5,
            "No tabular attribution",
            ha="center",
            va="center",
        )
        ax_tabular.set_axis_off()

    probability = result.get("probability")
    true_label = result.get("true_label")
    prediction = result.get("prediction")
    threshold = result.get("prediction_threshold")
    title = "Local ECG explanation"

    if probability is not None:
        title += f" | Probability: " f"{probability:.3f}"
    if true_label is not None:
        title += f" | True: " f"{int(true_label)}"
    if prediction is not None:
        title += f" | Predicted: " f"{int(prediction)}"
    if threshold is not None:
        title += f" | Threshold: " f"{threshold:.2f}"

    fig.suptitle(
        title,
        fontsize=14,
        y=0.975,
    )
    return fig


def plot_global_lead_importance(
    result,
    title=None,
    figsize=(10, 6),
):
    """Plot global ECG lead importance for one disease."""
    lead_names = np.asarray(result["lead_names"])
    median = np.asarray(result["median"])
    q25 = np.asarray(result["q25"])
    q75 = np.asarray(result["q75"])

    order = np.argsort(median)[::-1]
    leads = lead_names[order]
    values = median[order]
    lower = values - q25[order]
    upper = q75[order] - values

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(leads))
    ax.barh(
        y,
        values,
        xerr=np.vstack([lower, upper]),
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(leads)
    ax.invert_yaxis()
    ax.set_xlabel("Median normalized absolute IG attribution")

    if title is None:
        title = f"Global ECG Lead Importance - {result['target']}"

    ax.set_title(title)
    ax.grid(
        axis="x",
        alpha=0.2,
    )
    plt.tight_layout()
    return fig


def plot_cnn_grad_cam(
    result,
    waveform,
    figsize=(12, 10),
):
    """Visualize 1D CNN Grad-CAM for a single ECG."""

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform)

    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError("Expected a single ECG sample.")
        waveform = waveform[0]

    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform shape [time, leads], " f"got {waveform.shape}."
        )

    n_time, n_leads = waveform.shape
    if n_leads != 12:
        raise ValueError(f"Expected 12 ECG leads, got {n_leads}.")

    grad_cam = np.asarray(
        result["grad_cam"],
        dtype=np.float32,
    )
    grad_cam = np.squeeze(grad_cam)

    if grad_cam.ndim != 1:
        raise ValueError(
            "Expected Grad-CAM to be one-dimensional, " f"got {grad_cam.shape}."
        )

    if len(grad_cam) != n_time:
        raise ValueError(
            "Grad-CAM and waveform have different temporal "
            f"lengths: {len(grad_cam)} vs {n_time}."
        )

    grad_cam = np.maximum(
        grad_cam,
        0,
    )
    max_value = np.max(grad_cam)
    if max_value > 0:
        grad_cam_norm = grad_cam / max_value
    else:
        grad_cam_norm = np.zeros_like(grad_cam)

    normalized_waveform = np.zeros_like(
        waveform,
        dtype=np.float32,
    )
    for lead_idx in range(n_leads):
        signal = waveform[:, lead_idx]
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        if signal_max > signal_min:
            normalized_waveform[:, lead_idx] = (
                2.0 * (signal - signal_min) / (signal_max - signal_min) - 1.0
            )

    fig = plt.figure(figsize=figsize)
    left = 0.13
    width = 0.77
    ax_ecg = fig.add_axes([left, 0.37, width, 0.55])
    ax_temporal = fig.add_axes([left, 0.10, width, 0.18])

    spacing = 2.8
    amplitude = 0.9
    time = np.arange(n_time)

    neutral_color = "black"
    grad_cam_cmap = plt.cm.Reds

    for lead_idx in range(n_leads):
        lead_center = (n_leads - lead_idx - 1) * spacing
        y = normalized_waveform[:, lead_idx] * amplitude + lead_center
        ax_ecg.plot(
            time,
            y,
            color=neutral_color,
            linewidth=0.8,
            alpha=0.85,
        )

        for t in range(n_time - 1):
            strength = grad_cam_norm[t]
            if strength <= 0:
                continue
            ax_ecg.plot(
                time[t : t + 2],
                y[t : t + 2],
                color=grad_cam_cmap(0.25 + 0.70 * strength),
                linewidth=1.8,
            )

        ax_ecg.text(
            -0.02,
            lead_center,
            LEAD_NAMES[lead_idx],
            transform=ax_ecg.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

    ax_ecg.set_xlim(
        0,
        n_time - 1,
    )
    ax_ecg.set_ylim(
        -amplitude - 0.5,
        (n_leads - 1) * spacing + amplitude + 0.5,
    )
    ax_ecg.set_yticks([])
    ax_ecg.set_xlabel("Time point")
    ax_ecg.set_title("ECG with CNN Grad-CAM")
    ax_ecg.grid(
        axis="x",
        alpha=0.15,
    )
    ax_temporal.plot(
        time,
        grad_cam_norm,
        linewidth=1.2,
    )
    ax_temporal.fill_between(
        time,
        0,
        grad_cam_norm,
        alpha=0.25,
    )
    ax_temporal.set_xlim(
        0,
        n_time - 1,
    )
    ax_temporal.set_ylim(
        0,
        1.05,
    )
    ax_temporal.set_ylabel("Grad-CAM")
    ax_temporal.set_xlabel("Time point")
    ax_temporal.set_title("CNN temporal activation")
    ax_temporal.grid(
        axis="y",
        alpha=0.2,
    )

    probability = result.get("probability")
    if probability is not None:
        fig.suptitle(
            f"CNN Grad-CAM Explanation - "
            f"{result['target']} - "
            f"Probability: {probability:.3f}",
            fontsize=14,
            fontweight="bold",
            y=0.97,
        )
    return fig


def plot_rnn_representation(
    result,
    figsize=(8, 6),
):
    embedding = result["embedding"]
    labels = result["labels"]
    predictions = result["predictions"]

    fig, ax = plt.subplots(figsize=figsize)

    negative = labels == 0
    positive = labels == 1

    incorrect = labels != predictions
    correct = ~incorrect

    mask = negative & correct
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        alpha=0.5,
        label="Negative",
    )

    mask = positive & correct
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        alpha=0.5,
        label="Positive",
    )

    ax.scatter(
        embedding[incorrect, 0],
        embedding[incorrect, 1],
        facecolors="none",
        edgecolors="grey",
        linewidths=1.5,
        s=70,
        alpha=0.7,
        label="Incorrect prediction",
    )

    variance = result["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    ax.set_title(f"RNN Representation - {result['target']}")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig
