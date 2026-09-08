import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colormaps
from pathlib import Path
from typing import List, Optional, Literal, Union
import pywt


def visualize_echonext_ecg(
    waveforms_path: Union[str, Path],
    index: int = 0,
    fs: int = 250,
    seconds: int = 10,
    leads: Optional[List[int]] = None,
    layout: Literal["stacked", "overlay"] = "stacked",
    figsize=(14, 10),
    title_prefix: str = "EchoNext ECG",
) -> Figure:
    waveforms_path = Path(waveforms_path)
    if not waveforms_path.exists():
        raise FileNotFoundError(f"File not found: {waveforms_path}")

    x = np.load(waveforms_path, mmap_mode="r")

    if x.ndim != 4:
        raise ValueError(f"Expected 4D array (N, 1, 2500, 12), got shape {x.shape}")
    if x.shape[1] != 1 or x.shape[3] != 12:
        raise ValueError(f"Expected shape (N, 1, 2500, 12), got {x.shape}")

    n, _, n_samples, n_leads = x.shape

    if not (0 <= index < n):
        raise IndexError(f"index={index} is out of range for N={n}")
    if leads is None:
        leads = list(range(n_leads))

    ecg_data = x[index, 0, :, :].copy()
    t = np.arange(n_samples) / fs
    expected_samples = int(seconds * fs)
    if n_samples != expected_samples:
        print(
            f"Warning: got {n_samples} samples, expected {expected_samples} for {seconds}s at fs={fs}Hz"
        )

    lead_names = [
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

    if layout == "stacked":
        fig, axes = plt.subplots(len(leads), 1, figsize=figsize, sharex=True)
        if len(leads) == 1:
            axes = [axes]

        y_min = ecg_data[:, leads].min()
        y_max = ecg_data[:, leads].max()
        y_margin = (y_max - y_min) * 0.05

        for ax, lead_idx in zip(axes, leads):
            ax.plot(t, ecg_data[:, lead_idx], linewidth=1.0, color="steelblue")
            label = (
                lead_names[lead_idx]
                if lead_idx < len(lead_names)
                else f"Lead {lead_idx}"
            )
            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.grid(alpha=0.25)

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"{title_prefix} | sample={index}", fontsize=11)
        fig.tight_layout()
        return fig

    elif layout == "overlay":
        fig, ax = plt.subplots(figsize=figsize)
        colors = colormaps["tab20"](np.linspace(0, 1, len(leads)))
        for color, lead_idx in zip(colors, leads):
            label = (
                lead_names[lead_idx]
                if lead_idx < len(lead_names)
                else f"Lead {lead_idx}"
            )
            ax.plot(t, ecg_data[:, lead_idx], linewidth=1.0, label=label, color=color)
        ax.set_title(f"{title_prefix} | sample={index}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=9)
        fig.tight_layout()
        return fig

    else:
        raise ValueError("layout must be 'stacked' or 'overlay'")


def visualize_echonext_ecg_scalogram(
    waveforms_path: Union[str, Path],
    index: int = 0,
    fs: int = 250,
    leads: Optional[List[int]] = None,
    figsize=(14, 10),
    title_prefix: str = "EchoNext ECG Scalogram",
    wavelet: str = "morlet",
) -> Figure:
    waveforms_path = Path(waveforms_path)
    if not waveforms_path.exists():
        raise FileNotFoundError(f"File not found: {waveforms_path}")

    x = np.load(waveforms_path, mmap_mode="r")

    if x.ndim != 4:
        raise ValueError(f"Expected 4D array (N, 1, 2500, 12), got shape {x.shape}")
    if x.shape[1] != 1 or x.shape[3] != 12:
        raise ValueError(f"Expected shape (N, 1, 2500, 12), got {x.shape}")

    n, _, n_samples, n_leads = x.shape

    if not (0 <= index < n):
        raise IndexError(f"index={index} is out of range for N={n}")

    if leads is None:
        leads = list(range(n_leads))

    lead_names = [
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

    ecg_signal = x[index, 0, :, :].copy()

    scales = np.arange(1, 129)

    fig, axes = plt.subplots(len(leads), 1, figsize=figsize, sharex=True)
    if len(leads) == 1:
        axes = [axes]

    for ax, lead_idx in zip(axes, leads):
        coefficients, frequencies = pywt.cwt(
            ecg_signal[:, lead_idx], scales, "morl", method="fft"
        )

        power = np.abs(coefficients)
        power_norm = (power - power.min()) / (power.max() - power.min() + 1e-6)

        extent = [0, n_samples / fs, frequencies[-1], frequencies[0]]
        im = ax.imshow(
            power_norm,
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="bilinear",
        )

        ax.set_ylabel(f"{lead_names[lead_idx]}\n(Hz)", fontsize=9)
        ax.set_ylim(frequencies[-1], frequencies[0])

        cbar = plt.colorbar(im, ax=ax, label="Normalized Power")
        cbar.ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time [s]", fontsize=10)
    fig.suptitle(f"{title_prefix} | sample={index}", fontsize=11)
    fig.tight_layout()
    return fig
