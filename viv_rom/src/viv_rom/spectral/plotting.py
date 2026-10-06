"""Figures for the spectral analysis.

Every function returns the created :class:`matplotlib.figure.Figure` and does
not save or close it; see :func:`viv_rom.spectral.pipeline.write_figures`.

``components`` arguments are ``{"fx"/"fz": array (n_stations, n_steps)}``
dicts, as returned by :func:`viv_rom.spectral.core.select_components`.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from viv_rom.spectral.core import (COMPONENT_LABELS, compute_psd,
                                   compute_spanwise_amplitude,
                                   compute_spanwise_phase,
                                   compute_spanwise_psd, compute_spectrogram)


def _panels(n: int):
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.5), squeeze=False)
    return fig, axes[0]


def plot_station_psd(y: np.ndarray, t: np.ndarray,
                     components: Mapping[str, np.ndarray],
                     stations: Sequence[int], f_forcing: float,
                     n_harmonics: int, nperseg: int | None) -> Figure:
    """PSD (Welch) at selected stations, one panel per force component.

    Integer harmonics of the forcing frequency are drawn as vertical lines.
    """
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))

    for ax, (key, data) in zip(axes, components.items()):
        for st in stations:  # Plot PSD of each station
            freqs, psd = compute_psd(data[st, :], dt, nperseg)
            ax.semilogy(freqs, psd, label=f"y = {y[st]:.1f} m")
        for h in range(1, n_harmonics + 1):  # Add harmonics as dashed lines
            ax.axvline(h * f_forcing, color="gray", ls="--", lw=0.8,
                       label="forcing harmonics" if h == 1 else None)
        ax.set_xlim(0, (n_harmonics + 1) * f_forcing)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.set_title(COMPONENT_LABELS[key])
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_spanwise_map(y: np.ndarray, t: np.ndarray,
                      components: Mapping[str, np.ndarray], f_forcing: float,
                      n_harmonics: int, nperseg: int | None) -> Figure:
    """Frequency-vs-span heatmap of log10(PSD), one panel per component."""
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (key, data) in zip(axes, components.items()):
        freqs, psd_map = compute_spanwise_psd(data, dt, nperseg)
        keep = freqs <= f_max
        pcm = ax.pcolormesh(y, freqs[keep],
                            np.log10(psd_map[:, keep].T + 1e-30),
                            shading="auto", cmap="magma")
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(COMPONENT_LABELS[key])
        fig.colorbar(pcm, ax=ax, label="log10(PSD)")

    fig.tight_layout()
    return fig


def plot_spanwise_amplitude_map(y: np.ndarray, t: np.ndarray,
                                components: Mapping[str, np.ndarray],
                                f_forcing: float, n_harmonics: int,
                                nperseg: int | None) -> Figure:
    """Frequency-vs-span heatmap of peak amplitude (linear, e.g. N/m).

    The "force contribution per frequency" view. A linear colour scale is
    used deliberately: the colour axis reads directly in force units rather
    than a squared, per-Hz, log-compressed quantity.
    """
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (key, data) in zip(axes, components.items()):
        freqs, amp_map = compute_spanwise_amplitude(data, dt, nperseg)
        keep = freqs <= f_max
        pcm = ax.pcolormesh(y, freqs[keep], amp_map[:, keep].T,
                            shading="auto", cmap="magma", vmin=0.0)
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{COMPONENT_LABELS[key]} amplitude")
        fig.colorbar(pcm, ax=ax, label="peak amplitude [N/m]")

    fig.tight_layout()
    return fig


def plot_spanwise_phase_map(y: np.ndarray, t: np.ndarray,
                            components: Mapping[str, np.ndarray],
                            f_forcing: float, n_harmonics: int,
                            mag_threshold_db: float | None) -> Figure:
    """Frequency-vs-span heatmap of phase [rad], one panel per component.

    Phase is only meaningful where there is real spectral content, so bins
    whose FFT magnitude is more than ``mag_threshold_db`` below that panel's
    peak magnitude are masked out; ``None`` disables masking.
    """
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (key, data) in zip(axes, components.items()):
        freqs, phase_map, mag_map = compute_spanwise_phase(data, dt)
        keep = freqs <= f_max
        phase_plot = phase_map[:, keep].T
        mag_plot = mag_map[:, keep].T

        if mag_threshold_db is not None:
            threshold = mag_plot.max() * 10 ** (mag_threshold_db / 20.0)
            phase_plot = np.where(mag_plot >= threshold, phase_plot, np.nan)

        pcm = ax.pcolormesh(y, freqs[keep], phase_plot, shading="auto",
                            cmap="twilight", vmin=-np.pi, vmax=np.pi)
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="lime", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{COMPONENT_LABELS[key]} phase")
        cbar = fig.colorbar(pcm, ax=ax, label="phase [rad]")
        cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                             r"$\pi/2$", r"$\pi$"])

    fig.tight_layout()
    return fig


def plot_spectrogram(t: np.ndarray, signal: np.ndarray, f_forcing: float,
                     n_harmonics: int, title: str,
                     nperseg: int | None = None) -> Figure:
    """Time-frequency spectrogram of a single signal (e.g. total force)."""
    times, freqs, Sxx = compute_spectrogram(t, signal, nperseg)
    keep = freqs <= (n_harmonics + 1) * f_forcing

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcm = ax.pcolormesh(times, freqs[keep], np.log10(Sxx[keep, :] + 1e-30),
                        shading="auto", cmap="magma")
    for h in range(1, n_harmonics + 1):
        ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="log10(PSD)")
    fig.tight_layout()
    return fig


def plot_frequency_sweep(U_arr: np.ndarray, f_dom: np.ndarray,
                         f_forcing: float) -> Figure:
    """Dominant response frequency vs. wind speed (lock-in diagram)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(U_arr, f_dom, "o-", color="C0")
    ax.axhline(f_forcing, color="r", ls="--", lw=1, label="forcing frequency")
    ax.set_xlabel("wind speed U [m/s]")
    ax.set_ylabel("dominant response frequency [Hz]")
    ax.set_title("Frequency lock-in diagram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
