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

from viv_rom.spectral.cycle_energy import CycleEnergy
from viv_rom.spectral.forcing_response import ForcingResponse
from viv_rom.spectral.shedding import SheddingFrequencies
from viv_rom.spectral.core import (COMPONENT_LABELS, compute_psd,
                                   compute_spanwise_amplitude,
                                   compute_spanwise_phase,
                                   compute_spanwise_psd, compute_spectrogram)


def _label(labels: Mapping[str, str] | None, key: str) -> str:
    return (labels or COMPONENT_LABELS).get(key, key)


def _panels(n: int):
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.5), squeeze=False)
    return fig, axes[0]


def plot_station_psd(y: np.ndarray, t: np.ndarray,
                     components: Mapping[str, np.ndarray],
                     stations: Sequence[int], f_forcing: float,
                     n_harmonics: int, nperseg: int | None,
                     noverlap: int | None,
                     labels: Mapping[str, str] | None = None) -> Figure:
    """PSD (Welch) at selected stations, one panel per force component.

    Integer harmonics of the forcing frequency are drawn as vertical lines.
    """
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))

    for ax, (key, data) in zip(axes, components.items()):
        for st in stations:  # Plot PSD of each station
            freqs, psd = compute_psd(data[st, :], dt=dt,
                                     nperseg=nperseg, noverlap=noverlap)
            ax.semilogy(freqs, psd, label=f"y = {y[st]:.1f} m")
        for h in range(1, n_harmonics + 1):  # Add harmonics as dashed lines
            ax.axvline(h * f_forcing, color="gray", ls="--", lw=0.8,
                       label="forcing harmonics" if h == 1 else None)
        ax.set_xlim(0, (n_harmonics + 1) * f_forcing)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.set_title(_label(labels, key))
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_spanwise_map(y: np.ndarray, t: np.ndarray,
                      components: Mapping[str, np.ndarray], f_forcing: float,
                      n_harmonics: int, nperseg: int | None,
                      noverlap: int | None,
                      labels: Mapping[str, str] | None = None) -> Figure:
    """Frequency-vs-span heatmap of log10(PSD), one panel per component."""
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (key, data) in zip(axes, components.items()):
        freqs, psd_map = compute_spanwise_psd(data, dt=dt, nperseg=nperseg,
                                              noverlap=noverlap)

        # Filter for selected frequency range
        keep = freqs <= f_max
        psd_map = psd_map[:, keep]

        # Filter out near zero values
        psd_map = np.where(psd_map >= 1e-30, psd_map, np.nan)

        pcm = ax.pcolormesh(y, freqs[keep],
                            np.log10(psd_map.T),
                            shading="auto", cmap="magma")
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(_label(labels, key))
        fig.colorbar(pcm, ax=ax, label="log10(PSD)")

    fig.tight_layout()
    return fig


def plot_spanwise_amplitude_map(y: np.ndarray, t: np.ndarray,
                                components: Mapping[str, np.ndarray],
                                f_forcing: float, n_harmonics: int,
                                nperseg: int | None, noverlap: int | None,
                                labels: Mapping[str, str] | None = None
                                ) -> Figure:
    """Frequency-vs-span heatmap of peak amplitude (linear, e.g. N/m).

    The "force contribution per frequency" view. A linear colour scale is
    used deliberately: the colour axis reads directly in force units rather
    than a squared, per-Hz, log-compressed quantity.
    """
    dt = t[1] - t[0]
    fig, axes = _panels(len(components))
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (key, data) in zip(axes, components.items()):
        freqs, amp_map = compute_spanwise_amplitude(data, dt=dt,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap)
        keep = freqs <= f_max
        pcm = ax.pcolormesh(y, freqs[keep], amp_map[:, keep].T,
                            shading="auto", cmap="magma", vmin=0.0)
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{_label(labels, key)} amplitude")
        fig.colorbar(pcm, ax=ax, label="peak amplitude [N/m]")

    fig.tight_layout()
    return fig


def plot_spanwise_phase_map(y: np.ndarray, t: np.ndarray,
                            components: Mapping[str, np.ndarray],
                            f_forcing: float, n_harmonics: int,
                            mag_threshold_db: float | None,
                            labels: Mapping[str, str] | None = None
                            ) -> Figure:
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
        ax.set_title(f"{_label(labels, key)} phase")
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


def plot_forcing_response(response: ForcingResponse, harmonic: int = 1,
                          labels: Mapping[str, str] | None = None) -> Figure:
    """Spanwise amplitude and phase of the force at ``harmonic * f_forcing``.

    Phase is the lead relative to the forced displacement (0: in phase with
    displacement, +90 deg: with velocity). For the along-motion component
    ``0 < phase < 180 deg`` means the harmonic injects energy.
    """
    h = int(np.flatnonzero(response.harmonics == harmonic)[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for key in response.amplitude:
        lbl = _label(labels, key)
        ax1.plot(response.y, response.amplitude[key][:, h], label=lbl)
        ax2.plot(response.y, response.phase_deg(key)[:, h], ".-", ms=3,
                 label=lbl)
    f_h = response.frequencies[h]
    ax1.set_xlabel("Spanwise station y [m]")
    ax1.set_ylabel("Amplitude [N/m]")
    ax1.set_title(f"Force amplitude at {harmonic} x f_forcing ({f_h:.3g} Hz)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.8, zorder=0)

    y_max = np.nanmax([np.degrees(phase[:, h])
                       for phase in response.phase.values()])
    y_min = np.nanmin([np.degrees(phase[:, h])
                       for phase in response.phase.values()])

    if y_max - y_min <= 360:
        for level in (-180, -90, 0, 90, 180):
            ax2.axhline(level, color="gray", lw=.8,
                        ls="-" if level == 0 else ":")
        ax2.set_yticks([-180, -90, 0, 90, 180])
        ax2.set_ylim(-190, 190)
    else:
        y_min_360 = np.ceil(abs(y_min) / 360) * 360 * int(y_min/abs(y_min))
        y_max_360 = np.ceil(abs(y_max) / 360) * 360 * int(y_max/abs(y_max))

        y_min_90 = np.ceil(abs(y_min) / 90) * 90 * int(y_min/abs(y_min))
        y_max_90 = np.ceil(abs(y_max) / 90) * 90 * int(y_max/abs(y_max))

        ax2.set_yticks(np.arange(y_min_360, y_max_360, 360))
        ax2.set_yticks(np.arange(y_min_90, y_max_90, 90), minor=True)
        ax2.set_ylim(y_min_90, y_max_90)

    ax2.set_xlabel("Spanwise station y [m]")
    ax2.set_ylabel("Phase lead vs. displacement [deg]")
    ax2.set_title("Phase (+90 deg = in phase with velocity)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.8, zorder=0, which="both")
    fig.tight_layout()
    return fig


def plot_cycle_energy(cycle: CycleEnergy, stations: Sequence[int]) -> Figure:
    """Energy injected per forcing cycle: history and spanwise spread.

    Left: whole-blade and selected-station energy per cycle, each relative to
    its own mean (stations with negligible injection are skipped).
    Right: spanwise mean energy per cycle with its +-1 std band over cycles.
    """
    stats = cycle.station_stats
    tot = cycle.total_stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cycles = np.arange(1, cycle.n_cycles + 1)
    significant = np.abs(stats.mean) >= 1e-2 * np.abs(stats.mean).max()
    for k in stations:
        if significant[k]:  # skip stations with practically no injection
            ax1.plot(cycles, cycle.energy[:, k] / stats.mean[k], ".-", lw=1,
                     alpha=.8, label=f"y = {cycle.y[k]:.1f} m")
    if tot.mean != 0:
        ax1.plot(cycles, cycle.total / tot.mean, "k-o", ms=4, lw=2,
                 label="blade total")
    ax1.axhline(1.0, color="gray", lw=.8, ls=":")
    ax1.set_xlabel("Forcing cycle")
    ax1.set_ylabel("Energy per cycle / its mean")
    ax1.set_title(f"Per-cycle energy injection "
                  f"(blade total CV = {100 * tot.cv:.2f} %)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=.8, zorder=0)

    ax2.plot(cycle.y, stats.mean, "k", lw=1.6, label="mean over cycles")
    ax2.fill_between(cycle.y, stats.mean - stats.std, stats.mean + stats.std,
                     color="C0", alpha=.3, label="+-1 std over cycles")
    ax2.axhline(0, color="gray", lw=.8)
    ax2.set_xlabel("Spanwise station y [m]")
    ax2.set_ylabel("Energy per cycle [J/m]")
    ax2.set_title("Spanwise energy injection per cycle")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.8, zorder=0)
    fig.tight_layout()
    return fig


def plot_shedding_frequency(shedding: SheddingFrequencies,
                            f_max_factor: float = 3.0) -> Figure:
    """Shedding-frequency band along the span against the lock-in range.

    The blue band spans the shedding frequency for the lower to the upper
    Strouhal number. The dashed line is the forcing (= natural) frequency,
    the orange band the lock-in range around it, and the shaded columns
    mark stations whose shedding band overlaps the lock-in range. The
    shedding frequency grows without bound where the projected width goes
    to zero (tip), so the frequency axis ends at ``f_max_factor`` times the
    forcing frequency; the right axis gives ``f / f_forcing``.
    """
    s = shedding
    f_n = s.f_forcing
    lock_lo, lock_hi = s.lock_in_frequencies
    f_top = f_max_factor * f_n

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(lock_lo, lock_hi, color="C1", alpha=0.25, lw=0,
               label=f"lock-in range ({s.lock_in_band[0]:g}-"
                     f"{s.lock_in_band[1]:g} $f_n$)")
    ax.fill_between(s.y, 0, f_top, where=s.possible_lock_in,
                    step="mid", color="0.5", alpha=0.15, lw=0,
                    label="shedding band overlaps lock-in range")
    ax.fill_between(s.y, s.f_low, s.f_high, color="C0", alpha=0.35, lw=0,
                    label=f"shedding frequency, St = {s.strouhal[0]:g}-"
                          f"{s.strouhal[1]:g}")
    ax.plot(s.y, s.f_low, color="C0", lw=1.2,
            label=f"St = {s.strouhal[0]:g}")
    ax.plot(s.y, s.f_high, color="C0", lw=1.2, ls="--",
            label=f"St = {s.strouhal[1]:g}")
    ax.axhline(f_n, color="k", ls="--", lw=1.2,
               label=f"natural / forcing frequency ({f_n:.3g} Hz)")

    ax.set_xlim(s.y[0], s.y[-1])
    ax.set_ylim(0, f_top)
    ax.set_xlabel("spanwise station y [m]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(f"Estimated vortex-shedding frequency, U = "
                 f"{s.wind_speed:g} m/s, D = c|sin $\\alpha$|")
    ax.secondary_yaxis("right", functions=(lambda f: f / f_n,
                                           lambda r: r * f_n)
                       ).set_ylabel("$f / f_n$")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig
