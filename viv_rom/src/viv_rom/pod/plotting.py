"""Figures for the POD analysis.

Every function returns the created :class:`matplotlib.figure.Figure` and does
not save or close it; see :func:`viv_rom.pipeline.write_figures` for that.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from viv_rom.domain.motion import ForcedMotion
from viv_rom.pod.core import FieldInfo, PODResult
from viv_rom.pod.energy_injection import EnergyInjection
from viv_rom.pod.phase import HarmonicPhases
from viv_rom.pod.power import PowerCheck
from viv_rom.signal_utils import dominant_frequency


def plot_energy_spectrum(pod: PODResult, n_show: int = 20) -> Figure:
    """Energy fraction per mode and cumulative energy of the leading modes."""
    n_show = min(n_show, pod.n_modes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    modes = np.arange(1, n_show + 1)
    ax1.bar(modes, 100 * pod.energy_frac[:n_show], zorder=10)
    ax1.set_xlabel("POD mode")
    ax1.set_ylabel("Energy fraction [%]")
    ax1.set_title("Energy per mode")
    ax1.grid(alpha=.8, zorder=0)

    ax2.plot(modes, 100 * pod.cum_energy[:n_show], "o-")
    ax2.axhline(99, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Number of modes")
    ax2.set_ylabel("Cumulative energy [%]")
    ax2.set_title("Cumulative energy")
    ax2.grid(alpha=.8, zorder=0)

    fig.tight_layout()
    return fig


def plot_spatial_modes(y: np.ndarray, pod: PODResult, field_info: FieldInfo,
                       motion: ForcedMotion, n_show: int = 4,
                       normalization: Sequence[str] = ()) -> Figure:
    """Leading spatial POD modes with the forced-motion mode shape overlaid.

    For a combined Fx/Fz decomposition the modes are split into their
    components on separate axes. The modes are unit-norm in the space the POD
    was computed in; if a normalisation was used it is named in the figure
    title (the modes are *not* un-normalised).
    """
    n_show = min(n_show, pod.modes.shape[1])

    if field_info.kind == "single":
        fig, ax = plt.subplots(figsize=(7, 5))
        for k in range(n_show):
            ax.plot(y, pod.modes[:, k],
                    label=f"mode {k+1} ({100*pod.energy_frac[k]:.1f}%)")
        ax.set_ylabel(f"{field_info.component} mode shape [-]")
        ax.grid(alpha=.8, zorder=0)

        ax_twin = ax.twinx()
        ax_twin.plot(y, motion.modeshape(field_info.component, y),
                     c="k", ls="--", label="Forced motion modeshape")
        ax_twin.set_ylabel("Forced motion modeshape [-]")
        ax.set_title("Leading POD spatial modes")
    else:  # combined
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
        for k in range(n_show):
            phi_x, phi_z = field_info.split(pod.modes[:, k])
            lbl = f"mode {k+1} ({100*pod.energy_frac[k]:.1f}%)"
            ax.plot(y, phi_x, label=lbl)
            ax2.plot(y, phi_z, label=lbl)
        ax2.set_xlabel("Spanwise station y [m]")
        ax2.set_ylabel("Fz-part mode shape [-]")
        ax2.set_title("Flapwise (Fz) component")
        ax2.legend(fontsize=8)
        ax.set_ylabel("Fx-part mode shape [-]")
        ax.set_title("Edgewise (Fx) component")
        ax.grid(alpha=.8, zorder=0)
        ax2.grid(alpha=.8, zorder=0)

        for axis, comp in ((ax, "Fx"), (ax2, "Fz")):
            twin = axis.twinx()
            twin.plot(y, motion.modeshape(comp, y), c="k", ls="--",
                      label="Forced motion modeshape")
            twin.set_ylabel("Forced motion modeshape [-]")

    ax.set_xlabel("Spanwise station y [m]")
    ax.legend(fontsize=8)
    if normalization:
        fig.suptitle("normalization: " + ", ".join(normalization), fontsize=9)
    fig.tight_layout()
    return fig


def plot_temporal_coefficients(t: np.ndarray, pod: PODResult, dt: float,
                               f_forcing: float, n_show: int = 4,
                               n_harmonics: int = 6) -> Figure:
    """Temporal coefficients and their spectra for the leading modes.

    The prescribed forcing frequency (and harmonics) and the dominant
    spectral peak are highlighted.
    """
    n_show = min(n_show, pod.time_coeff.shape[0])
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.4 * n_show),
                             squeeze=False)

    for k in range(n_show):
        a_k = pod.time_coeff[k, :]
        f_dom, freqs, amp = dominant_frequency(a_k, dt)

        axes[k, 0].plot(t, a_k)
        axes[k, 0].set_ylabel(f"a_{k+1}(t)")
        axes[k, 0].set_title(f"Mode {k+1} time coefficient "
                             f"({100*pod.energy_frac[k]:.1f}% energy)")
        axes[k, 0].grid(alpha=.8, zorder=0)

        axes[k, 1].plot(freqs, amp)
        axes[k, 1].axvline(f_forcing, color="r", ls="--", lw=1,
                           label="Forcing frequency")

        for i in range(2, n_harmonics + 1):
            axes[k, 1].axvline(f_forcing * i, color="r", ls="--", lw=1,
                               label="")
        axes[k, 1].axvline(f_dom, color="g", ls=":", lw=1.5,
                           label="Dominant peak")
        axes[k, 1].set_xlim(0, max(n_harmonics * f_forcing, 2 * f_dom, 1e-6))
        axes[k, 1].set_title(f"Spectrum (peak = {f_dom:.3f} Hz, "
                             f"{f_dom / f_forcing:.2f} x forcing)")
        axes[k, 1].legend(fontsize=7)
        axes[k, 1].grid(alpha=.8, zorder=0)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    return fig


def plot_reconstruction(y: np.ndarray, t: np.ndarray, D_full: np.ndarray,
                        reconstruct: Callable[[int], np.ndarray],
                        field_info: FieldInfo,
                        station: int = 0,
                        r_values: Iterable[int] = (1, 2, 4)) -> Figure:
    """Compare truncated POD reconstructions with the original field.

    ``reconstruct(r)`` must return the rank-``r`` field in physical units
    (see :class:`~viv_rom.pod.preprocessing.PhysicalReconstructor`).

    Left: time history at spanwise index ``station``. Right: spanwise
    distribution at the final snapshot.
    """
    if field_info.kind == "single":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ax1.plot(t, D_full[station, :], "k", lw=1.5, label="original")
        ax2.plot(y, D_full[:, -1], "k", lw=1.5, label="original")

        for r in r_values:
            D_rec = reconstruct(r)
            ax1.plot(t, D_rec[station, :], lw=1, label=f"{r} mode(s)")
            ax2.plot(y, D_rec[:, -1], lw=1, label=f"{r} mode(s)")

        comp = field_info.component
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Force {comp} [N/m] at specified station")
        ax1.set_title(f"Reconstruction quality of {comp} (time series)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=.8, zorder=0)

        ax2.set_xlabel("Spanwise station y [m]")
        ax2.set_ylabel(f"Force {comp} [N/m] at last time step")
        ax2.set_title(f"Reconstruction quality of {comp} (spanwise, "
                      "last snapshot)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=.8, zorder=0)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fx_full, fz_full = field_info.split(D_full)

        axes[0, 0].plot(t, fx_full[station, :], "k", lw=1.5,
                        label="original")
        axes[1, 0].plot(t, fz_full[station, :], "k", lw=1.5,
                        label="original")
        axes[0, 1].plot(y, fx_full[:, -1], "k", lw=1.5, label="original")
        axes[1, 1].plot(y, fz_full[:, -1], "k", lw=1.5, label="original")

        for r in r_values:
            fx_rec, fz_rec = field_info.split(reconstruct(r))
            axes[0, 0].plot(t, fx_rec[station, :], lw=1, label=f"{r} mode(s)")
            axes[1, 0].plot(t, fz_rec[station, :], lw=1, label=f"{r} mode(s)")
            axes[0, 1].plot(y, fx_rec[:, -1], lw=1, label=f"{r} mode(s)")
            axes[1, 1].plot(y, fz_rec[:, -1], lw=1, label=f"{r} mode(s)")

        for i, comp in enumerate(["Fx", "Fz"]):
            axes[i, 0].set_xlabel("Time [s]")
            axes[i, 0].set_ylabel(f"Force {comp} [N/m] at specified station")
            axes[i, 0].set_title(f"Reconstruction quality of {comp} "
                                 "(time series)")
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].grid(alpha=.8, zorder=0)

            axes[i, 1].set_xlabel("Spanwise station y [m]")
            axes[i, 1].set_ylabel(f"Force {comp} [N/m] at last time step")
            axes[i, 1].set_title(f"Reconstruction quality of {comp} "
                                 "(spanwise, last snapshot)")
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].grid(alpha=.8, zorder=0)

    fig.tight_layout()
    return fig


def plot_force_orbit(t: np.ndarray, Fx: np.ndarray, Fz: np.ndarray,
                     reconstruct: Callable[[int], np.ndarray],
                     field_info: FieldInfo, station: int = 0,
                     r_values: Sequence[int] = (2, 4)) -> Figure:
    """Sectional-force orbit (Fx vs. Fz) at one station.

    Overlays the orbit of the original data (coloured by time) with orbits
    reconstructed from truncated POD bases. Requires a POD of the *combined*
    Fx/Fz field.

    Raises
    ------
    ValueError
        If ``field_info.kind`` is not ``"combined"``.
    """
    if field_info.kind != "combined":
        raise ValueError(
            "plot_force_orbit requires a POD computed on the combined "
            "Fx/Fz field (field_info.kind == 'combined')."
        )

    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(Fx[station, :], Fz[station, :], c=t, cmap="viridis",
                    s=8, label="original")
    fig.colorbar(sc, ax=ax, label="time [s]")

    for r in r_values:
        Fx_rec, Fz_rec = field_info.split(reconstruct(r))
        ax.plot(Fx_rec[station, :], Fz_rec[station, :], lw=1.2, alpha=.8,
                label=f"{r} mode(s)")

    ax.set_xlabel("Fx [N/m] (edgewise)")
    ax.set_ylabel("Fz [N/m] (flapwise)")
    ax.set_title(f"Sectional-force orbit at station index {station}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=.8, zorder=0)
    return fig


def plot_power_vs_mode1(t: np.ndarray, pod: PODResult, power: PowerCheck
                        ) -> Figure:
    """Instantaneous aerodynamic power against the first POD coefficient."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax2 = ax.twinx()
    ax.plot(t, power.power, "k", label="Instantaneous power P(t)")
    ax2.plot(t, pod.time_coeff[0, :], "C0", label="a_1(t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [W]")
    ax2.set_ylabel("a_1(t)", color="C0")
    ax.set_title("Aerodynamic power vs. leading POD temporal coefficient")
    ax.grid(alpha=.8, zorder=0)
    ax2.grid(alpha=.8, zorder=0)
    fig.tight_layout()
    return fig


def plot_sectional_energy(y: np.ndarray, power: PowerCheck) -> Figure:
    """Spanwise energy-injection distribution, original and reconstructed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y, power.sectional_energy, "k", label="Sectional energy injection")
    for r, energy in power.reconstructed_sectional_energy.items():
        ax.plot(y, energy, ls="--", alpha=.8,
                label=f"Reconstruction with {r} mode(s)")
    ax.set_xlabel("Spanwise station y [m]")
    ax.set_ylabel("Sectional energy injection [J/m]")
    ax.set_title("Sectional aerodynamic energy injection")
    ax.grid(alpha=.8, zorder=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_mode_energy_injection(energy: EnergyInjection, n_show: int = 4
                               ) -> Figure:
    """Spanwise energy injection of the leading modes, and per-mode totals.

    Left: ``E(y)`` of the raw field and of the leading modes individually
    (positive: exciting). Right: span-integrated ``E_k`` for the leading
    modes.
    """
    n_show = min(n_show, energy.n_modes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(energy.y, energy.original, "k", lw=1.8, label="raw field")
    ax1.plot(energy.y, energy.mean_part, "k:", lw=1.2, label="mean field")
    for k in range(n_show):
        ax1.plot(energy.y, energy.modes[k], lw=1.2, label=f"mode {k+1}")
    ax1.axhline(0, color="gray", lw=.8)
    ax1.set_xlabel("Spanwise station y [m]")
    ax1.set_ylabel("Energy injection E(y) [J/m]")
    ax1.set_title("Spanwise energy injection per mode")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.8, zorder=0)

    n_bar = min(max(n_show, 10), energy.n_modes)
    ax2.bar(np.arange(1, n_bar + 1), energy.mode_totals[:n_bar], zorder=10)
    ax2.axhline(0, color="gray", lw=.8)
    ax2.set_xlabel("POD mode")
    ax2.set_ylabel("Integrated energy injection [J]")
    ax2.set_title(f"Per-mode total (raw field: {energy.total_original:.3g} J)")
    ax2.grid(alpha=.8, zorder=0)

    fig.tight_layout()
    return fig


def plot_energy_reconstruction_error(energy: EnergyInjection,
                                     n_show: int = 20) -> Figure:
    """Error of the reconstructed ``E(y)`` against the number of modes.

    Left: aggregated error for each metric, with the tolerance and the
    number of modes needed. Right: station-wise error for the smallest
    ``r`` that meets the tolerance (or the last ``r`` shown).
    """
    n_show = min(n_show, energy.n_modes)
    r = np.arange(0, n_show + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for i, (name, curve) in enumerate(energy.curves.items()):
        line, = ax1.semilogy(r, np.maximum(curve[:n_show + 1], 1e-16), "o-",
                             ms=3, label=f"{name} error")
        n_req = energy.n_required[name]
        if n_req is not None and n_req <= n_show:
            ax1.plot(n_req, max(curve[n_req], 1e-16), "s", ms=9,
                     mfc="none", color=line.get_color())
    ax1.axhline(energy.tolerance, color="gray", ls="--", lw=1,
                label=f"tolerance ({100 * energy.tolerance:g} %)")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_xlabel("Number of modes r (0 = mean field only)")
    ax1.set_ylabel("Aggregated relative error of E(y)")
    ax1.set_title("Energy-injection reconstruction error")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.8, zorder=0)

    first = next((n for n in energy.n_required.values() if n is not None),
                 n_show)
    r_plot = int(min(first, energy.n_modes))
    ax2.plot(energy.y, energy.original, "k", lw=1.8, label="raw field")
    ax2.plot(energy.y, energy.reconstructed(r_plot), "--",
             label=f"mean + {r_plot} mode(s)")
    ax2.set_xlabel("Spanwise station y [m]")
    ax2.set_ylabel("Energy injection E(y) [J/m]")
    ax2.set_title(f"E(y) reconstructed with r = {r_plot}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.8, zorder=0)

    fig.tight_layout()
    return fig


def plot_harmonic_phases(phases: HarmonicPhases) -> Figure:
    """Phase of each mode's coefficient at the forcing harmonics.

    One polar panel per harmonic: angle = phase lead relative to the forced
    displacement (0 deg: in phase with displacement, +90 deg: with velocity),
    radius = amplitude relative to the largest mode at that harmonic.
    """
    n_h = len(phases.harmonics)
    fig, axes = plt.subplots(1, n_h, figsize=(3.8 * n_h, 4),
                             subplot_kw={"projection": "polar"},
                             squeeze=False)
    for i, ax in enumerate(axes[0]):
        amp = phases.amplitude[:, i]
        rel = amp / amp.max() if amp.max() > 0 else amp
        for k in range(phases.n_modes):
            if np.isnan(phases.phase[k, i]):
                continue
            weak = phases.explained[k] < 0.5  # not periodic at f_forcing
            ax.plot(phases.phase[k, i], rel[k], "o", ms=8,
                    color=f"C{k}", mfc="none" if weak else f"C{k}")
            ax.annotate(str(k + 1), (phases.phase[k, i], rel[k]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{phases.harmonics[i]} x f_forcing "
                     f"({phases.frequencies[i]:.3g} Hz)", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=f"C{k}",
                          label=f"mode {k+1}") for k in range(phases.n_modes)]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none",
                              color="gray", label="hollow: R$^2$ < 0.5"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False)
    fig.suptitle("Phase of temporal coefficients vs. forced displacement "
                 "(radius: relative amplitude)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
