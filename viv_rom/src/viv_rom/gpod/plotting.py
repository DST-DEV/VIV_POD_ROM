"""Figures for the global POD.

Every function returns the created :class:`matplotlib.figure.Figure` and does
not save or close it (see :func:`viv_rom.gpod.pipeline.write_figures`). The
global modes and spectra reuse the single-case plots of
:mod:`viv_rom.pod.plotting`.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from viv_rom.gpod.analysis import (GlobalPODResult, LeaveOneOutResult,
                                   SubspaceAngles)
from viv_rom.signal_utils import dominant_frequency


def _value_colors(values: np.ndarray):
    """Sequential colours for the sweep values, and a matching mappable."""
    norm = colors.Normalize(vmin=float(np.min(values)),
                            vmax=float(np.max(values)))
    cmap = plt.get_cmap("inferno")
    return [cmap(norm(v)) for v in values], cm.ScalarMappable(norm=norm,
                                                              cmap=cmap)


def _label(parameter: str) -> str:
    return {"wind_speed": "Wind speed [m/s]", "pitch": "Pitch angle [deg]",
            "inclination": "Inclination angle [deg]"}.get(parameter, parameter)


def plot_error_curves(result: GlobalPODResult,
                      loo: LeaveOneOutResult | None = None,
                      n_show: int = 20) -> Figure:
    """Relative fluctuation error of every simulation versus modes kept.

    Left: simulations projected onto the basis they helped build. Right (if
    ``loo`` is given): each simulation projected onto the basis fitted
    *without* it -- the honest measure for an unseen simulation.
    """
    panels = [("In-sample (basis fitted to all simulations)",
               result.relative_error, result.values)]
    if loo is not None:
        panels.append(("Leave-one-out (basis without the simulation)",
                       loo.relative_error, loo.values))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.4),
                             sharey=True, squeeze=False)
    cols, mappable = _value_colors(result.values)
    for ax, (title, err, values) in zip(axes[0], panels):
        n = min(n_show, err.shape[1])
        r = np.arange(1, n + 1)
        for c, e in zip(cols, err):
            ax.semilogy(r, np.maximum(e[:n], 1e-16), color=c, lw=1.2)
        ax.set_xlabel("Number of global modes r")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.8, zorder=0)
        ax.set_xticks(r[::max(1, n // 10)])
    axes[0, 0].set_ylabel(r"Relative error $\|D' - D'_r\| / \|D'\|$")
    fig.colorbar(mappable, ax=axes.ravel().tolist(),
                 label=_label(result.parameter))
    return fig


def plot_modes_required(result: GlobalPODResult,
                        loo: LeaveOneOutResult | None = None) -> Figure:
    """Modes needed per simulation: global basis versus its own local POD.

    Left: modes for 99 % of the fluctuation energy. Right: modes for the
    configured energy-injection tolerance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    v = result.values

    n99 = result.n99_cases
    ax1.plot(v, np.where(n99 < 0, np.nan, n99), "o-", label="global basis")
    if result.has_local:
        ax1.plot(v, result.local_n99, "s--", label="local POD")
    ax1.set_title("Modes for 99 % of the fluctuation energy", fontsize=10)

    ne = result.energy_modes_required()
    if np.any(ne >= 0) or result.has_local:
        ax2.plot(v, np.where(ne < 0, np.nan, ne), "o-", label="global basis")
        if result.has_local:
            nl = result.local_energy_modes_required()
            ax2.plot(v, np.where(nl < 0, np.nan, nl), "s--",
                     label="local POD")
        if loo is not None:
            nlo = loo.energy_modes_required()
            if np.any(nlo >= 0):
                ax2.plot(v, np.where(nlo < 0, np.nan, nlo), "^:",
                         label="global, leave-one-out")
    ax2.set_title("Modes for the energy-injection tolerance", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel(_label(result.parameter))
        ax.set_ylabel("Number of modes")
        ax.grid(alpha=.8, zorder=0)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_mode_share(result: GlobalPODResult, n_show: int = 10) -> Figure:
    """Share of each simulation's fluctuation energy in the global modes."""
    share = result.mode_share[:, :n_show]
    n = share.shape[1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(result.values) + 2.2))
    im = ax.imshow(100 * share, aspect="auto", origin="lower",
                   cmap="inferno", vmin=0, vmax=100)
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)])
    ax.set_yticks(range(len(result.values)),
                  [str(v) for v in result.values])
    ax.set_xlabel("Global mode")
    ax.set_ylabel(_label(result.parameter))
    ax.set_title("Share of each simulation's fluctuation energy [%]",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="[%]")
    fig.tight_layout()
    return fig


def plot_subspace_angles(angles: SubspaceAngles, parameter: str) -> Figure:
    """Principal angles between local POD subspaces and to the global basis.

    Left: between neighbouring simulations (small angles: one basis can
    serve both). Right: between each local subspace and the global one.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    mid = 0.5 * (angles.values[:-1] + angles.values[1:])
    for i in range(angles.k):
        ax1.plot(mid, angles.neighbour[:, i], "o-", ms=4,
                 label=f"angle {i + 1}")
        ax2.plot(angles.values, angles.to_global[:, i], "o-", ms=4,
                 label=f"angle {i + 1}")
    ax1.set_xlabel(f"{_label(parameter)} (midpoint of the pair)")
    ax1.set_title(f"Local {angles.k}-D subspaces of neighbouring "
                  "simulations", fontsize=10)
    ax2.set_xlabel(_label(parameter))
    ax2.set_title(f"Local {angles.k}-D subspace vs leading {angles.k} "
                  "global modes", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_ylabel("Principal angle [deg]")
        ax.set_ylim(0, 90)
        ax.grid(alpha=.8, zorder=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_mode_injection(result: GlobalPODResult, n_show: int = 4) -> Figure:
    """Energy injected by each leading global mode versus the parameter.

    Shows the total energy injection of the raw field (black) and the
    contribution of the leading global modes (and of the mean field); these
    are the quantities a parametric model would have to interpolate.
    """
    cases = [c for c in result.cases if c.energy is not None]
    if not cases:
        raise ValueError("No energy-injection analysis in the result.")
    v = np.array([c.value for c in cases])
    n = min(n_show, min(c.energy.n_modes for c in cases))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(v, [c.energy.total_original for c in cases], "k-o", lw=2,
            label="total (raw field)")
    for k in range(n):
        ax.plot(v, [c.energy.mode_totals[k] for c in cases], "o-", ms=4,
                label=f"global mode {k + 1}")
    ax.plot(v, [c.energy.mean_total for c in cases], "x--", color="gray",
            label="mean field")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xlabel(_label(result.parameter))
    ax.set_ylabel("Injected energy over the window [J]")
    ax.set_title("Energy injection by global mode (positive = excites)",
                 fontsize=10)
    ax.grid(alpha=.8, zorder=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_harmonic_response(result: GlobalPODResult) -> Figure:
    """Amplitude and phase of the leading coefficients at the forcing
    frequency versus the parameter (phase 0: displacement, +90: velocity).

    Smooth curves here mean the locked response of the global modes can be
    interpolated; jumps mark changes of regime (e.g. lock-in boundaries).
    """
    cases = [c for c in result.cases if c.phases is not None]
    if not cases:
        raise ValueError("No harmonic-phase analysis in the result.")
    v = np.array([c.value for c in cases])
    n = min(c.phases.n_modes for c in cases)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for k in range(n):
        amp = np.array([c.phases.amplitude[k, 0] for c in cases])
        phase = np.array([c.phases.phase_deg[k, 0] for c in cases])
        ax1.semilogy(v, amp, "o-", ms=4, label=f"mode {k + 1}")
        ax2.plot(v, phase, "o-", ms=4, label=f"mode {k + 1}")
    ax1.set_ylabel("Amplitude at the forcing frequency")
    ax2.set_ylabel("Phase lead [deg]")
    ax2.set_ylim(-180, 180)
    ax2.set_yticks(range(-180, 181, 90))
    ax2.axhline(0, color="gray", lw=.8)
    ax2.set_xlabel(_label(result.parameter))
    ax1.set_title("Fundamental of the leading global coefficients",
                  fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=.8, zorder=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_temporal_spectra(result: GlobalPODResult, n_show: int = 4,
                          n_harmonics: int = 6,
                          f_forcing: float | None = None,
                          log_amplitude: bool = True) -> Figure:
    """Amplitude spectra of the leading global temporal coefficients.

    One subplot per global mode, one line per simulation (coloured by the
    sweep parameter). The coefficients are the projections ``a_k(t)`` of each
    simulation onto the global mode. The forcing frequency and its harmonics
    are marked by red dashed lines, the dominant (non-zero) peak of every
    simulation by a dot.

    Parameters
    ----------
    result : GlobalPODResult
    n_show : int
        Number of leading global modes (subplots).
    n_harmonics : int
        Harmonics of the forcing frequency that are marked; the frequency
        axis extends at least to the last one.
    f_forcing : float, optional
        Forcing frequency [Hz] (default: that of the simulations).
    log_amplitude : bool
        Logarithmic amplitude axis (default), so that the weak lines next to
        the locked fundamental stay visible.

    Notes
    -----
    The amplitude is ``2 |rfft(a_k - mean)| / n``, the peak amplitude of a
    sinusoid in the units of the (normalised) coefficients. This differs from
    the single-case plot, which shows the unscaled ``|rfft|`` and therefore
    depends on the record length; the simulations of a sweep may differ in
    length.
    """
    cases = result.cases
    n_show = min(n_show, *(c.pod.time_coeff.shape[0] for c in cases))
    if f_forcing is None:
        f_forcing = cases[0].config.f_forcing
    cols, mappable = _value_colors(result.values)

    # spectra[k][i] = (freqs, amplitude, dominant frequency) of case i, mode k
    spectra = [[] for _ in range(n_show)]
    for c in cases:
        for k in range(n_show):
            a_k = c.pod.time_coeff[k]
            f_dom, freqs, amp = dominant_frequency(a_k, c.dt)
            spectra[k].append((freqs, 2.0 * amp / len(a_k), f_dom))

    f_dom_max = max(f for k in range(n_show) for _, _, f in spectra[k])
    x_max = max(n_harmonics * f_forcing, 1.2 * f_dom_max, 1e-6)

    fig, axes = plt.subplots(n_show, 1, figsize=(9, 2.4 * n_show),
                             sharex=True, squeeze=False,
                             layout="constrained")
    for k in range(n_show):
        ax = axes[k, 0]
        for i in range(1, n_harmonics + 1):
            ax.axvline(i * f_forcing, color="r", ls="--", lw=1, zorder=1)
        for col, (freqs, amp, f_dom) in zip(cols, spectra[k]):
            ax.plot(freqs[1:], amp[1:], color=col, lw=1.1, zorder=2)
            j = int(np.argmin(np.abs(freqs - f_dom)))
            ax.plot(f_dom, amp[j], "o", ms=4, color=col, mec="k", mew=.6,
                    zorder=3)

        ax.set_xlim(0, x_max)
        in_window = np.concatenate([amp[(f > 0) & (f <= x_max)]
                                    for f, amp, _ in spectra[k]])
        top = in_window.max()
        if log_amplitude:
            ax.set_yscale("log")
            floor = in_window[in_window > 0].min() if np.any(
                in_window > 0) else top
            # keep numerically empty bins from stretching the axis
            ax.set_ylim(max(floor, 1e-6 * top) / 1.5, 1.5 * top)
        else:
            ax.set_ylim(0, 1.05 * top)
        ax.set_ylabel(f"Amplitude of $a_{{{k + 1}}}$")
        ax.set_title(f"Global mode {k + 1} "
                     f"({100 * result.basis.energy_frac[k]:.1f}% of the "
                     "pooled energy)", fontsize=10)
        ax.grid(alpha=.8, zorder=0)

    axes[0, 0].legend(handles=[
        Line2D([], [], color="r", ls="--", lw=1,
               label=f"Forcing frequency and harmonics "
                     f"({f_forcing:.3f} Hz)"),
        Line2D([], [], marker="o", ls="", ms=4, color="gray", mec="k",
               mew=.6, label="Dominant peak of each simulation")],
        fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("Frequency [Hz]")
    fig.colorbar(mappable, ax=axes.ravel().tolist(),
                 label=_label(result.parameter))
    return fig
