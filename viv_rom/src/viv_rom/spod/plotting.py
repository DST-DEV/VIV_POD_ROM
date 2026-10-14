"""Figures for the SPOD analysis.

Every function returns the created :class:`matplotlib.figure.Figure` and does
not save or close it (see :func:`viv_rom.spod.pipeline.write_figures`).
Frequencies are shown relative to the forcing frequency.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from viv_rom.domain.motion import ForcedMotion
from viv_rom.pod.core import FieldInfo
from viv_rom.spod.analysis import HarmonicContent
from viv_rom.spod.core import SPODResult
from viv_rom.spod.injection import SPODInjection
from viv_rom.spod.locking import PhaseLockedResponse

_COMPONENT_LABEL = {"Fx": "Fx (edgewise)", "Fz": "Fz (flapwise)"}


def _components(field_info: FieldInfo) -> list[str]:
    return ["Fx", "Fz"] if field_info.kind == "combined" \
        else [field_info.component]


def _split(arr: np.ndarray, field_info: FieldInfo) -> list[np.ndarray]:
    """Per-component parts of an array whose first axis is the point axis."""
    return list(field_info.split(arr)) if field_info.kind == "combined" \
        else [arr]


def _mark_harmonics(ax, n_harmonics: int) -> None:
    for n in range(1, n_harmonics + 1):
        ax.axvline(n, color="r", ls="--", lw=.8,
                   label="forcing harmonics" if n == 1 else None)


def plot_eigenvalue_spectrum(spod: SPODResult, f_forcing: float,
                             n_harmonics: int, n_show: int = 3) -> Figure:
    """SPOD eigenvalue spectrum and low-rank measure ``lambda_1 / sum``.

    Left: the leading eigenvalues versus frequency. A deterministic locked
    response is a narrow line of the first mode, towering over the others.
    Right: the share of mode 1 at each frequency; values near 1 mean a
    single coherent structure (rank-1) at that frequency.
    """
    n_show = min(n_show, spod.n_modes)
    x = spod.freqs / f_forcing
    xmax = n_harmonics + 1.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    positive = spod.energy > 0
    ax1.semilogy(x[positive], spod.energy[positive], c="0.6", ls="--", lw=1,
                 label="all modes")
    for k in range(n_show):
        ok = spod.eigenvalues[:, k] > 0
        ax1.semilogy(x[ok], spod.eigenvalues[ok, k], lw=1.4,
                     label=f"mode {k + 1}")
    _mark_harmonics(ax1, n_harmonics)
    ax1.set_xlim(0, xmax)
    ax1.set_xlabel("Frequency / forcing frequency [-]")
    ax1.set_ylabel("SPOD eigenvalue (weighted one-sided PSD)")
    ax1.set_title("SPOD energy spectrum")
    ax1.grid(alpha=.5, which="both")
    ax1.legend(fontsize=8)

    lead = spod.mode_fractions()[:, 0]
    ax2.plot(x, lead, "-", lw=1.2)
    _mark_harmonics(ax2, n_harmonics)
    ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Frequency / forcing frequency [-]")
    ax2.set_ylabel(r"$\lambda_1 / \sum_k \lambda_k$ [-]")
    ax2.set_title("Share of the leading mode (1 = coherent / low rank)")
    ax2.grid(alpha=.5)
    fig.tight_layout()
    return fig


def plot_harmonic_modal_content(content: HarmonicContent,
                                locked: PhaseLockedResponse | None = None
                                ) -> Figure:
    """Modal structure at the forcing harmonics.

    Panels: power share of the leading modes per harmonic; cumulative power
    versus number of modes; share of the power that is phase-locked to the
    forced motion.
    """
    n_h, n_show = content.mode_fractions.shape
    n_panels = 3 if locked is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4))
    h = content.harmonics
    width = 0.8 / n_show

    ax = axes[0]
    for k in range(n_show):
        ax.bar(h + (k - (n_show - 1) / 2) * width,
               100 * content.mode_fractions[:, k], width=width,
               label=f"mode {k + 1}", zorder=10)
    ax.set_xticks(h)
    ax.set_xlabel("Harmonic n of the forcing frequency")
    ax.set_ylabel("Share of the power at the bin [%]")
    ax.set_title("Leading modes per harmonic")
    ax.grid(alpha=.5, zorder=0)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axis("on")
    for i, n in enumerate(h):
        cum = np.cumsum(content.mode_fractions[i])
        ax.plot(np.arange(1, n_show + 1), 100 * cum, "o-",
                label=f"n = {n}")
    ax.axhline(100 * content.threshold, c="gray", ls="--", lw=1)
    ax.set_xticks(np.arange(1, n_show + 1))
    ax.set_xlabel("Number of SPOD modes")
    ax.set_ylabel("Cumulative power at the bin [%]")
    ax.set_title("Convergence per harmonic")
    ax.grid(alpha=.5)
    ax.legend(fontsize=8)

    if locked is not None:
        ax = axes[2]
        ax.bar(locked.harmonics, 100 * locked.locked_fraction, zorder=10,
               color="C2")
        ax.set_xticks(locked.harmonics)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Harmonic n of the forcing frequency")
        ax.set_ylabel("Phase-locked share of the power [%]")
        ax.set_title("Locking to the forced motion")
        ax.grid(alpha=.5, zorder=0)
    fig.tight_layout()
    return fig


def plot_leading_mode_map(spod: SPODResult, y: np.ndarray,
                          field_info: FieldInfo, f_forcing: float,
                          n_harmonics: int,
                          scaling: np.ndarray | None = None,
                          decades: float = 5.0) -> Figure:
    """Where along the span the leading SPOD mode carries power, per frequency.

    Shows ``log10( lambda_1(f) |Phi_1(y, f)|^2 )`` per force component
    (the leading mode's PSD at each station; with a normalisation, the
    normalisation is undone so the units are those of the force).
    """
    comps = _components(field_info)
    fig, axes = plt.subplots(1, len(comps), figsize=(5.6 * len(comps), 4.6),
                             sharey=True, squeeze=False)
    phi = spod.modes[:, :, 0]                                  # (n_f, n_pts)
    if scaling is not None:
        phi = phi / np.asarray(scaling)[None, :]
    power = spod.eigenvalues[:, 0][:, None] * np.abs(phi) ** 2
    parts = _split(power.T, field_info)                        # (n_sec, n_f)
    vmax = max(p.max() for p in parts)
    x = spod.freqs / f_forcing
    for ax, comp, part in zip(axes[0], comps, parts):
        with np.errstate(divide="ignore"):
            z = np.log10(np.maximum(part, vmax * 10 ** -decades))
        mesh = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis",
                             vmin=np.log10(vmax) - decades,
                             vmax=np.log10(vmax))
        _mark_harmonics(ax, n_harmonics)
        ax.set_xlim(0, n_harmonics + 1.5)
        ax.set_xlabel("Frequency / forcing frequency [-]")
        ax.set_title(f"{_COMPONENT_LABEL[comp]}: mode 1 PSD per station")
        fig.colorbar(mesh, ax=ax, label="log10 PSD")
    axes[0, 0].set_ylabel("Spanwise station y [m]")
    fig.tight_layout()
    return fig


def plot_locked_modes(y: np.ndarray, response: np.ndarray,
                      field_info: FieldInfo, motion: ForcedMotion,
                      mode_power_share: np.ndarray, n_show: int = 3,
                      mask_fraction: float = 0.02) -> Figure:
    """Phase-locked spatial modes of the fundamental (amplitude and phase).

    ``response`` (n_points, n_modes) is ``H_k`` in physical units: the force
    of mode ``k`` is ``|H_k| sin(omega t + arg H_k)``, with the phase
    relative to the forced *displacement*. ``+90 deg`` is in phase with the
    velocity of the motion (energy injection where the force acts along the
    motion); the phase is hidden where the amplitude is below
    ``mask_fraction`` of its maximum.
    """
    comps = _components(field_info)
    n_show = min(n_show, response.shape[1])
    fig, axes = plt.subplots(len(comps), 2, figsize=(12, 3.8 * len(comps)),
                             sharex=True, squeeze=False)
    parts = _split(response, field_info)
    for row, (comp, part) in enumerate(zip(comps, parts)):
        amp_ax, ph_ax = axes[row]
        amp = np.abs(part[:, :n_show])
        cutoff = mask_fraction * max(amp.max(), np.finfo(float).tiny)
        for k in range(n_show):
            label = (f"mode {k + 1} ({100 * mode_power_share[k]:.0f}% of "
                     "locked power)")
            amp_ax.plot(y, amp[:, k], label=label)
            phase = np.degrees(np.angle(part[:, k]))
            phase = np.where(amp[:, k] > cutoff, phase, np.nan)
            ph_ax.plot(y, phase, label=f"mode {k + 1}")
        twin = amp_ax.twinx()
        twin.plot(y, motion.modeshape(comp, y), c="k", ls="--", lw=1)
        twin.set_ylabel("Forced motion mode shape [-]")
        amp_ax.set_ylabel(f"|{comp}| of the locked mode [N/m]")
        amp_ax.set_title(f"{_COMPONENT_LABEL[comp]}: amplitude")
        amp_ax.grid(alpha=.5)
        ph_ax.axhline(90, c="g", ls=":", lw=1.2)
        ph_ax.axhline(0, c="0.5", ls=":", lw=1)
        ph_ax.set_ylim(-185, 185)
        ph_ax.set_yticks(np.arange(-180, 181, 90))
        ph_ax.set_ylabel("Phase vs forced displacement [deg]")
        ph_ax.set_title(f"{_COMPONENT_LABEL[comp]}: phase "
                        "(+90 deg = in phase with velocity)")
        ph_ax.grid(alpha=.5)
        if row == 0:
            amp_ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Spanwise station y [m]")
    fig.suptitle("Phase-locked SPOD modes at the forcing frequency",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_injection(injection: SPODInjection, n_show: int = 6,
                   r_values: tuple[int, ...] = (1, 2)) -> Figure:
    """Mean power injected by each locked SPOD mode and its spanwise profile.

    Left: power of the leading modes (bars) and the cumulative total
    (line). Right: the spanwise distribution ``p(y)`` of the full locked
    response against reconstructions from the leading modes.
    """
    n_show = min(n_show, injection.n_modes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    k = np.arange(1, n_show + 1)
    power = injection.mode_power[:n_show]
    ax1.bar(k, power, zorder=10)
    ax1.set_xticks(k)
    ax1.set_xlabel("SPOD mode")
    ax1.set_ylabel("Mean injected power [W]")
    ax1.set_title("Power injected per locked mode")
    ax1.axhline(0, c="k", lw=.8)
    ax1.grid(alpha=.5, zorder=0)
    twin = ax1.twinx()
    total = injection.total_power
    if total != 0:
        cum = 100 * np.cumsum(injection.mode_power)[:n_show] / total
        twin.plot(k, cum, "o-", c="C1")
        twin.set_ylim(min(0.0, cum.min()) - 5, max(105.0, cum.max() + 5))
    twin.axhline(100, c="gray", ls="--", lw=1)
    twin.set_ylabel("Cumulative share of the total [%]")

    y = injection.y
    ax2.plot(y, injection.total_density, "k", lw=1.8, label="all modes")
    wanted = list(r_values)
    for key in ("rms", "mean", "max"):
        n = injection.n_required.get(key)
        if n is not None and n not in wanted:
            wanted.append(n)
            break
    for r in sorted(set(r for r in wanted if 0 < r < injection.n_modes)):
        ax2.plot(y, injection.reconstructed(r), lw=1.1,
                 label=f"{r} mode(s)")
    ax2.axhline(0, c="k", lw=.6)
    ax2.set_xlabel("Spanwise station y [m]")
    ax2.set_ylabel("Mean injected power p(y) [W/m]")
    ax2.set_title("Spanwise power injection (locked fundamental)")
    ax2.grid(alpha=.5)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig
