from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from viv_rom.pod.config import PODConfig
from viv_rom.pod.pipeline import analyse, PODAnalysisResult
from viv_rom.signal_utils import dominant_frequency
from viv_rom.pod.power import PowerCheck
from viv_rom.pod.core import FieldInfo, PODResult


# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see README
WIND_SPEEDS = list(range(6, 23))  # free-stream wind speed [m/s] to analyse (6-22)
FIELD = "combined"
CMAP = "magma"
OUTPUT_DIR = Path("./results") / CASE / "wind_sweep"  # Output folder


# =============================================================================
# Calculate POD
# =============================================================================
results = {}
for wind_speed in WIND_SPEEDS:
    config = PODConfig.from_case(case=CASE, wind_speed=wind_speed,
                                 data_dir=DATA_DIR, field="combined", )
    results[wind_speed] = analyse(config)


# =============================================================================
# Plot temporal coefficients
# =============================================================================
def plot_temporal_coefficients_overlayed(
        pod_res: Mapping[int, PODAnalysisResult],
        f_forcing: float, n_show: int = 4,
        n_harmonics: int = 6) -> Figure:
    """Temporal coefficients and their spectra for the leading modes.

    The prescribed forcing frequency (and harmonics) and the dominant
    spectral peak are highlighted.
    """
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.4 * n_show),
                             squeeze=False)

    cmap = mpl.colormaps[CMAP]
    colors = cmap(np.linspace(0, 1, len(pod_res)))
    param_vals = list(pod_res.keys())

    for k in range(n_show):
        for i, (param, res) in enumerate(pod_res.items()):
            a_k = res.pod.time_coeff[k, :]
            norm_factor = np.max(a_k)
            f_dom, freqs, amp = dominant_frequency(a_k/norm_factor, res.dt)

            axes[k, 0].plot(res.force_file.data.t, a_k/norm_factor,
                            c=colors[i])
            axes[k, 1].plot(freqs, amp, c=colors[i])

        axes[k, 0].set_ylabel(f"a_{k+1}(t)")
        axes[k, 0].set_title(f"Mode {k+1} normalized time coefficient "
                             f"({100*res.pod.energy_frac[k]:.1f}% energy)")
        axes[k, 0].grid(alpha=.8, zorder=0)

        axes[k, 1].axvline(f_forcing, color="r", ls="--", lw=1,
                           label="Forcing frequency")

        for i in range(2, n_harmonics + 1):
            axes[k, 1].axvline(f_forcing * i, color="r", ls="--", lw=1,
                               label="")
        axes[k, 1].axvline(f_dom, color="g", ls=":", lw=1.5,
                           label="Dominant peak")
        axes[k, 1].set_xlim(0,
                            max(n_harmonics * f_forcing, 2 * f_dom, 1e-6))
        axes[k, 1].set_title(f"Spectrum (peak = {f_dom:.3f} Hz, "
                             f"{f_dom / f_forcing:.2f} x forcing)")
        axes[k, 1].grid(alpha=.8, zorder=0)

        # Create colorbar
        colorizer = mpl.colorizer.Colorizer(
            norm=mpl.colors.Normalize(np.min(param_vals), np.max(param_vals)),
            cmap=CMAP)

        fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer),
                     ax=axes[k, 1], orientation='vertical',
                     label='Wind speed')

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    return fig


fig_time_coeffs = plot_temporal_coefficients_overlayed(
    results, f_forcing=config.f_forcing)
fig_time_coeffs.savefig(OUTPUT_DIR / "temporal_coefficients.png", dpi=150)


# =============================================================================
# Plot sectional energy
# =============================================================================
def plot_sectional_energy_overlayed(y: np.ndarray,
                                    power: Mapping[int, PowerCheck],
                                    normalize: bool = True,
                                    ) -> Figure:
    """Spanwise energy-injection distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = mpl.colormaps[CMAP](np.linspace(0, 1, len(power)))
    for i, (param, pwr) in enumerate(power.items()):
        factor = np.max(pwr.sectional_energy) if normalize else 1.0
        ax.plot(y, pwr.sectional_energy/factor,
                c=colors[i], label=param, zorder=1)
    ax.set_xlabel("Spanwise station y [m]")
    ax.set_ylabel("Sectional energy injection [J/m]")
    ax.set_title("Sectional aerodynamic energy injection")
    ax.grid(alpha=.8, zorder=0)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


power_dict = {wind_speed: results[wind_speed].power
              for wind_speed in WIND_SPEEDS}
fig_pwr = plot_sectional_energy_overlayed(
    y=results[WIND_SPEEDS[0]].force_file.data.y,
    power=power_dict)
fig_pwr.savefig(OUTPUT_DIR / "power_sectional.png", dpi=150)


# =============================================================================
# Plot mode orbits
# =============================================================================
def plot_force_orbit(t: np.ndarray, pod_res: Mapping[int, PODAnalysisResult]
                     ) -> Figure:
    """Sectional-mode orbit (mode 1 vs. mode 2) at one station."""

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = mpl.colormaps[CMAP]
    colors = cmap(np.linspace(0, 1, len(pod_res)))
    param_vals = list(pod_res.keys())

    for i, (param, res) in enumerate(pod_res.items()):
        # Reconstruct forces from first mode
        Fx_1, Fz_1 = res.field_info.split(res.pod.reconstruct(1))

        # Reconstruct forces from second mode
        field_2 = res.pod.mean[:, None] \
            + res.pod.modes[:, 1][:, None] @ res.pod.time_coeff[1, :][None, :]
        Fx_2, Fz_2 = res.field_info.split(field_2)

        # Choose station to plot (position of max energy injection)
        station = np.argwhere(res.power.sectional_energy)[0][0]

        # Calculate norm factor
        norm_factor = max(np.max(np.abs(Fx_1[station, :])),
                          np.max(np.abs(Fx_2[station, :])))

        ax.plot(Fx_1[station, :]/norm_factor, Fx_2[station, :]/norm_factor,
                lw=1.2, alpha=.8, c=colors[i])

    colorizer = mpl.colorizer.Colorizer(
        norm=mpl.colors.Normalize(np.min(param_vals), np.max(param_vals)),
        cmap=CMAP)

    fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer),
                 ax=ax, orientation='vertical', label='Wind speed')

    ax.set_xlabel("Mode 1")
    ax.set_ylabel("Mode 2")
    ax.set_title("Sectional-force orbit at energy injection maximum")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.8, zorder=0)
    return fig

fig_orbit = plot_force_orbit(
    t=results[WIND_SPEEDS[0]].force_file.data.t,
    pod_res=results)
fig_orbit.savefig(OUTPUT_DIR / "force_orbit.png", dpi=150)
