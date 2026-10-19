from pathlib import Path
from typing import Mapping
from enum import Enum

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from viv_rom.pod.config import PODConfig
from viv_rom.pod.pipeline import analyse, PODAnalysisResult
from viv_rom.signal_utils import dominant_frequency
from viv_rom.pod.power import PowerCheck
from viv_rom.domain.blade import BladeGeometry


class AbcissaType(Enum):
    SPAN_POS = "spanwise position (root y-coordinate) [m]"
    RED_VEL = "reduced velocity [-]"
    CHORD = "chord length [m]"


class CharacteristicLength(Enum):
    CHORD = "chord length"
    PROJECTED_CHORD = "Chord projected with pitch and twist angle"


# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
# CASE = "a01_p60_i50"  # amplitude/pitch/inclination code, see README
PITCH = 85
INCLINATION = 50
AMPLITUDE = 1  # Forced motion amplitude factor
WIND_SPEEDS = list(range(6, 23))  # free-stream wind speed [m/s] to analyse (6-22)
FIELD = "combined"
normalizer = "dynamic_pressure"
R_ROOT = 2.8  # [m] Blade root diameter
F_FORCING = .624  # [Hz] Forcing frequency (= first edgewise natural frequency)

ABSCISSA_COORD = AbcissaType.SPAN_POS
CHAR_LENGTH = CharacteristicLength.PROJECTED_CHORD
CMAP = "magma"

case = f"a{AMPLITUDE:02d}_p{PITCH:02d}_i{INCLINATION:02d}"
OUTPUT_DIR = Path("./results") / case / "wind_sweep" / "pod"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Calculate POD
# =============================================================================
results = {}
for wind_speed in WIND_SPEEDS:
    config = PODConfig(amplitude=AMPLITUDE, pitch=PITCH,
                       inclination=INCLINATION, wind_speed=wind_speed,
                       data_dir=DATA_DIR, field=FIELD,
                       normalization=normalizer)
    results[wind_speed] = analyse(config)


span_pos = results[wind_speed].y
# span_pos = np.linspace(0, 96, 100)
span_mask = np.where(span_pos >= R_ROOT)
span_pos = span_pos[span_mask]

if ABSCISSA_COORD is AbcissaType.SPAN_POS:
    abscissa = span_pos
elif ABSCISSA_COORD is AbcissaType.CHORD:
    blade_file_path = Path(__file__).parent / "data" / "IEA_10MW" \
        / "IEA-10-198-RWT.yaml"
    blade = BladeGeometry.from_windio(blade_file_path)
    abscissa = blade.chord_at(span_pos)
elif ABSCISSA_COORD is AbcissaType.RED_VEL:
    blade_file_path = Path(__file__).parent / "data" / "IEA_10MW" \
        / "IEA-10-198-RWT.yaml"
    blade = BladeGeometry.from_windio(blade_file_path)

    chord = blade.chord_at(span_pos)

    if CHAR_LENGTH == CharacteristicLength.CHORD:
        char_length = chord
    elif CHAR_LENGTH == CharacteristicLength.PROJECTED_CHORD:
        # Calculate local chord angle (assumption: twist and pitch defined
        # relative to blade tip)
        twist = blade.twist_at(span_pos)
        local_angle = np.deg2rad(90 - PITCH) - twist

        char_length = chord * np.cos(local_angle)
    else:
        raise ValueError("Invalid characteristic length selection "
                         f"{CHAR_LENGTH}")

    abscissa = [wind_speed / (F_FORCING * char_length)
                for wind_speed in WIND_SPEEDS]
else:
    raise ValueError(f"Invalid abscissa coordinate selection {ABSCISSA_COORD}")


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
fig_time_coeffs_name = "temporal_coefficients"
fig_time_coeffs.savefig(OUTPUT_DIR / (fig_time_coeffs_name + ".png"), dpi=150)
fig_time_coeffs.savefig(OUTPUT_DIR / (fig_time_coeffs_name + ".svg"))


# =============================================================================
# Plot sectional energy
# =============================================================================
def plot_sectional_energy_overlayed(abscissa: np.ndarray,
                                    power: Mapping[int, PowerCheck],
                                    normalize: bool = True,
                                    ) -> Figure:
    """Spanwise energy-injection distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = mpl.colormaps[CMAP](np.linspace(0, 1, len(power)))
    for i, (param, pwr) in enumerate(power.items()):
        if isinstance(abscissa, list):
            abscissa_i = abscissa[i]
        else:
            abscissa_i = abscissa

        factor = param**2 if normalize else 1.0
        ax.plot(abscissa_i, pwr.sectional_energy[span_mask]/factor,
                c=colors[i], label=param, zorder=1)
    ax.set_xlabel(ABSCISSA_COORD.value)
    ax.set_ylabel("Sectional energy injection [J/m]")
    ax.set_title("Sectional aerodynamic energy injection")
    ax.grid(alpha=.8, zorder=0)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


power_dict = {wind_speed: results[wind_speed].power
              for wind_speed in WIND_SPEEDS}
fig_pwr = plot_sectional_energy_overlayed(abscissa=abscissa, power=power_dict)
fig_pwr_name = "power_sectional_" + ABSCISSA_COORD.name.lower()
fig_pwr.savefig(OUTPUT_DIR / (fig_pwr_name + ".png"), dpi=150)
fig_pwr.savefig(OUTPUT_DIR / (fig_pwr_name + ".svg"))


def plot_sectional_energy_pcolormesh(
    abscissa: np.ndarray | list[np.ndarray],
    power: Mapping[int, PowerCheck],
    normalize: bool = True,
) -> Figure:
    """Plot sectional energy as a pcolormesh."""

    wind_speeds = np.array(list(power.keys()))

    # Assemble data matrix
    Z = []
    for wind_speed, pwr in power.items():
        factor = wind_speed**2 if normalize else 1.0
        Z.append(pwr.sectional_energy[span_mask] / factor)

    Z = np.asarray(Z)        # shape = (n_wind, n_sections)

    # Coordinate arrays
    if isinstance(abscissa, list):
        # Reduced velocity case: x changes with wind speed
        X = np.asarray(abscissa)
    else:
        # Span position or chord
        X = np.tile(abscissa, (len(wind_speeds), 1))

    Y = np.tile(wind_speeds[:, None], (1, X.shape[1]))

    fig, ax = plt.subplots(figsize=(7, 4))

    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=CMAP)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Sectional energy injection [J/m]")

    ax.set_xlabel(ABSCISSA_COORD.value)
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title("Sectional aerodynamic energy injection")

    fig.tight_layout()
    return fig


fig_pwr_mesh = plot_sectional_energy_pcolormesh(abscissa=abscissa,
                                                power=power_dict)
fig_pwr_mesh_name = "power_sectional_mesh_" + ABSCISSA_COORD.name.lower()
fig_pwr_mesh.savefig(OUTPUT_DIR / (fig_pwr_mesh_name + ".png"), dpi=150)
fig_pwr_mesh.savefig(OUTPUT_DIR / (fig_pwr_mesh_name + ".svg"))


# =============================================================================
# Plot mode orbits
# =============================================================================
# def plot_force_orbit(t: np.ndarray, pod_res: Mapping[int, PODAnalysisResult]
#                      ) -> Figure:
#     """Sectional-mode orbit (mode 1 vs. mode 2) at one station."""

#     fig, ax = plt.subplots(figsize=(6, 6))

#     cmap = mpl.colormaps[CMAP]
#     colors = cmap(np.linspace(0, 1, len(pod_res)))
#     param_vals = list(pod_res.keys())

#     for i, (param, res) in enumerate(pod_res.items()):
#         # Reconstruct forces from first mode
#         Fx_1, Fz_1 = res.field_info.split(res.pod.reconstruct(1))

#         # Reconstruct forces from second mode
#         field_2 = res.pod.mean[:, None] \
#             + res.pod.modes[:, 1][:, None] @ res.pod.time_coeff[1, :][None, :]
#         Fx_2, Fz_2 = res.field_info.split(field_2)

#         # Choose station to plot (position of max energy injection)
#         station = np.argwhere(res.power.sectional_energy)[0][0]

#         # Calculate norm factor
#         norm_factor = max(np.max(np.abs(Fx_1[station, :])),
#                           np.max(np.abs(Fx_2[station, :])))

#         ax.plot(Fx_1[station, :]/norm_factor, Fx_2[station, :]/norm_factor,
#                 lw=1.2, alpha=.8, c=colors[i])

#     colorizer = mpl.colorizer.Colorizer(
#         norm=mpl.colors.Normalize(np.min(param_vals), np.max(param_vals)),
#         cmap=CMAP)

#     fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer),
#                  ax=ax, orientation='vertical', label='Wind speed')

#     ax.set_xlabel("Mode 1")
#     ax.set_ylabel("Mode 2")
#     ax.set_title("Sectional-force orbit at energy injection maximum")
#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(alpha=.8, zorder=0)
#     return fig

# fig_orbit = plot_force_orbit(
#     t=results[WIND_SPEEDS[0]].force_file.data.t,
#     pod_res=results)
# fig_orbit.savefig(OUTPUT_DIR / "force_orbit.png", dpi=150)
