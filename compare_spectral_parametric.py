from pathlib import Path
from typing import Mapping
from enum import Enum

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from intersect import intersection

from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.pipeline import analyse
from viv_rom.spectral.forcing_response import ForcingResponse
from viv_rom.spectral.shedding import SheddingFrequencies
from viv_rom.domain.blade import BladeGeometry
from viv_rom.domain.inflow import InflowConvention
from viv_rom.io.results.export import load_result


class AbcissaType(Enum):
    SPAN_POS = "spanwise position (root y-coordinate) [m]"
    RED_VEL = "reduced velocity [-]"
    CHORD = "chord length [m]"


class CharacteristicLength(Enum):
    CHORD = "chord length"
    PROJECTED_CHORD = "Chord projected with pitch and twist angle"


class SheddingPlotMode(Enum):
    MEAN_RATIO = "mean f_vs/f_n"
    MEAN_REL_DIFF = "mean abs(f_vs-f_n)/f_n"
    POT_LOCK_IN = "Potential lock in region"


# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
LOAD_RES = True  # Whether to load results from previous analysis (if
# False, the spectral analysis is rerun for all wind speeds)

PITCH = 85
INCLINATION = 50
AMPLITUDE = 1  # Forced motion amplitude factor
WIND_SPEEDS = list(range(6, 23))  # free-stream wind speed [m/s] to analyse (6-22)
FIELD = "both"
FRAME = "motion"

R_ROOT = 2.8  # [m] Blade root diameter
F_FORCING = .624  # [Hz] Forcing frequency (= first edgewise natural frequency)

NORMALIZE_WIND_SPEED = True
UNWRAP_PHASE = False
ABSCISSA_COORD = AbcissaType.SPAN_POS
RED_VEL_AXIS_MAX = 50  # Upper axis limit for reduced velocity abscissa
CHAR_LENGTH = CharacteristicLength.PROJECTED_CHORD
SHEDDING_PLOT_MODE = SheddingPlotMode.MEAN_REL_DIFF
CMAP = "magma"

case = f"a{AMPLITUDE:02d}_p{PITCH:02d}_i{INCLINATION:02d}"
OUTPUT_DIR = Path("./results") / case / "wind_sweep"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Run spectral analysis
# =============================================================================

configs = {}
results = {}
for wind_speed in WIND_SPEEDS:
    if LOAD_RES:
        res_path = Path("./results") / case / f"u{wind_speed:02d}" \
            / "spectral" / "spectral_results.npz"
        res = load_result(res_path)
        configs[wind_speed] = res.config
        results[wind_speed] = res.result
        config = res.config
    else:
        config = SpectralConfig(amplitude=AMPLITUDE, pitch=PITCH,
                                inclination=INCLINATION, wind_speed=wind_speed,
                                data_dir=DATA_DIR, field=FIELD,
                                frame=FRAME, unwrap_phase=UNWRAP_PHASE)

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
        inflow = InflowConvention()

        # Calculate local chord angle
        # Note: twist is currently arbitrarily defined (directly read from the
        # windIO file)
        twist = blade.twist_at(span_pos)
        angle_of_attack = inflow.angle_of_attack(pitch_deg=PITCH,
                                                 twist_rad=twist)
        # local_angle = np.deg2rad(PITCH) - twist  # This is essentially what the InflowConvention defines

        char_length = chord * np.sin(angle_of_attack)
    else:
        raise ValueError("Invalid characteristic length selection "
                         f"{CHAR_LENGTH}")

    abscissa = [wind_speed / (F_FORCING * char_length)
                for wind_speed in WIND_SPEEDS]
else:
    raise ValueError(f"Invalid abscissa coordinate selection {ABSCISSA_COORD}")


# =============================================================================
# Plot forcing response
# =============================================================================
def plot_forcing_response(
    responses: Mapping[float, ForcingResponse],
    comp: str,
    harmonic: int = 1,
    cmap: str = CMAP,
) -> Figure:
    """Plot spanwise amplitude and phase for a single response component.

    Parameters
    ----------
    responses
        Mapping from wind speed to ``ForcingResponse``.
    idx
        Response component to plot (key into ``response.amplitude`` and
        ``response.phase_deg()``).
    harmonic
        Harmonic to display.
    cmap
        Name of the matplotlib colormap used for the wind-speed colouring.
    """
    if not responses:
        raise ValueError("responses is empty.")

    wind_speeds = np.asarray(sorted(responses.keys()), dtype=float)

    norm = mpl.colors.Normalize(vmin=wind_speeds.min(),
                                vmax=wind_speeds.max())
    cmap = mpl.colormaps[CMAP]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True,
                                   )

    for i, U in enumerate(wind_speeds):
        response = responses[U]
        h = int(np.flatnonzero(response.harmonics == harmonic)[0])
        color = cmap(norm(U))

        if isinstance(abscissa, list):
            abscissa_i = abscissa[i]
        else:
            abscissa_i = abscissa

        amplitude = response.amplitude[comp][:, h]
        if NORMALIZE_WIND_SPEED:
            amplitude = amplitude / U**2

        ax1.plot(
            abscissa_i,
            amplitude[span_mask],
            color=color,
            lw=2,
        )

        ax2.plot(
            abscissa_i,
            response.phase_deg(comp)[:, h][span_mask],
            ".-",
            ms=3,
            color=color,
        )

    # Use first response to obtain harmonic frequency
    first_response = responses[wind_speeds[0]]
    h = int(np.flatnonzero(first_response.harmonics == harmonic)[0])
    f_h = first_response.frequencies[h]

    ax1.set_xlabel(ABSCISSA_COORD.value)
    y_unit1 = "N/m" if not NORMALIZE_WIND_SPEED else "N/m * (m/s)^-2"
    ax1.set_ylabel("Amplitude [" + y_unit1 + "]")
    ax1.set_title(f"Amplitude at {harmonic} × f_forcing ({f_h:.3g} Hz)")
    if ABSCISSA_COORD is AbcissaType.RED_VEL:
        ax1.set_xlim([0, RED_VEL_AXIS_MAX])
    ax1.grid(alpha=0.8)

    if not UNWRAP_PHASE:
        for level in (-180, -90, 0, 90, 180):
            ax2.axhline(
                level,
                color="gray",
                lw=0.8,
                ls="-" if level == 0 else ":",
            )

        ax2.set_yticks([-180, -90, 0, 90, 180])
        ax2.set_ylim(-190, 190)
    else:
        y_max = np.nanmax([response.phase_deg(comp)[:, h]
                           for response in responses.values()])
        y_min = np.nanmin([response.phase_deg(comp)[:, h]
                           for response in responses.values()])

        y_min_360 = np.ceil(abs(y_min) / 360) * 360 * int(y_min/abs(y_min))
        y_max_360 = np.ceil(abs(y_max) / 360) * 360 * int(y_max/abs(y_max))

        y_min_90 = np.ceil(abs(y_min) / 90) * 90 * int(y_min/abs(y_min))
        y_max_90 = np.ceil(abs(y_max) / 90) * 90 * int(y_max/abs(y_max))

        ax2.set_yticks(np.arange(y_min_360, y_max_360, 360))
        ax2.set_yticks(np.arange(y_min_90, y_max_90, 90), minor=True)
        ax2.set_ylim(y_min_90, y_max_90)

    if ABSCISSA_COORD is AbcissaType.RED_VEL:
        ax2.set_xlim([0, RED_VEL_AXIS_MAX])

    ax2.set_xlabel(ABSCISSA_COORD.value)
    ax2.set_ylabel("Phase lead vs. displacement [deg]")
    ax2.set_title("Phase (+90° = in phase with velocity)")
    ax2.grid(alpha=0.8, which="both")

    # Shared colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.02)
    cbar.set_label("Wind speed [m/s]")

    fig.suptitle(f"Force response for {comp}")
    # fig.tight_layout()

    return fig


force_responses = {ws: res.forcing_response for ws, res in results.items()}
fig_force_response = plot_forcing_response(responses=force_responses,
                                           comp="fpar")

fig_force_response_name = "force_response_" + ABSCISSA_COORD.name.lower()
if UNWRAP_PHASE:
    fig_force_response_name += "_unwrapped"
else:
    fig_force_response_name += "_wrapped"

fig_force_response.savefig(OUTPUT_DIR / "spectral"
                           / (fig_force_response_name + ".png"),
                           dpi=150)
fig_force_response.savefig(OUTPUT_DIR / "spectral"
                           / (fig_force_response_name + ".svg"))


# =============================================================================
# Shedding frequency plot
# =============================================================================

def plot_shedding_frequency(sheddings: dict[int, SheddingFrequencies],
                            f_max_factor: float = 3.0) -> Figure:
    """Shedding-frequency band along the span against the lock-in range.
    """
    wind_speeds = list(sheddings.keys())
    f_n = sheddings[wind_speeds[0]].f_forcing
    f_top = f_max_factor * f_n

    norm = mpl.colors.Normalize(vmin=np.min(wind_speeds),
                                vmax=np.max(wind_speeds))
    cmap = mpl.colormaps[CMAP]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (wind_speed, shedding) in enumerate(sheddings.items()):
        lock_lo, lock_hi = shedding.lock_in_frequencies
        color = cmap(norm(wind_speed))

        if isinstance(abscissa, list):
            abscissa_i = abscissa[i]
        else:
            abscissa_i = abscissa

        # ax.fill_between(s.y, 0, f_top, where=s.possible_lock_in,
        #                 step="mid", color="0.5", alpha=0.15, lw=0,
        #                 label="shedding band overlaps lock-in range")
        ax.fill_between(abscissa_i, shedding.f_low, shedding.f_high,
                        color=color, alpha=0.35, lw=0)
        ax.plot(abscissa_i, shedding.f_low, color=color, lw=1.2,
                label=f"St = {shedding.strouhal[0]:g}" if i == 0 else "")
        ax.plot(abscissa_i, shedding.f_high, color=color, lw=1.2, ls="--",
                label=f"St = {shedding.strouhal[1]:g}" if i == 0 else "")

    ax.axhspan(lock_lo, lock_hi, color="C1", alpha=0.25, lw=0,
               label=f"lock-in range ({shedding.lock_in_band[0]:g}-"
               f"{shedding.lock_in_band[1]:g} $f_n$)",
               zorder=1)
    ax.axhline(f_n, color="k", ls="--", lw=1.2,
               label=f"natural / forcing frequency ({f_n:.3g} Hz)",
               zorder=2)

    if ABSCISSA_COORD is AbcissaType.RED_VEL:
        ax.set_xlim([0, RED_VEL_AXIS_MAX])
    ax.set_ylim(0, f_top)
    ax.set_xlabel(ABSCISSA_COORD.value)
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("Estimated vortex-shedding frequency")
    ax.secondary_yaxis("right", functions=(lambda f: f / f_n,
                                           lambda r: r * f_n)
                       ).set_ylabel("$f / f_n$")
    ax.legend(fontsize=7, loc="upper left")

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Wind speed [m/s]")

    fig.tight_layout()
    return fig


sheddings = {ws: res.shedding for ws, res in results.items()}
fig_shedding = plot_shedding_frequency(sheddings=sheddings)

fig_shedding_name = "shedding_frequency_" + ABSCISSA_COORD.name.lower()

fig_shedding.savefig(OUTPUT_DIR / "spectral" / (fig_shedding_name + ".png"),
                     dpi=150)
fig_shedding.savefig(OUTPUT_DIR / "spectral" / (fig_shedding_name + ".svg"))


def plot_shedding_frequency_mesh(sheddings: dict[int, SheddingFrequencies],
                                 f_max_factor: float = 3.0) -> Figure:
    """Shedding-frequency band along the span against the lock-in range.
    """
    wind_speeds = np.array(list(sheddings.keys()))

    # Coordinate arrays
    if isinstance(abscissa, list):
        # Reduced velocity case: x changes with wind speed
        X = np.asarray(abscissa)
    else:
        # Span position or chord
        X = np.tile(abscissa, (len(wind_speeds), 1))

    Y = np.tile(wind_speeds[:, None], (1, X.shape[1]))

    if SHEDDING_PLOT_MODE is SheddingPlotMode.MEAN_RATIO:
        Z = []
        for shedding in sheddings.values():
            Z.append(np.mean(
                np.stack([shedding.ratio_low, shedding.ratio_high], axis=0),
                axis=0))
        Z = np.clip(np.asarray(Z), None, f_max_factor
                    )  # shape = (n_wind, n_sections)
    elif SHEDDING_PLOT_MODE is SheddingPlotMode.MEAN_REL_DIFF:
        f_n = sheddings[wind_speeds[0]].f_forcing
        mean_shed_freq = []
        for shedding in sheddings.values():
            mean_shed_freq.append(np.mean(
                np.stack([shedding.f_low, shedding.f_high], axis=0),
                axis=0))
        Z = np.abs(np.asarray(mean_shed_freq) - f_n) / f_n
        Z = np.clip(np.asarray(Z), None, f_max_factor
                    )  # shape = (n_wind, n_sections)

        # Calculate position of lock-in at f_n and harmonics
        span_lock_1, span_lock_2 = [], []
        for i in range(len(wind_speeds)):
            inter_1 = intersection(X[i, :], Z[i, :],
                                   X[i, :], np.full_like(X[i, :], 0.05))
            span_lock_1.append(inter_1[0][-1] if len(inter_1[0])>0 else np.nan)

            inter_2 = intersection(X[i, :], Z[i, :],
                                   X[i, :], np.full_like(X[i, :], 1.0))
            span_lock_2.append(inter_2[0][-1] if len(inter_2[0])>0 else np.nan)
    elif SHEDDING_PLOT_MODE is SheddingPlotMode.POT_LOCK_IN:
        Z = np.zeros(len(sheddings),
                     len(sheddings[wind_speeds[0]].f_low)
                     )  # shape = (n_wind, n_sections)
        for i, shedding in sheddings.values():
            Z[i, shedding.possible_lock_in] = 1
    else:
        raise ValueError("Invalid shedding frequency plot mode "
                         f"{SHEDDING_PLOT_MODE}")

    fig, ax = plt.subplots(figsize=(7, 4))

    fcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=CMAP)

    # Plot lock in curve
    if SHEDDING_PLOT_MODE is SheddingPlotMode.MEAN_REL_DIFF:
        cmap_inv = mpl.colormaps[CMAP + "_r"]
        norm = mpl.colors.Normalize(vmin=np.nanmin(Z),
                                    vmax=np.nanmax(Z))
        ax.plot(span_lock_1, wind_speeds, ls="--", lw=1, c=cmap_inv(norm(0.0)))
        ax.plot(span_lock_2, wind_speeds, ls="--", lw=1, c=cmap_inv(norm(1.0)))

    if ABSCISSA_COORD is AbcissaType.RED_VEL:
        ax.set_xlim([0, RED_VEL_AXIS_MAX])

    cbar = fig.colorbar(fcm, ax=ax)
    cbar.set_label(SHEDDING_PLOT_MODE.value)

    ax.set_xlabel(ABSCISSA_COORD.value)
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title("Lock-in region estimate")

    fig.tight_layout()
    return fig


fig_shedding_mesh = plot_shedding_frequency_mesh(sheddings=sheddings)

fig_shedding_mesh_name = "shedding_frequency_mesh_" \
    + ABSCISSA_COORD.name.lower()

fig_shedding_mesh.savefig(OUTPUT_DIR / "spectral"
                          / (fig_shedding_mesh_name + ".png"),
                          dpi=150)
fig_shedding_mesh.savefig(OUTPUT_DIR / "spectral"
                          / (fig_shedding_mesh_name + ".svg"))

# =============================================================================
# Additional plots
# =============================================================================

if ABSCISSA_COORD in (AbcissaType.CHORD, AbcissaType.RED_VEL):
    # Plot characteristic length over span
    fig, ax = plt.subplots(figsize=(8, 6))

    if ABSCISSA_COORD is AbcissaType.CHORD:
        ordinate = abscissa
        y_label = "chord [m]"
        fig_chord_name = "chord_length_vs_span"
    else:
        ordinate = char_length
        y_label = CHAR_LENGTH.value + " [m]"
        fig_chord_name = CHAR_LENGTH.value.lower().replace(" ", "_") \
            + "_vs_span"

    ax.plot(span_pos, ordinate)
    ax.set_ylabel(y_label)
    ax.set_xlabel(AbcissaType.SPAN_POS.value)
    ax.set_title("Spanwise chord distribution")
    ax.grid(alpha=0.8)

    fig.savefig(OUTPUT_DIR / (fig_chord_name + ".png"), dpi=150)
    fig.savefig(OUTPUT_DIR / (fig_chord_name + ".svg"))

if ABSCISSA_COORD is AbcissaType.RED_VEL:
    # Plot reduced velocity over span
    fig, ax = plt.subplots(figsize=(8, 6))

    norm = mpl.colors.Normalize(vmin=min(WIND_SPEEDS),
                                vmax=max(WIND_SPEEDS))
    cmap = mpl.colormaps[CMAP]

    for U, red_vel in zip(WIND_SPEEDS, abscissa):
        color = cmap(norm(U))
        ax.plot(span_pos, red_vel, color=color, ls="-")

    ax.set_ylabel(AbcissaType.RED_VEL.value)
    ax.set_xlabel(AbcissaType.SPAN_POS.value)
    ax.set_title("Spanwise reduced velocity distribution")
    ax.grid(alpha=0.8)

    # Colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Wind speed [m/s]")

    fig.suptitle("Spanwise reduced velocity distribution")
    fig_red_vel_name = "red_vel_vs_span"
    fig.savefig(OUTPUT_DIR / (fig_red_vel_name + ".png"), dpi=150)
    fig.savefig(OUTPUT_DIR / (fig_red_vel_name + ".svg"))
