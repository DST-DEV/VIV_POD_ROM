from pathlib import Path
from typing import Mapping
from enum import Enum

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.pipeline import analyse
from viv_rom.spectral.forcing_response import ForcingResponse
from viv_rom.domain.blade import BladeGeometry
from viv_rom.io.results.export import load_result


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
CHAR_LENGTH = CharacteristicLength.PROJECTED_CHORD
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
    # ax1.set_xlim([0, min(np.max(abscissa), 100)])
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
