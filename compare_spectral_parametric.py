from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.pipeline import analyse
from viv_rom.spectral.forcing_response import ForcingResponse

# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see README
WIND_SPEEDS = list(range(6, 23))  # free-stream wind speed [m/s] to analyse (6-22)
FIELD = "both"
FRAME = "motion"
NORMALIZE_WIND_SPEED = True
UNWRAP_PHASE = True
CMAP = "magma"
OUTPUT_DIR = Path("./results") / CASE / "wind_sweep"  # Output folder
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Run spectral analysis
# =============================================================================

results = {}
for wind_speed in WIND_SPEEDS:
    config = SpectralConfig.from_case(case=CASE, wind_speed=wind_speed,
                                      data_dir=DATA_DIR, field=FIELD,
                                      frame=FRAME, unwrap_phase=UNWRAP_PHASE)
    results[wind_speed] = analyse(config)


# =============================================================================
# Plot
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

    for U in wind_speeds:
        response = responses[U]
        h = int(np.flatnonzero(response.harmonics == harmonic)[0])
        color = cmap(norm(U))

        amplitude = response.amplitude[comp][:, h]
        if NORMALIZE_WIND_SPEED:
            amplitude = amplitude / U**2

        ax1.plot(
            response.y,
            amplitude,
            color=color,
            lw=2,
        )

        ax2.plot(
            response.y,
            response.phase_deg(comp)[:, h],
            ".-",
            ms=3,
            color=color,
        )

    # Use first response to obtain harmonic frequency
    first_response = responses[wind_speeds[0]]
    h = int(np.flatnonzero(first_response.harmonics == harmonic)[0])
    f_h = first_response.frequencies[h]

    ax1.set_xlabel("Spanwise station y [m]")
    y_unit1 = "N/m" if not NORMALIZE_WIND_SPEED else "N/m * (m/s)^-2"
    ax1.set_ylabel("Amplitude [" + y_unit1 + "]")
    ax1.set_title(f"Amplitude at {harmonic} × f_forcing ({f_h:.3g} Hz)")
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

    ax2.set_xlabel("Spanwise station y [m]")
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
fig_force_response_name = "force_response"
fig_force_response.savefig(OUTPUT_DIR / (fig_force_response_name + ".png"),
                           dpi=150)
fig_force_response.savefig(OUTPUT_DIR / (fig_force_response_name + ".svg"))
