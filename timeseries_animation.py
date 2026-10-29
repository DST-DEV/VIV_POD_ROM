from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from viv_rom.io.cfd.cfd_reader import HyReader
from viv_rom.domain.frames import MotionAlignedFrame
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE


class PlotMetric(Enum):
    FORCE = "force"
    POWER = "power"


class AbscissaMetric(Enum):
    TIME = "Time [s]"
    PERIOD = "Forcing period"


@dataclass
class Case:
    """One time series, fully specified by the parameters of a .1HY file."""
    pitch: float
    inclination: float
    wind_speed: float  # free-stream wind speed [m/s]
    amplitude: float = 1  # forced motion amplitude factor
    f_forcing: float = .624  # [Hz]
    label: str | None = None  # legend label (auto-generated if None)

    def __post_init__(self):
        if self.label is None:
            self.label = (f"A={self.amplitude}, pitch={self.pitch}, "
                          f"incl={self.inclination}, U={self.wind_speed}, "
                          f"f={self.f_forcing}")


# Define diverging colormap (only used if CENTER_CONTOURS_AT_ZERO = True)
colors = [
    "#05bdfa",  # light blue
    "#055be4",  # vibrant blue
    "#0c17bd",  # deep vibrant blue
    "#200677",  # deep blue
    "#000000",  # black at zero
    "#7f0000",  # dark red
    "#d94801",  # orange-red
    "#fca50a",  # orange
    "#fafda1",  # inferno-like yellow
]

cmap_diverging = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_black_inferno", colors, N=256)

# =============================================================================
# User config
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
N_SKIP = 200

# Add one Case per line to be shown in each axis
F_FORCING = .624  # [Hz]
CASES = [
    Case(pitch=85, inclination=50, wind_speed=6, amplitude=1,
         f_forcing=F_FORCING),
    Case(pitch=85, inclination=50, wind_speed=10, amplitude=1,
         f_forcing=F_FORCING),
    Case(pitch=85, inclination=50, wind_speed=18, amplitude=1,
         f_forcing=F_FORCING),
    Case(pitch=85, inclination=50, wind_speed=22, amplitude=1,
         f_forcing=F_FORCING),
]

ALIGN_FRAME = True
REMOVE_MEAN = False
NORMALIZE_ANIMATION = True
PLOTMETRIC = PlotMetric.POWER
ABSCISSA_METRIC = AbscissaMetric.TIME
PLOT_CONTOURS = True  # one contour figure (2 axes) per case
N_LEVELS = 50  # Number of levels for the contour plot
CENTER_CONTOURS_AT_ZERO = True  # Whether to center the colormap for the contours at zero
CONTOURS_EQUAL_LIMS = False  # Whether the positive and negative cbar lims of the contour plot should be equal
OUTPUT_DIR = Path(__file__).parent / "results"
SAVE_FIGS = True


# =============================================================================
# Data loading
# =============================================================================
def compute_case(reader, case):
    """Return (data, line_data1, line_data2, label_1, label_2, unit)."""
    data = reader.retrieve_force_field(amplitude=case.amplitude,
                                       pitch=case.pitch,
                                       inclination=case.inclination,
                                       wind_speed=case.wind_speed,
                                       n_skip=N_SKIP, n_last=None).data

    if PLOTMETRIC is PlotMetric.FORCE:
        if ALIGN_FRAME:
            frame = MotionAlignedFrame(FIRST_EDGEWISE_MODE)
            forces_transformed = frame.transform(data.y, data.Fx, data.Fz)
            line_data1 = forces_transformed["fpar"]
            line_data2 = forces_transformed["fperp"]

            label_1, label_2 = frame.labels.values()
        else:
            line_data1, line_data2 = data.Fx, data.Fz
            label_1, label_2 = "Fx", "Fz"

        unit = "N/m"
    elif PLOTMETRIC is PlotMetric.POWER:
        omega = 2 * np.pi * case.f_forcing
        vx = FIRST_EDGEWISE_MODE.velocity(component="x", y=data.y, t=data.t,
                                          amplitude=case.amplitude,
                                          omega=omega)
        vz = FIRST_EDGEWISE_MODE.velocity(component="z", y=data.y, t=data.t,
                                          amplitude=case.amplitude,
                                          omega=omega)

        if ALIGN_FRAME:
            line_data1 = data.Fx * vx + data.Fz * vz
            line_data2 = np.zeros_like(line_data1)

            label_1, label_2 = "P∥", "P⊥"
        else:
            line_data1 = data.Fx * vx
            line_data2 = data.Fz * vz

            label_1, label_2 = "Px", "Pz"

        unit = "W/m"
    else:
        raise ValueError("Invalid plot metric selected")

    if REMOVE_MEAN:
        # Not in-place, so the arrays stored in `data` are never modified
        line_data1 = line_data1 - line_data1.mean(axis=1)[:, None]
        line_data2 = line_data2 - line_data2.mean(axis=1)[:, None]

    return data, line_data1, line_data2, label_1, label_2, unit


reader = HyReader(DATA_DIR)
results = [compute_case(reader, case) for case in CASES]

# Labels/units are identical for all cases (they only depend on the metric)
label_1, label_2, lines_unit = results[0][3:]
labels = (label_1, label_2)

# One distinct color per case
colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(CASES))]

# =============================================================================
# 2D contour plots (one figure per case)
# =============================================================================
if PLOT_CONTOURS:
    for case, (data, ld1, ld2, *_) in zip(CASES, results):
        if PLOTMETRIC is PlotMetric.POWER and ALIGN_FRAME:
            # Power perpendicular to motion is always zero => second subplot
            # omitted
            fig_cont, ax_cont = plt.subplots(figsize=(8, 7))
            ax_cont = [ax_cont,]
        else:
            fig_cont, ax_cont = plt.subplots(figsize=(16, 7), ncols=2)
        fig_cont.suptitle(case.label)
        if ABSCISSA_METRIC is AbscissaMetric.TIME:
            Y, X = np.meshgrid(data.y, data.t)
            xticks = None
        elif ABSCISSA_METRIC is AbscissaMetric.PERIOD:
            period_len = 1 / F_FORCING
            period_fraction = data.t / period_len
            Y, X = np.meshgrid(data.y, period_fraction)
            xticks = np.arange(period_fraction[0], period_fraction[-1]+.1, 1)
        else:
            raise ValueError("Invalid abscissa metric.")

        line_data = (ld1, ld2)
        for idx_ax, ax in enumerate(ax_cont):
            ax.set_title(labels[idx_ax])

            Z = line_data[idx_ax].T
            if np.nanmin(Z) == np.nanmax(Z):
                continue

            vmin = np.floor(np.nanmin(Z) / 10) * 10
            vmax = np.ceil(np.nanmax(Z) / 10) * 10
            if CENTER_CONTOURS_AT_ZERO:
                if CONTOURS_EQUAL_LIMS:
                    vmax = np.nanmax([np.abs(vmin), np.abs(vmax)])
                    vmin = -vmax

                norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                cmap_contourf = cmap_diverging
            else:
                norm = None
                cmap_contourf = "inferno"
            levels = np.linspace(vmin, vmax, N_LEVELS+1)
            CS = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_contourf,
                             norm=norm)

            ax.set_xlabel(ABSCISSA_METRIC.value)
            if xticks is not None:
                ax.set_xticks(xticks)
            ax.set_ylabel("y [m]")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            cbar = fig_cont.colorbar(CS, cax=cax, orientation='vertical')
            cbar.ax.set_ylabel(labels[idx_ax] + " " + lines_unit)

            # Set ticks for colorbar
            cbar_mag_log = np.log10(max(abs(vmin), abs(vmax)))
            if cbar_mag_log-int(cbar_mag_log) >= .35:
                cbar_tick_step = 10**int(cbar_mag_log)
            else:
                cbar_tick_step = 2 * 10**(int(cbar_mag_log) - 1)  # 20 steps

            cbar_ticks = np.arange(
                np.ceil(vmin / cbar_tick_step) * cbar_tick_step,
                np.floor(vmax / cbar_tick_step) * cbar_tick_step + 1e-3,
                cbar_tick_step)
            cbar.set_ticks(cbar_ticks)

        if len(ax_cont) == 1:
            fig_cont.subplots_adjust(left=0.1, right=0.9,
                                     top=0.92, bottom=0.1,
                                     wspace=0.25)
        else:
            fig_cont.subplots_adjust(left=0.05, right=0.95,
                                     top=0.92, bottom=0.1,
                                     wspace=0.25)

        if SAVE_FIGS:
            case_dir = OUTPUT_DIR / (f"a{case.amplitude:02d}"
                                     + f"_p{case.pitch:02d}"
                                     + f"_i{case.inclination:02d}") \
                                     / f"u{case.wind_speed:02d}" / "data"
            case_dir.mkdir(exist_ok=True)
            fig_cont_name = "Hovmoller_" + PLOTMETRIC.value
            fig_cont.savefig(case_dir / (fig_cont_name + ".png"), dpi=150)
            fig_cont.savefig(case_dir / (fig_cont_name + ".svg"))

# =============================================================================
# Animation
# =============================================================================
fig_anim, (ax1_anim, ax2_anim) = plt.subplots(figsize=(10, 8), nrows=2)

if NORMALIZE_ANIMATION:
    norm_factors_1 = [np.abs(r[1]).max() or 1 for r in results]
    norm_factors_2 = [np.abs(r[2]).max() or 1 for r in results]
else:
    norm_factors_1, norm_factors_2 = [[1.0,] * len(results)] * 2

lines_1, lines_2 = [], []
for case, color in zip(CASES, colors):
    l1, = ax1_anim.plot([], [], color=color, label=case.label)
    l2, = ax2_anim.plot([], [], color=color, label=case.label)
    lines_1.append(l1)
    lines_2.append(l2)

title = ax1_anim.text(0.5, 0.85, "",
                      bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                      transform=ax1_anim.transAxes, ha="center")

# Axis limits cover all cases
span_min = min(r[0].y.min() for r in results)
span_max = max(r[0].y.max() for r in results)
for ax, idx, norm in ((ax1_anim, 1, norm_factors_1),
                      (ax2_anim, 2, norm_factors_2)):
    lo = min(r[idx].min()/n for r, n in zip(results, norm))
    hi = max(r[idx].max()/n for r, n in zip(results, norm))
    if lo == hi:  # e.g. the all-zero P⊥ line: avoid a degenerate axis
        lo, hi = lo - 1, hi + 1
    ax.set_xlim(span_min, span_max)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('y [m]')
    ax.grid(True)
ax1_anim.set_ylabel(f"{label_1} [{lines_unit}]")
ax2_anim.set_ylabel(f"{label_2} [{lines_unit}]")
ax1_anim.legend(loc="upper right", fontsize="small")

# In case the number of time steps doesn't match: animate up to the shortest
n_frames = min(len(r[0].t) for r in results)


def update(frame):
    for idx, (res, l1, l2) in enumerate(zip(results, lines_1, lines_2)):
        data, ld1, ld2 = res[:3]
        l1.set_data(data.y, ld1[:, frame]/norm_factors_1[idx])
        l2.set_data(data.y, ld2[:, frame]/norm_factors_2[idx])

    # Time of the first case is shown (cases are stepped frame by frame)
    title.set_text(f"t = {results[0][0].t[frame]:.2f} s")

    return (*lines_1, *lines_2, title)


ani = FuncAnimation(
    fig_anim,
    update,
    frames=n_frames,
    interval=10,
    blit=True
)

plt.show()
