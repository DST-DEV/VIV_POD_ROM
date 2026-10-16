from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from viv_rom.io.cfd_reader import HyReader
from viv_rom.domain.frames import MotionAlignedFrame
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE


class PlotMetric(Enum):
    FORCE = "force"
    POWER = "power"


# =============================================================================
# User config_animuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
PITCH = 85
INCLINATION = 50
WIND_SPEED = 10  # free-stream wind speed [m/s]
AMPLITUDE = 1  # Forced motion amplitude factor
F_FORCING = 0.624  # [Hz]
N_SKIP = 0

ALIGN_FRAME = True
REMOVE_MEAN = False
PLOTMETRIC = PlotMetric.POWER

# Load some data (to test)
reader = HyReader(DATA_DIR)
data = reader.retrieve_force_field(amplitude=1, pitch=PITCH,
                                   inclination=INCLINATION,
                                   wind_speed=WIND_SPEED,
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

    lines_unit = "N/m"
elif PLOTMETRIC is PlotMetric.POWER:
    vx = FIRST_EDGEWISE_MODE.velocity(component="x", y=data.y, t=data.t,
                                      amplitude=AMPLITUDE,
                                      omega=2*np.pi*F_FORCING)
    vz = FIRST_EDGEWISE_MODE.velocity(component="z", y=data.y, t=data.t,
                                      amplitude=AMPLITUDE,
                                      omega=2*np.pi*F_FORCING)

    if ALIGN_FRAME:
        line_data1 = data.Fx * vx + data.Fz * vz
        line_data2 = np.zeros_like(line_data1)

        label_1, label_2 = "P∥", "P⊥"
    else:
        line_data1 = data.Fx * vx
        line_data2 = data.Fz * vz

        label_1, label_2 = "Px", "Pz"

    lines_unit = "W/m"
else:
    raise ValueError("Invalid plot metric selected")

if REMOVE_MEAN:
    line_data1 -= line_data1.mean(axis=1)[:, None]
    line_data2 -= line_data2.mean(axis=1)[:, None]

# =============================================================================
# 2D contour plot
# =============================================================================
fig_cont, ax_cont = plt.subplots(figsize=(16, 7),
                                 ncols=2)
Y, T = np.meshgrid(data.y, data.t)

line_data = (line_data1, line_data2)
labels = (label_1, label_2)
for idx_ax, ax in enumerate(ax_cont):
    CS = ax_cont[idx_ax].contourf(T, Y, line_data[idx_ax].T, 50,
                                  cmap="inferno")
    # CS2 = ax_cont[idx_ax].contour(CS, levels=CS.levels[::2], colors='r')

    ax_cont[idx_ax].set_title(labels[idx_ax])
    ax_cont[idx_ax].set_xlabel("Time [s]")
    ax_cont[idx_ax].set_ylabel("y [m]")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig_cont.colorbar(CS, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(labels[idx_ax] + " " + lines_unit)
    # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)
plt.subplots_adjust(left=0.05, right=0.95,
                    top=0.95, bottom=0.1,
                    wspace=0.25)

# =============================================================================
# Animation
# =============================================================================
fig_anim, (ax1_anim, ax2_anim) = plt.subplots(figsize=(10, 8), nrows=2)

lline_1, = ax1_anim.plot([], [], 'k')
lline_2, = ax2_anim.plot([], [], 'k')

title = ax1_anim.text(0.5, 0.85, "",
                      bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                      transform=ax1_anim.transAxes, ha="center")

ax1_anim.set_xlim(0, 100)
ax2_anim.set_xlim(0, 100)
ax1_anim.set_ylim(line_data1.min(), line_data1.max())
ax2_anim.set_ylim(line_data2.min(), line_data2.max())
ax1_anim.set_xlabel('y [m]')
ax2_anim.set_xlabel('y [m]')
ax1_anim.set_ylabel(label_1 + " [N/m]")
ax2_anim.set_ylabel(label_2 + " [N/m]")
ax1_anim.grid(True)
ax2_anim.grid(True)


def update(frame):
    lline_1.set_data(data.y, line_data1[:, frame])
    lline_2.set_data(data.y, line_data2[:, frame])

    title.set_text(f"t = {data.t[frame]:.2f} s")

    return lline_1, lline_2, title


ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(data.t),
    interval=10,
    blit=True
)

plt.show()
