from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

from viv_rom.io.cfd_reader import HyReader
from viv_rom.domain.frames import MotionAlignedFrame
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE


# =============================================================================
# User config_animuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
PITCH = 85
INCLINATION = 50
WIND_SPEED = 18  # free-stream wind speed [m/s]
N_SKIP = 1200
ALIGN_FRAME = False
REMOVE_MEAN = False

# Load some data (to test)
reader = HyReader(DATA_DIR)
data = reader.retrieve_force_field(amplitude=1, pitch=PITCH,
                                   inclination=INCLINATION,
                                   wind_speed=WIND_SPEED,
                                   n_skip=N_SKIP, n_last=None).data

if ALIGN_FRAME:
    frame = MotionAlignedFrame(FIRST_EDGEWISE_MODE)
    forces_transformed = frame.transform(data.y, data.Fx, data.Fz)
    f1 = forces_transformed["fpar"]
    f2 = forces_transformed["fperp"]

    label_f1, label_f2 = frame.labels.values()
else:
    f1, f2 = data.Fx, data.Fz
    label_f1, label_f2 = "Fx", "Fz"

if REMOVE_MEAN:
    f1 -= f1.mean(axis=1)[:, None]
    f2 -= f2.mean(axis=1)[:, None]

# =============================================================================
# 2D contour plot
# =============================================================================
fig_cont, ax_cont = plt.subplots(figsize=(16, 7),
                                 ncols=2)
Y, T = np.meshgrid(data.y, data.t)

forces = (f1, f2)
labels = (label_f1, label_f2)
for idx_ax, ax in enumerate(ax_cont):
    CS = ax_cont[idx_ax].contourf(T, Y, forces[idx_ax].T, 50, cmap="inferno")
    # CS2 = ax_cont[idx_ax].contour(CS, levels=CS.levels[::2], colors='r')

    ax_cont[idx_ax].set_title(labels[idx_ax])
    ax_cont[idx_ax].set_xlabel("Time [s]")
    ax_cont[idx_ax].set_ylabel("y [m]")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig_cont.colorbar(CS, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(labels[idx_ax] + " [N/m]")
    # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)
plt.subplots_adjust(left=0.05, right=0.95,
                    top=0.95, bottom=0.1,
                    wspace=0.25)

# =============================================================================
# Animation
# =============================================================================
fig_anim, (ax1_anim, ax2_anim) = plt.subplots(figsize=(10, 8), nrows=2)

line_f1, = ax1_anim.plot([], [], 'k')
line_f2, = ax2_anim.plot([], [], 'k')

title = ax1_anim.text(0.5, 0.85, "",
                      bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                      transform=ax1_anim.transAxes, ha="center")

ax1_anim.set_xlim(0, 100)
ax2_anim.set_xlim(0, 100)
ax1_anim.set_ylim(f1.min(), f1.max())
ax2_anim.set_ylim(f2.min(), f2.max())
ax1_anim.set_xlabel('y [m]')
ax2_anim.set_xlabel('y [m]')
ax1_anim.set_ylabel(label_f1 + " [N/m]")
ax2_anim.set_ylabel(label_f2 + " [N/m]")
ax1_anim.grid(True)
ax2_anim.grid(True)


def update(frame):
    line_f1.set_data(data.y, f1[:, frame])
    line_f2.set_data(data.y, f2[:, frame])

    title.set_text(f"t = {data.t[frame]:.2f} s")

    return line_f1, line_f2, title


ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(data.t),
    interval=10,
    blit=True
)

plt.show()
