"""Instantaneous spanwise power at the power peak of the two dominant regions.

For every wind speed of a sweep:
  1. The time series are split based on a specified y-position.
  2. The regional power P_R(t) = int_region P(y, t) dy is computed.
  3. The time of the maximum of P_R(t) is found for each region.
  4. The spanwise power distribution P(y, t_peak) is extracted at that time.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viv_rom.io.cfd.cfd_reader import HyReader
from viv_rom.domain.frames import MotionAlignedFrame
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE


# =============================================================================
# User config
# =============================================================================
DATA_DIR = Path("./data/archive_1D")
N_SKIP = 200
PITCH = 85
INCLINATION = 50
AMPLITUDE = 1
F_FORCING = 0.624  # [Hz]
WIND_SPEEDS = np.arange(6, 23, 1)  # 6..22 m/s in 1 m/s steps

FIXED_SPLIT_Y = 80

# True: fold the window onto one forcing period (average over all periods) and
# take the peak of that mean cycle. False: use the raw time series of the
# window (the single highest peak over all periods).
PHASE_AVERAGE = True

SHARED_COLOR_SCALE = True  # same color limits for both regions
OUTPUT_DIR = Path(__file__).parent / "results" / "peak_power"
SAVE_FIGS = True

REGION_NAMES = ("region 1 (inboard)", "region 2 (outboard)")

# Diverging colormap (blue - black at zero - inferno-like)
cmap_diverging = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_black_inferno",
    ["#05bdfa", "#055be4", "#0c17bd", "#200677", "#000000",
     "#7f0000", "#d94801", "#fca50a", "#fafda1"], N=256)

case = f"a{AMPLITUDE:02d}_p{PITCH:02d}_i{INCLINATION:02d}"
OUTPUT_DIR = Path("./results") / case / "wind_sweep" / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def window_slice(t, f_forcing):
    """Slice of the samples covering an integer number of forcing periods."""
    n_periods = int(np.floor((t[-1] - t[0]) * f_forcing + 1e-9))
    if n_periods < 1:
        raise ValueError("Time series shorter than one forcing period")
    t_end = t[0] + n_periods / f_forcing
    idx_end = int(np.argmin(np.abs(t - t_end)))
    return slice(0, idx_end + 1), n_periods


def fourier_coeff(x, t, omega, k):
    """Complex amplitude c_k(y) such that x ~ Re[c_k exp(i k omega t)].

    x has shape (ny, nt). k = 0 returns the (real) time mean.
    """
    span = t[-1] - t[0]
    if k == 0:
        return np.trapezoid(x, t, axis=-1) / span
    return 2.0 / span * np.trapezoid(x * np.exp(-1j * k * omega * t), t,
                                     axis=-1)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def region_weights(y, lo, hi):
    """Weights w such that  int_lo^hi f dy = w @ f  (trapezoid, exact edges)."""
    w = np.zeros_like(y, dtype=float)
    for i in range(len(y) - 1):
        a, b = max(y[i], lo), min(y[i + 1], hi)  # Clip to interval limits
        if b <= a:
            continue
        dy = y[i + 1] - y[i]

        # Calculate fractional interval
        sa, sb = (a - y[i]) / dy, (b - y[i]) / dy

        # Calculate weight of left and right edge of interval
        w[i] += 0.5 * (b - a) * ((1 - sa) + (1 - sb))
        w[i + 1] += 0.5 * (b - a) * (sa + sb)
    return w


def analyse_speed(reader, wind_speed):
    data = reader.retrieve_force_field(amplitude=AMPLITUDE, pitch=PITCH,
                                       inclination=INCLINATION,
                                       wind_speed=int(wind_speed),
                                       n_skip=N_SKIP, n_last=None).data
    y, t = data.y, data.t
    omega = 2 * np.pi * F_FORCING

    vx = FIRST_EDGEWISE_MODE.velocity(component="x", y=y, t=t,
                                      amplitude=AMPLITUDE, omega=omega)
    vz = FIRST_EDGEWISE_MODE.velocity(component="z", y=y, t=t,
                                      amplitude=AMPLITUDE, omega=omega)
    frame = MotionAlignedFrame(FIRST_EDGEWISE_MODE)
    v_par = frame.transform(y, vx, vz)["fpar"]
    P = data.Fx * vx + data.Fz * vz  # instantaneous power [W/m]

    # Slice data into full period window
    sl, n_periods = window_slice(t, F_FORCING)
    tw, Pw = t[sl], P[:, sl]

    # Regions
    Pbar = fourier_coeff(Pw, tw, omega, 0).real  # Time mean sectional power
    Vh = fourier_coeff(v_par[:, sl], tw, omega, 1)
    v_amp = np.abs(Vh)

    regions = [(y[0], FIXED_SPLIT_Y), (FIXED_SPLIT_Y, y[-1])]

    # Cycle to search for the peaks
    if PHASE_AVERAGE:
        dt = np.median(np.diff(tw))
        nb = max(int(round(1 / (F_FORCING * dt))), 8)  # bins per period
        u = ((tw - tw[0]) * F_FORCING) % 1.0  # fractional position in forcing period
        ib = np.minimum((u * nb).astype(int), nb - 1)  # bin index of each sample

        # Calculate mean power over all samples in each bin
        Pc = np.full((len(y), nb), np.nan)
        for b in range(nb):
            m = ib == b
            if m.any():
                Pc[:, b] = Pw[:, m].mean(axis=1)

        # Calculate bin center times
        t_c = tw[0] + (np.arange(nb) + 0.5) / (nb * F_FORCING)
    else:
        Pc, t_c = Pw, tw

    # Phase of the motion: theta = 0 at maximum of the parallel velocity
    # (measured at the station with the largest velocity amplitude)
    phi_v = np.angle(Vh[int(np.argmax(v_amp))])
    theta_c = wrap_deg(np.degrees(omega * t_c + phi_v))

    # Peak time per region and spanwise power at that time
    out = {"y": y, "regions": np.array(regions), "n_periods": n_periods,
           "p_peak": np.zeros((2, len(y))), "t_peak": np.zeros(2),
           "theta": np.zeros(2), "P_R_peak": np.zeros(2),
           "P_R_mean": np.zeros(2)}
    for r, (lo, hi) in enumerate(regions):
        w = region_weights(y, lo, hi)
        P_R = w @ Pc  # regional power vs time [W]
        ip = int(np.nanargmax(P_R))
        out["p_peak"][r] = Pc[:, ip]
        out["t_peak"][r] = (t_c[ip] - tw[0]) if PHASE_AVERAGE else t_c[ip]
        out["theta"][r] = theta_c[ip]
        out["P_R_peak"][r] = P_R[ip]
        out["P_R_mean"][r] = w @ Pbar
    return out


# =============================================================================
# Sweep
# =============================================================================
reader = HyReader(DATA_DIR)
speeds = np.asarray(WIND_SPEEDS, dtype=float)
nU = len(speeds)

y_ref = None
results = [None] * nU
for i, U in enumerate(speeds):
    try:
        results[i] = analyse_speed(reader, U)
    except Exception as exc:  # keep the sweep going, leave a gap in the plot
        print(f"U = {U:g} m/s skipped: {exc!r}")
        continue
    if y_ref is None:
        y_ref = results[i]["y"]

if y_ref is None:
    raise SystemExit("No wind speed could be analysed")

p_peak = np.full((nU, 2, len(y_ref)), np.nan)
reg = np.full((nU, 2, 2), np.nan)
theta = np.full((nU, 2), np.nan)
t_peak = np.full((nU, 2), np.nan)
pr_peak = np.full((nU, 2), np.nan)
pr_mean = np.full((nU, 2), np.nan)
for i, res in enumerate(results):
    if res is None:
        continue
    for r in range(2):  # interpolate onto a common spanwise grid
        p_peak[i, r] = np.interp(y_ref, res["y"], res["p_peak"][r])
    reg[i] = res["regions"]
    theta[i] = res["theta"]
    t_peak[i] = res["t_peak"]
    pr_peak[i] = res["P_R_peak"]
    pr_mean[i] = res["P_R_mean"]
dtheta = np.abs(wrap_deg(theta[:, 1] - theta[:, 0]))

# =============================================================================
# Printed results
# =============================================================================
tlabel = "tau [s]" if PHASE_AVERAGE else "t [s]"
print("\nPeak of the regional power "
      f"({'phase-averaged cycle' if PHASE_AVERAGE else 'raw time series'}); "
      "theta = phase rel. to max. velocity")
print(f"{'U':>4} | {'region 1: y [m]':>16} {tlabel:>8} {'theta':>7} "
      f"{'P_R,pk [W]':>11} | {'region 2: y [m]':>16} {tlabel:>8} "
      f"{'theta':>7} {'P_R,pk [W]':>11} | {'dtheta':>7}")
for i, U in enumerate(speeds):
    if results[i] is None:
        print(f"{U:4g} | skipped")
        continue
    cells = []
    for r in range(2):
        cells.append(f"{reg[i, r, 0]:6.1f}-{reg[i, r, 1]:<8.1f} "
                     f"{t_peak[i, r]:8.3f} {theta[i, r]:7.1f} "
                     f"{pr_peak[i, r]:11.1f}")
    print(f"{U:4g} | {cells[0]} | {cells[1]} | {dtheta[i]:7.1f}")

# =============================================================================
# Plots
# =============================================================================
# Figure 1: spanwise power at the peak time of each region
vmin_all = min(np.nanmin(p_peak), -1e-12)
vmax_all = max(np.nanmax(p_peak), 1e-12)

fig1, axes = plt.subplots(ncols=2, sharey=True, figsize=(15, 6.5))
for r, ax in enumerate(axes):
    if SHARED_COLOR_SCALE:
        vmin, vmax = vmin_all, vmax_all
    else:
        vmin = min(np.nanmin(p_peak[:, r]), -1e-12)
        vmax = max(np.nanmax(p_peak[:, r]), 1e-12)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    mesh = ax.pcolormesh(y_ref, speeds, p_peak[:, r, :], shading="nearest",
                         cmap=cmap_diverging, norm=norm)
    # ax.plot(reg[:, r, 0], speeds, "--", color="lime", lw=1.2,
    #         label="region limits")
    # ax.plot(reg[:, r, 1], speeds, "--", color="lime", lw=1.2)
    ax.set_title(f"P(y) at the peak of {REGION_NAMES[r]}")
    ax.set_xlabel("y [m]")
    cbar = fig1.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("P [W/m]")
axes[0].set_ylabel("Wind speed [m/s]")
axes[0].legend(loc="lower left", fontsize="small")
fig1.tight_layout()

# Figure 2: when the peaks occur and how large they are
fig2, (axa, axb, axc) = plt.subplots(ncols=3, figsize=(16, 4.8))
for r in range(2):
    axa.plot(speeds, theta[:, r], "o-", label=REGION_NAMES[r])
    axc.plot(speeds, pr_peak[:, r], "o-", label=REGION_NAMES[r])
axa.set_ylabel("Peak phase [deg] (0 = max. velocity)")
axa.set_ylim(-190, 190)
axa.set_yticks(np.arange(-180, 181, 90))
axb.plot(speeds, dtheta, "ko-")
axb.axhline(180, color="0.5", ls="--", lw=0.8)
axb.set_ylabel("|Phase difference| of the peaks [deg]")
axb.set_ylim(0, 190)
axc.set_ylabel("Peak regional power [W]")
for a in (axa, axb, axc):
    a.set_xlabel("Wind speed [m/s]")
    a.grid(True, alpha=0.3)
axa.legend(fontsize="small")
axc.legend(fontsize="small")
fig2.tight_layout()

if SAVE_FIGS:
    fig1.savefig(OUTPUT_DIR / "peak_power_spanwise.png", dpi=150)
    fig1.savefig(OUTPUT_DIR / "peak_power_spanwise.svg")
    fig2.savefig(OUTPUT_DIR / "peak_phase.png", dpi=150)

plt.show()
