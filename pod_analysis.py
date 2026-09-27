"""
pod_analysis.py

Proper Orthogonal Decomposition (POD) of the spanwise sectional-force
distribution on an oscillating wind-turbine blade (vortex-induced-vibration
CFD sweep, EllipSys3D).

POD is computed via the "direct method" (economy SVD of the fluctuation
snapshot matrix), using numpy.linalg.svd. No external POD/SPOD library is
needed (for now):

    D'(y, t) = D(y, t) - mean_t[D(y, t)]           (fluctuation field)
    D' = Phi @ diag(sigma) @ Psi^T                  (economy SVD)

    Phi[:, k]  -> k-th spatial POD mode  (unit norm)
    sigma[k]   -> k-th singular value
    a_k(t) = sigma[k] * Psi[:, k]  -> k-th temporal coefficient
    sigma[k]**2 / sum(sigma**2)    -> fraction of fluctuation energy in mode k
"""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import Dict, Optional, Sequence, Tuple, Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless / batch runs
import matplotlib.pyplot as plt  # noqa: E402

# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the data
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see data README
U_WIND = 10  # free-stream wind speed [m/s] to analyse (6-22)

# Selection for which sectional force component(s) to consider in the POD.
# "fx", "fz", or "combined"
FIELD = "fx"

# Rows (time steps) dropped as initial transient (Note: 200 is one full period)
N_SKIP = 1200
# Rows to keep after N_SKIP. If None: keep everything (recommended: run POD
# over several periods, not just the last one, so lock-in / beating behaviour
# is visible).
N_LAST = None

N_MODES_SHOW = 4  # number of leading POD modes to plot/report
F_FORCING = 0.624  # [Hz] prescribed excitation frequency (cf. README)
AMPLITUDE = 1.0  # motion amplitude factor A (cf. README, case a01)

DO_POWER_CHECK = True  # optional physical cross-check, see section 5

OUTPUT_DIR = Path(__file__).parent / "results" / "pod_test_single"

# Motion polynomials from the README (highest power first), evaluated at
# (y - 2.8)
COEFF_X = np.array([-1.63363204579525e-10, 3.98744338033108e-08,
                    -2.55298610905188e-06, 1.36855887568269e-04,
                    -8.05337508940684e-04, 9.35366136110291e-04])
COEFF_Z = np.array([-7.37789434610509e-14, 1.96005460184524e-11,
                    -1.94444451601558e-09, 9.84504021700715e-08,
                    -2.59892457459754e-06, 2.09320617905598e-05,
                    -5.07715323316064e-05, 1.81743886489348e-06])


# =============================================================================
# Data loading
# =============================================================================
def load_hy(path: PurePath, n_skip: int = 0, n_last: None | int = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a *.1HY sectional-force file.

    Expected file layout (see README of simulations):
    3000 rows x (2 + 4*n_sections) columns, whitespace separated, 3 leading
    '#' comment lines.
        col 0         nstep
        col 1         time      [s]
        col 2 + 4*j   x[j]      [m]  spanwise station (constant in time)
        col 3 + 4*j   fx[j]     [N/m] sectional force, edgewise
        col 4 + 4*j   fy[j]     [N/m] unused
        col 5 + 4*j   fz[j]     [N/m] sectional force, flapwise

    Parameters
    ----------
    path : PurePath
        Path to the .1HY file
    n_skip : int, optional
        Number of rows to skip at the beginning of the file, by default 0
    n_last : _type_, optional
        Number or rows to keep after n_skip. If None, keep all rows.
        Default is None

    Returns
    -------
    y : ndarray of shape (n_sections,)
        Spanwise coordinate [m]
    t : (n_rows,)
        Time [s] (absolute simulation time)
    Fx : ndarray of shape (n_rows, n_sections)
        Edgewise sectional force [N/m]
    Fz : ndarray of shape (n_rows, n_sections)
        Flapwise sectional force [N/m]
    """
    data = np.loadtxt(path, comments="#")
    if n_skip:
        data = data[n_skip:]
    if n_last:
        data = data[-n_last:]

    n_sections = (data.shape[1] - 2) // 4
    j = np.arange(n_sections)

    y = data[0, 2 + 4 * j]
    t = data[:, 1]
    Fx = data[:, 3 + 4 * j]
    Fz = data[:, 5 + 4 * j]
    return y, t, Fx, Fz


def build_field_matrix(Fx, Fz, mode):
    """Build the (n_snapshots, n_points) matrix POD will be applied to.

    Parameters
    ----------
    Fx : ndarray of shape (n_rows, n_sections)
        The force time series in x-direction.
    Fz : ndarray of shape (n_rows, n_sections)
        The force time series in z-direction.
    mode : str
        Which force time series should be considered
        - "fx" or "fz": that component alone, n_points = n_sections
        - "combined":  Fx and Fz concatenated along the spatial axis, each
          individually scaled by its own overall standard deviation so that
          both components contribute on a comparable footing.
          n_points = 2 * n_sections.

    Returns
    -------
    ndarray
        The field array plus a small dict describing how to interpret / invert
        the scaling when a "combined" spatial mode is split back into its
        Fx- and Fz- parts for plotting.
    """
    if mode == "fx":
        return Fx, {"kind": "single", "component": "Fx"}
    elif mode == "fz":
        return Fz, {"kind": "single", "component": "Fz"}
    elif mode == "combined":
        sx, sz = Fx.std(), Fz.std()
        field = np.hstack([Fx / sx, Fz / sz])
        return field, {"kind": "combined", "n": Fx.shape[1],
                       "scale_x": sx, "scale_z": sz}
    else:
        raise ValueError(f"Unknown FIELD option: {mode!r} (use 'fx', 'fz', "
                         "or 'combined')")


# =============================================================================
# POD via SVD
# =============================================================================
def compute_pod(field: np.ndarray) -> dict:
    """Proper Orthogonal Decomposition of a snapshot set via the direct SVD
    method.

    Parameters
    ----------
    field : ndarray of shape (n_snapshots, n_points)
        Snapshot matrix. Rows are time snapshots, columns are spatial points
        (e.g. spanwise stations)

    Returns
    -------
    dict
        Decomposed snapshot matrix as a dict with fields:
        - mean : (n_points,)
            temporal mean field, subtracted before SVD
        - modes : (n_points, n_modes)
            spatial POD modes (orthonormal columns)
        - sigma : (n_modes,)
            singular values, descending
        - energy_frac : (n_modes,)
            sigma_k^2 / sum(sigma^2)
        - cum_energy : (n_modes,)
            cumulative energy fraction
        - time_coeff : (n_modes, n_snapshots)
            Time coefficient matrix (a_k(t) = sigma_k * Psi[:, k])
    """
    D = np.asarray(field, dtype=float).T  # (n_points, n_snapshots)
    mean_field = D.mean(axis=1, keepdims=True)
    D_fluc = D - mean_field

    # Economy SVD:  D_fluc = Phi @ diag(sigma) @ PsiT
    Phi, sigma, PsiT = np.linalg.svd(D_fluc, full_matrices=False)
    # Equivalent implementation:
    # Phi, sigma, PsiT = scipy.linalg.svd(D_fluc, full_matrices=False)

    energy_frac = sigma**2 / np.sum(sigma**2)
    cum_energy = np.cumsum(energy_frac)
    time_coeff = sigma[:, None] * PsiT  # (n_modes, n_snapshots)

    return {
        "mean": mean_field.ravel(),
        "modes": Phi,
        "sigma": sigma,
        "energy_frac": energy_frac,
        "cum_energy": cum_energy,
        "time_coeff": time_coeff,
    }


def reconstruct(pod: dict, r: int):
    """Reconstruct the field using the mean plus the leading r POD modes.

    Parameters
    ----------
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    r : int
        Number of leading modes to use for the reconstruction

    Returns
    -------
    ndarray
        Reconstructed field using the r-truncated POD basis
    """
    Phi_r = pod["modes"][:, :r]
    a_r = pod["time_coeff"][:r, :]
    return pod["mean"][:, None] + Phi_r @ a_r  # (n_points, n_snapshots)


def dominant_frequency(signal: np.ndarray, dt: float) -> tuple:
    """Return the frequency of the largest non-zero peak in the absolute FFT

    Parameters
    ----------
    signal : ndarray
        Signal to analyse
    dt : float
        Time step of the signal

    Returns
    -------
    dominant_freq : float
        Dominant non-zero frequency in the signal
    freqs : ndarray
        FFT of the signal
    amp : ndarray
        Amplitude of the reverse FFT of the signal
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(signal - signal.mean()))
    if len(amp) < 2:
        return 0.0, freqs, amp
    k = np.argmax(amp[1:]) + 1  # skip the DC bin
    return freqs[k], freqs, amp


# =============================================================================
# Plotting
# =============================================================================
def plot_energy_spectrum(pod: dict, out_dir: PurePath, n_show: int = 20
                         ) -> None:
    """
    Plot the energy distribution of the leading POD modes.

    Generates two figures: the energy fraction captured by each mode and the
    cumulative energy captured by the leading modes. The figure is saved to
    ``01_energy_spectrum.png``.

    Parameters
    ----------
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    out_dir : pathlib.PurePath
        Directory where the figure will be written.
    n_show : int, optional
        Maximum number of POD modes to display. If greater than the available
        number of modes, all modes are shown. Default is 20.

    Returns
    -------
    None
    """
    n_show = min(n_show, len(pod["sigma"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    modes = np.arange(1, n_show + 1)
    ax1.bar(modes, 100 * pod["energy_frac"][:n_show])
    ax1.set_xlabel("POD mode")
    ax1.set_ylabel("Energy fraction [%]")
    ax1.set_title("Energy per mode")

    ax2.plot(modes, 100 * pod["cum_energy"][:n_show], "o-")
    ax2.axhline(99, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Number of modes")
    ax2.set_ylabel("Cumulative energy [%]")
    ax2.set_title("Cumulative energy")

    fig.tight_layout()
    fig.savefig(out_dir / "01_energy_spectrum.png", dpi=150)
    plt.close(fig)


def plot_spatial_modes(y: np.ndarray, pod: dict, field_info: dict,
                       out_dir: PurePath, n_show: int = 4) -> None:
    """
    Plot the leading spatial POD modes.

    For a single force component, the leading spatial modes are plotted on a
    single axis. For a combined Fx/Fz decomposition, the spatial modes are
    split back into their respective components using the stored scaling
    factors and displayed on separate axes.

    Parameters
    ----------
    y : ndarray of shape (n_sections,)
        Spanwise station coordinates.
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    field_info : dict
        Metadata returned by :func:`build_field_matrix` describing whether the
        POD was performed on a single component or on the combined field.
    out_dir : pathlib.PurePath
        Directory where the figure will be saved.
    n_show : int, optional
        Number of leading modes to plot. Default is 4.

    Returns
    -------
    None
    """
    n_show = min(n_show, pod["modes"].shape[1])
    fig, ax = plt.subplots(figsize=(7, 5))

    if field_info["kind"] == "single":
        for k in range(n_show):
            ax.plot(y, pod["modes"][:, k],
                    label=f"mode {k+1} ({100*pod['energy_frac'][k]:.1f}%)")
        ax.set_ylabel(f"{field_info['component']} mode shape [-]")
    else:  # combined
        n = field_info["n"]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
        for k in range(n_show):
            phi = pod["modes"][:, k]
            lbl = f"mode {k+1} ({100*pod['energy_frac'][k]:.1f}%)"
            ax.plot(y, phi[:n] * field_info["scale_x"], label=lbl)
            ax2.plot(y, phi[n:] * field_info["scale_z"], label=lbl)
        ax2.set_xlabel("Spanwise station y [m]")
        ax2.set_ylabel("Fz-part mode shape [N/m, rescaled]")
        ax2.set_title("Flapwise (Fz) component")
        ax2.legend(fontsize=8)
        ax.set_ylabel("Fx-part mode shape [N/m, rescaled]")
        ax.set_title("Edgewise (Fx) component")

    ax.set_xlabel("Spanwise station y [m]")
    ax.set_title("Leading POD spatial modes"
                 if field_info["kind"] == "single" else ax.get_title())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_spatial_modes.png", dpi=150)
    plt.close(fig)


def plot_temporal_coefficients(t: np.ndarray, pod: dict, dt: float,
                               f_forcing: float, out_dir: PurePath,
                               n_show: int = 4) -> None:
    """
    Plot POD temporal coefficients and their frequency spectra.

    For each leading POD mode, the temporal coefficient is plotted together
    with its FFT magnitude spectrum. The prescribed forcing frequency and the
    dominant spectral peak are highlighted.

    Parameters
    ----------
    t : ndarray of shape (n_snapshots,)
        Time vector.
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    dt : float
        Constant sampling interval.
    f_forcing : float
        Prescribed excitation frequency in Hz.
    out_dir : pathlib.PurePath
        Directory where the figure will be saved.
    n_show : int, optional
        Number of leading modes to visualize. Default is 4.

    Returns
    -------
    None
    """
    n_show = min(n_show, pod["time_coeff"].shape[0])
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.4 * n_show),
                             squeeze=False)

    for k in range(n_show):
        a_k = pod["time_coeff"][k, :]
        f_dom, freqs, amp = dominant_frequency(a_k, dt)

        axes[k, 0].plot(t, a_k)
        axes[k, 0].set_ylabel(f"a_{k+1}(t)")
        axes[k, 0].set_title(f"Mode {k+1} time coefficient "
                             f"({100*pod['energy_frac'][k]:.1f}% energy)")

        axes[k, 1].plot(freqs, amp)
        axes[k, 1].axvline(f_forcing, color="r", ls="--", lw=1,
                           label="Forcing f")
        axes[k, 1].axvline(f_dom, color="g", ls=":", lw=1.5,
                           label="Dominant peak")
        axes[k, 1].set_xlim(0, max(6 * f_forcing, 2 * f_dom, 1e-6))
        axes[k, 1].set_title(f"Spectrum (peak = {f_dom:.3f} Hz, "
                             f"{f_dom / f_forcing:.2f} x forcing)")
        axes[k, 1].legend(fontsize=7)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    fig.savefig(out_dir / "03_temporal_coefficients.png", dpi=150)
    plt.close(fig)


def plot_reconstruction(y: np.ndarray, t: np.ndarray, D_full: np.ndarray,
                        pod: dict, station: int = 0,
                        r_values: Iterable[int] = (1, 2, 4),
                        out_dir: Optional[Path] = None
                        ) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
    """
    Compare POD reconstructions using different truncation levels.

    The reconstruction quality is assessed by comparing the original field
    with truncated POD reconstructions. Two comparisons are shown: the time
    history at the specified spanwise position and the spanwise distribution
    at the final snapshot.


    Parameters
    ----------
    y : ndarray of shape (n_sections,)
        Spanwise station coordinates.
    t : ndarray of shape (n_snapshots,)
        Time vector.
    D_full : ndarray of shape (n_snapshots, n_sections)
        Original field used for comparison.
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    station : int, optional
        Index of the spanwise station to plot the orbit for. Default is
        ``0`` (the blade root).
    r_values : iterable of int, optional
        Numbers of POD modes used for the truncated reconstructions.
        Default is ``(1, 2, 4)``.
    out_dir : pathlib.Path, optional
        Directory the figure is saved to. If ``None``, the
        figure is not saved to disk (only returned).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : Tuple[matplotlib.axes.Axes]
        The axes containing the orbit plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(t, D_full.T[station, :], "k", lw=1.5, label="original")
    ax2.plot(y, D_full.T[:, -1], "k", lw=1.5, label="original")

    for r in r_values:
        D_rec = reconstruct(pod, r)
        ax1.plot(t, D_rec[station, :], lw=1, label=f"{r} mode(s)")
        ax2.plot(y, D_rec[:, -1], lw=1, label=f"{r} mode(s)")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Force [N/m] at specified station")
    ax1.set_title("Reconstruction quality (time series)")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Spanwise station y [m]")
    ax2.set_ylabel("Force [N/m] at last time step")
    ax2.set_title("Reconstruction quality (spanwise, last snapshot)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(Path(out_dir) / "04_reconstruction_check.png", dpi=150)

    return fig, (ax1, ax2)


def plot_force_orbit(t: np.ndarray, Fx: np.ndarray, Fz: np.ndarray,
                     pod: Dict, field_info: Dict, station: int = 0,
                     r_values: Sequence[int] = (2, 4),
                     out_dir: Optional[Path] = None
                     ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the sectional-force orbit (Fx vs. Fz trajectory) at one station.

    At a fixed spanwise station the instantaneous sectional force is a 2-D
    vector (Fx(t), Fz(t)). Tracing this vector's tip over time gives a
    force "orbit" -- a line, ellipse or more complex (e.g. figure-of-eight)
    loop whose shape reflects the coupling between edgewise and flapwise
    loading, which is characteristic of vortex-induced-vibration behaviour.
    This function overlays the orbit from the original data with the
    orbit(s) reconstructed from a truncated POD basis, so that the number
    of modes needed to reproduce the orbit's shape can be assessed. It
    requires a POD computed on the *combined* Fx/Fz field (i.e. the ``pod``
    and ``field_info`` returned by ``compute_pod`` and ``build_field_matrix``
    with ``mode="combined"``), since only then do the POD modes carry
    information about both force components jointly.

    Parameters
    ----------
    t : np.ndarray of shape (n_snapshots,)
        Time vector, shape ``(n_snapshots,)``, in seconds. Used to colour
        the original orbit by time so the direction of travel is visible.
    Fx : np.ndarray of shape (n_snapshots, n_sections)
        Original edgewise sectional force.
    Fz : np.ndarray of shape (n_snapshots, n_sections)
        Original flapwise sectional force.
    pod : dict of {str : np.ndarray}
        POD decomposition returned by :func:`compute_pod`, computed on the
        combined field built by
        ``build_field_matrix(Fx, Fz, mode="combined")``.
    field_info : dict of {str : Any}
        Metadata returned alongside the combined field by
        :func:`build_field_matrix`. Must have ``field_info["kind"] ==
        "combined"`` and provide ``"n"``, ``"scale_x"`` and ``"scale_z"``.
    station : int, optional
        Index of the spanwise station to plot the orbit for. Default is
        ``0`` (the blade root).
    r_values : Sequence[int], optional
        Numbers of leading POD modes to include in the reconstructed
        orbit(s) overlaid for comparison. Default is ``(2, 4)``.
    out_dir : pathlib.Path, optional
        Directory the figure is saved to. If ``None``, the
        figure is not saved to disk (only returned).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the orbit plot.

    Raises
    ------
    ValueError
        If ``field_info["kind"]`` is not ``"combined"``.

    Notes
    -----
    The combined POD field stacks the (standardised) Fx and Fz snapshot
    matrices column-wise, so a reconstruction with ``r`` modes,
    ``D_rec = reconstruct(pod, r)``, has its first ``field_info["n"]`` rows
    corresponding to the (rescaled) Fx component and the remaining rows to
    the Fz component; this function performs that split and the rescaling
    by ``field_info["scale_x"]`` / ``field_info["scale_z"]`` internally.
    """
    if field_info.get("kind") != "combined":
        raise ValueError(
            "plot_force_orbit requires a POD computed on the combined "
            "Fx/Fz field (field_info['kind'] == 'combined')."
        )

    n = field_info["n"]
    sx = field_info["scale_x"]
    sz = field_info["scale_z"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Original orbit at the chosen station, coloured by time to show
    # the direction/rate of travel around the loop.
    sc = ax.scatter(Fx[:, station], Fz[:, station], c=t, cmap="viridis",
                    s=8, label="original")
    fig.colorbar(sc, ax=ax, label="time [s]")

    # POD-reconstructed orbit(s) for comparison.
    for r in r_values:
        D_rec = reconstruct(pod, r)  # (2*n, n_snapshots), combined units
        Fx_rec = D_rec[:n, :][station, :] * sx
        Fz_rec = D_rec[n:, :][station, :] * sz
        ax.plot(Fx_rec, Fz_rec, lw=1.2, alpha=.8, label=f"{r} mode(s)")

    ax.set_xlabel("Fx [N/m] (edgewise)")
    ax.set_ylabel("Fz [N/m] (flapwise)")
    ax.set_title(f"Sectional-force orbit at station index {station}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if out_dir is not None:
        fig.savefig(Path(out_dir) / "05_force_orbit.png", dpi=150)

    return fig, ax


# =============================================================================
# Physical cross-check
# =============================================================================
def power_correlation_check(y: np.ndarray, t: np.ndarray, Fx: np.ndarray,
                            Fz: np.ndarray, pod: dict, out_dir: PurePath,
                            amplitude: float = AMPLITUDE,
                            omega: float = 2 * np.pi * F_FORCING,
                            n_modes: int = 4,
                            work_file: None | PurePath = None):
    """
    Correlate aerodynamic power with the leading POD temporal coefficients.

    Computes the instantaneous aerodynamic power transferred from the fluid to
    the prescribed blade motion using the sectional force distributions and
    prescribed motion kinematics. The total power is correlated with the
    leading POD temporal coefficients to identify modes associated with energy
    transfer. Optionally compares the computed mean power with data from a
    ``*.1work`` file.

    Parameters
    ----------
    y : ndarray of shape (n_sections,)
        Spanwise station coordinates.
    t : ndarray of shape (n_snapshots,)
        Time vector.
    Fx : ndarray of shape (n_snapshots, n_sections)
        Edgewise sectional force.
    Fz : ndarray of shape (n_snapshots, n_sections)
        Flapwise sectional force.
    pod : dict
        Dictionary returned by :func:`compute_pod`.
    out_dir : pathlib.PurePath
        Directory where output figures are written.
    amplitude : float, optional
        Blade motion amplitude scaling factor.
    omega : float, optional
        Angular excitation frequency in rad/s.
    n_show : int, optional
        Number of leading modes to consider for the correlation calculation.
        Default is 4.
    work_file : pathlib.PurePath or None, optional
        Solver-generated ``*.1work`` file used for validation of the computed
        aerodynamic power. If ``None`` or the file does not exist, no
        comparison is performed.

    Returns
    -------
    None

    Notes
    -----
    The instantaneous aerodynamic power is computed by integrating the
    sectional power density along the blade span,

    ``P(t) = ∫ (Fx * vx + Fz * vz) dy``,

    where the prescribed blade velocities are obtained from the motion
    polynomials defined in the simulation setup.
    """
    vx = np.polyval(COEFF_X, y - 2.8) * amplitude * omega \
        * np.cos(omega * t)[:, None]
    vz = np.polyval(COEFF_Z, y - 2.8) * amplitude * omega \
        * np.cos(omega * t)[:, None]
    # p_sectional = Fx * vx + Fz * vz  # [W/m] Instantaneous power
    p_sectional = Fx * vx  # [W/m] Instantaneous power
    P_total = np.trapezoid(p_sectional, y, axis=1)  # [W] Power per time step

    print("\n--- Physical power cross-check ---")
    print(f"Mean input power (from Fx, Fz, and prescribed velocity): "
          f"{P_total.mean():.4g} W  "
          f"({'exciting' if P_total.mean() > 0 else 'damping'} the motion)")

    if work_file is not None and work_file.exists():
        wk = np.loadtxt(work_file, comments="#")
        dt = t[1] - t[0]
        n = min(len(wk), len(P_total))
        P_work = wk[-n:, 2] / dt  # wx column -> power
        print("Mean power from *.1work file (cross-check): "
              f"{P_work.mean():.4g} W")
    else:
        print("(*.1work file not found/skipped. No cross-check against "
              "solver output)")

    corr = [np.corrcoef(P_total, pod["time_coeff"][k, :])[0, 1]
            for k in range(min(n_modes, pod["time_coeff"].shape[0]))]
    print("Correlation of each leading POD mode's a_k(t) with instantaneous "
          "power:")
    for k, c in enumerate(corr, start=1):
        print(f"  mode {k}: r = {c:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, P_total, "k", label="instantaneous power P(t)")
    ax2 = ax.twinx()
    ax2.plot(t, pod["time_coeff"][0, :], "C0", label="a_1(t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [W]")
    ax2.set_ylabel("a_1(t)", color="C0")
    ax.set_title("Aerodynamic power vs. leading POD temporal coefficient")
    fig.tight_layout()
    fig.savefig(out_dir / "06_power_vs_mode1.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main execution
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hy_path = DATA_DIR / f"{CASE}_u{U_WIND}.1HY"
    work_path = DATA_DIR / f"{CASE}_u{U_WIND}.1work"

    print(f"Loading {hy_path} ...")
    y, t, Fx, Fz = load_hy(hy_path, n_skip=N_SKIP, n_last=N_LAST)
    dt = t[1] - t[0]
    n_periods = (t[-1] - t[0]) / (1.0 / F_FORCING)
    print(f"  {len(t)} snapshots, dt = {dt:.6g} s, "
          f"spanning ~{n_periods:.1f} forcing periods")
    print(f"  {len(y)} spanwise stations, y = {y[0]:.1f} to {y[-1]:.1f} m")

    field, field_info = build_field_matrix(Fx=Fx, Fz=Fz, mode=FIELD)
    pod = compute_pod(field=field)

    n_report = min(N_MODES_SHOW, len(pod["sigma"]))
    print(f"\n--- POD summary (field = '{FIELD}') ---")
    print(f"{'mode':>4} {'energy %':>10} {'cum %':>8} "
          f"{'dom. freq [Hz]':>15}r {'freq / f_forcing':>17}")
    for k in range(n_report):
        f_dom, _, _ = dominant_frequency(signal=pod["time_coeff"][k, :], dt=dt)
        print(f"{k+1:>4} {100*pod['energy_frac'][k]:>10.2f} "
              f"{100*pod['cum_energy'][k]:>8.2f} {f_dom:>15.4f} "
              f"{f_dom / F_FORCING:>17.3f}")

    n99 = int(np.searchsorted(pod["cum_energy"], 0.99) + 1)
    print(f"\nModes needed to capture 99% of the fluctuation energy: {n99}")

    plot_energy_spectrum(pod=pod, out_dir=OUTPUT_DIR)
    plot_spatial_modes(y=y, pod=pod, field_info=field_info,
                       out_dir=OUTPUT_DIR, n_show=N_MODES_SHOW)
    plot_temporal_coefficients(t=t, pod=pod, dt=dt, f_forcing=F_FORCING,
                               out_dir=OUTPUT_DIR, n_show=N_MODES_SHOW)

    # For the reconstruction check, use whichever raw field matches FIELD
    # ("combined" mixes units, so default the sanity-check plot to Fz).
    D_full_check = Fz if FIELD != "fx" else Fx
    pod_check = pod if field_info["kind"] == "single" \
        else compute_pod(D_full_check)
    idx_onethird = np.where(y >= y[-1]/3)[0][0]
    plot_reconstruction(y=y, t=t, D_full=D_full_check, pod=pod_check,
                        station=idx_onethird, out_dir=OUTPUT_DIR)

    if field_info["kind"] == "combined":
        plot_force_orbit(t=t, Fx=Fx, Fz=Fz, pod=pod, field_info=field_info,
                         station=idx_onethird,
                         r_values=(2, 4),
                         out_dir=OUTPUT_DIR)

    if DO_POWER_CHECK:
        power_correlation_check(y=y, t=t, Fx=Fx, Fz=Fz, pod=pod, n_modes=n99,
                                out_dir=OUTPUT_DIR, work_file=work_path)

    print(f"\nFigures written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
