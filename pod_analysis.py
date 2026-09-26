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

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless / batch runs
import matplotlib.pyplot as plt

# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the data
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see data README
U_WIND = 10  # free-stream wind speed [m/s] to analyse (6-22)

FIELD = "fz"  # "fx", "fz", or "combined" -> which sectional force
                # component(s) feed the POD. "fz" (flapwise) is the natural
                # default for VIV.

N_SKIP = 400  # rows (time steps) dropped as initial transient
                # Default (400) equals two full periods.
N_LAST = None  # keep only the last N rows after N_SKIP;
                # None = keep everything (recommended: run POD over several
                # periods, not just the last one, so lock-in / beating
                # behaviour is visible).

N_MODES_SHOW = 4  # number of leading POD modes to plot/report
F_FORCING = 0.624  # [Hz] prescribed excitation frequency (cf. README)
AMPLITUDE = 1.0  # motion amplitude factor A (cf. README, case a01)

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
        Results of POD
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
# Main execution
# =============================================================================
def main():
    hy_path = DATA_DIR / f"{CASE}_u{U_WIND}.1HY"

    print(f"Loading {hy_path} ...")
    y, t, Fx, Fz = load_hy(hy_path, n_skip=N_SKIP, n_last=N_LAST)
    dt = t[1] - t[0]
    n_periods = (t[-1] - t[0]) / (1.0 / F_FORCING)
    print(f"  {len(t)} snapshots, dt = {dt:.6g} s, "
          f"spanning ~{n_periods:.1f} forcing periods")
    print(f"  {len(y)} spanwise stations, y = {y[0]:.1f} to {y[-1]:.1f} m")

    field, field_info = build_field_matrix(Fx, Fz, FIELD)
    pod = compute_pod(field)

    n_report = min(N_MODES_SHOW, len(pod["sigma"]))
    print(f"\n--- POD summary (field = '{FIELD}') ---")
    print(f"{'mode':>4} {'energy %':>10} {'cum %':>8} "
          f"{'dom. freq [Hz]':>15}r {'freq / f_forcing':>17}")
    for k in range(n_report):
        f_dom, _, _ = dominant_frequency(pod["time_coeff"][k, :], dt)
        print(f"{k+1:>4} {100*pod['energy_frac'][k]:>10.2f} "
              f"{100*pod['cum_energy'][k]:>8.2f} {f_dom:>15.4f} "
              f"{f_dom / F_FORCING:>17.3f}")

    n99 = int(np.searchsorted(pod["cum_energy"], 0.99) + 1)
    print(f"\nModes needed to capture 99% of the fluctuation energy: {n99}")

if __name__ == "__main__":
    main()
