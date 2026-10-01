"""
spod_analysis.py

Spectral Proper Orthogonal Decomposition (SPOD) of the spanwise sectional-
force distribution on an oscillating wind-turbine blade (vortex-induced-
vibration CFD sweep, EllipSys3D).

Where pod_analysis.py finds spatial structures ranked by total energy (with
no guarantee that a mode oscillates at a single frequency) and
spectral_analysis.py characterises frequency content station-by-station
(with no notion of a coherent spatial structure), SPOD combines both: it
finds spatial modes that are simultaneously energy-ranked *and* tied to a
single frequency. This is a natural next step for data -- like this one --
that appears to be dominated by oscillation at one (prescribed) frequency:
SPOD directly tests that hypothesis by revealing, at each frequency,
whether the field is well described by one dominant coherent structure
(a large gap between the 1st and 2nd SPOD eigenvalue) or by a broad mix of
comparably-energetic structures (little or no gap -- more consistent with
broadband/stochastic content such as unlocked vortex shedding or numerical
noise).

The implementation follows the standard batch SPOD algorithm of Towne,
Schmidt & Colonius (2018, J. Fluid Mech. 847:821-867): the record is split
into overlapping, windowed blocks (exactly as in Welch's method); at each
frequency, the ensemble of block-wise Fourier coefficients is assembled
into a matrix and decomposed via the SVD, which is algebraically equivalent
to eigen-decomposing the cross-spectral density (CSD) matrix at that
frequency but cheaper here since the number of blocks is smaller than the
number of spatial points (though not by a huge lot for this specific dataset).
===============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import ShortTimeFFT, get_window, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the *.1HY files
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see README
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

N_MODES_SHOW = 4  # number of leading eigenvalue curves to plot
N_HARMONICS = 4  # number of integer harmonics of F_FORCING to
                 # report eigenvalues/mode shapes for
F_FORCING = 0.624  # [Hz] prescribed excitation frequency (README)

SPOD_NPERSEG = 800  # block length [samples]; None -> n // 6 (see
  # compute_spod for the resolution/convergence
  # trade-off this controls)
SPOD_NOVERLAP = None  # overlap [samples]; None -> SPOD_NPERSEG // 2
SPOD_WINDOW = "hann"  # window applied to each block

STATION_FOR_PHASE_REF = 0  # station used to remove an arbitrary overall
                           # phase offset when plotting mode phase (purely
                           # cosmetic: SPOD modes are only defined up to a
                           # unit-modulus complex factor)

OUTPUT_DIR = Path(__file__).parent / "results" / "pod_test_single" / "spod"

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
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def build_field_matrix(Fx: np.ndarray, Fz: np.ndarray, mode: str
                       ) -> Tuple[np.ndarray, dict]:
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
    field_array: np.ndarray
        The field array containing the data for the selected mode
    field_info: dict
        Description of how to interpret / invert the scaling when a "combined"
        spatial mode is split back into its Fx- and Fz- parts for plotting.
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
# SPOD
# =============================================================================
def compute_spod(field: np.ndarray, dt: float, nperseg: Optional[int] = None,
                 noverlap: Optional[int] = None, window: str = "hann"
                 ) -> Dict[str, np.ndarray]:
    """Spectral Proper Orthogonal Decomposition via the batch/Welch algorithm.

    Follows Towne, Schmidt & Colonius (2018): the record is split into
    overlapping, windowed blocks; at every frequency, the ensemble of
    block-wise Fourier coefficients across blocks is assembled into a
    ``(n_points, n_blocks)`` matrix and decomposed via the SVD, which is
    algebraically equivalent to (but cheaper than) eigen-decomposing the
    ``(n_points, n_points)`` cross-spectral density matrix directly,
    since here ``n_blocks`` is typically much smaller than ``n_points``.

    Unlike plain POD, SPOD modes are both spatially *and* temporally
    orthogonal: each mode oscillates at a single frequency, and modes at
    different frequencies are statistically uncorrelated. Unlike a plain
    PSD, SPOD retains the spatial *shape* associated with the energy at
    each frequency, not just how much energy is present.

    Parameters
    ----------
    field : ndarray, shape (n_snapshots, n_points)
        Time series of the quantity to decompose, rows = time snapshots,
        columns = spatial points (e.g. spanwise stations).
    dt : float
        Sample interval [s].
    nperseg : int, optional
        Block length in samples. This is the central trade-off in SPOD: a
        longer block gives finer frequency resolution but fewer blocks
        (a noisier estimate of the cross-spectral density, i.e. a less
        converged eigenvalue/eigenvector estimate at each frequency); a
        shorter block gives more blocks (better statistical convergence)
        at the cost of frequency resolution. Default is ``n // 6``.
    noverlap : int, optional
        Overlap between consecutive blocks, in samples. Default is
        ``nperseg // 2`` (50% overlap, the standard choice for a Hann
        window).
    window : str, optional
        Name of the window applied to each block before the FFT (passed
        to ``scipy.signal.get_window``). Default ``"hann"``.

    Returns
    -------
    result : dict
        Dictionary with keys:

        freqs : ndarray, shape (n_freqs,)
            Frequency axis [Hz].
        eigenvalues : ndarray, shape (n_freqs, n_modes)
            SPOD eigenvalues (energy) at each frequency, descending within
            each row, where ``n_modes = min(n_points, n_blocks)``. In
            units comparable to a one-sided PSD (e.g. (N/m)^2/Hz): summing
            all of a row's eigenvalues gives (approximately) the same
            value as summing the single-point Welch PSDs over all spatial
            points at that frequency.
        modes : ndarray, shape (n_freqs, n_points, n_modes), complex
            SPOD spatial modes, unit-norm and mutually orthogonal within
            each frequency; ``modes[k, :, 0]`` is the leading (most
            energetic) mode at frequency ``freqs[k]``.
        n_blocks : int
            Number of blocks used to estimate the cross-spectral density.

    Raises
    ------
    ValueError
        If fewer than 2 usable blocks result from the chosen ``nperseg``/
        ``noverlap`` and the record length.

    Notes
    -----
    ``scipy.signal.ShortTimeFFT`` is used with ``scale_to="psd"``, which
    (unlike ``scipy.signal.welch``'s ``scaling="density"``) does not itself
    double the power of the non-DC/Nyquist bins to account for the folded
    negative frequencies of a one-sided spectrum; that factor of
    ``sqrt(2)`` is applied explicitly below so the resulting eigenvalues
    are directly comparable to a Welch ``"density"`` PSD (the same
    convention used in the companion ``spectral_analysis.py`` script's
    ``compute_spanwise_amplitude``).
    """
    n, n_points = field.shape
    seg = nperseg if nperseg is not None else max(16, n // 6)
    seg = min(seg, n)
    novl = noverlap if noverlap is not None else seg // 2
    hop = seg - novl
    if hop < 1:
        raise ValueError("noverlap must be smaller than nperseg.")

    win = get_window(window, seg, fftbins=True)
    fluctuation = field - field.mean(axis=0, keepdims=True)

    sft = ShortTimeFFT(win, hop=hop, fs=1.0 / dt, scale_to="psd",
                       fft_mode="onesided")
    Q = sft.stft(fluctuation, axis=0)  # (n_freqs, n_points, n_blocks_total)
    freqs = sft.f

    # Drop the first/last blocks: by default ShortTimeFFT centres a window
    # on every sample, including near the edges, where it is zero-padded
    # rather than built from genuine data. Excluding those keeps every
    # block used a fully-windowed segment of real data.
    if Q.shape[-1] > 2:
        Q = Q[:, :, 1:-1]
    n_blocks = Q.shape[-1]
    if n_blocks < 2:
        raise ValueError(
            f"Only {n_blocks} usable block(s) with nperseg={seg}, "
            f"noverlap={novl}; SPOD needs at least 2 blocks to estimate a "
            "cross-spectral density. Reduce nperseg/noverlap or provide a "
            "longer record."
        )

    # One-sided PSD correction (see Notes above).
    onesided_factor = np.full(len(freqs), np.sqrt(2.0))
    onesided_factor[0] = 1.0
    if seg % 2 == 0:
        onesided_factor[-1] = 1.0
    Q = Q * onesided_factor[:, None, None]

    n_modes = min(n_points, n_blocks)
    eigenvalues = np.zeros((len(freqs), n_modes))
    modes = np.zeros((len(freqs), n_points, n_modes), dtype=complex)

    for k in range(len(freqs)):
        Qf = Q[k, :, :] / np.sqrt(n_blocks)  # (n_points, n_blocks)
        Phi, sigma, _ = np.linalg.svd(Qf, full_matrices=False)
        eigenvalues[k, :] = sigma**2
        modes[k, :, :] = Phi

    return {"freqs": freqs, "eigenvalues": eigenvalues, "modes": modes,
            "n_blocks": n_blocks}


def nearest_freq_index(freqs: np.ndarray, target: float) -> int:
    """Return the index of the frequency bin closest to ``target``.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency axis to search.
    target : float
        Target frequency [Hz].

    Returns
    -------
    int
        Index ``k`` such that ``freqs[k]`` is closest to ``target``.
    """
    return int(np.argmin(np.abs(freqs - target)))


def spanwise_psd_trace(field: np.ndarray, dt: float, nperseg: int,
                       noverlap: int, window: str
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of the single-point Welch PSDs over all spatial points.

    Used purely as an independent cross-check on :func:`compute_spod`'s
    eigenvalues: since the SPOD eigenvalues at a frequency are the
    eigenvalues of the cross-spectral density matrix at that frequency,
    their sum (the matrix trace) should approximately equal the sum of
    the ordinary (phase-blind) Welch PSD of every individual spatial
    point, computed with the same block settings.

    Parameters
    ----------
    field : ndarray, shape (n_snapshots, n_points)
        Same field passed to :func:`compute_spod`.
    dt : float
        Sample interval [s].
    nperseg, noverlap : int
        Must match the values effectively used by :func:`compute_spod`
        for the comparison to be meaningful.
    window : str
        Window name, passed to ``scipy.signal.welch``.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
    trace_psd : ndarray, shape (n_freqs,)
        Sum over spatial points of the Welch ("density") PSD.
    """
    freqs, psd_map = welch(field, fs=1.0 / dt, window=window, nperseg=nperseg,
                           noverlap=noverlap, axis=0, detrend="constant",
                           scaling="density")
    return freqs, psd_map.sum(axis=1)


# =============================================================================
# Plotting
# =============================================================================
def plot_spod_spectrum(
        spod: Dict[str, np.ndarray], f_forcing: float,
        n_harmonics: int, n_modes_show: int, out_dir: Path,
        trace_check: Optional[Tuple[np.ndarray, np.ndarray]] = None
        ) -> None:
    """Plot the leading SPOD eigenvalue curves vs. frequency.

    This is the classic "SPOD spectrum" plot: each curve traces how one
    eigenvalue rank (1st, 2nd, ...) varies with frequency. A pronounced,
    well-separated 1st curve at a given frequency indicates a single
    dominant coherent structure there (e.g. the prescribed-motion
    frequency); curves bunched together indicate energy spread over
    several comparably-important structures (more consistent with
    broadband/stochastic content).

    Parameters
    ----------
    spod : dict
        Return value of :func:`compute_spod`.
    f_forcing : float
        Prescribed excitation frequency [Hz], marked with the first
        dashed line.
    n_harmonics : int
        Number of integer harmonics of ``f_forcing`` to mark with dashed
        vertical lines.
    n_modes_show : int
        Number of leading eigenvalue curves to plot.
    out_dir : pathlib.Path
        Directory the figure is saved to, as ``"01_spod_spectrum.png"``.
    trace_check : tuple of (ndarray, ndarray), optional
        If given, as ``(freqs, trace_psd)`` from :func:`spanwise_psd_trace`,
        overlaid as a dashed black line for comparison against the sum of
        the plotted SPOD eigenvalues.
    """
    freqs = spod["freqs"]
    eigenvalues = spod["eigenvalues"]
    n_show = min(n_modes_show, eigenvalues.shape[1])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k in range(n_show):
        ax.semilogy(freqs, eigenvalues[:, k], label=f"SPOD mode {k + 1}")

    if trace_check is not None:
        f_chk, trace_chk = trace_check
        ax.semilogy(f_chk, trace_chk, "k--", lw=1,
                    label="sum of single-point PSDs (cross-check)")

    for h in range(1, n_harmonics + 1):
        ax.axvline(h * f_forcing, color="gray", ls="--", lw=0.8,
                   label="forcing harmonics" if h == 1 else None)

    ax.set_xlim(0, (n_harmonics + 1) * f_forcing)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("SPOD eigenvalue (energy)")
    ax.set_title("SPOD eigenvalue spectrum")
    ax.legend(fontsize=8)

    ax.grid(alpha=.8, zorder=0)

    fig.tight_layout()
    fig.savefig(out_dir / "01_spod_spectrum.png", dpi=150)
    plt.close(fig)


def plot_spod_mode_shapes(y: np.ndarray, spod: Dict[str, np.ndarray],
                          field_info: Dict[str, Any],
                          target_freqs: Sequence[float],
                          out_dir: Path, phase_ref_station: int = 0,
                          n_modes_show: int = 1) -> None:
    """Plot the leading SPOD spatial mode shape(s) at chosen frequencies.

    For each requested frequency, the nearest available SPOD frequency bin
    is used and its leading mode(s) are plotted as magnitude and phase vs.
    spanwise station. SPOD modes are complex and only defined up to an
    arbitrary unit-modulus factor, so before plotting, each mode is
    rotated to make its value at ``phase_ref_station`` real and positive;
    this is a purely cosmetic step that does not affect the mode shape's
    physical content, only where its phase reference is set to zero.

    Parameters
    ----------
    y : ndarray, shape (n_sections,)
        Spanwise coordinate [m].
    spod : dict
        Return value of :func:`compute_spod`.
    field_info : dict
        Metadata from :func:`build_field_matrix`, used to split a
        ``"combined"`` mode back into its Fx- and Fz- parts.
    target_freqs : sequence of float
        Frequencies [Hz] to plot mode shapes for (e.g. the forcing
        frequency and its harmonics); each is snapped to the nearest
        available SPOD frequency bin.
    out_dir : pathlib.Path
        Directory the figure is saved to, as
        ``"02_spod_mode_shapes.png"``.
    phase_ref_station : int, optional
        Station index used to remove the arbitrary overall phase of each
        mode before plotting. Default 0.
    n_modes_show : int, optional
        Number of leading modes to overlay at each frequency. Default 1
        (only the leading, most energetic mode).
    """
    freqs = spod["freqs"]
    modes = spod["modes"]
    is_combined = field_info["kind"] == "combined"

    n_rows = 2 if is_combined else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.5 * n_rows),
                             squeeze=False)

    n_freqs = len(target_freqs)
    for i_freq, f_target in enumerate(target_freqs):
        k = nearest_freq_index(freqs, f_target)
        for m in range(n_modes_show):
            phi = modes[k, :, m]
            phi = phi * np.exp(-1j * np.angle(phi[phase_ref_station])
                               )  # phase reference
            label = f"f = {freqs[k]:.3f} Hz, mode {m + 1}"

            # Layer lines for each mode with the lowest mode on top and
            # increase transparency with mode number
            zorder = n_freqs + 1 - i_freq
            alpha = 1 - i_freq/n_freqs * .5

            if is_combined:
                n = field_info["n"]
                phi_x = phi[:n] * field_info["scale_x"]
                phi_z = phi[n:] * field_info["scale_z"]
                axes[0, 0].plot(y, np.abs(phi_x), label=label, alpha=alpha,
                                zorder=zorder)
                axes[0, 1].plot(y, np.angle(phi_x), label=label, alpha=alpha,
                                zorder=zorder)
                axes[1, 0].plot(y, np.abs(phi_z), label=label, alpha=alpha,
                                zorder=zorder)
                axes[1, 1].plot(y, np.angle(phi_z), label=label, alpha=alpha,
                                zorder=zorder)
            else:
                axes[0, 0].plot(y, np.abs(phi), label=label, alpha=alpha,
                                zorder=zorder)
                axes[0, 1].plot(y, np.angle(phi), label=label, alpha=alpha,
                                zorder=zorder)

    axes[0, 0].grid(alpha=.8, zorder=0)
    axes[0, 1].grid(alpha=.8, zorder=0)

    if is_combined:
        # Plot modeshapes of forced motion (for reference)
        modeshape_x = np.polyval(COEFF_X, y - 2.8)
        modeshape_x /= np.max(modeshape_x)
        ax01_twin = axes[0, 0].twinx()
        ax01_twin.plot(y, modeshape_x, c="k", ls="--",
                       label="Forced motion modeshape")
        ax01_twin.set_ylabel("Forced motion modeshape [-]")

        modeshape_z = np.polyval(COEFF_Z, y - 2.8)
        modeshape_z /= np.max(modeshape_z)
        ax02_twin = axes[0, 1].twinx()
        ax02_twin.plot(y, modeshape_z, c="k", ls="--",
                       label="Forced motion modeshape")
        ax02_twin.set_ylabel("Forced motion modeshape [-]")

        axes[1, 0].grid(alpha=.8, zorder=0)
        axes[1, 1].grid(alpha=.8, zorder=0)
    else:
        if field_info["component"] == "Fx":
            modeshape = np.polyval(COEFF_X, y - 2.8)
            modeshape /= np.max(modeshape)
        elif field_info["component"] == "Fz":
            modeshape = np.polyval(COEFF_Z, y - 2.8)
            modeshape /= np.max(modeshape)

        # Plot modeshape of forced motion (for reference)
        if modeshape is not None:
            ax0_twin = axes[0, 0].twinx()
            ax0_twin.plot(y, modeshape, c="k", ls="--",
                          label="Forced motion modeshape")
            ax0_twin.set_ylabel("Forced motion modeshape [-]")

    comp_labels = (["Fx (edgewise)", "Fz (flapwise)"] if is_combined
                   else [field_info["component"]])
    for row, comp_label in enumerate(comp_labels):
        axes[row, 0].set_ylabel(f"{comp_label} |mode| [-]")
        axes[row, 0].set_title(f"{comp_label} mode magnitude")
        axes[row, 1].set_ylabel(f"{comp_label} phase [rad]")
        axes[row, 1].set_title(f"{comp_label} mode phase")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("spanwise station y [m]")

    fig.tight_layout()
    fig.savefig(out_dir / "02_spod_mode_shapes.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main execution
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hy_path = DATA_DIR / f"{CASE}_u{U_WIND}.1HY"

    print(f"Loading {hy_path} ...")
    y, t, Fx, Fz = load_hy(hy_path, n_skip=N_SKIP, n_last=N_LAST)
    dt = t[1] - t[0]
    n_periods = (t[-1] - t[0]) / (1.0 / F_FORCING)
    print(f"  {len(t)} snapshots, dt = {dt:.6g} s, "
            f"spanning ~{n_periods:.1f} forcing periods")
    print(f"  {len(y)} spanwise stations, y = {y[0]:.1f} to {y[-1]:.1f} m")

    field, field_info = build_field_matrix(Fx=Fx, Fz=Fz, mode=FIELD)

    spod = compute_spod(field, dt, nperseg=SPOD_NPERSEG,
                        noverlap=SPOD_NOVERLAP, window=SPOD_WINDOW)
    seg = SPOD_NPERSEG if SPOD_NPERSEG is not None else max(16, len(t) // 6)
    seg = min(seg, len(t))
    novl = SPOD_NOVERLAP if SPOD_NOVERLAP is not None else seg // 2
    df = spod["freqs"][1] - spod["freqs"][0]
    print(f"\nSPOD block settings: nperseg = {seg} samples, "
          f"noverlap = {novl} samples, "
          f"n_blocks = {spod['n_blocks']}, frequency resolution = {df:.4f} Hz")

    print("\n--- SPOD eigenvalues at the forcing harmonics ---")
    print(f"{'harmonic':>9} {'f target':>9} {'f bin':>9} {'lambda_1':>12} "
          f"{'lambda_2':>12} {'lambda_1/lambda_2':>18}")
    for h in range(1, N_HARMONICS + 1):
        f_target = h * F_FORCING
        k = nearest_freq_index(spod["freqs"], f_target)
        lam1 = spod["eigenvalues"][k, 0]
        lam2 = spod["eigenvalues"][k, 1] if spod["eigenvalues"].shape[1] > 1 \
            else np.nan
        ratio = lam1 / lam2 if lam2 > 0 else np.inf
        print(f"{h:>9} {f_target:>9.3f} {spod['freqs'][k]:>9.3f} "
              f"{lam1:>12.4g} {lam2:>12.4g} {ratio:>18.2f}")
    print("(a large lambda_1/lambda_2 ratio means the field is well described "
          "by a single coherent structure at that frequency, consistent with "
          "a deterministic, forced response; a ratio near 1 indicates energy "
          "spread over several comparably-important structures, more "
          "consistent with broadband/stochastic content)")

    # cross-check: SPOD eigenvalue sum (trace) vs. independent single-point
    # PSD sum
    trace_check = spanwise_psd_trace(field, dt, seg, novl, SPOD_WINDOW)

    plot_spod_spectrum(spod, F_FORCING, N_HARMONICS, N_MODES_SHOW, OUTPUT_DIR,
                        trace_check=trace_check)

    target_freqs = [h * F_FORCING for h in range(1, N_HARMONICS + 1)]
    plot_spod_mode_shapes(y, spod, field_info, target_freqs, OUTPUT_DIR,
                          phase_ref_station=STATION_FOR_PHASE_REF,
                          n_modes_show=1)

    print(f"\nFigures written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
