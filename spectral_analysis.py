"""
spectral_analysis.py
Spectral analysis of the spanwise sectional-force distribution on an
oscillating wind-turbine blade (vortex-induced-vibration CFD sweep,
EllipSys3D).

Where pod_analysis.py decomposes the force field into energy-ranked spatial
modes (POD via SVD), this script instead characterises the *frequency
content* of the same data using standard tools from scipy.signal:

    - Welch's method (scipy.signal.welch) for power-spectral-density (PSD)
      estimates of the force at individual spanwise stations and of the
      span-integrated (total) force.
    - A frequency-vs-span PSD map, to see whether the dominant frequency is
      constant along the span or varies from root to tip.
    - A time-frequency spectrogram, to catch transients, amplitude
      modulation or beating that a single, time-averaged PSD would hide.
    - An optional sweep over the wind-speed series described in the README,
      tracking how the dominant response frequency of the total force moves
      with wind speed U -- the standard way to spot "lock-in" (the response
      clamped to the prescribed-motion frequency) versus a free
      vortex-shedding response that instead scales with U.
"""

from pathlib import Path, PurePath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# =============================================================================
# User configuration
# =============================================================================
DATA_DIR = Path("./data/archive_1D")  # folder containing the data
CASE = "a01_p85_i50"  # amplitude/pitch/inclination code, see README
U_WIND = 10  # wind speed [m/s] for the detailed single-case run

FIELD = "both"  # "fx", "fz", or "both" -> which sectional-force
                # component(s) to analyse station-by-station

STATIONS = (0, 25, 50, 75, 99)  # spanwise station indices for the PSD plots
                                # (0 = root, 99 = tip, for 100 sections)

# Rows (time steps) dropped as initial transient (Note: 200 is one full period)
N_SKIP = 1200
N_LAST = None  # Rows to keep after N_SKIP

F_FORCING = 0.624  # [Hz] prescribed excitation frequency (README)
N_HARMONICS = 6  # number of integer harmonics of F_FORCING to mark
NPERSEG = None  # Welch segment length; None -> len(signal) // 8

DO_SPANWISE_MAP = True  # frequency-vs-span PSD heatmap
DO_SPANWISE_AMPLITUDE_MAP = True  # frequency-vs-span peak-amplitude heatmap (linear, N/m)
DO_SPANWISE_PHASE_MAP = True  # Frequency-vs-span phase heatmap
DO_SPECTROGRAM = True  # time-frequency spectrogram of the total force
DO_FREQUENCY_SWEEP = True  # repeat the dominant-frequency analysis over U

PHASE_MASK_THRESHOLD_DB: Optional[float] = -30   # mask phase where magnitude is more than
                                                  # this many dB below the panel's peak
                                                  # magnitude (phase is meaningless where
                                                  # there is no real spectral content);
                                                  # set to None to disable masking

U_SWEEP_LIST = list(range(6, 23))  # wind speeds [m/s] to include in the sweep
SWEEP_FIELD = "fx"  # component used for the frequency sweep

OUTPUT_DIR = Path(__file__).parent / "results" / "pod_test_single" / "spectral"


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
    n_last : None | int, optional
        Number or rows to keep after n_skip. If None, keep all rows.
        Default is None

    Returns
    -------
    y : ndarray, shape (n_sections,)
        Spanwise coordinate [m]
    t : (n_rows,)
        Time [s] (absolute simulation time)
    Fx : ndarray, shape (n_rows, n_sections)
        Edgewise sectional force [N/m]
    Fz : ndarray, shape (n_rows, n_sections)
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


# =============================================================================
# Spectral estimates
# =============================================================================
def compute_psd(signal: np.ndarray, dt: float, nperseg: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density of a single time signal via Welch's method.

    Parameters
    ----------
    signal  : (n_samples,) array
    dt      : sample interval [s]
    nperseg : Welch segment length; default is the full record (nperseg = n),
              i.e. a single windowed periodogram with no segment-averaging.
              This dataset is short relative to the frequencies of interest
              (order 10-15 forcing periods per file), so splitting it into
              several sub-segments (the usual Welch default) destroys the
              frequency resolution needed to distinguish nearby frequencies
              (e.g. a 0.45 Hz response from the 0.624 Hz forcing). Pass an
              explicit nperseg to trade resolution for variance reduction
              if the signal is noisy and long enough to afford it.

    Returns
    -------
    freqs, psd : (n_freqs,) arrays
    """
    n = len(signal)
    seg = nperseg if nperseg is not None else n
    seg = min(seg, n)
    freqs, psd = welch(signal, fs=1.0 / dt, nperseg=seg, detrend="constant")
    return freqs, psd


def compute_spanwise_psd(field: np.ndarray, dt: float,
                         nperseg: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    PSD at every spanwise station at once (vectorised Welch along axis 0).

    Parameters
    ----------
    field : ndarray, shape (n_snapshots, n_stations)
            The data to analyse
        dt : float
            Sample interval [s]
    nperseg : see compute_psd (defaults to the full record for the same reason)

    Returns
    -------
    freqs   : (n_freqs,)
    psd_map : (n_freqs, n_stations)
    """
    n = field.shape[0]
    seg = nperseg if nperseg is not None else n
    seg = min(seg, n)
    freqs, psd_map = welch(field, fs=1.0 / dt, axis=0, nperseg=seg,
                           detrend="constant")
    return freqs, psd_map


def compute_spanwise_phase(field: np.ndarray, dt: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase of every frequency component at every spanwise station, from a
    single direct FFT of the (mean-removed) full record. Consistent with
    the full-record Welch PSD used elsewhere in this script (Welch's method
    itself discards phase information through its segment-averaging, so a
    direct FFT is used here instead).

    Parameters
    ----------
    field : ndarray, shape (n_snapshots, n_stations)
        The data to analyse
    dt : float
        Sample interval [s]

    Returns
    -------
    freqs : (n_freqs,)
        Frequency axis
    phase_map : (n_freqs, n_stations)
        Phase in radians, wrapped to (-pi, pi]
    mag_map : (n_freqs, n_stations)
        Corresponding FFT magnitude (for masking phase values where there is
        no real spectral content - cf. plot_spanwise_phase_map)
    """
    n = field.shape[0]
    fluctuation = field - field.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(fluctuation, axis=0)
    return freqs, np.angle(spectrum), np.abs(spectrum)


def compute_spanwise_amplitude(field: np.ndarray, dt: float,
                               nperseg: Optional[int] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided *peak amplitude* spectrum at every spanwise station, in the
    same physical units as `field` (e.g. N/m) -- i.e. how much each
    frequency actually contributes to the force, as opposed to the PSD's
    (N/m)^2/Hz "density" units.

    Uses scipy.signal.welch with scaling="spectrum" rather than a manual
    FFT, so that scipy's own window-gain normalization is used instead of
    a hand-rolled correction (important if nperseg is ever reduced below
    the full record and a taper window is applied). With scaling="spectrum",
    Welch returns the mean-square (one-sided) magnitude spectrum, whose
    square root is the RMS amplitude of a sinusoidal component at that
    frequency; the corresponding peak (zero-to-peak) amplitude is sqrt(2)
    times larger for every bin *except* DC (f=0) and, if the segment length
    is even, the Nyquist bin (f = fs/2) -- neither of which is mirrored in
    the one-sided spectrum and so must not get that extra factor.

    Parameters
    ----------
    field   : (n_snapshots, n_stations) array
    dt      : sample interval [s]
    nperseg : Welch segment length; default = full record (see compute_psd).
              With the default, this is numerically close to reading the
              amplitude straight off a single FFT (as compute_spanwise_phase
              does for its magnitude), which is exact for this dataset's
              forcing harmonics since the record spans an exact number of
              forcing periods. Passing a shorter nperseg trades resolution
              for a lower-variance amplitude estimate of any broadband
              content, at the cost of some leakage-related peak broadening.

    Returns
    -------
    freqs   : (n_freqs,)
    amp_map : (n_freqs, n_stations)   peak amplitude, same units as `field`
    """
    n = field.shape[0]
    seg = nperseg if nperseg is not None else n
    seg = min(seg, n)
    freqs, Pxx_spec = welch(field, fs=1.0 / dt, axis=0, nperseg=seg,
                            detrend="constant", scaling="spectrum")

    rms_amp = np.sqrt(Pxx_spec)
    peak_amp = rms_amp * np.sqrt(2.0)
    peak_amp[0, :] = rms_amp[0, :]  # DC bin is not mirrored
    if seg % 2 == 0:  # even segment -> unique Nyquist bin
        peak_amp[-1, :] = rms_amp[-1, :]
    return freqs, peak_amp


def dominant_peaks(freqs: np.ndarray, psd: np.ndarray, n_peaks: int = 3,
                   f_min: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the dominant spectral peaks in a power spectral density estimate.

    Local maxima in the PSD are identified using
    :func:`scipy.signal.find_peaks`.
    The detected peaks are ranked according to spectral power, and the
    strongest ``n_peaks`` peaks are returned in descending order. Frequencies
    below ``f_min`` are ignored to avoid selecting the zero-frequency (DC)
    component.

    If no local maxima are detected, the global maximum of the PSD above
    ``f_min`` is returned instead.

    Parameters
    ----------
    freqs : ndarray, shape (n_frequencies,)
        Frequency vector corresponding to the PSD estimate, in Hz.
    psd : ndarray, shape (n_frequencies,)
        Power spectral density values.
    n_peaks : int, optional
        Maximum number of peaks to return. The default is 3.
    f_min : float, optional
        Minimum frequency considered during the peak search. Frequencies below
        this threshold are discarded before peak detection. The default is
        ``1e-6`` Hz.

    Returns
    -------
    peak_freqs : ndarray, shape (n_found,)
        Frequencies of the strongest spectral peaks, sorted by decreasing PSD.
    peak_powers : ndarray, shape (n_found,)
        PSD values corresponding to ``peak_freqs``.
    """
    mask = freqs >= f_min
    f, p = freqs[mask], psd[mask]
    idx, _ = find_peaks(p)
    if len(idx) == 0:
        k = int(np.argmax(p))
        return np.array([f[k]]), np.array([p[k]])
    order = np.argsort(p[idx])[::-1][:n_peaks]
    return f[idx[order]], p[idx[order]]


def integrate_over_span(y: np.ndarray, field: np.ndarray) -> np.ndarray:
    """
    Compute the span-integrated force time history.

    The sectional force distribution is numerically integrated along the blade
    span using the composite trapezoidal rule to obtain the total aerodynamic
    force acting on the blade at each time step.

    Parameters
    ----------
    y : ndarray, shape (n_stations,)
        Spanwise coordinates of the blade stations.
    field : ndarray, shape (n_snapshots, n_stations)
        Sectional force distribution. Each row corresponds to one time snapshot
        and each column to one spanwise station.

    Returns
    -------
    total_force : ndarray, shape (n_snapshots,)
        Span-integrated force as a function of time.
    """
    return np.trapezoid(field, y, axis=1)


# =============================================================================
# Plotting
# =============================================================================
def plot_station_psd(y: np.ndarray, t: np.ndarray,
                     Fx: np.ndarray, Fz: np.ndarray,
                     stations: Sequence[int], field: str, f_forcing: float,
                     n_harmonics: int, nperseg: Optional[int], out_dir: Path
                     ) -> None:
    """
    Plot power spectral density estimates at selected blade stations.

    Power spectral density (PSD) estimates are computed using Welch's method
    for the requested spanwise stations. Separate panels are produced for the
    edgewise and flapwise force components when both are analysed. Integer
    harmonics of the prescribed forcing frequency are drawn as vertical
    reference lines.

    The resulting figure is saved as ``01_station_psd.png``.

    Parameters
    ----------
    y : ndarray, shape (n_stations,)
        Spanwise station coordinates.
    t : ndarray, shape (n_snapshots,)
        Time vector.
    Fx : ndarray, shape (n_snapshots, n_stations)
        Edgewise sectional force distribution.
    Fz : ndarray, shape (n_snapshots, n_stations)
        Flapwise sectional force distribution.
    stations : sequence of int
        Indices of the spanwise stations whose spectra are plotted.
    field : {"fx", "fz", "both"}
        Force component(s) to visualise.
    f_forcing : float
        Prescribed excitation frequency in Hz.
    n_harmonics : int
        Number of forcing harmonics shown as reference lines.
    nperseg : int or None
        Segment length supplied to Welch's method.
    out_dir : pathlib.Path
        Output directory for the generated figure.

    Returns
    -------
    None
    """
    dt = t[1] - t[0]
    components = {"fx": ("Fx (edgewise)", Fx), "fz": ("Fz (flapwise)", Fz)}
    if field != "both":
        components = {field: components[field]}

    fig, axes = plt.subplots(1, len(components),
                             figsize=(6.5 * len(components), 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, (label, data) in zip(axes, components.values()):
        for st in stations:
            freqs, psd = compute_psd(data[:, st], dt, nperseg)
            ax.semilogy(freqs, psd, label=f"y = {y[st]:.1f} m")
        for h in range(1, n_harmonics + 1):
            ax.axvline(h * f_forcing, color="gray", ls="--", lw=0.8,
                       label="forcing harmonics" if h == 1 else None)
        ax.set_xlim(0, (n_harmonics + 1) * f_forcing)
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "01_station_psd.png", dpi=150)
    plt.close(fig)


def plot_spanwise_map(y: np.ndarray, t: np.ndarray,
                      Fx: np.ndarray, Fz: np.ndarray,
                      field: str, f_forcing: float, n_harmonics: int,
                      nperseg: Optional[int], out_dir: Path) -> None:
    """Frequency-vs-span heatmap of log10(PSD), one panel per force component.
    """
    dt = t[1] - t[0]
    components = {"fx": ("Fx (edgewise)", Fx), "fz": ("Fz (flapwise)", Fz)}
    if field != "both":
        components = {field: components[field]}

    fig, axes = plt.subplots(1, len(components),
                             figsize=(6.5 * len(components), 4.5),
                             squeeze=False)
    axes = axes[0]
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (label, data) in zip(axes, components.values()):
        freqs, psd_map = compute_spanwise_psd(data, dt, nperseg)
        keep = freqs <= f_max
        pcm = ax.pcolormesh(y, freqs[keep], np.log10(psd_map[keep, :] + 1e-30),
                            shading="auto", cmap="magma")
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(label)
        fig.colorbar(pcm, ax=ax, label="log10(PSD)")

    fig.tight_layout()
    fig.savefig(out_dir / "02_spanwise_psd_map.png", dpi=150)
    plt.close(fig)


def plot_spanwise_amplitude_map(y: np.ndarray, t: np.ndarray,
                                Fx: np.ndarray, Fz: np.ndarray,
                                field: str, f_forcing: float, n_harmonics: int,
                                nperseg: Optional[int], out_dir: Path) -> None:
    """
    Frequency-vs-span heatmap of peak amplitude (linear units, e.g. N/m),
    one panel per force component -- the "force contribution per frequency"
    view, complementing the log-scaled PSD map with a physically-scaled,
    linear one. Unlike the PSD/phase maps, a linear color scale is used
    deliberately here: it's the point of using an amplitude spectrum in the
    first place that the colour axis reads directly in force units rather
    than a squared, per-Hz, log-compressed quantity.
    """
    dt = t[1] - t[0]
    components = {"fx": ("Fx (edgewise)", Fx), "fz": ("Fz (flapwise)", Fz)}
    if field != "both":
        components = {field: components[field]}

    fig, axes = plt.subplots(1, len(components),
                             figsize=(6.5 * len(components), 4.5),
                             squeeze=False)
    axes = axes[0]
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (label, data) in zip(axes, components.values()):
        freqs, amp_map = compute_spanwise_amplitude(data, dt, nperseg)
        keep = freqs <= f_max
        pcm = ax.pcolormesh(y, freqs[keep], amp_map[keep, :], shading="auto",
                            cmap="magma", vmin=0.0)
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{label} amplitude")
        fig.colorbar(pcm, ax=ax, label="peak amplitude [N/m]")

    fig.tight_layout()
    fig.savefig(out_dir / "03_spanwise_amplitude_map.png", dpi=150)
    plt.close(fig)


def plot_spanwise_phase_map(y: np.ndarray, t: np.ndarray,
                            Fx: np.ndarray, Fz: np.ndarray,
                            field: str, f_forcing: float, n_harmonics: int,
                            mag_threshold_db: Optional[float], out_dir: Path
                            ) -> None:
    """
    Frequency-vs-span heatmap of phase (radians), one panel per force
    component, with dashed lines marking the forcing-frequency harmonics.

    Phase is only meaningful where there is real spectral content, so bins
    whose FFT magnitude is more than mag_threshold_db below that panel's
    peak magnitude are masked out (left blank) rather than showing
    essentially-random phase; set mag_threshold_db=None to disable this.
    """
    dt = t[1] - t[0]
    components = {"fx": ("Fx (edgewise)", Fx), "fz": ("Fz (flapwise)", Fz)}
    if field != "both":
        components = {field: components[field]}

    fig, axes = plt.subplots(1, len(components),
                             figsize=(6.5 * len(components), 4.5),
                             squeeze=False)
    axes = axes[0]
    f_max = (n_harmonics + 1) * f_forcing

    for ax, (label, data) in zip(axes, components.values()):
        freqs, phase_map, mag_map = compute_spanwise_phase(data, dt)
        keep = freqs <= f_max
        phase_plot = phase_map[keep, :]
        mag_plot = mag_map[keep, :]

        if mag_threshold_db is not None:
            threshold = mag_plot.max() * 10 ** (mag_threshold_db / 20.0)
            phase_plot = np.where(mag_plot >= threshold, phase_plot, np.nan)

        pcm = ax.pcolormesh(y, freqs[keep], phase_plot, shading="auto",
                            cmap="twilight", vmin=-np.pi, vmax=np.pi)
        for h in range(1, n_harmonics + 1):
            ax.axhline(h * f_forcing, color="lime", ls="--", lw=0.8)
        ax.set_xlabel("spanwise station y [m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{label} phase")
        cbar = fig.colorbar(pcm, ax=ax, label="phase [rad]")
        cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                             r"$\pi/2$", r"$\pi$"])

    fig.tight_layout()
    fig.savefig(out_dir / "03_spanwise_phase_map.png", dpi=150)
    plt.close(fig)


def plot_spectrogram(t: np.ndarray, signal: np.ndarray, f_forcing: float,
                     n_harmonics: int, title: str, filename: str,
                     out_dir: Path, nperseg: Optional[int] = None) -> None:
    """
    Time-frequency spectrogram of a single signal (e.g. the total force),
    computed via the modern scipy.signal.ShortTimeFFT class (the legacy
    scipy.signal.spectrogram function is deprecated in favour of it).

    A Hann window is used with 50% overlap between successive segments;
    ShortTimeFFT.spectrogram returns the same quantity (the squared
    magnitude of the STFT) as the old function, here scaled to a PSD via
    scale_to='psd'.
    """
    dt = t[1] - t[0]
    fs = 1.0 / dt
    n = len(signal)
    seg = nperseg if nperseg is not None else max(64, n // 12)
    seg = min(seg, n)
    hop = max(1, seg // 2)  # 50% overlap

    win = hann(seg, sym=False)
    sft = ShortTimeFFT(win, hop=hop, fs=fs, scale_to="psd")
    Sxx = sft.spectrogram(signal, detr="constant")   # (n_freqs, n_slices)
    freqs = sft.f
    times = sft.t(n) + t[0]

    f_max = (n_harmonics + 1) * f_forcing
    keep = freqs <= f_max

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcm = ax.pcolormesh(times, freqs[keep], np.log10(Sxx[keep, :] + 1e-30),
                        shading="auto", cmap="magma")
    for h in range(1, n_harmonics + 1):
        ax.axhline(h * f_forcing, color="cyan", ls="--", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, label="log10(PSD)")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_frequency_sweep(U_arr: np.ndarray, f_dom: np.ndarray,
                         f_forcing: float, out_dir: Path) -> None:
    """Dominant response frequency vs. wind speed (lock-in diagram)."""
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.5), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    ax.plot(U_arr, f_dom, "o-", color="C0")
    ax.axhline(f_forcing, color="r", ls="--", lw=1, label="forcing frequency")
    ax.set_xlabel("wind speed U [m/s]")
    ax.set_ylabel("dominant response frequency [Hz]")
    ax.set_title("Frequency lock-in diagram")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "04_frequency_sweep.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Wind speed sweep
# =============================================================================
def frequency_sweep(data_dir: Path, case: str, u_list: Sequence[int],
                    field: str, f_forcing: float, n_skip: int = 0,
                    n_last: Optional[int] = None, nperseg: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repeat the "dominant frequency of the span-integrated force" analysis
    for every wind speed in u_list, skipping any file that is missing.

    Returns
    -------
    U_found : (n_found,) wind speeds for which a file was loaded
    f_dom   : (n_found,) dominant frequency of the total force at each U
    """
    U_found: List[int] = []
    f_dom: List[float] = []

    for U in u_list:
        path = data_dir / f"{case}_u{U}.1HY"
        if not path.exists():
            print(f"  [sweep] skipping U = {U} m/s. "
                  f"file not found ({path.name})")
            continue

        y, t, Fx, Fz = load_hy(path, n_skip=n_skip, n_last=n_last)
        dt = t[1] - t[0]
        data = Fx if field == "fx" else Fz
        total = integrate_over_span(y, data)
        freqs, psd = compute_psd(total, dt, nperseg)
        f_peak, _ = dominant_peaks(freqs, psd, n_peaks=1)

        U_found.append(U)
        f_dom.append(float(f_peak[0]))

    return np.array(U_found, dtype=float), np.array(f_dom)


# =============================================================================
# Main
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

    # Station-level PSDs + summary table
    plot_station_psd(y, t, Fx, Fz, STATIONS, FIELD, F_FORCING, N_HARMONICS,
                     NPERSEG, OUTPUT_DIR)

    components = {"fx": Fx, "fz": Fz} if FIELD == "both" \
        else {FIELD: Fx if FIELD == "fx" else Fz}
    print("\n--- Dominant frequencies by station ---")
    print(f"{'component':>9} {'station':>9} {'y [m]':>8} "
          f"{'f_peak [Hz]':>12} {'f/f_forcing':>12}")
    for comp_name, data in components.items():
        for st in STATIONS:
            freqs, psd = compute_psd(data[:, st], dt, NPERSEG)
            f_peak, _ = dominant_peaks(freqs, psd, n_peaks=1)
            print(f"{comp_name:>9} {st:>9} {y[st]:>8.1f} {f_peak[0]:>12.4f} "
                  f"{f_peak[0] / F_FORCING:>12.3f}")

    # Frequency-vs-span map
    if DO_SPANWISE_MAP:
        plot_spanwise_map(y, t, Fx, Fz, FIELD, F_FORCING, N_HARMONICS,
                          NPERSEG, OUTPUT_DIR)

    # Frequency-vs-span amplitude ("force contribution") map
    if DO_SPANWISE_AMPLITUDE_MAP:
        plot_spanwise_amplitude_map(y, t, Fx, Fz, FIELD, F_FORCING,
                                    N_HARMONICS, NPERSEG, OUTPUT_DIR)

    # Frequency-vs-span phase map
    if DO_SPANWISE_PHASE_MAP:
        plot_spanwise_phase_map(y, t, Fx, Fz, FIELD, F_FORCING, N_HARMONICS,
                                PHASE_MASK_THRESHOLD_DB, OUTPUT_DIR)

    # Time-frequency spectrogram of the total (span-integrated) force
    if DO_SPECTROGRAM:
        total_fz = integrate_over_span(y, Fz)
        plot_spectrogram(t, total_fz, F_FORCING, N_HARMONICS,
                         "Spectrogram of span-integrated Fz",
                         "03_spectrogram_fz.png", OUTPUT_DIR)
        if FIELD == "both":
            total_fx = integrate_over_span(y, Fx)
            plot_spectrogram(t, total_fx, F_FORCING, N_HARMONICS,
                             "Spectrogram of span-integrated Fx",
                             "03_spectrogram_fx.png", OUTPUT_DIR)

    # Wind-speed sweep: lock-in diagram
    if DO_FREQUENCY_SWEEP:
        print(f"\n--- Frequency sweep over U for case '{CASE}' "
              f"(field = '{SWEEP_FIELD}') ---")
        U_arr, f_dom = frequency_sweep(DATA_DIR, CASE, U_SWEEP_LIST,
                                       SWEEP_FIELD, F_FORCING, N_SKIP, N_LAST,
                                       NPERSEG)
        if len(U_arr) > 0:
            for U, f in zip(U_arr, f_dom):
                tag = "  <-- locked to forcing" \
                    if abs(f - F_FORCING) < 0.05 * F_FORCING else ""
                print(f"  U = {U:5.1f} m/s : f_dom = {f:.4f} Hz "
                      f"({f / F_FORCING:.2f} x forcing){tag}")
            plot_frequency_sweep(U_arr, f_dom, F_FORCING, OUTPUT_DIR)
        else:
            print("  No files found for the requested sweep. "
                  "Skipping sweep plot.")

    print(f"\nFigures written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
