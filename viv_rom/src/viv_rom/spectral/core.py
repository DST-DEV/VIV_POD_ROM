"""Spectral estimates (scipy.signal based).

Array convention: sectional fields have shape ``(n_stations, n_steps)`` and
spanwise spectral maps are returned as ``(n_stations, n_freqs)``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
# from scipy.signal import ShortTimeFFT, welch
from scipy.signal import ShortTimeFFT, find_peaks, welch
from scipy.signal.windows import hann

# Labels of the blade-frame components; other frames bring their own
# (``ForceFrame.labels``) and the plotting functions accept them.
COMPONENT_LABELS = {"fx": "Fx (edgewise)", "fz": "Fz (flapwise)"}


def select_components(field: str, components: Mapping[str, np.ndarray]
                      ) -> dict[str, np.ndarray]:
    """Pick ``"both"`` (all) or one component from ``{key: data}``."""
    if field == "both":
        return dict(components)
    if field not in components:
        raise ValueError(f"Unknown field option: {field!r} "
                         f"(use one of {(*components, 'both')}).")
    return {field: components[field]}


def _segment_length(n: int, nperseg: int | None) -> int:
    seg = nperseg if nperseg is not None else n
    return min(seg, n)


def compute_psd(signal: np.ndarray, dt: float, nperseg: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of a single time signal via Welch's method.

    Parameters
    ----------
    signal : (n_samples,) array
    dt : float
        Sample interval [s].
    nperseg : int, optional
        Welch segment length; default is the full record (a single windowed
        periodogram, no segment-averaging). This dataset is short relative to
        the frequencies of interest (order 10-15 forcing periods per file),
        so splitting it into sub-segments destroys the frequency resolution
        needed to distinguish nearby frequencies (e.g. a 0.45 Hz response
        from the 0.624 Hz forcing). Pass an explicit ``nperseg`` to trade
        resolution for variance reduction if the signal is noisy and long
        enough to afford it.

    Returns
    -------
    freqs, psd : (n_freqs,) arrays
    """
    seg = _segment_length(len(signal), nperseg)
    return welch(signal, fs=1.0 / dt, nperseg=seg, detrend="constant")


def compute_spanwise_psd(field: np.ndarray, dt: float,
                         nperseg: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """PSD at every spanwise station at once (vectorised Welch).

    Parameters
    ----------
    field : ndarray, shape (n_stations, n_steps)
    dt : float
        Sample interval [s].
    nperseg : see :func:`compute_psd` (defaults to the full record).

    Returns
    -------
    freqs : (n_freqs,)
    psd_map : (n_stations, n_freqs)
    """
    seg = _segment_length(field.shape[-1], nperseg)
    return welch(field, fs=1.0 / dt, axis=-1, nperseg=seg, detrend="constant")


def compute_spanwise_phase(field: np.ndarray, dt: float
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase of every frequency component at every spanwise station.

    Uses a single direct FFT of the (mean-removed) full record, consistent
    with the full-record Welch PSD (Welch's segment-averaging discards phase,
    so a direct FFT is used instead).

    Parameters
    ----------
    field : ndarray, shape (n_stations, n_steps)
    dt : float
        Sample interval [s].

    Returns
    -------
    freqs : (n_freqs,)
    phase_map : (n_stations, n_freqs)
        Phase in radians, wrapped to (-pi, pi].
    mag_map : (n_stations, n_freqs)
        FFT magnitude, for masking phase where there is no real spectral
        content.
    """
    n = field.shape[-1]
    fluctuation = field - field.mean(axis=-1, keepdims=True)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(fluctuation, axis=-1)
    return freqs, np.angle(spectrum), np.abs(spectrum)


def compute_spanwise_amplitude(field: np.ndarray, dt: float,
                               nperseg: int | None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
    """One-sided *peak amplitude* spectrum at every spanwise station.

    The result is in the same physical units as ``field`` (e.g. N/m), i.e.
    how much each frequency contributes to the force, as opposed to the
    PSD's (N/m)^2/Hz density units.

    Uses ``scipy.signal.welch(scaling="spectrum")`` so scipy's own
    window-gain normalisation applies. That returns the one-sided mean-square
    spectrum, whose square root is the RMS amplitude of a sinusoid at that
    frequency. The zero-to-peak amplitude is sqrt(2) times larger for every
    bin *except* DC and, for an even segment length, the Nyquist bin
    (neither is mirrored in the one-sided spectrum).

    Parameters
    ----------
    field : ndarray, shape (n_stations, n_steps)
    dt : float
        Sample interval [s].
    nperseg : int, optional
        Welch segment length; default = full record (see
        :func:`compute_psd`). With the default this is numerically close to
        reading the amplitude off a single FFT, which is exact for this
        dataset's forcing harmonics because the record spans an exact number
        of forcing periods.

    Returns
    -------
    freqs : (n_freqs,)
    amp_map : (n_stations, n_freqs)
        Peak amplitude, same units as ``field``.
    """
    seg = _segment_length(field.shape[-1], nperseg)
    freqs, Pxx_spec = welch(field, fs=1.0 / dt, axis=-1, nperseg=seg,
                            detrend="constant", scaling="spectrum")

    rms_amp = np.sqrt(Pxx_spec)
    peak_amp = rms_amp * np.sqrt(2.0)
    peak_amp[:, 0] = rms_amp[:, 0]  # DC bin is not mirrored
    if seg % 2 == 0:  # even segment -> unique Nyquist bin
        peak_amp[:, -1] = rms_amp[:, -1]
    return freqs, peak_amp


def dominant_peaks(freqs: np.ndarray, psd: np.ndarray, n_peaks: int = 3,
                   f_min: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Dominant spectral peaks of a PSD estimate.

    Local maxima are found with :func:`scipy.signal.find_peaks`, ranked by
    spectral power, and the strongest ``n_peaks`` are returned in descending
    order. Frequencies below ``f_min`` are ignored (avoids the DC bin). If no
    local maximum exists, the global maximum above ``f_min`` is returned.

    Parameters
    ----------
    freqs : ndarray, shape (n_frequencies,)
        Frequency vector [Hz].
    psd : ndarray, shape (n_frequencies,)
        Power spectral density values.
    n_peaks : int, optional
        Maximum number of peaks to return. Default is 3.
    f_min : float, optional
        Minimum frequency considered. Default is ``1e-6`` Hz.

    Returns
    -------
    peak_freqs : ndarray, shape (n_found,)
        Frequencies of the strongest peaks, by decreasing PSD.
    peak_powers : ndarray, shape (n_found,)
        PSD values at ``peak_freqs``.
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
    """Span-integrated force time history (composite trapezoidal rule).

    Parameters
    ----------
    y : ndarray, shape (n_stations,)
        Spanwise coordinates.
    field : ndarray, shape (n_stations, n_steps)
        Sectional force distribution.

    Returns
    -------
    total_force : ndarray, shape (n_steps,)
    """
    return np.trapezoid(field, y, axis=0)


def compute_spectrogram(t: np.ndarray, signal: np.ndarray,
                        nperseg: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-frequency PSD of a single signal via :class:`ShortTimeFFT`.

    A Hann window with 50 % overlap is used; the squared STFT magnitude is
    scaled to a PSD (``scale_to="psd"``).

    Parameters
    ----------
    t : ndarray, shape (n_steps,)
        Time vector (uniform sampling).
    signal : ndarray, shape (n_steps,)
    nperseg : int, optional
        Segment length; default ``max(64, n_steps // 12)``.

    Returns
    -------
    times : (n_slices,)
    freqs : (n_freqs,)
    Sxx : (n_freqs, n_slices)
    """
    fs = 1.0 / (t[1] - t[0])
    n = len(signal)
    seg = nperseg if nperseg is not None else max(64, n // 2)
    seg = min(seg, n)
    hop = max(1, seg // 2)  # 50% overlap

    sft = ShortTimeFFT(hann(seg, sym=False), hop=hop, fs=fs, scale_to="psd")
    Sxx = sft.spectrogram(signal, detr="constant")
    return sft.t(n) + t[0], sft.f, Sxx
