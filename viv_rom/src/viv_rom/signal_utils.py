"""Signal-processing helpers shared by the analysis methods."""
from __future__ import annotations

import numpy as np


def dominant_frequency(signal: np.ndarray, dt: float
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """Frequency of the largest non-zero peak in the amplitude spectrum.

    Parameters
    ----------
    signal : ndarray
        Signal to analyse (its mean is removed).
    dt : float
        Sampling interval.

    Returns
    -------
    dominant_freq : float
        Dominant non-zero frequency in the signal.
    freqs : ndarray
        FFT frequency bins [1/dt units].
    amp : ndarray
        Amplitude spectrum ``|rfft(signal - mean)|``.
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(signal - signal.mean()))
    if len(amp) < 2:
        return 0.0, freqs, amp
    k = np.argmax(amp[1:]) + 1  # skip the DC bin
    return freqs[k], freqs, amp
