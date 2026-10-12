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


def fit_harmonics(signals: np.ndarray, t: np.ndarray, f: float,
                  n_harmonics: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit of ``f`` and its harmonics to each signal.

    Each signal is modelled as

        ``s(t) ~ c_0 + sum_{n=1..H} A_n * sin(n * 2 pi f * t + phi_n)``

    with all harmonics fitted jointly. Unlike reading FFT bins this is exact
    for a non-integer number of periods and keeps the harmonics from
    leaking into each other.

    Parameters
    ----------
    signals : ndarray, shape (n_signals, n_t)
    t : ndarray, shape (n_t,)
        Absolute time [s]; the origin defines the zero of the phase.
    f : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Harmonics ``1..n_harmonics``.

    Returns
    -------
    amplitude : ndarray, shape (n_signals, n_harmonics)
        ``A_n`` in the units of ``signals``.
    phase : ndarray, shape (n_signals, n_harmonics)
        ``phi_n`` [rad] in ``(-pi, pi]``: the lead relative to
        ``sin(n * 2 pi f * t)``. Not masked (meaningless where ``A_n`` is
        negligible).
    explained : ndarray, shape (n_signals,)
        Fraction of each signal's variance captured by the fit.

    Raises
    ------
    ValueError
        If the highest harmonic is not below the Nyquist frequency or the
        record is shorter than one period.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=float))
    t = np.asarray(t, dtype=float)
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1.")
    dt = np.median(np.diff(t))
    if n_harmonics * f >= 0.5 / dt:
        raise ValueError(
            f"Harmonic {n_harmonics} ({n_harmonics * f:.3g} Hz) is above the "
            f"Nyquist frequency ({0.5 / dt:.3g} Hz).")
    if t[-1] - t[0] < 1.0 / f:
        raise ValueError("The time series is shorter than one forcing "
                         "period; phases are not identifiable.")

    omega = 2 * np.pi * f
    n = np.arange(1, n_harmonics + 1)
    X = np.column_stack(
        [np.ones_like(t)]
        + [fn(k * omega * t) for k in n for fn in (np.cos, np.sin)])
    a = signals.T                                      # (n_t, n_signals)
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)       # (1 + 2H, n_signals)

    alpha = coef[1::2].T                               # cos coefficients
    beta = coef[2::2].T                                # sin coefficients
    amplitude = np.hypot(alpha, beta)
    phase = np.arctan2(alpha, beta)                    # A sin(n w t + phi)

    resid = np.sum((a - X @ coef) ** 2, axis=0)
    total = np.sum((a - a.mean(axis=0)) ** 2, axis=0)
    explained = np.where(total > 0,
                         1.0 - resid / np.where(total > 0, total, 1), 0.0)
    return amplitude, phase, explained
