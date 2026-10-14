"""Welch-type Fourier blocks: the estimator behind SPOD.

The (de-meaned) snapshot matrix ``x(p, t)`` is cut into ``n_b`` overlapping
segments of ``n_fft`` samples, each tapered with a window and Fourier
transformed. For every frequency bin ``k`` the result is a matrix
``Q_k`` of shape ``(n_points, n_b)`` whose columns are realisations of the
Fourier mode at that frequency.

Scaling is chosen so the eigenvalues of the SPOD are a *one-sided power
spectral density*: ``sum_k lambda(f_k) * df`` equals the (weighted)
mean-square of the signal. A deterministic sinusoid exactly on a bin is
therefore reported with the right power, whatever the window.

This module knows nothing about SPOD, weights or the blade; it only
produces blocks, so the same estimator is also applied to reference signals
(e.g. the forced displacement) to get directly comparable Fourier blocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window


@dataclass(frozen=True)
class SegmentSpec:
    """How a record is cut into blocks.

    Attributes
    ----------
    n_fft : int
        Samples per block. The frequency resolution is ``1 / (n_fft dt)``.
    n_overlap : int
        Samples shared by consecutive blocks.
    window : str
        Any name accepted by :func:`scipy.signal.get_window`.
    """
    n_fft: int
    n_overlap: int
    window: str = "hann"

    def __post_init__(self):
        if self.n_fft < 8:
            raise ValueError(f"n_fft must be >= 8, got {self.n_fft}.")
        if not 0 <= self.n_overlap < self.n_fft:
            raise ValueError("n_overlap must satisfy 0 <= n_overlap < n_fft "
                             f"(got {self.n_overlap} for n_fft = "
                             f"{self.n_fft}).")

    @classmethod
    def for_period(cls, dt: float, f_forcing: float, periods: float = 2.0,
                   overlap: float = 0.5, window: str = "hann"
                   ) -> "SegmentSpec":
        """Blocks spanning ``periods`` forcing periods.

        With a whole number of samples per period this puts the forcing
        frequency and all its harmonics exactly on frequency bins, which
        keeps their power from leaking into neighbouring bins.
        """
        if periods <= 0 or not 0 <= overlap < 1:
            raise ValueError("periods must be > 0 and 0 <= overlap < 1.")
        n_fft = max(8, int(round(periods / (f_forcing * dt))))
        return cls(n_fft, int(round(overlap * n_fft)), window)

    @property
    def hop(self) -> int:
        return self.n_fft - self.n_overlap

    def n_blocks(self, n_samples: int) -> int:
        if n_samples < self.n_fft:
            return 0
        return (n_samples - self.n_fft) // self.hop + 1

    def taper(self) -> np.ndarray:
        return get_window(self.window, self.n_fft)


@dataclass(frozen=True, eq=False)
class FourierBlocks:
    """Fourier coefficients of all blocks.

    Attributes
    ----------
    freqs : ndarray, shape (n_f,)
        Bin frequencies ``0 ... Nyquist`` [Hz].
    q : ndarray, complex, shape (n_f, n_rows, n_b)
        Scaled coefficients (see module docstring).
    spec : SegmentSpec
    dt : float
    """
    freqs: np.ndarray
    q: np.ndarray
    spec: SegmentSpec
    dt: float

    @property
    def n_blocks(self) -> int:
        return self.q.shape[2]

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])

    def bin_of(self, f: float) -> int:
        """Index of the bin nearest to ``f``."""
        return int(np.argmin(np.abs(self.freqs - f)))

    def offset_bins(self, f: float) -> float:
        """``f`` minus its nearest bin, in bin widths (0 = exactly on bin)."""
        return float(f / self.df - self.bin_of(f))


def fourier_blocks(data: np.ndarray, dt: float, spec: SegmentSpec
                   ) -> FourierBlocks:
    """Windowed Fourier transform of the overlapping blocks of ``data``.

    Parameters
    ----------
    data : ndarray, shape (n_rows, n_t) or (n_t,)
        Real signals, one per row (already de-meaned if desired).
    dt : float
        Sampling interval [s].
    spec : SegmentSpec

    Raises
    ------
    ValueError
        If the record is too short for at least two blocks.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_t = data.shape[1]
    n_b = spec.n_blocks(n_t)
    if n_b < 2:
        raise ValueError(
            f"A record of {n_t} samples gives {n_b} block(s) of "
            f"{spec.n_fft} samples; SPOD needs at least 2. Use a shorter "
            "segment (smaller --segment-periods / --n-fft), less overlap "
            "or a longer record (smaller --n-skip).")
    taper = spec.taper()
    segments = sliding_window_view(data, spec.n_fft, axis=1)
    segments = segments[:, ::spec.hop][:, :n_b]          # (rows, n_b, n_fft)
    spectrum = np.fft.rfft(segments * taper, axis=-1)    # (rows, n_b, n_f)

    freqs = np.fft.rfftfreq(spec.n_fft, d=dt)
    one_sided = np.full(len(freqs), 2.0)
    one_sided[0] = 1.0
    if spec.n_fft % 2 == 0:
        one_sided[-1] = 1.0
    scale = np.sqrt(one_sided * dt / np.sum(taper**2))
    q = np.transpose(spectrum, (2, 0, 1)) * scale[:, None, None]
    return FourierBlocks(freqs=freqs, q=q, spec=spec, dt=dt)
