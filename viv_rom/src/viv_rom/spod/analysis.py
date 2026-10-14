"""Descriptive diagnostics of an SPOD: harmonic content and spectral peaks.

All results are frozen dataclasses with ``summary_lines()`` so the text
report is assembled from the pieces in one place (the pipeline) and printed
and saved identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from viv_rom.spod.core import SPODResult


def _band_energy(energy: np.ndarray, bins: np.ndarray, band_bins: int
                 ) -> np.ndarray:
    """Power within ``+-band_bins`` of each bin, counting every bin once.

    With coarse resolution neighbouring harmonics are only a few bins
    apart and their bands overlap. A bin within reach of several centres is
    credited to the one with the strongest centre bin: its spectral leakage
    dominates there, so the weaker line is not credited with the skirt of
    the stronger one.
    """
    bins = np.asarray(bins)
    owner = {}
    for i, b in enumerate(bins):
        for j in range(max(b - band_bins, 0),
                       min(b + band_bins, len(energy) - 1) + 1):
            if j not in owner or energy[bins[owner[j]]] < energy[b]:
                owner[j] = i
    out = np.zeros(len(bins))
    for j, i in owner.items():
        out[i] += energy[j]
    return out


@dataclass(frozen=True, eq=False)
class HarmonicContent:
    """Power and modal structure at the forcing harmonics.

    Attributes
    ----------
    harmonics, bins, freqs, offset_bins : ndarray, shape (H,)
        The harmonic numbers, nearest frequency bins, exact frequencies and
        bin offsets (in bin widths).
    bin_power : ndarray, shape (H,)
        ``sum_k lambda_k`` at the bin (PSD).
    band_share : ndarray, shape (H,)
        Fraction of the total variance within ``+-band_bins`` bins (a bin
        shared by two harmonics is credited to the stronger one).
    mode_fractions : ndarray, shape (H, n_show)
        ``lambda_k / sum lambda`` at the bin for the leading modes.
    n_for_threshold : ndarray of int, shape (H,)
        Modes holding ``threshold`` of the power at the bin.
    threshold : float
    band_bins : int
    """
    harmonics: np.ndarray
    bins: np.ndarray
    freqs: np.ndarray
    offset_bins: np.ndarray
    bin_power: np.ndarray
    band_share: np.ndarray
    mode_fractions: np.ndarray
    n_for_threshold: np.ndarray
    threshold: float
    band_bins: int

    def summary_lines(self) -> list[str]:
        n_show = self.mode_fractions.shape[1]
        lines = [f"--- SPOD content at the forcing harmonics "
                 f"(band = +-{self.band_bins} bin) ---",
                 f"{'n':>3} {'f [Hz]':>8} {'bin offset':>11} "
                 f"{'variance share':>15} "
                 + " ".join(f"{'mode ' + str(k + 1):>8}"
                            for k in range(n_show))
                 + " {'modes for "
                 + f"{format(100 * self.threshold, 'g') + '%':>3}"]
        for i, n in enumerate(self.harmonics):
            shares = " ".join(f"{100 * v:7.2f}%"
                              for v in self.mode_fractions[i])
            lines.append(
                f"{n:3d} {self.freqs[i]:8.4f} {self.offset_bins[i]:+11.3f} "
                f"{100 * self.band_share[i]:14.2f}% {shares} "
                f"{self.n_for_threshold[i]:11d}")
        lines.append("(mode columns: share of the power at the harmonic's "
                     "bin; a coherent, locked response sits in mode 1)")
        return lines


def harmonic_content(spod: SPODResult, f_forcing: float, n_harmonics: int,
                     n_show: int = 3, band_bins: int = 1,
                     threshold: float = 0.99) -> HarmonicContent:
    """Characterise the SPOD at ``f_forcing`` and its harmonics."""
    if n_harmonics < 1 or band_bins < 0:
        raise ValueError("n_harmonics must be >= 1 and band_bins >= 0.")
    nyquist = spod.freqs[-1]
    if n_harmonics * f_forcing >= nyquist:
        raise ValueError(
            f"Harmonic {n_harmonics} ({n_harmonics * f_forcing:.3g} Hz) is "
            f"not below the Nyquist frequency ({nyquist:.3g} Hz).")
    n_show = min(n_show, spod.n_modes)
    harmonics = np.arange(1, n_harmonics + 1)
    freqs = harmonics * f_forcing
    bins = np.array([spod.bin_of(f) for f in freqs])
    energy, var = spod.energy, spod.variance
    band = _band_energy(energy, bins, band_bins) * spod.df
    fractions = spod.mode_fractions()[bins, :n_show]
    return HarmonicContent(
        harmonics=harmonics, bins=bins, freqs=freqs,
        offset_bins=np.array([spod.offset_bins(f) for f in freqs]),
        bin_power=energy[bins],
        band_share=band / var if var > 0 else np.zeros(len(bins)),
        mode_fractions=fractions,
        n_for_threshold=np.array([spod.modes_for_energy(b, threshold)
                                  for b in bins]),
        threshold=threshold, band_bins=band_bins)


@dataclass(frozen=True, eq=False)
class SpectralPeaks:
    """Strongest peaks of the total SPOD power spectrum.

    Attributes
    ----------
    freqs, ratio, share, lead_share : ndarray, shape (n_peaks,)
        Peak frequency [Hz], its ratio to the forcing frequency, share of
        the total variance within the peak's band, and share of mode 1 at
        the peak bin.
    """
    freqs: np.ndarray
    ratio: np.ndarray
    share: np.ndarray
    lead_share: np.ndarray
    band_bins: int

    def summary_lines(self) -> list[str]:
        lines = ["--- Strongest spectral peaks of the SPOD power ---"]
        if len(self.freqs) == 0:
            return lines + ["  no peaks found"]
        lines.append(f"{'f [Hz]':>9} {'f/f_forcing':>12} "
                     f"{'variance share':>15} {'mode 1 share':>13}")
        for f, r, s, l in zip(self.freqs, self.ratio, self.share,
                              self.lead_share):
            lines.append(f"{f:9.4f} {r:12.3f} {100 * s:14.2f}% "
                         f"{100 * l:12.2f}%")
        return lines


def spectral_peaks(spod: SPODResult, f_forcing: float, n_peaks: int = 5,
                   band_bins: int = 1, min_share: float = 1e-3
                   ) -> SpectralPeaks:
    """Find the strongest spectral lines in the total SPOD power spectrum.

    The total power spectrum is ``E(f) = sum_k lambda_k(f)``, the sum of the
    SPOD eigenvalues over all modes at each frequency bin (a one-sided,
    weighted power spectral density). The function locates its local
    maxima, discards negligible ones, ranks the rest by peak height and
    reports for each of the strongest ones where it is, how much of the
    total variance it holds, and how coherent it is.

    It is mainly meant to reveal lines that are *not* forcing harmonics,
    for example a free vortex-shedding frequency appearing next to the
    forced response. The harmonics themselves are tabulated separately by
    :func:`harmonic_content`.

    Parameters
    ----------
    spod : SPODResult
        Result of :func:`~viv_rom.spod.core.compute_spod`.
    f_forcing : float
        Forcing frequency in Hz. Only used to express each peak frequency
        as a multiple of it (``SpectralPeaks.ratio``).
    n_peaks : int, default 5
        Maximum number of peaks returned, strongest first. Fewer are
        returned if fewer peaks survive the ``min_share`` filter.
    band_bins : int, default 1
        Half-width, in frequency bins, of the band summed around each
        peak when computing its power. ``1`` (three bins in total) captures
        a line that sits on a bin centre when a Hann window is used. Use a
        larger value for broader lines.
    min_share : float, default 1e-3
        Minimum fraction of the total variance that a peak's band must
        hold for the peak to be kept. Removes the many small random
        maxima of a broadband spectrum. Lower it to look for weak lines.

    Returns
    -------
    SpectralPeaks
        Arrays of length ``<= n_peaks``, ordered from the strongest peak:

        ``freqs`` : peak frequencies in Hz.
        ``ratio`` : ``freqs / f_forcing``. Close to an integer for forcing
        harmonics, non-integer for lines at their own frequency.
        ``share`` : band power divided by the total variance (0 to 1).
        ``lead_share`` : ``lambda_1 / sum_k lambda_k`` at the peak bin.
        Near 1 means one coherent structure, low means the power is
        spread over several modes (incoherent / broadband content).
        ``band_bins`` : the value used.

    Notes
    -----
    * The first and last bins (0 Hz and the Nyquist frequency) cannot be
      peaks, because a local maximum needs a neighbour on each side.
    * A line closer than two bins to a much stronger line is not a local
      maximum, because leakage of the stronger line is larger than it at
      the bin between them. At the default two-period blocks the second
      harmonic is only two bins from the fundamental and can be missed.
      Use longer blocks (``--segment-periods 4``) to separate such lines.
    * ``share`` sums plain overlapping bands. Peaks closer than
      ``2 * band_bins + 1`` bins both count the bins between them, so
      their shares can add up to slightly more than their true total.
    * Peaks are ranked by the height of the single peak bin, not by band
      power.

    See Also
    --------
    harmonic_content : Power and modal structure at the forcing harmonics.
    """
    energy = spod.energy  # total power per bin, sum over modes
    variance = spod.variance  # sum(energy) * df: total weighted power
    n_bins = len(energy)

    def band_power(bin_index: int) -> float:
        """Power (PSD summed over the band, times df) around one bin."""
        lo = max(bin_index - band_bins, 0)  # clamp at 0 Hz
        hi = min(bin_index + band_bins + 1, n_bins)  # clamp at Nyquist
        return float(energy[lo:hi].sum() * spod.df)

    # Every local maximum of the spectrum. height=0 filters nothing
    # (all energies are >= 0)
    idx, props = find_peaks(energy, height=0.0)
    heights = props["peak_heights"]

    # Drop noise ripples: keep a peak only if its band holds at least
    # `min_share` of the total variance.
    keep = np.array([band_power(b) >= min_share * variance for b in idx],
                    dtype=bool)
    idx, heights = idx[keep], heights[keep]

    # Rank by the height of the peak bin, strongest first, and keep the best
    # `n_peaks`.
    order = np.argsort(heights)[::-1][:n_peaks]
    idx = idx[order]

    # 4) Quantities reported for each surviving peak.
    band = np.array([band_power(b) for b in idx])
    share = band / variance if variance > 0 else np.zeros(len(idx))
    # Fraction of the power at the peak bin that mode 1 carries.
    lead_share = (spod.mode_fractions()[idx, 0] if len(idx)
                  else np.array([]))

    return SpectralPeaks(
        freqs=spod.freqs[idx],
        ratio=spod.freqs[idx] / f_forcing,
        share=share,
        lead_share=lead_share,
        band_bins=band_bins)


def fundamental_mode_lines(spod: SPODResult, f_forcing: float,
                           locked=None, injection=None, n_show: int = 5
                           ) -> list[str]:
    """Table of the leading modes at the forcing frequency.

    ``locked`` (:class:`~viv_rom.spod.locking.PhaseLockedResponse`) and
    ``injection`` (:class:`~viv_rom.spod.injection.SPODInjection`) are
    optional; their columns are left out when they are ``None``.
    """
    b = spod.bin_of(f_forcing)
    n_show = min(n_show, spod.n_modes)
    frac = spod.mode_fractions()[b]
    header = f"{'mode':>4} {'power %':>9} {'cum %':>8}"
    if locked is not None:
        header += f" {'coherence':>10} {'locked %':>9}"
    if injection is not None:
        header += f" {'P_k [W]':>11} {'share of P':>11}"
    lines = [f"--- SPOD modes at the forcing frequency "
             f"(bin {spod.freqs[b]:.4f} Hz) ---", header]
    for k in range(n_show):
        row = f"{k + 1:4d} {100 * frac[k]:9.2f} " \
              f"{100 * frac[:k + 1].sum():8.2f}"
        if locked is not None:
            own = locked.locked_energy[0, k] / max(spod.eigenvalues[b, k],
                                                   np.finfo(float).tiny)
            row += (f" {locked.mode_locking[0, k]:10.3f} "
                    f"{100 * min(own, 1.0):8.1f}%")
        if injection is not None:
            p = injection.mode_power[k]
            row += (f" {p:11.4g} "
                    f"{100 * p / injection.total_power:10.1f}%"
                    if injection.total_power != 0 else f" {p:11.4g} {'-':>11}")
        lines.append(row)
    lines.append("(coherence: squared correlation of the mode's coefficients "
                 "with the forced displacement; locked %: share of the "
                 "mode's own power that is phase-locked)")
    return lines
