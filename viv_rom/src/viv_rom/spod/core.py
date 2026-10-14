"""Spectral proper orthogonal decomposition (Towne, Schmidt & Colonius 2018).

For every frequency ``f_k`` the cross-spectral density (CSD) of the field is
estimated from the Fourier blocks ``Q_k`` (see :mod:`viv_rom.spod.blocks`) and
its eigen-decomposition under the weight ``W`` is computed::

    S_k = Q_k Q_k^H / n_b            (CSD estimate)
    S_k W Phi_k = Phi_k Lambda_k     <=>   (Q_k^H W Q_k / n_b) Psi = Psi Lambda

    Phi_k = Q_k Psi Lambda^{-1/2} / sqrt(n_b)        (W-orthonormal modes)

``lambda_{k,1} >= lambda_{k,2} >= ...`` is the power (one-sided PSD,
weighted) carried by the SPOD modes at that frequency; their number is
limited by the number of blocks. A mode is a *complex* spatial shape: its
amplitude is the spanwise distribution of the oscillation and its phase the
travelling/standing character along the span. Because every mode oscillates
at a single frequency, a deterministic, coherent response (e.g. lock-in)
collapses onto one mode, whereas the POD of the same data needs several.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viv_rom.spod.blocks import FourierBlocks, SegmentSpec


@dataclass(frozen=True, eq=False)
class SPODResult:
    """Result of :func:`compute_spod`.

    Attributes
    ----------
    freqs : ndarray, shape (n_f,)
        Bin frequencies [Hz].
    eigenvalues : ndarray, shape (n_f, n_modes)
        ``lambda_k`` per frequency, descending (zero where a mode is not
        resolved). One-sided weighted PSD.
    modes : ndarray, complex, shape (n_f, n_points, n_modes)
        Spatial modes, ``Phi^H W Phi = I`` (all-zero where unresolved).
        Their overall phase is arbitrary.
    weights : ndarray, shape (n_points,)
        Diagonal of ``W``.
    spec : SegmentSpec
    dt : float
    """
    freqs: np.ndarray
    eigenvalues: np.ndarray
    modes: np.ndarray
    weights: np.ndarray
    spec: SegmentSpec
    dt: float

    @property
    def n_blocks(self) -> int:
        return self.eigenvalues.shape[1]

    @property
    def n_modes(self) -> int:
        return self.eigenvalues.shape[1]

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])

    @property
    def energy(self) -> np.ndarray:
        """Total power per frequency, ``sum_k lambda_k``, shape (n_f,)."""
        return self.eigenvalues.sum(axis=1)

    @property
    def variance(self) -> float:
        """``sum_f energy(f) df``: the weighted mean-square of the field."""
        return float(self.energy.sum() * self.df)

    def mode_fractions(self) -> np.ndarray:
        """``lambda_k / sum lambda`` per frequency, shape (n_f, n_modes)."""
        total = self.energy[:, None]
        return np.divide(self.eigenvalues, total,
                         out=np.zeros_like(self.eigenvalues), where=total > 0)

    def bin_of(self, f: float) -> int:
        return int(np.argmin(np.abs(self.freqs - f)))

    def offset_bins(self, f: float) -> float:
        return float(f / self.df - self.bin_of(f))

    def modes_for_energy(self, bin_index: int, threshold: float = 0.99
                         ) -> int:
        """Smallest number of modes holding ``threshold`` of a bin's power."""
        cum = np.cumsum(self.eigenvalues[bin_index])
        if cum[-1] <= 0:
            return 0
        return int(np.searchsorted(cum / cum[-1], threshold) + 1)

    def coefficients(self, blocks: FourierBlocks) -> np.ndarray:
        """Modal coefficients ``a_k = Phi_k^H W Q``, shape (n_f, n_modes, n_b).

        ``sum_l |a_kl|^2 / n_b = lambda_k`` and ``Q = sum_k Phi_k a_k`` for
        the resolved modes.
        """
        return np.einsum("fpk,p,fpb->fkb", self.modes.conj(), self.weights,
                         blocks.q, optimize=True)

    def coefficients_at(self, blocks: FourierBlocks, bin_index: int
                        ) -> np.ndarray:
        """:meth:`coefficients` of a single bin, shape (n_modes, n_b)."""
        return np.einsum("pk,p,pb->kb", self.modes[bin_index].conj(),
                         self.weights, blocks.q[bin_index], optimize=True)


def compute_spod(blocks: FourierBlocks, weights: np.ndarray,
                 rel_tol: float = 1e-12) -> SPODResult:
    """SPOD of Fourier blocks under the diagonal weight ``weights``.

    Parameters
    ----------
    blocks : FourierBlocks
        Output of :func:`~viv_rom.spod.blocks.fourier_blocks`.
    weights : ndarray, shape (n_points,)
        Positive diagonal of the energy norm.
    rel_tol : float
        Eigenvalues below ``rel_tol`` times the largest one are treated as
        unresolved (their modes are not defined).
    """
    w = np.asarray(weights, dtype=float)
    q = blocks.q
    if w.shape != (q.shape[1],) or np.any(w <= 0):
        raise ValueError("weights must be positive with one entry per "
                         f"point ({q.shape[1]}).")
    n_b = q.shape[2]

    # Small (n_b x n_b) problem per frequency instead of (n_p x n_p)
    gram = np.einsum("fpa,p,fpb->fab", q.conj(), w, q, optimize=True) / n_b
    gram = 0.5 * (gram + np.conj(np.swapaxes(gram, 1, 2)))
    lam, psi = np.linalg.eigh(gram)
    lam, psi = lam[:, ::-1], psi[:, :, ::-1]
    lam = np.clip(lam, 0.0, None)

    resolved = lam > rel_tol * max(lam.max(), np.finfo(float).tiny)
    safe = np.where(resolved, lam, 1.0)
    modes = (q @ psi) * np.where(resolved, 1.0 / np.sqrt(n_b * safe),
                                 0.0)[:, None, :]
    return SPODResult(freqs=blocks.freqs, eigenvalues=np.where(resolved,
                                                               lam, 0.0),
                      modes=modes, weights=w, spec=blocks.spec,
                      dt=blocks.dt)
