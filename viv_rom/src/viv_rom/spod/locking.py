"""Phase-locked part of the SPOD modes at the forcing harmonics.

The eigenvectors of an SPOD have an arbitrary overall phase, so on their own
they say nothing about the timing relative to the prescribed motion. The
motion is deterministic: its Fourier blocks ``R_l`` (of ``sin(n omega t)``,
transformed exactly like the data) are the phase reference. For harmonic
``n`` (bin ``k``) each mode's coefficient series ``a_kl`` is regressed on it::

    s_k = sum_l a_kl conj(R_l) / sum_l |R_l|^2         (complex)
    H_k = Phi_k s_k                                    (locked mode shape)

``H_k(y)`` is the complex amplitude of the force that oscillates *in a fixed
phase relation* to the forced displacement: the force is
``|H_k| sin(n omega t + arg H_k)``, with the same sign convention as the POD
and spectral modules (``arg H = +90 deg`` is in phase with the velocity of the
motion). ``sum_k H_k`` is the transfer function force / displacement of the
whole field, the SPOD merely splits it into coherent modes and the global
phase ambiguity is gone.

Two ratios quantify how locked a response is:

* ``locked_fraction``: share of the power at the bin that is phase-locked
  to the motion (1: fully deterministic, ~0: independent of the forcing,
  e.g. a free vortex-shedding frequency);
* ``mode_locking``: squared coherence between one mode's coefficient
  series and the reference (1: the mode follows the motion exactly).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viv_rom.spod.blocks import FourierBlocks
from viv_rom.spod.core import SPODResult


@dataclass(frozen=True, eq=False)
class PhaseLockedResponse:
    """Phase-locked modal response at the forcing harmonics.

    Attributes
    ----------
    harmonics : ndarray of int, shape (H,)
    bins : ndarray of int, shape (H,)
        Frequency bin used for each harmonic.
    freqs : ndarray, shape (H,)
        Exact harmonic frequencies ``n f_forcing`` [Hz].
    offset_bins : ndarray, shape (H,)
        Distance of the harmonic from its bin centre in bin widths.
    coefficients : complex ndarray, shape (H, n_modes)
        Regression coefficients ``s_k``.
    response : complex ndarray, shape (H, n_points, n_modes)
        ``H_k = Phi_k s_k`` in the space the SPOD was computed in.
    locked_energy : ndarray, shape (H, n_modes)
        Part of each eigenvalue that is phase-locked (same units as
        ``lambda``).
    total_energy : ndarray, shape (H,)
        ``sum_k lambda_k`` at the bin.
    mode_locking : ndarray, shape (H, n_modes)
        Squared coherence of each mode with the reference, in [0, 1].
    """
    harmonics: np.ndarray
    bins: np.ndarray
    freqs: np.ndarray
    offset_bins: np.ndarray
    coefficients: np.ndarray
    response: np.ndarray
    locked_energy: np.ndarray
    total_energy: np.ndarray
    mode_locking: np.ndarray
    n_blocks: int = 0

    @property
    def chance_level(self) -> float:
        """Locked share expected from unrelated noise, about ``1 / n_b``.

        (Higher with overlapping blocks, which are not independent.)
        """
        return 1.0 / self.n_blocks if self.n_blocks else float("nan")

    @property
    def n_modes(self) -> int:
        return self.coefficients.shape[1]

    @property
    def locked_fraction(self) -> np.ndarray:
        """Phase-locked share of the power at each harmonic, shape (H,)."""
        return np.divide(self.locked_energy.sum(axis=1), self.total_energy,
                         out=np.zeros(len(self.total_energy)),
                         where=self.total_energy > 0)

    def total_response(self) -> np.ndarray:
        """``sum_k H_k`` (all modes), shape (H, n_points)."""
        return self.response.sum(axis=2)

    def summary_lines(self, f_forcing: float) -> list[str]:
        lines = ["--- Phase-locking to the forced motion ---",
                 "(share of the power at each harmonic that keeps a fixed "
                 "phase relation to the motion)",
                 f"{'n':>3} {'f [Hz]':>8} {'bin offset':>11} "
                 f"{'locked power':>13} {'mode 1 coherence':>17}"]
        # (chance level is appended below the table)
        for i, n in enumerate(self.harmonics):
            lines.append(
                f"{n:3d} {self.freqs[i]:8.4f} {self.offset_bins[i]:+11.3f} "
                f"{100 * self.locked_fraction[i]:12.1f}% "
                f"{self.mode_locking[i, 0]:17.3f}")
        chance = 100 * self.chance_level
        lines.append(f"(unrelated noise would show ~{chance:.0f}% locked "
                     f"power with {self.n_blocks} blocks; only values "
                     "clearly above that indicate locking)")
        return lines


def phase_locked_response(spod: SPODResult, blocks: FourierBlocks,
                          reference: FourierBlocks, f_forcing: float,
                          harmonics: np.ndarray | None = None
                          ) -> PhaseLockedResponse:
    """Regress the SPOD coefficients on the forced displacement.

    Parameters
    ----------
    spod : SPODResult
    blocks : FourierBlocks
        The blocks ``spod`` was computed from.
    reference : FourierBlocks
        Blocks (same spec) of ``sin(n omega t)``, one row per harmonic,
        rows in the order of ``harmonics``; time must be the absolute time
        of the data.
    f_forcing : float
    harmonics : ndarray of int, optional
        Harmonic numbers of the reference rows (default ``1..n_rows``).
    """
    n_rows = reference.q.shape[1]
    harmonics = (np.arange(1, n_rows + 1) if harmonics is None
                 else np.asarray(harmonics, dtype=int))
    if len(harmonics) != n_rows:
        raise ValueError("One reference row per harmonic is required.")
    if reference.q.shape[2] != blocks.n_blocks:
        raise ValueError("Reference and data blocks must use the same "
                         "segmentation.")

    n_b = blocks.n_blocks
    freqs = harmonics * f_forcing
    bins = np.array([spod.bin_of(f) for f in freqs])
    offsets = np.array([spod.offset_bins(f) for f in freqs])

    coeff, energy_locked, coherence = [], [], []
    for i, b in enumerate(bins):
        a = spod.coefficients_at(blocks, b)                  # (n_modes, n_b)
        r = reference.q[b, i, :]                             # (n_b,)
        r_power = np.sum(np.abs(r) ** 2)
        cross = a @ r.conj()                                 # (n_modes,)
        s = cross / r_power if r_power > 0 else np.zeros_like(cross)
        a_power = np.sum(np.abs(a) ** 2, axis=1)
        denom = a_power * r_power
        coh = np.divide(np.abs(cross) ** 2, denom,
                        out=np.zeros(len(cross)), where=denom > 0)
        coeff.append(s)
        energy_locked.append(np.abs(s) ** 2 * r_power / n_b)
        coherence.append(coh)

    coeff = np.array(coeff)
    return PhaseLockedResponse(
        harmonics=harmonics, bins=bins, freqs=freqs, offset_bins=offsets,
        coefficients=coeff,
        response=np.einsum("hpk,hk->hpk", spod.modes[bins], coeff),
        locked_energy=np.array(energy_locked),
        total_energy=spod.energy[bins], mode_locking=np.array(coherence),
        n_blocks=n_b)
