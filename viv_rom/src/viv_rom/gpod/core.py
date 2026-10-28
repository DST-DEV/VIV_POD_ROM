"""Numerics of the global POD: pooled SVD of time-concatenated snapshots.

Given simulations ``c = 1..N`` with fluctuation fields ``D'_c``
(``n_points x n_t,c``, normalised and centred, see
:mod:`viv_rom.pod.preprocessing`) and weights ``w_c``, the global basis is the
POD of the weighted concatenation

    D_glob = [ sqrt(w_1) D'_1 | sqrt(w_2) D'_2 | ... | sqrt(w_N) D'_N ]
           = Phi diag(sigma) Psi^T,

i.e. the eigenvectors of the pooled covariance ``sum_c w_c D'_c D'_c^T``. The
order of the columns does not matter.

Time coefficients of a simulation are obtained by projection,
``a_c(t) = Phi^T D'_c(t)`` (with its *unweighted* data); weights only select
the basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import subspace_angles

from viv_rom.gpod.config import CENTERING_OPTIONS, WEIGHTING_OPTIONS
from viv_rom.gpod.snapshots import CaseSnapshots
from viv_rom.pod.energy_injection import work_alignment_weights
from viv_rom.pod.phase import canonical_signs


# =============================================================================
# Small numerical building blocks
# =============================================================================
def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles [rad] between the column spaces of ``a`` and ``b``.

    Returns ``min(a.shape[1], b.shape[1])`` angles in ascending order; 0
    means a shared direction, ``pi/2`` an orthogonal one. Uses SciPy's
    sine-based algorithm, which stays accurate for very small angles (where
    ``arccos`` of the singular values of ``a.T @ b`` loses half the digits).
    """
    return np.sort(subspace_angles(a, b))


def case_weights(fluctuations: Sequence[np.ndarray], scheme: str
                 ) -> np.ndarray:
    """Weight ``w_c`` of each simulation in the pooled covariance.

    See :class:`~viv_rom.gpod.config.GlobalPODConfig` (``weighting``).
    """
    if scheme not in WEIGHTING_OPTIONS:
        raise ValueError(f"Unknown weighting {scheme!r} "
                         f"(use one of {WEIGHTING_OPTIONS}).")
    if scheme == "snapshots":
        return np.ones(len(fluctuations))
    if scheme == "time_average":
        return np.array([1.0 / d.shape[1] for d in fluctuations])
    energy = np.array([np.sum(d * d) for d in fluctuations])
    if np.any(energy <= 0):
        raise ValueError("A simulation has zero fluctuation energy; "
                         "'unit_energy' weighting is undefined.")
    return 1.0 / energy


def pooled_svd(fluctuations: Sequence[np.ndarray], weights: Sequence[float]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Left singular vectors and values of the weighted concatenation.

    Parameters
    ----------
    fluctuations : sequence of ndarray, each (n_points, n_t_c)
    weights : sequence of float
        ``w_c``; block ``c`` enters as ``sqrt(w_c) * D'_c``.

    Returns
    -------
    modes : ndarray, shape (n_points, n_modes)
        Orthonormal spatial modes, ``n_modes = min(n_points, sum_c n_t_c)``.
    sigma : ndarray, shape (n_modes,)
        Singular values of the weighted concatenation, descending.
    """
    m = fluctuations[0].shape[0]
    C = np.zeros((m, m))
    for d, w in zip(fluctuations, weights):
        d = np.asarray(d, dtype=float)
        C += w * (d @ d.T)                 # one block at a time
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    return U, np.sqrt(np.clip(S, 0.0, None))


def work_signs(cases: Sequence[CaseSnapshots], modes: np.ndarray
               ) -> np.ndarray:
    """Sign of each global mode from its alignment with the forced motion.

    Same rule as for the single-case POD (``canonical_signs``): the sign is
    chosen such that the mode's work factor is positive, so a fundamental
    phase of +90 deg means energy injection. The work weights of the
    simulations (which differ by their normalisation) are each scaled to a
    peak of one and averaged.
    """
    w = np.zeros(modes.shape[0])
    for c in cases:
        wc = work_alignment_weights(c.y, c.config.motion, c.field_info,
                                    c.scaling.factors)
        w += wc / np.max(np.abs(wc))
    return canonical_signs(modes, w)


# =============================================================================
# The global basis
# =============================================================================
@dataclass(frozen=True, eq=False)
class GlobalBasis:
    """Spatial basis fitted to a set of simulations, plus how to use it.

    Modes live in the *normalised* space of the simulations (see
    :mod:`viv_rom.pod.normalization`), exactly like the modes of a
    single-case POD.

    Attributes
    ----------
    values : ndarray of int, shape (n_cases,)
        Sweep-parameter values of the simulations the basis was fitted to.
    modes : ndarray, shape (n_points, n_modes)
        Orthonormal spatial modes, ordered by pooled energy and oriented by
        :func:`work_signs`.
    sigma : ndarray, shape (n_modes,)
        Singular values of the (weighted) global snapshot matrix.
    energy_frac : ndarray, shape (n_modes,)
        ``sigma_k**2 / sum(sigma**2)`` (sum over *all* modes, also if the
        basis was truncated).
    cum_energy : ndarray, shape (n_modes,)
        Cumulative energy of the modes (also if the basis was truncated).
    centering : str
        ``"case"``, ``"global"`` or ``"none"``.
    weighting : str
        Case-weighting scheme.
    weights : ndarray, shape (n_cases,)
        Weights ``w_c`` of the fitted simulations.
    n_snapshots : ndarray of int, shape (n_cases,)
        Snapshots per simulation.
    global_mean : ndarray, shape (n_points,) or None
        The common mean (for ``centering == "global"``).
    """
    values: np.ndarray
    modes: np.ndarray
    sigma: np.ndarray
    energy_frac: np.ndarray
    cum_energy: np.ndarray
    centering: str
    weighting: str
    weights: np.ndarray
    n_snapshots: np.ndarray
    global_mean: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    @property
    def n_points(self) -> int:
        return self.modes.shape[0]

    def modes_for_energy(self, threshold: float = 0.99) -> int | None:
        """Smallest number of modes with ``cum_energy >= threshold``
        (``None`` if the kept modes never reach it)."""
        idx = np.searchsorted(self.cum_energy, threshold - 1e-12)
        return int(idx + 1) if idx < self.n_modes else None

    def mean_for(self, case: CaseSnapshots) -> np.ndarray:
        """Mean field subtracted from ``case`` (normalised space)."""
        if self.centering == "case":
            return case.scaled.mean(axis=1)
        if self.centering == "global":
            return self.global_mean
        return np.zeros(case.n_points)

    def fluctuation(self, case: CaseSnapshots
                    ) -> tuple[np.ndarray, np.ndarray]:
        """``(D'_c, mean)`` of a simulation, whether or not it was fitted."""
        mean = self.mean_for(case)
        return case.scaled - mean[:, None], mean

    def coefficients(self, fluctuation: np.ndarray) -> np.ndarray:
        """Temporal coefficients ``Phi^T D'``, shape (n_modes, n_t)."""
        return self.modes.T @ fluctuation


def fit_global_basis(cases: Sequence[CaseSnapshots],
                     centering: str = "case",
                     weighting: str = "time_average",
                     n_modes_keep: int | None = None) -> GlobalBasis:
    """Fit the global basis to the time-concatenated simulations.

    Parameters
    ----------
    cases : sequence of CaseSnapshots
        The training simulations (at least two, compatible layouts).
    centering, weighting : str
        See :class:`~viv_rom.gpod.config.GlobalPODConfig`.
    n_modes_keep : int, optional
        Keep only the leading modes.
    """
    if centering not in CENTERING_OPTIONS:
        raise ValueError(f"Unknown centering {centering!r} "
                         f"(use one of {CENTERING_OPTIONS}).")
    cases = list(cases)
    if len(cases) < 2:
        raise ValueError("A global basis needs at least two simulations.")

    scaled = [c.scaled for c in cases]
    global_mean = None
    if centering == "case":
        means = [s.mean(axis=1) for s in scaled]
    elif centering == "global":
        # equal weight per case, independent of the record lengths
        global_mean = np.mean([s.mean(axis=1) for s in scaled], axis=0)
        means = [global_mean] * len(cases)
    else:
        means = [np.zeros(s.shape[0]) for s in scaled]
    fluctuations = [s - m[:, None] for s, m in zip(scaled, means)]
    del scaled

    weights = case_weights(fluctuations, weighting)
    modes, sigma = pooled_svd(fluctuations, weights)
    del fluctuations  # Clear up memory

    energy_frac = sigma**2 / np.sum(sigma**2)
    cum_energy = np.cumsum(energy_frac)
    modes = modes * work_signs(cases, modes)

    if n_modes_keep is not None:
        k = min(n_modes_keep, modes.shape[1])
        modes, sigma = modes[:, :k], sigma[:k]
        energy_frac, cum_energy = energy_frac[:k], cum_energy[:k]

    return GlobalBasis(
        values=np.array([c.value for c in cases], dtype=int), modes=modes,
        sigma=sigma, energy_frac=energy_frac, cum_energy=cum_energy,
        centering=centering, weighting=weighting, weights=weights,
        n_snapshots=np.array([c.n_t for c in cases], dtype=int),
        global_mean=global_mean)
