"""Proper Orthogonal Decomposition (POD) via the direct SVD method.

The decomposition is applied to a *fluctuation* field ``D'`` that has already
been prepared (normalised and mean-removed, see
:mod:`viv_rom.pod.preprocessing`); this module only does the SVD.

    D' = Phi @ diag(sigma) @ Psi^T          (economy SVD)

    Phi[:, k]  -> k-th spatial POD mode  (unit norm)
    sigma[k]   -> k-th singular value
    a_k(t) = sigma[k] * Psi[:, k]  -> k-th temporal coefficient
    sigma[k]**2 / sum(sigma**2)    -> fraction of fluctuation energy in mode k

    Array convention: sectional fields have shape ``(n_stations, n_steps)`` and
    spanwise spectral maps are returned as ``(n_stations, n_freqs)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELD_OPTIONS = ("fx", "fz", "combined")


@dataclass(frozen=True)
class FieldInfo:
    """Describes how to interpret / invert the snapshot-matrix construction.

    Attributes
    ----------
    kind : {"single", "combined"}
        Whether a single force component or both combined are considered.
    component : {"Fx", "Fz"} or None
        Force component, for ``kind == "single"``.
    n : int or None
        Number of spanwise sections, for ``kind == "combined"``.
    """
    kind: str
    component: str | None = None
    n: int | None = None

    def split(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a combined-field quantity into rescaled (Fx, Fz) parts.

        ``arr`` has the stacked Fx/Fz points along axis 0, e.g. a spatial
        mode of shape (2n,) or a reconstruction of shape (2n, n_snapshots).
        """
        if self.kind != "combined":
            raise ValueError("split() requires a combined field "
                             "(kind == 'combined').")
        return arr[:self.n], arr[self.n:]

    def layout(self, fx_part: np.ndarray, fz_part: np.ndarray) -> np.ndarray:
        """Arrange per-section (Fx, Fz) quantities like the snapshot matrix.

        The inverse of :meth:`split` for 1-D per-point quantities such as
        scaling factors: returns shape ``(n_points,)`` -- the Fx part for a
        single-Fx field, the Fz part for a single-Fz field, and both stacked
        for a combined field.
        """
        if self.kind == "combined":
            return np.concatenate([fx_part, fz_part])
        return fx_part if self.component == "Fx" else fz_part


@dataclass(frozen=True, eq=False)
class PODResult:
    """Result of :func:`compute_pod`, expressed in the space it was given.

    If the input was normalised (e.g. work-weighted), modes and coefficients
    live in that normalised space; undo it with
    :class:`viv_rom.pod.preprocessing.PhysicalReconstructor`.

    Attributes
    ----------
    modes : ndarray, shape (n_points, n_modes)
        Spatial POD modes (orthonormal columns).
    sigma : ndarray, shape (n_modes,)
        Singular values, descending.
    energy_frac : ndarray, shape (n_modes,)
        ``sigma_k**2 / sum(sigma**2)``.
    cum_energy : ndarray, shape (n_modes,)
        Cumulative energy fraction.
    time_coeff : ndarray, shape (n_modes, n_snapshots)
        Temporal coefficients ``a_k(t) = sigma_k * Psi[:, k]``.
    """
    modes: np.ndarray
    sigma: np.ndarray
    energy_frac: np.ndarray
    cum_energy: np.ndarray
    time_coeff: np.ndarray

    @property
    def n_modes(self) -> int:
        return len(self.sigma)

    def reconstruct(self, r: int) -> np.ndarray:
        """Reconstruct the *fluctuation* field from the leading ``r`` modes.

        The mean is not included (it is not part of the decomposition).

        Returns
        -------
        ndarray, shape (n_points, n_snapshots)
        """
        return self.modes[:, :r] @ self.time_coeff[:r, :]

    def modes_for_energy(self, threshold: float = 0.99) -> int:
        """Smallest number of modes capturing ``threshold`` of the energy."""
        return int(np.searchsorted(self.cum_energy, threshold) + 1)


def build_field_matrix(Fx: np.ndarray, Fz: np.ndarray, mode: str
                       ) -> tuple[np.ndarray, FieldInfo]:
    """Build the (n_points, n_snapshots) matrix POD will be applied to.

    Parameters
    ----------
    Fx, Fz : ndarray, shape (n_sections, n_snapshots)
        Edgewise and flapwise force time series.
    mode : {"fx", "fz", "combined"}
        - "fx" or "fz": that component alone, n_points = n_sections
        - "combined": Fx and Fz concatenated along the spatial axis,
           n_points = 2 * n_sections.

    Returns
    -------
    field : ndarray
        The snapshot matrix for the selected mode.
    field_info : FieldInfo
        Needed to split a "combined" spatial mode back into its Fx and Fz
        parts.
    """
    if mode == "fx":
        return Fx, FieldInfo(kind="single", component="Fx")
    if mode == "fz":
        return Fz, FieldInfo(kind="single", component="Fz")
    if mode == "combined":
        field = np.vstack([Fx, Fz])
        return field, FieldInfo(kind="combined", n=Fx.shape[0])
    raise ValueError(f"Unknown field option: {mode!r} "
                     f"(use one of {FIELD_OPTIONS}).")


def compute_pod(fluctuation: np.ndarray) -> PODResult:
    """POD of a fluctuation field via the direct (economy SVD) method.

    No mean removal or scaling is done here. Prepare the input with
    :class:`viv_rom.pod.preprocessing.Preprocessor`.

    Parameters
    ----------
    fluctuation : ndarray, shape (n_points, n_snapshots)
        Rows are spatial points, columns are time snapshots.
    """
    D = np.asarray(fluctuation, dtype=float)

    # Economy SVD:  D = Phi @ diag(sigma) @ PsiT
    Phi, sigma, PsiT = np.linalg.svd(D, full_matrices=False)

    energy_frac = sigma**2 / np.sum(sigma**2)
    return PODResult(
        modes=Phi,
        sigma=sigma,
        energy_frac=energy_frac,
        cum_energy=np.cumsum(energy_frac),
        time_coeff=sigma[:, None] * PsiT,
    )
