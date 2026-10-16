"""Preparing a snapshot matrix for POD: optional normalisation, then mean
removal.

    physical field D  --scaling-->  S D  --subtract mean-->  D' = S D - m

``D'`` is what :func:`viv_rom.pod.core.compute_pod` decomposes. Normalisation
(see :mod:`viv_rom.pod.normalization`) is optional; mean removal is always
done here and never inside the POD. Because the scaling is a fixed per-point
factor, scaling before or after taking the mean gives the same ``D'``.

The inverse chain, ``D = (D' + m) / S``, is provided by
:class:`Preprocessing` so reconstructions can be compared with the raw data in
physical units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viv_rom.pod.core import FieldInfo, PODResult
from viv_rom.pod.normalization import (FieldScaling, NormalizationContext,
                                       Normalizer)


def remove_mean(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a snapshot matrix into temporal mean and fluctuation.

    Parameters
    ----------
    field : ndarray, shape (n_points, n_snapshots)

    Returns
    -------
    fluctuation : ndarray, shape (n_points, n_snapshots)
    mean : ndarray, shape (n_points,)
    """
    field = np.asarray(field, dtype=float)
    mean = field.mean(axis=1)
    return field - mean[:, None], mean


@dataclass(frozen=True, eq=False)
class Preprocessing:
    """What was done to a field, so it can be repeated and undone.

    Attributes
    ----------
    scaling : FieldScaling
        Normalisation applied before mean removal (identity if none).
    mean : ndarray, shape (n_points,)
        Temporal mean of the *normalised* field.
    """
    scaling: FieldScaling
    mean: np.ndarray

    @property
    def normalization_names(self) -> tuple[str, ...]:
        return self.scaling.names

    def forward(self, field: np.ndarray) -> np.ndarray:
        """Physical field -> fluctuation, using the stored scaling and mean."""
        return self.scaling.apply(field) - self.mean[:, None]

    def inverse(self, fluctuation: np.ndarray) -> np.ndarray:
        """Fluctuation (e.g. a truncated reconstruction) -> physical field."""
        return self.scaling.invert(fluctuation + self.mean[:, None])


class Preprocessor:
    """Fits a :class:`Preprocessing` to a field and applies it.

    Parameters
    ----------
    normalizers : sequence of Normalizer, optional
        Strategies applied (multiplicatively) before mean removal. Empty
        means no normalisation, only mean removal.
    """

    def __init__(self, normalizers: Sequence[Normalizer] = ()):
        self.normalizers = tuple(normalizers)

    def fit_transform(self, field: np.ndarray, field_info: FieldInfo,
                      ctx: NormalizationContext
                      ) -> tuple[np.ndarray, Preprocessing]:
        """Normalise and de-mean ``field``.

        Returns
        -------
        fluctuation : ndarray, shape (n_points, n_snapshots)
            Input for :func:`~viv_rom.pod.core.compute_pod`.
        preprocessing : Preprocessing
            Record of the scaling and mean for inverting later.
        """
        scaling = FieldScaling.from_normalizers(self.normalizers, ctx,
                                                field_info)
        fluctuation, mean = remove_mean(scaling.apply(np.asarray(field,
                                                                 dtype=float)))
        return fluctuation, Preprocessing(scaling=scaling, mean=mean)


@dataclass(frozen=True, eq=False)
class PhysicalReconstructor:
    """Truncated POD reconstruction in physical units.

    Callable as ``reconstructor(r)`` -> ndarray ``(n_points, n_snapshots)``,
    i.e. the leading ``r`` modes, plus the mean, with the normalisation
    undone. Plotting and power-check code depend only on this callable.
    """
    pod: PODResult
    preprocessing: Preprocessing

    def __call__(self, r: int) -> np.ndarray:
        return self.preprocessing.inverse(self.pod.reconstruct(r))

    @property
    def n_modes(self) -> int:
        return self.pod.n_modes

    def mean(self) -> np.ndarray:
        """Temporal-mean field in physical units, shape ``(n_points, 1)``."""
        return self.preprocessing.scaling.invert(
            self.preprocessing.mean[:, None])

    def mode(self, k: int) -> np.ndarray:
        """Mode ``k`` (0-based) alone, in physical units, without the mean.

        Shape ``(n_points, n_snapshots)``. Because the normalisation is
        linear, ``mean() + sum_k mode(k)`` over all modes is the raw field.
        """
        fluct = np.outer(self.pod.modes[:, k], self.pod.time_coeff[k, :])
        return self.preprocessing.scaling.invert(fluct)
