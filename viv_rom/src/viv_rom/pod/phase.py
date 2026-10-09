"""Phase of the POD temporal coefficients relative to the forced motion.

Each temporal coefficient ``a_k(t)`` is fitted by least squares with the
forcing frequency and its harmonics,

    a_k(t) ~ c_0 + sum_{n=1..H} A_n * sin(n * omega * t + phi_n),

``omega = 2 pi f_forcing``. A joint fit (rather than reading FFT bins) is
exact for a non-integer number of periods and keeps the harmonics from
leaking into each other.

Reference and sign convention
-----------------------------
* The forced displacement is ``x(t) = A sin(omega t)`` (velocity
  ``A omega cos(omega t)``, as used throughout the package), so ``phi_n`` is
  the phase *lead* relative to the ``n``-th harmonic of the motion, with
  ``t`` the absolute simulation time. ``phi_1 = 0``: in phase with the
  displacement; ``+90 deg``: in phase with the velocity.
* A POD mode and its coefficient are only defined up to a common sign. To
  make phases reproducible, each mode is oriented so that the largest-
  magnitude entry of its spatial mode is positive (see
  :func:`canonical_signs`); the phase then describes ``+phi_k(y) * a_k(t)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def canonical_signs(modes: np.ndarray) -> np.ndarray:
    """Sign (+1/-1) making the largest-|.| entry of each mode positive.

    Parameters
    ----------
    modes : ndarray, shape (n_points, n_modes)
    """
    idx = np.argmax(np.abs(modes), axis=0)
    picked = modes[idx, np.arange(modes.shape[1])]
    return np.where(picked < 0, -1.0, 1.0)


@dataclass(frozen=True, eq=False)
class HarmonicPhases:
    """Result of :func:`harmonic_phase_analysis`.

    Attributes
    ----------
    harmonics : ndarray of int, shape (H,)
        Harmonic orders ``n`` (1 = forcing frequency).
    frequencies : ndarray, shape (H,)
        ``n * f_forcing`` [Hz].
    amplitude : ndarray, shape (K, H)
        ``A_n`` of each analysed mode (units of the coefficients).
    phase : ndarray, shape (K, H)
        ``phi_n`` [rad], wrapped to ``(-pi, pi]``; ``nan`` where the
        harmonic is negligible (see ``amplitude_floor``).
    sign : ndarray, shape (K,)
        Sign applied to each mode (see module docstring).
    explained : ndarray, shape (K,)
        Fraction of the variance of ``a_k`` captured by the fit; low values
        mean ``a_k`` is not periodic at the forcing frequency (phases then
        carry little meaning).
    """
    harmonics: np.ndarray
    frequencies: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    sign: np.ndarray
    explained: np.ndarray

    @property
    def n_modes(self) -> int:
        return self.phase.shape[0]

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(self.phase)

    def summary_lines(self) -> list[str]:
        head = "".join(f"{f'phase h{n} [deg]':>16}{f'amp h{n}':>11}"
                       for n in self.harmonics)
        lines = ["--- Phase of temporal coefficients relative to forced "
                 "motion (lead; 0 = displacement, +90 = velocity) ---",
                 f"{'mode':>4}{head}{'fit R^2':>9}"]
        for k in range(self.n_modes):
            cells = "".join(f"{self.phase_deg[k, i]:16.1f}"
                            f"{self.amplitude[k, i]:11.3g}"
                            for i in range(len(self.harmonics)))
            lines.append(f"{k+1:4d}{cells}{self.explained[k]:9.3f}")
        return lines


def harmonic_phase_analysis(time_coeff: np.ndarray, t: np.ndarray,
                            f_forcing: float, n_modes: int = 4,
                            n_harmonics: int = 3,
                            signs: np.ndarray | None = None,
                            amplitude_floor: float = 1e-3
                            ) -> HarmonicPhases:
    """Phase/amplitude of the leading coefficients at the forcing harmonics.

    Parameters
    ----------
    time_coeff : ndarray, shape (n_all_modes, n_t)
        Temporal coefficients (rows = modes).
    t : ndarray, shape (n_t,)
        Absolute time [s] (the origin matters: it is the motion's).
    f_forcing : float
        Forcing frequency [Hz].
    n_modes : int
        Number of leading modes to analyse.
    n_harmonics : int
        Harmonics ``1..n_harmonics`` of the forcing frequency.
    signs : ndarray, optional
        Per-mode sign to apply (default: no flipping). Use
        :func:`canonical_signs` for reproducible phases.
    amplitude_floor : float
        Harmonics whose amplitude is below this fraction of the mode's
        largest harmonic amplitude have no meaningful phase and get ``nan``.
    """
    if n_modes < 1 or n_harmonics < 1:
        raise ValueError("n_modes and n_harmonics must be >= 1.")
    K = min(n_modes, time_coeff.shape[0])
    t = np.asarray(t, dtype=float)
    dt = np.median(np.diff(t))
    if n_harmonics * f_forcing >= 0.5 / dt:
        raise ValueError(
            f"Harmonic {n_harmonics} ({n_harmonics * f_forcing:.3g} Hz) is "
            f"above the Nyquist frequency ({0.5 / dt:.3g} Hz).")
    if t[-1] - t[0] < 1.0 / f_forcing:
        raise ValueError("The time series is shorter than one forcing "
                         "period; phases are not identifiable.")

    sign = (np.ones(K) if signs is None else np.asarray(signs, float)[:K])
    a = (time_coeff[:K] * sign[:, None]).T            # (n_t, K)

    omega = 2 * np.pi * f_forcing
    n = np.arange(1, n_harmonics + 1)
    X = np.column_stack(
        [np.ones_like(t)]
        + [f(k * omega * t)
           for k in n for f in (np.cos, np.sin)])  # (n_t, 1 + 2H)
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)  # (1 + 2H, K)

    alpha = coef[1::2].T  # cos coefficients
    beta = coef[2::2].T  # sin coefficients
    amplitude = np.hypot(alpha, beta)
    phase = np.arctan2(alpha, beta)  # a = A sin(n w t + phi)
    negligible = amplitude < amplitude_floor * amplitude.max(axis=1,
                                                             keepdims=True)
    phase = np.where(negligible, np.nan, phase)

    resid = np.sum((a - X @ coef) ** 2,
                   axis=0)  # residual sum of squares (K,)
    total = np.sum((a - a.mean(axis=0)) ** 2,
                   axis=0)  # total sum of squares (K,)
    explained = np.where(total > 0,
                         1.0 - resid / np.where(total > 0, total, 1),
                         0.0)
    return HarmonicPhases(harmonics=n, frequencies=n * f_forcing,
                          amplitude=amplitude, phase=phase, sign=sign,
                          explained=explained)
