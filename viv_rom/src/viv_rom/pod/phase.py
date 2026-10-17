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
  make phases reproducible *and* physically readable, each mode is oriented
  by :func:`canonical_signs` so that ``+phi_k(y) * a_k(t)`` has positive work
  alignment with the motion: with the weights of
  :func:`viv_rom.pod.energy_injection.work_alignment_weights` a mode whose
  fundamental has ``phi_1 = +90 deg`` then injects energy and
  ``phi_1 = -90 deg`` damps (``0 < phi_1 < 180 deg``: injection). Without
  weights the largest-magnitude entry of the spatial mode is made positive,
  which carries no such meaning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viv_rom.signal_utils import fit_harmonics


def canonical_signs(modes: np.ndarray,
                    work_weights: np.ndarray | None = None,
                    rtol: float = 1e-6) -> np.ndarray:
    """Sign (+1/-1) that orients each mode for a reproducible phase.

    Parameters
    ----------
    modes : ndarray, shape (n_points, n_modes)
        Spatial POD modes (in the space they were computed in).
    work_weights : ndarray, shape (n_points,), optional
        Weights ``w`` such that ``w @ mode`` is proportional to the
        span-integrated work factor of the mode (see
        :func:`viv_rom.pod.energy_injection.work_alignment_weights`). With
        them, each mode is flipped so that ``w @ mode > 0``: the sign is
        decided by how the whole mode aligns with the *motion*, not by one
        entry that may sit where the section hardly moves (or only in the
        other of the edgewise/flapwise directions). A mode with (numerically)
        no net work alignment (``|w @ mode| <= rtol * sum|w * mode|``) falls
        back to the sign of its largest *work-weighted* entry ``w * mode``.
        Without weights, the largest-magnitude entry of the mode is made
        positive.
    rtol : float
        Relative tolerance below which the net alignment counts as zero.
    """
    modes = np.asarray(modes, dtype=float)
    cols = np.arange(modes.shape[1])
    if work_weights is None:
        # Choose sign so that largest value of the mode shape is positive
        picked = modes[np.argmax(np.abs(modes), axis=0), cols]
        return np.where(picked < 0, -1.0, 1.0)

    w = np.asarray(work_weights, dtype=float)
    if w.shape != (modes.shape[0],):
        raise ValueError(f"work_weights must have shape "
                         f"({modes.shape[0]},), got {w.shape}.")
    weighted = modes * w[:, None]  # per-entry work factor
    net = weighted.sum(axis=0)
    fallback = weighted[np.argmax(np.abs(weighted), axis=0),
                        cols]  # Fallback if net work of a mode is near zero
    decisive = np.where(np.abs(net) > rtol * np.abs(weighted).sum(axis=0),
                        net, fallback)
    return np.where(decisive < 0, -1.0, 1.0)


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
                            amplitude_floor: float = 1e-3,
                            unwrap_phase: float = False
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
    sign = (np.ones(K) if signs is None else np.asarray(signs, float)[:K])

    amplitude, phase, explained = fit_harmonics(
        signals=time_coeff[:K] * sign[:, None], t=t, f=f_forcing,
        n_harmonics=n_harmonics, unwrap_phase=unwrap_phase)
    negligible = amplitude < amplitude_floor * amplitude.max(axis=1,
                                                             keepdims=True)
    phase = np.where(negligible, np.nan, phase)
    n = np.arange(1, n_harmonics + 1)
    return HarmonicPhases(harmonics=n, frequencies=n * f_forcing,
                          amplitude=amplitude, phase=phase, sign=sign,
                          explained=explained)
