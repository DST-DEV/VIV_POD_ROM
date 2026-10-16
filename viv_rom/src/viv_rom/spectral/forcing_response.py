"""Amplitude and phase of the sectional force at the forcing frequency.

At every station each force component is fitted by least squares with the
forcing frequency and its first harmonics (see
:func:`viv_rom.signal_utils.fit_harmonics`)::

    F(y, t) ~ c_0 + sum_{n=1..H} A_n(y) * sin(n * omega * t + phi_n(y))

Conventions (identical to :mod:`viv_rom.pod.phase`):

* ``phi_n`` is the lead relative to the ``n``-th harmonic of the forced
  displacement ``x(t) ~ sin(omega t)``, with ``t`` the absolute simulation
  time. ``0``: in phase with displacement; ``+90 deg``: in phase with the
  velocity.
* For the along-motion component ``F_par`` of
  :class:`~viv_rom.domain.frames.MotionAlignedFrame` the fundamental
  contributes a mean power ``0.5 * A_1 * |v| * sin(phi_1)`` to the motion
  (``|v|`` the velocity amplitude), so ``0 < phi_1 < 180 deg`` means the
  harmonic injects energy and ``-180 deg < phi_1 < 0`` means it damps; higher
  harmonics inject nothing over whole periods.
* Phase is ``nan`` where the amplitude is below ``amplitude_floor`` times the
  largest amplitude along the span, or is mere round-off next to the signal
  level (no meaningful phase without content).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from viv_rom.signal_utils import fit_harmonics


@dataclass(frozen=True, eq=False)
class ForcingResponse:
    """Result of :func:`forcing_frequency_response`.

    All per-component dicts share the keys of the analysed components.

    Attributes
    ----------
    y : ndarray, shape (n_stations,)
    harmonics : ndarray of int, shape (H,)
        Harmonic orders (1 = forcing frequency).
    frequencies : ndarray, shape (H,)
        ``n * f_forcing`` [Hz].
    amplitude : dict of {str: ndarray, shape (n_stations, H)}
        ``A_n`` [N/m].
    phase : dict of {str: ndarray, shape (n_stations, H)}
        ``phi_n`` [rad] in ``(-pi, pi]`` (``nan`` where negligible).
    explained : dict of {str: ndarray, shape (n_stations,)}
        Fraction of the force variance captured by the harmonic fit. Low
        values flag content not periodic at the forcing frequency.
    """
    y: np.ndarray
    harmonics: np.ndarray
    frequencies: np.ndarray
    amplitude: dict[str, np.ndarray]
    phase: dict[str, np.ndarray]
    explained: dict[str, np.ndarray]

    def phase_deg(self, key: str) -> np.ndarray:
        return np.degrees(self.phase[key])

    def summary_lines(self, stations: Sequence[int],
                      labels: Mapping[str, str] | None = None
                      ) -> list[str]:
        """Table of the fundamental at the given station indices."""
        labels = labels or {}
        lines = ["--- Force at the forcing frequency (fundamental; phase = "
                 "lead vs. forced displacement, +90 deg = velocity) ---",
                 f"{'component':>22} {'station':>8} {'y [m]':>7} "
                 f"{'amp [N/m]':>11} {'phase [deg]':>12} {'fit R^2':>8}"]
        for key in self.amplitude:
            for st in stations:
                ph = self.phase_deg(key)[st, 0]
                ph_txt = "      n/a" if np.isnan(ph) else f"{ph:12.1f}"
                lines.append(
                    f"{labels.get(key, key):>22} {st:8d} {self.y[st]:7.1f} "
                    f"{self.amplitude[key][st, 0]:11.4g} {ph_txt:>12} "
                    f"{self.explained[key][st]:8.3f}")
        return lines


def forcing_frequency_response(components: Mapping[str, np.ndarray],
                               y: np.ndarray, t: np.ndarray,
                               f_forcing: float, n_harmonics: int = 3,
                               amplitude_floor: float = 1e-3
                               ) -> ForcingResponse:
    """Amplitude and phase of each component at ``n * f_forcing``.

    Parameters
    ----------
    components : mapping of {str: ndarray, shape (n_stations, n_steps)}
        Force components (any frame, see :mod:`viv_rom.domain.frames`).
    y, t : ndarray
        Station positions and absolute time vector.
    f_forcing : float
        Forcing frequency [Hz].
    n_harmonics : int
        Harmonics ``1..n_harmonics`` are fitted jointly and reported.
    amplitude_floor : float
        Phase is ``nan`` where ``A_n`` is below this fraction of its maximum
        over the span.
    """
    amplitude, phase, explained = {}, {}, {}
    for key, data in components.items():
        amp, ph, r2 = fit_harmonics(data, t, f_forcing, n_harmonics)
        scale = np.abs(data).max(axis=-1, keepdims=True)   # per station
        negligible = ((amp < amplitude_floor * amp.max(axis=0,
                                                       keepdims=True))
                      | (amp <= 1e-9 * scale))              # round-off only
        amplitude[key] = amp
        phase[key] = np.where(negligible, np.nan, ph)
        explained[key] = r2
    n = np.arange(1, n_harmonics + 1)
    return ForcingResponse(y=np.asarray(y), harmonics=n,
                           frequencies=n * f_forcing, amplitude=amplitude,
                           phase=phase, explained=explained)
