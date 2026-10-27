"""Vortex-shedding frequency estimate along the span and its lock-in range.

Every section is treated as a bluff body at its local angle of attack. Its
shedding frequency follows from the Strouhal number::

    f_vs(y) = St * U / D(y),         D(y) = c(y) |sin(alpha(y))|

with ``c`` the chord, ``alpha`` the angle of attack (chord to flow, see
:mod:`viv_rom.domain.inflow`) and ``D`` the chord projected normal to the
flow, i.e. the cross-flow width of the section (the "projected area" of
Pellegrino & Meskell, 2013, per unit span). ``U`` is the *nominal* wind
speed: it is not reduced by the inclination angle.

``St`` is not one number but a range. Pellegrino & Meskell found
``0.11 < St < 0.13`` for the NREL S809 airfoil in 2D flow at ``Re = 1e6``
and high angles of attack near 90°. That does not transfer directly to other
airfoils and 3D flow, but the range is used nonetheless for an estimate of a
*band* of shedding frequencies per station, ``[St_low U / D, St_high U / D]``,
to be compared with the forcing (= natural) frequency ``f_n``.

Lock-in may occur when ``f_vs / f_n`` is roughly 0.8 to 1.1 (depends on the
motion amplitude). A station is flagged as *possible lock-in* if its band
overlaps ``[0.8 f_n, 1.1 f_n]`` (this says where lock-in cannot be excluded,
not that it happens with certainty).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viv_rom.domain.frames import InflowAlignedFrame

STROUHAL_RANGE = (0.11, 0.13)
LOCK_IN_BAND = (0.8, 1.1)


def validate_range(name: str, bounds: Sequence[float]) -> tuple[float, float]:
    """``(low, high)`` as floats; raises if not ``0 < low <= high``."""
    try:
        low, high = (float(b) for b in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (low, high) pair, got "
                         f"{bounds!r}.") from None
    if not 0 < low <= high:
        raise ValueError(f"{name} must satisfy 0 < low <= high, got "
                         f"({low}, {high}).")
    return low, high


@dataclass(frozen=True, eq=False)
class SheddingFrequencies:
    """Shedding-frequency band per spanwise station.

    Attributes
    ----------
    y : ndarray, shape (n_stations,)
        Spanwise coordinates [m].
    chord : ndarray, shape (n_stations,)
        Chord [m].
    angle_of_attack : ndarray, shape (n_stations,)
        Angle from chord to flow [rad], ``(-pi, pi]``.
    width : ndarray, shape (n_stations,)
        Projected (cross-flow) width ``D = c |sin(alpha)|`` [m].
    wind_speed : float
        Nominal wind speed ``U`` [m/s].
    strouhal : tuple of float
        ``(St_low, St_high)``.
    f_low, f_high : ndarray, shape (n_stations,)
        Shedding frequency [Hz] for ``St_low`` / ``St_high``. ``inf`` where
        ``D`` is zero (flow along the chord).
    f_forcing : float
        Forcing (natural) frequency ``f_n`` [Hz].
    lock_in_band : tuple of float
        ``(low, high)`` as multiples of ``f_forcing``.
    """
    y: np.ndarray
    chord: np.ndarray
    angle_of_attack: np.ndarray
    width: np.ndarray
    wind_speed: float
    strouhal: tuple[float, float]
    f_low: np.ndarray
    f_high: np.ndarray
    f_forcing: float
    lock_in_band: tuple[float, float]

    @property
    def lock_in_frequencies(self) -> tuple[float, float]:
        """Lock-in band in Hz, ``(low * f_forcing, high * f_forcing)``."""
        return (self.lock_in_band[0] * self.f_forcing,
                self.lock_in_band[1] * self.f_forcing)

    @property
    def ratio_low(self) -> np.ndarray:
        """``f_vs / f_forcing`` for the lower Strouhal bound."""
        return self.f_low / self.f_forcing

    @property
    def ratio_high(self) -> np.ndarray:
        """``f_vs / f_forcing`` for the upper Strouhal bound."""
        return self.f_high / self.f_forcing

    @property
    def possible_lock_in(self) -> np.ndarray:
        """Stations whose shedding band overlaps the lock-in band."""
        lo, hi = self.lock_in_frequencies
        return (self.f_high >= lo) & (self.f_low <= hi)

    @property
    def certain_lock_in(self) -> np.ndarray:
        """Stations whose whole shedding band lies inside the lock-in band.
        """
        lo, hi = self.lock_in_frequencies
        return (self.f_low >= lo) & (self.f_high <= hi)

    def spans(self, mask: np.ndarray | None = None
              ) -> list[tuple[float, float]]:
        """Spanwise intervals ``(y_first, y_last)`` of consecutive stations
        where ``mask`` (default: :attr:`possible_lock_in`) holds."""
        mask = self.possible_lock_in if mask is None else np.asarray(mask)
        runs, start = [], None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(mask) - 1):
                end = i if flag else i - 1
                runs.append((float(self.y[start]), float(self.y[end])))
                start = None
        return runs

    def summary_lines(self, stations: Sequence[int]) -> list[str]:
        """Text report: table at ``stations`` and the lock-in intervals."""
        lo, hi = self.lock_in_frequencies
        lines = [f"--- Vortex shedding estimate (St = {self.strouhal[0]:g}"
                 f"-{self.strouhal[1]:g}, U = {self.wind_speed:g} m/s, "
                 "D = c |sin(alpha)|) ---",
                 f"{'station':>8} {'y [m]':>7} {'c [m]':>6} "
                 f"{'alpha [deg]':>11} {'D [m]':>6} {'f_vs [Hz]':>15} "
                 f"{'f_vs/f_n':>13}  lock-in"]
        for st in stations:
            flag = ("certain" if self.certain_lock_in[st]
                    else "possible" if self.possible_lock_in[st] else "-")
            lines.append(
                f"{st:8d} {self.y[st]:7.1f} {self.chord[st]:6.2f} "
                f"{np.degrees(self.angle_of_attack[st]):11.1f} "
                f"{self.width[st]:6.2f} "
                f"{self.f_low[st]:7.3f}-{self.f_high[st]:<7.3f} "
                f"{self.ratio_low[st]:6.2f}-{self.ratio_high[st]:<6.2f}  "
                f"{flag}")
        lines.append(
            f"Lock-in band {self.lock_in_band[0]:g}-{self.lock_in_band[1]:g}"
            f" x f_n = {lo:.3f}-{hi:.3f} Hz (f_n = {self.f_forcing:g} Hz).")
        for label, mask in (("possible (band overlaps)",
                             self.possible_lock_in),
                            ("certain (band inside)", self.certain_lock_in)):
            spans = self.spans(mask)
            text = (", ".join(f"{a:.1f}-{b:.1f} m" for a, b in spans)
                    or "none")
            lines.append(f"Lock-in {label}: y = {text}")
        return lines


def shedding_frequencies(frame: InflowAlignedFrame, y: np.ndarray,
                         wind_speed: float, f_forcing: float,
                         strouhal: Sequence[float] = STROUHAL_RANGE,
                         lock_in_band: Sequence[float] = LOCK_IN_BAND
                         ) -> SheddingFrequencies:
    """Shedding-frequency band of every station.

    Parameters
    ----------
    frame : InflowAlignedFrame
        Provides the chord, the angle of attack and the projected width
        (needs the blade geometry).
    y : ndarray, shape (n_stations,)
        Station coordinates [m].
    wind_speed : float
        Nominal wind speed [m/s] (not projected by the inclination).
    f_forcing : float
        Forcing / natural frequency [Hz].
    strouhal : (low, high)
        Strouhal range, default ``(0.11, 0.16)``.
    lock_in_band : (low, high)
        Lock-in range as multiples of ``f_forcing``, default ``(0.8, 1.1)``.
    """
    strouhal = validate_range("strouhal", strouhal)
    lock_in_band = validate_range("lock_in_band", lock_in_band)
    if wind_speed <= 0 or f_forcing <= 0:
        raise ValueError("wind_speed and f_forcing must be positive.")
    y = np.asarray(y, float)
    width = np.asarray(frame.projected_width(y), float)
    with np.errstate(divide="ignore"):
        scale = np.where(width > 0, wind_speed / width, np.inf)
    return SheddingFrequencies(
        y=y, chord=np.asarray(frame.geometry.chord_at(y), float),
        angle_of_attack=np.asarray(frame.angle_of_attack(y), float),
        width=width, wind_speed=float(wind_speed), strouhal=strouhal,
        f_low=strouhal[0] * scale, f_high=strouhal[1] * scale,
        f_forcing=float(f_forcing), lock_in_band=lock_in_band)
