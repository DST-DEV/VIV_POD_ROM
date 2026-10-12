"""Cycle-by-cycle energy injection and how much it fluctuates.

The sectional power transferred from the flow to the prescribed motion is
``p(y, t) = Fx*vx + Fz*vz`` (frame independent: rotating ``(Fx, Fz)`` into the
motion-aligned frame leaves it unchanged). Its integral over each forcing
period ``T = 1/f_forcing`` is the energy injected in that cycle,

    ``E_c(y) = integral_{t_c}^{t_c + T} p(y, t) dt``     [J/m]

and ``E_c = integral E_c(y) dy`` [J] for the whole blade. For a strictly
periodic (locked) response every cycle injects the same energy; a spread of
``E_c`` over the cycles signals beating, intermittency or an unfinished
transient. :class:`CycleStatistics` quantifies it:

``cv``
    standard deviation over cycles / ``|mean|`` (the headline measure).
``rel_range``
    (max - min) / ``|mean|``.
``slope``, ``std_detrended``
    linear drift per cycle, and the spread left after removing it: a
    settling transient shows up as drift, genuine fluctuation as
    ``std_detrended``.
``n_negative``
    cycles with net negative injection (the flow *damped* the motion).

Cycles start at ``t_start`` (default the first sample) and only complete
cycles are used. Cycle boundaries need not coincide with samples: the power
is interpolated linearly at the boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def injected_power(Fx: np.ndarray, Fz: np.ndarray, vx: np.ndarray,
                   vz: np.ndarray) -> np.ndarray:
    """Sectional power ``p = Fx*vx + Fz*vz`` [W/m], shape of ``Fx``."""
    return Fx * vx + Fz * vz


def per_cycle_integral(values: np.ndarray, t: np.ndarray, period: float,
                       t_start: float | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Integral of ``values`` over each complete period (trapezoidal rule).

    Parameters
    ----------
    values : ndarray, shape (n_series, n_steps)
    t : ndarray, shape (n_steps,)
        Increasing time vector (need not be uniform).
    period : float
    t_start : float, optional
        Start of the first cycle (default ``t[0]``).

    Returns
    -------
    integrals : ndarray, shape (n_cycles, n_series)
    edges : ndarray, shape (n_cycles + 1,)
        Cycle boundaries.
    """
    t = np.asarray(t, dtype=float)
    t0 = t[0] if t_start is None else t_start
    if t0 < t[0]:
        raise ValueError("t_start precedes the first sample.")
    n_cycles = int(np.floor((t[-1] - t0) / period + 1e-9))
    if n_cycles < 1:
        raise ValueError("The record does not contain a complete forcing "
                         "period after t_start.")
    edges = t0 + period * np.arange(n_cycles + 1)

    # Cumulative trapezoid integral C(t_i) and exact value at the edges:
    # C(b) = C(t_i) + (b - t_i) * (p_i + p(b)) / 2, with p linear on [t_i, t_i+1].
    dt = np.diff(t)
    cum = np.concatenate(
        [np.zeros(values.shape[:-1] + (1,)),
         np.cumsum(0.5 * (values[..., 1:] + values[..., :-1]) * dt, axis=-1)],
        axis=-1)

    # Interpolate linearly at edges (if the edge is not exactly at a time step)
    left_edge_idx = np.clip(np.searchsorted(t, edges, side="right") - 1,
                            0, len(t) - 2)
    frac = (edges - t[left_edge_idx]) \
        / dt[left_edge_idx]  # fraction to next time step
    p_edge = values[..., left_edge_idx] \
        + (values[..., left_edge_idx + 1] - values[..., left_edge_idx]) * frac

    # Add partial trapezoid from the closest left time step to the edge to the
    # cumulative trapezoid
    cum_edge = cum[..., left_edge_idx] \
        + (edges - t[left_edge_idx]) \
        * (values[..., left_edge_idx] + p_edge) / 2
    return np.diff(cum_edge, axis=-1).T, edges


@dataclass(frozen=True, eq=False)
class CycleStatistics:
    """Spread of a per-cycle quantity (arrays over the non-cycle axes)."""
    n_cycles: int
    mean: np.ndarray
    std: np.ndarray
    cv: np.ndarray
    rel_range: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    slope: np.ndarray
    std_detrended: np.ndarray
    n_negative: np.ndarray


def cycle_statistics(x: np.ndarray) -> CycleStatistics:
    """Statistics over axis 0 (cycles) of ``x``, shape ``(n_cycles, ...)``.

    Quantities that need more cycles than available (``std``: 2,
    ``slope``: 2, ``std_detrended``: 3) are ``nan``; ratios by a mean that is
    negligible against the largest mean are ``nan`` too.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    nan = np.full(x.shape[1:], np.nan)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1) if n > 1 else nan
    lo, hi = x.min(axis=0), x.max(axis=0)

    safe = np.abs(mean) > 1e-9 * max(np.abs(mean).max(), 1e-300)
    denom = np.where(safe, np.abs(mean), np.nan)

    if n > 1:
        c = (np.arange(n) - (n - 1) / 2.0).reshape((-1,) + (1,) * (x.ndim - 1))
        slope = (c * (x - mean)).sum(axis=0) / (c**2).sum()
    else:
        slope = nan
    if n > 2:
        resid = x - mean - slope * c
        std_detrended = np.sqrt((resid**2).sum(axis=0) / (n - 2))
    else:
        std_detrended = nan

    return CycleStatistics(
        n_cycles=n, mean=mean, std=std, cv=std / denom,
        rel_range=(hi - lo) / denom, minimum=lo, maximum=hi, slope=slope,
        std_detrended=std_detrended, n_negative=(x < 0).sum(axis=0))


@dataclass(frozen=True, eq=False)
class CycleEnergy:
    """Result of :func:`cycle_energy_analysis`.

    Attributes
    ----------
    y : ndarray, shape (n_stations,)
    edges : ndarray, shape (n_cycles + 1,)
        Cycle boundaries [s].
    energy : ndarray, shape (n_cycles, n_stations)
        ``E_c(y)`` [J/m].
    total : ndarray, shape (n_cycles,)
        Whole-blade energy per cycle [J] (``E_c(y)`` integrated over span).
    """
    y: np.ndarray
    edges: np.ndarray
    energy: np.ndarray
    total: np.ndarray

    @property
    def n_cycles(self) -> int:
        return self.energy.shape[0]

    @property
    def station_stats(self) -> CycleStatistics:
        return cycle_statistics(self.energy)

    @property
    def total_stats(self) -> CycleStatistics:
        return cycle_statistics(self.total)

    def summary_lines(self, stations: Sequence[int]) -> list[str]:
        s = self.total_stats
        lines = [f"--- Cycle-by-cycle energy injection "
                 f"({self.n_cycles} complete forcing cycles) ---",
                 f"Whole blade: mean {s.mean:.4g} J/cycle, std {s.std:.4g} J, "
                 f"CV {100 * s.cv:.2f} %, range {100 * s.rel_range:.2f} % "
                 f"of mean, drift {s.slope:+.3g} J/cycle "
                 f"(detrended std {s.std_detrended:.3g} J), "
                 f"{int(s.n_negative)} damping cycle(s)"]
        if self.n_cycles < 3:
            lines.append("  (fewer than 3 cycles: spread estimates are "
                         "unreliable)")
        st = self.station_stats
        lines.append(f"{'station':>8} {'y [m]':>7} {'mean [J/m]':>11} "
                     f"{'std [J/m]':>10} {'CV [%]':>8} {'neg':>4}")
        for k in stations:
            cv = ("     n/a" if np.isnan(st.cv[k])
                  else f"{100 * st.cv[k]:8.2f}")
            lines.append(f"{k:8d} {self.y[k]:7.1f} {st.mean[k]:11.4g} "
                         f"{st.std[k]:10.4g} {cv} {int(st.n_negative[k]):4d}")
        return lines


def cycle_energy_analysis(Fx: np.ndarray, Fz: np.ndarray, vx: np.ndarray,
                          vz: np.ndarray, y: np.ndarray, t: np.ndarray,
                          f_forcing: float, t_start: float | None = None
                          ) -> CycleEnergy:
    """Energy injected per forcing cycle, per station and for the blade.

    Parameters
    ----------
    Fx, Fz : ndarray, shape (n_stations, n_steps)
        Blade-frame sectional forces [N/m].
    vx, vz : ndarray, shape (n_stations, n_steps)
        Prescribed velocities [m/s] (e.g. ``ForcedMotion.velocity``).
    y, t : ndarray
        Station positions and time vector.
    f_forcing : float
        Forcing frequency [Hz]; one cycle is ``1 / f_forcing``.
    t_start : float, optional
        Start of the first cycle (default: first sample).
    """
    power = injected_power(Fx, Fz, vx, vz)
    energy, edges = per_cycle_integral(power, t, 1.0 / f_forcing, t_start)
    return CycleEnergy(y=np.asarray(y), edges=edges, energy=energy,
                       total=np.trapezoid(energy, y, axis=1))
