"""Energy injection of the phase-locked SPOD modes into the forced motion.

The prescribed velocity is ``v_i(y, t) = vhat_i(y) cos(omega t)`` with
``vhat_i = psi_i A omega`` (see :class:`~viv_rom.domain.motion.ForcedMotion`).
A force ``F_i = |H| sin(omega t + phi)`` therefore delivers the mean sectional
power

    p_i(y) = 1/2 vhat_i |H_i| sin(phi_i) = 1/2 vhat_i Im(H_i)      [W/m]

(positive = the flow drives the motion). It is linear in ``H``, so the
power of the locked response splits exactly into the contribution of the
SPOD modes, ``p = sum_k p_k``, and one can ask (as for the POD) how many
modes are needed to reproduce the spanwise distribution ``p(y)``::

    station error  e_j(r) = |sum_{k<=r} p_k(y_j) - p(y_j)| / ref_j
    one number     m(r)   = aggregate_j e_j(r)       (mean, rms, max, ...)

Only the fundamental carries net power: the motion has no content at the
higher harmonics, so those inject nothing over whole periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viv_rom.pod.core import FieldInfo
from viv_rom.pod.energy_injection import (
    ErrorReference,
    StationErrorAggregator,
    resolve_aggregators,
    resolve_reference,
)


@dataclass(frozen=True, eq=False)
class SPODInjection:
    """Mean power injected by the locked fundamental, per SPOD mode.

    Attributes
    ----------
    y : ndarray, shape (n_sections,)
    density : ndarray, shape (n_modes, n_sections)
        ``p_k(y)`` [W/m]: contribution of each mode (Fx and Fz summed for
        a combined field).
    errors : ndarray, shape (n_modes + 1, n_sections)
        Station errors; row ``r`` uses the ``r`` leading modes.
    curves, n_required, tolerance
        As in :class:`~viv_rom.pod.energy_injection.EnergyInjection`.
    f_forcing : float
        Used to express power as energy per cycle.
    """
    y: np.ndarray
    density: np.ndarray
    errors: np.ndarray
    curves: dict[str, np.ndarray]
    n_required: dict[str, int | None]
    tolerance: float
    f_forcing: float

    @property
    def n_modes(self) -> int:
        return self.density.shape[0]

    @property
    def total_density(self) -> np.ndarray:
        """``p(y)`` of the whole locked response [W/m]."""
        return self.density.sum(axis=0)

    @property
    def mode_power(self) -> np.ndarray:
        """Span-integrated power of each mode [W]."""
        return np.trapezoid(self.density, self.y, axis=1)

    @property
    def total_power(self) -> float:
        return float(self.mode_power.sum())

    @property
    def energy_per_cycle(self) -> float:
        """Injected energy per forcing cycle [J]."""
        return self.total_power / self.f_forcing

    def reconstructed(self, r: int) -> np.ndarray:
        """``p(y)`` from the ``r`` leading modes."""
        return self.density[:r].sum(axis=0)

    def summary_lines(self) -> list[str]:
        kind = ("driving the motion" if self.total_power > 0
                else "damping the motion")
        lines = ["--- Energy injection of the locked fundamental ---",
                 f"Mean power {self.total_power:.4g} W "
                 f"= {self.energy_per_cycle:.4g} J per cycle ({kind})",
                 f"Modes needed to reproduce p(y) within "
                 f"{100 * self.tolerance:g} %:"]
        for name, n in self.n_required.items():
            got = ("not reached with the modes analysed" if n is None
                   else f"{n}  (error {100 * self.curves[name][n]:.3g} %)")
            lines.append(f"  {name:>5}-error: {got}")
        return lines


def spod_injection(response: np.ndarray, velocity_amplitude: np.ndarray,
                   field_info: FieldInfo, y: np.ndarray, f_forcing: float,
                   tolerance: float = 0.01,
                   aggregators: Sequence[str | StationErrorAggregator] = (
                       "mean", "rms", "max"),
                   reference: str | ErrorReference = "peak"
                   ) -> SPODInjection:
    """Power injected by each locked SPOD mode of the fundamental.

    Parameters
    ----------
    response : complex ndarray, shape (n_points, n_modes)
        Locked mode shapes ``H_k`` of the fundamental in *physical* units
        (N/m), relative to ``sin(omega t)``.
    velocity_amplitude : ndarray, shape (n_points,)
        ``vhat`` laid out like the field (``ForcedMotion.velocity_amplitude``
        for ``Fx`` / ``Fz``).
    field_info : FieldInfo
    y : ndarray, shape (n_sections,)
    f_forcing : float
    tolerance, aggregators, reference
        As for the POD energy-injection analysis.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    aggs = resolve_aggregators(aggregators)
    ref_fn = resolve_reference(reference)

    p = 0.5 * velocity_amplitude[:, None] * np.imag(response)  # (n_pts, K)
    if field_info.kind == "combined":
        px, pz = field_info.split(p)
        p = px + pz
    density = p.T                                              # (K, n_sec)

    total = density.sum(axis=0)
    cumulative = np.vstack([np.zeros_like(total), np.cumsum(density, axis=0)])
    ref = np.broadcast_to(np.asarray(ref_fn(total), dtype=float),
                          total.shape)
    errors = np.divide(np.abs(cumulative - total), ref,
                       out=np.zeros_like(cumulative), where=ref > 0)
    curves = {name: np.array([agg(e) for e in errors])
              for name, agg in aggs.items()}
    n_required = {}
    for name, curve in curves.items():
        ok = np.flatnonzero(curve <= tolerance)
        n_required[name] = int(ok[0]) if ok.size else None
    return SPODInjection(y=np.asarray(y), density=density, errors=errors,
                         curves=curves, n_required=n_required,
                         tolerance=tolerance, f_forcing=f_forcing)
