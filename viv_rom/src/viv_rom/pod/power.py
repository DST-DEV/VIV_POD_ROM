"""Physical cross-check: aerodynamic power vs. POD temporal coefficients."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from viv_rom.domain.motion import ForcedMotion
from viv_rom.pod.core import PODResult, FieldInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PowerCheck:
    """Result of :func:`power_correlation_check`.

    Attributes
    ----------
    power : ndarray, shape (n_snapshots,)
        Instantaneous total power ``P(t)`` [W].
    mean_power : float
        Time-averaged ``P(t)`` [W]. Positive means exciting the motion.
    mean_power_work : float or None
        Mean power from the solver ``*.1work`` file, if available.
    correlations : ndarray, shape (n_modes,)
        Pearson correlation of ``P(t)`` with each leading ``a_k(t)``.
    sectional_energy : ndarray, shape (n_sections,)
        Energy injected per unit span, ``integral p(y, t) dt`` [J/m].
    reconstructed_sectional_energy : dict of {int: ndarray}
        Same quantity, using Fx reconstructed with ``r`` Fx-POD modes.
    """
    power: np.ndarray
    mean_power: float
    mean_power_work: float | None
    correlations: np.ndarray
    sectional_energy: np.ndarray
    reconstructed_sectional_energy: dict[int, np.ndarray] = field(
        default_factory=dict)


def power_correlation_check(y: np.ndarray, t: np.ndarray,
                            Fx: np.ndarray, Fz: np.ndarray,
                            pod: PODResult, field_info: FieldInfo,
                            motion: ForcedMotion, amplitude: float,
                            omega: float, n_modes: int = 4,
                            r_values: Iterable[int] = (1, 2, 4),
                            work_file: Path | None = None) -> PowerCheck:
    """Correlate aerodynamic power with the leading POD temporal coefficients.

    The instantaneous power transferred from the fluid to the prescribed
    blade motion is computed from the edgewise sectional force and the
    prescribed edgewise velocity,

        ``P(t) = integral( Fx * vx +  Fz * vz) dy``.

    Parameters
    ----------
    y, t : ndarray
        Spanwise coordinates and time vector.
    Fx : ndarray, shape (n_snapshots, n_sections)
        Edgewise sectional force.
    Fz : ndarray, shape (n_snapshots, n_sections)
        Flapwise sectional force.
    pod : PODResult
        POD results.
    field_info : FieldInfo
        Needed to split a "combined" spatial mode back into its Fx and Fz
        parts.
    motion : ForcedMotion
        Prescribed motion shapes.
    amplitude : float
        Motion amplitude factor ``A``.
    omega : float
        Angular excitation frequency [rad/s].
    n_modes : int, optional
        Number of leading modes to correlate. Default is 4.
    r_values : iterable of int, optional
        Truncation levels for the sectional-energy reconstructions.
    work_file : Path or None, optional
        Solver ``*.1work`` file for validating the mean power. Skipped if
        ``None`` or non-existent.
    """
    vx = motion.velocity(component="x", y=y, t=t, amplitude=amplitude,
                         omega=omega)
    vz = motion.velocity(component="z", y=y, t=t, amplitude=amplitude,
                         omega=omega)
    p_sectional = Fx * vx + Fz * vz  # [W/m]
    P_total = np.trapezoid(p_sectional, y, axis=0)  # [W]
    mean_power = float(P_total.mean())

    mean_power_work = None
    if work_file is not None and Path(work_file).exists():
        wk = np.loadtxt(work_file, comments="#")
        dt = t[1] - t[0]
        n = min(len(wk), len(P_total))
        mean_power_work = float(((wk[-n:, 2] + wk[-n:, 4]) / dt).mean())
    else:
        logger.info("(*.1work file not found/skipped. No cross-check "
                    "against solver output)")

    n_corr = min(n_modes, pod.time_coeff.shape[0])
    correlations = np.array([np.corrcoef(P_total, pod.time_coeff[k, :])[0, 1]
                             for k in range(n_corr)])

    reconstructed = {}
    for r in r_values:
        if field_info.component == "fx":
            p_rec = pod.reconstruct(r) * vx  # [W/m]
        elif field_info.component == "fz":
            p_rec = pod.reconstruct(r) * vz  # [W/m]
        else:
            fx_rec, fz_rec = field_info.split(pod.reconstruct(r))
            p_rec = fx_rec * vx + fz_rec * vz  # [W/m]

        reconstructed[r] = np.trapezoid(p_rec, t, axis=1)

    return PowerCheck(
        power=P_total,
        mean_power=mean_power,
        mean_power_work=mean_power_work,
        correlations=correlations,
        sectional_energy=np.trapezoid(p_sectional, t, axis=1),
        reconstructed_sectional_energy=reconstructed,
    )
