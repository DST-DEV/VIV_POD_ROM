"""Loading the simulations that enter a global POD.

Every simulation becomes a :class:`CaseSnapshots`: its raw (physical) snapshot
matrix, cropped to whole forcing periods, together with the per-point scaling
of the configured normalisation. Nothing is centred or decomposed here, so
the same snapshots can be used to fit any number of bases (all simulations,
leave-one-out subsets, ...).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from viv_rom.domain.data import ForceFieldData, ForceFieldFile
from viv_rom.gpod.config import GlobalPODConfig
from viv_rom.loading import load_case
from viv_rom.pod.config import PODConfig
from viv_rom.pod.core import FieldInfo, build_field_matrix
from viv_rom.pod.normalization import FieldScaling, NormalizationContext

logger = logging.getLogger(__name__)

# A period counts as a whole number of samples if it is within this
# (relative) distance of an integer.
_SAMPLES_PER_PERIOD_RTOL = 1e-3


@dataclass(frozen=True, eq=False)
class CaseSnapshots:
    """One simulation prepared for pooling.

    Attributes
    ----------
    value : int
        Value of the sweep parameter.
    config : PODConfig
        Config of this simulation.
    force_file : ForceFieldFile
        The loaded force field, cropped like ``field``.
    field : ndarray, shape (n_points, n_t)
        Raw snapshot matrix in physical units (N/m).
    field_info : FieldInfo
        Layout of ``field`` (single component or combined Fx/Fz).
    scaling : FieldScaling
        Per-point normalisation of this case (identity if none).
    dt : float
        Sampling interval [s].
    n_periods : float
        Number of forcing periods in ``field``.
    n_cropped : int
        Samples dropped from the start of the retained window to obtain whole
        periods.
    """
    value: int
    config: PODConfig
    force_file: ForceFieldFile
    field: np.ndarray
    field_info: FieldInfo
    scaling: FieldScaling
    dt: float
    n_periods: float
    n_cropped: int

    @property
    def y(self) -> np.ndarray:
        return self.force_file.data.y

    @property
    def t(self) -> np.ndarray:
        return self.force_file.data.t

    @property
    def n_points(self) -> int:
        return self.field.shape[0]

    @property
    def n_t(self) -> int:
        return self.field.shape[1]

    @property
    def scaled(self) -> np.ndarray:
        """Normalised (not yet centred) snapshot matrix."""
        return self.scaling.apply(self.field)


def whole_period_length(n_t: int, dt: float, f_forcing: float,
                        n_periods: int | None = None) -> int:
    """Number of trailing samples that make up whole forcing periods.

    Parameters
    ----------
    n_t : int
        Samples available.
    dt, f_forcing : float
        Sampling interval [s] and forcing frequency [Hz].
    n_periods : int, optional
        Exactly this many periods; ``ValueError`` if not available. Default:
        as many whole periods as fit.
    """
    per_period = 1.0 / (f_forcing * dt)
    if abs(per_period - round(per_period)) <= _SAMPLES_PER_PERIOD_RTOL * \
            per_period:
        per_period = float(round(per_period))   # absorb file round-off in dt
    else:
        logger.warning("  forcing period is %.3f samples (not an integer); "
                       "whole-period cropping is approximate.", per_period)
    available = int(np.floor(n_t / per_period + 1e-9))
    wanted = available if n_periods is None else n_periods
    if wanted < 1 or wanted > available:
        raise ValueError(f"{wanted} forcing period(s) requested but only "
                         f"{n_t / per_period:.2f} are available.")
    return int(np.floor(wanted * per_period + 1e-9))


def crop_to_last(force_file: ForceFieldFile, n_keep: int) -> ForceFieldFile:
    """Keep the last ``n_keep`` samples of a force field file."""
    d = force_file.data
    if n_keep >= len(d.t):
        return force_file
    data = ForceFieldData(t=d.t[-n_keep:], y=d.y, Fx=d.Fx[:, -n_keep:],
                          Fz=d.Fz[:, -n_keep:])
    return ForceFieldFile(
        source_file=force_file.source_file, amplitude=force_file.amplitude,
        pitch=force_file.pitch, inclination=force_file.inclination,
        wind_speed=force_file.wind_speed, data=data)


def prepare_case(config: PODConfig, value: int, *, whole_periods: bool = True,
                 n_periods: int | None = None) -> CaseSnapshots:
    """Load and prepare the simulation described by ``config``.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    force_file = load_case(config)
    t = force_file.data.t
    dt = float((t[-1] - t[0]) / (len(t) - 1))   # mean step: robust to the
    n_cropped = 0                               # round-off of the file
    if whole_periods or n_periods is not None:
        n_t = len(t)
        n_keep = whole_period_length(n_t, dt, config.f_forcing, n_periods)
        n_cropped = n_t - n_keep
        force_file = crop_to_last(force_file, n_keep)

    data = force_file.data
    field, field_info = build_field_matrix(Fx=data.Fx, Fz=data.Fz,
                                           mode=config.field)
    context = NormalizationContext(
        y=data.y, wind_speed=force_file.wind_speed, motion=config.motion,
        amplitude=config.effective_motion_amplitude,
        omega=2 * np.pi * config.f_forcing, air_density=config.air_density)
    scaling = FieldScaling.from_normalizers(config.normalizers(), context,
                                            field_info)
    return CaseSnapshots(
        value=int(value), config=config, force_file=force_file, field=field,
        field_info=field_info, scaling=scaling, dt=dt,
        n_periods=(data.t[-1] - data.t[0] + dt) * config.f_forcing,
        n_cropped=n_cropped)


def check_compatible(cases: list[CaseSnapshots]) -> None:
    """Raise if the simulations cannot share one snapshot matrix.

    All cases need the same spanwise stations and the same field layout; a
    differing time step is only reported.
    """
    ref = cases[0]
    for c in cases[1:]:
        if c.field_info != ref.field_info or c.n_points != ref.n_points:
            raise ValueError(
                f"Case {c.value} has a different field layout "
                f"({c.field_info}, {c.n_points} points) than case "
                f"{ref.value} ({ref.field_info}, {ref.n_points} points).")
        if not np.allclose(c.y, ref.y):
            raise ValueError(f"Case {c.value} has different spanwise "
                             f"stations than case {ref.value}.")
        if abs(c.dt - ref.dt) > 1e-6 * ref.dt:
            logger.warning("Case %s has dt = %.6g s, case %s has %.6g s.",
                           c.value, c.dt, ref.value, ref.dt)


def load_snapshots(config: GlobalPODConfig
                   ) -> tuple[list[CaseSnapshots], list[int]]:
    """Load all simulations of a global POD.

    Returns
    -------
    cases : list of CaseSnapshots
        In ascending order of the sweep parameter.
    missing : list of int
        Values skipped because their file does not exist (empty unless
        ``config.skip_missing``).
    """
    cases: list[CaseSnapshots] = []
    missing: list[int] = []
    for value in config.values:
        case_config = config.case_config(value)
        try:
            case = prepare_case(case_config, value,
                                whole_periods=config.whole_periods,
                                n_periods=config.n_periods)
        except FileNotFoundError as err:
            if not config.skip_missing:
                raise
            logger.warning("Skipping %s = %s: %s", config.parameter, value,
                           err)
            missing.append(value)
            continue
        logger.info("  %s = %-3s %6d snapshots, %.1f periods%s",
                    config.parameter, value, case.n_t, case.n_periods,
                    f" (dropped {case.n_cropped} to get whole periods)"
                    if case.n_cropped else "")
        cases.append(case)

    if len(cases) < 2:
        raise ValueError(f"Only {len(cases)} simulation(s) found for "
                         f"{config.parameter} in {list(config.values)}; a "
                         "global POD needs at least two.")
    check_compatible(cases)
    return cases, missing
