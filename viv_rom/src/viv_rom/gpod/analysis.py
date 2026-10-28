"""Global POD of a set of simulations and its per-simulation diagnostics.

Typical use (see :mod:`viv_rom.gpod.pipeline` for the file-loading wrapper)::

    cases, _ = load_snapshots(config)
    result = analyse_snapshots(cases, config)
    result.cases[0].reconstruct(4)            # rank-4 field in N/m
    result.relative_error                      # (n_cases, n_modes)
    loo = leave_one_out_snapshots(cases, config)   # unseen-case projection

Every simulation is seen through the global basis as a
:class:`CaseProjection`, which carries a :class:`~viv_rom.pod.core.PODResult`
*view* (global modes, but that case's temporal coefficients and energies), so
all single-case tools (:class:`PhysicalReconstructor`, the energy-injection
and phase analyses) work on it unchanged.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viv_rom.domain.data import ForceFieldFile
from viv_rom.gpod.config import GlobalPODConfig
from viv_rom.gpod.core import GlobalBasis, fit_global_basis, principal_angles
from viv_rom.gpod.snapshots import CaseSnapshots
from viv_rom.pod.config import PODConfig
from viv_rom.pod.core import FieldInfo, PODResult, compute_pod
from viv_rom.pod.energy_injection import (EnergyInjection,
                                          energy_injection_analysis)
from viv_rom.pod.phase import HarmonicPhases, harmonic_phase_analysis
from viv_rom.pod.preprocessing import PhysicalReconstructor, Preprocessing
from viv_rom.signal_utils import dominant_frequency

logger = logging.getLogger(__name__)


# =============================================================================
# Results
# =============================================================================
@dataclass(frozen=True, eq=False)
class LocalReference:
    """The single-case POD of a simulation (reference for the global basis).

    Computed exactly as ``viv pod`` does: the case's own mean removed, its own
    SVD.

    Attributes
    ----------
    pod : PODResult
    n99 : int
        Modes capturing 99 % of the fluctuation energy.
    energy : EnergyInjection or None
    """
    pod: PODResult
    n99: int
    energy: EnergyInjection | None = None


@dataclass(frozen=True, eq=False)
class CaseProjection:
    """One simulation seen through a (global) basis.

    Attributes
    ----------
    value : int
        Sweep-parameter value.
    config : PODConfig
    force_file : ForceFieldFile
    field : ndarray or None
        Raw snapshot matrix (physical units), ``None`` if exported without
        raw data.
    field_info : FieldInfo
    preprocessing : Preprocessing
        This case's scaling and the mean that was subtracted.
    pod : PODResult
        View of the decomposition: ``modes`` are the global modes,
        ``time_coeff`` this case's coefficients, ``sigma[k]`` the norm of
        ``a_k`` and ``energy_frac[k]`` the share of *this case's* fluctuation
        energy in global mode ``k`` (modes are ordered by pooled energy, so
        these are not sorted); ``cum_energy`` is the fraction captured by
        the leading ``r`` modes.
    dominant_freqs : ndarray
        Dominant frequency [Hz] of the leading coefficients.
    dt, n_periods : float
    n99 : int or None
        Leading global modes needed for 99 % of this case's fluctuation
        energy (``None``: not reached with the kept modes).
    energy : EnergyInjection or None
        Energy-injection analysis with the global modes.
    phases : HarmonicPhases or None
    local : LocalReference or None
    """
    value: int
    config: PODConfig
    force_file: ForceFieldFile
    field: np.ndarray | None
    field_info: FieldInfo
    preprocessing: Preprocessing
    pod: PODResult
    dominant_freqs: np.ndarray
    dt: float
    n_periods: float
    n99: int | None
    energy: EnergyInjection | None = None
    phases: HarmonicPhases | None = None
    local: LocalReference | None = None

    @property
    def reconstruct(self) -> PhysicalReconstructor:
        """``reconstruct(r)``: rank-``r`` field in physical units (N/m)."""
        return PhysicalReconstructor(self.pod, self.preprocessing)

    @property
    def relative_error(self) -> np.ndarray:
        """Relative fluctuation error after ``r = 1..n_modes`` modes.

        ``||D' - D'_r||_F / ||D'||_F`` in the normalised space (exact, from
        the orthonormality of the basis).
        """
        return np.sqrt(np.clip(1.0 - self.pod.cum_energy, 0.0, None))

    @property
    def t(self) -> np.ndarray:
        return self.force_file.data.t

    @property
    def y(self) -> np.ndarray:
        return self.force_file.data.y

    def energy_modes_required(self, metric: str | None = None
                              ) -> int | None:
        """Modes needed for the energy-injection tolerance (``None`` if
        never reached or not analysed). ``metric`` defaults to the first of
        the configured aggregators."""
        return _n_required(self.energy, metric)


@dataclass(frozen=True, eq=False)
class SubspaceAngles:
    """Principal angles [deg] between leading-``k`` POD subspaces.

    Attributes
    ----------
    values : ndarray of int, shape (n_cases,)
    k : int
        Subspace dimension.
    neighbour : ndarray, shape (n_cases - 1, k)
        Angles between the local subspaces of case ``i`` and ``i + 1``
        (ascending). Small angles: the dominant structures hardly change
        between neighbours, so one basis can represent both.
    to_global : ndarray, shape (n_cases, k)
        Angles between each local subspace and the leading-``k`` global
        subspace.
    """
    values: np.ndarray
    k: int
    neighbour: np.ndarray
    to_global: np.ndarray


@dataclass(frozen=True, eq=False)
class GlobalPODResult:
    """Everything computed by :func:`analyse_snapshots`.

    Attributes
    ----------
    parameter : str
        Name of the varying parameter.
    values : ndarray of int
        Its values (ascending), one per simulation.
    y : ndarray
        Spanwise stations.
    field_info : FieldInfo
    basis : GlobalBasis
    normalization_names : tuple of str
    cases : tuple of CaseProjection
        One per value, in the order of ``values``.
    n99 : int or None
        Global modes for 99 % of the *pooled* energy.
    angles : SubspaceAngles or None
        Needs local references (``compute_local``).
    missing : tuple of int
        Requested values without a data file.
    """
    parameter: str
    values: np.ndarray
    y: np.ndarray
    field_info: FieldInfo
    basis: GlobalBasis
    normalization_names: tuple[str, ...]
    cases: tuple[CaseProjection, ...]
    n99: int | None
    angles: SubspaceAngles | None = None
    missing: tuple[int, ...] = ()

    # The basis
    @property
    def modes(self) -> np.ndarray:
        return self.basis.modes

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes

    @property
    def pod(self) -> PODResult:
        """The global basis as a :class:`PODResult` without time coefficients
        (so the single-case plotting of modes and spectra can be reused)."""
        b = self.basis
        return PODResult(modes=b.modes, sigma=b.sigma,
                         energy_frac=b.energy_frac, cum_energy=b.cum_energy,
                         time_coeff=np.empty((b.n_modes, 0)))

    # Per case
    def case(self, value: int) -> CaseProjection:
        for c in self.cases:
            if c.value == value:
                return c
        raise KeyError(f"No simulation with {self.parameter} = {value} "
                       f"(have {self.values.tolist()}).")

    def reconstruct(self, value: int, r: int) -> np.ndarray:
        """Rank-``r`` field of one simulation in physical units (N/m)."""
        return self.case(value).reconstruct(r)

    @property
    def relative_error(self) -> np.ndarray:
        """``(n_cases, n_modes)`` relative fluctuation error versus ``r``."""
        return _stack_errors(self.cases)

    @property
    def mode_share(self) -> np.ndarray:
        """``(n_cases, n_modes)``: share of each case's fluctuation energy in
        each global mode."""
        return np.array([c.pod.energy_frac for c in self.cases])

    @property
    def n99_cases(self) -> np.ndarray:
        """Global modes needed for 99 % per case (``-1`` if not reached)."""
        return _ints([c.n99 for c in self.cases])

    def energy_modes_required(self, metric: str | None = None) -> np.ndarray:
        """Per case: modes for the energy-injection tolerance (-1: n/a)."""
        return _ints([c.energy_modes_required(metric) for c in self.cases])

    # Local reference
    @property
    def has_local(self) -> bool:
        return all(c.local is not None for c in self.cases)

    @property
    def local_n99(self) -> np.ndarray:
        return _ints([None if c.local is None else c.local.n99
                      for c in self.cases])

    def local_energy_modes_required(self, metric: str | None = None
                                    ) -> np.ndarray:
        return _ints([None if c.local is None
                      else _n_required(c.local.energy, metric)
                      for c in self.cases])

    # Export
    def without_raw(self) -> "GlobalPODResult":
        """Copy without the raw force fields (much smaller to save)."""
        return dataclasses.replace(
            self, cases=tuple(_without_raw(c) for c in self.cases))


@dataclass(frozen=True, eq=False)
class LeaveOneOutResult:
    """Projection of every simulation onto a basis fitted *without* it.

    This measures how well the global basis represents an *unseen*
    simulation (no leakage, unlike projecting a case onto a basis it helped
    build). Each held-out case is centred like the training cases are
    (``"case"``: its own mean, i.e. the mean is assumed to be modelled
    separately; ``"global"``: the training mean).

    Attributes
    ----------
    parameter : str
    values : ndarray of int
    cases : tuple of CaseProjection
        ``cases[i]`` is simulation ``values[i]`` projected onto the basis of
        all the others.
    bases : tuple of GlobalBasis
        The basis used for each.
    """
    parameter: str
    values: np.ndarray
    cases: tuple[CaseProjection, ...]
    bases: tuple[GlobalBasis, ...]

    @property
    def relative_error(self) -> np.ndarray:
        """``(n_cases, n_modes)`` relative fluctuation error versus ``r``."""
        return _stack_errors(self.cases)

    def energy_modes_required(self, metric: str | None = None) -> np.ndarray:
        return _ints([c.energy_modes_required(metric) for c in self.cases])

    def without_raw(self) -> "LeaveOneOutResult":
        """Copy without the raw force fields (much smaller to save)."""
        return dataclasses.replace(
            self, cases=tuple(_without_raw(c) for c in self.cases))


# =============================================================================
# Helpers
# =============================================================================
def _ints(values: Sequence[int | None]) -> np.ndarray:
    return np.array([-1 if v is None else v for v in values], dtype=int)


def _n_required(energy: EnergyInjection | None, metric: str | None
                ) -> int | None:
    if energy is None:
        return None
    if metric is None:
        metric = next(iter(energy.n_required))
    return energy.n_required[metric]


def _stack_errors(cases: Sequence[CaseProjection]) -> np.ndarray:
    n = min(len(c.relative_error) for c in cases)
    return np.array([c.relative_error[:n] for c in cases])


def _without_raw(case: CaseProjection) -> CaseProjection:
    data = dataclasses.replace(case.force_file.data, Fx=None, Fz=None)
    return dataclasses.replace(
        case, field=None,
        force_file=dataclasses.replace(case.force_file, data=data))


def _velocities(case: CaseSnapshots) -> tuple[np.ndarray, np.ndarray]:
    cfg = case.config
    omega = 2 * np.pi * cfg.f_forcing
    amp = cfg.effective_motion_amplitude
    return (cfg.motion.velocity("x", case.y, case.t, amp, omega),
            cfg.motion.velocity("z", case.y, case.t, amp, omega))


def _energy(case: CaseSnapshots, model: PhysicalReconstructor,
            config: GlobalPODConfig) -> EnergyInjection:
    base = config.base
    vx, vz = _velocities(case)
    return energy_injection_analysis(
        model=model, field=case.field, field_info=case.field_info,
        y=case.y, t=case.t, vx=vx, vz=vz, tolerance=base.energy_tolerance,
        aggregators=base.energy_error_metrics,
        reference=base.energy_error_reference,
        max_modes=config.energy_max_modes)


def local_reference(case: CaseSnapshots, config: GlobalPODConfig,
                    energy: bool = True) -> LocalReference:
    """Single-case POD of ``case`` (own mean removed)."""
    preprocessing = Preprocessing(scaling=case.scaling,
                                  mean=case.scaled.mean(axis=1))
    pod = compute_pod(preprocessing.forward(case.field))
    result = None
    if energy and config.base.do_energy_analysis:
        result = _energy(case, PhysicalReconstructor(pod, preprocessing),
                         config)
    return LocalReference(pod=pod, n99=pod.modes_for_energy(0.99),
                          energy=result)


# =============================================================================
# Projection of one simulation onto a basis
# =============================================================================
def project_case(case: CaseSnapshots, basis: GlobalBasis,
                 config: GlobalPODConfig, *, with_local: bool = False,
                 with_energy: bool = True) -> CaseProjection:
    """Project a simulation onto ``basis`` and run the per-case analyses.

    The simulation need not have been used to fit the basis.

    Parameters
    ----------
    with_local : bool
        Also compute the single-case POD of the simulation as a reference.
    with_energy : bool
        Run the energy-injection analysis (if enabled in ``config.base``).
    """
    base = config.base
    fluctuation, mean = basis.fluctuation(case)
    coeff = basis.coefficients(fluctuation)
    total = float(np.sum(fluctuation * fluctuation))
    mode_energy = np.sum(coeff * coeff, axis=1)
    denom = total if total > 0 else 1.0
    pod = PODResult(modes=basis.modes, sigma=np.sqrt(mode_energy),
                    energy_frac=mode_energy / denom,
                    cum_energy=np.cumsum(mode_energy) / denom,
                    time_coeff=coeff)
    idx_99 = np.searchsorted(pod.cum_energy, 0.99 - 1e-12)
    n99 = int(idx_99 + 1) if idx_99 < len(pod.cum_energy) else None

    n_report = min(base.n_modes_show, basis.n_modes)
    dominant = np.array([dominant_frequency(coeff[k], case.dt)[0]
                         for k in range(n_report)])
    preprocessing = Preprocessing(scaling=case.scaling, mean=mean)
    model = PhysicalReconstructor(pod, preprocessing)

    energy = None
    if with_energy and base.do_energy_analysis:
        energy = _energy(case, model, config)

    phases = None
    if base.phase_modes > 0:
        try:
            phases = harmonic_phase_analysis(
                time_coeff=coeff, t=case.t, f_forcing=base.f_forcing,
                n_modes=base.phase_modes, n_harmonics=base.phase_harmonics,
                unwrap_phase=base.unwrap_phase)
        except ValueError as err:
            logger.warning("  %s = %s: phase analysis skipped (%s)",
                           config.parameter, case.value, err)

    local = local_reference(case, config, energy=with_energy) \
        if with_local else None

    return CaseProjection(
        value=case.value, config=case.config, force_file=case.force_file,
        field=case.field, field_info=case.field_info,
        preprocessing=preprocessing, pod=pod, dominant_freqs=dominant,
        dt=case.dt, n_periods=case.n_periods, n99=n99, energy=energy,
        phases=phases, local=local)


def subspace_angles(cases: Sequence[CaseProjection], basis: GlobalBasis,
                    k: int) -> SubspaceAngles:
    """Angles between the local subspaces of neighbouring simulations and to
    the global basis (needs ``case.local``)."""
    k = min(k, basis.n_modes, *(c.local.pod.n_modes for c in cases))
    local = [c.local.pod.modes[:, :k] for c in cases]
    neighbour = np.degrees([principal_angles(a, b)
                            for a, b in zip(local[:-1], local[1:])])
    to_global = np.degrees([principal_angles(a, basis.modes[:, :k])
                            for a in local])
    return SubspaceAngles(values=np.array([c.value for c in cases], int),
                          k=k, neighbour=neighbour, to_global=to_global)


# =============================================================================
# Entry points on prepared snapshots
# =============================================================================
def analyse_snapshots(cases: Sequence[CaseSnapshots], config: GlobalPODConfig,
                      missing: Sequence[int] = ()) -> GlobalPODResult:
    """Fit the global basis to ``cases`` and analyse every simulation.

    Parameters
    ----------
    cases : sequence of CaseSnapshots
        See :func:`viv_rom.gpod.snapshots.load_snapshots`.
    config : GlobalPODConfig
    missing : sequence of int
        Requested values that had no file (only recorded in the result).
    """
    cases = list(cases)
    basis = fit_global_basis(cases, centering=config.centering,
                             weighting=config.weighting,
                             n_modes_keep=config.n_modes_keep)
    logger.info("  global basis: %d modes from %d simulations (%d "
                "snapshots); 99%% of the pooled energy in %s modes",
                basis.n_modes, len(cases), basis.n_snapshots.sum(),
                basis.modes_for_energy(0.99))

    projections = []
    for case in cases:
        logger.info("  projecting %s = %s ...", config.parameter, case.value)
        projections.append(project_case(
            case, basis, config, with_local=config.compute_local))

    angles = None
    if config.compute_local:
        angles = subspace_angles(projections, basis, config.angle_modes)

    return GlobalPODResult(
        parameter=config.parameter, values=basis.values, y=cases[0].y,
        field_info=cases[0].field_info, basis=basis,
        normalization_names=cases[0].scaling.names, cases=tuple(projections),
        n99=basis.modes_for_energy(0.99), angles=angles,
        missing=tuple(missing))


def leave_one_out_snapshots(cases: Sequence[CaseSnapshots],
                            config: GlobalPODConfig, *,
                            with_energy: bool = False) -> LeaveOneOutResult:
    """Project each simulation onto the basis fitted to all the others.

    Parameters
    ----------
    with_energy : bool
        Also run the energy-injection analysis of every held-out simulation
        (slower: one analysis per case).
    """
    cases = list(cases)
    if len(cases) < 3:
        raise ValueError("Leave-one-out needs at least three simulations.")
    projections, bases = [], []
    for i, case in enumerate(cases):
        logger.info("  leave out %s = %s ...", config.parameter, case.value)
        train = cases[:i] + cases[i + 1:]
        basis = fit_global_basis(train, centering=config.centering,
                                 weighting=config.weighting,
                                 n_modes_keep=config.n_modes_keep)
        projections.append(project_case(case, basis, config,
                                        with_energy=with_energy))
        bases.append(basis)
    return LeaveOneOutResult(
        parameter=config.parameter,
        values=np.array([c.value for c in cases], dtype=int),
        cases=tuple(projections), bases=tuple(bases))
