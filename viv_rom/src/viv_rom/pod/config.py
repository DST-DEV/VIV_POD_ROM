"""Configuration of a POD analysis run (replaces the script's globals)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from viv_rom.config import CaseConfig
from viv_rom.pod.core import FIELD_OPTIONS
from viv_rom.pod.energy_injection import (resolve_aggregators,
                                          resolve_reference)
from viv_rom.pod.normalization import (RHO_AIR, Normalizer, make_normalizer)


@dataclass
class PODConfig(CaseConfig):
    """Case selection (see :class:`CaseConfig`) plus POD options.

    Attributes
    ----------
    field : {"fx", "fz", "combined"}
        Sectional force component(s) to include in the POD.
    n_modes_show : int
        Number of leading POD modes to plot/report.
    station_y : float
        Spanwise position [m]; the first section at or beyond it is used for
        the time-series/orbit plots (81 m is the energy-injection peak).
    do_power_check : bool
        Run the physical power cross-check.
    normalization : sequence of (str or Normalizer)
        Optional normalisation applied before mean removal and POD. Empty
        (default) means none. Items are registered names
        (``"dynamic_pressure"``, ``"work"``; see
        :func:`~viv_rom.pod.normalization.available_normalizers`) or
        :class:`~viv_rom.pod.normalization.Normalizer` instances for custom
        options, e.g. ``(WorkWeightNormalizer(floor=1e-2),)``. A single name
        is accepted too.
    air_density : float
        ``rho`` [kg/m^3] for the dynamic pressure ``q = 0.5 * rho * U**2``.
    do_energy_analysis : bool
        Compute the energy injection of each mode and the number of modes
        needed to reproduce the spanwise energy injection.
    energy_tolerance : float
        Accepted aggregated error of the reconstructed energy injection;
        0.01 means 99 % accuracy.
    energy_error_metrics : sequence of (str or callable)
        How per-station errors are combined: any of ``"mean"``, ``"rms"``,
        ``"max"`` or a callable ``station_errors -> float``.
    energy_error_reference : {"peak", "local"} or callable
        Station errors are scaled by the peak ``|E|`` over the span
        (default) or by the local ``|E_j|``.
    phase_modes : int
        Number of leading modes whose temporal-coefficient phase is
        reported (0 disables the phase analysis).
    phase_harmonics : int
        Harmonics ``1..n`` of the forcing frequency used for the phases.
    """
    field: str = "combined"
    n_modes_show: int = 4
    station_y: float = 81.0
    do_power_check: bool = True
    normalization: tuple[str | Normalizer, ...] = ()
    air_density: float = RHO_AIR
    do_energy_analysis: bool = True
    energy_tolerance: float = 0.01
    energy_error_metrics: tuple = ("mean", "rms", "max")
    energy_error_reference: str | Callable = "peak"
    phase_modes: int = 4
    phase_harmonics: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.field not in FIELD_OPTIONS:
            raise ValueError(f"Unknown field option: {self.field!r} "
                             f"(use one of {FIELD_OPTIONS}).")
        if isinstance(self.normalization, (str, Normalizer)):
            self.normalization = (self.normalization,)
        self.normalization = tuple(self.normalization)
        self.normalizers()  # fail early on unknown names

        self.energy_error_metrics = tuple(self.energy_error_metrics)
        resolve_aggregators(self.energy_error_metrics)
        resolve_reference(self.energy_error_reference)

        if self.energy_tolerance < 0:
            raise ValueError("energy_tolerance must be non-negative.")
        if self.phase_modes < 0 or self.phase_harmonics < 1:
            raise ValueError("phase_modes must be >= 0 and phase_harmonics "
                             ">= 1.")

    def normalizers(self) -> tuple[Normalizer, ...]:
        """The configured normalisation strategies as instances."""
        return tuple(make_normalizer(n) if isinstance(n, str) else n
                     for n in self.normalization)

    @property
    def default_output_dir(self):
        return self.output_dir_for("pod")
