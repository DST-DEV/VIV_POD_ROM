"""Configuration of a POD analysis run (replaces the script's globals)."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from viv_rom.config import CaseConfig
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE, ForcedMotion
from viv_rom.pod.core import FIELD_OPTIONS


@dataclass
class PODConfig(CaseConfig):
    """Case selection (see :class:`CaseConfig`) plus POD options.

    Attributes
    ----------
    field : {"fx", "fz", "combined"}
        Sectional force component(s) to include in the POD.
    n_modes_show : int
        Number of leading POD modes to plot/report.
    motion_amplitude : float or None
        Motion amplitude factor ``A`` used in the power check. ``None`` uses
        ``float(amplitude)`` (case a01 -> A = 1.0).
    station_y : float
        Spanwise position [m]; the first section at or beyond it is used for
        the time-series/orbit plots (81 m is the energy-injection peak).
    do_power_check : bool
        Run the physical power cross-check.
    motion : ForcedMotion
        Prescribed motion shapes.
    """
    field: str = "combined"
    n_modes_show: int = 4
    motion_amplitude: float | None = None
    station_y: float = 81.0
    do_power_check: bool = True
    motion: ForcedMotion = dc_field(default=FIRST_EDGEWISE_MODE, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.field not in FIELD_OPTIONS:
            raise ValueError(f"Unknown field option: {self.field!r} "
                             f"(use one of {FIELD_OPTIONS}).")

    @property
    def effective_motion_amplitude(self) -> float:
        return (float(self.amplitude) if self.motion_amplitude is None
                else self.motion_amplitude)

    @property
    def default_output_dir(self):
        return self.output_dir_for("pod")
