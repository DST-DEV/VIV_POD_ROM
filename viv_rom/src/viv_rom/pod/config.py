"""Configuration of a POD analysis run (replaces the script's globals)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from viv_rom.domain.motion import FIRST_EDGEWISE_MODE, ForcedMotion
from viv_rom.pod.core import FIELD_OPTIONS

_CASE_RE = re.compile(r"^a(?P<amplitude>\d+)_p(?P<pitch>\d+)_i(?P<incl>\d+)$")


def parse_case(case: str) -> dict[str, int]:
    """Parse a case code such as ``"a01_p85_i50"`` into its parameters."""
    m = _CASE_RE.match(case)
    if m is None:
        raise ValueError(f"Invalid case code {case!r}; expected the form "
                         "'a01_p85_i50'.")
    return {"amplitude": int(m["amplitude"]), "pitch": int(m["pitch"]),
            "inclination": int(m["incl"])}


@dataclass
class PODConfig:
    """Parameters of one POD analysis.

    Attributes
    ----------
    data_dir : Path
        Folder containing the ``.1HY`` (and optionally ``.1work``) files.
    amplitude, pitch, inclination : int
        Case parameters, as encoded in the file name (``a01_p85_i50`` ->
        1, 85, 50).
    wind_speed : int
        Free-stream wind speed [m/s] to analyse (6-22).
    field : {"fx", "fz", "combined"}
        Sectional force component(s) to include in the POD.
    n_skip : int
        Rows dropped as initial transient (200 rows is one full period).
    n_last : int or None
        Rows kept after ``n_skip``. ``None`` keeps everything (recommended:
        several periods, so lock-in / beating behaviour is visible).
    n_modes_show : int
        Number of leading POD modes to plot/report.
    f_forcing : float
        Prescribed excitation frequency [Hz].
    motion_amplitude : float or None
        Motion amplitude factor ``A`` used in the power check. ``None`` uses
        ``float(amplitude)`` (case a01 -> A = 1.0).
    station_y : float
        Spanwise position [m]; the first section at or beyond it is used for
        the time-series/orbit plots.
    do_power_check : bool
        Run the physical power cross-check.
    motion : ForcedMotion
        Prescribed motion shapes.
    """
    data_dir: Path = Path("./data/archive_1D")
    amplitude: int = 1
    pitch: int = 85
    inclination: int = 50
    wind_speed: int = 10
    field: str = "combined"
    n_skip: int = 1200
    n_last: int | None = None
    n_modes_show: int = 4
    f_forcing: float = 0.624
    motion_amplitude: float | None = None
    station_y: float = 81.0
    do_power_check: bool = True
    motion: ForcedMotion = dc_field(default=FIRST_EDGEWISE_MODE, repr=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.field not in FIELD_OPTIONS:
            raise ValueError(f"Unknown field option: {self.field!r} "
                             f"(use one of {FIELD_OPTIONS}).")

    @classmethod
    def from_case(cls, case: str, **kwargs) -> "PODConfig":
        """Build a config from a case code, e.g. ``"a01_p85_i50"``."""
        return cls(**parse_case(case), **kwargs)

    @property
    def case(self) -> str:
        return (f"a{self.amplitude:02d}_p{self.pitch:02d}"
                f"_i{self.inclination:02d}")

    @property
    def effective_motion_amplitude(self) -> float:
        return (float(self.amplitude) if self.motion_amplitude is None
                else self.motion_amplitude)

    @property
    def default_output_dir(self) -> Path:
        return Path("results") / f"{self.case}_u{self.wind_speed:02d}" / "pod"
