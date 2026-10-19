"""Configuration shared by all analysis methods."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from viv_rom.domain.blade import DEFAULT_BLADE_FILE
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE, ForcedMotion

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
class CaseConfig:
    """Which CFD case to load and how to trim it.

    Method-specific configs (``PODConfig``, ``SpectralConfig``) extend this.

    Attributes
    ----------
    data_dir : Path
        Folder containing the ``.1HY`` (and optionally ``.1work``) files.
    amplitude, pitch, inclination : int
        Case parameters, as encoded in the file name (``a01_p85_i50`` ->
        1, 85, 50).
    wind_speed : int
        Free-stream wind speed [m/s] to analyse (6-22).
    n_skip : int
        Rows dropped as initial transient (200 rows is one full period).
    n_last : int or None
        Rows kept after ``n_skip``. ``None`` keeps everything (recommended:
        several periods, so lock-in / beating behaviour is visible).
    f_forcing : float
        Prescribed excitation frequency [Hz].
    motion : ForcedMotion
        Prescribed motion shapes (direction/velocity along the span).
    motion_amplitude : float or None
        Motion amplitude factor ``A`` used wherever the prescribed velocity
        enters (power, energy injection, work weighting). ``None`` uses
        ``float(amplitude)`` (case a01 -> A = 1.0).
    blade_file : Path
        windIO ``.yaml`` description of the blade (chord, twist), used by
        the ``inflow`` frame. Default ``data/IEA_10MW/IEA-10-198-RWT.yaml``.
    """
    data_dir: Path = Path("./data/archive_1D")
    amplitude: int = 1
    pitch: int = 85
    inclination: int = 50
    wind_speed: int = 10
    n_skip: int = 1200
    n_last: int | None = None
    f_forcing: float = 0.624
    motion: ForcedMotion = field(default=FIRST_EDGEWISE_MODE, repr=False)
    motion_amplitude: float | None = None
    blade_file: Path = DEFAULT_BLADE_FILE

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.blade_file = Path(self.blade_file)

    @classmethod
    def from_case(cls, case: str, **kwargs):
        """Build a config from a case code, e.g. ``"a01_p85_i50"``."""
        return cls(**parse_case(case), **kwargs)

    @property
    def effective_motion_amplitude(self) -> float:
        return (float(self.amplitude) if self.motion_amplitude is None
                else self.motion_amplitude)

    @property
    def case(self) -> str:
        return (f"a{self.amplitude:02d}_p{self.pitch:02d}"
                f"_i{self.inclination:02d}")

    def output_dir_for(self, method: str) -> Path:
        """Default output folder: ``results/<case>_u<speed>/<method>``."""
        return Path("results") / f"{self.case}" / f"u{self.wind_speed:02d}" \
            / method
