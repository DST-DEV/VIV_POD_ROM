from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True, frozen=True)
class ForceFieldData:
    """Container for data from a force field (.HY) file."""
    t: np.ndarray = field(doc="Time [s]")
    y: np.ndarray = field(doc="Spanwise blade section position [m] "
                          "(Measured in the blade root coordinate system).")
    Fx: np.ndarray = field(doc="Edgewise sectional force [N/m] "
                           "(Measured in the blade root coordinate system)."
                           "Shape (n_sections, n_steps)")
    Fz: np.ndarray = field(doc="Flapwise sectional force [N/m] "
                           "(Measured in the blade root coordinate system)."
                           "Shape (n_sections, n_steps)")


@dataclass(slots=True, frozen=True)
class ForceFieldFile:
    """Container for data and metadata of a force field (.HY) file."""
    source_file: Path
    amplitude: float = field(doc="Forced motion oscillation [m]")
    pitch: float = field(doc="Pitch angle of the blade [deg]")
    inclination: float = field(doc="Inclination angle of the blade [deg]")
    wind_speed: float = field(doc="Mean wind speed [m/s]")
    data: ForceFieldData = field(repr=False, compare=True)
