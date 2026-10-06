"""Loading of CFD force fields for a configured case."""
from __future__ import annotations

from viv_rom.config import CaseConfig
from viv_rom.domain.data import ForceFieldFile
from viv_rom.io.cfd_reader import HyReader


def load_case(config: CaseConfig, wind_speed: int | None = None
              ) -> ForceFieldFile:
    """Read the force field selected by ``config``.

    Parameters
    ----------
    config : CaseConfig
        Case selection and trimming (``n_skip``, ``n_last``).
    wind_speed : int, optional
        Override ``config.wind_speed`` (used by wind-speed sweeps).

    Raises
    ------
    FileNotFoundError
        If the file for the requested parameters does not exist.
    """
    reader = HyReader(config.data_dir)
    return reader.retrieve_force_field(
        amplitude=config.amplitude, pitch=config.pitch,
        inclination=config.inclination,
        wind_speed=config.wind_speed if wind_speed is None else wind_speed,
        n_skip=config.n_skip, n_last=config.n_last)
