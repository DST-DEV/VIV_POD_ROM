"""Utilities for reading ``.1HY`` force field files.

This module provides the :class:`HyReader` class for loading sectional force
data from ``.1HY`` files corresponding to a given set of simulation parameters.

The reader assumes that force field files follow the naming convention

    ``a{amp:02d}_p{pitch:02d}_i{incl:02d}_u{speed:02d}.HY``

where ``amp``, ``pitch``, ``incl``, and ``speed`` denote the forced motion
oscillation amplitude, pitch angle, inclination angle, and wind speed,
respectively.

The loaded data are returned as :class:`ForceFieldData` and
:class:`ForceFieldFile` dataclasses defined in ``viv_rom.domain.data``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from viv_rom.domain.data import ForceFieldFile, ForceFieldData


class HyReader:
    FILE_PATTERN = "a{amp:02d}_p{pitch:02d}_i{incl:02d}_u{speed:d}"

    def __init__(self, data_dir):
        """Initialize a ``.1HY`` file reader.

        Parameters
        ----------
        data_dir : pathlib.Path or str
            Directory containing the force field (``.1HY``) files. A string
            is converted to a :class:`pathlib.Path`.

        Raises
        ------
        TypeError
            If ``data_dir`` is not a :class:`pathlib.Path` or ``str``.
        OSError
            If ``data_dir`` does not exist or is not a directory.
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        elif not isinstance(data_dir, Path):
            raise TypeError("Expected a pathlib.Path object for parameter "
                            f"'data_dir', got '{type(data_dir).name}'.")

        if not data_dir.is_dir():
            raise OSError("Data directory does not exist.")

        self.data_dir = Path(data_dir)

    def retrieve_force_field(self, amplitude: float, pitch: float,
                             inclination: float, wind_speed: float,
                             n_skip: int = 0, n_last: None | int = None,
                             y_root: None | float = None
                             ) -> ForceFieldFile:
        """Retrieve a force field file given a simulation parameter set.

        amplitude : float
            Amplitude of the forced motion oscillation used in the
            simulation [m].
        pitch : float
            Pitch angle used in the simulation [deg].
        inclination : float
            Inclination angle used in the simulation [deg].
        wind_speed : float
            Wind speed used in the simulation [m/s].
        n_skip : int, optional
            Number of rows to skip at the beginning of the file. Default is 0.
        n_last : None | int, optional
            Number of rows to retain after ``n_skip``. If ``None``, all
            remaining rows are retained. Default is ``None``.
        y_root : None | float, optional
            Blade root diameter [m]. Only spanwise stations with y larger or
            equal than this value are kept. If None, keep all values.
            Default is None.

        Returns
        -------
        ForceFieldFile
            Force field corresponding to the requested simulation parameters.
            The returned dataclass contains the source file path, simulation
            metadata (amplitude, pitch, inclination, and wind speed), and the
            associated :class:`ForceFieldData`.
        """
        file_name = self.FILE_PATTERN.format(amp=amplitude, pitch=pitch,
                                             incl=inclination,
                                             speed=wind_speed)

        file_path = self.data_dir / (file_name + ".1HY")
        data = self.read_hy(path=file_path, n_skip=n_skip, n_last=n_last,
                            y_root=y_root)

        return ForceFieldFile(
            source_file=file_path,
            amplitude=amplitude,
            pitch=pitch,
            inclination=inclination,
            wind_speed=wind_speed,
            data=data
        )

    @staticmethod
    def read_hy(path: Path, n_skip: int = 0, n_last: None | int = None,
                y_root: None | float = None
                ) -> ForceFieldData:
        """Load a ``.1HY`` sectional-force file.

        Expected file layout:
        n_time rows x (2 + 4*n_sections) columns, whitespace separated,
        3 leading '#' comment lines.
            col 0         nstep
            col 1         time      [s]
            col 2 + 4*j   x[j]      [m]  spanwise station (constant in time)
            col 3 + 4*j   fx[j]     [N/m] sectional force, edgewise
            col 4 + 4*j   fy[j]     [N/m] unused
            col 5 + 4*j   fz[j]     [N/m] sectional force, flapwise

        Parameters
        ----------
        path : PurePath
            Path to the .1HY file
        n_skip : int, optional
            Number of rows to skip at the beginning of the file. Default is 0.
        n_last : None | int, optional
            Number or rows to keep after n_skip. If None, keep all rows.
            Default is None.
        y_root : None | float, optional
            Blade root diameter [m]. Only spanwise stations with y larger or
            equal than this value are kept. If None, keep all values.
            Default is None.

        Returns
        -------
        ForceFieldData:
            Force field data read from the provided ``.1HY`` file.
            The returned dataclass contains the time steps, spanwise section
            positions, edgewise sectional forces (Fx), and flapwise sectional
            forces (Fz).
        """
        if not path.suffix == ".1HY":
            raise ValueError("Invalid file type. Expected a .HY file, not "
                             f"'{path.suffix}'.")

        data = np.loadtxt(path, comments="#")
        if n_skip:
            data = data[n_skip:]
        if n_last:
            data = data[-n_last:]

        n_sections = (data.shape[1] - 2) // 4
        j = np.arange(n_sections)

        y = data[0, 2 + 4 * j]
        t = data[:, 1]
        Fx = data[:, 3 + 4 * j]
        Fz = data[:, 5 + 4 * j]

        if y_root:
            y_mask = y >= y_root
            y = y[y_mask]
            Fx = Fx[:, y_mask]
            Fz = Fz[:, y_mask]
        return ForceFieldData(y=y, t=t, Fx=Fx.T, Fz=Fz.T)
