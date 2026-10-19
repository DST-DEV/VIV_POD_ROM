"""Blade geometry (chord, twist) read from a windIO turbine description.

Only what the force frames and the Strouhal analysis need is read: the
spanwise coordinate of the reference axis and the ``chord`` and ``twist``
distributions of ``components.blade`` (windIO ``reference_axis.z``,
``outer_shape.chord`` / ``outer_shape.twist``). Each distribution is given on
its own non-dimensional grid along the blade; it is converted to the
dimensional span coordinate ``r`` [m] through ``reference_axis.z``.

``r`` is the distance from the blade root along the blade root axis. The CFD
station coordinate ``y`` is measured from the rotor axis, so
``r = y - y_offset`` (the hub radius, 2.8 m for the forced motion of
:data:`~viv_rom.domain.motion.FIRST_EDGEWISE_MODE`).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_BLADE_FILE = Path("data") / "IEA_10MW" / "IEA-10-198-RWT.yaml"

#: Stations further outside the blade than this fraction of the span raise
#: an error instead of being clamped to the root / tip value.
_EXTRAPOLATION_TOLERANCE = 0.005


@dataclass(frozen=True, eq=False)
class BladeGeometry:
    """Chord and twist along the blade.

    Attributes
    ----------
    name : str
        Turbine name from the windIO file.
    span_chord, chord : ndarray
        Span coordinate ``r`` [m] and chord [m] of the chord distribution.
    span_twist, twist : ndarray
        Span coordinate ``r`` [m] and twist [rad] of the twist distribution
        (windIO sign: positive twist turns the nose towards the wind, as in
        HAWC2).
    y_offset : float
        Hub radius [m]: CFD coordinate ``y`` of the blade root.
    """
    name: str
    span_chord: np.ndarray
    chord: np.ndarray
    span_twist: np.ndarray
    twist: np.ndarray
    y_offset: float = 2.8

    # io
    @classmethod
    def from_windio(cls, path: str | Path, y_offset: float = 2.8,
                    twist_unit: str = "deg") -> "BladeGeometry":
        """Read the blade of a windIO ``.yaml`` turbine description.

        Parameters
        ----------
        path : str or Path
            windIO file.
        y_offset : float
            Hub radius [m] (CFD ``y`` of the blade root).
        twist_unit : {"deg", "rad"}
            Unit of ``outer_shape.twist.values`` in the file (the IEA-10
            file is in degrees).
        """
        import yaml

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Blade geometry file not found: {path}")
        if twist_unit not in ("deg", "rad"):
            raise ValueError("twist_unit must be 'deg' or 'rad'.")
        with path.open(encoding="utf-8") as fh:
            turbine = yaml.safe_load(fh)
        try:
            blade = turbine["components"]["blade"]
            axis_z = blade["reference_axis"]["z"]
            shape = blade["outer_shape"]
            chord, twist = shape["chord"], shape["twist"]
        except (KeyError, TypeError) as err:
            raise ValueError(f"{path} is not a windIO blade description "
                             f"(missing {err}).") from None

        grid_z = np.asarray(axis_z["grid"], float)
        z = np.asarray(axis_z["values"], float)
        if np.any(np.diff(z) <= 0):
            raise ValueError("reference_axis.z must increase from root to "
                             "tip.")

        def on_span(entry):
            r = np.interp(np.asarray(entry["grid"], float), grid_z, z)
            return r, np.asarray(entry["values"], float)

        r_c, c = on_span(chord)
        r_t, tw = on_span(twist)
        if twist_unit == "deg":
            tw = np.deg2rad(tw)
        return cls(name=str(turbine.get("name", path.stem)),
                   span_chord=r_c, chord=c, span_twist=r_t, twist=tw,
                   y_offset=y_offset)

    # Geometry
    @property
    def length(self) -> float:
        """Blade length along the root axis [m]."""
        return float(max(self.span_chord[-1], self.span_twist[-1]))

    def span_coordinate(self, y: np.ndarray) -> np.ndarray:
        """Blade span coordinate ``r = y - y_offset`` [m] of CFD stations."""
        r = np.asarray(y, float) - self.y_offset
        tol = _EXTRAPOLATION_TOLERANCE * self.length
        if r.min() < -tol or r.max() > self.length + tol:
            raise ValueError(
                f"Stations y = {np.min(y):.2f} ... {np.max(y):.2f} m lie "
                f"outside the blade (y_offset = {self.y_offset} m, length "
                f"{self.length:.2f} m); check y_offset.")
        return r

    def chord_at(self, y: np.ndarray) -> np.ndarray:
        """Chord [m] at CFD stations ``y`` (clamped inside the tolerance)."""
        return np.interp(self.span_coordinate(y), self.span_chord,
                         self.chord)

    def twist_at(self, y: np.ndarray) -> np.ndarray:
        """Twist [rad] at CFD stations ``y`` (windIO sign)."""
        return np.interp(self.span_coordinate(y), self.span_twist,
                         self.twist)


@functools.lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int, y_offset: float,
                 twist_unit: str) -> BladeGeometry:
    return BladeGeometry.from_windio(path, y_offset, twist_unit)


def load_blade_geometry(path: str | Path | None = None, y_offset: float = 2.8,
                        twist_unit: str = "deg") -> BladeGeometry:
    """:meth:`BladeGeometry.from_windio` with caching (reloaded if the file
    changes). ``path=None`` uses :data:`DEFAULT_BLADE_FILE`."""
    path = Path(DEFAULT_BLADE_FILE if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(f"Blade geometry file not found: {path} "
                                f"(default location: {DEFAULT_BLADE_FILE})")
    return _load_cached(str(path.resolve()), path.stat().st_mtime_ns,
                        float(y_offset), twist_unit)
