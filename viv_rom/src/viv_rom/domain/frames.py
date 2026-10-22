"""Coordinate frames in which the sectional forces can be analysed.

The CFD forces are given in the blade-root frame (edgewise ``Fx``, flapwise
``Fz``). The prescribed motion of a section, however, points in a
station-dependent direction in the ``(x, z)`` plane, so ``Fx`` and ``Fz``
each mix the force that does work on the motion with the force that does
not. :class:`MotionAlignedFrame` rotates ``(Fx, Fz)`` at every station into

    ``F_par  =  Fx*c_x + Fz*c_z``      along the local direction of motion
    ``F_perp = -Fx*c_z + Fz*c_x``      normal to it (+90 deg from the motion
                                       in the (x, z) plane)

with ``(c_x, c_z)`` the unit direction of the motion
(:meth:`ForcedMotion.direction`). The rotation is orthonormal, so the force
magnitude is unchanged, and the sectional power is carried entirely by
``F_par`` (``p = F_par * |v|``).

:class:`InflowAlignedFrame` (``"inflow"``) instead aligns the force with the
free stream: ``F_D`` along the inflow (drag direction) and ``F_L`` normal to
it (+90 deg in the (x, z) plane): the components along and normal to the
local angle of attack. The shedding force that carries the Strouhal
frequency is ``F_L``; the drag oscillates at twice that. The inflow direction
follows from the pitch angle (fixed ``z_0`` inflow, root frame rotated by the
pitch; see :mod:`viv_rom.domain.inflow`); the blade geometry (windIO file) gives the
local angle of attack and projected width for a Strouhal analysis.

A frame is a strategy: anything with ``name``, ``labels`` and
``transform(y, Fx, Fz)`` can be used. Register new ones with
:func:`register_frame` to make them selectable by name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from viv_rom.domain.blade import BladeGeometry, load_blade_geometry
from viv_rom.domain.inflow import (InflowConvention, normal_speed_factor,
                                   spanwise_speed_factor)
from viv_rom.domain.motion import ForcedMotion


@runtime_checkable
class ForceFrame(Protocol):
    """Strategy interface: express ``(Fx, Fz)`` in some coordinate frame."""

    name: str
    labels: Mapping[str, str]  # component key -> axis/plot label

    def transform(self, y: np.ndarray, Fx: np.ndarray, Fz: np.ndarray
                  ) -> dict[str, np.ndarray]:
        """Components ``{key: array (n_stations, n_steps)}`` (ordered)."""
        ...


# ---------------------------------------------------------------------------
# Selection by name
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., ForceFrame]] = {}


def register_frame(name: str):
    """Decorator registering a factory by name.

    The factory is called as ``factory(motion, **context)``; ``context`` is
    whatever :func:`make_frame` was given (the case's ``pitch``,
    ``inclination`` and ``blade_file``). Factories ignore what they do not
    need (``lambda motion, **_: ...``).
    """
    def decorator(factory):
        if name in _REGISTRY:
            raise ValueError(f"Frame {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_frames() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def make_frame(name: str, motion: ForcedMotion, **context) -> ForceFrame:
    """Build the registered frame ``name``.

    ``context`` (``pitch`` [deg], ``inclination`` [deg], ``blade_file``) is
    handed to the factory; frames that do not depend on it ignore it.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown frame {name!r} "
                         f"(use one of {available_frames()}).") from None
    return factory(motion, **context)


# ---------------------------------------------------------------------------
# Implementation of ForceFrames
# ---------------------------------------------------------------------------
class BladeFrame:
    """The blade-root frame as delivered by the CFD (no rotation)."""
    name = "blade"
    labels = {"fx": "Fx (edgewise)", "fz": "Fz (flapwise)"}

    def transform(self, y, Fx, Fz):
        return {"fx": Fx, "fz": Fz}


class MotionAlignedFrame:
    """Per-station frame with one axis along the local direction of motion.

    Parameters
    ----------
    motion : ForcedMotion
        Prescribed motion; its mode shapes give the direction at each
        station.
    """
    name = "motion"
    labels = {"fpar": "F∥ (along motion)", "fperp": "F⊥ (normal to motion)"}

    def __init__(self, motion: ForcedMotion):
        self.motion = motion

    def transform(self, y, Fx, Fz):
        cx, cz = self.motion.direction(y)
        cx, cz = cx[:, None], cz[:, None]
        return {"fpar": Fx * cx + Fz * cz,
                "fperp": -Fx * cz + Fz * cx}


class InflowAlignedFrame:
    """Force frame aligned with the local flow (drag / lift directions).

    ``(Fx, Fz)`` of every section is resolved along the root direction.
    The section's chord-aligned force angle thus only depends on the twist::

        F_D =  Fx*d_x - Fz*d_z     along the inflow
        F_L = Fx*d_z + Fz*d_x     normal to it, +90 deg from the inflow

    with ``(d_x, d_z) = (cos beta, sin beta)``. The rotation is orthonormal.

    The blade geometry is read lazily from the windIO file, never by
    :meth:`transform`.

    Parameters
    ----------
    pitch : float
        Blade pitch angle [deg] of the case.
    inclination : float
        Inclination angle [deg] of the case (see :mod:`viv_rom.domain.
        inflow`); only changes the speed normal to the blade.
    convention : InflowConvention, optional
        Angle conventions; the defaults are documented there.
    geometry : BladeGeometry, optional
        Blade geometry. If omitted it is read from ``blade_file``.
    blade_file : str or Path, optional
        windIO ``.yaml`` file (default
        ``data/IEA_10MW/IEA-10-198-RWT.yaml``).
    y_offset : float, optional
        Hub radius [m]; default ``2.8`` (that of the forced motion).
    """
    name = "inflow"
    labels = {"fdrag": "F_D (along inflow)", "flift": "F_L (normal to inflow)"}

    def __init__(self, pitch: float, inclination: float = 0.0, *,
                 convention: InflowConvention = InflowConvention(),
                 geometry: BladeGeometry | None = None,
                 blade_file: str | Path | None = None,
                 y_offset: float = 2.8):
        self.pitch = float(pitch)
        self.inclination = float(inclination)
        self.convention = convention
        self.blade_file = blade_file
        self.y_offset = y_offset
        self._geometry = geometry

    # The rotation
    @property
    def flow_angle(self) -> float:
        """Free-stream direction in the (x, z) plane [rad from +x to +z]."""
        return self.convention.flow_angle(self.pitch)

    def direction(self) -> tuple[float, float]:
        """Unit vector ``(d_x, d_z)`` of the free stream."""
        theta = self.convention.flow_angle(self.pitch)
        return float(np.cos(theta)), float(np.sin(theta))

    def transform(self, y, Fx, Fz):
        dx, dz = self.direction()
        dx, dz = dx[:, None], dz[:, None]
        return {"fdrag": Fx * dx - Fz * dz,
                "flift": -Fx * dz - Fz * dx}

    # Local flow quantities
    @property
    def geometry(self) -> BladeGeometry:
        if self._geometry is None:
            self._geometry = load_blade_geometry(self.blade_file,
                                                 self.y_offset)
        return self._geometry

    def angle_of_attack(self, y) -> np.ndarray:
        """Angle from chord to flow at stations ``y`` [rad], ``(-pi, pi]``.

        Twist-dependent, so it varies along the span.
        """
        return self.convention.angle_of_attack(
            self.pitch, self.geometry.twist_at(y))

    def projected_width(self, y) -> np.ndarray:
        """Cross-flow width ``D(y) = c(y) |sin(alpha(y))|`` [m]."""
        return (self.geometry.chord_at(y)
                * np.abs(np.sin(self.angle_of_attack(y))))

    def normal_speed(self, wind_speed: float) -> float:
        """Wind speed normal to the blade axis, ``U cos(inclination)``."""
        return wind_speed * normal_speed_factor(self.inclination)

    def spanwise_speed(self, wind_speed: float) -> float:
        """Wind speed along the blade axis, ``U sin(inclination)``."""
        return wind_speed * spanwise_speed_factor(self.inclination)

    def describe(self) -> str:
        """One-line description for logs and summaries."""
        dx, dz = self.direction
        return (f"flow direction {np.degrees(self.flow_angle):.1f} deg "
                f"(d = [{dx:.3f}, {dz:.3f}]), pitch "
                f"{self.pitch:g} deg, inclination {self.inclination:g} deg")


register_frame("blade")(lambda motion, **_: BladeFrame())
register_frame("motion")(lambda motion, **_: MotionAlignedFrame(motion))
register_frame("inflow")(
    lambda motion, pitch=0.0, inclination=0.0, blade_file=None, **_:
    InflowAlignedFrame(pitch, inclination, blade_file=blade_file,
                       y_offset=motion.y_offset))
