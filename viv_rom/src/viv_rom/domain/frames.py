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

:class:`RelativeInflowFrame` (``"rel_inflow"``) additionally accounts for the
motion of the section: the air reaches a moving section with the *relative*
velocity ``U_n d - v(y, t)``, whose direction changes along the span and in
time. The rotation then depends on ``(y, t)``.

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
                                   spanwise_speed_factor, wrap_angle)
from viv_rom.domain.motion import ForcedMotion


@runtime_checkable
class ForceFrame(Protocol):
    """Strategy interface: express ``(Fx, Fz)`` in some coordinate frame."""

    name: str
    labels: Mapping[str, str]  # component key -> axis/plot label

    def transform(self, y: np.ndarray, Fx: np.ndarray, Fz: np.ndarray
                  ) -> dict[str, np.ndarray]:
        """Components ``{key: array (n_stations, n_steps)}`` (ordered).

        A frame that also depends on time sets ``time_dependent = True`` and
        takes the time vector as a fourth argument ``t`` (shape
        ``(n_steps,)``); call frames through :func:`transform_forces`.
        """
        ...


def transform_forces(frame: ForceFrame, y, Fx, Fz, t=None
                     ) -> dict[str, np.ndarray]:
    """Apply any frame; time is handed only to frames that depend on it."""
    if getattr(frame, "time_dependent", False):
        if t is None:
            raise ValueError(f"Frame {frame.name!r} depends on time; the "
                             "time vector t is required.")
        return frame.transform(y, Fx, Fz, t=t)
    return frame.transform(y, Fx, Fz)


# ---------------------------------------------------------------------------
# Selection by name
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., ForceFrame]] = {}


def register_frame(name: str):
    """Decorator registering a factory by name.

    The factory is called as ``factory(motion, **context)``; ``context`` is
    whatever :func:`make_frame` was given (the case's ``pitch``,
    ``inclination``, ``blade_file``, ``wind_speed``, ``amplitude`` and
    ``f_forcing``). Factories ignore what they do not
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

    ``context`` (``pitch`` [deg], ``inclination`` [deg], ``blade_file``,
    ``wind_speed`` [m/s], ``amplitude``, ``f_forcing`` [Hz]) is handed to
    the factory; frames that do not depend on it ignore it.
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

    ``(Fx, Fz)`` of every section is resolved along the local flow, i.e.
    along the direction at the angle of attack ``alpha(y)`` to the chord, and
    normal to it. The section's chord angle ``theta_c(y)`` (twist) and
    ``alpha(y)`` add up to the flow direction ``phi`` in the root frame
    (``theta_c + alpha = phi``), so the twist cancels and the rotation angle
    is the same at every station::

        F_D =  Fx*d_x + Fz*d_z     along the inflow
        F_L = -Fx*d_z + Fz*d_x     normal to it, +90 deg from the inflow

    with ``(d_x, d_z) = (cos phi, sin phi)`` the flow direction,
    ``phi = convention.flow_angle(pitch)``. The rotation is orthonormal.
    ``F_L`` has the sign of the +90 deg direction in the ``(x, z)`` plane,
    not of an aerodynamic lift (which would depend on the side of the
    chord the flow meets).

    The blade geometry is only needed for the local flow quantities
    (:meth:`angle_of_attack`, :meth:`projected_width`) and is read
    lazily from the windIO file, never by :meth:`transform`.

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

    @property
    def direction(self) -> tuple[float, float]:
        """Unit vector ``(d_x, d_z)`` of the free stream.

        Describes the angle between the fixed inflow coordinate system
        (x0, z0) and the root coordinate system (xr, zr)
        """
        theta = self.convention.flow_angle(self.pitch)
        return float(np.cos(theta)), float(np.sin(theta))

    def transform(self, y, Fx, Fz):
        dx, dz = self.direction
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

    def angle_of_attack(self, y, *args, **kwargs) -> np.ndarray:
        """Angle from chord to flow at stations ``y`` [rad], ``(-pi, pi]``.

        Twist-dependent, so it varies along the span.
        """
        return self.convention.angle_of_attack(
            self.pitch, self.geometry.twist_at(y))

    def projected_width(self, y, *args, **kwargs) -> np.ndarray:
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


class RelativeInflowFrame:
    """Force frame aligned with the inflow *relative to the moving section*.

    The section moves with the prescribed velocity ``v(y, t)`` of the forced
    motion (:meth:`ForcedMotion.velocity`, in the root ``(x, z)`` plane), so
    the air reaches it with the relative velocity::

        w(y, t) = U_n d - v(y, t),     U_n = U cos(inclination)

    with ``d`` the free-stream direction of :class:`InflowAlignedFrame` and
    ``U_n`` the wind speed in the section plane. ``(Fx, Fz)`` is resolved
    along ``w`` and normal to it, with the direction angle
    ``phi_rel(y, t) = atan2(w_z, w_x)``::

        F_D,rel =  Fx*cos(phi_rel) + Fz*sin(phi_rel)    along the relative flow
        F_L,rel = -Fx*sin(phi_rel) + Fz*cos(phi_rel)    normal to it

    The rotation is orthonormal at every ``(y, t)`` but varies along the span
    (the motion shape differs per station) and in time (the velocity
    oscillates at the forcing frequency). The effective angle of attack is
    ``alpha_eff(y, t) = phi_rel(y, t) - theta_c(y)``. Relative to the
    static one it changes by ``phi_rel - phi``, a swing of several degrees to
    tens of degrees at low wind speed and large tip velocity.

    Because the rotation oscillates at the forcing frequency, a force that is
    harmonic in the root frame gets sum / difference frequencies in this
    frame (e.g. at the shedding frequency +/- the forcing frequency). The
    components are therefore not a drop-in replacement for the static frames
    in phase / energy analyses.

    Parameters
    ----------
    inflow : InflowAlignedFrame
        The static inflow frame (pitch, inclination, conventions, geometry).
    motion : ForcedMotion
        Prescribed motion shapes.
    wind_speed : float
        Free-stream speed ``U`` [m/s].
    amplitude : float
        Motion amplitude factor ``A`` (as ``CaseConfig.effective_motion_
        amplitude``).
    f_forcing : float
        Forcing frequency [Hz]. The time vector passed to :meth:`transform`
        is the simulation time of the data (displacement ``A phi sin(omega
        t)``, velocity ``A omega phi cos(omega t)``), as everywhere else.
    """
    name = "rel_inflow"
    time_dependent = True
    labels = {"fd_rel": "F_D,rel (along relative inflow)",
              "fl_rel": "F_L,rel (normal to relative inflow)"}

    def __init__(self, inflow: InflowAlignedFrame, motion: ForcedMotion,
                 wind_speed: float, amplitude: float = 1.0,
                 f_forcing: float = 0.624):
        if wind_speed < 0:
            raise ValueError("wind_speed must be non-negative.")
        self.inflow = inflow
        self.motion = motion
        self.wind_speed = float(wind_speed)
        self.amplitude = float(amplitude)
        self.f_forcing = float(f_forcing)

    @property
    def geometry(self):
        return self.inflow.geometry

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.f_forcing

    # Relative flow field
    def section_velocity(self, y, t) -> tuple[np.ndarray, np.ndarray]:
        """Prescribed section velocity ``(v_x, v_z)``, ``(n_stations, n_t)``."""
        y, t = np.asarray(y, float), np.asarray(t, float)
        return (self.motion.velocity("x", y, t, self.amplitude, self.omega),
                self.motion.velocity("z", y, t, self.amplitude, self.omega))

    def relative_velocity(self, y, t) -> tuple[np.ndarray, np.ndarray]:
        """Air velocity relative to the section, ``(w_x, w_z)`` [m/s]."""
        dx, dz = self.inflow.direction
        un = self.wind_speed
        vx, vz = self.section_velocity(y, t)
        return un * dx - vx, un * dz - vz

    def relative_speed(self, y, t) -> np.ndarray:
        """Magnitude of the relative velocity [m/s], ``(n_stations, n_t)``."""
        return np.hypot(*self.relative_velocity(y, t))

    def flow_angle(self, y, t) -> np.ndarray:
        """Direction ``phi_rel`` of the relative flow [rad], ``(n, n_t)``."""
        wx, wz = self.relative_velocity(y, t)
        return np.arctan2(wz, wx)

    def angle_shift(self, y, t) -> np.ndarray:
        """``phi_rel - phi``: change of the flow direction by the motion
        [rad], wrapped to ``(-pi, pi]``."""
        return wrap_angle(self.flow_angle(y, t) - self.inflow.flow_angle)

    # The rotation
    def transform(self, y, Fx, Fz, t=None):
        if t is None:
            raise ValueError("RelativeInflowFrame needs the time vector t.")
        t = np.asarray(t, float)
        Fx, Fz = np.asarray(Fx), np.asarray(Fz)
        if Fx.shape != Fz.shape or Fx.ndim != 2 or Fx.shape[1] != t.size \
                or Fx.shape[0] != np.size(y):
            raise ValueError(
                f"Fx/Fz must have shape (len(y), len(t)) = "
                f"({np.size(y)}, {t.size}), got {Fx.shape} and {Fz.shape}.")
        phi = self.flow_angle(y, t)
        c, s = np.cos(phi), np.sin(phi)
        return {"fd_rel": -Fx * c - Fz * s,
                "fl_rel": -Fx * s + Fz * c}

    # Local flow quantities (geometry)
    def angle_of_attack(self, y, t) -> np.ndarray:
        """Effective angle of attack ``phi_rel - theta_c(y)`` [rad],
        ``(n_stations, n_t)``, wrapped to ``(-pi, pi]``."""
        theta_c = self.inflow.convention.chord_angle(
            self.inflow.geometry.twist_at(y))
        return wrap_angle(self.flow_angle(y, t) - theta_c[:, None])

    def projected_width(self, y, t) -> np.ndarray:
        """``D(y, t) = c(y) |sin alpha_eff(y, t)|`` [m]."""
        return (self.inflow.geometry.chord_at(y)[:, None]
                * np.abs(np.sin(self.angle_of_attack(y, t))))

    def describe(self) -> str:
        un = self.inflow.normal_speed(self.wind_speed)
        y = np.linspace(self.inflow.y_offset, self.inflow.y_offset + 100, 200)
        t = np.linspace(0, 1 / self.f_forcing, 64, endpoint=False)
        swing = np.degrees(np.abs(self.angle_shift(y, t))).max()
        vmax = np.hypot(*self.section_velocity(y, t)).max()
        return (f"relative inflow: static {self.inflow.describe()}; "
                f"U = {self.wind_speed:g} m/s (normal {un:.2f} m/s), "
                f"A = {self.amplitude:g}, f = {self.f_forcing:g} Hz, "
                f"peak section speed {vmax:.2f} m/s, flow-angle swing up to "
                f"{swing:.1f} deg")


register_frame("blade")(lambda motion, **_: BladeFrame())
register_frame("motion")(lambda motion, **_: MotionAlignedFrame(motion))
register_frame("inflow")(
    lambda motion, pitch=0.0, inclination=0.0, blade_file=None, **_:
    InflowAlignedFrame(pitch, inclination, blade_file=blade_file,
                       y_offset=motion.y_offset))
register_frame("rel_inflow")(
    lambda motion, pitch=0.0, inclination=0.0, blade_file=None,
    wind_speed=10.0, amplitude=1.0, f_forcing=0.624, **_:
    RelativeInflowFrame(
        InflowAlignedFrame(pitch, inclination, blade_file=blade_file,
                           y_offset=motion.y_offset),
        motion, wind_speed=wind_speed, amplitude=amplitude,
        f_forcing=f_forcing))
