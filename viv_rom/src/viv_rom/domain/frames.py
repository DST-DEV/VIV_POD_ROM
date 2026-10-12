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

A frame is a strategy: anything with ``name``, ``labels`` and
``transform(y, Fx, Fz)`` can be used. Register new ones with
:func:`register_frame` to make them selectable by name.
"""
from __future__ import annotations

from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

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
_REGISTRY: dict[str, Callable[[ForcedMotion], ForceFrame]] = {}


def register_frame(name: str):
    """Decorator registering a factory ``motion -> ForceFrame`` by name."""
    def decorator(factory):
        if name in _REGISTRY:
            raise ValueError(f"Frame {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_frames() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def make_frame(name: str, motion: ForcedMotion) -> ForceFrame:
    try:
        return _REGISTRY[name](motion)
    except KeyError:
        raise ValueError(f"Unknown frame {name!r} "
                         f"(use one of {available_frames()}).") from None


# ---------------------------------------------------------------------------
# Implementation of ForceFrames
# ---------------------------------------------------------------------------
# @register_frame("blade")
class BladeFrame:
    """The blade-root frame as delivered by the CFD (no rotation)."""
    name = "blade"
    labels = {"fx": "Fx (edgewise)", "fz": "Fz (flapwise)"}

    def transform(self, y, Fx, Fz):
        return {"fx": Fx, "fz": Fz}


# @register_frame("motion")
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

register_frame("blade")(lambda motion: BladeFrame())
register_frame("motion")(lambda motion: MotionAlignedFrame(motion))
