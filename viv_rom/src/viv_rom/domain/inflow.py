"""Where the free stream points in the blade-root frame.

THE angle conventions of the project live here, in one place.

**Definition (from the CFD set-up).** The free stream has a fixed direction
``x_0`` that does not depend on the pitch. Forces and motion are given in the
blade-root frame, which is ``x_0`` rotated *clockwise* by the pitch angle
``p``; ``x_r`` is the edgewise and ``z_r`` the flapwise axis. The twist rotates
each section *counter-clockwise* relative to the root frame (``x_c`` and
``z_c``).

All angles below are measured in the root ``(x_r, z_r)`` plane from ``+x_r`
towards ``+z_r``.

* **Flow**: rotating the frame clockwise by ``p`` rotates the flow
  by ``p``. The air moves along ``+x_0``, i.e. ``phi = pitch``: at zero pitch
  the flow runs along ``+x_r``, at ``p = 90 deg`` along ``+z_r``.
* **Chord**: a section's chord line (leading to trailing edge) points along
  ``theta_c = twist(y)`` from the root ``+x_r`` axis.
* **Angle of attack** ``alpha(y) = phi - theta_c(y) = pitch - twist(y)``,
  wrapped to ``(-180, 180]`` deg: the angle from the chord to
  the flow. With the chord along ``+x_c`` (leading to trailing edge) a flow at
  ``alpha = 0`` runs from the leading to the trailing edge, ``alpha =
  +/-180`` deg is a reversed flow, ``+/-90`` deg a flow normal to the chord.

Because a force component *along the local flow* and one *normal to it* is
a projection, the rotation that takes ``(F_x, F_z)`` there is
``theta_c(y) + alpha(y) = phi``: the twist cancels, and the rotation is the
same at every station. Twist only enters ``alpha(y)`` (and thereby the
projected width), not the force rotation.

**Inclination** ``i`` is the angle by which the blade axis is turned out of
the plane normal to the wind. It is taken to leave the *direction* of the
in-plane flow unchanged: the wind splits into a speed ``U cos(i)`` normal to
the blade axis (the flow that drives the sectional forces) and ``U sin(i)``
along it.

Open choices that this file fixes, and where to change them: the air moves
towards ``+x_0`` (``flow_angle_at_zero_pitch_deg = 0``), and the chord line
from leading to trailing edge points along root ``+x_r`` at zero twist
(``chord_angle_at_zero_twist_deg = 0``). Both only shift ``alpha`` / flip
signs; they do not change ``|sin alpha|`` or the drag / lift axes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle):
    """Wrap an angle [rad] to ``(-pi, pi]``."""
    return np.pi - np.mod(np.pi - np.asarray(angle, float), 2 * np.pi)


@dataclass(frozen=True)
class InflowConvention:
    """Angle conventions (see the module docstring).

    Attributes
    ----------
    flow_angle_at_zero_pitch_deg : float
        Direction of the free stream in the ``(x0, z0)`` plane at zero pitch
        . 0 (default): air moves along ``+x0`` (= ``+x0``).
    pitch_sign : float
        ``+1`` or ``-1``: sign with which the pitch angle adds to the flow
        angle. +1 (default): the root frame turns clockwise with the pitch.
    twist_sign : float
        ``+1`` or ``-1``: sign with which the windIO twist adds to the
        chord angle. -1 (default): counter-clockwise.
    chord_angle_at_zero_twist_deg : float
        Direction of the chord line (leading to trailing edge) at zero
        twist in the blade root coordinate system [deg from ``+xr`` towards
        ``+zr``]. 0 (default): along ``+xr``.
    """
    flow_angle_at_zero_pitch_deg: float = 0.0
    pitch_sign: float = 1.0
    twist_sign: float = -1.0
    chord_angle_at_zero_twist_deg: float = 0.0

    def __post_init__(self):
        if self.pitch_sign not in (-1, 1) or self.twist_sign not in (-1, 1):
            raise ValueError("pitch_sign and twist_sign must be +1 or -1.")

    def flow_angle(self, pitch_deg: float) -> float:
        """Calculate root direction in the free stream plane [rad].

            Parameters
            ----------
            pitch_deg : _type_
                Pitch angle in degrees (= Angle from the inflow coordinate
                system to the root coordinate system.
                See :attr:`InflowConvention.pitch_sign` for definition of
                the direction)

            Returns
            -------
            np.ndarray
                Angle from the inflow coordinate system to the the root
                coordinate system system in radians (positive values consitute
                clockwise rotation)
        """
        return float(wrap_angle(np.deg2rad(
            self.flow_angle_at_zero_pitch_deg
            + self.pitch_sign * pitch_deg)))

    def chord_angle(self, twist_rad):
        """Calculate chord direction in the root ``(x, z)`` plane [rad].

        Parameters
        ----------
        twist_rad : _type_
            Twist angle in radians (= Angle from the root coordinate system
            to the chord coordinate system.
            See :attr:`InflowConvention.twist_sign` for definition of the
            direction)

        Returns
        -------
        np.ndarray
            Angle from the root coordinate system to the chord coordinate
            system in radians (positive values consitute clockwise rotation)
        """
        return (np.deg2rad(self.chord_angle_at_zero_twist_deg)
                + self.twist_sign * np.asarray(twist_rad, float))

    def angle_of_attack(self, pitch_deg: float, twist_rad):
        """Calculate angle from the flow to the chord [rad], in ``(-pi, pi]``.

            Parameters
            ----------
            pitch_deg : _type_
                Pitch angle in degrees (= Angle from the inflow coordinate
                system to the root coordinate system.
                See :attr:`InflowConvention.pitch_sign` for definition of
                the direction)
            twist_rad : _type_
                Twist angle in radians (= Angle from the root coordinate system
                to the chord coordinate system.
                See :attr:`InflowConvention.twist_sign` for definition of the
                direction)

            Returns
            -------
            np.ndarray
                Angle from the inflow coordinate system to the the chord
                coordinate system in radians (positive values consitute
                clockwise rotation)
        """
        """"""
        return wrap_angle(self.flow_angle(pitch_deg)
                          + self.chord_angle(twist_rad))


def normal_speed_factor(inclination_deg: float) -> float:
    """``cos(i)``: share of the wind speed normal to the blade axis."""
    return float(np.cos(np.deg2rad(inclination_deg)))


def spanwise_speed_factor(inclination_deg: float) -> float:
    """``sin(i)``: share of the wind speed along the blade axis."""
    return float(np.sin(np.deg2rad(inclination_deg)))
