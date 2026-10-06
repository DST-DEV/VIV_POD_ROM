"""Prescribed (forced) blade motion used in the CFD simulations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ForcedMotion:
    """Polynomial description of the forced-motion mode shapes.

    The polynomials (highest power first, as for :func:`numpy.polyval`) are
    evaluated at ``y - y_offset``.
    """
    coeff_x: np.ndarray
    coeff_z: np.ndarray
    y_offset: float = 2.8

    def modeshape(self, component: str, y: np.ndarray,
                  normalize: bool = True) -> np.ndarray:
        """Mode shape of the forced motion along the span.

        Parameters
        ----------
        component : {"Fx", "Fz"}
            Edgewise (``"Fx"``/``"x"``) or flapwise (``"Fz"``/``"z"``)
            direction. Case-insensitive.
        y : ndarray
            Spanwise coordinates [m].
        normalize : bool, optional
            Divide by the maximum of the mode shape. Default is True.
        """
        key = component.lower().lstrip("f")
        if key == "x":
            coeff = self.coeff_x
        elif key == "z":
            coeff = self.coeff_z
        else:
            raise ValueError(f"Unknown component {component!r} "
                             "(use 'Fx' or 'Fz').")
        shape = np.polyval(coeff, np.asarray(y) - self.y_offset)
        return shape / np.max(shape) if normalize else shape

    def velocity(self, component: str, y: np.ndarray, t: np.ndarray,
                 amplitude: float, omega: float) -> np.ndarray:
        """Velocity [m/s], shape (len(t), len(y)).

        ``v_i(y, t) = phi_i(y) * A * omega * cos(omega * t)``
        """
        phi = self.modeshape(component, y, normalize=False)
        return phi[:, None] * amplitude * omega * np.cos(omega * np.asarray(t))


# Motion polynomials from the simulation README (first edgewise mode).
FIRST_EDGEWISE_MODE = ForcedMotion(
    coeff_x=np.array([-1.63363204579525e-10, 3.98744338033108e-08,
                      -2.55298610905188e-06, 1.36855887568269e-04,
                      -8.05337508940684e-04, 9.35366136110291e-04]),
    coeff_z=np.array([-7.37789434610509e-14, 1.96005460184524e-11,
                      -1.94444451601558e-09, 9.84504021700715e-08,
                      -2.59892457459754e-06, 2.09320617905598e-05,
                      -5.07715323316064e-05, 1.81743886489348e-06]),
    y_offset=2.8,
)
