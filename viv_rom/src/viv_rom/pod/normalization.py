"""Optional normalisation strategies applied to the force field before POD.

A normaliser answers one question: *by what factor should each force
component at each spanwise section be multiplied?* It returns that as
:class:`ComponentScales` and never touches the data itself. Several
normalisers can be chained; their factors multiply. Applying/inverting the
combined factors is the job of :class:`FieldScaling`, so strategies stay tiny
and independent of the snapshot-matrix layout.

Built-in strategies
-------------------
``"dynamic_pressure"`` -- :class:`DynamicPressureNormalizer`
    ``F / q`` with ``q = 0.5 * rho * U**2``: makes cases at different wind
    speeds comparable (the quantity becomes a length, i.e. a force
    coefficient times chord).
``"work"`` -- :class:`WorkWeightNormalizer`
    ``F * v_amp(y)`` with ``v_amp = phi(y) * A * omega`` the amplitude of the
    prescribed velocity. The weighted field is the (signed) sectional power
    amplitude [W/m], so the POD energy measures how strongly each section
    takes part in the energy injection instead of how large its force is.

Adding a strategy (Open/Closed)
-------------------------------
Implement :class:`Normalizer` and register it; nothing else changes::

    @register_normalizer("chord")
    class ChordNormalizer:
        name = "chord"
        def scales(self, ctx):
            return ComponentScales.uniform(1 / my_chord(ctx.y))

It is then available as ``PODConfig(normalization=("chord",))`` and on the
command line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from viv_rom.domain.motion import ForcedMotion
from viv_rom.pod.core import FieldInfo

RHO_AIR = 1.225  # [kg/m^3]


# ---------------------------------------------------------------------------
# Data passed to / returned from strategies
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NormalizationContext:
    """Everything a strategy may need to know about the case.

    Attributes
    ----------
    y : ndarray, shape (n_sections,)
        Spanwise section positions [m].
    wind_speed : float
        Free-stream wind speed ``U`` [m/s].
    motion : ForcedMotion
        Prescribed motion shapes.
    amplitude : float
        Motion amplitude factor ``A``.
    omega : float
        Angular excitation frequency [rad/s].
    air_density : float
        ``rho`` [kg/m^3].
    """
    y: np.ndarray
    wind_speed: float
    motion: ForcedMotion
    amplitude: float
    omega: float
    air_density: float = RHO_AIR


@dataclass(frozen=True, eq=False)
class ComponentScales:
    """Multiplicative factors for the Fx and Fz components per section.

    Both arrays have shape ``(n_sections,)``.
    """
    fx: np.ndarray
    fz: np.ndarray

    @classmethod
    def uniform(cls, value, n_sections: int | None = None
                ) -> "ComponentScales":
        """Same factor(s) for Fx and Fz.

        ``value`` is a scalar (then ``n_sections`` is required) or an
        array of shape ``(n_sections,)``.
        """
        v = np.asarray(value, dtype=float)
        if v.ndim == 0:
            if n_sections is None:
                raise ValueError("n_sections is required for a scalar.")
            v = np.full(n_sections, float(v))
        return cls(fx=v, fz=v.copy())

    def __mul__(self, other: "ComponentScales") -> "ComponentScales":
        return ComponentScales(self.fx * other.fx, self.fz * other.fz)


@runtime_checkable
class Normalizer(Protocol):
    """Strategy interface: how to scale the force components."""

    name: str

    def scales(self, ctx: NormalizationContext) -> ComponentScales:
        """Factors ``s`` such that ``normalised = s * F``."""
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[[], Normalizer]] = {}


def register_normalizer(name: str):
    """Class decorator registering a no-argument factory under ``name``."""
    def decorator(factory):
        if name in _REGISTRY:
            raise ValueError(f"Normalizer {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_normalizers() -> tuple[str, ...]:
    """Names accepted by :func:`make_normalizer` / ``PODConfig``."""
    return tuple(_REGISTRY)


def make_normalizer(name: str) -> Normalizer:
    """Instantiate a registered normaliser with its default options."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown normalization {name!r} "
                         f"(use one of {available_normalizers()}).") from None


@register_normalizer("dynamic_pressure")
@dataclass(frozen=True)
class DynamicPressureNormalizer:
    """Divide by the free-stream dynamic pressure ``q = 0.5 * rho * U**2``."""
    name: str = "dynamic_pressure"

    def scales(self, ctx: NormalizationContext) -> ComponentScales:
        if ctx.wind_speed <= 0:
            raise ValueError("Dynamic-pressure scaling needs a positive "
                             f"wind speed, got {ctx.wind_speed}.")
        q = 0.5 * ctx.air_density * ctx.wind_speed**2
        return ComponentScales.uniform(1.0 / q, len(ctx.y))


@register_normalizer("work")
@dataclass(frozen=True)
class WorkWeightNormalizer:
    """Weight each section by the prescribed velocity amplitude.

    With ``v_i(y, t) = phi_i(y) * A * omega * cos(omega t)`` the sectional
    power is ``p = Fx*vx + Fz*vz``, so ``F_i * phi_i * A * omega`` is the
    power amplitude carried by component ``i``. The sign of ``phi`` is kept,
    which makes the weighting exactly invertible and the sum of the weighted
    components proportional to the power.

    Parameters
    ----------
    floor : float
        Smallest allowed ``|weight|`` as a fraction of the largest one.
        Near the root the mode shape (and hence the weight) tends to zero,
        which would make un-weighting (reconstruction in N/m) blow up
        round-off and truncation error. The floor bounds that; set it to 0
        to disable (an exactly zero weight is then rejected).
    """
    floor: float = 1e-3
    name: str = "work"

    def scales(self, ctx: NormalizationContext) -> ComponentScales:
        w = {}
        for comp in ("Fx", "Fz"):
            w[comp] = ctx.motion.velocity_amplitude(
                comp, ctx.y, amplitude=ctx.amplitude, omega=ctx.omega)
        w_max = max(np.max(np.abs(w["Fx"])), np.max(np.abs(w["Fz"])))
        if w_max == 0:
            raise ValueError("The prescribed motion is zero; work weighting "
                             "is undefined.")
        w_min = self.floor * w_max
        for comp, vec in w.items():
            sign = np.where(vec < 0, -1.0, 1.0)
            w[comp] = sign * np.maximum(np.abs(vec), w_min)
        return ComponentScales(fx=w["Fx"], fz=w["Fz"])


# ---------------------------------------------------------------------------
# Combining strategies and applying them to a snapshot matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FieldScaling:
    """Per-point multiplicative scaling of a snapshot matrix.

    Attributes
    ----------
    factors : ndarray, shape (n_points,)
        ``normalised = factors[:, None] * field``. All finite and non-zero.
    names : tuple of str
        Names of the strategies that produced the factors (for reports).
    """
    factors: np.ndarray
    names: tuple[str, ...] = ()

    def __post_init__(self):
        f = np.asarray(self.factors, dtype=float)
        if f.ndim != 1 or not np.all(np.isfinite(f)) or np.any(f == 0):
            raise ValueError("Scaling factors must be a 1-D array of finite, "
                             "non-zero values.")
        object.__setattr__(self, "factors", f)

    @classmethod
    def identity(cls, n_points: int) -> "FieldScaling":
        return cls(np.ones(n_points), ())

    @classmethod
    def from_normalizers(cls, normalizers: Sequence[Normalizer],
                         ctx: NormalizationContext, field_info: FieldInfo
                         ) -> "FieldScaling":
        """Combine the strategies' factors for the given field layout."""
        scales = ComponentScales.uniform(1.0, len(ctx.y))
        for normalizer in normalizers:
            scales = scales * normalizer.scales(ctx)
        return cls(field_info.layout(scales.fx, scales.fz),
                   tuple(n.name for n in normalizers))

    @property
    def is_identity(self) -> bool:
        return not self.names

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Physical -> normalised. ``field`` has shape (n_points, ...)."""
        return self._expand(field) * field

    def invert(self, field: np.ndarray) -> np.ndarray:
        """Normalised -> physical."""
        return field / self._expand(field)

    def _expand(self, field: np.ndarray) -> np.ndarray:
        if field.shape[0] != len(self.factors):
            raise ValueError(f"Field has {field.shape[0]} points but the "
                             f"scaling has {len(self.factors)}.")
        return self.factors.reshape((-1,) + (1,) * (field.ndim - 1))
