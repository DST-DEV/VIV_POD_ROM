"""Inner products ("weights") that define the energy SPOD optimises.

SPOD maximises the energy ``x^H W x`` captured by each mode, so ``W`` decides
what "energy" means. It is a strategy: a small object returning the diagonal
of ``W`` for a given field layout. The default integrates over the span
(``sum_j w_j |x_j|^2 ~ integral |F|^2 dy``), which makes the result
independent of how the 100 sections happen to be distributed.

Adding a weighting (Open/Closed)::

    @register_inner_product("root_heavy")
    class RootHeavy:
        name = "root_heavy"
        def weights(self, y, field_info):
            return field_info.layout(w(y), w(y))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from viv_rom.pod.core import FieldInfo
from viv_rom.pod.energy_injection import trapezoid_weights


@runtime_checkable
class InnerProduct(Protocol):
    """Strategy interface: the diagonal weights of the energy norm."""

    name: str

    def weights(self, y: np.ndarray, field_info: FieldInfo) -> np.ndarray:
        """Positive weights, shape ``(n_points,)``."""
        ...


_REGISTRY: dict[str, Callable[[], InnerProduct]] = {}


def register_inner_product(name: str):
    """Class decorator registering a no-argument factory under ``name``."""
    def decorator(factory):
        if name in _REGISTRY:
            raise ValueError(f"Inner product {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory
    return decorator


def available_inner_products() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def make_inner_product(name: str) -> InnerProduct:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown weighting {name!r} "
                         f"(use one of {available_inner_products()}).") \
            from None


@register_inner_product("span")
@dataclass(frozen=True)
class SpanwiseIntegral:
    """Trapezoid quadrature weights: energy is ``integral |F|^2 dy``."""
    name: str = "span"

    def weights(self, y, field_info):
        q = trapezoid_weights(y)
        return field_info.layout(q, q)


@register_inner_product("identity")
@dataclass(frozen=True)
class Identity:
    """Plain sum over the sections (the weighting of the POD module)."""
    name: str = "identity"

    def weights(self, y, field_info):
        ones = np.ones(len(y))
        return field_info.layout(ones, ones)
