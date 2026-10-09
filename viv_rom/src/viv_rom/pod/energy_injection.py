"""Energy injection of the POD modes into the forced blade motion.

The sectional power transferred from the flow to the prescribed motion is
``p(y, t) = Fx*vx + Fz*vz`` and the energy injected per unit span over the
analysed window is ``E(y) = integral p(y, t) dt`` [J/m].

Both ``p`` and the integral are linear in the force field, and so is the
POD (and the optional normalisation, once undone). The energy of any
truncated reconstruction therefore decomposes exactly::

    E_rec(y; r) = E_mean(y) + sum_{k=1..r} E_k(y)

with ``E_k`` the energy injection of mode ``k`` alone and ``E_mean`` that of
the temporal-mean field. :func:`energy_injection_analysis` computes these
terms once and derives, for every truncation ``r``, how well the spanwise
distribution ``E(y)`` is reproduced:

    station error   e_j(r) = |E_rec_j(r) - E_j| / ref_j
    one number      m(r)   = aggregate_j( e_j(r) )     (mean, rms, max, ...)

The number of modes required is the smallest ``r`` with ``m(r) <= tol``
(``tol = 0.01`` for "99 % accuracy"). ``ref_j`` is the peak ``|E|`` over the
span (``reference="peak"``, robust where ``E_j`` crosses zero) or the local
``|E_j|`` (``reference="local"``).

Extension points: the *aggregator* (any callable
``station_errors -> float``) and the *reference* (any callable
``E -> array broadcastable to E``) are plain strategies passed in by name or
as callables; the model of the field only has to satisfy :class:`FieldModel`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from viv_rom.pod.core import FieldInfo
from viv_rom.pod.power import sectional_power

StationErrorAggregator = Callable[[np.ndarray], float]
ErrorReference = Callable[[np.ndarray], np.ndarray]

AGGREGATORS: Mapping[str, StationErrorAggregator] = {
    "mean": lambda e: float(np.mean(e)),
    "rms": lambda e: float(np.sqrt(np.mean(np.square(e)))),
    "max": lambda e: float(np.max(e)),
}

REFERENCES: Mapping[str, ErrorReference] = {
    "peak": lambda E: np.max(np.abs(E)),
    "local": lambda E: np.abs(E),
}


class FieldModel(Protocol):
    """What the analysis needs to know about a decomposed field.

    Satisfied by :class:`viv_rom.pod.preprocessing.PhysicalReconstructor`.
    All fields are in physical units (N/m).
    """
    n_modes: int

    def mean(self) -> np.ndarray:
        """Mean field, shape ``(n_points, 1)``."""

    def mode(self, k: int) -> np.ndarray:
        """Mode ``k`` (0-based) alone, shape ``(n_points, n_t)``."""


def resolve_aggregators(spec: Sequence[str | StationErrorAggregator]
                        ) -> dict[str, StationErrorAggregator]:
    """Map names (``AGGREGATORS``) or callables to ``{label: callable}``."""
    out = {}
    for item in spec:
        if isinstance(item, str):
            if item not in AGGREGATORS:
                raise ValueError(f"Unknown aggregator {item!r} "
                                 f"(use one of {tuple(AGGREGATORS)}).")
            out[item] = AGGREGATORS[item]
        else:
            out[getattr(item, "__name__", repr(item))] = item
    if not out:
        raise ValueError("At least one error aggregator is required.")
    return out


def resolve_reference(spec: str | ErrorReference) -> ErrorReference:
    if callable(spec):
        return spec
    if spec not in REFERENCES:
        raise ValueError(f"Unknown error reference {spec!r} "
                         f"(use one of {tuple(REFERENCES)}).")
    return REFERENCES[spec]


@dataclass(frozen=True, eq=False)
class EnergyInjection:
    """Result of :func:`energy_injection_analysis`.

    Attributes
    ----------
    y : ndarray, shape (n_sections,)
    original : ndarray, shape (n_sections,)
        ``E(y)`` of the raw field [J/m].
    mean_part : ndarray, shape (n_sections,)
        ``E_mean(y)``: contribution of the temporal-mean field [J/m].
    modes : ndarray, shape (n_modes, n_sections)
        ``E_k(y)``: contribution of each mode alone [J/m].
    errors : ndarray, shape (n_modes + 1, n_sections)
        Station errors ``e_j(r)``; row ``r`` uses the mean plus ``r`` modes
        (row 0: mean only).
    curves : dict of {str: ndarray, shape (n_modes + 1,)}
        Aggregated error ``m(r)`` per aggregator.
    n_required : dict of {str: int or None}
        Smallest ``r`` with ``m(r) <= tolerance`` (``None`` if never met).
    tolerance : float
    """
    y: np.ndarray
    original: np.ndarray
    mean_part: np.ndarray
    modes: np.ndarray
    errors: np.ndarray
    curves: dict[str, np.ndarray]
    n_required: dict[str, int | None]
    tolerance: float

    @property
    def n_modes(self) -> int:
        return self.modes.shape[0]

    @property
    def total_original(self) -> float:
        """Span-integrated energy injection of the raw field [J]."""
        return float(np.trapezoid(self.original, self.y))

    @property
    def mode_totals(self) -> np.ndarray:
        """Span-integrated energy injection of each mode [J]."""
        return np.trapezoid(self.modes, self.y, axis=1)

    @property
    def mode_fractions(self) -> np.ndarray:
        """``mode_totals / total_original`` (negative: damping mode)."""
        return self.mode_totals / self.total_original

    @property
    def mean_total(self) -> float:
        return float(np.trapezoid(self.mean_part, self.y))

    def reconstructed(self, r: int) -> np.ndarray:
        """``E_rec(y; r)``: energy injection of mean + ``r`` leading modes."""
        return self.mean_part + self.modes[:r].sum(axis=0)

    def summary_lines(self, n_show: int = 4) -> list[str]:
        lines = ["--- Energy injection ---",
                 f"Total injected energy (raw field): "
                 f"{self.total_original:.4g} J "
                 f"(mean-field part: {self.mean_total:.4g} J)",
                 f"{'mode':>4} {'E_k [J]':>12} {'share of total':>15}"]
        totals, shares = self.mode_totals, self.mode_fractions
        for k in range(min(n_show, self.n_modes)):
            lines.append(f"{k+1:4d} {totals[k]:12.4g} "
                         f"{100 * shares[k]:14.2f}%")
        lines.append(f"Modes needed to reproduce E(y) within "
                     f"{100 * self.tolerance:g} % "
                     f"(mean + r modes):")
        for name, n in self.n_required.items():
            got = ("not reached with the modes analysed" if n is None
                   else f"{n}  (error {100 * self.curves[name][n]:.3g} %)")
            lines.append(f"  {name:>5}-error: {got}")
        return lines


def energy_injection_analysis(
        model: FieldModel, field: np.ndarray, field_info: FieldInfo,
        y: np.ndarray, t: np.ndarray, vx: np.ndarray, vz: np.ndarray,
        tolerance: float = 0.01,
        aggregators: Sequence[str | StationErrorAggregator] = (
            "mean", "rms", "max"),
        reference: str | ErrorReference = "peak",
        max_modes: int | None = None) -> EnergyInjection:
    """Energy injection per mode and modes needed to reproduce ``E(y)``.

    Parameters
    ----------
    model : FieldModel
        Decomposed field in physical units.
    field : ndarray, shape (n_points, n_t)
        The raw snapshot matrix that was decomposed (the reference).
    field_info : FieldInfo
        Layout of ``field``.
    y, t : ndarray
        Section positions and time vector.
    vx, vz : ndarray, shape (n_sections, n_t)
        Prescribed velocities (e.g. ``ForcedMotion.velocity``).
    tolerance : float
        Accepted aggregated error; ``0.01`` means 99 % accuracy.
    aggregators : sequence of str or callable
        How the per-station errors are combined (names in ``AGGREGATORS``).
    reference : str or callable
        Per-station error scale (``"peak"`` or ``"local"``).
    max_modes : int, optional
        Analyse only the leading modes (default: all).
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    aggs = resolve_aggregators(aggregators)
    ref_fn = resolve_reference(reference)
    n_modes = model.n_modes if max_modes is None else min(max_modes,
                                                          model.n_modes)

    def energy(f: np.ndarray) -> np.ndarray:
        return np.trapezoid(sectional_power(f, field_info, vx, vz), t, axis=1)

    original = energy(field)
    mean_part = energy(model.mean())
    modes = np.array([energy(model.mode(k)) for k in range(n_modes)])

    cumulative = np.vstack([np.zeros_like(original),
                            np.cumsum(modes, axis=0)])
    errors = np.abs(mean_part + cumulative - original) / ref_fn(original)

    curves = {name: np.array([agg(e) for e in errors])
              for name, agg in aggs.items()}
    n_required = {}
    for name, curve in curves.items():
        ok = np.flatnonzero(curve <= tolerance)
        n_required[name] = int(ok[0]) if ok.size else None

    return EnergyInjection(y=np.asarray(y), original=original,
                           mean_part=mean_part, modes=modes, errors=errors,
                           curves=curves, n_required=n_required,
                           tolerance=tolerance)
