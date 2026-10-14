"""Configuration of an SPOD run."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from viv_rom.config import CaseConfig
from viv_rom.pod.core import FIELD_OPTIONS
from viv_rom.pod.energy_injection import (resolve_aggregators,
                                          resolve_reference)
from viv_rom.pod.normalization import (RHO_AIR, Normalizer, make_normalizer)
from viv_rom.spod.blocks import SegmentSpec
from viv_rom.spod.weights import make_inner_product


@dataclass
class SPODConfig(CaseConfig):
    """Case selection (see :class:`CaseConfig`) plus SPOD options.

    Attributes
    ----------
    field : {"fx", "fz", "combined"}
        Force component(s) decomposed.
    weighting : str
        Energy norm (see :mod:`viv_rom.spod.weights`): ``"span"`` (default,
        integral over the span) or ``"identity"``.
    normalization : sequence of (str or Normalizer)
        Optional normalisation before the decomposition, exactly as for the
        POD (``"dynamic_pressure"``, ``"work"``; default none).
    air_density : float
        ``rho`` [kg/m^3] for the dynamic pressure.
    segment_periods : float
        Length of one Welch block in forcing periods (default 2). With a
        whole number of samples per period the forcing frequency and its
        harmonics fall exactly on bins; the frequency resolution is
        ``f_forcing / segment_periods``.
    n_fft : int or None
        Block length in samples; overrides ``segment_periods``.
    overlap : float
        Fraction of overlap between blocks, ``0 <= overlap < 1``.
    window : str
        Taper applied to each block (:func:`scipy.signal.get_window`).
    n_harmonics : int
        Harmonics of the forcing frequency analysed (phase locking, modal
        content).
    n_modes_show : int
        Leading SPOD modes plotted / tabulated.
    band_bins : int
        Half-width (in bins) of the band summed around a harmonic when its
        share of the variance is reported (+-1 holds a Hann main lobe).
    energy_threshold : float
        Power fraction used for "modes needed at a harmonic".
    n_peaks : int
        Spectral peaks listed in the report.
    do_injection : bool
        Energy injection of the locked fundamental, per mode.
    energy_tolerance, energy_error_metrics, energy_error_reference
        Modes needed to reproduce the spanwise injection (as in POD).
    """
    field: str = "combined"
    weighting: str = "span"
    normalization: tuple[str | Normalizer, ...] = ()
    air_density: float = RHO_AIR
    segment_periods: float = 4.0
    n_fft: int | None = None
    overlap: float = 0.5
    window: str = "hann"
    n_harmonics: int = 3
    n_modes_show: int = 3
    band_bins: int = 1
    energy_threshold: float = 0.99
    n_peaks: int = 5
    do_injection: bool = True
    energy_tolerance: float = 0.01
    energy_error_metrics: tuple = ("mean", "rms", "max")
    energy_error_reference: str | Callable = "peak"

    def __post_init__(self):
        super().__post_init__()
        if self.field not in FIELD_OPTIONS:
            raise ValueError(f"Unknown field option: {self.field!r} "
                             f"(use one of {FIELD_OPTIONS}).")
        make_inner_product(self.weighting)             # fail early
        if isinstance(self.normalization, (str, Normalizer)):
            self.normalization = (self.normalization,)
        self.normalization = tuple(self.normalization)
        self.normalizers()
        if self.segment_periods <= 0:
            raise ValueError("segment_periods must be positive.")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")
        if self.n_harmonics < 1 or self.n_modes_show < 1 \
                or self.band_bins < 0 or self.n_peaks < 0:
            raise ValueError("n_harmonics and n_modes_show must be >= 1; "
                             "band_bins and n_peaks >= 0.")
        if not 0 < self.energy_threshold <= 1:
            raise ValueError("energy_threshold must be in (0, 1].")
        if self.energy_tolerance < 0:
            raise ValueError("energy_tolerance must be non-negative.")
        self.energy_error_metrics = tuple(self.energy_error_metrics)
        resolve_aggregators(self.energy_error_metrics)
        resolve_reference(self.energy_error_reference)

    def normalizers(self) -> tuple[Normalizer, ...]:
        """The configured normalisation strategies as instances."""
        return tuple(make_normalizer(n) if isinstance(n, str) else n
                     for n in self.normalization)

    def segment_spec(self, dt: float) -> SegmentSpec:
        """Block layout for a record sampled every ``dt`` seconds."""
        if self.n_fft is not None:
            return SegmentSpec(int(self.n_fft),
                               int(round(self.overlap * self.n_fft)),
                               self.window)
        return SegmentSpec.for_period(dt, self.f_forcing,
                                      self.segment_periods, self.overlap,
                                      self.window)

    @property
    def default_output_dir(self):
        return self.output_dir_for("spod")
