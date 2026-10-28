"""Configuration of a global POD run (one varying simulation parameter)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from viv_rom.pod.config import PODConfig

SWEEP_PARAMETERS = ("wind_speed", "pitch", "inclination")
CENTERING_OPTIONS = ("case", "global", "none")
WEIGHTING_OPTIONS = ("time_average", "snapshots", "unit_energy")


@dataclass
class GlobalPODConfig:
    """Which simulations to pool and how.

    The global snapshot matrix is the time-concatenation of the (scaled and
    centred) snapshot matrices of one simulation per value of ``parameter``,
    with every other parameter taken from ``base``.

    Attributes
    ----------
    base : PODConfig
        Template for every simulation: case code (amplitude, pitch,
        inclination), data folder, trimming, field selection, normalisation
        (applied *per case*, with that case's wind speed), energy and phase
        options. ``base.<parameter>`` is replaced by each of ``values``.
    parameter : {"wind_speed", "pitch", "inclination"}
        The varying parameter.
    values : sequence of int
        Values of ``parameter`` to pool (the files must exist; see
        ``skip_missing``).
    centering : {"case", "global", "none"}
        What is removed before the SVD.

        ``"case"`` (default): every simulation's own temporal mean. The modes
        describe fluctuations only. The mean field has to be modelled
        separately.
        ``"global"``: one common mean (average of the case means). The
        case-to-case differences of the mean stay in the snapshots and
        occupy modes.
        ``"none"``: nothing is removed.
    weighting : {"time_average", "snapshots", "unit_energy"}
        How much each simulation counts in the pooled covariance
        ``sum_c w_c D_c D_c^T``.

        ``"time_average"`` (default): ``w_c = 1/n_c``, so every simulation
        contributes its time-averaged covariance regardless of record
        length.
        ``"snapshots"``: ``w_c = 1``. Plain concatenation. Longer records
        weigh more.
        ``"unit_energy"``: ``w_c = 1/||D_c||^2``: Every simulation has the
        same total fluctuation energy (removes amplitude differences between
        cases that normalisation did not).

        The weights only select the basis. Projections and reconstructions
        of each case use its unweighted data.
    whole_periods : bool
        Crop every simulation to an integer number of forcing periods (the
        oldest samples of the retained window are dropped), so the mean and
        the energy integrals are not biased by a partial cycle.
    n_periods : int or None
        If given, keep exactly the last ``n_periods`` forcing periods of every
        simulation (all cases then have the same length). A case with fewer
        periods raises an error.
    n_skip_overrides : dict of {int: int}
        Per-value replacement of ``base.n_skip`` (initial transient rows),
        e.g. a longer transient at low wind speed.
    skip_missing : bool
        Skip values without a data file (reported in the result). If False,
        a missing file raises ``FileNotFoundError``.
    n_modes_keep : int or None
        Keep only this many leading modes of the basis. If None all modes are
        kept. Defaults to all.
    compute_local : bool
        Also decompose every simulation on its own (local POD) as the
        reference for the global basis: modes needed, energy-injection
        accuracy and the angles between the local subspaces.
    angle_modes : int
        Subspace dimension ``k`` for the principal angles between the local
        bases of neighbouring simulations and to the global basis.
    energy_max_modes : int or None
        Limit the number of modes used in the per-case energy-injection
        analysis. If None all modes are kept. Defaults to all.
    """
    base: PODConfig = field(default_factory=PODConfig)
    parameter: str = "wind_speed"
    values: tuple[int, ...] = ()
    centering: str = "case"
    weighting: str = "time_average"
    whole_periods: bool = True
    n_periods: int | None = None
    n_skip_overrides: dict[int, int] = field(default_factory=dict)
    skip_missing: bool = True
    n_modes_keep: int | None = None
    compute_local: bool = True
    angle_modes: int = 4
    energy_max_modes: int | None = None

    def __post_init__(self):
        if self.parameter not in SWEEP_PARAMETERS:
            raise ValueError(f"Unknown sweep parameter {self.parameter!r} "
                             f"(use one of {SWEEP_PARAMETERS}).")
        if self.centering not in CENTERING_OPTIONS:
            raise ValueError(f"Unknown centering {self.centering!r} "
                             f"(use one of {CENTERING_OPTIONS}).")
        if self.weighting not in WEIGHTING_OPTIONS:
            raise ValueError(f"Unknown weighting {self.weighting!r} "
                             f"(use one of {WEIGHTING_OPTIONS}).")
        values = []
        for v in self.values:
            if int(v) != v:
                raise ValueError(f"{self.parameter} values must be integers "
                                 "(they are part of the file name), "
                                 f"got {v!r}.")
            values.append(int(v))
        self.values = tuple(sorted(set(values)))
        if len(self.values) < 2:
            raise ValueError("A global POD needs at least two distinct "
                             f"{self.parameter} values, got {self.values}.")
        if self.n_periods is not None and self.n_periods < 1:
            raise ValueError("n_periods must be >= 1.")
        if self.n_modes_keep is not None and self.n_modes_keep < 1:
            raise ValueError("n_modes_keep must be >= 1.")
        if self.angle_modes < 1:
            raise ValueError("angle_modes must be >= 1.")
        if self.energy_max_modes is not None and self.energy_max_modes < 1:
            raise ValueError("energy_max_modes must be >= 1.")
        self.n_skip_overrides = {int(k): int(v)
                                 for k, v in self.n_skip_overrides.items()}

    def case_config(self, value: int) -> PODConfig:
        """Config of the single simulation at ``parameter = value``."""
        changes = {self.parameter: int(value)}
        if int(value) in self.n_skip_overrides:
            changes["n_skip"] = self.n_skip_overrides[int(value)]
        return dataclasses.replace(self.base, **changes)

    @property
    def parameter_label(self) -> str:
        """Axis label of the varying parameter."""
        return {"wind_speed": "Wind speed [m/s]", "pitch": "Pitch angle [deg]",
                "inclination": "Inclination angle [deg]"}[self.parameter]

    @property
    def case_label(self) -> str:
        """Case code with the varying parameter masked, e.g. ``a01_p85_iXX``.
        """
        b = self.base
        p = "XX" if self.parameter == "pitch" else f"{b.pitch:02d}"
        i = "XX" if self.parameter == "inclination" else f"{b.inclination:02d}"
        return f"a{b.amplitude:02d}_p{p}_i{i}"

    @property
    def default_output_dir(self) -> Path:
        return Path("results") / self.case_label / f"gpod_{self.parameter}"
