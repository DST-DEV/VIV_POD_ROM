"""Configuration of a spectral analysis run."""
from __future__ import annotations

from dataclasses import dataclass

from viv_rom.config import CaseConfig

FIELD_OPTIONS = ("fx", "fz", "both")
SWEEP_FIELD_OPTIONS = ("fx", "fz")


@dataclass
class SpectralConfig(CaseConfig):
    """Case selection (see :class:`CaseConfig`) plus spectral options.

    Attributes
    ----------
    field : {"fx", "fz", "both"}
        Sectional-force component(s) analysed station-by-station.
    stations : tuple of int
        Spanwise station indices for the PSD plots / summary table
        (0 = root; the defaults assume 100 sections).
    n_harmonics : int
        Number of integer harmonics of ``f_forcing`` to mark.
    nperseg : int or None
        Welch segment length. ``None`` uses the full record (single windowed
        periodogram), which keeps the frequency resolution needed to tell
        nearby frequencies apart on these short records.
    spectrogram_nperseg : int or None
        STFT segment length. ``None`` -> ``max(64, n_steps // 12)``.
    do_spanwise_map, do_spanwise_amplitude_map, do_spanwise_phase_map : bool
        Frequency-vs-span heatmaps of log10(PSD), peak amplitude and phase.
    do_spectrogram : bool
        Time-frequency spectrogram of the span-integrated force.
    do_frequency_sweep : bool
        Repeat the dominant-frequency analysis over ``u_sweep_list``.
    phase_mask_threshold_db : float or None
        Mask phase where the magnitude is more than this many dB below the
        panel's peak (phase is meaningless without spectral content).
        ``None`` disables masking.
    u_sweep_list : tuple of int
        Wind speeds [m/s] included in the sweep (missing files are skipped).
    sweep_field : {"fx", "fz"}
        Component used for the frequency sweep.
    """
    field: str = "both"
    stations: tuple[int, ...] = (0, 25, 50, 75, 99)
    n_harmonics: int = 6
    nperseg: int | None = None
    spectrogram_nperseg: int | None = None
    do_spanwise_map: bool = True
    do_spanwise_amplitude_map: bool = True
    do_spanwise_phase_map: bool = True
    do_spectrogram: bool = True
    do_frequency_sweep: bool = True
    phase_mask_threshold_db: float | None = -30.0
    u_sweep_list: tuple[int, ...] = tuple(range(6, 23))
    sweep_field: str = "fx"

    def __post_init__(self):
        super().__post_init__()
        if self.field not in FIELD_OPTIONS:
            raise ValueError(f"Unknown field option: {self.field!r} "
                             f"(use one of {FIELD_OPTIONS}).")
        if self.sweep_field not in SWEEP_FIELD_OPTIONS:
            raise ValueError(f"Unknown sweep_field: {self.sweep_field!r} "
                             f"(use one of {SWEEP_FIELD_OPTIONS}).")
        self.stations = tuple(self.stations)
        self.u_sweep_list = tuple(self.u_sweep_list)

    @property
    def default_output_dir(self):
        return self.output_dir_for("spectral")
