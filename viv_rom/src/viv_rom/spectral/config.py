"""Configuration of a spectral analysis run."""
from __future__ import annotations

from dataclasses import dataclass

from viv_rom.config import CaseConfig
from viv_rom.domain.frames import (ForceFrame,
                                   available_frames, make_frame)
from viv_rom.domain.motion import FIRST_EDGEWISE_MODE
from viv_rom.spectral.shedding import (LOCK_IN_BAND, STROUHAL_RANGE,
                                       validate_range)


def available_fields() -> tuple[str, ...]:
    """Component keys of every registered frame, then ``"both"``."""
    keys: list[str] = []
    for name in available_frames():
        for key in make_frame(name, FIRST_EDGEWISE_MODE).labels:
            if key not in keys:
                keys.append(key)
    return (*keys, "both")


@dataclass
class SpectralConfig(CaseConfig):
    """Case selection (see :class:`CaseConfig`) plus spectral options.

    Attributes
    ----------
    frame : str or ForceFrame
        Coordinate frame the forces are rotated into before every analysis:
        ``"blade"`` (the CFD ``Fx``/``Fz``), ``"motion"`` (default; along /
        normal to the local direction of motion, ``F_par``/``F_perp``),
        ``"inflow"`` (along / normal to the free stream, ``F_D``/``F_L``;
        uses the case's pitch and inclination and ``blade_file``), or a
        :class:`~viv_rom.domain.frames.ForceFrame` instance.
    field : str
        Component(s) analysed station-by-station: ``"both"`` or one of the
        frame's keys (``"fx"``/``"fz"`` in the blade frame,
        ``"fpar"``/``"fperp"`` in the motion frame, ``"fdrag"``/``"flift"``
        in the inflow frame).
    stations : tuple of int
        Spanwise station indices for the PSD plots / summary table as a
        percentage of the number of stations (0 = root, 100 = tip).
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
    unwrap_phase : bool
        If False (default), phases are wrapped to ``(-pi, pi]``.
        If True, return the continuous (unwrapped) phase.
    u_sweep_list : tuple of int
        Wind speeds [m/s] included in the sweep (missing files are skipped).
    sweep_field : str or None
        Component (a key of the frame) used for the frequency sweep.
        ``None`` uses the frame's first component (``"fx"`` / ``"fpar"``).
    do_forcing_response : bool
        Amplitude and phase of the force at the forcing frequency (and
        harmonics) along the span.
    response_harmonics : int
        Harmonics ``1..n`` of the forcing frequency fitted for it.
    do_cycle_energy : bool
        Energy injected per forcing cycle and its cycle-to-cycle spread.
    do_shedding : bool
        Estimate the vortex-shedding frequency band along the span from the
        Strouhal range and compare it with the forcing frequency (needs the
        blade geometry, ``blade_file``; skipped with a warning if missing).
    strouhal_range : (float, float)
        ``(St_low, St_high)`` used for the shedding estimate; the default
        0.11-0.16 is that of Pellegrino & Meskell (2013, NREL S809, 2D).
    lock_in_band : (float, float)
        Lock-in range as multiples of ``f_forcing`` (default 0.8-1.1).
    """
    frame: str | ForceFrame = "motion"
    field: str = "both"
    stations: tuple[int, ...] = (0, 25, 50, 75, 100)
    n_harmonics: int = 6
    nperseg: int | None = None
    spectrogram_nperseg: int | None = None
    do_spanwise_map: bool = True
    do_spanwise_amplitude_map: bool = True
    do_spanwise_phase_map: bool = True
    do_spectrogram: bool = True
    do_frequency_sweep: bool = True
    phase_mask_threshold_db: float | None = -30.0
    unwrap_phase: bool = False
    u_sweep_list: tuple[int, ...] = tuple(range(6, 23))
    sweep_field: str | None = None
    do_forcing_response: bool = True
    response_harmonics: int = 3
    do_cycle_energy: bool = True
    do_shedding: bool = True
    strouhal_range: tuple[float, float] = STROUHAL_RANGE
    lock_in_band: tuple[float, float] = LOCK_IN_BAND

    def __post_init__(self):
        super().__post_init__()
        keys = tuple(self.frame_instance().labels)
        if self.field != "both" and self.field not in keys:
            raise ValueError(f"Unknown field option: {self.field!r} for the "
                             f"{self.frame_instance().name!r} frame "
                             f"(use one of {(*keys, 'both')}).")
        if self.sweep_field is not None and self.sweep_field not in keys:
            raise ValueError(f"Unknown sweep_field: {self.sweep_field!r} for "
                             f"the {self.frame_instance().name!r} frame "
                             f"(use one of {keys}).")
        if self.response_harmonics < 1:
            raise ValueError("response_harmonics must be >= 1.")
        self.strouhal_range = validate_range("strouhal_range",
                                             self.strouhal_range)
        self.lock_in_band = validate_range("lock_in_band", self.lock_in_band)
        self.stations = tuple(self.stations)
        self.u_sweep_list = tuple(self.u_sweep_list)

    def frame_instance(self) -> ForceFrame:
        """The configured force frame as an object."""
        if isinstance(self.frame, str):
            return make_frame(self.frame, self.motion, pitch=self.pitch,
                              inclination=self.inclination,
                              blade_file=self.blade_file)
        return self.frame

    @property
    def effective_sweep_field(self) -> str:
        return (self.sweep_field if self.sweep_field is not None
                else next(iter(self.frame_instance().labels)))

    @property
    def default_output_dir(self):
        return self.output_dir_for("spectral")
