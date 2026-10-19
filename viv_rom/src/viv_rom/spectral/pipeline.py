"""End-to-end spectral analysis of one CFD case (plus optional U sweep).

Typical interactive use::

    from viv_rom.spectral import SpectralConfig, analyse
    cfg = SpectralConfig.from_case("a01_p85_i50", wind_speed=12,
                                   data_dir="data/archive_1D")
    result = analyse(cfg)
    result.station_peaks["fx"]        # dominant frequency per station [Hz]
    result.sweep.f_dom                # lock-in diagram data

The steps are also available separately: :func:`analyse` (numbers only, no
plotting), :func:`write_figures`, and :func:`run` (both, plus a summary).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viv_rom.domain.data import ForceFieldFile
from viv_rom.domain.frames import ForceFrame
from viv_rom.io.results.export import save_result
from viv_rom.loading import load_case
from viv_rom.spectral import plotting
from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.core import (compute_psd,
                                   dominant_peaks,
                                   integrate_over_span, select_components)
from viv_rom.spectral.cycle_energy import (CycleEnergy,
                                           cycle_energy_analysis)
from viv_rom.spectral.forcing_response import (ForcingResponse,
                                               forcing_frequency_response)
from viv_rom.spectral.sweep import SweepResult, frequency_sweep

logger = logging.getLogger(__name__)
EXPORT_FILENAME = "spectral_results.npz"


@dataclass(frozen=True, eq=False)
class SpectralAnalysisResult:
    """Everything computed by :func:`analyse`.

    Attributes
    ----------
    station_peaks : dict of {"fx"/"fz": ndarray}
        Dominant frequency [Hz] at each of ``config.stations``.
    frame : ForceFrame
        Frame the forces were rotated into before analysis.
    sweep : SweepResult or None
        Wind-speed sweep, if requested.
    forcing_response : ForcingResponse or None
        Amplitude/phase of the force at the forcing frequency.
    cycle_energy : CycleEnergy or None
        Per-cycle energy injection.
    """
    force_file: ForceFieldFile
    dt: float
    n_periods: float
    station_peaks: dict[str, np.ndarray]
    frame: ForceFrame
    sweep: SweepResult | None = None
    forcing_response: ForcingResponse | None = None
    cycle_energy: CycleEnergy | None = None

    def components(self, field: str = "both") -> dict[str, np.ndarray]:
        """Forces in the analysis frame, ``{key: (n_stations, n_steps)}``."""
        all_components = self.frame.transform(self.y, self.Fx, self.Fz)
        return select_components(field, all_components)

    @property
    def y(self) -> np.ndarray:
        return self.force_file.data.y

    @property
    def t(self) -> np.ndarray:
        return self.force_file.data.t

    @property
    def Fx(self) -> np.ndarray:
        return self.force_file.data.Fx

    @property
    def Fz(self) -> np.ndarray:
        return self.force_file.data.Fz


def analyse(config: SpectralConfig) -> SpectralAnalysisResult:
    """Load a case, find the dominant frequency per station, run the sweep."""
    logger.info("Loading case %s, u = %s m/s from %s ...", config.case,
                config.wind_speed, config.data_dir)
    force_file = load_case(config)
    y, t = force_file.data.y, force_file.data.t
    frame = config.frame_instance()
    components = select_components(
        config.field,
        frame.transform(y, force_file.data.Fx, force_file.data.Fz))

    n_sections = len(y)
    if config.stations and max(config.stations) >= n_sections:
        raise ValueError(
            f"Station index {max(config.stations)} out of range: the file "
            f"has {n_sections} sections (valid indices 0-{n_sections - 1}).")

    dt = t[1] - t[0]
    n_periods = (t[-1] - t[0]) / (1.0 / config.f_forcing)
    logger.info("  %d snapshots, dt = %.6g s, spanning ~%.1f forcing periods",
                len(t), dt, n_periods)
    logger.info("  %d spanwise stations, y = %.1f to %.1f m",
                n_sections, y[0], y[-1])

    station_peaks = {}
    for key, data in components.items():
        peaks = []
        for st in config.stations:
            freqs, psd = compute_psd(data[st, :], dt, config.nperseg)
            peaks.append(dominant_peaks(freqs, psd, n_peaks=1)[0][0])
        station_peaks[key] = np.array(peaks)

    sweep = frequency_sweep(config) if config.do_frequency_sweep else None

    forcing_response = None
    if config.do_forcing_response:
        forcing_response = forcing_frequency_response(
            components=components, y=y, t=t, f_forcing=config.f_forcing,
            n_harmonics=config.response_harmonics,
            unwrap_phase=config.unwrap_phase)

    cycle_energy = None
    if config.do_cycle_energy:
        amplitude = config.effective_motion_amplitude
        omega = 2 * np.pi * config.f_forcing
        cycle_energy = cycle_energy_analysis(
            force_file.data.Fx, force_file.data.Fz,
            vx=config.motion.velocity("x", y, t, amplitude, omega),
            vz=config.motion.velocity("z", y, t, amplitude, omega),
            y=y, t=t, f_forcing=config.f_forcing)

    return SpectralAnalysisResult(
        force_file=force_file, dt=dt, n_periods=n_periods,
        station_peaks=station_peaks, frame=frame, sweep=sweep,
        forcing_response=forcing_response, cycle_energy=cycle_energy)


SUMMARY_FILENAME = "spectral_summary.txt"


def spectral_summary_text(result: SpectralAnalysisResult,
                          config: SpectralConfig) -> str:
    """The text summary of an analysis as one string (no I/O).

    This is the single source for both the console output and
    ``spectral_summary.txt``.
    """
    lines = [f"--- Spectral summary: case {config.case}, "
             f"u = {config.wind_speed} m/s ---",
             f"frame = '{result.frame.name}', field = '{config.field}', "
             f"f_forcing = {config.f_forcing} Hz, "
             f"{len(result.t)} snapshots spanning ~{result.n_periods:.1f} "
             f"forcing periods"]
    lines += ["",
             "--- Dominant frequencies by station ---",
             f"{'component':>9} {'station':>9} {'y [m]':>8} "
             f"{'f_peak [Hz]':>12} {'f/f_forcing':>12}"]
    # for key, peaks in result.station_peaks.items():
    #     for st, f_peak in zip(config.stations, peaks):
    #         lines.append(f"{key:>9} {st:9d} {result.y[st]:8.1f} "
    #                      f"{f_peak:12.4f} {f_peak / config.f_forcing:12.3f}")

    if result.forcing_response is not None:
        lines += [""] + result.forcing_response.summary_lines(
            config.stations, result.frame.labels)
    if result.cycle_energy is not None:
        lines += [""] + result.cycle_energy.summary_lines(config.stations)

    if result.sweep is not None:
        lines += ["", f"--- Frequency sweep over U for case '{config.case}' "
                      f"(field = '{config.effective_sweep_field}') ---"]
        sweep = result.sweep
        if len(sweep.wind_speeds) == 0:
            lines.append("  No files found for the requested sweep. "
                         "Skipping sweep plot.")
        for U, f in zip(sweep.wind_speeds, sweep.f_dom):
            locked = abs(f - config.f_forcing) < 0.05 * config.f_forcing
            lines.append(
                f"  U = {U:5.1f} m/s : f_dom = {f:.4f} Hz "
                f"({f / config.f_forcing:.2f} x forcing)"
                + ("  <-- locked to forcing" if locked else ""))
    return "\n".join(lines)


def log_summary(result: SpectralAnalysisResult, config: SpectralConfig
                ) -> None:
    """Print the text summary to the console (via the logger)."""
    logger.info("\n%s", spectral_summary_text(result, config))


def write_summary(result: SpectralAnalysisResult, config: SpectralConfig,
                  output_dir: Path) -> Path:
    """Save the text summary to ``output_dir / spectral_summary.txt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_FILENAME
    path.write_text(spectral_summary_text(result, config) + "\n",
                    encoding="utf-8")
    return path


def write_figures(result: SpectralAnalysisResult, config: SpectralConfig,
                  output_dir: Path, dpi: int = 150) -> list[Path]:
    """Create the requested figures and save them to ``output_dir``.

    Returns the list of written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, t = result.y, result.t
    components = result.components(config.field)
    labels = result.frame.labels
    common = dict(f_forcing=config.f_forcing, n_harmonics=config.n_harmonics,
                  labels=labels)

    figures = {"01_station_psd": plotting.plot_station_psd(
        y, t, components, config.stations, nperseg=config.nperseg, **common)}

    if config.do_spanwise_map:
        figures["02_spanwise_psd_map"] = plotting.plot_spanwise_map(
            y, t, components, nperseg=config.nperseg, **common)
    if config.do_spanwise_amplitude_map:
        figures["03_spanwise_amplitude_map"] = (
            plotting.plot_spanwise_amplitude_map(
                y, t, components, nperseg=config.nperseg, **common))
    if config.do_spanwise_phase_map:
        figures["03_spanwise_phase_map"] = (
            plotting.plot_spanwise_phase_map(
                y, t, components,
                mag_threshold_db=config.phase_mask_threshold_db, **common))

    if config.do_spectrogram:
        # The frame's first component is always shown; the second
        # additionally when both are analysed (behaviour of the original
        # script, where these were Fx and Fz).
        all_components = result.components("both")
        keys = list(all_components)
        spectrogram_keys = keys if config.field == "both" else keys[:1]
        for key in spectrogram_keys:
            figures[f"03_spectrogram_{key}"] = plotting.plot_spectrogram(
                t, integrate_over_span(y, all_components[key]),
                config.f_forcing, config.n_harmonics,
                title=("Spectrogram of span-integrated "
                       f"{labels.get(key, key)}"),
                nperseg=config.spectrogram_nperseg)

    if result.sweep is not None and len(result.sweep.wind_speeds) > 0:
        figures["04_frequency_sweep"] = plotting.plot_frequency_sweep(
            result.sweep.wind_speeds, result.sweep.f_dom, config.f_forcing)

    if result.forcing_response is not None:
        figures["05_forcing_response"] = plotting.plot_forcing_response(
            result.forcing_response, harmonic=1, labels=labels)
    if result.cycle_energy is not None:
        figures["06_cycle_energy"] = plotting.plot_cycle_energy(
            result.cycle_energy, config.stations)

    paths = []
    for name, fig in figures.items():
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def write_results(result: SpectralAnalysisResult, config: SpectralConfig,
                  output_dir: Path, include_raw: bool = True) -> Path:
    """Save result and config to ``output_dir / spectral_results.npz``.

    Reload with :func:`viv_rom.export.load_result`. ``include_raw=False``
    leaves out the raw force field (``Fx`` and ``Fz``).
    """
    return save_result(result, config, Path(output_dir) / EXPORT_FILENAME,
                       include_raw=include_raw)


def run(config: SpectralConfig, output_dir: Path | None = None,
        export: bool = True, export_raw: bool = True
        ) -> SpectralAnalysisResult:
    """Analyse a case, print and save the summary, write figures and export.

    ``export`` saves result and config to ``spectral_results.npz``
    (:func:`write_results`); ``export_raw=False`` omits the raw force field
    from it.
    """
    output_dir = Path(output_dir or config.default_output_dir)
    result = analyse(config)
    log_summary(result, config)
    summary_path = write_summary(result, config, output_dir)
    write_figures(result, config, output_dir)
    logger.info("\nSummary written to: %s", summary_path.resolve())
    logger.info("Figures written to: %s", output_dir.resolve())
    if export:
        path = write_results(result, config, output_dir, export_raw)
        logger.info("Results exported to: %s", path.resolve())
    return result
