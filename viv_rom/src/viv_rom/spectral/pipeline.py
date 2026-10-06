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
from viv_rom.loading import load_case
from viv_rom.spectral import plotting
from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.core import (compute_psd,
                                   dominant_peaks,
                                   integrate_over_span, select_components)
from viv_rom.spectral.sweep import SweepResult, frequency_sweep

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class SpectralAnalysisResult:
    """Everything computed by :func:`analyse`.

    Attributes
    ----------
    station_peaks : dict of {"fx"/"fz": ndarray}
        Dominant frequency [Hz] at each of ``config.stations``.
    sweep : SweepResult or None
        Wind-speed sweep, if requested.
    """
    force_file: ForceFieldFile
    dt: float
    n_periods: float
    station_peaks: dict[str, np.ndarray]
    sweep: SweepResult | None = None

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
    components = select_components(config.field, force_file.data.Fx,
                                   force_file.data.Fz)

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

    return SpectralAnalysisResult(force_file=force_file, dt=dt,
                                  n_periods=n_periods,
                                  station_peaks=station_peaks, sweep=sweep)


def log_summary(result: SpectralAnalysisResult, config: SpectralConfig
                ) -> None:
    """Log the per-station table and the sweep results."""
    logger.info("\n--- Dominant frequencies by station ---")
    logger.info("%9s %9s %8s %12s %12s", "component", "station", "y [m]",
                "f_peak [Hz]", "f/f_forcing")
    # for key, peaks in result.station_peaks.items():
    #     for st, f_peak in zip(config.stations, peaks):
    #         logger.info("%9s %9d %8.1f %12.4f %12.3f", key, st,
    #                     result.y[st], f_peak, f_peak / config.f_forcing)

    if result.sweep is None:
        return
    logger.info("\n--- Frequency sweep over U for case '%s' "
                "(field = '%s') ---", config.case, config.sweep_field)
    sweep = result.sweep
    if len(sweep.wind_speeds) == 0:
        logger.info("  No files found for the requested sweep. "
                    "Skipping sweep plot.")
        return
    for U, f in zip(sweep.wind_speeds, sweep.f_dom):
        locked = abs(f - config.f_forcing) < 0.05 * config.f_forcing
        logger.info("  U = %5.1f m/s : f_dom = %.4f Hz (%.2f x forcing)%s",
                    U, f, f / config.f_forcing,
                    "  <-- locked to forcing" if locked else "")


def write_figures(result: SpectralAnalysisResult, config: SpectralConfig,
                  output_dir: Path, dpi: int = 150) -> list[Path]:
    """Create the requested figures and save them to ``output_dir``.

    Returns the list of written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, t = result.y, result.t
    components = select_components(config.field, result.Fx, result.Fz)
    common = dict(f_forcing=config.f_forcing, n_harmonics=config.n_harmonics)

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
        # Fx is always shown; Fz additionally when both components are
        # analysed (behaviour of the original script).
        spectrogram_signals = {"fx": result.Fx}
        if config.field == "both":
            spectrogram_signals["fz"] = result.Fz
        for key, data in spectrogram_signals.items():
            figures[f"03_spectrogram_{key}"] = plotting.plot_spectrogram(
                t, integrate_over_span(y, data), config.f_forcing,
                config.n_harmonics,
                title=f"Spectrogram of span-integrated {key.capitalize()}",
                nperseg=config.spectrogram_nperseg)

    if result.sweep is not None and len(result.sweep.wind_speeds) > 0:
        figures["04_frequency_sweep"] = plotting.plot_frequency_sweep(
            result.sweep.wind_speeds, result.sweep.f_dom, config.f_forcing)

    paths = []
    for name, fig in figures.items():
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def run(config: SpectralConfig, output_dir: Path | None = None
        ) -> SpectralAnalysisResult:
    """Analyse a case, log a summary and write the figures."""
    output_dir = Path(output_dir or config.default_output_dir)
    result = analyse(config)
    log_summary(result, config)
    write_figures(result, config, output_dir)
    logger.info("\nFigures written to: %s", output_dir.resolve())
    return result
