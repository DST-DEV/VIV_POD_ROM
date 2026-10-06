"""End-to-end POD analysis of one CFD case.

Typical interactive use::

    from viv_rom import PODConfig, analyse
    result = analyse(PODConfig.from_case("a01_p85_i50", wind_speed=12,
                                         data_dir="data/archive_1D"))
    result.pod.energy_frac[:4]

The three steps are also available separately: :func:`analyse` (no plotting,
no file output), :func:`write_figures`, and :func:`run` (both, plus a
printed summary).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viv_rom.pod import plotting
from viv_rom.loading import load_case
from viv_rom.pod.config import PODConfig
from viv_rom.domain.data import ForceFieldFile
from viv_rom.pod.core import (FieldInfo, PODResult, build_field_matrix,
                              compute_pod)
from viv_rom.pod.power import PowerCheck, power_correlation_check
from viv_rom.signal_utils import dominant_frequency

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PODAnalysisResult:
    """Everything computed by :func:`analyse`."""
    force_file: ForceFieldFile
    field: np.ndarray  # Force field array on which the pod was applied
    field_info: FieldInfo
    pod: PODResult
    dominant_freqs: np.ndarray      # per reported mode [Hz]
    n99: int                        # modes needed for 99 % of the energy
    dt: float
    n_periods: float
    station_index: int
    power: PowerCheck | None = None

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


def analyse(config: PODConfig) -> PODAnalysisResult:
    """Load a case and run the POD (and optional power check)."""
    logger.info("Loading case %s, u = %s m/s from %s ...", config.case,
                config.wind_speed, config.data_dir)
    force_file = load_case(config)
    y, t = force_file.data.y, force_file.data.t
    Fx, Fz = force_file.data.Fx, force_file.data.Fz

    dt = t[1] - t[0]
    n_periods = (t[-1] - t[0]) / (1.0 / config.f_forcing)
    logger.info("  %d snapshots, dt = %.6g s, spanning ~%.1f forcing periods",
                len(t), dt, n_periods)
    logger.info("  %d spanwise stations, y = %.1f to %.1f m",
                len(y), y[0], y[-1])

    field, field_info = build_field_matrix(Fx=Fx, Fz=Fz, mode=config.field)
    pod = compute_pod(field=field)

    n_report = min(config.n_modes_show, pod.n_modes)
    dominant_freqs = np.array([
        dominant_frequency(signal=pod.time_coeff[k, :], dt=dt)[0]
        for k in range(n_report)])
    n99 = pod.modes_for_energy(0.99)

    candidates = np.where(y >= config.station_y)[0]
    if len(candidates) == 0:
        raise ValueError(f"station_y = {config.station_y} m is beyond the "
                         f"last section (y = {y[-1]:.1f} m).")
    station_index = int(candidates[0])

    power = None
    if config.do_power_check:
        work_file = force_file.source_file.with_suffix(".1work")
        power = power_correlation_check(
            y=y, t=t, Fx=Fx, Fz=Fz, pod=pod, field_info=field_info,
            motion=config.motion, amplitude=config.effective_motion_amplitude,
            omega=2 * np.pi * config.f_forcing, n_modes=n99,
            work_file=work_file)

    return PODAnalysisResult(
        force_file=force_file, field=field, field_info=field_info, pod=pod,
        dominant_freqs=dominant_freqs, n99=n99, dt=dt, n_periods=n_periods,
        station_index=station_index, power=power)


def log_summary(result: PODAnalysisResult, config: PODConfig) -> None:
    """Log the POD summary table and the power cross-check."""
    pod = result.pod
    logger.info("\n--- POD summary (field = '%s') ---", config.field)
    logger.info("%4s %10s %8s %15s %17s", "mode", "energy %", "cum %",
                "dom. freq [Hz]", "freq / f_forcing")
    for k, f_dom in enumerate(result.dominant_freqs):
        logger.info("%4d %10.2f %8.2f %15.4f %17.3f", k + 1,
                    100 * pod.energy_frac[k], 100 * pod.cum_energy[k],
                    f_dom, f_dom / config.f_forcing)
    logger.info("\nModes needed to capture 99%% of the fluctuation energy: %d",
                result.n99)

    p = result.power
    if p is not None:
        logger.info("\n--- Physical power cross-check ---")
        logger.info("Mean input power (from Fx and prescribed velocity): "
                    "%.4g W  (%s the motion)", p.mean_power,
                    "exciting" if p.mean_power > 0 else "damping")
        if p.mean_power_work is not None:
            logger.info("Mean power from *.1work file (cross-check): %.4g W",
                        p.mean_power_work)
        logger.info("Correlation of each leading POD mode's a_k(t) with "
                    "instantaneous power:")
        for k, c in enumerate(p.correlations, start=1):
            logger.info("  mode %d: r = %+.3f", k, c)


def write_figures(result: PODAnalysisResult, config: PODConfig,
                  output_dir: Path, dpi: int = 150) -> list[Path]:
    """Create all figures and save them to ``output_dir``.

    Returns the list of written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_show = config.n_modes_show
    y, t, pod = result.y, result.t, result.pod

    figures = {
        "01_energy_spectrum":
            plotting.plot_energy_spectrum(pod),
        "02_spatial_modes":
            plotting.plot_spatial_modes(y=y, pod=pod,
                                        field_info=result.field_info,
                                        motion=config.motion, n_show=n_show),
        "03_temporal_coefficients":
            plotting.plot_temporal_coefficients(t=t, pod=pod, dt=result.dt,
                                                f_forcing=config.f_forcing,
                                                n_show=n_show),
        "04_reconstruction_check":
            plotting.plot_reconstruction(y=y, t=t, D_full=result.field,
                                         pod=result.pod,
                                         field_info=result.field_info,
                                         station=result.station_index),
    }
    if result.field_info.kind == "combined":
        figures["05_force_orbit"] = plotting.plot_force_orbit(
            t=t, Fx=result.Fx, Fz=result.Fz,
            pod=pod, field_info=result.field_info,
            station=result.station_index, r_values=(2, 4, result.n99))
    if result.power is not None:
        figures["06_power_vs_mode1"] = plotting.plot_power_vs_mode1(
            t=t, pod=pod, power=result.power)
        figures["07_power_sectional"] = plotting.plot_sectional_energy(
            y=y, power=result.power)

    paths = []
    for name, fig in figures.items():
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def run(config: PODConfig, output_dir: Path | None = None
        ) -> PODAnalysisResult:
    """Analyse a case, log a summary and write the figures."""
    output_dir = Path(output_dir or config.default_output_dir)
    result = analyse(config)
    log_summary(result, config)
    write_figures(result, config, output_dir)
    logger.info("\nFigures written to: %s", output_dir.resolve())
    return result
