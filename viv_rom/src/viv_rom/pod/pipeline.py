"""End-to-end POD analysis of one CFD case.

Typical interactive use::

    from viv_rom import PODConfig, analyse
    result = analyse(PODConfig.from_case("a01_p85_i50", wind_speed=12,
                                         data_dir="data/archive_1D"))

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

from viv_rom.io.results.export import save_result
from viv_rom.pod import plotting
from viv_rom.loading import load_case
from viv_rom.pod.config import PODConfig
from viv_rom.domain.data import ForceFieldFile
from viv_rom.pod.core import (FieldInfo, PODResult, build_field_matrix,
                              compute_pod)
from viv_rom.pod.energy_injection import (EnergyInjection,
                                          energy_injection_analysis,
                                          work_alignment_weights)
from viv_rom.pod.normalization import NormalizationContext
from viv_rom.pod.phase import (HarmonicPhases, canonical_signs,
                               harmonic_phase_analysis)
from viv_rom.pod.power import PowerCheck, power_correlation_check
from viv_rom.pod.preprocessing import (PhysicalReconstructor, Preprocessing,
                                       Preprocessor)
from viv_rom.signal_utils import dominant_frequency

logger = logging.getLogger(__name__)
EXPORT_FILENAME = "pod_results.npz"


@dataclass(frozen=True, eq=False)
class PODAnalysisResult:
    """Everything computed by :func:`analyse`."""
    force_file: ForceFieldFile
    field: np.ndarray  # Raw (physical) snapshot matrix, shape (n_points, n_t)
    field_info: FieldInfo
    preprocessing: Preprocessing  # normalisation + mean applied before POD
    pod: PODResult  # decomposition of the preprocessed (fluctuation) field
    dominant_freqs: np.ndarray  # per reported mode [Hz]
    n99: int  # modes needed for 99 % of the energy
    dt: float
    n_periods: float
    station_index: int
    power: PowerCheck | None = None
    energy: EnergyInjection | None = None
    phases: HarmonicPhases | None = None

    @property
    def reconstruct(self) -> PhysicalReconstructor:
        """``reconstruct(r)``: rank-``r`` field in physical units (N/m)."""
        return PhysicalReconstructor(self.pod, self.preprocessing)

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
    # Load force field data
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

    # Retrieve selected force components and build field matrix
    field, field_info = build_field_matrix(Fx=Fx, Fz=Fz, mode=config.field)

    # Preprocess the data: normalization (optional) and mean removal
    omega = 2 * np.pi * config.f_forcing
    context = NormalizationContext(
        y=y, wind_speed=force_file.wind_speed, motion=config.motion,
        amplitude=config.effective_motion_amplitude, omega=omega,
        air_density=config.air_density)
    preprocessor = Preprocessor(config.normalizers())
    fluctuation, preprocessing = preprocessor.fit_transform(
        field, field_info, context)
    logger.info("  normalization: %s",
                ", ".join(preprocessing.normalization_names) or "none")

    # Run proper orthogonal decomposition and reconstruct physical field
    pod = compute_pod(fluctuation)
    reconstruct = PhysicalReconstructor(pod, preprocessing)

    # Report energy and dominant frequency of each mode
    n_report = min(config.n_modes_show, pod.n_modes)
    dominant_freqs = np.array([
        dominant_frequency(signal=pod.time_coeff[k, :], dt=dt)[0]
        for k in range(n_report)])
    n99 = pod.modes_for_energy(0.99)

    # Select station for plotting later
    candidates = np.where(y >= config.station_y)[0]
    if len(candidates) == 0:
        raise ValueError(f"station_y = {config.station_y} m is beyond the "
                         f"last section (y = {y[-1]:.1f} m).")
    station_index = int(candidates[0])

    # Run power analysis incl. reconstruction from the first couple of modes
    power = None
    if config.do_power_check:
        work_file = force_file.source_file.with_suffix(".1work")
        power = power_correlation_check(
            y=y, t=t, Fx=Fx, Fz=Fz, pod=pod, reconstruct=reconstruct,
            field_info=field_info, motion=config.motion,
            amplitude=config.effective_motion_amplitude,
            omega=omega, n_modes=n99,
            work_file=work_file, n_skip=config.n_skip, n_last=config.n_last)

    # Analyse energy injection contribution of each mode
    energy = None
    if config.do_energy_analysis:
        amp = config.effective_motion_amplitude
        energy = energy_injection_analysis(
            model=reconstruct, field=field, field_info=field_info, y=y, t=t,
            vx=config.motion.velocity("x", y, t, amp, omega),
            vz=config.motion.velocity("z", y, t, amp, omega),
            tolerance=config.energy_tolerance,
            aggregators=config.energy_error_metrics,
            reference=config.energy_error_reference)

    # Analyse amplitude & phase of each mode at the forcing frequency
    # (+ harmonics)
    phases = None
    if config.phase_modes > 0:
        # orient the modes by their work alignment with the motion, so that
        # a fundamental phase of +90 deg means energy injection
        work_weights = work_alignment_weights(
            y, config.motion, field_info, preprocessing.scaling.factors)
        phases = harmonic_phase_analysis(
            time_coeff=pod.time_coeff, t=t, f_forcing=config.f_forcing,
            n_modes=config.phase_modes, n_harmonics=config.phase_harmonics,
            unwrap_phase=config.unwrap_phase,
            signs=canonical_signs(pod.modes, work_weights))

    return PODAnalysisResult(
        force_file=force_file, field=field, field_info=field_info,
        preprocessing=preprocessing, pod=pod,
        dominant_freqs=dominant_freqs, n99=n99, dt=dt, n_periods=n_periods,
        station_index=station_index, power=power, energy=energy,
        phases=phases)


def pod_summary_text(result: PODAnalysisResult, config: PODConfig) -> str:
    pod = result.pod

    lines = []
    lines.append(
        f"--- POD summary (field = '{config.field}', normalization = "
        f"{', '.join(result.preprocessing.normalization_names) or 'none'}) ---"
        )
    lines.append(
        f"{'mode':>4} {'energy %':>10} {'cum %':>8} "
        f"{'dom. freq [Hz]':>15} {'freq / f_forcing':>17}"
    )

    for k, f_dom in enumerate(result.dominant_freqs):
        lines.append(
            f"{k+1:4d}"
            f"{100*pod.energy_frac[k]:10.2f}"
            f"{100*pod.cum_energy[k]:8.2f}"
            f"{f_dom:15.4f}"
            f"{f_dom/config.f_forcing:17.3f}"
        )

    lines.append("")
    lines.append(
        f"Modes needed to capture 99% of the fluctuation energy: {result.n99}"
    )
    if not result.preprocessing.scaling.is_identity:
        lines.append("(energy is that of the normalized fluctuation field)")

    p = result.power
    if p is not None:
        lines.append("")
        lines.append("--- Physical power cross-check ---")
        lines.append(
            f"Mean input power (from Fx and prescribed velocity): "
            f"{p.mean_power:.4g} W "
            f"({'exciting' if p.mean_power > 0 else 'damping'} the motion)"
        )

        if p.mean_power_work is not None:
            lines.append(
                f"Mean power from *.1work file (cross-check): "
                f"{p.mean_power_work:.4g} W"
            )

        lines.append(
            "Correlation of each leading POD mode's a_k(t) with "
            "instantaneous power:"
        )

        for k, c in enumerate(p.correlations, start=1):
            lines.append(f"  mode {k}: r = {c:+.3f}")

    for section in (result.energy, result.phases):
        if section is not None:
            lines.append("")
            lines.extend(section.summary_lines())

    return "\n".join(lines)


def log_summary(result: PODAnalysisResult, config: PODConfig,
                output_dir: Path | None = None) -> None:
    summary = pod_summary_text(result, config)

    # Console/log output
    logger.info("\n%s", summary)

    # Save to file
    output_dir = Path(output_dir or config.default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "pod_summary.txt").write_text(summary, encoding="utf-8")


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
            plotting.plot_spatial_modes(
                y=y, pod=pod, field_info=result.field_info,
                motion=config.motion, n_show=n_show,
                normalization=result.preprocessing.normalization_names),
        "03_temporal_coefficients":
            plotting.plot_temporal_coefficients(t=t, pod=pod, dt=result.dt,
                                                f_forcing=config.f_forcing,
                                                n_show=n_show),
        "04_reconstruction_check":
            plotting.plot_reconstruction(y=y, t=t, D_full=result.field,
                                         reconstruct=result.reconstruct,
                                         field_info=result.field_info,
                                         station=result.station_index),
    }
    if result.field_info.kind == "combined":
        figures["05_force_orbit"] = plotting.plot_force_orbit(
            t=t, Fx=result.Fx, Fz=result.Fz,
            reconstruct=result.reconstruct, field_info=result.field_info,
            station=result.station_index, r_values=(2, 4, result.n99))
    if result.power is not None:
        figures["06_power_vs_mode1"] = plotting.plot_power_vs_mode1(
            t=t, pod=pod, power=result.power)
        figures["07_power_sectional"] = plotting.plot_sectional_energy(
            y=y, power=result.power)

    if result.energy is not None:
        figures["08_mode_energy_injection"] = (
            plotting.plot_mode_energy_injection(result.energy, n_show=n_show))
        figures["09_energy_reconstruction_error"] = (
            plotting.plot_energy_reconstruction_error(result.energy))
    if result.phases is not None:
        figures["10_harmonic_phase"] = plotting.plot_harmonic_phases(
            result.phases)

    paths = []
    for name, fig in figures.items():
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def write_results(result: PODAnalysisResult, config: PODConfig,
                  output_dir: Path, include_raw: bool = True) -> Path:
    """Save result and config to ``output_dir / pod_results.npz``.

    Reload with :func:`viv_rom.export.load_result`. ``include_raw=False``
    leaves out the raw force field (``Fx``, ``Fz`` and ``field``).
    """
    return save_result(result, config, Path(output_dir) / EXPORT_FILENAME,
                       include_raw=include_raw)


def run(config: PODConfig, output_dir: Path | None = None,
        export: bool = True, export_raw: bool = True) -> PODAnalysisResult:
    """Analyse a case, log a summary, write the figures and the export.

    ``export`` saves result and config to ``pod_results.npz``
    (:func:`write_results`); ``export_raw=False`` omits the raw force field
    from it.
    """
    output_dir = Path(output_dir or config.default_output_dir)
    result = analyse(config)
    log_summary(result, config, output_dir)
    write_figures(result, config, output_dir)
    logger.info("\nFigures written to: %s", output_dir.resolve())
    if export:
        path = write_results(result, config, output_dir, export_raw)
        logger.info("Results exported to: %s", path.resolve())
    return result
