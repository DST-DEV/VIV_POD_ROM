"""End-to-end SPOD analysis of one CFD case.

Typical interactive use::

    from viv_rom.spod import SPODConfig, analyse
    result = analyse(SPODConfig.from_case("a01_p85_i50", wind_speed=12,
                                          data_dir="data/archive_1D"))
    result.spod.mode_fractions()[result.content.bins]

:func:`analyse` computes without plotting or file output;
:func:`write_figures` and :func:`run` add the figures and the text report,
which is printed and saved as ``spod_summary.txt``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viv_rom.domain.data import ForceFieldFile
from viv_rom.loading import load_case
from viv_rom.pod.core import FieldInfo, build_field_matrix
from viv_rom.pod.normalization import NormalizationContext
from viv_rom.pod.preprocessing import Preprocessing, Preprocessor
from viv_rom.spod import plotting
from viv_rom.spod.analysis import (HarmonicContent, SpectralPeaks,
                                   fundamental_mode_lines, harmonic_content,
                                   spectral_peaks)
from viv_rom.spod.blocks import FourierBlocks, fourier_blocks
from viv_rom.spod.config import SPODConfig
from viv_rom.spod.core import SPODResult, compute_spod
from viv_rom.spod.injection import SPODInjection, spod_injection
from viv_rom.spod.locking import PhaseLockedResponse, phase_locked_response
from viv_rom.spod.weights import make_inner_product

logger = logging.getLogger(__name__)
SUMMARY_FILENAME = "spod_summary.txt"


@dataclass(frozen=True, eq=False)
class SPODAnalysisResult:
    """Everything computed by :func:`analyse`."""
    force_file: ForceFieldFile
    field_info: FieldInfo
    preprocessing: Preprocessing
    blocks: FourierBlocks
    spod: SPODResult
    content: HarmonicContent
    peaks: SpectralPeaks
    locked: PhaseLockedResponse
    injection: SPODInjection | None
    dt: float
    n_periods: float
    time_mean_square: float      # weighted mean-square of the fluctuation

    @property
    def y(self) -> np.ndarray:
        return self.force_file.data.y

    @property
    def t(self) -> np.ndarray:
        return self.force_file.data.t

    @property
    def locked_physical(self) -> np.ndarray:
        """Locked fundamental modes ``H_k`` in N/m, shape (n_pts, n_modes)."""
        return self.preprocessing.scaling.invert(self.locked.response[0])


def analyse(config: SPODConfig) -> SPODAnalysisResult:
    """Load a case and run the SPOD with its lock-in diagnostics."""
    logger.info("Loading case %s, u = %s m/s from %s ...", config.case,
                config.wind_speed, config.data_dir)
    force_file = load_case(config)
    y, t = force_file.data.y, force_file.data.t
    dt = float(t[1] - t[0])
    n_periods = (t[-1] - t[0]) * config.f_forcing
    logger.info("  %d snapshots, dt = %.6g s, ~%.1f forcing periods, "
                "%d stations", len(t), dt, n_periods, len(y))

    omega = 2 * np.pi * config.f_forcing
    amp = config.effective_motion_amplitude
    field, field_info = build_field_matrix(
        Fx=force_file.data.Fx, Fz=force_file.data.Fz, mode=config.field)
    context = NormalizationContext(
        y=y, wind_speed=force_file.wind_speed, motion=config.motion,
        amplitude=amp, omega=omega, air_density=config.air_density)
    fluctuation, preprocessing = Preprocessor(config.normalizers()) \
        .fit_transform(field, field_info, context)
    logger.info("  normalization: %s",
                ", ".join(preprocessing.normalization_names) or "none")

    weights = make_inner_product(config.weighting).weights(y, field_info)
    spec = config.segment_spec(dt)
    blocks = fourier_blocks(fluctuation, dt, spec)
    spod = compute_spod(blocks, weights)
    logger.info("  %d blocks of %d samples (%.2f periods), df = %.4g Hz",
                blocks.n_blocks, spec.n_fft, spec.n_fft * dt
                * config.f_forcing, spod.df)

    content = harmonic_content(
        spod, config.f_forcing, config.n_harmonics, config.n_modes_show,
        config.band_bins, config.energy_threshold)
    peaks = spectral_peaks(spod, config.f_forcing, config.n_peaks,
                           config.band_bins)

    harmonics = np.arange(1, config.n_harmonics + 1)
    reference = fourier_blocks(np.sin(np.outer(harmonics * omega, t)), dt,
                               spec)
    locked = phase_locked_response(spod, blocks, reference,
                                   config.f_forcing, harmonics)

    injection = None
    if config.do_injection:
        v_amp = field_info.layout(
            config.motion.velocity_amplitude("Fx", y, amp, omega),
            config.motion.velocity_amplitude("Fz", y, amp, omega))
        physical = preprocessing.scaling.invert(locked.response[0])
        injection = spod_injection(
            physical, v_amp, field_info, y, config.f_forcing,
            tolerance=config.energy_tolerance,
            aggregators=config.energy_error_metrics,
            reference=config.energy_error_reference)

    mean_square = float(np.mean(np.sum(weights[:, None] * fluctuation ** 2,
                                       axis=0)))
    return SPODAnalysisResult(
        force_file=force_file, field_info=field_info,
        preprocessing=preprocessing, blocks=blocks, spod=spod,
        content=content, peaks=peaks, locked=locked, injection=injection,
        dt=dt, n_periods=n_periods, time_mean_square=mean_square)


def spod_summary_text(result: SPODAnalysisResult, config: SPODConfig) -> str:
    """The text report: single source for console and ``spod_summary.txt``."""
    spod, spec = result.spod, result.spod.spec
    norm = ", ".join(result.preprocessing.normalization_names) or "none"
    ratio = (spod.variance / result.time_mean_square
             if result.time_mean_square > 0 else float("nan"))
    lines = [
        f"--- SPOD summary: case {config.case}, u = {config.wind_speed} "
        f"m/s ---",
        f"field = '{config.field}', weighting = '{config.weighting}', "
        f"normalization = {norm}, f_forcing = {config.f_forcing} Hz",
        f"{len(result.t)} snapshots (~{result.n_periods:.1f} forcing "
        f"periods); blocks of {spec.n_fft} samples "
        f"({spec.n_fft * result.dt * config.f_forcing:.2f} periods), "
        f"overlap {spec.n_overlap} ({100 * spec.n_overlap / spec.n_fft:.0f}"
        f"%), window '{spec.window}'",
        f"{spod.n_blocks} blocks -> at most {spod.n_modes} SPOD modes per "
        f"frequency; df = {spod.df:.4g} Hz "
        f"({spod.df / config.f_forcing:.3g} x forcing)",
        f"Parseval check: sum(lambda) df = {spod.variance:.4g} vs "
        f"time-domain weighted mean-square {result.time_mean_square:.4g} "
        f"(ratio {ratio:.3f}; Welch windowing makes this approximate)",
        "Eigenvalues are one-sided PSDs of the weighted fluctuation field "
        "(mean removed" + ("" if norm == "none" else ", normalised")
        + ").",
        ""]
    sections = [result.peaks.summary_lines(),
                result.content.summary_lines(),
                fundamental_mode_lines(spod, config.f_forcing, result.locked,
                                       result.injection, config.n_modes_show
                                       + 2),
                result.locked.summary_lines(config.f_forcing)]
    if result.injection is not None:
        sections.append(result.injection.summary_lines())
    for section in sections:
        lines += section + [""]
    return "\n".join(lines).rstrip("\n")


def log_summary(result: SPODAnalysisResult, config: SPODConfig) -> None:
    """Print the text summary to the console (via the logger)."""
    logger.info("\n%s", spod_summary_text(result, config))


def write_summary(result: SPODAnalysisResult, config: SPODConfig,
                  output_dir: Path) -> Path:
    """Save the text summary to ``output_dir / spod_summary.txt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_FILENAME
    path.write_text(spod_summary_text(result, config) + "\n",
                    encoding="utf-8")
    return path


def write_figures(result: SPODAnalysisResult, config: SPODConfig,
                  output_dir: Path, dpi: int = 150) -> list[Path]:
    """Create all figures and save them (PNG and SVG) to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spod, y = result.spod, result.y
    n_show = config.n_modes_show

    figures = {
        "01_eigenvalue_spectrum": plotting.plot_eigenvalue_spectrum(
            spod, config.f_forcing, config.n_harmonics, n_show),
        "02_harmonic_modal_content": plotting.plot_harmonic_modal_content(
            result.content, result.locked),
        "03_leading_mode_map": plotting.plot_leading_mode_map(
            spod, y, result.field_info, config.f_forcing,
            config.n_harmonics, result.preprocessing.scaling.factors),
    }
    locked = result.locked
    share = np.divide(locked.locked_energy[0],
                      locked.locked_energy[0].sum() or 1.0)
    figures["04_locked_modes"] = plotting.plot_locked_modes(
        y, result.locked_physical, result.field_info, config.motion, share,
        n_show)
    if result.injection is not None:
        figures["05_injection"] = plotting.plot_injection(result.injection)

    paths = []
    for name, fig in figures.items():
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def run(config: SPODConfig, output_dir: Path | None = None
        ) -> SPODAnalysisResult:
    """Analyse a case, print and save the summary, write the figures."""
    output_dir = Path(output_dir or config.default_output_dir)
    result = analyse(config)
    log_summary(result, config)
    summary_path = write_summary(result, config, output_dir)
    write_figures(result, config, output_dir)
    logger.info("\nSummary written to: %s", summary_path.resolve())
    logger.info("Figures written to: %s", output_dir.resolve())
    return result
