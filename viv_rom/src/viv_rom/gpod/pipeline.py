"""End-to-end global POD of a parameter sweep.

Typical interactive use::

    from viv_rom.gpod import GlobalPODConfig, analyse
    from viv_rom.pod import PODConfig

    base = PODConfig.from_case("a01_p85_i50", data_dir="data/archive_1D",
                               normalization=("dynamic_pressure", "work"))
    config = GlobalPODConfig(base=base, parameter="wind_speed",
                             values=range(6, 23))
    result = analyse(config)
    result.relative_error            # (n_cases, n_modes)
    result.case(12).reconstruct(4)   # rank-4 field of the 12 m/s case, N/m

:func:`analyse` never plots or writes files; :func:`write_figures`,
:func:`write_results` and :func:`run` do.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from viv_rom.gpod import plotting
from viv_rom.gpod.analysis import (GlobalPODResult, LeaveOneOutResult,
                                   analyse_snapshots,
                                   leave_one_out_snapshots)
from viv_rom.gpod.config import GlobalPODConfig
from viv_rom.gpod.snapshots import load_snapshots
from viv_rom.io.results.export import save_result
from viv_rom.pod import plotting as pod_plotting

logger = logging.getLogger(__name__)
EXPORT_FILENAME = "gpod_results.npz"
LOO_EXPORT_FILENAME = "gpod_loo_results.npz"


# =============================================================================
# Analysis
# =============================================================================
def analyse(config: GlobalPODConfig) -> GlobalPODResult:
    """Load the simulations of ``config`` and compute the global POD."""
    logger.info("Global POD over %s = %s (case %s, field '%s')",
                config.parameter, list(config.values), config.case_label,
                config.base.field)
    cases, missing = load_snapshots(config)
    return analyse_snapshots(cases, config, missing=missing)


def leave_one_out(config: GlobalPODConfig, with_energy: bool = False
                  ) -> LeaveOneOutResult:
    """Leave-one-out projection of every simulation (reloads the data)."""
    cases, _ = load_snapshots(config)
    return leave_one_out_snapshots(cases, config, with_energy=with_energy)


# =============================================================================
# Reporting
# =============================================================================
def _fmt_modes(n: int) -> str:
    return "-" if n < 0 else str(n)


def summary_text(result: GlobalPODResult, config: GlobalPODConfig,
                 loo: LeaveOneOutResult | None = None) -> str:
    b = result.basis
    names = ", ".join(result.normalization_names) or "none"
    lines = [
        f"--- Global POD summary ({config.parameter} sweep, field = "
        f"'{config.base.field}', normalization = {names}) ---",
        f"{len(result.cases)} simulations, {int(b.n_snapshots.sum())} "
        f"snapshots pooled; centering = {b.centering}, case weighting = "
        f"{b.weighting}, whole periods = {config.whole_periods}",
    ]
    if result.missing:
        lines.append(f"no data file for {list(result.missing)}")

    lines += ["", f"{'mode':>4} {'pooled energy %':>16} {'cum %':>8}"]
    for k in range(min(config.base.n_modes_show + 4, b.n_modes)):
        lines.append(f"{k + 1:4d} {100 * b.energy_frac[k]:16.2f} "
                     f"{100 * b.cum_energy[k]:8.2f}")
    lines.append(f"Modes for 99% of the pooled fluctuation energy: "
                 f"{result.n99 if result.n99 is not None else 'not reached'}")

    metric = None
    if any(c.energy is not None for c in result.cases):
        metric = next(iter(next(c.energy for c in result.cases
                                if c.energy is not None).n_required))
    lines += ["", "--- Per simulation (errors relative to the fluctuation "
              "energy, normalised space) ---"]
    head = (f"{config.parameter[:10]:>10} {'snaps':>6} {'periods':>8} "
            f"{'err r=1':>9} {'err r=2':>9} {'err r=4':>9} "
            f"{'n99 glob':>9}")
    if result.has_local:
        head += f" {'n99 loc':>8}"
    if metric:
        head += f" {'E glob':>7}"
        if result.has_local:
            head += f" {'E loc':>6}"
    lines.append(head)
    err = result.relative_error
    n99, e_g = result.n99_cases, result.energy_modes_required(metric)
    n99_l = result.local_n99 if result.has_local else None
    e_l = result.local_energy_modes_required(metric) \
        if result.has_local else None

    def e(i, r):
        return f"{err[i, r - 1]:9.2e}" if err.shape[1] >= r else f"{'-':>9}"

    for i, c in enumerate(result.cases):
        row = (f"{c.value:10d} {c.pod.time_coeff.shape[1]:6d} "
               f"{c.n_periods:8.1f} {e(i, 1)} {e(i, 2)} {e(i, 4)} "
               f"{_fmt_modes(n99[i]):>9}")
        if n99_l is not None:
            row += f" {n99_l[i]:8d}"
        if metric:
            row += f" {_fmt_modes(e_g[i]):>7}"
            if e_l is not None:
                row += f" {_fmt_modes(e_l[i]):>6}"
        lines.append(row)
    if metric:
        lines.append(f"(E = modes for the energy-injection error "
                     f"({metric}) below {100 * config.base.energy_tolerance:g}"
                     " %; '-' = not reached)")

    if result.angles is not None:
        a = result.angles
        lines += ["", f"--- Principal angles of the leading {a.k}-D local "
                  "subspaces [deg] ---",
                  f"neighbouring simulations: largest angle "
                  f"{a.neighbour.max():.1f}, mean of the largest per pair "
                  f"{a.neighbour[:, -1].mean():.1f}",
                  f"local vs global: largest angle {a.to_global.max():.1f}, "
                  f"mean of the largest per simulation "
                  f"{a.to_global[:, -1].mean():.1f}"]

    if loo is not None:
        le = loo.relative_error
        lines += ["", "--- Leave-one-out (basis fitted without the "
                  "simulation) ---",
                  f"{config.parameter[:10]:>10} {'err r=1':>9} {'err r=2':>9} "
                  f"{'err r=4':>9} {'err r=8':>9}"]
        for i, v in enumerate(loo.values):
            cells = "".join(f"{le[i, r - 1]:9.2e}" if le.shape[1] >= r
                            else f"{'-':>9}" for r in (1, 2, 4, 8))
            lines.append(f"{v:10d} {cells}")
        if any(c.energy is not None for c in loo.cases):
            ne = loo.energy_modes_required(metric)
            lines.append("energy-injection modes: " + ", ".join(
                f"{v}: {_fmt_modes(n)}" for v, n in zip(loo.values, ne)))
    return "\n".join(lines)


def log_summary(result: GlobalPODResult, config: GlobalPODConfig,
                output_dir: Path | None = None,
                loo: LeaveOneOutResult | None = None) -> str:
    summary = summary_text(result, config, loo)
    logger.info("\n%s", summary)
    output_dir = Path(output_dir or config.default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gpod_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def write_figures(result: GlobalPODResult, config: GlobalPODConfig,
                  output_dir: Path, loo: LeaveOneOutResult | None = None,
                  dpi: int = 150) -> list[Path]:
    """Create all figures and save them (PNG and SVG) to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = config.base
    figures = {
        "01_energy_spectrum":
            lambda: pod_plotting.plot_energy_spectrum(result.pod),
        "02_spatial_modes":
            lambda: pod_plotting.plot_spatial_modes(
                y=result.y, pod=result.pod, field_info=result.field_info,
                motion=base.motion, n_show=base.n_modes_show,
                normalization=result.normalization_names),
        "03_reconstruction_error":
            lambda: plotting.plot_error_curves(result, loo),
        "04_modes_required":
            lambda: plotting.plot_modes_required(result, loo),
        "05_mode_share":
            lambda: plotting.plot_mode_share(result),
    }
    if result.angles is not None:
        figures["06_subspace_angles"] = lambda: plotting.plot_subspace_angles(
            result.angles, result.parameter)
    if any(c.energy is not None for c in result.cases):
        figures["07_mode_energy_injection"] = (
            lambda: plotting.plot_mode_injection(result, base.n_modes_show))
    if any(c.phases is not None for c in result.cases):
        figures["08_harmonic_response"] = (
            lambda: plotting.plot_harmonic_response(result))
    figures["09_temporal_spectra"] = (
        lambda: plotting.plot_temporal_spectra(
            result, n_show=base.n_modes_show, f_forcing=base.f_forcing))

    paths = []
    for name, make in figures.items():
        fig = make()
        path = output_dir / (name + ".png")
        fig.savefig(path, dpi=dpi)
        fig.savefig(output_dir / (name + ".svg"))
        plt.close(fig)
        paths.append(path)
    return paths


def write_results(result: GlobalPODResult, config: GlobalPODConfig,
                  output_dir: Path, include_raw: bool = True,
                  loo: LeaveOneOutResult | None = None) -> list[Path]:
    """Save result (and leave-one-out result) with the config as ``.npz``.

    Reload with :func:`viv_rom.io.results.export.load_result`.
    ``include_raw=False`` leaves out the raw force fields of ``result``; the
    leave-one-out file never contains them (they are in the main file).
    """
    output_dir = Path(output_dir)
    written = [save_result(result.without_raw() if not include_raw else result,
                           config, output_dir / EXPORT_FILENAME)]
    if loo is not None:
        written.append(save_result(loo.without_raw(), config,
                                   output_dir / LOO_EXPORT_FILENAME))
    return written


def run(config: GlobalPODConfig, output_dir: Path | None = None,
        export: bool = True, export_raw: bool = True, loo: bool = False,
        loo_energy: bool = False) -> GlobalPODResult:
    """Analyse a sweep, log a summary, write figures and the export.

    Parameters
    ----------
    loo : bool
        Also run the leave-one-out projection (a basis per held-out
        simulation); reported and plotted next to the in-sample results.
    loo_energy : bool
        With ``loo``: also the energy-injection analysis of the held-out
        simulations (slower).
    """
    output_dir = Path(output_dir or config.default_output_dir)
    cases, missing = load_snapshots(config)
    result = analyse_snapshots(cases, config, missing=missing)
    loo_result = None
    if loo:
        logger.info("Leave-one-out projection ...")
        loo_result = leave_one_out_snapshots(cases, config,
                                             with_energy=loo_energy)
    del cases
    log_summary(result, config, output_dir, loo_result)
    write_figures(result, config, output_dir, loo_result)
    logger.info("\nFigures written to: %s", output_dir.resolve())
    if export:
        for path in write_results(result, config, output_dir, export_raw,
                                  loo_result):
            logger.info("Results exported to: %s", path.resolve())
    return result
