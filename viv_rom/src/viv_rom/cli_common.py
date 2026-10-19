"""Pieces shared by the command-line interfaces of all methods.

Kept free of matplotlib and pipeline imports so ``viv --help`` stays fast.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from viv_rom.config import CaseConfig, parse_case


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options selecting and trimming a CFD case."""
    d = CaseConfig()
    parser.add_argument("--data-dir", type=Path, default=d.data_dir,
                        help="folder containing the .1HY files")
    parser.add_argument("--case", default=d.case,
                        help="amplitude/pitch/inclination code "
                             "(default: %(default)s)")
    parser.add_argument("-u", "--wind-speed", type=int, default=d.wind_speed,
                        help="wind speed [m/s] (default: %(default)s)")
    parser.add_argument("--n-skip", type=int, default=d.n_skip,
                        help="initial-transient rows to drop "
                             "(default: %(default)s)")
    parser.add_argument("--n-last", type=int, default=d.n_last,
                        help="rows to keep after --n-skip (default: all)")
    parser.add_argument("--f-forcing", type=float, default=d.f_forcing,
                        help="excitation frequency [Hz] "
                             "(default: %(default)s)")
    parser.add_argument("--blade-file", type=Path, default=d.blade_file,
                        help="windIO blade description (chord, twist) used "
                             "by the 'inflow' frame (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="figure output folder (default: "
                             "results/<case>/u<speed>/<method>); with a "
                             "wind-speed range, one u<speed> subfolder "
                             "per speed is created inside it")
    sweep = parser.add_argument_group(
        "wind-speed range",
        "Run the analysis for every wind speed start, start+step, ... "
        "<= stop instead of the single --wind-speed. Speeds without a "
        "data file are skipped.")
    sweep.add_argument("--u-start", type=int, default=None, metavar="U",
                       help="first wind speed [m/s]")
    sweep.add_argument("--u-stop", type=int, default=None, metavar="U",
                       help="last wind speed [m/s], inclusive")
    sweep.add_argument("--u-step", type=int, default=None, metavar="DU",
                       help="wind-speed increment [m/s] (default: 1)")


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options controlling the export of the result to a file."""
    group = parser.add_argument_group(
        "result export",
        "Result and config are saved to <method>_results.npz in the output "
        "folder (reload with viv_rom.export.load_result).")
    group.add_argument("--no-export", action="store_true",
                       help="do not save the results to a file")
    group.add_argument("--no-export-raw", action="store_true",
                       help="leave the raw force fields out of the export "
                            "(much smaller file)")


def export_kwargs(args: argparse.Namespace) -> dict:
    """Keyword arguments for a pipeline's ``run`` from the export options."""
    return {"export": not getattr(args, "no_export", False),
            "export_raw": not getattr(args, "no_export_raw", False)}


def case_kwargs(args: argparse.Namespace) -> dict:
    """:class:`CaseConfig` keyword arguments from parsed CLI arguments."""
    return {**parse_case(args.case), "data_dir": args.data_dir,
            "wind_speed": args.wind_speed, "n_skip": args.n_skip,
            "n_last": args.n_last, "f_forcing": args.f_forcing,
            "blade_file": args.blade_file}


def setup_runtime() -> None:
    """Headless plotting and plain-message logging for batch runs."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# =============================================================================
# Wind-speed range
# =============================================================================
def wind_speed_range(start: int, stop: int, step: int = 1) -> list[int]:
    """Inclusive range ``start, start+step, ... <= stop`` (``step > 0``)."""
    if step <= 0:
        raise ValueError(f"Wind-speed step must be positive, got {step}.")
    if stop < start:
        raise ValueError(f"Wind-speed stop ({stop}) is below start "
                         f"({start}).")
    return list(range(start, stop + 1, step))


def wind_speeds_from_args(args: argparse.Namespace) -> list[int] | None:
    """Wind speeds requested by ``--u-start/--u-stop/--u-step``.

    Returns ``None`` when no range was given (single-speed run). Start and
    stop must come together; an omitted step defaults to 1 m/s.
    """
    start, stop, step = args.u_start, args.u_stop, args.u_step
    if start is None and stop is None:
        if step is not None:
            raise ValueError("--u-step needs --u-start and --u-stop.")
        return None
    if start is None or stop is None:
        raise ValueError("--u-start and --u-stop must be given together.")
    return wind_speed_range(start, stop, 1 if step is None else step)


def run_over_wind_speeds(args: argparse.Namespace,
                         make_config: Callable[[argparse.Namespace], object],
                         run: Callable[..., object],
                         adapt: Callable[[object, bool], object] | None = None
                         ) -> int:
    """Run one method for a single wind speed or for a requested range.

    Parameters
    ----------
    make_config : callable(args) -> config
        Builds the method config from the parsed arguments (including
        ``args.wind_speed``).
    run : callable(config, output_dir=...)
        The method pipeline's ``run``.
    adapt : callable(config, is_first) -> config, optional
        Per-speed adjustment, e.g. to do speed-independent work only once.

    Without a range this is exactly ``run(make_config(args), output_dir=
    args.output_dir)``. With a range every speed gets its own config; with
    ``-o DIR`` the results go to ``DIR/u<speed>``, otherwise to the
    method's default folder of that speed. Speeds without a data file are
    skipped and reported; other errors are logged, the sweep continues and
    the exit code is 1.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 if nothing ran or a speed failed.
    """
    log = logging.getLogger(__name__)
    try:
        speeds = wind_speeds_from_args(args)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from None
    if speeds is None:
        run(make_config(args), output_dir=args.output_dir)
        return 0

    done, missing, failed = [], [], []
    for u in speeds:
        log.info("\n%s\nWind speed %d m/s\n%s", "=" * 60, u, "=" * 60)
        config = make_config(argparse.Namespace(**{**vars(args),
                                                   "wind_speed": u}))
        if adapt is not None:
            config = adapt(config, not done)
        out = None if args.output_dir is None else \
            Path(args.output_dir) / f"u{u:02d}"
        try:
            run(config, output_dir=out)
        except FileNotFoundError as err:
            log.warning("Skipping u=%d m/s: %s", u, err)
            missing.append(u)
        except Exception:                      # keep the sweep going
            log.exception("Analysis failed for u=%d m/s", u)
            failed.append(u)
        else:
            done.append(u)

    log.info("\nWind-speed range %s: completed %s", speeds, done or "none")
    if missing:
        log.info("  no data file for %s", missing)
    if failed:
        log.info("  failed for %s", failed)
    return 0 if done and not failed else 1
