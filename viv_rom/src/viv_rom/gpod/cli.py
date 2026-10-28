"""Global POD command line: ``viv gpod`` / ``viv-gpod`` /
``python -m viv_rom.gpod``.

Accepts every option of ``viv pod`` (case, normalisation, energy and phase
settings. They apply to each simulation of the sweep) plus the sweep and
pooling options below.
"""
from __future__ import annotations

import argparse

from viv_rom.cli_common import (export_kwargs, setup_runtime,
                                wind_speeds_from_args)
from viv_rom.gpod.config import (CENTERING_OPTIONS, SWEEP_PARAMETERS,
                                 WEIGHTING_OPTIONS, GlobalPODConfig)
from viv_rom.pod import cli as pod_cli


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pod_cli.add_arguments(parser)
    d = GlobalPODConfig(values=(0, 1))  # only for the defaults
    g = parser.add_argument_group(
        "global POD",
        "Pool one simulation per value of the varying parameter (all other "
        "parameters from --case). For --parameter wind_speed the values "
        "may also be given as --u-start/--u-stop/--u-step.")
    g.add_argument("--parameter", choices=SWEEP_PARAMETERS,
                   default=d.parameter,
                   help="varying parameter (default: %(default)s)")
    g.add_argument("--values", type=int, nargs="+", metavar="V",
                   help="values of the parameter to pool")
    g.add_argument("--centering", choices=CENTERING_OPTIONS,
                   default=d.centering,
                   help="mean removed before the SVD: each simulation's own "
                        "('case'), one common mean ('global') or none "
                        "(default: %(default)s)")
    g.add_argument("--weighting", choices=WEIGHTING_OPTIONS,
                   default=d.weighting,
                   help="weight of each simulation in the pooled covariance: "
                        "'time_average' (equal per simulation), 'snapshots' "
                        "(plain concatenation) or 'unit_energy' "
                        "(default: %(default)s)")
    g.add_argument("--no-whole-periods", action="store_true",
                   help="do not crop simulations to whole forcing periods")
    g.add_argument("--n-periods", type=int, default=d.n_periods,
                   help="keep exactly the last N forcing periods of every "
                        "simulation")
    g.add_argument("--n-skip-override", nargs="+", default=[],
                   metavar="VALUE=ROWS",
                   help="initial-transient rows to drop for single values, "
                        "e.g. 6=2400 8=1800 (others use --n-skip)")
    g.add_argument("--strict", action="store_true",
                   help="fail if a requested simulation has no data file "
                        "(default: skip it)")
    g.add_argument("--n-modes-keep", type=int, default=d.n_modes_keep,
                   help="keep only the leading modes of the basis")
    g.add_argument("--no-local", action="store_true",
                   help="skip the single-simulation reference PODs and "
                        "subspace angles")
    g.add_argument("--angle-modes", type=int, default=d.angle_modes,
                   help="subspace dimension for the principal angles "
                        "(default: %(default)s)")
    g.add_argument("--energy-max-modes", type=int,
                   default=d.energy_max_modes,
                   help="limit the modes of the energy-injection analysis")
    g.add_argument("--loo", action="store_true",
                   help="also project every simulation onto the basis fitted "
                        "without it (leave-one-out)")
    g.add_argument("--loo-energy", action="store_true",
                   help="with --loo: also the energy-injection analysis of "
                        "the held-out simulations (slower)")


def _parse_overrides(items: list[str]) -> dict[int, int]:
    out = {}
    for item in items:
        try:
            value, rows = item.split("=")
            out[int(value)] = int(rows)
        except ValueError:
            raise SystemExit(f"error: --n-skip-override expects VALUE=ROWS, "
                             f"got {item!r}") from None
    return out


def build_config(args: argparse.Namespace) -> GlobalPODConfig:
    values = args.values
    if values is None and args.parameter == "wind_speed":
        try:
            values = wind_speeds_from_args(args)
        except ValueError as err:
            raise SystemExit(f"error: {err}") from None
    if values is None:
        raise SystemExit("error: give the values to pool with --values"
                         + (" or --u-start/--u-stop"
                            if args.parameter == "wind_speed" else "")
                         + ".")
    try:
        return GlobalPODConfig(
            base=pod_cli.build_config(args), parameter=args.parameter,
            values=tuple(values), centering=args.centering,
            weighting=args.weighting,
            whole_periods=not args.no_whole_periods,
            n_periods=args.n_periods,
            n_skip_overrides=_parse_overrides(args.n_skip_override),
            skip_missing=not args.strict, n_modes_keep=args.n_modes_keep,
            compute_local=not args.no_local, angle_modes=args.angle_modes,
            energy_max_modes=args.energy_max_modes)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from None


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    from viv_rom.gpod.pipeline import run  # heavy import, deferred
    run(build_config(args), output_dir=args.output_dir,
        loo=args.loo, loo_energy=args.loo_energy, **export_kwargs(args))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv-gpod",
        description="Global POD of the sectional-force field over a sweep of "
                    "one simulation parameter (time-concatenated "
                    "snapshots).")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
