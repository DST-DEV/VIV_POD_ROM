"""POD command line: ``viv pod`` / ``viv-pod`` / ``python -m viv_rom.pod``."""
from __future__ import annotations

import argparse

from viv_rom.cli_common import add_case_arguments, case_kwargs, setup_runtime
from viv_rom.pod.config import PODConfig


def add_arguments(parser: argparse.ArgumentParser) -> None:
    default_config = PODConfig()
    add_case_arguments(parser)
    parser.add_argument("--field", choices=("fx", "fz", "combined"),
                        default=default_config.field,
                        help="force component(s) for POD")
    parser.add_argument("--n-modes-show", type=int,
                        default=default_config.n_modes_show)
    parser.add_argument("--station-y", type=float,
                        default=default_config.station_y,
                        help="spanwise position [m] for time-series/orbit "
                             "plots")
    parser.add_argument("--no-power-check", action="store_true",
                        help="skip the physical power cross-check")


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    config = PODConfig(
        **case_kwargs(args), field=args.field,
        n_modes_show=args.n_modes_show, station_y=args.station_y,
        do_power_check=not args.no_power_check)

    from viv_rom.pod.pipeline import run  # heavy import, deferred
    run(config, output_dir=args.output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv-pod",
        description="POD of the spanwise sectional-force distribution of an "
                    "oscillating blade (forced-motion CFD).")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
