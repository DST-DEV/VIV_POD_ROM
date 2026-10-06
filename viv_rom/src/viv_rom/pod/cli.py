"""Command-line interface: ``pod`` / ``python -m viv_rom.pod.cli``."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from viv_rom.pod.config import PODConfig, parse_case


def build_parser() -> argparse.ArgumentParser:
    defaults = PODConfig()
    p = argparse.ArgumentParser(
        prog="viv-pod",
        description="POD of the spanwise sectional-force distribution of an "
                    "oscillating blade (forced-motion CFD).")
    p.add_argument("--data-dir", type=Path, default=defaults.data_dir,
                   help="folder containing the .1HY files")
    p.add_argument("--case", default=defaults.case,
                   help="amplitude/pitch/inclination code "
                   "(default: %(default)s)")
    p.add_argument("-u", "--wind-speed", type=int, default=defaults.wind_speed,
                   help="wind speed [m/s] (default: %(default)s)")
    p.add_argument("--field", choices=("fx", "fz", "combined"),
                   default=defaults.field, help="force component(s) for POD")
    p.add_argument("--n-skip", type=int, default=defaults.n_skip,
                   help="initial-transient rows to drop")
    p.add_argument("--n-last", type=int, default=None,
                   help="rows to keep after --n-skip (default: all)")
    p.add_argument("--n-modes-show", type=int, default=defaults.n_modes_show)
    p.add_argument("--f-forcing", type=float, default=defaults.f_forcing,
                   help="excitation frequency [Hz]")
    p.add_argument("--station-y", type=float, default=defaults.station_y,
                   help="spanwise position [m] for time-series/orbit plots")
    p.add_argument("--no-power-check", action="store_true",
                   help="skip the physical power cross-check")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="figure output folder (default: "
                        "results/<case>_u<speed>/pod)")
    return p


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg", force=True)  # headless / batch runs

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = PODConfig(
        data_dir=Path(args.data_dir).resolve(), **parse_case(args.case),
        wind_speed=args.wind_speed, field=args.field, n_skip=args.n_skip,
        n_last=args.n_last, n_modes_show=args.n_modes_show,
        f_forcing=args.f_forcing, station_y=args.station_y,
        do_power_check=not args.no_power_check)

    from viv_rom.pod.pipeline import run
    run(config, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
