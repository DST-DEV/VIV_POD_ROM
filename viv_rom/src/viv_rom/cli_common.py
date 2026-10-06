"""Pieces shared by the command-line interfaces of all methods.

Kept free of matplotlib and pipeline imports so ``viv --help`` stays fast.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="figure output folder (default: "
                             "results/<case>_u<speed>/<method>)")


def case_kwargs(args: argparse.Namespace) -> dict:
    """:class:`CaseConfig` keyword arguments from parsed CLI arguments."""
    return {**parse_case(args.case), "data_dir": args.data_dir,
            "wind_speed": args.wind_speed, "n_skip": args.n_skip,
            "n_last": args.n_last, "f_forcing": args.f_forcing}


def setup_runtime() -> None:
    """Headless plotting and plain-message logging for batch runs."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
