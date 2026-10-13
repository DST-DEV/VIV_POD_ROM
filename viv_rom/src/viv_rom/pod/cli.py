"""POD command line: ``viv pod`` / ``viv-pod`` / ``python -m viv_rom.pod``."""
from __future__ import annotations

import argparse

from viv_rom.cli_common import (add_case_arguments, case_kwargs,
                                run_over_wind_speeds, setup_runtime)
from viv_rom.pod.config import PODConfig
from viv_rom.pod.normalization import available_normalizers


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
    parser.add_argument("--normalization", nargs="*", default=[],
                        choices=available_normalizers(), metavar="NAME",
                        help="optional normalization(s) applied before mean "
                             f"removal and POD, any of "
                             f"{', '.join(available_normalizers())} "
                             "(default: none)")
    parser.add_argument("--air-density", type=float,
                        default=default_config.air_density,
                        help="air density [kg/m^3] for dynamic-pressure "
                             "scaling (default: %(default)s)")
    parser.add_argument("--no-energy-analysis", action="store_true",
                        help="skip the energy-injection analysis")
    parser.add_argument("--energy-tolerance", type=float,
                        default=default_config.energy_tolerance,
                        help="accepted error of the reconstructed energy "
                             "injection (0.01 = 99%% accuracy)")
    parser.add_argument("--energy-metrics", nargs="+",
                        default=list(default_config.energy_error_metrics),
                        choices=("mean", "rms", "max"),
                        help="how station errors are combined")
    parser.add_argument("--energy-reference", choices=("peak", "local"),
                        default=default_config.energy_error_reference,
                        help="station errors relative to the peak |E| or "
                             "the local |E_j|")
    parser.add_argument("--phase-modes", type=int,
                        default=default_config.phase_modes,
                        help="leading modes whose coefficient phase is "
                             "reported (0 = off)")
    parser.add_argument("--phase-harmonics", type=int,
                        default=default_config.phase_harmonics,
                        help="harmonics of the forcing frequency for the "
                             "phases")
    parser.add_argument("--no-power-check", action="store_true",
                        help="skip the physical power cross-check")


def build_config(args: argparse.Namespace) -> PODConfig:
    return PODConfig(
        **case_kwargs(args), field=args.field,
        n_modes_show=args.n_modes_show, station_y=args.station_y,
        do_power_check=not args.no_power_check,
        normalization=tuple(args.normalization), air_density=args.air_density,
        do_energy_analysis=not args.no_energy_analysis,
        energy_tolerance=args.energy_tolerance,
        energy_error_metrics=tuple(args.energy_metrics),
        energy_error_reference=args.energy_reference,
        phase_modes=args.phase_modes, phase_harmonics=args.phase_harmonics)


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    from viv_rom.pod.pipeline import run  # heavy import, deferred
    return run_over_wind_speeds(args, build_config, run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv-pod",
        description="POD of the spanwise sectional-force distribution of an "
                    "oscillating blade (forced-motion CFD).")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
