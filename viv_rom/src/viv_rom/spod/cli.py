"""SPOD command line: ``viv spod`` / ``viv-spod`` / ``python -m viv_rom.spod``.
"""
from __future__ import annotations

import argparse

from viv_rom.cli_common import (add_case_arguments, case_kwargs,
                                run_over_wind_speeds, setup_runtime)
from viv_rom.pod.normalization import available_normalizers
from viv_rom.spod.config import SPODConfig
from viv_rom.spod.weights import available_inner_products


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = SPODConfig()
    add_case_arguments(parser)
    parser.add_argument("--field", choices=("fx", "fz", "combined"),
                        default=d.field, help="force component(s) for SPOD")
    parser.add_argument("--weighting", choices=available_inner_products(),
                        default=d.weighting,
                        help="energy norm: 'span' integrates over the span, "
                             "'identity' sums the sections "
                             "(default: %(default)s)")
    parser.add_argument("--normalization", nargs="*", default=[],
                        choices=available_normalizers(), metavar="NAME",
                        help="optional normalization(s) applied before the "
                             f"decomposition, any of "
                             f"{', '.join(available_normalizers())} "
                             "(default: none)")
    parser.add_argument("--air-density", type=float, default=d.air_density,
                        help="air density [kg/m^3] for dynamic-pressure "
                             "scaling (default: %(default)s)")
    seg = parser.add_argument_group(
        "spectral estimation (Welch blocks)",
        "Frequency resolution is f_forcing / segment-periods; more blocks "
        "(shorter segments) give more reliable but fewer SPOD modes.")
    seg.add_argument("--segment-periods", type=float,
                     default=d.segment_periods,
                     help="block length in forcing periods "
                          "(default: %(default)s)")
    seg.add_argument("--n-fft", type=int, default=None,
                     help="block length in samples (overrides "
                          "--segment-periods)")
    seg.add_argument("--overlap", type=float, default=d.overlap,
                     help="block overlap fraction (default: %(default)s)")
    seg.add_argument("--window", default=d.window,
                     help="taper window name (default: %(default)s)")
    parser.add_argument("--n-harmonics", type=int, default=d.n_harmonics,
                        help="forcing harmonics analysed "
                             "(default: %(default)s)")
    parser.add_argument("--n-modes-show", type=int, default=d.n_modes_show,
                        help="leading modes plotted / tabulated "
                             "(default: %(default)s)")
    parser.add_argument("--band-bins", type=int, default=d.band_bins,
                        help="bins either side of a harmonic summed for its "
                             "variance share (default: %(default)s)")
    parser.add_argument("--n-peaks", type=int, default=d.n_peaks,
                        help="spectral peaks listed (default: %(default)s)")
    parser.add_argument("--energy-threshold", type=float,
                        default=d.energy_threshold,
                        help="power fraction for 'modes needed at a "
                             "harmonic' (default: %(default)s)")
    parser.add_argument("--no-injection", action="store_true",
                        help="skip the energy injection of the locked modes")
    parser.add_argument("--energy-tolerance", type=float,
                        default=d.energy_tolerance,
                        help="accepted error of the reconstructed injection "
                             "(0.01 = 99%% accuracy)")
    parser.add_argument("--energy-metrics", nargs="+",
                        default=list(d.energy_error_metrics),
                        choices=("mean", "rms", "max"),
                        help="how station errors are combined")
    parser.add_argument("--energy-reference", choices=("peak", "local"),
                        default=d.energy_error_reference,
                        help="station errors relative to the peak |p| or "
                             "the local |p_j|")


def build_config(args: argparse.Namespace) -> SPODConfig:
    return SPODConfig(
        **case_kwargs(args), field=args.field, weighting=args.weighting,
        normalization=tuple(args.normalization),
        air_density=args.air_density, segment_periods=args.segment_periods,
        n_fft=args.n_fft, overlap=args.overlap, window=args.window,
        n_harmonics=args.n_harmonics, n_modes_show=args.n_modes_show,
        band_bins=args.band_bins, n_peaks=args.n_peaks,
        energy_threshold=args.energy_threshold,
        do_injection=not args.no_injection,
        energy_tolerance=args.energy_tolerance,
        energy_error_metrics=tuple(args.energy_metrics),
        energy_error_reference=args.energy_reference)


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    from viv_rom.spod.pipeline import run  # heavy import, deferred
    return run_over_wind_speeds(args, build_config, run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv-spod",
        description="Spectral POD of the spanwise sectional-force "
                    "distribution of an oscillating blade (forced-motion "
                    "CFD).")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
