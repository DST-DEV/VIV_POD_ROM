"""Spectral-analysis command line: ``viv spectral`` / ``viv-spectral``."""
from __future__ import annotations

import argparse

from viv_rom.cli_common import add_case_arguments, case_kwargs, setup_runtime
from viv_rom.spectral.config import SpectralConfig


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = SpectralConfig()
    add_case_arguments(parser)
    parser.add_argument("--field", choices=("fx", "fz", "both"),
                        default=d.field,
                        help="force component(s) analysed per station")
    parser.add_argument("--stations", type=int, nargs="+",
                        default=list(d.stations), metavar="IDX",
                        help="spanwise station indices (0 = root) "
                             "(default: %(default)s)")
    parser.add_argument("--n-harmonics", type=int, default=d.n_harmonics,
                        help="forcing harmonics to mark")
    parser.add_argument("--nperseg", type=int, default=d.nperseg,
                        help="Welch segment length (default: full record)")
    parser.add_argument("--spectrogram-nperseg", type=int,
                        default=d.spectrogram_nperseg,
                        help="STFT segment length "
                             "(default: max(64, n_steps // 12))")
    parser.add_argument("--no-spanwise-map", action="store_true",
                        help="skip the frequency-vs-span PSD map")
    parser.add_argument("--no-amplitude-map", action="store_true",
                        help="skip the frequency-vs-span amplitude map")
    parser.add_argument("--no-phase-map", action="store_true",
                        help="skip the frequency-vs-span phase map")
    parser.add_argument("--no-spectrogram", action="store_true",
                        help="skip the spectrogram of the total force")
    parser.add_argument("--phase-mask-db", type=float,
                        default=d.phase_mask_threshold_db, metavar="DB",
                        help="mask phase this many dB below the panel peak "
                             "(default: %(default)s)")
    parser.add_argument("--no-phase-mask", action="store_true",
                        help="do not mask low-magnitude phase")
    parser.add_argument("--no-sweep", action="store_true",
                        help="skip the wind-speed sweep")
    parser.add_argument("--u-sweep", type=int, nargs="+",
                        default=list(d.u_sweep_list), metavar="U",
                        help="wind speeds [m/s] for the sweep "
                             "(default: 6 ... 22)")
    parser.add_argument("--sweep-field", choices=("fx", "fz"),
                        default=d.sweep_field,
                        help="component used for the sweep")


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    config = SpectralConfig(
        **case_kwargs(args), field=args.field,
        stations=tuple(args.stations), n_harmonics=args.n_harmonics,
        nperseg=args.nperseg, spectrogram_nperseg=args.spectrogram_nperseg,
        do_spanwise_map=not args.no_spanwise_map,
        do_spanwise_amplitude_map=not args.no_amplitude_map,
        do_spanwise_phase_map=not args.no_phase_map,
        do_spectrogram=not args.no_spectrogram,
        do_frequency_sweep=not args.no_sweep,
        phase_mask_threshold_db=(None if args.no_phase_mask
                                 else args.phase_mask_db),
        u_sweep_list=tuple(args.u_sweep), sweep_field=args.sweep_field)

    from viv_rom.spectral.pipeline import run  # heavy import, deferred
    run(config, output_dir=args.output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv-spectral",
        description="Spectral analysis (PSD, phase, spectrogram, lock-in "
                    "sweep) of the sectional forces of an oscillating "
                    "blade (forced-motion CFD).")
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
