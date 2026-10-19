"""Spectral-analysis command line: ``viv spectral`` / ``viv-spectral``."""
from __future__ import annotations

import argparse
import dataclasses
from functools import partial

from viv_rom.cli_common import (add_case_arguments, add_export_arguments,
                                case_kwargs, export_kwargs,
                                run_over_wind_speeds, setup_runtime)
from viv_rom.domain.frames import available_frames
from viv_rom.spectral.config import SpectralConfig, available_fields


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = SpectralConfig()
    add_case_arguments(parser)
    add_export_arguments(parser)
    parser.add_argument("--frame", choices=available_frames(),
                        default=d.frame,
                        help="coordinate frame the forces are rotated into "
                             "before analysis: 'blade' (Fx/Fz), 'motion' "
                             "(along/normal to the local motion, "
                             "fpar/fperp) or 'inflow' (along/normal to the "
                             "free stream, fdrag/flift) "
                             "(default: %(default)s)")
    parser.add_argument("--field", choices=available_fields(),
                        default=d.field,
                        help="force component(s) analysed per station; "
                             "must belong to the chosen frame")
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
    parser.add_argument("--unwrap-phase", action="store_true",
                        help="whether phases should be unwrapped from the "
                        "(-pi, pi] window to show continous increase/decrease")
    parser.add_argument("--no-sweep", action="store_true",
                        help="skip the wind-speed sweep")
    parser.add_argument("--u-sweep", type=int, nargs="+",
                        default=list(d.u_sweep_list), metavar="U",
                        help="wind speeds [m/s] for the sweep "
                             "(default: 6 ... 22)")
    parser.add_argument("--sweep-field",
                        choices=available_fields()[:-1],
                        default=d.sweep_field,
                        help="component used for the sweep "
                             "(default: first component of the frame)")
    parser.add_argument("--no-forcing-response", action="store_true",
                        help="skip amplitude/phase at the forcing frequency")
    parser.add_argument("--response-harmonics", type=int,
                        default=d.response_harmonics,
                        help="harmonics of the forcing frequency fitted for "
                             "the amplitude/phase")
    parser.add_argument("--no-cycle-energy", action="store_true",
                        help="skip the cycle-by-cycle energy injection")
    parser.add_argument("--no-shedding", action="store_true",
                        help="skip the vortex-shedding frequency estimate "
                             "along the span")
    parser.add_argument("--strouhal", type=float, nargs=2,
                        default=list(d.strouhal_range),
                        metavar=("LOW", "HIGH"),
                        help="Strouhal range for the shedding estimate "
                             "(default: %(default)s)")
    parser.add_argument("--lock-in-band", type=float, nargs=2,
                        default=list(d.lock_in_band),
                        metavar=("LOW", "HIGH"),
                        help="lock-in range as multiples of the forcing "
                             "frequency (default: %(default)s)")


def build_config(args: argparse.Namespace) -> SpectralConfig:
    return SpectralConfig(
        **case_kwargs(args), frame=args.frame, field=args.field,
        stations=tuple(args.stations), n_harmonics=args.n_harmonics,
        nperseg=args.nperseg, spectrogram_nperseg=args.spectrogram_nperseg,
        do_spanwise_map=not args.no_spanwise_map,
        do_spanwise_amplitude_map=not args.no_amplitude_map,
        do_spanwise_phase_map=not args.no_phase_map,
        do_spectrogram=not args.no_spectrogram,
        do_frequency_sweep=not args.no_sweep,
        phase_mask_threshold_db=(None if args.no_phase_mask
                                 else args.phase_mask_db),
        unwrap_phase=args.unwrap_phase,
        u_sweep_list=tuple(args.u_sweep), sweep_field=args.sweep_field,
        do_forcing_response=not args.no_forcing_response,
        response_harmonics=args.response_harmonics,
        do_cycle_energy=not args.no_cycle_energy,
        do_shedding=not args.no_shedding,
        strouhal_range=tuple(args.strouhal),
        lock_in_band=tuple(args.lock_in_band))


def run_from_args(args: argparse.Namespace) -> int:
    setup_runtime()
    from viv_rom.spectral.pipeline import run  # heavy import, deferred
    # The --u-sweep frequency sweep does not depend on --wind-speed, so in a
    # wind-speed range it is done once (for the first speed) only.
    return run_over_wind_speeds(args, build_config,
                                partial(run, **export_kwargs(args)),
                                adapt=without_frequency_sweep)


def without_frequency_sweep(config, is_first: bool):
    """Adapter: keep a speed-independent sweep only for the first speed."""
    if is_first or not getattr(config, "do_frequency_sweep", False):
        return config
    return dataclasses.replace(config, do_frequency_sweep=False)


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
