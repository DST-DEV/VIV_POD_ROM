"""Umbrella command-line interface: ``viv <method> ...``.

Each method provides a ``cli`` module with ``add_arguments(parser)`` and
``run_from_args(args)``. Those modules import their (heavy) pipeline only
inside ``run_from_args``, so building this parser stays cheap.
"""
from __future__ import annotations

import argparse

from viv_rom.pod import cli as pod_cli
from viv_rom.spectral import cli as spectral_cli
from viv_rom.spod import cli as spod_cli

METHODS = {
    "pod": (pod_cli,
            "proper orthogonal decomposition of the sectional forces"),
    "spectral": (spectral_cli,
                 "PSD / phase / spectrogram / lock-in analysis"),
    "spod": (spod_cli,
             "spectral proper orthogonal decomposition with lock-in "
             "diagnostics"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viv", description="Analysis of forced-motion VIV CFD data.")
    sub = parser.add_subparsers(dest="method", required=True,
                                metavar="{" + ",".join(METHODS) + "}")
    for name, (module, help_text) in METHODS.items():
        sp = sub.add_parser(name, help=help_text, description=help_text)
        module.add_arguments(sp)
        sp.set_defaults(run=module.run_from_args)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
