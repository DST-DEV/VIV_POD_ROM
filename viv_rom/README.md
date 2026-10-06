# viv_rom

Analysis and reduced-order modelling of vortex-induced vibration (VIV) from
forced-motion CFD (EllipSys3D) of a wind turbine blade.

Methods (one subpackage each):

| Subpackage | What it does |
|---|---|
| `viv_rom.pod` | Proper orthogonal decomposition of the sectional-force distribution, plus a physical power cross-check |
| `viv_rom.spectral` | Welch PSD per station, frequency-vs-span PSD / amplitude / phase maps, spectrogram, wind-speed sweep (lock-in diagram) |

## Install

    pip install -e .[dev]
    pytest

## Command line

    viv pod      --data-dir data/archive_1D --case a01_p85_i50 -u 10 --field combined
    viv spectral --data-dir data/archive_1D --case a01_p85_i50 -u 10 --field both
    viv pod --help               # all options of a method

Shortcuts `viv-pod` and `viv-spectral`, and `python -m viv_rom.pod` /
`python -m viv_rom.spectral`, run a single method. Figures go to
`results/<case>_u<speed>/<method>` unless `-o` is given.

## Python

    from viv_rom.pod import PODConfig, analyse
    from viv_rom.spectral import SpectralConfig
    from viv_rom.spectral import analyse as spectral_analyse, write_figures

    cfg = SpectralConfig.from_case("a01_p85_i50", wind_speed=12,
                                   data_dir="data/archive_1D")
    res = spectral_analyse(cfg)          # numbers only: no plotting, no files
    res.station_peaks["fx"], res.sweep.f_dom
    write_figures(res, cfg, "results/u12")   # optional

`analyse` never plots or writes files; `write_figures` and `run` do.

## Conventions

* Sectional fields (`Fx`, `Fz`, POD snapshot matrices) have shape
  `(n_sections, n_steps)`; spanwise spectral maps `(n_stations, n_freqs)`.
* `CaseConfig` (`viv_rom/config.py`) holds what every method shares (case,
  wind speed, trimming, forcing frequency); `PODConfig` and `SpectralConfig`
  extend it.

## Layout

    viv_rom/
      config.py  loading.py  signal_utils.py    shared by all methods
      cli.py  cli_common.py                      `viv <method>` umbrella CLI
      domain/   data.py, motion.py               data containers, forced motion
      readers/  cfd_reader.py                    HyReader (.1HY files)
      pod/      config core power plotting pipeline cli
      spectral/ config core plotting sweep pipeline cli

## Adding a method (e.g. SPOD)

1. Create `viv_rom/spod/` with `config.py` (`SPODConfig(CaseConfig)`),
   numerics, `pipeline.py` (`analyse`, `write_figures`, `run`) and `cli.py`.
2. `cli.py` exposes `add_arguments(parser)` and `run_from_args(args)`;
   import the pipeline *inside* `run_from_args`.
3. Register it in `METHODS` in `viv_rom/cli.py` and add a `viv-spod`
   script to `pyproject.toml` if wanted.
