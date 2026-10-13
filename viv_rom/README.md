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
`results/<case>/u<speed>/<method>` unless `-o` is given.

### Wind-speed range

Both methods accept `--u-start`, `--u-stop` and `--u-step` (default 1) to
run the whole analysis for every speed `start, start+step, ... <= stop`
(stop inclusive) instead of the single `-u`:

    viv pod      --case a01_p85_i50 --u-start 6 --u-stop 22 --u-step 2
    viv spectral --case a01_p85_i50 --u-start 6 --u-stop 22 --u-step 2 -o results/sweep

Each speed writes to its own folder (`results/<case>/u<speed>/<method>`,
or `<-o>/u<speed>` when `-o` is given). Speeds without a `.1HY` file are
skipped and listed at the end; a failing speed is logged and the range
continues (exit code 1). The spectral lock-in frequency sweep (`--u-sweep`)
does not depend on `-u`, so inside a range it is done once, for the first
speed that runs.

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

## POD preprocessing and normalization

`analyse` prepares the snapshot matrix in two separate, explicit steps (`viv_rom/pod/preprocessing.py`): optional **normalization**, then **mean removal**. `compute_pod` only does the SVD of the prepared fluctuation field.
Normalization is off by default.

| name | factor | effect |
|---|---|---|
| `dynamic_pressure` | `F / q`, `q = 0.5 rho U^2` | removes the wind-speed scaling, so cases at different `U` are comparable |
| `work` | `F * phi(y) * A * omega` | weights sections by the prescribed velocity amplitude: energy then measures contribution to the energy injection (field in W/m) |

    viv pod ... --normalization dynamic_pressure work     # CLI (default: none)

    PODConfig.from_case("a01_p85_i50", normalization=())                  # none
    PODConfig.from_case("a01_p85_i50", normalization=("dynamic_pressure",))
    PODConfig.from_case("a01_p85_i50", normalization=(WorkWeightNormalizer(floor=1e-2),))

Modes, singular values and temporal coefficients live in the normalized space; `result.reconstruct(r)` returns the rank-`r` field back in N/m. To add a strategy, implement `Normalizer.scales(ctx)` and decorate it with `@register_normalizer("name")` -- nothing else changes.

## Energy injection and harmonic phase

Two further POD diagnostics run in `analyse` (both configurable on
`PODConfig`, both can be switched off):

* **Energy injection** (`pod/energy_injection.py`, `--no-energy-analysis`).
  `E(y) = integral (Fx*vx + Fz*vz) dt` [J/m] is linear in the field, so
  `E_rec(y; r) = E_mean(y) + sum_{k<=r} E_k(y)` exactly (normalization
  undone). This gives the energy injection of each mode, and per station
  `e_j(r) = |E_rec_j - E_j| / ref_j`, combined over the span by
  `mean`/`rms`/`max` (or any callable). The number of modes required is the
  smallest `r` whose aggregated error is below `energy_tolerance` (0.01 =
  99 %). `ref_j` is the peak `|E|` (`"peak"`, default) or local `|E_j|`.
* **Harmonic phase** (`pod/phase.py`, `--phase-modes K --phase-harmonics H`).
  Least-squares fit of `a_k(t)` with the forcing frequency and harmonics
  `1..H`. Phase is the lead relative to the forced displacement
  `sin(omega t)` (0 deg: displacement, +90 deg: velocity). The sign of a POD
  mode is arbitrary, so each mode is oriented by its work alignment with the
  motion (`canonical_signs` with `work_alignment_weights`): the sign of
  `sum_i integral phi_k^i(y) psi_i(y) dy` over the physical mode shape is made
  positive, so a fundamental phase in (0, 180 deg) means the mode injects
  energy (+90 deg: maximum) and in (-180, 0) it damps. This replaces the
  "largest entry positive" rule, which can be decided at a station that hardly
  moves. `phase_modes=0` disables it.

    viv pod ... --energy-metrics rms max --energy-tolerance 0.01 --phase-modes 4

Results: `result.energy` (`.n_required`, `.modes`, `.mode_totals`,
`.curves`) and `result.phases` (`.phase_deg`, `.amplitude`, `.explained`).

## Spectral: motion-aligned frame, force at the forcing frequency, cycle energy

* **Frame** (`viv_rom/domain/frames.py`, `--frame {blade,motion}`,
  `SpectralConfig(frame=...)`). Forces are rotated into the chosen frame
  *before* every spectral analysis (PSDs, maps, spectrogram, sweep, and the
  analyses below). `blade` (default) is the CFD `Fx`/`Fz`; `motion` rotates
  each station into `F_par` (along the local direction of motion, from the
  un-normalised mode shapes) and `F_perp` (+90 deg from it in the (x, z)
  plane); `--field` and `--sweep-field` then take `fpar`/`fperp`. The
  rotation is orthonormal and `F_par` carries all of the sectional power.
  New frames: implement `ForceFrame` (`name`, `labels`, `transform`) and
  `@register_frame("name")`.
* **Force at the forcing frequency** (`spectral/forcing_response.py`,
  `--response-harmonics H`, `--no-forcing-response`). Joint least-squares
  fit of `f_forcing` and harmonics `1..H` per station and component: amplitude
  [N/m], phase and fit R^2. Phase is the lead relative to the forced
  displacement `sin(omega t)` (0: displacement, +90 deg: velocity). For
  `F_par` the fundamental injects `0.5 * A_1 * |v| * sin(phase)` of mean power:
  `0 < phase < 180 deg` excites, `-180 < phase < 0` damps; harmonics >= 2
  inject nothing over whole periods.
* **Cycle-by-cycle energy injection** (`spectral/cycle_energy.py`,
  `--no-cycle-energy`). Energy `E_c(y)` [J/m] and blade total [J] injected in
  each complete forcing cycle (frame independent: `p = Fx*vx + Fz*vz`), and how
  much it fluctuates: coefficient of variation, relative range, linear drift
  vs. detrended spread (transient vs. real fluctuation) and the number of
  damping cycles, per station and for the blade. Needs several complete cycles
  (with fewer than 3 the spread is flagged unreliable).

    viv spectral ... --frame motion --field both --response-harmonics 3

`viv spectral` prints its text summary and also saves it as
`spectral_summary.txt` next to the figures (`-o` folder).

Results: `result.forcing_response` (`.amplitude`, `.phase`, `.explained`) and
`result.cycle_energy` (`.energy`, `.total`, `.station_stats`, `.total_stats`).
`motion` and `motion_amplitude` are now part of the shared `CaseConfig`.

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
      domain/   data.py, motion.py, frames.py     data containers, forced motion,
                                                 force frames
      readers/  cfd_reader.py                    HyReader (.1HY files)
      pod/      config core normalization preprocessing energy_injection phase
                power plotting pipeline cli
      spectral/ config core forcing_response cycle_energy plotting sweep pipeline
                cli

## Adding a method (e.g. SPOD)

1. Create `viv_rom/spod/` with `config.py` (`SPODConfig(CaseConfig)`),
   numerics, `pipeline.py` (`analyse`, `write_figures`, `run`) and `cli.py`.
2. `cli.py` exposes `add_arguments(parser)` and `run_from_args(args)`;
   import the pipeline *inside* `run_from_args`.
3. Register it in `METHODS` in `viv_rom/cli.py` and add a `viv-spod`
   script to `pyproject.toml` if wanted.
