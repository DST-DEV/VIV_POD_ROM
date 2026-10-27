# viv_rom

Analysis and reduced-order modelling of vortex-induced vibration (VIV) from
forced-motion CFD (EllipSys3D) of a wind turbine blade.

Methods (one subpackage each):

| Subpackage | What it does |
|---|---|
| `viv_rom.pod` | Proper orthogonal decomposition of the sectional-force distribution, plus a physical power cross-check |
| `viv_rom.spod` | Spectral POD (Welch/Towne): SPOD modes per frequency, harmonic content, phase-locking to the forced motion, energy injection per mode |
| `viv_rom.spectral` | Welch PSD per station, frequency-vs-span PSD / amplitude / phase maps, spectrogram, wind-speed sweep (lock-in diagram) |

## Install

    pip install -e .[dev]     # pulls in numpy, scipy, matplotlib, pyyaml
    pytest

## Command line

    viv pod      --data-dir data/archive_1D --case a01_p85_i50 -u 10 --field combined
    viv spectral --data-dir data/archive_1D --case a01_p85_i50 -u 10 --field both
    viv spod     --data-dir data/archive_1D --case a01_p85_i50 -u 10 --field combined
    viv pod --help               # all options of a method

Shortcuts `viv-pod`, `viv-spectral` and `viv-spod`, and `python -m viv_rom.pod` /
`python -m viv_rom.spectral` / `python -m viv_rom.spod`, run a single method. Figures go to
`results/<case>/u<speed>/<method>` unless `-o` is given.

### Wind-speed range

All methods accept `--u-start`, `--u-stop` and `--u-step` (default 1) to
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

### Saving and reloading results

Every run also saves its result and configuration to one file in the output
folder: `pod_results.npz`, `spectral_results.npz` or `spod_results.npz`
(next to the figures, so one file per wind speed in a range).

    viv pod --case a01_p85_i50 -u 10 --no-export        # do not save
    viv pod --case a01_p85_i50 -u 10 --no-export-raw    # smaller file

`--no-export-raw` leaves out the raw force field (`Fx`, `Fz`, and the POD
snapshot matrix `field`; `t` and `y` are kept). Everything computed is kept,
but the loaded result then has `None` for the raw arrays, so methods that
need them (e.g. `SpectralAnalysisResult.components`) are unavailable. The
SPOD Fourier blocks (`blocks.q`) are kept as well and can be large.

    from viv_rom.io.export import load_result, save_result

    result, config = load_result("results/a01_p85_i50/u10/pod/pod_results.npz")
    # -> PODAnalysisResult, PODConfig (SpectralAnalysisResult/SpectralConfig
    #    and SPODAnalysisResult/SPODConfig for the other methods)
    save_result(result, config, "copy.npz", include_raw=False)

The file is a compressed NumPy archive: the arrays are stored natively and
the nested dataclasses as a JSON description (`__meta__`) naming their
classes. No pickle is used, and loading only instantiates classes from
`viv_rom`, so a file cannot run foreign code. Consequently results that
hold objects from outside the package (a user-defined `Normalizer`, a
lambda as `energy_error_reference`) cannot be exported; `save_result`
raises a `TypeError` naming the offending field. From Python,
`pipeline.run(config, export=False, export_raw=False)` and
`pipeline.write_results(result, config, output_dir)` do the same as the
flags.

## POD preprocessing and normalization

`analyse` prepares the snapshot matrix in two separate, explicit steps
(`viv_rom/pod/preprocessing.py`): optional **normalization**, then
**mean removal**. `compute_pod` only does the SVD of the prepared fluctuation
field.
Normalization is off by default.

| name | factor | effect |
|---|---|---|
| `dynamic_pressure` | `F / q`, `q = 0.5 rho U^2` | removes the wind-speed scaling, so cases at different `U` are comparable |
| `work` | `F * phi(y) * A * omega` | weights sections by the prescribed velocity amplitude: energy then measures contribution to the energy injection (field in W/m) |

    viv pod ... --normalization dynamic_pressure work     # CLI (default: none)

    PODConfig.from_case("a01_p85_i50", normalization=())                  # none
    PODConfig.from_case("a01_p85_i50", normalization=("dynamic_pressure",))
    PODConfig.from_case("a01_p85_i50", normalization=(WorkWeightNormalizer(floor=1e-2),))

Modes, singular values and temporal coefficients live in the normalized space;
`result.reconstruct(r)` returns the rank-`r` field back in N/m. To add a
strategy, implement `Normalizer.scales(ctx)` and decorate it with
`@register_normalizer("name")` -- nothing else changes.

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
* **Inflow frame** (`--frame inflow`, `--blade-file`). Resolves `(Fx, Fz)` of
  every section along the local flow and normal to it: `F_D` (`fdrag`) and
  `F_L` (`flift`, +90 deg from it in the (x, z) plane). The vortex-shedding
  force that carries the Strouhal frequency is `F_L`; the drag oscillates at
  twice that. Convention (`viv_rom/domain/inflow.py`): the inflow has a
  fixed direction `x_0`; the root frame (forces, motion) is `x_0` rotated
  clockwise by the pitch (angles from `+x` towards `+z`); the twist turns
  each section counter-clockwise, so `alpha(y) = pitch - twist(y)`. Since the
  forces are given in the blade root coordinate frame, only the twist is
  needed for the force rotation into the flow-aligned frame (neglecting the
  movement of the blade).
  Chord and twist come from a windIO file (default
  `data/IEA_10MW/IEA-10-198-RWT.yaml`, loaded lazily and only for the local
  flow quantities): `frame.angle_of_attack(y)`, `frame.projected_width(y)`
  (`D = c |sin alpha|`, the length scale for a local Strouhal number) and
  `frame.normal_speed(U)` (`U cos(inclination)`; the inclination does not
  change the in-plane flow direction). The CFD station `y` is measured from
  the rotor axis, the blade span is `r = y - 2.8 m`.

        from viv_rom.domain import InflowAlignedFrame
        frame = InflowAlignedFrame(pitch=85, inclination=50)
        frame.angle_of_attack(y)         # rad, per station
        viv spectral ... --frame inflow --field flift
* **Relative-inflow frame** (`--frame rel_inflow`, components `fd_rel` /
  `fl_rel`). Like the inflow frame, but the air is taken relative to the
  *moving* section: `w(y, t) = U cos(i) d - v(y, t)` with `d` the static flow
  direction and `v` the prescribed section velocity (`A omega phi(y)
  cos(omega t)`, the same velocity used for power and energy injection).
  `(Fx, Fz)` is resolved along / normal to `w` with the angle
  `phi_rel(y, t) = atan2(w_z, w_x)`, so the rotation depends on span *and*
  time. Helpers: `frame.relative_velocity(y, t)`, `relative_speed`,
  `flow_angle`, `angle_shift` (`phi_rel - phi`), `angle_of_attack(y, t)` and
  `projected_width(y, t)` (`c |sin alpha_eff|`). It uses the case's wind
  speed, motion amplitude and forcing frequency; with amplitude 0 it equals
  the inflow frame. Because the rotation oscillates at the forcing
  frequency, a harmonic force acquires sum / difference frequencies in this
  frame (e.g. shedding frequency +/- forcing frequency): it is meant for
  looking at the load relative to the instantaneous flow, not as a drop-in
  for the phase / energy analyses of the static frames. Frames are applied
  through `viv_rom.domain.frames.transform_forces(frame, y, Fx, Fz, t)`,
  which hands `t` only to frames with `time_dependent = True`; the
  wind-speed sweep builds the frame for each speed.

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

* **Vortex-shedding frequency and lock-in range** (`spectral/shedding.py`,
  `--no-shedding`, `--strouhal LOW HIGH`, `--lock-in-band LOW HIGH`). Each
  section is treated as a bluff body: `f_vs(y) = St * U / D(y)` with
  `D = c |sin(alpha)|`, the chord projected normal to the flow (the
  cross-flow width, from the inflow frame and the windIO blade), and `U`
  the *nominal* wind speed (not reduced by the inclination). `St` is the
  range 0.11-0.16 that Pellegrino & Meskell (2013) found for the NREL S809
  in 2D at high angles of attack: a rough guide here, since other airfoils
  and 3D flow are involved. The result is a band `[f_low, f_high]` per
  station. Lock-in may occur for `f_vs / f_n` of about 0.8-1.1 (amplitude
  dependent), so stations whose band overlaps `[0.8, 1.1] * f_forcing` are
  reported as *possible* lock-in, those entirely inside as *certain*
  (`result.shedding.possible_lock_in`, `.certain_lock_in`, `.spans()`).
  Figure `07_shedding_frequency` shows the band along the span with the
  forcing frequency and the lock-in range. Needs the blade file
  (`--blade-file`); without it the estimate is skipped with a warning.

    viv spectral ... --frame motion --field both --response-harmonics 3

`viv spectral` prints its text summary and also saves it as
`spectral_summary.txt` next to the figures (`-o` folder).

Results: `result.forcing_response` (`.amplitude`, `.phase`, `.explained`) and
`result.cycle_energy` (`.energy`, `.total`, `.station_stats`, `.total_stats`).
`motion` and `motion_amplitude` are now part of the shared `CaseConfig`.

## Spectral POD (SPOD)

`viv_rom.spod` implements the Welch-type SPOD of Towne, Schmidt & Colonius
(2018). The de-meaned (and optionally normalised, as for the POD) field is
cut into overlapping, windowed blocks; at every frequency the eigen-problem
of the cross-spectral density gives the SPOD modes -- complex spanwise shapes
that oscillate at a single frequency -- and their power
`lambda_k(f)` (one-sided weighted PSD, so `sum(lambda) df` is the
mean-square of the field). A coherent, locked response collapses onto *one*
SPOD mode, whereas the real POD of the same data needs several.

    viv spod --case a01_p85_i50 -u 10 --field combined --segment-periods 2
    viv spod ... --normalization dynamic_pressure --weighting span

* **Blocks:** `--segment-periods` (default 2) sets the block length in
  forcing periods, hence the resolution `df = f_forcing / segment_periods`
  and, with a whole number of samples per period, puts the forcing
  frequency and harmonics exactly on bins. Shorter blocks give more blocks
  (= more modes, steadier estimates), longer blocks resolve close lines
  (e.g. a free shedding frequency next to the forcing). `--n-fft`,
  `--overlap` and `--window` give full control. At least two blocks are
  required.
* **Weighting:** `--weighting span` (default, integral over the span) or
  `identity`; add your own with `register_inner_product`.
* **Phase locking** (`spod/locking.py`): the motion is deterministic, so
  each mode's coefficients are regressed on the blocks of `sin(n w t)`. This
  fixes the arbitrary eigenvector phase and gives the locked mode shape
  `H_k(y)`: force `|H_k| sin(n w t + arg H_k)` with the phase convention of the
  POD/spectral modules (+90 deg = in phase with the velocity). It also
  yields the *locked fraction* of the power at each harmonic and the
  coherence of each mode with the motion -- a lock-in strength measure that
  can be compared between wind speeds / pitch angles. Unrelated noise
  gives a floor of about `1/n_blocks`, which the report states.
* **Energy injection** (`spod/injection.py`): the mean power of the locked
  fundamental is `p = 1/2 v_amp Im(H)`; it splits exactly over the SPOD
  modes, with the same "modes needed for 99 %" metrics as the POD
  (`--energy-tolerance`, `--energy-metrics`, `--energy-reference`).
* **Output** (`-o` folder): `spod_summary.txt` (also printed) and figures
  `01_eigenvalue_spectrum`, `02_harmonic_modal_content`,
  `03_leading_mode_map`, `04_locked_modes`, `05_injection`.

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
                blade.py, inflow.py                force frames, windIO blade
                                                 geometry, inflow conventions
      io/       cfd_reader, export               Load .1HY files, save/load results (.npz)
      pod/      config core normalization preprocessing energy_injection phase
                power plotting pipeline cli
      spectral/ config core forcing_response cycle_energy shedding plotting sweep
                pipeline cli
      spod/     config weights blocks core locking injection analysis plotting
                pipeline cli

## Adding a method (worked example: `spod`)

1. Create `viv_rom/spod/` with `config.py` (`SPODConfig(CaseConfig)`),
   numerics, `pipeline.py` (`analyse`, `write_figures`, `run`) and `cli.py`.
2. `cli.py` exposes `add_arguments(parser)` and `run_from_args(args)`;
   import the pipeline *inside* `run_from_args`.
3. Register it in `METHODS` in `viv_rom/cli.py` and add a `viv-spod`
   script to `pyproject.toml` if wanted.
