# Reduced-Order Modelling of Vortex-Induced Vibrations using Proper Orthogonal Decomposition

This repository contains the code developed for the Master's thesis Wind-M-1206 at the Danish Technical University (DTU) on **Proper Orthogonal Decomposition (POD)** for reduced-order modelling of **vortex-induced vibrations (VIVs)** in wind turbine blades.

The project analyses spanwise integral load time series obtained from high-fidelity CFD simulations to analyze energy transfer between the flow and a vibrating wind turbine blade. The ultimate goal is to investigate whether low-dimensional POD models can accurately represent the aerodynamic loading and predict energy injection for operating conditions that were not directly simulated.

---

## Project Background

Vortex-induced vibrations (VIVs) are an important aeroelastic phenomenon for modern wind turbines due to increasing rotor sizes and decreasing structural natural frequencies. Under certain operating conditions, vortex shedding can synchronize with a structural eigenmode, leading to sustained oscillations and increased fatigue loading.

The CFD simulations analysed in this project were generated in a previous project, where the blade motion was prescribed according to the **first edgewise eigenmode**. Simulations were performed for multiple combinations of

- Wind speed
- Blade pitch angle
- Rotor inclination angle

For each simulation, spanwise distributions of integral aerodynamic forces are available as time series.

---

## Objectives

The repository contains tools for

- Importing and preprocessing CFD simulation results
- Constructing POD bases from spanwise force data
- Analysing modal energy content
- Reconstructing force fields using truncated POD expansions
- Computing aerodynamic energy transfer
- Comparing POD representations across different operating conditions
- Constructing and evaluating global POD bases
- Investigating interpolation between simulated operating points

---

## Repository Structure (WIP)

```text
.
├── data/
├── figures/                  # Generated figures
├── results/                  # Numerical results
├── tests/                    # Unit tests
└── README.md
```

---

## Data

The CFD simulations are **not included** in this repository due to their size and ownership.

Each simulation consists of

- Time vector
- Spanwise positions
- Integral aerodynamic force time series
- Simulation metadata
  - Wind speed
  - Pitch angle
  - Inclination angle
  - Oscillation frequency
  - Motion amplitude

The code assumes a consistent directory structure for all simulations.

---

## Main Research Questions

The code is designed to support the following investigations:

- How do the dominant aerodynamic structures change with wind speed, pitch angle, and inclination angle?
- How many POD modes are required to accurately reconstruct the aerodynamic force field and energy transfer?
- Can a global POD basis represent multiple operating conditions?
- Can the reduced-order model interpolate aerodynamic energy injection at unseen operating conditions?

---

## Requirements

Typical dependencies include

- Python ≥ 3.11
- NumPy
- SciPy
- Matplotlib

---

## Citation

If you use this repository in your own work, please cite the associated Master's thesis.

```text
Davis Stöwer, 2027.
Reduced-order modelling of fluid-structure energy transfer in vortex-induced vibrations.
Master's Thesis, DTU.
```

---

## Acknowledgements

This work was carried out as part of a Master's thesis at the Technical University of Denmark (DTU).

The CFD simulations analysed in this project were produced in a previous research project and are used here as the basis for reduced-order modelling and aerodynamic energy transfer analysis.