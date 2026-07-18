# Exciton Self-Trapping Dynamics in Libra

## Context

This is the second sub-project of the moiré chain excitons project (tight-binding +
real-time electron dynamics on Ag atomic chains), continuing directly from
[`1.Ag_chain_recalibration_pySCF`](../1.Ag_chain_recalibration_pySCF). With the TB
model's band gap and electron-hole coupling anchored to ab initio quantum chemistry
there, this sub-project ports the model into
[Libra](https://github.com/Quantum-Dynamics-Hub/libra-code), the nonadiabatic
molecular dynamics package used throughout the workshop, and runs real-time excited-
state dynamics to observe exciton self-trapping.

**Start here:** [`../report.md`](../report.md), Part II, is the main deliverable for
this sub-project -- methodology, the three production simulations, results, and a
future-work plan for Libra's broader method suite. Everything in this folder is the
code that produces those results.

## What's here

- **`model/`** -- the TB→Libra port itself:
  - `AgChain.py` -- the electronic (hopping) Hamiltonian, ported into Libra's diabatic
    `compute_model` contract and cross-validated against the project's original Julia
    implementation.
  - `cis_exciton.py`, `cis_gradient.py` -- the exciton reformulation: Libra's dynamics
    engine propagates a single quantum amplitude (`rho = C*C^dagger`, a pure state),
    not the many-electron mean-field density matrix the original model used, so the
    exciton is recast as a Configuration Interaction Singles (CIS), Tamm-Dancoff
    problem over a window of near-gap configurations, with analytic nuclear gradients
    (including a degenerate-perturbation-theory extension needed because every orbital
    in this ring except the HOMO/LUMO band edge is exactly degenerate).
  - `cis_time_overlap.py` -- a nonadiabatic coupling formula between electronic states,
    derived via Slater-Condon rules for this model's orthonormal site basis. Used by
    ongoing work (see `../report.md` Section 2.5); not part of the three production runs.
  - `cis_compute_adi.py` -- the full Libra on-the-fly adiabatic `compute_model` contract
    (state energies, analytic gradients, ground-state baseline, repulsive potential),
    built against the same template Libra's DFTB+ workflows use. **`get_default_params()`
    here holds the production parameters** -- the ab initio-corrected Hamiltonian from
    Part I and the electron-hole coupling used for the runs below; see its docstring.
  - `exciton_density.py` -- post-processing: real-space hole/electron density and
    inverse participation ratio (IPR), the localization diagnostic used throughout.
- **`recipes/ehrenfest_onthefly.py`** -- the Libra dynamics recipe (mean-field Ehrenfest
  force, on-the-fly adiabatic Hamiltonian) used by every run below.
- **`production_runs/`** -- the three simulations behind `../report.md` Part II, Section
  2.4, and their analysis scripts. See "Reproducing the production runs" below.
- **`report/figures/`** -- figures generated from the production runs, referenced by
  `../report.md`.

## Reproducing the production runs

Needs the `libra` kernel (`liblibra_core`/`cyglibra_core`, `util.libutil`, `libra_py`)
on the Python path -- these runs are not reproducible on a machine without a working
Libra install (e.g. a CHPC cluster build).

1. `production_runs/run_1_unseeded.py` -- exciton dynamics from the exactly symmetric
   equilibrium geometry, zero initial velocity. Expected: no localization (nothing
   breaks the ring's symmetry). Output: `run1_unseeded/`.
2. `production_runs/run_2_selftrapping_and_control.py` -- runs both remaining
   simulations from the *same* random velocity seed: the exciton run (self-trapping)
   and the ground-state control (no exciton). Output: `run2_selftrapping/`,
   `run3_control/`.
3. `production_runs/analyze_dynamics.py <prefix>` -- bond-distortion and hole/electron
   density (IPR) diagnostics for one run, e.g. `analyze_dynamics.py run2_selftrapping`.
   Produces `<prefix>_maps.png` (hole density, electron density, and bond-distortion
   heatmaps together, so the exciton's localized region can be checked directly against
   where the lattice is maximally distorted) and `<prefix>_localization.png` (IPR vs.
   RMS bond distortion).
4. `production_runs/check_energy_conservation.py <prefix>` -- confirms any observed
   growth is real dynamics, not integration drift.
5. `production_runs/compare_runs.py` -- overlays all three runs' bond-distortion traces
   on one plot, once all three have been run.

All three runs use a 32-unit-cell (64-site) ring and the same physical parameters
(only the initial velocity and the propagated electronic state differ) -- see
`../report.md` Section 2.3 for the full setup and Section 2.4 for the results.

## Development history

The methodology above -- the CIS/TDA reformulation, its analytic gradients, the
degenerate-orbital fix, and the Libra `compute_model` port -- was built and validated
incrementally, including an earlier round of exploratory dynamics runs (under the
original, pre-Part-I hopping parameters) that first established the self-trapping
signature this sub-project's production runs repeat under the corrected Hamiltonian.
That full development log, including diagnostics and dead ends not needed for the
final result, lives in the separate development repository,
[`TB_Ag_excitons/libra_port`](https://github.com/estebangadea/TB_Ag_excitons/tree/main/libra_port)
-- kept there deliberately as an active working folder, not duplicated here. This
folder holds only the code needed to reproduce the report's actual results.

## Status

Part II is complete. All three production runs (`production_runs/`) have been run on
the corrected Hamiltonian from Part I; results, figures, and discussion are in
`../report.md` Section 2.4. Headline result: Run 2 (seeded exciton) shows a clean,
energy-conserving self-trapping signature not previously seen in the original
real-time Ehrenfest implementation (`../report.md` Section 2.1). Remaining open
items are tracked in `../report.md` Sections 2.5-2.7 (ongoing nonadiabatic-coupling
work, near-crossing search, FSSH/decoherence, two-chain moiré extension).
