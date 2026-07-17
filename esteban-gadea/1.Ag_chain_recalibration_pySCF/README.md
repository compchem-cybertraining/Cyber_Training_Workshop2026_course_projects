# Ag Chain Ab Initio Recalibration

## Context

This is a sub-project of the moire chain excitons project (tight-binding + real-time
electron dynamics on Ag atomic chains, developed at the Quantum Dynamics Workshop).
The existing single-chain TB model (`Ag_chains_parametrization.pdf`) reproduces a PBE
(Quantum ESPRESSO, plane-wave) band gap and uses a phenomenological electron-hole
interaction term,

    f_xc^A = -alpha * sum_B  dq_B / sqrt(r_AB^2 + gamma^2)

to generate a bound sub-gap excitonic peak in real-time Ehrenfest dynamics. Neither the
band gap (PBE-derived) nor alpha (hand-picked) was anchored to a level of theory beyond
semi-local DFT. **Objective 1 of this project (`report/objective1_report.md`) uses
pySCF-based quantum chemistry to correct both.**

**Start here:** [`report/objective1_report.md`](report/objective1_report.md) is the main
deliverable -- objective, strategy, results (4 figures), limitations, and the resulting
TB parameters. Everything else in this repository is supporting methodology,
investigation history, and the scripts needed to reproduce every number in that report.

## Project history (methodology investigations)

Getting to Objective 1's result took several dead ends, each documented rather than
erased, since they rule out approaches a reader might otherwise reasonably try first:

- **`Step0/`** -- convergence/feasibility recipe (k-points, vacuum spacing, cluster
  size, CASSCF robustness) for periodic HF on Ag chains. See `Step0/README.md`.
- **`Step2/`** -- the full electron-hole binding energy investigation: six methods
  attempted in order (frozen CASCI IP/EA, relaxed UHF delta-SCF, closed-shell
  Koopmans+CIS, EOM-IP/EA-CCSD, CAM-B3LYP TDDFT, phenomenological dielectric-scaled
  CIS), the multiple-local-minima problems that ruled out the open-shell routes, and
  the k(delta) rescaling calibration data. See `Step2/README.md`.
- **`Step3_calibration/`** -- formalizes the k_inf = k(delta) fit and applies it to the
  verified beta(r) parameters from `Ag_chains_parametrization.pdf`. This is where
  Objective 1's final TB parameters actually come from.
- **`step1_abandoned/`** -- a separate plan to rebuild beta(r) from a full new
  periodic-HF (l, delta) grid (originally "Step 1"), abandoned once it became clear
  that HF's own shape is *worse* than PBE's near the metallic limit (Figure 1 of the
  report) -- keeping PBE's shape and rescaling its magnitude (`Step3_calibration/`)
  sidesteps that problem entirely rather than trading it for a different one.
  `step1_surface_scan.py`, `step1_preflight_check.py`, and `submit_step1_array.slurm`
  are kept as a historical record; they were never run to completion and are not part
  of the Objective 1 result.

## Reproducing Objective 1's result

```
conda env create -f environment.yml
conda activate ag-chain-ai-recalibration
```

1. `Step2/exciton_binding_convergence.py` -- Koopmans+CIS baseline binding energy
   (natoms 2-24). Already-run output: `exciton_binding_convergence.csv`.
2. `Step2/exciton_binding_eom_ccsd.py` -- EOM-IP/EA-CCSD fundamental gap + CIS optical
   gap (natoms 2-12). Already-run output: `exciton_binding_eom_ccsd.csv`.
3. `Step2/exciton_binding_lc_tddft.py` -- CAM-B3LYP TDDFT optical gap screening test
   (natoms 2-8). Already-run output: `exciton_binding_lc_tddft.csv`.
4. `Step2/beta_rescaling_scan.py` then `Step2/beta_rescaling_summary.py` -- the 8-point
   k(delta) calibration scan along the equilibrium ridge. Already-run output:
   `beta_rescaling_k_summary.csv`.
5. `Step2/generate_nto.py` -- Natural Transition Orbital analysis + cube files
   (`report/nto_cubes/`), confirming the excited electron's Ag 5s character.
6. `Step3_calibration/fit_k_delta.py` -- fits k(delta) = k_inf + A/delta, produces
   `k_inf_fit.csv` and Figure 2.
7. `Step3_calibration/rescale_beta.py` -- applies k_inf to beta(r), produces
   `rescaled_beta_parameters.csv` and `gap_old_vs_new.csv`.
8. `Step3_calibration/fig1_metal_transition_hf.py` -- cheap illustrative periodic HF
   scan near delta=0 for Figure 1 (rerun a few times; checkpointed, ~9s/point).
9. `figures/make_report_figures.py` -- builds Figures 1, 3, 4 from the CSVs above
   (Figure 2 comes from step 6 directly).

All scripts checkpoint to CSV and are safe to re-run. Steps 1-3 and 5 are the
expensive ones (EOM-CCSD in particular; natoms=12 took ~2.4 min in step 2, and cost
rises steeply with natoms -- see script docstrings for details before extending any
natoms grid).

## Reference data and background

- `Ag_chains_parametrization.pdf` -- the original TB model report; Section 3 gives the
  beta(r)/repulsive-potential functional forms and parameters that Step3_calibration/
  rescales.
- `Step3_calibration/DFT_bgap.csv` -- the full PBE (l, delta) band gap grid.
- `Step3_calibration/TB_bgap.csv` -- the original TB model's own fit to that grid
  (used as the calibration denominator, NOT the raw DFT grid directly -- the two
  differ by the TB fit's own small residual, ~0.003 eV mean absolute deviation per
  the original report).
- `Step3_calibration/refit_data/TBenergy.dat`, `DFT_energysurface.dat` -- raw
  potential-energy-surface data (TB, 80-pair chain; DFT, single cell/many k-points)
  behind the Section 6 repulsive-potential re-fit. The Monte Carlo fitting code
  itself lives in the separate [`TB_Ag_excitons`](https://github.com/estebangadea/TB_Ag_excitons)
  repository, not here -- see `report/objective1_report.md` Section 6 for the
  resulting parameters and fitting conventions (relative-to-minimum energies,
  80:1 TB:DFT cell scaling, weighting favoring (E-E_min) < 0.1 eV configurations).

## Key technical conventions

- Pseudopotential/basis: `gth-pbe` + `gth-szv-molopt-sr` -- the only Ag combination
  bundled by default with pySCF. Treats 4d^10 5s^1 (11 valence electrons) explicitly,
  i.e. it carries the full d-manifold, not just the 5s band the TB model represents.
- Periodicity (Step0/HF convergence work only): genuine 3D "wire array" (a real
  3D-periodic Cell with a large vacuum gap in the transverse directions), not pySCF's
  native `dimension=1` mode -- the low-dimensional Coulomb kernel for 1D wires is
  flagged by pySCF itself as inaccurate.
- Gamma-only finite supercells only sample the true zone-boundary band edge when
  natoms/2 is even (proven via exact zone-folding matches in Step 0). Not relevant to
  the finite-cluster (Step2/Step3) work, where any even natoms is a valid data point.
- `mcscf.CASSCF(mf, ...)` does NOT automatically inherit density fitting from a
  DF-based mean field -- always call `.density_fit()` explicitly.
- Open-shell SCF (UHF) on charged states of this system has the same multiple-minima
  pathology as CASSCF and is worse to diagnose. Avoid charged-state/open-shell routes
  for this system; prefer closed-shell alternatives (Koopmans' theorem, EOM-IP/EA-CCSD).
- HF-level fundamental gaps are systematically too large relative to the true
  (correlated/screened) answer -- confirmed via the delta=0 gap not vanishing (Figure 1)
  and via exciton binding energies not dropping enough even with real dynamical
  correlation (CCSD) or a range-separated DFT kernel (CAM-B3LYP) added on top.
- beta(r)'s equilibrium hopping (beta_eq) and the dimerization-sensitive slope (A) are
  rescaled independently -- the fundamental gap depends only on A (beta_eq cancels in
  gap = 2|beta1-beta2|), so only A is corrected by k_inf; beta_eq is untouched.

## Status

Objective 1 complete -- see `report/objective1_report.md`. Final model parameters
(Section 6, repulsive-potential re-fit): beta_eq = -0.668653 eV (unchanged),
r_eq = 3.989152 Angstrom, A = 1.150209 eV/Angstrom (unchanged from the k_inf
correction), B = 0.032205, p = 8. Section 7's alpha-vs-binding-energy scan
(0.03-0.21 eV over alpha=0.8-1.6) falls well short of the ab initio EOM-CCSD
estimate (~1.2-1.4 eV) and of the model's own gap-based ceiling at the l=6.0 Angstrom
equilibrium geometry (1.32 eV) -- open question for follow-up work: whether higher
alpha (untested here) remains dynamically stable and can close that gap, or whether
the ab initio estimate itself is still an overestimate.
