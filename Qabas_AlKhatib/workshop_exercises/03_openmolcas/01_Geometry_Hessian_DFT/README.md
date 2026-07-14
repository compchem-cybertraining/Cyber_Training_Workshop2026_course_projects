# Example 1 — Ground-State Geometry & Hessian (DFT)

## What this calculation does

This example optimizes the ground-state (S₀) geometry of ethylene with **DFT (B3LYP)**
and then computes the **analytic Hessian**. The Hessian is the starting point for the
whole dynamics workflow: we shall convert it into the  **pySpawn `hessian.hdf5`** format 
in the second part of today's session and then use it to **Wigner-sample the initial conditions** 
for the AIMS trajectories.

The input:

1. **GATEWAY** — molecule and basis (6-31G\*), `Group = NoSym`.
2. **`Do While` … `EndDo`** — geometry-optimization loop:
   - **SEWARD** — integrals each step.
   - **SCF with `KSDFT = B3LYP`** — Kohn–Sham DFT energy and gradient.
   - **SLAPAF** — updates the geometry until the forces vanish.
3. **MCKINLEY with `ShowHessian`** — analytic second derivatives at the optimized
   geometry, printing the **mass-weighted Hessian** and harmonic frequencies.

## What to look for in the output

- The **converged S₀ geometry** (planar, D₂ₕ ethylene).
- The **MCKINLEY** section: the Hessian and the **harmonic vibrational frequencies**.
  All frequencies should be **real (positive)** — a confirmation that you are at a true
  minimum (no imaginary modes).
- The Hessian / frequency data is what the **`Molcas_2_pySpawn_hessian.py`** converter
  (in the pySpawn tutorial) reads to build `hessian.hdf5`.

## How to run

```bash
sbatch 01_Ethylene_Geom_Hess_DFT.job
```

The job loads `openmolcas/26.02` and runs `pymolcas 01_Ethylene_Geom_Hess_DFT.in`.

## Notes

- DFT is used here only to get a **cheap, reliable Hessian** for sampling — the excited-
  state energies in the dynamics come from CASSCF/CASPT2, not from DFT.
- Make sure the frequency calculation runs at the **optimized** geometry; SLAPAF must
  converge before MCKINLEY is meaningful.
- This Hessian feeds **Example 1 → pySpawn** directly: optimized geometry + Hessian →
  `hessian.hdf5` → Wigner initial conditions.
