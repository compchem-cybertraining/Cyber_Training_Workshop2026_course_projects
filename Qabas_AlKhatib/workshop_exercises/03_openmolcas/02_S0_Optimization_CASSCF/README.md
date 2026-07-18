# Example 2 — S₀ Geometry Optimization (CASSCF)

## What this calculation does

This example optimizes the **ground-state (S₀)** geometry of ethylene at the
**SA-3-CAS(2,2)SCF/6-31G\*** level — the same active space and state-averaging which we 
plan to use in the pySpawn dynamics. 

The input:

1. **GATEWAY** — molecule, basis (6-31G\*), `Group = NoSym`.
2. **`Do While` … `EndDo`** — optimization loop:
   - **SEWARD** — integrals each step.
   - **SCF** — starting orbitals, computed only on the **first iteration** (`IF ITER = 1`).
   - **RASSCF** — state-averaged CAS(2,2) over **3 singlet roots**; **`RLXRoot = 1`**
     tells SLAPAF to optimize the **ground state** (root 1).
   - **SLAPAF** — geometry update.
3. **MCKINLEY with `ShowHessian`** — analytic Hessian / frequencies at the minimum.

## What to look for in the output

- The **converged S₀ geometry** at the CASSCF level (planar ethylene).
- The **RASSCF** energies of the 3 state-averaged roots.
- The **MCKINLEY** Hessian and **harmonic frequencies** — all real at a true minimum.

## How to run

```bash
sbatch 02_Ethylene_Geom_S0_CASSCF.job
```

The job loads `openmolcas/26.02` and runs `pymolcas 02_Ethylene_Geom_S0_CASSCF.in`.

## Notes

- `RLXRoot = 1` is the key line: in a state-averaged RASSCF, it selects **which root the
  gradient (and therefore the optimization) follows**. Root 1 = S₀.
- State-averaging over 3 roots keeps the orbitals balanced for the excited states even
  while optimizing the ground state — important for a smooth connection to Example 3.
- Compare this CASSCF minimum to the DFT minimum of Example 1; small differences are
  expected and instructive.
