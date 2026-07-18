# Example 3 — S₁ Geometry Optimization (XMS-CASPT2)

## What this calculation does

This example optimizes the **first excited-state (S₁)** minimum of ethylene at the
**XMS-CASPT2 // SA-3-CAS(2,2)SCF/6-31G\*** level, and computes the state-to-state
transition properties at the optimized geometry with RASSI. The S₁ minimum is where an
excited wavepacket relaxes before reaching the conical intersection — a key reference
point for interpreting the dynamics.

The input:

1. **GATEWAY** — molecule, basis (6-31G\*), `Group = NoSym`, `RICD` integrals.
2. **`Do While` … `EndDo`** — optimization loop:
   - **SEWARD** — integrals each step.
   - **SCF** — starting orbitals on the first iteration only.
   - **RASSCF** — state-averaged CAS(2,2) over **3 singlet roots**.
   - **CASPT2** — `multi = all` performs **XMS-CASPT2**; **`RLXRoot = 2`** optimizes the
     **first excited state** (root 2 = S₁). `IMAGinary = 0.25` is the imaginary level
     shift to avoid intruder states; `IPEA = 0.00` sets the IPEA shift.
   - **ALASKA** — analytic CASPT2 gradient (`CutOFF = 1.0D-9`, `SHOW`).
   - **SLAPAF** — geometry update.
3. **RASSI** — at the optimized S₁ geometry, reads the 3 states (`NR OF JOBIPHS = 1 3`),
   computes **transition dipoles** (`TRDI`), prints CI info (`CIPRint`), and uses the
   CASPT2 energies (`EJob`).

## What to look for in the output

- The **converged S₁ geometry** — typically distorted (pyramidalized/twisting onset)
  relative to the planar S₀ minimum.
- The **CASPT2** energies of the 3 roots → the **S₁–S₀ gap** at the S₁ minimum.
- The **RASSI** transition moments at this geometry.

## How to run

```bash
sbatch 03_Ethylene_Geom_S1_CASPT2.job
```

The job loads `openmolcas/26.02` and runs `pymolcas 03_Ethylene_Geom_S1_CASPT2.in`.

## Notes

- `RLXRoot = 2` inside **CASPT2** is what makes this an **excited-state** optimization —
  the relaxation root is selected at the correlated (CASPT2) level here, unlike Example 2
  where it was set at the RASSCF level.
- Excited-state optimizations are more delicate: if SLAPAF struggles, check that the
  state ordering does not switch between iterations (root flipping).
- This S₁ minimum, the S₀ minimum (Example 2), and the MECI (Example 8) together map the
  S₁ relaxation pathway that the AIMS trajectories sample.
