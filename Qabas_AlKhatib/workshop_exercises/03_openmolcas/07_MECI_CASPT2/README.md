# Example 7 — S₁/S₀ Minimum-Energy Conical Intersection (MECI) of Ethylene

## What this calculation does

This example optimizes the **minimum-energy conical intersection (MECI)** between the ground
state (S₀) and the first excited singlet (S₁) of ethylene, at the **XMS-CASPT2 //
SA-3-CAS(2,2)SCF/6-31G\*** level.

A conical intersection is the funnel through which nonadiabatic population transfer occurs —
it is exactly the geometry that the **AIMS trajectories spawn around** in the pySpawn tutorial.
Optimizing the MECI shows students *where* the dynamics is heading. Using **CASPT2** rather than
bare CASSCF places the crossing at a correlated, production-quality level consistent with the
energies used elsewhere in the workshop.

### How the optimization works — the energy-difference constraint

The crossing is located by **minimizing the energy while constraining the two state energies to
be equal**. That constraint is defined in GATEWAY:

```
Constraints
  a = Ediff 1 2        ! define "a" = E(state 2) - E(state 1)
  Value
  a = 0.000            ! force that difference to zero
End of Constraints
```

Then the optimization loop runs at the CASPT2 level:

```
>>> Do While
  &SEWARD
  >>> IF ( ITER = 1 ) <<<           ! SCF only on the first iteration (starting orbitals)
    &SCF
  >>> ENDIF <<<
  &RASSCF
    ciroot  = 3 3 1                 ! state-averaged over 3 singlets
  &CASPT2
    RLXRoot     = 1                 ! relax (take the gradient) on the LOWER state, S0
    XMultiState = All               ! XMS-CASPT2 over all SA states
    IPEA        = 0.00
    IMAGinary   = 0.20              ! imaginary level shift (avoid intruders)
    FROZen      = 0
  &SLAPAF                           ! geometry step, honouring the Ediff = 0 constraint
>>> EndDo
```

SLAPAF moves the geometry downhill on the **CASPT2** S₀ surface (`RLXRoot = 1`) **subject to**
the constraint that S₁ and S₀ stay degenerate. The converged structure is the **lowest-energy
point on the S₁/S₀ crossing seam** at the CASPT2 level.

> Key differences from a CASSCF MECI run:
> - `RLXRoot` is now set **inside the `&CASPT2` block** (the relaxation root is chosen at the
>   correlated level), not in `&RASSCF`.
> - `XMultiState = All` makes this an **XMS-CASPT2** optimization.
> - The constraint recipe does **not** require the derivative coupling vector, so it is robust.

## How to run

A SLURM job file (`Job.slurm`) is provided inside this directory. It loads `openmolcas/26.02`
and runs `pymolcas` on this example's input:

```bash
sbatch Job.slurm
```

> A reference output is included in this directory in case the optimization takes too long to
> finish during the session. Starting from the geometry provided, it converges in only ~9 steps.

---

## Interpreting the output

### 1. Did it reach the crossing? — check the energy difference

At convergence SLAPAF prints:

```
A : Energy difference =  0.00000140 hartree,  0.00367029 kJ/mol
Geometry is converged in 9 iterations to a Minimum Energy Crossing Point Structure
```

The **S₁–S₀ gap is ~1.4×10⁻⁶ hartree** — effectively **zero** (the constraint is satisfied to
~0.004 kJ/mol). The message *"converged to a Minimum Energy Crossing Point Structure"* confirms
a genuine MECI. Starting near the crossing, it converged in just **9 geometry steps**.

### 2. The MECI geometry — twisted and pyramidalized

The final structure shows the classic ethylene S₁/S₀ intersection:

| Feature | Value | Meaning |
|---------|-------|---------|
| C1–C2 distance | **1.426 Å** | stretched relative to the planar GS (~1.33 Å) |
| C1–C2–H6 angle | **83.4°** | one CH strongly bent → **pyramidalized** carbon |
| H–C1–C2–H dihedrals | **~37° / −54° / 117°** | the CH₂ groups are **twisted** about the C=C |

The molecule is **twisted about the C=C bond and pyramidalized at one carbon** — the
"twisted-pyramidalized" ethylene intersection, in sharp contrast to the planar Franck–Condon
geometry the AIMS wavepacket starts from.

### 3. Conical intersection characterization

OpenMolcas analyzes the topography of the crossing (following *J. Chem. Theory Comput.* **12**,
3636–3653 (2016)):

```
 Pitch (delta_gh):                 8.60782E-02 Eh/a0
 Asymmetry (Delta_gh):             4.30468E-01
 Relative tilt (sigma):            6.19409E-01
 P:    2.95857E-01
 B:    1.15554E+00
 Type: peaked (P<1) single-path (B>1)
```

- **Pitch / asymmetry** describe the shape of the double cone in the branching plane.
- **Type: peaked, single-path** is the key qualitative result. A **peaked** intersection
  (P < 1) funnels population efficiently **straight through** the cone — it is a good, fast
  decay funnel (contrast a *sloped* CI, which is tilted so the wavepacket can pass and return).
  The **single-path** label (B > 1) indicates one dominant decay direction.
- This peaked topography is consistent with ethylene's known ultrafast S₁→S₀ internal
  conversion, and it directly affects the hopping/branching you extract from the AIMS dynamics.

> **CASPT2 vs CASSCF note:** at this correlated level the intersection comes out **peaked**
> (P ≈ 0.30). A bare CASSCF optimization of the "same" MECI can give a different topography
> (e.g. sloped) and a flatter cone (smaller pitch). This is a nice illustration that **dynamic
> correlation changes not only energies but the shape of the crossing**, which is part of why
> production dynamics uses CASPT2-quality surfaces.

### 4. The branching-space vectors

The `Local x` and `Local y` matrices are the two vectors that lift the degeneracy — the
**gradient-difference** and **derivative-coupling** directions. Together they span the 2D
**branching space**; displacing along them splits S₁ and S₀, while motion in the other 3N−8
directions preserves the degeneracy (the intersection *seam*). The CASPT2 derivative-coupling
vector itself is printed just above in the ALASKA section (`Total derivative coupling`).

---

## Notes

- **`RLXRoot = 1` inside `&CASPT2`** relaxes on the lower (CASPT2) state; the `Ediff` constraint
  keeps the optimization on the seam — without it you would slide to the S₀ minimum.
- `XMultiState = All` (XMS-CASPT2) gives a balanced multi-state treatment, important near a
  crossing where states mix strongly; `IMAGinary = 0.20` guards against intruder states.
- The **starting geometry** is placed near the twisted–pyramidalized region; an MECI search from
  the planar FC point is much harder and slower.
- Compare the converged MECI geometry (C=C twist, pyramidalization) with the values your
  **pySpawn `analysis.py`** reports for the spawned trajectories — they should cluster here.
- This example pairs with **Example 9** (derivative coupling), which computes one of the two
  branching-space vectors explicitly at a chosen geometry.
