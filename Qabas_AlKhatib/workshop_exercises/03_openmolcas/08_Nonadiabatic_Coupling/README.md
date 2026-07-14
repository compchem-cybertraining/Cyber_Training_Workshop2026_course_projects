# Example 8 — Nonadiabatic (Derivative) Coupling Vector of Ethylene

## What this calculation does

This example computes the **nonadiabatic derivative coupling vector (NAC)** between the ground
state (S₀) and the first excited singlet (S₁) of ethylene at a fixed geometry, at the
**SA-2-CAS(2,2)SCF/6-31G\*** level (analytic NAC via the ALASKA module).

The derivative coupling

```
d_IJ = ⟨ Ψ_I | ∂/∂R | Ψ_J ⟩
```

is the quantity that drives population transfer between electronic states. It is exactly the
coupling that the **AIMS spawning criterion** in pySpawn monitors — when it becomes large (near
a conical intersection), new trajectory basis functions are spawned. Computing it here makes the
spawning condition concrete, and it is one of the two vectors that span the **branching space**
of the conical intersection from Example 7.

The calculation runs RASSCF, then **ALASKA** with the NAC request (`NAC = 1 2`), which
assembles the coupling from its individual contributions and prints the final vector.

## How to run

A SLURM job file (`08_Nonadiabatic_Coupling.joc`) is provided inside this directory. It loads `openmolcas/26.02`
and runs `pymolcas` on this example's input:

```bash
sbatch 08_Nonadiabatic_Coupling.job
```

---

## Interpreting the output

ALASKA prints several intermediate blocks (Renormalization, Kinetic, Nuclear Attraction,
Two-electron, CSF, CI derivative coupling). These are the **pieces** that add up to the coupling
— you do not need to read them individually. The result is the last block:

### 1. The Total derivative coupling vector

```
 *      Total derivative coupling       *

              X                 Y               Z
  C1    2.94868809E-01   -1.23307975E-02    ~0
  C2    2.94868809E-01   -1.23307975E-02    ~0
  H3   -1.12863913E-02   -3.26720372E-03    ~0
  H4   -1.09759006E-02    4.19865034E-03    ~0
  H5   -1.09759006E-02    4.19865034E-03    ~0
  H6   -1.12863913E-02   -3.26720372E-03    ~0

              norm:  0.4180
```

This is a **3N vector** (x, y, z on each atom) describing the direction in nuclear space along
which S₀ and S₁ are coupled. How to read it:

- **Magnitude (norm = 0.4180 a.u.)** — the overall strength of the S₁–S₀ coupling at this
  geometry. It is moderate here because this is a geometry *away* from the crossing; the norm
  **grows sharply** as you approach the MECI (Example 7) and formally diverges at the seam.
- **Which atoms dominate** — the two **carbons carry essentially all the coupling**
  (|d| ≈ 0.295 each), while the hydrogens contribute little (|d| ≈ 0.012 each). The coupling is
  concentrated on the **C=C unit**.
- **Direction** — the large components are along **x on C1 and C2 with the same sign**, i.e. the
  coupling points along the **C=C stretch / pyramidalization coordinate** that carries ethylene
  toward its twisted-pyramidalized intersection.
- **Planarity** — all **z-components are ~10⁻¹² (zero)**, so the coupling vector lies entirely
  **in the molecular plane**, as required by symmetry at this planar geometry.

### 2. The energy difference

```
 Energy difference: -3.801055E-01
```

The S₁–S₀ gap at this geometry is **0.380 hartree ≈ 10.3 eV** — a large gap, confirming this is
the **planar Franck–Condon-type geometry**, far from any crossing. (Compare: at the MECI in
Example 7 this difference is ~0.) The derivative coupling and the energy gap are the two
ingredients of the spawning probability: large coupling **and** small gap → spawn.

> Note: the bare **CI derivative coupling** block (printed earlier) is the term that, divided by
> the energy gap, gives the dominant contribution to the total coupling. This is why the total
> coupling blows up as the gap closes near a conical intersection.

---

## Connecting to the dynamics and to Example 7

- This vector is one of the **two branching-space vectors** (with the gradient difference) that
  define the conical intersection topography in Example 7. There, the same information appears as
  the `Local y` (derivative-coupling) direction.
- In AIMS, the **magnitude of this coupling along the trajectory is the spawning trigger**:
  evaluate it at the planar FC geometry (here, norm 0.418) and then near the MECI to *see* it
  grow — that growth is what makes the dynamics spawn new basis functions.

## Notes

- **State-averaging over both states** (`ciroot = 2 2 1`, with S₀ and S₁ in the average) is
  required — the analytic NAC needs a balanced description of both states.
- **This example is CASSCF only — and intentionally so.** OpenMolcas computes the derivative
  coupling **analytically only at the CASSCF level**; there is no analytic CASPT2 NAC, and the
  numerical fallback is not implemented (ALASKA returns *"Numerical nonadiabatic coupling not
  implemented"* if you place a `&CASPT2` step before it). This is fine in practice: the coupling
  **direction** is far less sensitive to dynamic correlation than the energies are, so AIMS-type
  dynamics typically uses **CASSCF derivative couplings** together with CASPT2-corrected
  **energies**. Use this CASSCF NAC as-is and refine the energies separately where needed.
- The coupling vector is defined only **up to a sign** (and, between runs, an overall phase);
  compare magnitudes and relative directions, not absolute signs.
- Re-run at the **MECI geometry from Example 7** to watch the norm grow by orders of magnitude as
  the S₁–S₀ gap collapses — a compelling, concrete illustration of why population transfers there.
