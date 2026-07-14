# Example 4 — Oscillator Strengths & NTOs

## What this calculation does

This example computes the **vertical excitation energies**, **oscillator strengths**, and
**Natural Transition Orbitals (NTOs)** for the S₀ → S₁ and S₀ → S₂ transitions of ethylene
at the **SA-3-CAS(2,2)SCF/6-31G\*** level.

The workflow:

1. **GATEWAY / SEWARD** — molecule, basis (6-31G\*), RICD integrals.
2. **SCF** — closed-shell starting orbitals.
3. **RASSCF** — state-averaged CAS(2,2) over 3 singlet roots. The wavefunction file is
   copied to **two** `JobIph` files (`JOB001`, `JOB002`) so RASSI can treat a "bra" set and
   a "ket" set.
4. **RASSI** — computes transition densities, dipole transition moments, oscillator
   strengths, and NTOs between the selected states.

### Reading the RASSI selection line

```
NR OF JOBIPHS = 2 1 2     ! 2 JobIph files: take 1 state from the first, 2 from the second
1                         ! from JOB001: state 1   (the ground state, S0)
2 3                       ! from JOB002: states 2 and 3  (S1 and S2)
```

So RASSI evaluates the transitions **1→2 (S₀→S₁)** and **1→3 (S₀→S₂)**.

Keywords:
- **`TRDI`** / **`TRD1`** — compute the transition density matrices / one-particle
  transition densities (needed for dipole moments and NTOs).
- **`NTOC`** — generate **Natural Transition Orbitals** for each transition.
- **`EJob`** — take state energies from the wavefunction (`JobIph`) files.

## How to run

A SLURM job file (`04_Oscillator_Strengths_NTOs.job`) is provided inside this directory. It loads
`openmolcas/26.02` and runs `pymolcas` on this example's input:

```bash
sbatch 04_Oscillator_Strengths_NTOs.job
```

---

## Interpreting the output

### 1. Excitation energies — "Spin-free section"

```
 SF State    Rel lowest level(eV)    cm**(-1)
     1            0.0000                 0.0
     2           10.3408             83404.2     <- S0 -> S1
     3           15.5674            125559.9     <- S0 -> S2
```

These are the **vertical excitation energies** (energy of each excited state relative to
the ground state). At SA-CASSCF the S₁ of ethylene comes out very high (~10 eV) because
the bare CAS(2,2) lacks dynamic correlation — this is exactly why the production dynamics
adds **(XMS-)CASPT2** on top. 

### 2. Oscillator strengths — "Dipole transition strengths"

```
   From  To    Osc. strength
     1   2     5.767E-01     <- S0 -> S1, bright
     2   3     6.470E-01     <- S1 -> S2 (excited-to-excited), bright
```

You see 1→2 and 2→3 because those are bright (f ≈ 0.58 and 0.65). The **1→3 (S₀→S₂)** row is missing 
because its oscillator strength is **below 1×10⁻⁵** — i.e., S₀→S₂ is essentially **dark** at this geometry and level.

> **On CASSCF vs CASPT2 oscillator strengths:** the **transition moments are always
> CASSCF-quality** — RASSI builds them from the RASSCF wavefunctions. What CASPT2 changes
> is the **energy** ΔE. If you add a `&CASPT2` step and feed RASSI the CASPT2 energies
> (`EJob` reads them from the `JobMix`), the same |μ|² is rescaled by the corrected ΔE,
> giving the CASPT2-level `f = (2/3) · ΔE · |μ|²`. 

### 3. Length vs velocity gauge

RASSI prints `f` in both the **length** (dipole) and **velocity** gauges and flags large
differences:

```
   From To   Difference (%)  Osc.(len.)  Osc.(vel.)
     1   2     11.5           0.577       0.517
     2   3     40.3           0.647       0.461
```

For an **exact** wavefunction the two gauges agree. A gap signals **basis-set / active-space
incompleteness**. The 6-31G\* basis with a CAS(2,2) is small, so an 11–40% spread is
expected and is the program telling you the description is approximate. 

### 4. Natural Transition Orbitals (NTOs)

NTOs re-express an electronic transition as a small number of **hole → particle** orbital
pairs, each with a weight (eigenvalue). They answer "**which orbitals does this excitation
actually move an electron between?**" far more compactly than listing CI coefficients.

From your output, **S₀→S₁ (1→2)**:

```
   EIGENVALUE   CONTRIBUTION(%)   HOLE NTO   PARTICLE NTO
   0.48137         96.27           a  9        a  9
   0.01863          3.73           a  8        a  8
```

One hole→particle pair dominates (**96%**) — a clean, essentially single-configuration
π → π\* excitation. The dominant eigenvalue (0.481, out of a normalized sum of 0.5 per spin
for a singlet) tells you the transition is well described by a single NTO pair.

**S₀→S₂ (1→3)**:

```
   EIGENVALUE   CONTRIBUTION(%)   HOLE NTO   PARTICLE NTO
   0.03587         50.00           a  9        a  9
   0.03587         50.00           a  8        a  8
```

Here the transition is split **50/50** over two NTO pairs — a **doubly-excited /
multiconfigurational** character, *not* a simple one-orbital promotion. The small
eigenvalues (0.036) also reflect weak one-particle transition character.

RASSI writes the NTOs as MOLDEN files (`*.nto.molden.SF.*.HOLE` / `.PART`); open them in a
viewer to **see** the hole and particle orbitals for each transition.

## Notes

- Copying the `JobIph` to both `JOB001` and `JOB002` is a convenient trick to let RASSI form
  bra/ket transition pairs from a single SA-RASSCF run.
- `Group = NoSym` keeps everything in one symmetry block — simplest for RASSI.
- To obtain CASPT2-level oscillator strengths, insert a `&CASPT2` step before RASSI and
  copy its `JobMix` so `EJob` reads the corrected energies.
