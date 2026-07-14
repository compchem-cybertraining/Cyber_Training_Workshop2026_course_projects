# Example 5 — Spin–Orbit Coupling of H₂CS via RASSI-SO

## What this calculation does

This example computes the **spin–orbit coupling (SOC)** between the low-lying singlet and
triplet states of **thioformaldehyde (H₂CS)** and the spin–orbit-corrected state energies,
at the **MS-CASPT2(12,10)/ANO-RCC-VQZP** level with the atomic mean-field integrals **AMFI** 
spin–orbit operator.

H₂CS is the standard small-molecule benchmark for intersystem crossing (ISC). This setup
reproduces the reference protocol of:

> **S. Mai, A. J. Atkins, F. Plasser, L. González**, *J. Chem. Theory Comput.* **15**,
> 3470–3480 (2019). [doi:10.1021/acs.jctc.9b00282](https://doi.org/10.1021/acs.jctc.9b00282)

The workflow:

1. **GATEWAY / SEWARD** — molecule, relativistic basis (**ANO-RCC-VQZP**), Second-order
   scalar-relativistic Douglas-Kroll-Hess (DKH2) Hamiltonian (`Relativistic = R02O02`), 
   and **AMFI** atomic mean-field SO integrals.
2. **SCF** — closed-shell starting orbitals.
3. **RASSCF + CASPT2 (singlets)** — SA-2 CAS(12,10), 2 singlet roots (S₀, S₁); MS-CASPT2;
   saved to `JOB001`.
4. **RASSCF + CASPT2 (triplets)** — SA-2 CAS(12,10), 2 triplet roots (T₁, T₂); MS-CASPT2;
   saved to `JOB002`.
5. **RASSI with `SPINorbit`** — builds and diagonalizes the spin–orbit Hamiltonian in the
   basis of the spin-free MS-CASPT2 states, returning the **SOC matrix elements (cm⁻¹)**
   and the **spin–orbit eigenstates**.

## How to run

A SLURM job file (`05_Spin_Orbit_Coupling.job`) is provided inside this directory. It loads
`openmolcas/26.02` and runs `pymolcas` on this example's input:

```bash
sbatch 05_Spin_Orbit_Coupling.job
```

---

## Interpreting the SO-RASSI states energies

The `Spin-orbit section` of the RASSI block reports the total energies including SO-coupling. 
This block contains energies of 8 states (instead of 4).

> **A triplet is not one state — it is three.** Each triplet has three spin projections
> (Mₛ = −1, 0, +1) that are degenerate without SOC but couple individually once SOC is
> turned on.

The spin-orbit Hamiltonian is built from **all spin components**:

| Spin-free states | Spin components | Count |
|------------------|-----------------|-------|
| 2 singlets (S₀, S₁) | 1 each | 2 |
| 2 triplets (T₁, T₂) | 3 each (Mₛ = −1, 0, +1) | 6 |
| **Total** | | **8** |

That is exactly your **8 SO-RASSI states**. The **spin-free section** still lists only
**4 states** (each triplet counted once); the **spin-orbit section** expands the triplets
into their three components and mixes everything.

You can see the degeneracy directly in the SO eigenvalues:

```
 SO State     cm⁻¹
   1            0.0      <- S0 (singlet)
   2,3      16107.9      <- T1 (3nπ*) components  ┐
   4        16111.7      <- T1 (3nπ*) component   ┘ split by only ~4 cm⁻¹ (zero-field splitting)
   5        18082.4      <- S1 (1nπ*, singlet)
   6,7      27993.5      <- T2 (3ππ*) components  ┐
   8        27994.0      <- T2 (3ππ*) component   ┘ split by <1 cm⁻¹
```

The tiny splittings of the triplet components (the **zero-field splitting**, here only a
few cm⁻¹) are the direct fingerprint of SOC acting within each triplet.

---

## Interpreting the Spin-Orbit section

### 1. Spin-free energies and state assignment

```
 SF State    Rel (eV)
     1        0.0000     S0
     3        1.9972     T1 (3nπ*)
     2        2.2420     S1 (1nπ*)
     4        3.4703     T2 (3ππ*)
```

> **Read the SF-state index carefully.** The header says *"index according to input order,
> order according to energy."* Input index 2 is the first **singlet** (S₁), index 3 is the
> first **triplet** (T₁). Because T₁ lies **below** S₁ in energy, the energy-ordered list
> shows state 3 before state 2 — this is physically correct, not a bug.

Comparison with the benchmark paper (vertical excitations):

| State | This calc | Mai 2019 MS-CASPT2(12,10) | Exp |
|-------|-----------|---------------------------|-----|
| T₁ (³nπ\*) | 2.00 eV | 2.00 eV | 1.80 eV |
| S₁ (¹nπ\*) | 2.24 eV | 2.25 eV | 2.03 eV |
| T₂ (³ππ\*) | 3.47 eV | 3.45 eV | — |
| **S₁–T₁ gap** | **0.25 eV** | 0.24 eV | 0.23 eV |
| **S₁–T₂ gap** | **1.23 eV** | ~1.2 eV | — |

Your energies should match the published benchmark essentially exactly — this is the correct,
high-quality result. Small variation is due to the slight difference in the coordinates. We are using
experimental standard gas phase microwave bond length and bond angles.

### 2. SOC matrix elements

The block *"Complex SO-Hamiltonian matrix elements over spin components"* lists each
⟨state, spin, Mₛ | Ĥ_SO | state, spin, Mₛ⟩ in **cm⁻¹** (Real, Imag, Absolute). The large
entries in your output:

| Coupling | |SOC| (cm⁻¹) |
|----------|-------------|
| T₂(³ππ\*) ↔ S₀ | 168.2 |
| T₂(³ππ\*) ↔ S₁(¹nπ\*) | 160.5 |
| within T₁ (³nπ\*) components | 155.1 |

The physically important one for ISC is **S₁(¹nπ\*) ↔ T₂(³ππ\*) ≈ 160 cm⁻¹**, in excellent
agreement with the paper's multireference value of **150–160 cm⁻¹** at the FC geometry.

> **Note the symmetry selection rule.** The **S₁(¹nπ\*) ↔ T₁(³nπ\*)** coupling is
> **symmetry-forbidden in C₂ᵥ** and comes out **≈ 0** in your matrix. This is why ISC in
> H₂CS proceeds preferentially through **T₂(³ππ\*)**, not T₁ — exactly the point made in the
> Mai 2019 paper. The vanishing S₁–T₁ SOC in your output is a feature of the molecule.

### 3. SO eigenstates and their composition

The *"Weights of the five most important spin-orbit-free states"* table shows how each SO
eigenstate is built from the spin-free states. In your output every SO state is **>99.9%**
a single spin-free state — i.e., **SOC mixing is very weak** at the FC geometry. This is the
electronic-structure reason H₂CS shows **little ISC after vertical excitation**: although the
SOC element (~160 cm⁻¹) is sizable, the **S₁–T₂ energy gap (1.2 eV) is too large** for
efficient mixing at the FC geometry. As the paper emphasizes, ISC only becomes efficient at
**stretched C–S geometries** where the gap collapses — so the static FC picture here is the
starting point, and the dynamics is what reveals the full story.

---

## Notes

- **Relativistic basis (ANO-RCC) + Douglas–Kroll + AMFI are all required** for meaningful
  SOC; a non-relativistic basis gives garbage couplings.
- The `SOCO = 0.0` keyword sets the energy-difference cutoff for which SOC elements are
  computed to zero, i.e. **all** pairs are included.
- The SOC magnitude is fairly insensitive to dynamic correlation; the **energies/gaps** are
  what require MS-CASPT2. CASSCF SOC values would be similar, but CASSCF gaps would be worse.
- For the ISC dynamics itself, the SOC and gaps must be evaluated **along the C–S stretch**,
  not only at the FC point (see Figs. 3–4 of the paper).
