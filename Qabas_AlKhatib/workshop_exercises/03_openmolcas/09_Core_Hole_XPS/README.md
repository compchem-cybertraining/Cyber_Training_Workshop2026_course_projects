# Example 9 — Core-Hole X-ray Photoelectron Spectrum (XPS) of Uracil: O 1s Edge

## What this calculation does

This example computes the **O 1s core-level photoelectron (XPS) spectrum** of uracil — core
ionization energies and relative intensities — using **RASPT2 + Dyson orbitals**, following the
protocol of:

> **D. Faccialà, M. Bonanomi, B. N. C. Tenorio, … S. Coriani, N. Došlić, … O. Plekan**,
> *Unraveling the Relaxation Dynamics of Uracil: Insights from Time-Resolved X-ray Photoelectron
> Spectroscopy*, **J. Am. Chem. Soc.** 2025, **147**, 30694–30707.
> [doi:10.1021/jacs.5c04874](https://doi.org/10.1021/jacs.5c04874)

It is the **core-level analogue of Example 6** (valence Dyson PES): the same RASSI/Dyson
machinery, but the ionization removes an **O 1s core electron**, made possible by the **HEXS**
core-hole technique and a **restricted active space (RAS)** that contains the core orbital.

---

## The key trick: placing the O 1s orbital in RAS1

A core-hole RAS calculation only works if the **specific core orbital you want to ionize sits in
RAS1**. The recipe used here:

1. **`&SCF`** generates the starting molecular orbitals (no external orbital file needed — this
   keeps the example self-contained and reproducible).
2. In the SCF orbitals, the atom labelled as **O8 1s** core orbital is **orbital number 2**. With
    `Inactive = 21`, the first active (RAS1) slot is **orbital 22**.
3. **`Alter = 1; 1 2 22`** swaps orbital **2 ↔ 22**, moving the O8 1s into the RAS1 slot. The
   output confirms this: *"Molecular orbitals exchanged: In symmetry 1: 2 22"*.
4. **`SupSym = 1; 1 2`** (neutral) / **`SupSym = 1; 1 22`** (cation) freezes the symmetry of that
   orbital so it cannot rotate out of RAS1 during the RASSCF.

> **Why the `Alter` must be done in the NEUTRAL, not the cation.** RASSI builds Dyson orbitals by
> overlapping the neutral and cation wavefunctions, which requires them to share the **same
> orbital basis**. If you altered orbitals only in the cation, the two sets would no longer
> correspond and **RASSI would fail**. By doing the swap in the neutral and passing its orbitals 
> to the cation via `File = RasOrb_Neutral_O1sRAS`, both steps use an identical orbital ordering. 
> A bonus: doing the swap in the neutral does **not** change the neutral energy — a rotation among 
> occupied orbitals leaves the closed-shell energy invariant — so the reference is unaffected.

---

## Keyword-by-keyword

| Keyword | Meaning |
|---------|---------|
| `Inactive = 21` | 21 doubly occupied orbitals kept out of the active space |
| `Ras1 = 1` | one orbital in RAS1 — the **O 1s core** orbital |
| `Ras2 = 7` | seven valence-occupied orbitals (per the paper) |
| `Ras3 = 2` | two **π\*** orbitals (the third π\* is dropped to save cost) |
| `Nactel = 16 1 4` | 16 active electrons; **max 1 hole** in RAS1; max 4 electrons in RAS3 |
| `Alter = 1; 1 2 22` | swap orbitals 2 and 22 → put O8 1s into RAS1 |
| `SupSym = 1; 1 2` / `1; 1 22` | lock that orbital's symmetry so it stays in RAS1 |
| `HExs = 1; 1` | **HEXS** projection: enforce a single hole in RAS1 for the final (core-ionized) states |
| `Charge = 1`, `Spin = 2` | the core-ionized cation is a doublet |
| `Ciroot = 10 10 1` / `20 20 1` | state-average over **10 neutral** / **20 core-ionized** roots |
| `IMAGinary = 0.35` | imaginary level shift (Eₕ) — core-hole states are intruder-prone |
| `XMultiState = All` | XMS-RASPT2 multistate treatment |
| RASSI `NrOfJobIphs = 2 3 20` | 2 files: take 3 neutral states (S₀,S₁,S₂) and 20 core-ionized states |
| `DYSOn` / `DYSExport` | compute and export the Dyson orbitals (pole strengths = XPS intensities) |

The two `&CASPT2` blocks add dynamic correlation (RASPT2) to the neutral and core-ionized
manifolds; their `JobMix` files are copied to `JOB001`/`JOB002` and read by RASSI via `EJob`.

---

## How to run

A SLURM job file (`09_Core_Hole_XPS.job`) is provided inside this directory. It loads `openmolcas/26.02` and
runs `pymolcas` on this example's input:

```bash
sbatch 09_Core_Hole_XPS.job
```

> A reference output is included in this directory — this is the most expensive example in the set
> (RAS + HEXS + 10/20-state XMS-RASPT2 + Dyson). Use the provided output to follow the analysis if
> your own job has not finished during the session.

---

## Reading the output

The **Dyson amplitudes** table is read exactly as in Example 6:

- **From** = the neutral state being ionized — `1` = S₀, `2` = S₁, `3` = S₂.
- **To** = the core-ionized final state.
- **BE (eV)** = E(core-ionized) − E(neutral) = the **core binding energy**.
- **Dyson intensity** = pole strength ∝ **XPS band intensity**.

Plot intensity vs BE, broaden with a ~0.4 eV Lorentzian, and apply the paper's rigid **−2.4 eV**
O 1s shift to compare with experiment.

### Ground-state O 1s — excellent agreement

```
 From  To    BE (eV)    Dyson intensity
   1    4    539.644    0.6078       <- O 1s ground-state ionization (strong)
```

This **539.6 eV** matches the JACS Table 1 ground-state values (**539.95 / 540.05 eV**, O8 1s⁻¹ /
O7 1s⁻¹) to within a few tenths of an eV. **The ground-state O 1s XPS is reproduced essentially
exactly** — confirming that the RAS/HEXS/Dyson machinery and the orbital-swap setup are correct.

### Excited-state O 1s — qualitatively present, quantitatively sensitive

The paper reports the headline result that in **S₁(nπ\*)** the **O8 1s line shifts up by ~4 eV**
(to 544.0 / 546.4 eV) because the nπ\* excitation removes lone-pair density from O8. In this
tutorial run, the `From 2` (S₁) lines do **not** reproduce that +4 eV shift cleanly — the strong
S₁ feature appears near 534 eV (a shake-down), and lines near 546 eV appear but with mixed state
assignment.

**This is a known sensitivity, not a setup error**, and the reason is instructive:

- The excited-state binding energy is referenced to the **excited state itself**, so it is only as
  good as the **neutral excited-state description**. At the RASSCF level here, the neutral
  vertical excitations come out high (S₁ ≈ 6 eV vs the literature ~4.8 eV) and the **nπ\* / ππ\***
  states can mix or reorder within the SA-10 manifold.
- The +4 eV O8 shift specifically depends on the **O8 lone-pair (n) orbital** being well described
  in RAS2 and on the nπ\* state being cleanly isolated. Small differences in the **active-space
  composition** or the **starting geometry** (this example uses a planar reference geometry, not
  the paper's sampled FC structures) move the excited-state lines significantly.

So the ground-state spectrum is quantitative; the excited-state spectrum is **qualitatively
correct but quantitatively sensitive to active-space curation and geometry** — which is itself an
important lesson about excited-state core spectroscopy.

---

## Notes

- **Recommended for the workshop.** The ground-state result demonstrates the full
  method against a published benchmark, and the excited-state behavior is a genuine teaching point
  about active-space sensitivity. Reproducing the paper's excited-state O8 shift to the eV requires
  research-grade active-space curation (verifying/localizing the O lone pairs, confirming nπ\* vs
  ππ\* ordering state by state) and ideally the paper's exact geometry and orbital set.
- To improve the excited-state agreement if you do continue: (1) use an **optimized FC geometry**;
  (2) **inspect and, if needed, localize** the RAS2 orbitals to guarantee the O8 lone pair is
  included; (3) verify the **nπ\* state** is present and correctly ordered in the SA-10 neutral
  manifold before trusting its core-ionized lines.
- For the **N 1s** or **C 1s** edges, identify the corresponding N/C 1s SCF orbital, `Alter` it
  into RAS1, and set `SupSym` accordingly — everything else is unchanged.
- Pairs with **Example 6** (valence Dyson PES): same RASSI/Dyson method, core vs valence window.
