# Example 6 — Photoelectron Spectrum of Uracil via Dyson Orbitals 

## What this calculation does

This example computes the **valence photoelectron (photoionization) spectrum** of uracil —
ionization energies and relative band intensities — using **Dyson orbitals** in OpenMolcas,
for **both the ground state (S₀) and an excited state (S₁)** in a single run. The method
follows the multireference Dyson-orbital approach of Tenorio, Ponzi, Coriani & Decleva
(*Molecules* **27**, 1203 (2022), [doi:10.3390/molecules27041203](https://doi.org/10.3390/molecules27041203)); the nucleobase active space
follows the thiouracil Dyson study of Ruckenbauer, Mai, Marquetand & González 
*J. Chem. Phys.* **144**, 074303 (2016),  [doi:10.1063/1.4941948](https://doi.org/10.1063/1.4941948).

The workflow:

1. **GATEWAY / SEWARD** — molecule, basis (**aug-cc-pVDZ**; diffuse functions are essential
   for ionization), RICD integrals.
2. **SCF** — closed-shell neutral starting orbitals.
3. **RASSCF (neutral)** — CAS(14,10), **state-averaged over 3 singlets** (S₀, S₁, S₂);
   → `JOB001`.
4. **RASSCF (cation)** — CAS(13,10), state-averaged over 20 doublets; → `JOB002`.
5. **RASSI with `DYSOn`** — couples the selected neutral state(s) to each cation state,
   producing a **Dyson orbital** and its **squared norm (pole strength / intensity)** for
   every ionization.

### The RASSI selection line — IMPORTANT

```
NrOfJobIphs = 2 2 20      ! 2 files: take 2 states from JOB001 (neutral), 20 from JOB002 (cation)
1 2                       ! neutral states to ionize FROM:  1 = S0,  2 = S1
1 2 3 ... 20              ! the 20 cation states to ionize INTO
SUBSets = 2               ! neutral and cation are two separate orbital subsets
EJob                      ! take state energies from the wavefunction files
DYSOn                     ! compute Dyson orbitals
DYSExport                 ! export Dyson orbitals to MOLDEN
CIPRint
```

By listing **`1 2`** you ask RASSI to build Dyson orbitals from **both** S₀ and S₁. This is
what lets a single job give you the ground-state PES *and* the excited-state (pump–probe)
PES at once.

## How to run

A SLURM job file (`06_Photoelectron_Spectrum.job`) is provided inside this directory. It loads
`openmolcas/26.02` and runs `pymolcas` on this example's input:

```bash
sbatch 06_Photoelectron_Spectrum.job
```

> **NOTE**: A complete reference output (`06_Photoelectron_Spectrum.out`) is included in this directory
> in case the calculation takes too long to finish during the session — you can read the results and
> perform analysis (discussed below) without waiting for your own job.

---

## ⭐ How to read the results — ground state vs excited state

This is the central skill for this example. RASSI assembles its internal state list from the
**selected** states only, in the order you listed them:

| RASSI index | What it is |
|-------------|------------|
| **1** | neutral **S₀** (first neutral you selected) |
| **2** | neutral **S₁** (second neutral you selected) |
| **3 – 22** | the 20 **cation** doublets |

So in the **Dyson amplitudes table**, the **"From"** column tells you which neutral state is
being ionized:

> **From = 1  →  GROUND-STATE photoelectron spectrum (ionizing S₀)**
> **From = 2  →  EXCITED-STATE photoelectron spectrum (ionizing S₁)**

The **"To"** column is the cation state (RASSI index; subtract 2 to get the cation root
number). **BE (eV)** is the binding energy = E(cation) − E(neutral state being ionized).
**Dyson intensity** is the squared Dyson-orbital norm ∝ relative band intensity.

### Ground-state PES — read the `From 1` rows

```
 From  To   BE (eV)   Dyson intensity
   1    3    8.844     0.9903     <- first ionization (HOMO), strong
   1    4   11.571     0.9938
   1    5   11.795     0.9942
   1    6   13.062     0.9936
   1    7   13.496     0.9851
   1    8   15.117     0.9912
```

The first ionization energy from S₀ is **8.84 eV** (experimental vertical IE of uracil is
~9.5 eV — CASSCF underestimates slightly, as expected without dynamic correlation). The
**Dyson norms are all ≈ 0.99**, meaning each of these is a clean **one-electron ionization**
(a single valence orbital is emptied) — exactly what you expect for the outer-valence band of
a closed-shell molecule. Higher cation states (To = 9, 11, …) appear with tiny norms (~10⁻⁴)
— these are shake-up / correlation satellites that borrow little intensity.

### Excited-state PES — read the `From 2` rows

```
 From  To   BE (eV)   Dyson intensity
   2    3    3.234     0.4924     <- lowest-BE ionization from S1
   2    9   12.514     0.4136
   2   10   13.310     0.6867
   2   13   14.007     0.6035
   2   16   14.533     0.4157
   2   17   14.594     0.3636
   2   18   14.928     0.6187
   ...
```

Ionizing from **S₁**, the **binding energies are much smaller** (the lowest is **3.23 eV**),
because S₁ already sits ~5.6 eV above S₀ — the molecule is part-way to threshold. The
intensity is also **spread over many cation states** with moderate norms (0.3–0.7) rather than
concentrated in a few near-unity peaks, reflecting the more complex (open-shell, mixed)
character of ionizing an excited state. This **lowering and reshaping** of the spectrum upon
excitation is precisely the signal exploited in **time-resolved photoelectron spectroscopy
(TR-PES)** to follow excited-state dynamics.

### The consistency check (why this is correct)

The two spectra are tied together by the excitation energy:

```
BE(from S0) = BE(from S1) + E(S1)
   8.844 eV  ≈   3.234 eV  +  5.61 eV   ✓
```

For the lowest cation state, ionizing S₀ directly (8.844 eV) equals ionizing S₁ (3.234 eV)
plus the S₁ excitation energy (5.61 eV). This near-exact match confirms both the state
indexing and the physics — the excited-state spectrum is genuinely the ground-state spectrum
shifted down by the excitation energy (for the states that correlate).

> The neutral **S₁ vertical excitation energy** itself is read from the spin-free table:
> SF state 2 lies **5.61 eV** above SF state 1.

---

## Building the spectra

For each source state, plot **Dyson intensity (y)** against **BE (eV) (x)** and convolve each
line with a Gaussian/Lorentzian (~0.4–0.5 eV FWHM):

- Use the **`From 1`** rows for the **ground-state** photoelectron spectrum.
- Use the **`From 2`** rows for the **S₁ excited-state** spectrum.

A ready-made script, `plot_pes.py`, is provided in this directory.
Run `python plot_pes.py 06_Photoelectron_Spectrum.out` to parse the Dyson table and 
produce the spectrum automatically (blue = ground state, red = excited state; 
use `--separate` for two panels or `--fwhm` to change the broadening).

RASSI exports each Dyson orbital to MOLDEN (`*.dys.molden.SF.*`) — open them to **see** the
orbital each ionization removes an electron from.

---

## Variations

- **Only the ground-state PES:** list a single neutral state in RASSI (`NrOfJobIphs = 2 1 20`
  then `1`), and you can even compute just one neutral root (`ciroot = 1 1 1`).
- **Ionize from S₂ as well:** the neutral RASSCF already has 3 roots (`ciroot = 3 3 1`), so
  add `3` to the neutral selection list (`1 2 3`) and read the `From 3` rows.
- **Quantitative IEs:** add **(XMS-)CASPT2** after each RASSCF and let `EJob` read the
  corrected energies — this shifts the ground-state first IE from 8.84 eV toward the
  experimental ~9.5 eV. The Dyson **norms** (intensities) barely change, so CASSCF already
  gives the correct intensity pattern; CASPT2 only refines the **energy axis**.

---

## Notes

- **Diffuse functions (aug-cc-pVDZ)** matter: ionization removes an electron toward the
  continuum, so the outer-valence orbitals need a diffuse description.
- The neutral and cation **must share the same inactive/active partition** (Inactive = 22,
  Ras2 = 10) for the Dyson orbitals to be well defined.
- **20 cation roots** is demanding; if it is too slow, reduce to 6–10 roots (covers the bright
  outer-valence bands) — only a handful carry significant Dyson intensity anyway.
- This is the **valence** analogue of the core-hole **XPS** example (RAS1 core + HEXS +
  XMS-RASPT2), prepared separately.
