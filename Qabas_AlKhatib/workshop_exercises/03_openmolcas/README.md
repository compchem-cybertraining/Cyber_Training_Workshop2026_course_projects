<div align="center">

# 🧪 OpenMolcas Tutorial — Electronic Structure for Nonadiabatic Dynamics

### Multiconfigurational Calculations for Excited-State Photochemistry

[![Workshop](https://img.shields.io/badge/CyberTraining-2026-1f6feb?style=flat-square)](https://compchem-cybertraining.github.io/Cyber_Training_Workshop_2026/)
[![OpenMolcas](https://img.shields.io/badge/OpenMolcas-26.02-orange?style=flat-square)](https://gitlab.com/Molcas/OpenMolcas)
[![Methods](https://img.shields.io/badge/Methods-CASSCF%20%7C%20CASPT2%20%7C%20RASSI-2ea44f?style=flat-square)](https://gitlab.com/Molcas/OpenMolcas)
[![Companion](https://img.shields.io/badge/Companion-PySpawn%20Tutorial-1f6feb?style=flat-square)](https://github.com/compchem-cybertraining/Tutorials_PySpawn)

*Hands-on electronic-structure examples for the **Multiple Spawning with PySpawn** session —
CyberTraining Summer School & Workshop, University at Buffalo, 2026.*

</div>

---

## 📖 Overview

This repository is a graded set of **OpenMolcas** examples that build the electronic-structure
foundation needed for **nonadiabatic (excited-state) dynamics**. Together they walk from a
ground-state geometry and Hessian, through excitation energies, couplings, and spectroscopy, to
the **conical intersection** that ab initio multiple spawning (AIMS) trajectories funnel through
— the same physics demonstrated in the companion
[**PySpawn tutorial**](https://github.com/compchem-cybertraining/Tutorials_PySpawn).

Each example is a **self-contained folder** with:

- a ready-to-run OpenMolcas input file,
- a focused **README** explaining what the calculation does, how to run it, and how to read the output,
- a SLURM job file (`*.job`),
- a reference output in some cases, so you can follow the analysis even if a long job hasn't finished.

> 🧬 The recurring test molecule is **ethylene** (for continuity with the AIMS tutorial), with
> **thioformaldehyde** and **uracil** used where heavier atoms or larger systems are needed.

---

## 🗂️ Examples

| # | Directory | What it computes | Key modules |
|---|-----------|------------------|-------------|
| 1 | `01_Geometry_Hessian_DFT/` | **DFT (B3LYP) geometry + Hessian** → feeds the pySpawn Hessian converter | SCF/KSDFT · SLAPAF · MCKINLEY |
| 2 | `02_S0_Optimization_CASSCF/` | **S₀ minimum** at SA-CASSCF (+ Hessian) | RASSCF (RLXRoot 1) · SLAPAF |
| 3 | `03_S1_Optimization_CASPT2/` | **S₁ minimum** at XMS-CASPT2 (+ transition moments) | RASSCF → CASPT2 (RLXRoot 2) → ALASKA → SLAPAF → RASSI |
| 4 | `04_Oscillator_Strengths_NTOs/` | **Oscillator strengths** & **Natural Transition Orbitals** | RASSCF → CASPT2 → RASSI (NTO) |
| 5 | `05_Spin_Orbit_Coupling/` | **Spin–orbit coupling** (singlet–triplet) in H₂CS | RASSCF → RASSI-SO |
| 6 | `06_Photoelectron_Spectrum/` | **Valence photoelectron spectrum** of uracil (ground & excited state) | RASSCF → RASSI (Dyson) |
| 7 | `07_MECI_CASPT2/` | **S₁/S₀ minimum-energy conical intersection** | RASSCF → CASPT2 → SLAPAF (Ediff constraint) |
| 8 | `08_Nonadiabatic_Coupling/` | **Nonadiabatic derivative coupling vector** | SA-RASSCF → ALASKA (NAC) |
| 9 | `09_Core_Hole_XPS/` | **Core-hole X-ray photoelectron spectrum** (O 1s) of uracil | RAS + HEXS → RASPT2 → RASSI (Dyson) |

> 🔬 Example 9 (core-hole **XPS**) is the core-level analogue of Example 6 — same Dyson/RASSI
> machinery, but the ionization removes a **1s core electron** via the HEXS technique.

---

## 🧭 How the examples connect to the dynamics

```
   DFT Hessian (1) ─────────► pySpawn Wigner sampling ──► AIMS trajectories
        │
   S0 / S1 minima (2, 3) ───► where the wavepacket starts and relaxes
        │
   Oscillator strengths
   & NTOs (4) ──────────────► which states are bright (initial excitation)
        │
   Spin–orbit coupling (5) ─► intersystem-crossing channels
        │
   Nonadiabatic coupling (8) ► the spawning criterion in AIMS
        │
   MECI (7) ────────────────► the funnel the trajectories spawn around
        │
   Photoelectron /
   XPS spectra (6, 9) ──────► how the dynamics is probed experimentally
```

---

## ⚙️ Running an example

These examples were prepared with **OpenMolcas 26.02**, installed in `cyberwksp21/SOFTWARE_2026/` directory.
From inside any example folder:

```bash
sbatch *.job
```

The job sets the standard OpenMolcas environment (`MOLCAS_PROJECT`, a local scratch directory
`MOLCAS_WORKDIR=scr_<project>`, `MOLCAS_MEM`, MOLDEN output, …) and runs
`pymolcas <input>.in > <input>.out`, cleaning up scratch at the end. 

> 📂 Some folders ship a **reference output** — if a calculation runs long (the CASPT2, RASSI, and
> core-hole examples can), read the provided output and follow that folder's README directly.

---

## 🚀 Suggested order

1. **Start with geometry & references** — Examples 1–3 (DFT Hessian → S₀ → S₁) establish the
   structures and the CASSCF/CASPT2 workflow.
2. **Spectroscopy & couplings** — Examples 4–6 (oscillator strengths/NTOs, SOC, photoelectron).
3. **The dynamics-defining pieces** — Examples 7–8 (MECI and nonadiabatic coupling), which tie
   directly to the AIMS spawning picture.
4. **Core-level probe** — Example 9 (XPS), the experimental window onto the excited-state dynamics.

---

## 🔗 Resources

- **OpenMolcas:** [gitlab.com/Molcas/OpenMolcas](https://gitlab.com/Molcas/OpenMolcas) — see the
  `Test/` directory for many more sample inputs
- **OpenMolcas documentation:** [molcas.gitlab.io/OpenMolcas/sphinx](https://molcas.gitlab.io/OpenMolcas/sphinx/)
- **Companion dynamics tutorial:** [Tutorials_PySpawn](https://github.com/compchem-cybertraining/Tutorials_PySpawn)
- **Workshop page:** [CyberTraining 2026](https://compchem-cybertraining.github.io/Cyber_Training_Workshop_2026/)

---

## 🙏 Acknowledgement

Material prepared by [**Arshad Mehmood**](https://arshadmehmood118.github.io/) (Institute for Advanced Computational Science, Stony Brook
University) for the **PySpawn / OpenMolcas** session led with [**Professor Benjamin G. Levine**](https://levinegroup.org/).
This work is supported by the **NSF-OAC CyberTraining** program.

📧 **Post-workshop questions?** Email **[arshad.mehmood@stonybrook.edu](mailto:arshad.mehmood@stonybrook.edu)**.

<div align="center">

*Reference geometries are literature/experimental structures cited in each example's README —
verify against your own optimizations before quoting absolute numbers.*

</div>
