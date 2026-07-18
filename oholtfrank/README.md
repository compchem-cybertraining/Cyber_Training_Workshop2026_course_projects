# Virtual Photon Decay in Ar·H₂O — complete ab initio mechanism

**Owen Holtfrank** · Georgia Institute of Technology (Kretchmer group) ·
CyberTraining 2026 course project

---

## Main initial finding.

An Ar 3s⁻¹ hole at 28.97 eV that is unable to decay by ICD,
transfers its energy by virtual photon to a **bright superexcited state of water
at 13.3 eV**. This state has been found to be bound, not directly dissociaive, with
fragmentation accessible through nonadiabatic internal conversion in the excited state
manifold onto the Ã surface, which ejects an H atom in 16.4 fs. This contrasts with 
golden-rule expressions for energy transfer into a dissociative continuum (VPD), with
the state populated at resonance unable to dissociate. Therefore the single-step fragmentation
is unable to be characterized and requires the inclusion of nonadiabatic dynamics.

| | value | method |
|---|---|---|
| donor | **28.97 eV** | XMS-CASPT2 (expt. 29.24) |
| acceptor | **13.30 eV**, f = 0.157 | SA-20-CASSCF + RASSI |
| coupling | **12.37** vs **9.35 meV** | two independent routes, 30% agreement |
| acceptor is bound | **0.1%** fragmentation | classical, on the computed surface |
| cascade | **225 spawns**, 16/16 trajectories, funnel to Ã | **AIMS (PySpawn)** |
| fragmentation | **31.4%**, 16.4 fs, H at 0.60 eV | classical, on the Ã surface |

---

## Workshop software: adoption, integration, future use

**Three workshop codes, integrated into one workflow**

```
OpenMolcas ──> surfaces, oscillator strengths, XMS-CASPT2 validation
     │
     └──> PySpawn ──> AIMS cascade (the central result)
     │         ↑
     │         └── patched for more than ten states.
     │
Libra ──────> NE-FGR attempt (documented negative) + analysis environment
```

### Patched PySpawn to make this project possible

`molcas_interface.py::get_overlap` **hard-exits above 10 singlet states.** Working
with claude code we developed a token-based parser that slices the cross-geometry block,
which was verified against a 13-state run (needed for this project)

→ [`project/patches/molcas_interface_get_overlap.diff`](project/patches/)

### Continued use

This connects directly to my research:

- **A JCP lett manuscript in preparation** (Ar_n·H₂O ICD/ETMD competition) — the
  fragmentation KER answers the "observable signature" question my candidacy
  committee raised.
- **A follow-up study** — Experimental collaborators (Orlando) have
  expressed interest in VPD in this system, so this is part of a future investigation.

---

## Repository

```
oholtfrank/
├── README.md          <- you are here
├── coursework/        <- workshop tutorials → what fed the project
└── project/           <- the VPD mechanism
    ├── README.md      <- full writeup: mechanism, rigor tiers, open items
    ├── 01_electronic_structure/    donor 28.97 eV; ICD Coulomb-blocked
    ├── 02_acceptor_identification/ state 12: resonant + bright + falling
    ├── 03_surfaces/                1D + 2D scans; the acceptor is flat/rising
    ├── 04_rates_and_coupling/      12.37 vs 9.35 meV; NE-FGR negative
    ├── 05_aims/                    the cascade — 225 spawns, funnel to Ã
    ├── 06_fragmentation/           bound acceptor (0.1%); Ã (31.4%, 16.4 fs)
    ├── 07_caspt2_validation/       13.344 eV — ordering and resonance survive
    └── patches/                    the get_overlap fix
```

**Start here:** [`project/README.md`](project/README.md) for the science,
[`coursework/README.md`](coursework/README.md) for the workshop through-line.

### Reproducing

Every figure is built from committed `.npz` files — no raw `.out` needed.
Raw AIMS trajectories (16 × `sim.hdf5`, ~295 MB) are archived separately
[Zenodo DOI: TBD]; `05_aims/results/cascade_stats.txt` holds the extracted
statistics. Environment and SLURM settings are documented in
[`coursework/README.md`](coursework/README.md#environment-notes).

## Programs used

**OpenMolcas** (SA-CASSCF, XMS-CASPT2, RASSI) · **PySpawn** (AIMS, patched) ·
**Libra** (NE-FGR, analysis env) · NumPy/SciPy/matplotlib · SLURM ·
PySCF/`pyscf_prism` (tetrazine benchmarking coursework)
