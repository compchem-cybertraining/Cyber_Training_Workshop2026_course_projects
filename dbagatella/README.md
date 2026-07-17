# Non-Adiabatic Molecular Dynamics (NAMD) of the Graphene-TiO2 Interface
**Cyber-Training Workshop on Computational Chemistry 2026 — Course Project**
**Author:** Diana Marlén Castañeda Bagatella (PhD Student, NJIT)
**Principal Investigator / Advisor:** Dr. Farnaz Shakib
**Supercomputing Resource:** PSC Bridges-2 (ACCESS / XSEDE)

---

## Project Overview
Ultrafast interfacial charge transfer in hybrid architectures combining two-dimensional (2D) materials and transition metal oxides is critical for advancing solar energy conversion and photocatalytic devices. This repository contains the complete, reproducible computational workflow and analytical scripts developed to study the electron-phonon coupling and non-radiative relaxation dynamics at the **Graphene-TiO2 interface** using an integrated **VASP-Libra** framework.

This project was developed as part of the **2026 Cyber-Training Workshop on Computational Chemistry** at the University at Buffalo. Preliminary exploratory runs (400 fs) presented here serve as the methodological foundation and proof-of-concept for our ongoing large-scale production simulations (1640 frames) on the Bridges-2 supercomputer.

---

## Repository Structure
```text
Proyecto_SummerSchool_Buffalo/
├── README.md               # Top-level project documentation (this file)
├── inputs/                 # VASP input parameters (INCAR, POSCAR, KPOINTS)
│   ├── README.md           # Guide to input parameters and active space setup
│   ├── INCAR_MD            # Input for NVT Born-Oppenheimer Molecular Dynamics (300 K)
│   ├── INCAR_static_example# Input for step-by-step static DFT wavefunction extraction
│   ├── POSCAR_example      # Initial atomic coordinates of the Graphene-TiO2 heterostructure
│   ├── KPOINTS             # Monokhorst-Pack k-point mesh configuration
│   └── POTCAR_LICENSE_NOTE # Explanation of PAW pseudopotential assembly (Ti_pv, O, C)
├── scripts/                # Automated NAMD workflow scripts (Python & Bash)
│   ├── README.md           # Step-by-step guide on how to execute the NAMD pipeline
│   ├── step1-LIBRA.py      # Extracts time-overlap matrices S(t, t+dt) from VASP WAVECARs
│   ├── step2_libra.py      # Performs phase correction and builds vibronic Hamiltonian (H_vib)
│   ├── generate_eng_plots.py # Automated plotting tool for publication-grade figures (300 DPI)
│   └── run_slurm_extraction.sh # Slurm submission script for high-performance computing
└── figures/                # Publication-quality plots and heatmaps
    ├── README.md           # Physical explanation of generated figures
    ├── energy_profiles.png # Active space electronic energy fluctuations at 300 K
    ├── nac_heatmap.png     # Time-averaged Non-Adiabatic Coupling (NAC) matrix heatmap
    └── state_populations.png # FSSH real-time electronic state population decay
```

---

## Computational Workflow & Methodology
The NAMD simulation pipeline bridges ab initio density functional theory (DFT) with semi-classical quantum dynamics through a 4-stage execution model:

1. **Born-Oppenheimer Molecular Dynamics (BOMD):** The heterostructure is equilibrated under the NVT ensemble at 300 K using a Nosé-Hoover thermostat (dt = 1.0 fs) in VASP.
2. **Static Electronic Structure Extraction:** Snapshots are extracted along the trajectory. Static DFT calculations force the convergence of an explicit active space of **400 bands**, utilizing real-space projection operators (`LREAL = Auto`) and fast minimization (`ALGO = Fast`) to optimize HPC node memory.
3. **Time-Overlaps & Phase Correction (Libra):** Binary wavefunctions (`WAVECAR`) are parsed using Libra's VASP interface to evaluate many-body time-overlaps S(t, t+dt). A phase-tracking algorithm eliminates mathematical sign discontinuities before deriving Non-Adiabatic Coupling (NAC) vectors.
4. **Tully's Fewest Switches Surface Hopping (FSSH):** The complex vibronic Hamiltonian (H_vib) is propagated in Libra within an active window enclosing the Fermi level (**Bands 280 to 350**), tracking real-time electron injection from photoexcited TiO2 into the graphene monolayer.

---

## Quick Start & Execution Guide
To reproduce the analysis and generate the project figures on an HPC environment:

```bash
# 1. Clone the repository
git clone [https://github.com/compchem-cybertraining/Cyber_Training_Workshop2026_course_projects.git](https://github.com/compchem-cybertraining/Cyber_Training_Workshop2026_course_projects.git)
cd Cyber_Training_Workshop2026_course_projects/dianamcb/

# 2. Load required HPC modules (Anaconda & MPI)
module load anaconda3
module load intelmpi/2021.3.0-intel2021.3.0

# 3. Execute the automated plotting and analysis script
python3 scripts/generate_eng_plots.py
```

---

## Key Findings & Scientific Impact
* **Ultrafast Electron Injection:** FSSH population dynamics confirm that graphene acts as an atomic-scale "electron sink," extracting over 55% of the photoexcited charge from TiO2 within ~200 fs, effectively suppressing carrier recombination.
* **Strong Interfacial Coupling:** Time-averaged NAC heatmaps reveal coupling intensities exceeding **15–25 meV**, driven by thermal lattice vibrations (phonons) of the Ti–O framework.
* **HPC Scaling Justification:** Exploratory data demonstrates that while sub-picosecond injection is fast, capturing slow lattice breathing modes and full quantum decoherence requires extended statistical sampling—justifying our current 1640-frame production scale-up on Bridges-2.

---

## References & Acknowledgments
* **Libra Software:** Akimov, A. V. (2016). J. Comput. Chem., 37(18), 1626-1649.
* **VASP Code:** Kresse, G., & Furthmüller, J. (1996). Phys. Rev. B, 54(16), 11169.
* **FSSH Theory:** Tully, J. C. (1990). J. Chem. Phys., 93(2), 1061-1071.
* **Acknowledgments:** Special thanks to the NSF Cyber-Training Workshop organizers at UB, Dr. Alexey Akimov for Libra support, and the Pittsburgh Supercomputing Center (ACCESS/XSEDE) for supercomputing resources.
