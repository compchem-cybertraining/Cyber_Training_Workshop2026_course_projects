# VASP Input Parameters & System Configuration

This directory contains the essential electronic structure parameters and structural files required to execute the VASP static and molecular dynamics stages of the NAMD workflow.

## File Descriptions

* **`INCAR_MD`**: Configuration for Born-Oppenheimer Molecular Dynamics (BOMD). Sets up an NVT ensemble at 300 K using a Nosé-Hoover thermostat (`MDALGO = 2`, `SMASS = 0.5`) with a 1.0 fs time step (`POTIM = 1.0`).
* **`INCAR_static_example`**: Optimized configuration for extracting static electronic wavefunctions along the MD trajectory.
  * **Critical Setting:** `NBANDS = 400` ensures explicit calculation of virtual conduction bands required for the active space.
  * **HPC Speed Optimization:** Utilizes `LREAL = Auto` (real-space projections) and `ALGO = Fast` to accelerate convergence on 128-core HPC nodes.
  * **Output Flag:** `LWAVE = .TRUE.` forces the writing of binary `WAVECAR` files needed by Libra.
* **`POSCAR_example`**: Initial atomic coordinates of the Graphene-TiO2 heterostructure (128 atoms total: Ti, O, and C monolayer).
* **`KPOINTS`**: Specifies a Gamma-centered Monokhorst-Pack mesh (1x1x1), optimized for large supercell Brillouin zone sampling.
* **`POTCAR_LICENSE_NOTE`**: Detailed instructions on assembling the PAW pseudopotentials (`Ti_pv`, `O`, `C`) from a licensed VASP distribution. Original POTCARs are omitted to comply with VASP copyright guidelines.
