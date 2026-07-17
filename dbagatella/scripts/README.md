# Automated NAMD Execution Scripts

This directory houses the Python modules and Slurm batch scripts that automate the extraction of quantum mechanical data and execution of Surface Hopping dynamics.

## Script Descriptions & Execution Order

### 1. `run_slurm_extraction.sh` (HPC Batch Automation)
* **Purpose:** Slurm submission script for the PSC Bridges-2 supercomputer.
* **Function:** Sequentially copies snapshots from the MD trajectory, executes MPI-parallelized VASP static calculations (`vasp_std`), and securely saves generated `WAVECAR` files while maintaining memory efficiency.
* **Usage:** `sbatch run_slurm_extraction.sh`

### 2. `step1-LIBRA.py` (Wavefunction Parsing & Overlaps)
* **Purpose:** Interfaces directly with VASP binary outputs using the Libra package.
* **Function:** Reads sequential `WAVECAR_XXXX` files, extracts Kohn-Sham eigenvalues, and computes the many-body time-overlap matrices S(t, t+dt) across the designated active space (**Bands 280 to 350**).
* **Usage:** `python3 step1-LIBRA.py` (Outputs matrices to `/res` folder).

### 3. `step2_libra.py` (Phase Correction & Hamiltonian Assembly)
* **Purpose:** Constructs the complex vibronic Hamiltonian (H_vib) required for Tully's FSSH.
* **Function:** Reads raw overlap matrices from `step1`, applies a rigorous phase-tracking algorithm to eliminate mathematical sign flips between consecutive time steps, and computes the real-time Non-Adiabatic Coupling (NAC) vectors.
* **Usage:** `python3 step2_libra.py` (Outputs clean matrices to `/res_step2` folder).

### 4. `generate_eng_plots.py` (Publication-Grade Visualization)
* **Purpose:** Automated analytical and plotting suite.
* **Function:** Recursively scans the project directory for NAMD output matrices, parses energy levels and coupling coefficients, and generates standardized English figures at 300 DPI. Includes a physical fail-safe data generator for standalone repository verification.
* **Usage:** `python3 generate_eng_plots.py` (Outputs PNGs to `/figures` folder).
