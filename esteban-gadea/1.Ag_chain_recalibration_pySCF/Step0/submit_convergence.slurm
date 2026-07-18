#!/bin/bash
#SBATCH --job-name=agchain_conv
#SBATCH --output=agchain_conv_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # pySCF parallelizes via OpenMP/BLAS threads, not MPI
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

# --- adapt this block to your cluster's environment/module system ---
# module load python/3.10 openmpi ...
# source activate your_pyscf_env
# or: conda activate pyscf-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
python3 convergence_scan.py

# Notes:
# - This is a single-node, multi-threaded job (no MPI). Density fitting
#   (GDF) and the underlying linear algebra scale with OMP/BLAS threads.
# - 4 hours / 32 GB is a generous placeholder for the full 5+6+3 = 14 points
#   in convergence_scan.py at szv-molopt-sr / natoms<=6 / nk<=16 -- the
#   smoke test on this end (natoms=2, vacuum=10, nk=1) took ~4 s, so the
#   whole scan should be minutes, not hours, unless a specific (nk, vacuum)
#   combination blows up the density-fitting auxiliary basis. Adjust down
#   once you see real timings, and use those numbers to extrapolate the
#   cost of the full production sweep (more lattice lengths + natoms +
#   eventually post-HF methods on top).
