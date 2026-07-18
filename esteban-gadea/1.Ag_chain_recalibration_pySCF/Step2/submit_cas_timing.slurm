#!/bin/bash
#SBATCH --job-name=agchain_cas_timing
#SBATCH --output=agchain_cas_timing_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00        # generous on purpose -- see note below
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

# --- adapt to your cluster's environment/module system ---
# module load python/3.10 ...
# conda activate pyscf-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
python3 cas_timing_scan.py

# Notes on this round:
# - natoms=12 is deliberately NOT in this job (cas_timing_scan.py now only
#   loops natoms in [4, 8]). The first run showed real multiple-minima /
#   convergence issues at those two sizes that need resolving before it's
#   worth spending cluster time on 12.
# - max_cycle_macro raised from 10 to 40: the first run showed the HF-guess
#   CASSCF hadn't converged at 10 iterations at either natoms=4 or 8, and
#   the three guesses (HF, MP2 natural orbitals, DFT/PBE) landed on
#   different energies with the ranking flipping between the two sizes --
#   a real multiple-local-minima signature, not just slow convergence.
#   40 macro iterations gives a real chance to see if they actually
#   converge to the same minimum given enough room.
# - MP2 natural orbitals crashed ("SVD did not converge") at natoms=8 last
#   time. Switched from the generic pyscf.mp.MP2 to the periodic-specific
#   pyscf.pbc.mp.MP2, and exceptions now print a full traceback to the log
#   instead of a one-line summary, so if it fails again you'll see exactly
#   where.
# - cas_timing_scan.py still runs all 3 guesses per natoms, and each is now
#   allowed up to 4x more macro iterations than before, so expect this run
#   to take noticeably longer than the first one -- 8h/32G is kept as a
#   generous placeholder. Run natoms=4 alone first if you want a cheaper
#   initial read before committing to both.
