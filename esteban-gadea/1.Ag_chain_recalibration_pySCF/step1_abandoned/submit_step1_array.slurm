#!/bin/bash
#SBATCH --job-name=agchain_step1_surface
#SBATCH --output=step1_surface_%a.log
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

# --- adapt to your cluster's environment/module system ---
# module load python/3.10 ...
# conda activate pyscf-env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
python3 step1_surface_scan.py   # picks up lattice_length from SLURM_ARRAY_TASK_ID

# One array task per lattice length (12 tasks, indices 0-11 -> L_VALUES in
# step1_surface_scan.py, i.e. 5.2, 5.3, ..., 6.3 A). Each task loops its own
# 21 delta values (0.00 to 0.20 step 0.01) and checkpoints to its own
# step1_surface_l<L>.csv after every point.
#
# Cost basis: Step 0's joint-converged HF point (nk=16, vacuum=30, the same
# recipe used here) took ~85s. 21 points/task -> ~30 min/task at that rate.
# 4h/task is a generous margin for delta values near the undimerized (small
# delta) end, which weren't part of Step 0's testing and could need the
# smearing rescue path (see script docstring) and/or more SCF cycles. A quick
# interactive check (smaller nk/vacuum, not the production setting) suggests
# plain RHF handles delta=0 fine, but that wasn't tested at the real nk=16
# production setting across all 12 lattice lengths -- hence the margin.
#
# If your cluster's array-size or queue policy makes 12 concurrent tasks
# awkward, run submit_step1_preflight.slurm (or step1_preflight_check.py
# directly) first -- a handful of points spanning delta=0 to 0.20 at a single
# lattice length -- before committing the full array.
