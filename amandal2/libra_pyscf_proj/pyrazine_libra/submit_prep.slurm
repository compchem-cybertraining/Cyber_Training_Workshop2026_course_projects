#!/bin/bash
#SBATCH --job-name=pyz_prep
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cluster=ub-hpc

# ---------------------------------------------------------------------------
# ONE-TIME prep: S0 geometry optimization -> analytic Hessian -> normal modes ->
# Wigner ensemble -> bright-state identification. Writes runs/prep.npz, which every
# array task then reads.
#
#   mkdir -p logs
#   sbatch submit_prep.sh
#   # wait for runs/prep.npz, then:
#   sbatch --array=0 submit_pyrazine.sh        # test ONE task
#   sbatch --array=1-49%20 submit_pyrazine.sh  # release the rest
#
# More cores here than in the trajectory jobs: the Hessian threads well and this
# runs once, so it is worth the wider allocation.
# ---------------------------------------------------------------------------

unset PYTHONPATH
unset PYTHONHOME
unset EBPYTHONPREFIXES
export PYTHONNOUSERSITE=1

module use /projects/academic/cyberwksp21/MODULES
module load libra_ava/devel

source /projects/academic/cyberwksp21/SOFTWARE/Conda/etc/profile.d/conda.sh
conda activate libra

export LD_LIBRARY_PATH=/projects/academic/cyberwksp21/SOFTWARE/libra/_build/src:$LD_LIBRARY_PATH
export LIBRA_EXTRAS=/user/amandal2/libra-extras

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"

echo "=== prep on $(hostname), ${SLURM_CPUS_PER_TASK} cores ==="
date

python run_pyrazine.py prep \
    --ntraj 50 \
    --basis "6-31G*" \
    --xc pbe0 \
    --nexc 5 \
    --temperature 0.0 \
    --seed 42 \
    --outdir runs

echo "=== done ==="
date
ls -la runs/
