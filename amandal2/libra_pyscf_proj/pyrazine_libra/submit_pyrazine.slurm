#!/bin/bash
#SBATCH --job-name=pyrazine
#SBATCH --output=logs/pyz_%A_%a.out
#SBATCH --error=logs/pyz_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-49%20          # 50 trajectories, at most 20 at once
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cluster=ub-hpc

# ---------------------------------------------------------------------------
# Usage:
#   mkdir -p logs
#   1) ONCE:   python run_pyrazine.py prep --ntraj 50 --outdir runs
#              (do this in an interactive session -- optimization + Hessian,
#               tens of minutes. It is NOT repeated per task, on purpose.)
#   2) THEN:   sbatch submit_pyrazine.sh
#
# Each array task runs ONE trajectory. generic_recipe runs trajectories serially
# inside a process, so ntraj=8 in one job would take 8x as long as ntraj=1 and buy
# nothing -- trajectories are independent. The array IS the parallelism.
# ---------------------------------------------------------------------------

# ===== environment: mirrors the libra kernel's launcher.sh exactly =====
# HARD CLEAN first. Order matters: the unset must happen BEFORE the module load,
# because libra_ava/devel puts Libra's build tree on PYTHONPATH and we need that.
unset PYTHONPATH
unset PYTHONHOME
unset EBPYTHONPREFIXES
export PYTHONNOUSERSITE=1

module use /projects/academic/cyberwksp21/MODULES
module load libra_ava/devel

source /projects/academic/cyberwksp21/SOFTWARE/Conda/etc/profile.d/conda.sh
conda activate libra

export LD_LIBRARY_PATH=/projects/academic/cyberwksp21/SOFTWARE/libra/_build/src:$LD_LIBRARY_PATH

# pyscf lives in a directory you own; pyscf_libra.py reads this and APPENDS it to
# sys.path (append, so the shared env still wins every name collision).
export LIBRA_EXTRAS=/user/amandal2/libra-extras

# PySCF is OpenMP-threaded. Match the threads to the cores you actually asked for --
# oversubscribing makes it slower, not faster.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ===== run =====
# cd BEFORE the sanity check: pyscf_libra.py lives here, and importing it is what
# puts $LIBRA_EXTRAS on sys.path.
cd "$SLURM_SUBMIT_DIR"

# ===== sanity: fail loudly and early rather than 6 hours in =====
# NOTE the import order. A bare `import pyscf` FAILS here and that is by design --
# pyscf lives in $LIBRA_EXTRAS, and nothing in the environment points at it. The
# sys.path.append happens inside pyscf_libra.py, so pyscf only becomes importable
# after that module loads. run_pyrazine.py imports them in this same order.
python - <<'EOF'
import sys, os
print("python:", sys.version.split()[0])
print("cwd   :", os.getcwd())

extras = os.environ.get("LIBRA_EXTRAS", "<unset>")
print("LIBRA_EXTRAS:", extras, "| exists:", os.path.isdir(extras))
if not os.path.isdir(extras):
    sys.exit("LIBRA_EXTRAS does not exist -- pip install --no-deps --target=$LIBRA_EXTRAS pyscf")

if not os.path.exists("pyscf_libra.py"):
    sys.exit(f"pyscf_libra.py not found in {os.getcwd()} -- it must sit beside run_pyrazine.py")

import numpy
print("numpy :", numpy.__version__, "from", os.path.dirname(numpy.__file__))
assert "libra-extras" not in numpy.__file__, "numpy shadowed by LIBRA_EXTRAS -- reinstall with --no-deps"

import liblibra_core
print("liblibra_core: OK")

import pyscf_libra          # <-- this is what makes pyscf importable
import pyscf
print("pyscf :", pyscf.__version__, "from", os.path.dirname(pyscf.__file__))

from recipes import ehrenfest_adi_ld, fssh
print("recipes: OK")
EOF
if [ $? -ne 0 ]; then echo "environment check FAILED -- not starting the trajectory"; exit 1; fi

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} cores ==="
date

python run_pyrazine.py run \
    --icond "${SLURM_ARRAY_TASK_ID}" \
    --method 2 \
    --nsteps 250 \
    --dt 10.0 \
    --outdir runs

echo "=== done ==="
date
