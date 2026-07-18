#!/bin/bash
#SBATCH --job-name=fms_nqdh
#SBATCH --array=0-599%150          # 600 tasks = 300 ICs x 2 rescale modes; %150 caps concurrency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_PARTITION # EDIT (we used nes2.8 on UCI Green Planet)
#SBATCH --output=logs/fms_%a.out

# *** RUNTIME WARNING **********************************************************
# * Full multiple spawning is EXPENSIVE: each 40 fs trajectory takes ~5-8 h    *
# * on one CPU core (cost grows with the number of spawned basis functions).  *
# * The full 600-run campaign is ~4,000 core-hours -- a queue job, never a    *
# * laptop job.  Disk: each run writes ~1.3 GB of pySpawn restart files;      *
# * --cleanup (below) keeps only the ~100 KB populations file per run.        *
# ******************************************************************************
#
# One-time environment setup (all in $HOME; tested on an old-glibc cluster):
#   module load python/3.10.18            # load BEFORE creating/activating the venv
#   python3 -m venv ~/envs/nqdh_cpu && source ~/envs/nqdh_cpu/bin/activate
#   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install "numpy<2.3" h5py==3.12.1 hippynn==0.1.3 "matplotlib<3.9" "pillow<11"
#   pip install --no-deps -e /path/to/pySpawn17_py3   # python-3 port!  --no-deps:
#                                                     # its setup.py pins ancient numpy
#   + clone this repo to ~/nqdh-beating-demo
# Gotchas we hit so you don't have to:
#   - upstream blevine37/pySpawn17 master is python 2; use a python-3 port
#   - TORCHDYNAMO_DISABLE=1 is REQUIRED on old toolchains (inductor C++ JIT fails)
#   - never run the model on a login node (resource limits kill it silently)

module load python/3.10.18                 # EDIT to your cluster's python module
source ~/envs/nqdh_cpu/bin/activate
export TORCHDYNAMO_DISABLE=1
export HIPPYNN_USE_CUSTOM_KERNELS=False
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=$HOME/nqdh-beating-demo               # EDIT
WORK=$HOME/fms_ensemble                    # EDIT: output root (needs ~100 MB total)
cd $REPO

K=$SLURM_ARRAY_TASK_ID
IC=$(( K % 300 ))
if [ $K -lt 300 ]; then MODE=velocity; else MODE=coupling; fi
OUT=$WORK/$MODE/ic$(printf "%03d" $IC)

# resumable: requeued/relaunched arrays skip finished tasks
[ -f "$OUT/fms_populations.npz" ] && { echo "already done"; exit 0; }
mkdir -p "$OUT"

python examples/run_fms_single.py --ic-index $IC --rescale $MODE \
  --tfinal 1650 --device cpu --out-dir "$OUT"
# keep only the small populations file (the sim/restart files are ~1.3 GB/run)
cd "$OUT" && rm -f sim*.hdf5 sim*.json working.hdf5
