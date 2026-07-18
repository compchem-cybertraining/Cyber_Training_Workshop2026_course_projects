#!/bin/bash
#SBATCH -J p4def
#SBATCH -A members -p cpu -N 1 -c 32 --mem=32G -t 06:00:00
#SBATCH -o slurm_p4def_%j.out
set +e                                 
S=${SLURM_SUBMIT_DIR}                    # code dir (this folder)
DATA="$S/data"; export P4DATA="$DATA"    # ALL generated data live here
PY3=/home2/users/yschoi/programs/anaconda3/bin/python          # base env: numpy + h5py + matplotlib
PY2=/home2/users/yschoi/programs/anaconda3/envs/pyspawn/bin/python   # py2.7 for pySpawn
/usr/bin/mkdir -p "$DATA"; cd "$DATA"
/usr/bin/cp "$S/pyrmod4.op" "$S/pyr4.inp" "$DATA/"     # so MCTDH finds the operator locally

echo "=== (1) MCTDH exact ==="
export MCTDH_DIR=/home2/users/yschoi/programs/mctdh86.9
source "$MCTDH_DIR/install/mctdh.profile" >/dev/null 2>&1
/usr/bin/rm -rf "$DATA/pyr4"
mctdh86 -mnd -w pyr4.inp > mctdh_run.log 2>&1    
OMP_NUM_THREADS=1 "$PY3" "$S/analyze_mctdh.py" pyr4       # $DATA/mctdh_pop.csv

echo "=== (2) ladder operator: full N=657 + convergence series toward the minimal N=183 ==="
CM="--merge=0.8 --prune=1e-3 --pruneage=15 --reg=1e-6 --tf=120 --dt=0.1"
/usr/bin/mkdir -p "$DATA/ci_combo"
# full N=657 (tag pyrmod4) -> $DATA/swarm_pop_pyrmod4.csv
( cd "$DATA"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY3" "$S/tower_pyr4.py" \
    --Kp=6,3,3,3 --cutoff=5 --spawnkp=3,2,2,2 --spawn=5 $CM --tag=pyrmod4 > tower_pyrmod4.log 2>&1 ) &
# convergence series -> $DATA/ci_combo/ 
( cd "$DATA/ci_combo"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY3" "$S/tower_pyr4.py" \
    --Kp=4,2,2,2 --cutoff=4 --spawnkp=2,1,1,1 --spawn=5 $CM --tag=k4 > tower_k4.log 2>&1 ) &
for g in 0.3 0.4 0.5; do
  ( cd "$DATA/ci_combo"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY3" "$S/tower_pyr4.py" \
      --Kp=6,3,3,3 --cutoff=2 --spawnkp=3,2,2,2 --spawn=5 --spawngap="$g" $CM --tag="c2g$g" > "tower_c2g$g.log" 2>&1 ) &
done
wait

echo "=== (3) AIMS: 96 Wigner ICs (pySpawn; each IC self-samples its Wigner IC from seed) ==="
runic(){ i=$1; d="$DATA/$(printf "ic%02d" $i)"; /usr/bin/mkdir -p "$d"; cd "$d"
  /usr/bin/rm -f sim.hdf5 sim.json working.hdf5 2>/dev/null
  PYTHONPATH="$S" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY2" "$S/start.py" "$i" > run.log 2>&1; cd "$DATA"; }
MAXJ=32                                
for i in $(seq 0 95); do
  runic "$i" &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJ" ]; do sleep 2; done
done
wait

echo "=== (4) adiabatic->diabatic projection ==="
OMP_NUM_THREADS=1 "$PY3" "$S/analyze_aims.py"       # $DATA/aims_pop.csv, aims_perrun.npz

echo "P4DEF_DONE" > "$DATA/DONE_definitive.flag"
