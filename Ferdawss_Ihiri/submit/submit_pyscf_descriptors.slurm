#!/bin/bash
#SBATCH -J cap_pyscf
#SBATCH -o pyscf_%j.out
#SBATCH -e pyscf_%j.err
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -p RM-shared
#SBATCH --account=che250019p

cd ../qc_ml_capstone
export PATH=your/path/to/qc_env/bin:$PATH
export OMP_NUM_THREADS=8
python pyscf_descriptors.py

