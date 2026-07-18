#!/bin/bash
#SBATCH --job-name=pyscf_example
#SBATCH --output=slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source /projects/academic/cyberwksp21/SOFTWARE_2026/pyscf_prism_environment.sh
python 04_sr_adc.py > 04_sr_adc.dat 2>&1
