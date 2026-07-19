#!/bin/bash
#SBATCH -J cap_chemml_plots
#SBATCH -o chemml_plots_%j.out
#SBATCH -e chemml_plots_%j.err
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3900M
#SBATCH -p RM-shared

cd ../qc_ml_capstone
export PATH=your/path/to/chem_ml/bin:$PATH
export MPLBACKEND=Agg

python chemml_plots.py --chemml-dir results/chemml_rerun
