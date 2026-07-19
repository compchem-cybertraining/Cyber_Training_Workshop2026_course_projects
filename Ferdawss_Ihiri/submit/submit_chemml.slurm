#!/bin/bash
#SBATCH -J cap_chemml
#SBATCH -o chemml_%j.out
#SBATCH -e chemml_%j.err
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -p RM-shared

cd ../qc_ml_capstone
export PATH=your/path/to/chem_ml/bin:$PATH
export MPLBACKEND=Agg

for target in GW_QP_BandGap GWBSE_EsC Singlet_binding; do
    python chemml_automl.py --target $target --dataset results/dataset.csv --out-dir results/chemml_rerun
done
