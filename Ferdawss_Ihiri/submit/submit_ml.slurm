#!/bin/bash
#SBATCH -J cap_ml
#SBATCH -o ml_%j.out
#SBATCH -e ml_%j.err
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -p RM-shared

cd ../qc_ml_capstone
export PATH=your/path/to/chem_ml/bin:$PATH
export MPLBACKEND=Agg

python build_dataset.py
python train_ml.py --target GW_QP_BandGap
python train_ml.py --target GWBSE_EsC
python train_ml.py --target Singlet_binding
python results_report.py
