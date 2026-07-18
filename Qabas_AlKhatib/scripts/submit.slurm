#!/bin/bash
#SBATCH --job-name=casscf_data
#SBATCH --output=/user/qkhatib/casscf_%j.out
#SBATCH --error=/user/qkhatib/casscf_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

cd ~/CyberTraining2026/ml-hessian-casscf
echo "Starting job at $(date)"
python3 scripts/generate_data.py
echo "Job finished at $(date)"
