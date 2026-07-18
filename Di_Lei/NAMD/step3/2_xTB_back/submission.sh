#!/bin/bash -l

#SBATCH --clusters=faculty

#SBATCH --partition=valhalla --qos=valhalla

#SBATCH --account=cyberwksp21

#SBATCH --time=24:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=16

#SBATCH --cpus-per-task=1

#SBATCH --mem=60000




source ~/.bashrc
conda activate libra

cd /vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/step3/2_xTB
python tutorial.py
