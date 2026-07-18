#!/bin/bash
##serial execution
#SBATCH -N1 -n48
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=300Gb
#SBATCH --job-name=molcas
#SBATCH --error  myJob.err
#SBATCH --output myJob.out
#SBATCH --account=IscrC_HyChall
#SBATCH --partition=g100_usr_dbg


# Load needed modules
module purge
module load profile/chem-phys
module load autoload openmolcas/21.10

PREFIX=`pwd`
export WD=$PREFIX
export SCRDIR=$CINECA_SCRATCH/MOLCAS_$SLURM_JOBID
mkdir  -p $SCRDIR
cp ./* $SCRDIR
cd $SCRDIR

export MOLCAS_NPROCS=$SLURM_NTASKS

export CurrDir=$(pwd)

dists=(0.5 0.75 1.0 1.25 1.50 1.75 2.0 2.5)

printf "dist ES0 ES1 gap\n" > gaps.dat

for dist in "${dists[@]}"; do
  cat > h2.inp <<EOF
&GATEWAY
    Coord
    2
    OpenCazzas
    H 0.000000 0.000000 0.000000
    H 0.000000 0.000000 $dist
    Basis = def2-TZVP
    Group = NoSym

&SEWARD

&SCF
    KSDFT = B3LYP
    Charge = 0
    Spin = 1

&RASSCF
    Title = H2 SA-2-CAS(2,2)SCF
    symmetry = 1
    Nactel = 2 0 0
    Spin = 1
    Ras2 = 4
    ciroot = 2 2 1
    RLXRoot = 1
EOF

  # replace this with your site's OpenMolcas launcher
  ${OPENMOLCAS_HOME}/pymolcas -np $MOLCAS_NPROCS h2.inp > h2_${dist}.log 2>&1

  # extract the two state energies from the log
  #mapfile -t energies < <(awk '/RASSCF root number/ {root=$NF} /Total energy/ {print root, $NF}' h2_${dist}.log | awk '$1==1 || $1==2 {print $2}')
  mapfile -t energies < <(
   awk '/RASSCF root number/ {print $NF}' h2_${dist}.log
  )

  es0=${energies[0]}
  es1=${energies[1]}
  gap=$(awk -v e0="$es0" -v e1="$es1" 'BEGIN{print (e1-e0)/2}')

  printf "%s %s %s %s\n" "$dist" "$es0" "$es1" "$gap" | tee -a gaps.dat
done

mkdir -p $WD/$SLURM_JOBID
cp ./* $WD/$SLURM_JOBID
cd ../
rm -rf MOLCAS_$SLURM_JOBID
