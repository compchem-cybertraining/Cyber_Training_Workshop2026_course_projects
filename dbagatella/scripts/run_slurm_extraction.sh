#!/bin/bash
#SBATCH -N 1                    # Regresamos a 1 nodo para eliminar el cuello de botella de red
#SBATCH -A CHE200011P
#SBATCH -p RM
#SBATCH --ntasks-per-node=128
#SBATCH --job-name=VASP_Stat_Auto
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dmc59@njit.edu

module purge
module load intelmpi/2021.3.0-intel2021.3.0
export OMP_NUM_THREADS=1

echo "--- INICIANDO EXTRACCIÓN AUTOMÁTICA CON 1 NODO ---"

# 1. Encontrar el último WAVECAR generado en la carpeta
ULTIMO_ARCHIVO=$(ls -v WAVECARS_OUT/WAVECAR_* 2>/dev/null | tail -n 1)

if [ -n "$ULTIMO_ARCHIVO" ]; then
    echo "Último archivo detectado: $ULTIMO_ARCHIVO"
    echo "Recuperando memoria cuántica para acelerar el cálculo..."
    cp $ULTIMO_ARCHIVO WAVECAR
    
    # Extraer el número del archivo y sumarle 1 matemáticamente
    NUMERO_ACTUAL=$(basename $ULTIMO_ARCHIVO | tr -dc '0-9' | sed 's/^0*//')
    PASO_INICIAL=$((NUMERO_ACTUAL + 1))
else
    echo "No se encontraron WAVECARs previos. Iniciando desde cero."
    PASO_INICIAL=1
fi

echo "Retomando el ciclo desde el paso $PASO_INICIAL hasta 1640..."

# 2. Correr el ciclo secuencial
for i in $(seq -f "%04g" $PASO_INICIAL 1640); do
    echo "Calculando fotograma $i..."
    cp POSCARS_FOTOGRAMAS/POSCAR_$i POSCAR
    
    mpirun -np $SLURM_NTASKS /opt/packages/VASP/VASP6/6.3/INTEL/vasp_std > vasp_static.log
    
    cp WAVECAR WAVECARS_OUT/WAVECAR_$i
    
    # Limpiamos basura, pero el WAVECAR se mantiene para el siguiente paso
    rm -f CHG CHGCAR DOSCAR EIGENVAL OSZICAR OUTCAR PCDAT REPORT vasprun.xml XDATCAR
done

echo "¡Mitad 1 completada exitosamente!"
