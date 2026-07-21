
## Step 0 - optimization-DFT (VASP) and post-DFT (bands, pdos, charge analysis, etc.)
#This step involves geometry optimization of the 2*2*2 supercell of CsPbI$_3$ doped with Zn atoms at Pb sites (12.5% doping considered here). After optimization, electronic structure is analysed. Band structure, density of states (DOS), p-DOS, charge distribution, etc are computed and plotted. VASP 6.2.0 was used along with other data analysis tools. This part was performed on JNCASR ParamYukti HPC facility.
(Note: Input files are not on the repo yet, will be pushed soon)

## Step 1 - AIMD (using VASP)
#After optimization, molecular dynamics was performed, equilibration in NVT ensemble for 500 fs, followed by production run in NVE & NVT  ensemble for 10,000 fs or 2 ps, at 300 K using the VASP 6.2.0. Radial distribution function, g(r) was calculated from AIMD data. This part was performed on JNCASR ParamYukti HPC facility.

## Step 2 - TD-DFPT calculations (CP2K)
#The Kohn-Sham time overlap matrices were calculated using time-dependent density functinoal perturbation theory (TD-DFPT) at 1000 snapshots obtained from 10,000 fs AIMD run at every 10 fs time-step in CP2K v2025 on UB CCR HPC facility.

## Step 3 - TD-DFT time-overlaps (CP2K/Libra)
#Time-resolved state energies and ovelap matrices were calculated post TD-DFPT calculations (log files) using the Libra code. This part was implemented using Libra code on UB CCR HPC facility.

## Step 4 - Pre-NAMD like excitation analysis, dynamical analysis (NAC)  (Libra)
#For pre-NAMD analysis, excitation and dynamical analyses were performed. Electron-phonon couplings were investigated by plotting the influence spectrum and spectral desnity w.r.t. frequency. Non-adiabatic coupling matrix was calculated between the occupied and unoccupied states to determine the extent of non-adibaticity in the system. This part was implemented using Libra code on UB CCR HPC facility.

## Step 5 - NAMD (Libra)
#Non-adiabatic molecular dynamics (NAMD) was implemented using the Ehrenfest dynamics, fewest-switches surface hopping (FSSH1/FSSH2), FSSH2 This part was implemented using Libra code on UB CCR HPC facility.
