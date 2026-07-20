# Interface of LIBRA with the Multireference Quantum Chemistry Package OpenMolcas

## Objective

The objective of this project is to develop an interface between the **LIBRA** nonadiabatic molecular dynamics package and the **OpenMolcas** multireference quantum chemistry software to enable **Neglect of Back Reaction Approximation (NBRA)** calculations using multireference electronic structure methods.

The interface automates the extraction of electronic structure information from OpenMolcas calculations and converts it into the format required by LIBRA for nonadiabatic molecular dynamics simulations.

## Workflow

The interface performs the following tasks:

### Step 1: Read Electronic Structure Data from OpenMolcas

The interface reads the following quantities from the OpenMolcas output files:

- Adiabatic electronic energies
- Molecular orbital information
- Atomic orbital (AO) overlap matrix
- Configuration Interaction (CI) coefficients

### Step 2: Construct LIBRA Input Files

Using the electronic structure information obtained from OpenMolcas, the interface generates the quantities required by LIBRA, including:

- State overlap matrix (`s_adi.txt`)
- Time-overlap matrix (`st_adi.txt`)
- Adiabatic Hamiltonian (`ham_adi.txt`)
- Vibronic Hamiltonian (`hvib_adi.txt`)

These files provide the electronic information needed for nonadiabatic dynamics within the NBRA framework.

### Step 3: Perform NBRA Dynamics

The generated files are used as input to LIBRA to perform **Step-4 Nonadiabatic Molecular Dynamics (NAMD)** calculations. LIBRA utilizes

- Adiabatic energies,
- Time-overlap matrices, and
- Vibronic Hamiltonians

to propagate the electronic populations and simulate excited-state nonadiabatic dynamics using the NBRA approach.

## Features

- Interface between OpenMolcas and LIBRA
- Supports multireference electronic structure methods (e.g., CASSCF/RASSCF)
- Automatic extraction of electronic structure information from OpenMolcas outputs
- Generation of LIBRA-compatible Hamiltonian and overlap files
- Enables NBRA-based excited-state nonadiabatic molecular dynamics simulations
- The pyrazine molecule was used as an example.
