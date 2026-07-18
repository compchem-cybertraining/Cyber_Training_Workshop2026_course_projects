import numpy as np
from pyscf import gto, scf, cc
from pyscf import lib
from pyscf.tools import cubegen
from pyscf.tools import molden

# Specify geometry and molecular information
mol = gto.Mole()
mol.atom = '/projects/academic/cyberwksp21/SOFTWARE_2026/pyscf_prism_examples/geometries/ethylene.xyz'
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.charge = 0
mol.max_memory = 10000
mol.build()

# Run reference (ground-state) Hartree-Fock calculation
mf = scf.RHF(mol).density_fit("aug-cc-pvdz-jkfit")
mf.conv_tol = 1e-8
mf.kernel()

# Run CCSD
mycc = cc.CCSD(mf).set_frozen().density_fit("aug-cc-pvdz-ri")
mycc.verbose = 5
mycc.kernel()

# Run EOM-CCSD for singlet states
eS = mycc.eomee_ccsd_singlet(nroots=10)[0]

# Print excitation energies
for state in range(len(eS)):
    print ("State #%d, dE = %9.6f (a.u.), %6.3f (eV)" % (state+1, eS[state], eS[state] * 27.2114))
