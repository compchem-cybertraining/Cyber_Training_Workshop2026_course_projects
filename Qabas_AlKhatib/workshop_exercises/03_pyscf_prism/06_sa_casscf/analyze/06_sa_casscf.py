import numpy as np
from pyscf import gto, scf, mcscf, mp, tdscf
from pyscf import lib
from pyscf.tools import cubegen
from pyscf.tools import molden
import prism.interface
import prism.nevpt

# Specify geometry and molecular information
mol = gto.Mole()
mol.atom = '/projects/academic/cyberwksp21/SOFTWARE_2026/pyscf_prism_examples/geometries/ethylene.xyz'
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.charge = 0
mol.max_memory = 100000
mol.build()

# Run reference Hartree-Fock calculation
mf = scf.RHF(mol).density_fit("aug-cc-pvdz-jkfit")
mf.conv_tol = 1e-8
mf.kernel()

# Use MP2 natural orbitals as guess orbitals for SA-CASSCF
pt = mp.MP2(mf).density_fit("aug-cc-pvdz-jkfit").run()
noon, uno = mcscf.addons.make_natural_orbitals(pt)
print("\nEigenvalues of MP2 density matrix:\n", noon)

n_states = 5
weights = np.ones(n_states)/n_states
mc = mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
emc = mc.mc1step(uno)[0]

# Analyze results
mc.analyze()

mo_mc = mc.mo_coeff.copy()

# Run CASCI calculation using SA-CASSCF orbitals to compute relative energies and wavefunctions for each state
mc = mcscf.CASCI(mf, 6, 6).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
mc.fcisolver.nroots = n_states
emc = mc.casci(mo_mc)[0]

for state in range(n_states):
    print("State #%d, dE = " % (state + 1), (emc[state] - emc[0]) * 27.2114, "eV")

from pyscf.tools import molden
molden.from_mo(mol, 'casscf.molden', mc.mo_coeff)
