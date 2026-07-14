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

# Run state-avereged CASSCF calculation for singlet states
n_states = 5
weights = np.ones(n_states)/n_states
mc = mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
emc = mc.mc1step()[0]

mo_mc = mc.mo_coeff.copy()

# Run CASCI calculation using SA-CASSCF orbitals to compute relative energies and wavefunctions for each state
mc = mcscf.CASCI(mf, 6, 6).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
mc.fcisolver.nroots = n_states
emc = mc.casci(mo_mc)[0]

for state in range(n_states):
    print("State #%d, dE = " % (state + 1), (emc[state] - emc[0]) * 27.2114, "eV")

# Analyze results
mc.analyze()

# Use MP2 natural orbitals as guess orbitals for SA-CASSCF
pt = mp.MP2(mf).density_fit("aug-cc-pvdz-jkfit").run()
noon, uno = mcscf.addons.make_natural_orbitals(pt)
print("\nEigenvalues of MP2 density matrix:\n", noon)

mc = mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
emc = mc.mc1step(uno)[0]

mo_mc = mc.mo_coeff.copy()

# Run CASCI calculation using SA-CASSCF orbitals to compute relative energies and wavefunctions for each state
mc = mcscf.CASCI(mf, 6, 6).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
mc.fcisolver.nroots = n_states
emc = mc.casci(mo_mc)[0]

for state in range(n_states):
    print("State #%d, dE = " % (state + 1), (emc[state] - emc[0]) * 27.2114, "eV")

# Use CIS natural orbitals as guess orbitals for SA-CASSCF [J. Chem. Phys. 142, 024102 (2015)]
def tda_density_matrix(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    return dm

# Obtain 1-rdm for RHF ground state
hf_coeff = mf.mo_coeff.copy()
hf_rdm1 = np.diag(mf.mo_occ)

# Perform CIS calculation for singlet states
mytd_singlet = tdscf.TDA(mf)
mytd_singlet.nstates = n_states
mytd_singlet.singlet = True
mytd_singlet.kernel()
mytd_singlet.analyze()

## Perform CIS calculation for triplet states
#mytd_triplet = tdscf.TDA(mf)
#mytd_triplet.nstates = 10
#mytd_triplet.singlet = False
#mytd_triplet.kernel()
#mytd_triplet.analyze()

# Obtain 1-RDM for CIS excited states
cis_rdm1_singlet = []
for state in range(mytd_singlet.nstates):
    cis_rdm1_singlet.append(tda_density_matrix(mytd_singlet, state))
cis_rdm1_singlet = np.array(cis_rdm1_singlet)

#cis_rdm1_triplet = []
#for state in range(mytd_triplet.nstates):
#    cis_rdm1_triplet.append(tda_density_matrix(mytd_triplet, state))
#cis_rdm1_triplet = np.array(cis_rdm1_triplet)

# Obtain CISNO eigenvalues and eigenvectors
cisno_rdm1 = hf_rdm1.copy()
for state in range(mytd_singlet.nstates):
    cisno_rdm1 += cis_rdm1_singlet[state] 

#for state in range(mytd_triplet.nstates):
#    cisno_rdm1 += cis_rdm1_triplet[state] 
#cisno_rdm1 /= (mytd_singlet.nstates + mytd_triplet.nstates + 1)

cisno_rdm1 /= (mytd_singlet.nstates + 1)

# Diagonalize the SA-CIS density matrix to get occupation numbers
sa_eigval, sa_eigvec = np.linalg.eigh(cisno_rdm1)

sa_eigval = sa_eigval[::-1]
sa_eigvec = sa_eigvec[:,::-1]

print("\nEigenvalues of CISNO density matrix:\n", sa_eigval)

# Transform MO coef into CISNO basis
cisno_coeff = np.dot(hf_coeff, sa_eigvec)

# Run SA-CASSCF starting with CISNO as a guess
mc = mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
emc = mc.mc1step(cisno_coeff)[0]

mo_mc = mc.mo_coeff.copy()

# Run CASCI calculation using SA-CASSCF orbitals to compute relative energies and wavefunctions for each state
mc = mcscf.CASCI(mf, 6, 6).density_fit('aug-cc-pvdz-jkfit')
mc.fix_spin_(ss=0) # Singlet states only
mc.fcisolver.nroots = n_states
emc = mc.casci(mo_mc)[0]

for state in range(n_states):
    print("State #%d, dE = " % (state + 1), (emc[state] - emc[0]) * 27.2114, "eV")

mc.analyze()

