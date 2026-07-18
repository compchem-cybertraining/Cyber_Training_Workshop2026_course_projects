import numpy as np
from pyscf import gto, scf, adc, tdscf
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
mf = scf.RHF(mol)
mf.kernel()

# Run time-dependent SCF calculation in the Tamm-Dancoff approximation (CIS)
mytd = tdscf.TDA(mf)
mytd.nstates = 10
mytd.singlet = True
mytd.kernel()

# Analyze the results
mytd.analyze()

# Compute natural transition orbitals
for i in range(mytd.nstates):
    weights, nto = mytd.get_nto(state=i+1, verbose=4)
    molden.from_mo(mol, 'nto-tda-%s.molden' % str(i+1), nto)

# Function for calculating TDA density matrix
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

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

# Density matrix for the 3rd excited state
dm = tda_density_matrix(mytd, 2)

# Write to cube format
from pyscf.tools import cubegen
cubegen.density(mol, 'tda_density.cube', dm)

# Write the density difference between excited state and ground state
cubegen.density(mol, 'density_diff.cube', dm - mf.make_rdm1())

# Run full time-dependent SCF calculation (RPA)
mytd = tdscf.TDHF(mf)
mytd.nstates = 10
mytd.singlet = True
mytd.kernel()

# Analyze the results
mytd.analyze()

# Compute natural transition orbitals
for i in range(mytd.nstates):
    weights, nto = mytd.get_nto(state=i+1, verbose=4)
    molden.from_mo(mol, 'nto-rpa-%s.molden' % str(i+1), nto)

