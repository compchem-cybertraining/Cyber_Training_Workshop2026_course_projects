import numpy as np
from pyscf import gto, scf, adc
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

# Run EE-ADC(2) calculation
myadc = adc.ADC(mf).density_fit("aug-cc-pvdz-ri")
myadc.method = "adc(2)"
myadc.conv_tol = 1e-5
myadc.tol_residual = 1e-3
myadc.method_type = "ee"
myadc.verbose = 5
myadc.approx_trans_moments = True
e,v,p,x = myadc.kernel(nroots = 5)

# Analyze the results
myadc.analyze()

# Compute reference (ground-state) one-particle density matrix
rdm1_ref = myadc.make_ref_rdm1(ao_repr = True)

# Compute excited-state one-particle density matrix
rdm1_ee = np.array(myadc.make_rdm1(ao_repr = True))

# Write density difference for each excited state
for state in range(rdm1_ee.shape[0]):
    rdm1_diff = rdm1_ee[state] - rdm1_ref
    cubegen.density(mol, 'state_%s.cube' % str(state + 1), rdm1_diff)

# Compute natural transition orbitals
import prism.interface
from prism.tools import trans_prop
interface = prism.interface.PYSCF(mf)
for state in range(x.shape[0]):
    trans_prop.compute_ntos(interface, x[state], initial_state=0, target_state=state+1)
