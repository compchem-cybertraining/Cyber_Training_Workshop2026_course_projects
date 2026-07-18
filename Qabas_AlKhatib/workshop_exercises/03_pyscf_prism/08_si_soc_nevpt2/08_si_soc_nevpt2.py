import numpy as np
import math
from pyscf import gto, scf, mcscf
import prism.interface
import prism.nevpt

# Specify geometry and molecular information
mol = gto.Mole()
mol.atom = [
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 4
mol.build()

# Use DFT (BP86) to compute guess orbitals
mf = scf.RKS(mol).x2c() # Include scalar relativistic effects
mf.xc = "bp86"
ehf = mf.scf()
mf.analyze()

# Run SA-CASSCF calculation
n_states = 3
weights = np.ones(n_states)/n_states
mc = mcscf.CASSCF(mf, 5, 3).state_average_(weights)
emc = mc.mc1step()[0]
mc.analyze()

# Run QD-NEVPT2 calculation without spin-orbit coupling
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
nevpt.method_type = "qd"
nevpt.verbose = 4
nevpt.kernel()

# Run QD-NEVPT2 calculation with state-interaction treatment of spin-orbit coupling
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
nevpt.method_type = "qd"
nevpt.soc = "DKH1" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
nevpt.verbose = 4
nevpt.gtensor = True # Compute magnetic properties (g-tensor)
nevpt.kernel()
