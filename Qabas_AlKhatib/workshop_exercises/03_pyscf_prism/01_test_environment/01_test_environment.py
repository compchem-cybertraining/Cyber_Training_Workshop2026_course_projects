#!/usr/bin/env python

'''
Basic MP2 calculation for H2O using Prism NEVPT2 code
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = '6-31G'
mol.symmetry = True
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# NEVPT2 with frozen core
interface = prism.interface.PYSCF(mf, None, backend = 'opt_einsum').density_fit()
nevpt = prism.nevpt.NEVPT(interface)
nevpt.nfrozen = 1
e_tot, e_corr, osc = nevpt.kernel()
