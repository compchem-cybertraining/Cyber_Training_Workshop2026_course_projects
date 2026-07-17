import time
import numpy as np
from pyscf.pbc import scf as pbcscf
from pyscf import mcscf
from ag_chain_lib import build_cell

t0 = time.time()
cell = build_cell(natoms=2, lattice_length=6.0, delta=0.0, vacuum=20.0)
mf = pbcscf.RHF(cell).density_fit()
mf.verbose = 0
mf.kernel()
print("HF done", time.time()-t0, "E=", mf.e_tot, "converged", mf.converged, flush=True)

# state-averaged CASSCF(2,2) on neutral system: ground + first excited (optical-like gap)
mc = mcscf.CASSCF(mf, 2, 2).density_fit()
mc.verbose = 0
mc.max_cycle_macro = 40
mc = mc.state_average_([0.5, 0.5])
t1 = time.time()
mc.kernel()
print("SA-CASSCF(2,2) neutral done", time.time()-t1, flush=True)
print("mc.converged", mc.converged, flush=True)
print("e_states (Ha):", mc.e_states, flush=True)
gap_optical = (mc.e_states[1]-mc.e_states[0])*27.211386245988
print("neutral (optical-like) excitation gap eV:", gap_optical, flush=True)
