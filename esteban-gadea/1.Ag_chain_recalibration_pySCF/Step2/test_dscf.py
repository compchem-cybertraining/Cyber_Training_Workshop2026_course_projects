import copy, time
from pyscf.pbc import scf as pbcscf
from pyscf import scf as molscf, tdscf
from ag_chain_lib import build_cell

HARTREE_TO_EV = 27.211386245988
t0 = time.time()

cell = build_cell(natoms=2, lattice_length=6.0, delta=0.0, vacuum=20.0)

# neutral RHF + CIS (optical / excitonic gap)
mf0 = pbcscf.RHF(cell).density_fit()
mf0.verbose = 0
mf0.kernel()
td = tdscf.TDA(mf0)
td.nstates = 1
td.kernel()
optical_gap = td.e[0] * HARTREE_TO_EV
print("neutral RHF E:", mf0.e_tot, " CIS/TDA optical gap eV:", optical_gap, flush=True)

def charged_cell(base, charge, spin):
    m = copy.copy(base)
    m.charge = charge
    m.spin = spin
    m.nelectron = base.nelectron - charge
    return m

# cation: relaxed UHF (self-consistent orbitals, not frozen)
cell_cat = charged_cell(cell, +1, 1)
mf_cat = pbcscf.UHF(cell_cat).density_fit()
mf_cat.verbose = 0
mf_cat.kernel()
print("cation UHF converged:", mf_cat.converged, "E:", mf_cat.e_tot, flush=True)

# anion
cell_an = charged_cell(cell, -1, 1)
mf_an = pbcscf.UHF(cell_an).density_fit()
mf_an.verbose = 0
mf_an.kernel()
print("anion UHF converged:", mf_an.converged, "E:", mf_an.e_tot, flush=True)

IP = (mf_cat.e_tot - mf0.e_tot) * HARTREE_TO_EV
EA = (mf0.e_tot - mf_an.e_tot) * HARTREE_TO_EV
fundamental_gap = IP - EA
print(f"IP={IP:.4f} EA={EA:.4f} fundamental(dSCF)={fundamental_gap:.4f} eV", flush=True)
print(f"optical(CIS)={optical_gap:.4f} eV  ballpark_binding={fundamental_gap-optical_gap:.4f} eV", flush=True)
print("total time", time.time()-t0, flush=True)
