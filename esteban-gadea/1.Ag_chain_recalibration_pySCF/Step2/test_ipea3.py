import copy, time
from pyscf.pbc import scf as pbcscf
from pyscf import mcscf
from ag_chain_lib import build_cell

HARTREE_TO_EV = 27.211386245988
t0 = time.time()
cell = build_cell(natoms=2, lattice_length=6.0, delta=0.0, vacuum=20.0)
mf = pbcscf.RHF(cell).density_fit()
mf.verbose = 0
mf.kernel()
print("HF done", time.time()-t0, "E=", mf.e_tot, flush=True)

NCORE = 10

mc0 = mcscf.CASCI(mf, 2, 2)
mc0.ncore = NCORE
mc0.fcisolver.nroots = 2
mc0.kernel()
e0_ground, e0_excited = mc0.e_tot[0], mc0.e_tot[1]
optical_gap = (e0_excited - e0_ground) * HARTREE_TO_EV
print("neutral ground/excited:", e0_ground, e0_excited, "optical gap eV:", optical_gap, flush=True)

def charged_mol(base_cell, nelectron):
    m = copy.copy(base_cell)
    m.nelectron = nelectron
    return m

mc_cat = mcscf.CASCI(mf, 2, (1, 0))
mc_cat.ncore = NCORE
mc_cat.mol = charged_mol(cell, 21)
mc_cat.kernel()
e_cat = mc_cat.e_tot
print("cation E:", e_cat, flush=True)

mc_an = mcscf.CASCI(mf, 2, (2, 1))
mc_an.ncore = NCORE
mc_an.mol = charged_mol(cell, 23)
mc_an.kernel()
e_an = mc_an.e_tot
print("anion E:", e_an, flush=True)

IP = (e_cat - e0_ground) * HARTREE_TO_EV
EA = (e0_ground - e_an) * HARTREE_TO_EV
fundamental_gap = IP - EA
print(f"IP={IP:.4f} EA={EA:.4f} fundamental(IP-EA)={fundamental_gap:.4f} eV", flush=True)
print(f"optical={optical_gap:.4f} eV  implied_exciton_binding={fundamental_gap-optical_gap:.4f} eV", flush=True)
print("total", time.time()-t0, flush=True)
