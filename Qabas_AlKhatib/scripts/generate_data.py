"""
CASSCF(6,6)/def2-SVP data generation script
Author: Qabas Mohammad Al Khatib
CyberTraining 2026
"""
import os
import glob
import pickle
import numpy as np
from pyscf import gto, scf, mcscf

XYZ_DIR = os.path.expanduser("~/CyberTraining2026/ml-hessian-casscf/data/raw/xyz")
OUT_DIR = os.path.expanduser("~/CyberTraining2026/ml-hessian-casscf/data/processed")
PKL_FILE = os.path.join(OUT_DIR, "training_data_def2svp.pkl")
BASIS = "def2-svp"
NCAS = 6
NELECAS = 6

os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(PKL_FILE):
    with open(PKL_FILE, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} existing results")
else:
    results = {}

xyz_files = sorted(glob.glob(os.path.join(XYZ_DIR, "*.xyz")))
print(f"Found {len(xyz_files)} molecules total")

def read_xyz(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    return ''.join(lines[2:])

for f in xyz_files:
    name = os.path.splitext(os.path.basename(f))[0]
    if name in results:
        print(f"Skipping {name} (already done)")
        continue
    print(f"Running {name}...")
    try:
        mol = gto.M(atom=read_xyz(f), basis=BASIS, verbose=0)
        mf = scf.RHF(mol).run()
        mc = mcscf.CASSCF(mf, NCAS, NELECAS)
        mc.kernel()
        if mc.converged:
            casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
            eris = mc.ao2mo()
            g_orb, _, h_op, hdiag = mc.gen_g_hop(mc.mo_coeff, mc.ci, casdm1, casdm2, eris)
            h1cas, ecore = mc.get_h1cas()
            h2cas = mc.get_h2cas()
            ci_hdiag = mc.fcisolver.make_hdiag(h1cas, h2cas, mc.ncas, mc.nelecas)
            results[name] = {
                'energy': mc.e_tot,
                'g_orb': g_orb,
                'hdiag': hdiag,
                'ci_hdiag': ci_hdiag,
                'ci_vector': mc.ci,
                'grad_norm': float(np.linalg.norm(g_orb)),
            }
            print(f"  OK: {mc.e_tot:.6f} Ha")
            with open(PKL_FILE, "wb") as pf:
                pickle.dump(results, pf)
        else:
            print(f"  FAILED: did not converge")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDone! {len(results)}/23 molecules succeeded.")
