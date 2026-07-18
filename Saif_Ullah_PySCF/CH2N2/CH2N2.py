import os
import numpy as np
from pyscf import gto, dft, tddft, hessian  # Removed cc
from pyscf.tools import molden, cubegen

# Set environmental multi-threading to match your 4-core allocation
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

mol_geometry = """
    C 0.00000 0.00000 0.00000
    N 0.00000 0.00000 1.30230
    N 0.00000 0.00000 2.44190
    H 0.93282 0.00000 -0.56306
    H -0.93282 0.00000 -0.56306
"""

basis_sets = ['aug-cc-pVDZ', 'aug-cc-pVTZ', 'aug-cc-pVQZ']

for basis in basis_sets:
    print(f"\n{'='*40}\nDIAZOMETHANE - Basis: {basis}\n{'='*40}")
    
    mol = gto.Mole()
    # Adjusted max_memory to 16000 MB (16 GB) for a 4-core environment
    mol.build(
        atom = mol_geometry, 
        basis = basis, 
        verbose = 4, 
        charge = 0, 
        max_memory = 16000
    )
    
    # 1. Ground State DFT with Density Fitting
    mf = dft.RKS(mol).density_fit()
    mf.xc = 'cam-b3lyp'
    mf.kernel()
    
    # 2. TDDFT / RPA Calculation
    mytd = tddft.TDDFT(mf)
    mytd.nstates = 10
    mytd.singlet = True
    mytd.kernel()
    mytd.analyze()
    
    # 3. Generate Unique NTO files
    for i in range(mytd.nstates):
        weights, nto = mytd.get_nto(state=i+1, verbose=4)
        molden.from_mo(mol, f"ch2n2_nto-rpa-{basis}-state_{i+1}.molden", nto)

    # 4. UPDATED: Density Difference Loop for ALL Calculated Excited States
    for state_idx in range(mytd.nstates):
        cis_t1 = np.asarray(mytd.xy[state_idx][0])

        dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
        dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

        dm_mo = np.diag(mf.mo_occ)
        nocc = cis_t1.shape[0]
        dm_mo[:nocc, :nocc] += dm_oo * 2
        dm_mo[nocc:, nocc:] += dm_vv * 2

        dm_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm_mo, mf.mo_coeff.conj())

        # Unique prefix for Diazomethane
        cube_filename = f'ch2n2_diff_{basis}_state_{state_idx+1}.cube'
        cubegen.density(mol, cube_filename, dm_ao - mf.make_rdm1())
        
    
    # 5. Vibrational Frequencies (Conditional Safety Block)
    if basis != 'aug-cc-pVQZ':
        print(f"Calculating Vibrational Hessian for CH2N2 ({basis})...")
        hobj = hessian.rks.Hessian(mf).kernel()
    else:
        print(f"Skipping heavy Vibrational Hessian for CH2N2 ({basis}) to optimize 4-core runtime.")

