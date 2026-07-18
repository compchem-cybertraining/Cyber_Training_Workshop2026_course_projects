from pyscf import gto, scf, symm
import numpy as np

mol = gto.M(
    atom="""
    O 0.0 0.0 -0.605
    O 0.0 0.0  0.605
    """,
    basis="cc-pvdz",
    charge=0,
    spin=2,
    unit="Angstrom",
    symmetry=True,
    verbose=4,
)

mf = scf.ROHF(mol)
mf.conv_tol = 1e-10
mf.kernel()

if not mf.converged:
    raise RuntimeError("ROHF did not converge")

orb_syms = symm.label_orb_symm(
    mol,
    mol.irrep_name,
    mol.symm_orb,
    mf.mo_coeff,
)

print("\nMO index   Energy (Ha)   Occupation   Symmetry")
print("------------------------------------------------")

for i, (energy, occ, irrep) in enumerate(
    zip(mf.mo_energy, mf.mo_occ, orb_syms)
):
    print(f"{i:8d}   {energy:11.6f}   {occ:10.1f}   {irrep}")

singly_occupied = np.where(np.isclose(mf.mo_occ, 1.0))[0]

print("\nSingly occupied orbitals:")
for i in singly_occupied:
    print(
        f"MO {i}: energy = {mf.mo_energy[i]:.8f} Ha, "
        f"symmetry = {orb_syms[i]}"
    )

print("\nSpin diagnostic:", mf.spin_square())
