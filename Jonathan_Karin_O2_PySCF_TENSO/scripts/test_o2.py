from pyscf import gto, scf

mol = gto.M(
    atom="""
    O 0.0 0.0 0.0
    O 0.0 0.0 1.21
    """,
    basis="cc-pvdz",
    charge=0,
    spin=2,
    unit="Angstrom",
)

mf = scf.ROHF(mol)
energy = mf.kernel()

print("Converged:", mf.converged)
print("Triplet O2 ROHF energy:", energy, "Hartree")
print("<S^2> and multiplicity:", mf.spin_square())
