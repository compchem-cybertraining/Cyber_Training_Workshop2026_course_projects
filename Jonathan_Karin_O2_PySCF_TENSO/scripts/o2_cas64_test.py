from pyscf import gto, scf, mcscf, fci

HARTREE_TO_EV = 27.211386245988

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
    verbose=3,
)

mf = scf.ROHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 100
mf.kernel()

if not mf.converged:
    raise RuntimeError("ROHF did not converge")

# pi_u(x), pi_u(y), pi_g*(x), pi_g*(y)
active_orbitals = [5, 6, 7, 8]

# ==========================================================
# Triplet CAS(6,4)
# Five alpha and one beta active electrons
# ==========================================================
mc_triplet = mcscf.CASCI(
    mf,
    ncas=4,
    nelecas=(4, 2),
)

mo_triplet = mcscf.sort_mo(
    mc_triplet,
    mf.mo_coeff,
    active_orbitals,
    base=0,
)

mc_triplet.fcisolver = fci.direct_spin1.FCI(mol)
mc_triplet.fcisolver.spin = 2

e_triplet, _, ci_triplet, _, _ = mc_triplet.kernel(mo_triplet)

ss_triplet, mult_triplet = mc_triplet.fcisolver.spin_square(
    ci_triplet,
    mc_triplet.ncas,
    mc_triplet.nelecas,
)

# ==========================================================
# Singlet CAS(6,4)
# Three alpha and three beta active electrons
# ==========================================================
mc_singlet = mcscf.CASCI(
    mf,
    ncas=4,
    nelecas=(3, 3),
)

mo_singlet = mcscf.sort_mo(
    mc_singlet,
    mf.mo_coeff,
    active_orbitals,
    base=0,
)

mc_singlet.fcisolver = fci.direct_spin0.FCI(mol)

e_singlet, _, ci_singlet, _, _ = mc_singlet.kernel(mo_singlet)

ss_singlet, mult_singlet = mc_singlet.fcisolver.spin_square(
    ci_singlet,
    mc_singlet.ncas,
    mc_singlet.nelecas,
)

gap_hartree = e_singlet - e_triplet
gap_ev = gap_hartree * HARTREE_TO_EV

print("\n" + "=" * 64)
print("O2 CAS(6,4) RESULTS AT R = 1.21 ANGSTROM")
print("=" * 64)

print(f"Triplet energy:        {e_triplet:.12f} Hartree")
print(f"Triplet <S^2>:         {ss_triplet:.8f}")
print(f"Triplet multiplicity:  {mult_triplet:.8f}")

print()

print(f"Singlet energy:        {e_singlet:.12f} Hartree")
print(f"Singlet <S^2>:         {ss_singlet:.8f}")
print(f"Singlet multiplicity:  {mult_singlet:.8f}")

print()

print(f"Singlet-triplet gap:   {gap_hartree:.12f} Hartree")
print(f"Singlet-triplet gap:   {gap_ev:.6f} eV")
print("=" * 64)
