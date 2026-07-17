"""
Helper library for building dimerized periodic Ag chains in pySCF and running
KRHF single points, timed, for convergence testing.

Approach: "3D wire array" — a genuinely 3D-periodic Cell (dimension=3, the
default and best-tested code path) with a large vacuum gap in the two
transverse directions, rather than pySCF's native dimension=1 mode (whose
low-dimensional Coulomb kernel is not well converged/accurate for 1D wires).
Convergence must therefore be checked explicitly in two places:
  (1) k-point sampling along the periodic (chain) axis -> approach to the
      long-chain / bulk limit.
  (2) transverse box size (interchain/interwire spacing) -> approach to the
      isolated-wire limit, removing spurious wire-wire coupling.

Pseudopotential / basis: gth-pbe (PBE-consistent GTH pseudopotential) with
gth-szv-molopt-sr basis. This is the only Ag combination bundled by default
with pySCF; it treats 4d^10 5s^1 (11 electrons) as valence -- i.e. it also
carries the filled 4d shell, not just the 5s band your TB model represents.
Keep that in mind when interpreting gaps/energies, but for a pure
convergence/timing scan it does not matter.
"""

import time
import numpy as np
from pyscf.pbc import gto, scf

HARTREE_TO_EV = 27.211386245988

DEFAULT_BASIS = 'gth-szv-molopt-sr'
DEFAULT_PSEUDO = 'gth-pbe'


def build_cell(natoms=2, lattice_length=6.0, delta=0.0864, vacuum=20.0,
               basis=DEFAULT_BASIS, pseudo=DEFAULT_PSEUDO, verbose=0):
    """Build a dimerized Ag chain as a 3D-periodic Cell (wire-in-a-box).

    natoms          : number of Ag atoms in the repeating cell (even; 2, 4, 6, ...)
    lattice_length  : repeat length of one dimer pair (A-A distance x2), Angstrom
                      (matches the "lattice length" convention in the TB report,
                      e.g. 6.0 A corresponds to bonds 2.74/3.26 A at delta=0.0864)
    delta           : dimerization amplitude (fractional), bonds = (lattice_length/2)*(1 -+ delta)
    vacuum          : transverse box size (Angstrom) in y and z -- the
                      interchain/interwire spacing to be converged
    """
    assert natoms % 2 == 0, "natoms must be even (dimerized chain of pairs)"

    half = lattice_length / 2.0
    d_short = half * (1 - delta)
    d_long = half * (1 + delta)

    positions = [0.0]
    for i in range(natoms - 1):
        bond = d_short if i % 2 == 0 else d_long
        positions.append(positions[-1] + bond)
    last_bond = d_short if (natoms - 1) % 2 == 0 else d_long
    axis_length = positions[-1] + last_bond  # should equal (natoms/2)*lattice_length

    atom_lines = "\n".join(f"Ag {p:.6f} 0.0 0.0" for p in positions)

    cell = gto.Cell()
    cell.a = np.array([[axis_length, 0.0, 0.0],
                        [0.0, vacuum, 0.0],
                        [0.0, 0.0, vacuum]])
    cell.atom = atom_lines
    cell.basis = basis
    cell.pseudo = pseudo
    cell.dimension = 3          # explicit: plain 3D periodicity, no low-dim kernel
    cell.verbose = verbose
    cell.build()
    return cell


def run_krhf(cell, nk):
    """Run KRHF with Gaussian density fitting, timed. nk = # k-points along chain axis."""
    kpts = cell.make_kpts([nk, 1, 1])
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.verbose = 0
    mf.max_cycle = 100

    t0 = time.time()
    mf.kernel()
    wall = time.time() - t0

    nocc = cell.nelectron // 2
    homo_k = [mo[nocc - 1] for mo in mf.mo_energy]
    lumo_k = [mo[nocc] for mo in mf.mo_energy]
    direct_gap_gamma = (lumo_k[0] - homo_k[0]) * HARTREE_TO_EV
    indirect_gap = (min(lumo_k) - max(homo_k)) * HARTREE_TO_EV

    return dict(
        converged=mf.converged,
        e_tot=mf.e_tot,
        e_tot_per_atom=mf.e_tot / cell.natm,
        wall_time_s=wall,
        direct_gap_gamma_eV=direct_gap_gamma,
        indirect_gap_eV=indirect_gap,
        nao=cell.nao,
        natm=cell.natm,
        nelectron=cell.nelectron,
        nk=nk,
    )
