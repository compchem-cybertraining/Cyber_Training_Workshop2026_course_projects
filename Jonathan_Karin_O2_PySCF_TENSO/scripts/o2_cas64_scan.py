#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
from pyscf import gto, scf, mcscf, fci

HARTREE_TO_EV = 27.211386245988

OUTPUT_FILE = Path("data/o2_cas64_scan.csv")

# Include additional short-bond points for a better minimum fit.
BOND_LENGTHS = np.arange(1.04, 1.341, 0.02)


def calculate_states(r_angstrom):
    mol = gto.M(
        atom=f"""
        O 0.0 0.0 {-r_angstrom / 2.0:.8f}
        O 0.0 0.0 { r_angstrom / 2.0:.8f}
        """,
        basis="cc-pvdz",
        charge=0,
        spin=2,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )

    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()

    if not mf.converged:
        mf = mf.newton()
        mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"ROHF failed at R = {r_angstrom:.4f} Angstrom"
        )

    # pi_u(x), pi_u(y), pi_g*(x), pi_g*(y)
    active_orbitals = [5, 6, 7, 8]

    # ---------------------------------------------------------
    # Triplet CAS(6,4): 4 alpha and 2 beta active electrons
    # ---------------------------------------------------------
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

    e_triplet, _, ci_triplet, _, _ = mc_triplet.kernel(
        mo_triplet
    )

    ss_triplet, mult_triplet = (
        mc_triplet.fcisolver.spin_square(
            ci_triplet,
            mc_triplet.ncas,
            mc_triplet.nelecas,
        )
    )

    # ---------------------------------------------------------
    # Singlet CAS(6,4): 3 alpha and 3 beta active electrons
    # ---------------------------------------------------------
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

    e_singlet, _, ci_singlet, _, _ = mc_singlet.kernel(
        mo_singlet
    )

    ss_singlet, mult_singlet = (
        mc_singlet.fcisolver.spin_square(
            ci_singlet,
            mc_singlet.ncas,
            mc_singlet.nelecas,
        )
    )

    gap_hartree = e_singlet - e_triplet
    gap_ev = gap_hartree * HARTREE_TO_EV

    return {
        "bond_length_angstrom": r_angstrom,
        "triplet_energy_hartree": e_triplet,
        "singlet_energy_hartree": e_singlet,
        "gap_hartree": gap_hartree,
        "gap_ev": gap_ev,
        "triplet_s2": ss_triplet,
        "singlet_s2": ss_singlet,
        "triplet_multiplicity": mult_triplet,
        "singlet_multiplicity": mult_singlet,
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "bond_length_angstrom",
        "triplet_energy_hartree",
        "singlet_energy_hartree",
        "gap_hartree",
        "gap_ev",
        "triplet_s2",
        "singlet_s2",
        "triplet_multiplicity",
        "singlet_multiplicity",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for r_angstrom in BOND_LENGTHS:
            print(
                f"Calculating CAS(6,4) at "
                f"R = {r_angstrom:.2f} Angstrom",
                flush=True,
            )

            result = calculate_states(r_angstrom)

            writer.writerow(result)
            csvfile.flush()

            print(
                f"  E_triplet = "
                f"{result['triplet_energy_hartree']:.10f} Ha",
                flush=True,
            )
            print(
                f"  E_singlet = "
                f"{result['singlet_energy_hartree']:.10f} Ha",
                flush=True,
            )
            print(
                f"  Gap       = "
                f"{result['gap_ev']:.6f} eV",
                flush=True,
            )

    print(f"\nSaved scan to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
