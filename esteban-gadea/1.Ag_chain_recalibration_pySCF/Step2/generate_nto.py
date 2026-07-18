#!/usr/bin/env python3
"""
Natural Transition Orbitals (NTOs) for the lowest CIS/TDA excited state, at
one representative geometry (natoms=8, lattice_length=6.0, delta=0.0864 --
the same reference geometry used throughout Step 2). Doesn't need to be the
best level of theory (plain RHF+CIS, already validated extensively in this
project, is enough) -- the point is a qualitative/visual one: show that the
excited electron predominantly occupies an Ag 5s-derived orbital, which is
the direct justification for representing the conduction band with a single
s-orbital-derived TB band in the first place.

Method: pyscf's tdscf.TDA.get_nto(state=1) decomposes the CIS transition
density into hole/electron NTO pairs ranked by weight. Character is checked
with a plain Mulliken population (mol.intor('int1e_ovlp') restricted to the
'5s' AO label subset) on the dominant pair. Results are in
report/objective1_report.md Section 4.5.

Outputs hole/electron NTO cube files (viewable in VMD, PyMOL, Avogadro,
VESTA, etc. -- isosurface at roughly +-0.02-0.05 is a reasonable starting
point for these) plus nto_character_summary.csv with the weights and
s-character numbers for the top few pairs.
"""

import csv
import numpy as np
from pyscf import scf, tdscf, tools
from exciton_binding_convergence import build_mole, HARTREE_TO_EV

NATOMS = 8
LATTICE_LENGTH = 6.0
DELTA = 0.0864
N_PAIRS_TO_REPORT = 3
OUT_DIR = '../report/nto_cubes'


def mulliken_weight(c, S, mask):
    Sc = S.dot(c)
    pop = np.real(np.conj(c) * Sc)
    return float(pop[mask].sum())


def main():
    mol = build_mole(NATOMS, LATTICE_LENGTH, DELTA)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    td = tdscf.TDA(mf)
    td.nstates = 3
    td.kernel()
    print(f"Lowest CIS/TDA excitation energy: {td.e[0]*HARTREE_TO_EV:.4f} eV")

    weights, nto_coeff = td.get_nto(state=1, verbose=0)
    nocc = mol.nelectron // 2

    S = mol.intor('int1e_ovlp')
    labels = mol.ao_labels()
    s_mask = np.array(['5s' in lab for lab in labels])

    rows = []
    for i in range(N_PAIRS_TO_REPORT):
        hole_idx = nocc - 1 - i
        elec_idx = nocc + i
        if hole_idx < 0 or elec_idx >= nto_coeff.shape[1]:
            break
        hole_nto = nto_coeff[:, hole_idx]
        elec_nto = nto_coeff[:, elec_idx]
        hole_s = mulliken_weight(hole_nto, S, s_mask)
        elec_s = mulliken_weight(elec_nto, S, s_mask)
        w = weights[i] if i < len(weights) else float('nan')
        print(f"  pair {i}: weight={w:.4f}, hole 5s-character={hole_s:.4f}, "
              f"electron 5s-character={elec_s:.4f}")
        rows.append(dict(pair=i, weight=w, hole_5s_character=hole_s, electron_5s_character=elec_s))

        if i == 0:
            tools.cubegen.orbital(mol, f'{OUT_DIR}/hole_nto.cube', hole_nto)
            tools.cubegen.orbital(mol, f'{OUT_DIR}/electron_nto.cube', elec_nto)
            print(f"  -> saved {OUT_DIR}/hole_nto.cube and {OUT_DIR}/electron_nto.cube "
                  f"(dominant pair)")

    with open('nto_character_summary.csv', 'w', newline='') as f:
        w_csv = csv.DictWriter(f, fieldnames=['pair', 'weight', 'hole_5s_character', 'electron_5s_character'])
        w_csv.writeheader()
        for r in rows:
            w_csv.writerow(r)
    print("Saved nto_character_summary.csv")
    print("\nOpen hole_nto.cube / electron_nto.cube in VMD/PyMOL/Avogadro/VESTA and render "
          "isosurfaces (try +-0.02 to +-0.05) side by side for the report figure -- the hole "
          "should look p/d-like (no amplitude concentrated ON the Ag cores in an s-like way), "
          "the electron should look like a string of s-like lobes centered on the Ag atoms.")


if __name__ == '__main__':
    main()
