#!/usr/bin/env python3
"""
Finite-size convergence of a fully CLOSED-SHELL "ballpark" exciton-binding
recipe:

    fundamental gap (Koopmans) = eps_LUMO - eps_HOMO, from the plain closed-
                                  shell RHF calculation on the neutral system
                                  -- no cation, no anion, no open-shell SCF
                                  at all.
    optical gap                = CIS/TDA lowest singlet excitation (also
                                  closed-shell, neutral system).
    binding energy              = koopmans_gap - optical_gap

Why the switch from the previous (open-shell dSCF/UHF) version: that
approach required converging UHF on the radical cation and anion at every
cluster size, and the data it produced was NOT converging -- EA plateaued
around 4.7-4.9 eV but IP kept falling without leveling off through
natoms=16, the two 1/natoms extrapolation fits (all points vs largest 4)
actively disagreed MORE as more data was added, and one point (natoms=14)
had an outright non-converged cation. The likely cause: UHF on a near-
degenerate, metal-like radical doesn't have one obvious minimum -- it has
several competing symmetry-broken solutions, and which one the SCF lands on
can shift as the cluster grows. That's the same multiple-local-minima
pathology CASSCF showed earlier, resurfacing in a cheaper method.

The TB model this is meant to inform is closed-shell throughout (the
exciton is a same-electron-count, two-particle excitation, not a charged
state), so there was never a strict need to leave the closed-shell world in
the first place -- IP/EA via charged states was only ever a means to
estimate the fundamental gap. Koopmans' theorem gives that same fundamental
gap directly from the neutral system's own orbital energies, with no
separate SCF, no radical, no multiple-minima risk.

Checked before writing this: natoms=2/4/6/8 at delta=0.0864 gives Koopmans
gaps 7.10 -> 5.67 -> 5.13 -> 4.86 eV and implied binding energies
4.84 -> 3.98 -> 3.52 -> 3.34 eV -- both series smooth and monotonic, with
steadily shrinking increments (0.86 -> 0.46 -> 0.18 eV). That is what an
actually-converging series looks like, unlike the open-shell data.

Honest caveats, both push the same direction (this recipe likely
OVERESTIMATES the true binding energy, i.e. gives an upper-bound-ish
ballpark, not a precise number):
- Koopmans neglects orbital relaxation (rigorously an upper bound on the
  true IP -- relaxation only ever lowers the cation's energy relative to
  the frozen/Koopmans estimate).
- CIS/TDA uses the bare, unscreened Coulomb interaction for the electron-
  hole attraction (the same "HF+CIS overbinds excitons" issue flagged
  earlier in this project) -- real screening from the rest of the system's
  electrons would reduce this.
If a better number is needed later than this ballpark provides, the next
rung that's STILL built on a single closed-shell reference (no open-shell
SCF, no multiple-minima risk) is extended Koopmans' theorem or EOM-IP/EA-
CCSD -- more expensive, not attempted here.

Grid: natoms in [2,4,6,8,10,12,14,16,20,24], delta=0.0864 only, at
lattice_length=6.0 A. Much cheaper than the old UHF-based version (no
open-shell SCF, no stability loop), so the grid is extended further to give
the 1/natoms extrapolation more of a tail to work with.
"""

import csv
import time
import numpy as np
from pyscf import gto, scf, tdscf

HARTREE_TO_EV = 27.211386245988

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
LATTICE_LENGTH = 6.0
DELTA = 0.0864
NATOMS_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24]


def build_mole(natoms, lattice_length, delta, verbose=0):
    assert natoms % 2 == 0, "natoms must be even (clean dimer pairs, no dangling end atom)"
    half = lattice_length / 2.0
    d_short = half * (1 - delta)
    d_long = half * (1 + delta)
    positions = [0.0]
    for i in range(natoms - 1):
        bond = d_short if i % 2 == 0 else d_long
        positions.append(positions[-1] + bond)
    atom_str = "; ".join(f"Ag {p:.6f} 0 0" for p in positions)
    return gto.M(atom=atom_str, basis=BASIS, pseudo=PSEUDO, verbose=verbose)


def run_point(natoms, lattice_length, delta):
    row = dict(natoms=natoms, lattice_length=lattice_length, delta=delta, status='ok')
    try:
        t0 = time.time()

        mol = build_mole(natoms, lattice_length, delta)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.max_cycle = 150
        mf.kernel()

        nocc = mol.nelectron // 2
        koopmans_gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV

        td = tdscf.TDA(mf)
        td.nstates = 3
        td.kernel()
        optical_gap = td.e[0] * HARTREE_TO_EV

        wall = time.time() - t0
        binding_energy = koopmans_gap - optical_gap

        row.update(
            converged=mf.converged,
            e_tot=mf.e_tot,
            koopmans_gap_eV=koopmans_gap,
            optical_gap_eV=optical_gap,
            binding_energy_eV=binding_energy,
            nao=mol.nao,
            wall_time_s=wall,
        )
    except Exception as exc:
        row['status'] = f'FAILED: {exc}'
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def extrapolate_1_over_n(natoms_list, values):
    x = 1.0 / np.array(natoms_list, dtype=float)
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def run_extrapolation(rows):
    ok_rows = [r for r in rows if r.get('status') == 'ok']
    if len(ok_rows) < 4:
        print("Not enough converged points to extrapolate (need >=4).", flush=True)
        return []

    ok_rows = sorted(ok_rows, key=lambda r: r['natoms'])
    natoms_list = [r['natoms'] for r in ok_rows]
    quantities = ['koopmans_gap_eV', 'optical_gap_eV', 'binding_energy_eV']

    summary_rows = []
    for q in quantities:
        values = [r[q] for r in ok_rows]
        intercept_all, slope_all = extrapolate_1_over_n(natoms_list, values)
        n4, v4 = natoms_list[-4:], values[-4:]
        intercept_last4, slope_last4 = extrapolate_1_over_n(n4, v4) if len(n4) == 4 else (None, None)
        agreement = abs(intercept_all - intercept_last4) if intercept_last4 is not None else None
        summary_rows.append(dict(
            quantity=q,
            intercept_all_points=intercept_all, slope_all_points=slope_all,
            intercept_largest_4=intercept_last4, slope_largest_4=slope_last4,
            agreement=agreement,
        ))
        print(f"{q}: extrapolated (all points) = {intercept_all:.4f} eV; "
              f"extrapolated (largest 4) = {intercept_last4:.4f} eV; "
              f"difference = {agreement:.4f} eV", flush=True)

    return summary_rows


def main():
    rows = []
    for natoms in NATOMS_VALUES:
        print(f"=== natoms={natoms}, delta={DELTA} ===", flush=True)
        row = run_point(natoms, LATTICE_LENGTH, DELTA)
        print(f"  -> {row}", flush=True)
        rows.append(row)
        write_csv('exciton_binding_convergence.csv', rows)  # checkpoint after every point

    print("\n--- 1/natoms extrapolation ---", flush=True)
    summary_rows = run_extrapolation(rows)
    write_csv('exciton_binding_extrapolation.csv', summary_rows)

    print("\nDone. See exciton_binding_convergence.csv and exciton_binding_extrapolation.csv.",
          flush=True)
    print("Expect this to look much better behaved than the old UHF-based version: "
          "koopmans_gap_eV and binding_energy_eV should both decrease smoothly and "
          "monotonically with natoms, with shrinking increments, no sign flips, no "
          "outlier points. If that holds out to natoms=20-24, the 'all points' vs "
          "'largest 4' extrapolation agreement is the thing to check before trusting "
          "the extrapolated binding_energy_eV as the ballpark -- and remember it's "
          "expected to be an OVERESTIMATE (missing relaxation + missing screening), "
          "not a precise number.", flush=True)


if __name__ == '__main__':
    main()
