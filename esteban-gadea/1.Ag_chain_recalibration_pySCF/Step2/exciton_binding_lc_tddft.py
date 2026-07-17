#!/usr/bin/env python3
"""
Test #1 of two, both aimed at the same question: is the ~3 eV (HF+CIS) /
~1.2-1.4 eV extrapolated (CCSD fundamental + CIS optical) binding energy
still too large mainly because the optical-gap step (CIS/TDA) uses the
BARE, unscreened Coulomb interaction for the electron-hole attraction?

This script swaps CIS/TDA for TDA on top of a range-separated hybrid KS
functional (default: CAM-B3LYP) -- a standard, literature-precedented way
to estimate exciton binding energies (E_b = QP gap - optical gap) in
molecular/organic-semiconductor contexts, where the range-separated
functional's response kernel carries genuine correlation/screening that
bare HF-based CIS lacks entirely.

Important framing: this changes ONLY the optical-gap side. The fundamental
gap paired with it is still the plain closed-shell RHF Koopmans gap
(eps_LUMO - eps_HOMO), computed in this SAME script via a separate RHF run
on the same geometry -- not the KS orbital energies (which aren't rigorous
IPs the way HF's are). That isolates the one change being tested, the same
way the EOM-CCSD script isolated a change to the fundamental-gap side while
holding the optical side fixed at CIS. Don't compare across both scripts
naively -- each only changes one leg of the subtraction relative to the
original all-HF baseline.

Checked before writing this (natoms=2, delta=0.0864, lattice_length=6.0):
  RHF Koopmans gap:      7.101 eV  (unchanged from before)
  CIS/TDA optical gap:   2.263 eV  (unchanged from before, for reference)
  CAM-B3LYP TDA optical: 3.153 eV  (noticeably larger than CIS)
  => binding (Koopmans - CAM-B3LYP): 3.948 eV, vs 4.838 eV via CIS.
At natoms=4: CAM-B3LYP optical=2.326 eV (vs CIS's 1.690 eV), binding=3.346
eV (vs Koopmans+CIS's 3.982 eV). Same direction and similar-sized effect to
what correlating the fundamental gap did with CCSD -- consistent with the
idea that the missing physics is the same thing (screening/correlation of
the e-h interaction) showing up on whichever side of the calculation you
let it into.

Cost warning -- steeper than CIS, though nowhere near CCSD's wall:
natoms=2 ran in ~2s, natoms=4 in ~14s, and a first attempt at natoms=6
(default density-fitting off, default grid level 3) did NOT finish in 44s.
Turning on density fitting for the KS Fock build (`.density_fit()`) and
dropping the DFT integration grid to level=2 (from the default 3) brought
natoms=6 down to ~22s total. Both are on by default below as the practical
recipe for this sensitivity test -- they introduce small (sub-0.05 eV
scale, not checked precisely) numerical looseness relative to the
default/production DFT settings, which is an acceptable trade for a
screening ballpark, not something to use for a final production number
without tightening back up. Default grid here is therefore kept modest,
[2, 4, 6, 8] -- try natoms=8 alone first if you want a cheaper read before
committing to the whole thing; expect it to take a few minutes.

Alternative functionals: CAM-B3LYP is the default (common literature choice
for exciton binding energy work) -- swap FUNCTIONAL below to 'wb97x' or
'lc-blyp' for other range-separated options if you want to check
sensitivity to the specific functional/range-separation parameter, though
that wasn't tested here.
"""

import csv
import time
from pyscf import scf, dft, tdscf
from exciton_binding_convergence import build_mole, HARTREE_TO_EV

LATTICE_LENGTH = 6.0
DELTA = 0.0864
NATOMS_VALUES = [2, 4, 6, 8]
FUNCTIONAL = 'camb3lyp'


def run_point(natoms, lattice_length, delta):
    row = dict(natoms=natoms, lattice_length=lattice_length, delta=delta,
               functional=FUNCTIONAL, status='ok')
    try:
        t0 = time.time()
        mol = build_mole(natoms, lattice_length, delta)

        # Plain RHF Koopmans gap -- same definition as every earlier script,
        # kept fixed so this test isolates the optical-gap change only.
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.max_cycle = 150
        mf.kernel()
        nocc = mol.nelectron // 2
        koopmans_gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV
        t_hf = time.time() - t0

        # Range-separated hybrid KS + TDA for the optical gap.
        t1 = time.time()
        mfk = dft.RKS(mol).density_fit()
        mfk.xc = FUNCTIONAL
        mfk.verbose = 0
        mfk.grids.level = 2
        mfk.max_cycle = 150
        mfk.kernel()
        t_ks = time.time() - t1

        td = tdscf.TDA(mfk)
        td.nstates = 3
        td.kernel()
        lc_optical_gap = td.e[0] * HARTREE_TO_EV
        t_td = time.time() - t1 - t_ks

        binding_energy = koopmans_gap - lc_optical_gap

        row.update(
            hf_converged=mf.converged,
            ks_converged=mfk.converged,
            koopmans_gap_eV=koopmans_gap,
            lc_optical_gap_eV=lc_optical_gap,
            binding_energy_eV=binding_energy,
            nao=mol.nao,
            wall_hf_s=t_hf,
            wall_ks_s=t_ks,
            wall_td_s=t_td,
            wall_total_s=time.time() - t0,
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


def main():
    rows = []
    for natoms in NATOMS_VALUES:
        print(f"=== natoms={natoms}, delta={DELTA}, xc={FUNCTIONAL} ===", flush=True)
        row = run_point(natoms, LATTICE_LENGTH, DELTA)
        print(f"  -> {row}", flush=True)
        rows.append(row)
        write_csv('exciton_binding_lc_tddft.csv', rows)  # checkpoint after every point

    print("\nDone. See exciton_binding_lc_tddft.csv.", flush=True)
    print("Compare binding_energy_eV here against the CIS-based Koopmans+CIS series "
          "(exciton_binding_convergence.csv) at the same natoms: if the range-separated "
          "functional's optical gap is consistently larger than CIS's (as seen at "
          "natoms=2 and 4), binding_energy_eV here should be consistently smaller -- "
          "further evidence that a chunk of the earlier ~3 eV number was the bare-Coulomb "
          "CIS optical gap sitting artificially low, not necessarily the fundamental gap "
          "being wrong.", flush=True)


if __name__ == '__main__':
    main()
