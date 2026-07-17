#!/usr/bin/env python3
"""
Test #2 of two: instead of switching to a different quantum-chemistry
method to get "some" screening (script #1, range-separated TDDFT), put in
a single explicit screening knob by hand and ask directly: what dielectric
constant would the electron-hole attraction need to be divided by for the
binding energy to land in your model's usable range (~0.5 eV)? Then judge
whether that number is a plausible dielectric constant for this system, or
an unreasonably large one that would point to something other than simple
dielectric screening being the missing physics.

How: the closed-shell singlet CIS/TDA matrix (Szabo & Ostlund form) is

    A_{ia,jb} = delta_ij delta_ab (eps_a - eps_i) + 2(ia|jb) - (ij|ab)

built here by hand from partial MO-basis two-electron integral transforms
(ao2mo.general on (occ,vir,occ,vir) and (occ,occ,vir,vir) orbital subsets --
NOT the full nmo^4 tensor, which would be prohibitively large in memory at
these sizes; the partial transforms only scale as nocc^2*nvir^2, e.g. ~60 MB
at natoms=12, versus ~1 GB for the full tensor). Validated against pyscf's
own machinery before writing this: diagonalizing this hand-built A at
natoms=2 gives 2.24957 eV, matching pyscf's own tdscf.TDA(mf).get_ab()
matrix diagonalized the same way (2.24957 eV) to 7 decimal places -- so the
formula and index bookkeeping are correct. (That's ~0.014 eV below pyscf's
own td.kernel() value of 2.26320 eV; get_ab() vs kernel() differ slightly
in how pyscf solves the same underlying problem -- a pyscf-internal detail,
not a bug in this construction.) At natoms=8, this reproduces the existing
CIS/TDA optical gap (1.52289 eV) essentially exactly, confirming the
approach scales correctly.

The screening model: divide the ENTIRE two-electron coupling term (both the
"2(ia|jb)" direct piece and the "(ij|ab)" exchange piece that actually binds
the exciton) by a single scalar epsilon:

    A(epsilon) = diag(eps_a - eps_i)  +  (1/epsilon) * [2(ia|jb) - (ij|ab)]

As epsilon -> 1, this is exactly bare CIS (no screening). As epsilon -> oo,
the coupling vanishes and the lowest eigenvalue -> the Koopmans gap, i.e.
binding_energy -> 0 -- the qualitatively correct limit (fully screened
system = no bound exciton, just the bare single-particle gap). This is a
crude, single-parameter stand-in for real GW+BSE screening (which would
treat the direct and exchange pieces differently, and would make epsilon
itself frequency- and length-scale dependent rather than a single number)
-- good for a sensitivity check / ballpark, not a rigorous screened-BSE
calculation.

Per natoms, this scans epsilon over a fixed grid and also solves (by linear
interpolation between bracketing grid points) for epsilon_for_target_eV:
the epsilon at which binding_energy_eV = TARGET_BINDING_EV (default 0.5 eV,
matching the TB model's own |alpha| instability threshold). That number is
the one to sanity-check: is it a plausible static dielectric constant for a
few-atom-wide Ag chain, or implausibly large?

Checked before writing this (natoms=2, delta=0.0864): epsilon_for_target
(0.5 eV) came out around ~7.7 (crossing between epsilon=5, binding=0.767 eV,
and epsilon=8, binding=0.472 eV). At natoms=8, the whole binding_energy(eps)
curve is already lower at every epsilon (since binding_energy(eps=1)=1.523
optical vs bare koopmans -- consistent with the already-established
finite-size trend), so epsilon_for_target should come down with natoms too;
worth checking whether it converges to something size-independent or keeps
dropping.

Cost: comparable to plain CIS, i.e. much cheaper than CCSD or LC-TDDFT.
natoms=8 ran in ~10.5s (RHF ~2.2s, the two partial ao2mo transforms ~4s each,
diagonalization ~0.2s); natoms=12 is in the ~35-45s range extrapolating
from a partial timing (RHF ~7s, one ao2mo transform ~12s). Grid below goes
to natoms=12 to match the existing CCSD/Koopmans data directly; extending
further is plausible (cost here scales much better than CCSD) but untested.
"""

import csv
import time
import numpy as np
from pyscf import scf, ao2mo
from exciton_binding_convergence import build_mole, HARTREE_TO_EV

LATTICE_LENGTH = 6.0
DELTA = 0.0864
NATOMS_VALUES = [2, 4, 6, 8, 10, 12]
EPS_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0, 30.0]
TARGET_BINDING_EV = 0.5


def build_scaled_cis_matrix(mol, mf):
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    mo_occ = mf.mo_coeff[:, :nocc]
    mo_vir = mf.mo_coeff[:, nocc:]

    iajb = ao2mo.general(mol, (mo_occ, mo_vir, mo_occ, mo_vir), compact=False)
    iajb = iajb.reshape(nocc, nvir, nocc, nvir)
    ijab = ao2mo.general(mol, (mo_occ, mo_occ, mo_vir, mo_vir), compact=False)
    ijab = ijab.reshape(nocc, nocc, nvir, nvir)

    nov = nocc * nvir
    term1 = iajb.reshape(nov, nov)                       # (ia|jb), already [i,a,j,b]
    term2 = ijab.transpose(0, 2, 1, 3).reshape(nov, nov)  # (ij|ab) reordered to [i,a,j,b]

    d = (mf.mo_energy[nocc:][None, :] - mf.mo_energy[:nocc][:, None]).reshape(nov)
    A0 = np.diag(d)
    V = 2 * term1 - term2  # coupling term to be divided by epsilon
    return A0, V


def optical_gap_for_eps(A0, V, eps):
    A = A0 + (1.0 / eps) * V
    w = np.linalg.eigvalsh(A)
    return float(w[0]) * HARTREE_TO_EV


def find_eps_for_target(eps_list, binding_list, target):
    """Linear interpolation between bracketing grid points where binding_energy
    crosses target. Assumes binding_list is monotonically non-increasing in
    eps_list (true for this screening model). Returns None if never bracketed
    (target outside the tested epsilon range)."""
    for i in range(len(eps_list) - 1):
        b0, b1 = binding_list[i], binding_list[i + 1]
        if (b0 - target) * (b1 - target) <= 0 and b0 != b1:
            e0, e1 = eps_list[i], eps_list[i + 1]
            frac = (b0 - target) / (b0 - b1)
            return e0 + frac * (e1 - e0)
    return None


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
        t_hf = time.time() - t0

        t1 = time.time()
        A0, V = build_scaled_cis_matrix(mol, mf)
        t_ao2mo = time.time() - t1

        t2 = time.time()
        optical_gaps = [optical_gap_for_eps(A0, V, eps) for eps in EPS_GRID]
        binding_energies = [koopmans_gap - og for og in optical_gaps]
        eps_for_target = find_eps_for_target(EPS_GRID, binding_energies, TARGET_BINDING_EV)
        t_scan = time.time() - t2

        row.update(
            hf_converged=mf.converged,
            koopmans_gap_eV=koopmans_gap,
            bare_optical_gap_eV=optical_gaps[0],   # eps=1, should match CIS/TDA
            bare_binding_energy_eV=binding_energies[0],
            eps_for_target_binding=eps_for_target,
            target_binding_eV=TARGET_BINDING_EV,
            nao=mol.nao,
            wall_hf_s=t_hf,
            wall_ao2mo_s=t_ao2mo,
            wall_scan_s=t_scan,
            wall_total_s=time.time() - t0,
        )
        # full epsilon-resolved curve, for plotting / manual inspection
        for eps, og, be in zip(EPS_GRID, optical_gaps, binding_energies):
            row[f'optical_gap_eV_eps{eps:g}'] = og
            row[f'binding_energy_eV_eps{eps:g}'] = be
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
        print(f"=== natoms={natoms}, delta={DELTA} ===", flush=True)
        row = run_point(natoms, LATTICE_LENGTH, DELTA)
        print(f"  -> koopmans={row.get('koopmans_gap_eV')}, "
              f"bare_binding(eps=1)={row.get('bare_binding_energy_eV')}, "
              f"eps_for_{TARGET_BINDING_EV}eV_binding={row.get('eps_for_target_binding')}, "
              f"status={row.get('status')}", flush=True)
        rows.append(row)
        write_csv('exciton_binding_screened_cis.csv', rows)  # checkpoint after every point

    print("\nDone. See exciton_binding_screened_cis.csv.", flush=True)
    print("Key column: eps_for_target_binding -- the dielectric constant that would need "
          "to screen the bare e-h attraction for binding_energy to equal "
          f"{TARGET_BINDING_EV} eV, at each natoms. Watch whether this number converges to "
          "something roughly size-independent (suggesting a real, physically meaningful "
          "screening strength) or keeps drifting with natoms (suggesting this single-"
          "parameter model is too crude to pin down, same caution as everywhere else in "
          "this project's finite-size scans). Also check bare_binding_energy_eV (eps=1) "
          "against the existing CIS values in exciton_binding_convergence.csv as a "
          "consistency check -- they should match almost exactly at each natoms.", flush=True)


if __name__ == '__main__':
    main()
