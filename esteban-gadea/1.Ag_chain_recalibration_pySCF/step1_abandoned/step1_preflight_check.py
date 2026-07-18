#!/usr/bin/env python3
"""
Preflight check for submit_step1_array.slurm. Two things checked per point,
at the real production recipe (nk=16, vacuum=30 A):

1. Convergence robustness at the low-delta (near-undimerized) end -- none of
   Step 0's testing touched this regime. (Same as before.)

2. Orbital-character contamination. A quick investigation at delta=0 found
   that the naive "just take nocc-1/nocc by energy" gap extraction can pick
   up the WRONG orbital at some k-points: at Gamma, the literal LUMO is a
   5p-derived state, not the 5s antibonding partner the TB model actually
   means -- the true s-antibonding orbital sits ~4 eV further up the virtual
   manifold at that k-point. This function does a proper Mulliken (overlap-
   weighted) population analysis at every k to find the actual dominant-5s
   occupied and virtual orbitals, independent of raw energy ordering, and
   compares the resulting "s-band-only" gap against the naive one.

   At delta=0 (nk=4, cheap settings, checked interactively) these two
   numbers came out IDENTICAL (3.4292 eV both ways) -- the contamination at
   Gamma doesn't actually corrupt the reported indirect gap there, because
   the true minimum across the k-mesh happens to live at k=pi, which was
   never contaminated in the first place. So the earlier concern about
   character contamination corrupting the numbers looks to be a non-issue
   AT THAT POINT -- but that was only checked at one (l, delta) pair and one
   k-mesh size. This diagnostic is built into the preflight (and left here
   as a standing check) so any OTHER grid point where naive and s-resolved
   gaps disagree gets caught before it contaminates the beta(r) fit.

   Separately: since the s-band gap does NOT vanish at delta=0 (~3 eV, not
   character contamination), the leading remaining explanation is HF's own
   well-documented tendency to overestimate/spuriously open gaps in 1D
   half-filled systems (missing screening/correlation -- the same effect
   long documented for HF treatments of polyacetylene). That's a property
   of the method, not a bug in this script, but worth keeping in mind when
   Step 1's surface is used to find "equilibrium" delta and fit beta(r): the
   delta=0 gap floor may need to be subtracted or otherwise accounted for
   rather than assumed to vanish.
"""

import time
import numpy as np
from pyscf.pbc import scf as pbcscf

from ag_chain_lib import build_cell

HARTREE_TO_EV = 27.211386245988

DELTAS_TO_CHECK = [0.00, 0.01, 0.02, 0.05, 0.0864, 0.20]
LATTICE_LENGTH = 6.0
NK = 16
VACUUM = 30.0
BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'


def mulliken_s_weight(c, Sk, s_mask):
    """Overlap-weighted (Mulliken) population of MO column c on the s_mask AO
    subset. Correctly normalized (populations over all AOs sum to ~1 for a
    normalized MO), unlike a raw sum of squared coefficients, which is biased
    toward high-lying diffuse virtuals with large raw coefficients."""
    Sc = Sk.dot(c)
    pop = np.real(np.conj(c) * Sc)
    return float(pop[s_mask].sum())


def find_s_band_gap(cell, mf, kpts, nocc, search_below=3, search_above=10):
    """Character-resolved analogue of the naive nocc-1/nocc gap: at every k,
    pick the occupied orbital (within search_below of nocc) and virtual
    orbital (within search_above of nocc) with the highest 5s Mulliken
    weight, rather than assuming nocc-1/nocc are automatically the right
    ones. Returns the resulting indirect gap plus per-k diagnostics."""
    labels = cell.ao_labels()
    s_mask = np.array(['5s' in lab for lab in labels])
    Ss = cell.pbc_intor('int1e_ovlp', kpts=kpts)

    s_homo_e, s_lumo_e, diag = [], [], []
    for k_idx, (mo_e, mo_c, Sk) in enumerate(zip(mf.mo_energy, mf.mo_coeff, Ss)):
        occ_range = range(max(0, nocc - search_below), nocc)
        virt_range = range(nocc, min(mo_c.shape[1], nocc + search_above))
        o_idx, o_w = max(((i, mulliken_s_weight(mo_c[:, i], Sk, s_mask)) for i in occ_range),
                          key=lambda x: x[1])
        v_idx, v_w = max(((i, mulliken_s_weight(mo_c[:, i], Sk, s_mask)) for i in virt_range),
                          key=lambda x: x[1])
        s_homo_e.append(mo_e[o_idx])
        s_lumo_e.append(mo_e[v_idx])
        diag.append(dict(k_idx=k_idx, occ_idx=o_idx, occ_s_weight=round(o_w, 3),
                          virt_idx=v_idx, virt_s_weight=round(v_w, 3)))

    s_indirect_gap = (min(s_lumo_e) - max(s_homo_e)) * HARTREE_TO_EV
    return s_indirect_gap, diag


def run_and_diagnose(lattice_length, delta):
    row = dict(lattice_length=lattice_length, delta=delta, status='ok')
    try:
        cell = build_cell(natoms=2, lattice_length=lattice_length, delta=delta,
                           vacuum=VACUUM, basis=BASIS, pseudo=PSEUDO)
        kpts = cell.make_kpts([NK, 1, 1])

        t0 = time.time()
        mf = pbcscf.KRHF(cell, kpts=kpts).density_fit()
        mf.verbose = 0
        mf.max_cycle = 100
        mf.kernel()
        used_smearing = False
        if not mf.converged:
            from pyscf import scf as molscf
            mf = pbcscf.KRHF(cell, kpts=kpts).density_fit()
            mf.verbose = 0
            mf.max_cycle = 100
            mf = molscf.addons.smearing_(mf, sigma=0.005, method='fermi')
            mf.kernel()
            used_smearing = True
        wall = time.time() - t0

        nocc = cell.nelectron // 2
        homo_k = [mo[nocc - 1] for mo in mf.mo_energy]
        lumo_k = [mo[nocc] for mo in mf.mo_energy]
        naive_gap = (min(lumo_k) - max(homo_k)) * HARTREE_TO_EV

        s_gap, diag = find_s_band_gap(cell, mf, kpts, nocc)

        row.update(
            converged=mf.converged, used_smearing=used_smearing, wall_time_s=wall,
            naive_indirect_gap_eV=naive_gap, s_band_indirect_gap_eV=s_gap,
            gap_mismatch=abs(naive_gap - s_gap) > 0.01,
            s_band_diagnostics=diag,
        )
    except Exception as exc:
        row['status'] = f'FAILED: {exc}'
    return row


if __name__ == '__main__':
    print(f"Preflight check at lattice_length={LATTICE_LENGTH} A, "
          f"deltas={DELTAS_TO_CHECK}, nk={NK}, vacuum={VACUUM}\n", flush=True)
    problems = []
    for delta in DELTAS_TO_CHECK:
        row = run_and_diagnose(LATTICE_LENGTH, delta)
        print(f"delta={delta}: naive={row.get('naive_indirect_gap_eV')}, "
              f"s_band={row.get('s_band_indirect_gap_eV')}, "
              f"mismatch={row.get('gap_mismatch')}, "
              f"used_smearing={row.get('used_smearing')}, status={row.get('status')}",
              flush=True)
        for d in row.get('s_band_diagnostics', []):
            print(f"    {d}", flush=True)
        if row.get('status') != 'ok' or row.get('used_smearing') or row.get('gap_mismatch'):
            problems.append((delta, row.get('status'), row.get('used_smearing'), row.get('gap_mismatch')))

    print("\n--- Summary ---", flush=True)
    if not problems:
        print("All points converged cleanly, no smearing needed, and naive/s-band gaps "
              "agree everywhere. Safe to submit the full array.", flush=True)
    else:
        print("Points needing attention (failed, needed smearing, or naive != s-band gap):", flush=True)
        for delta, status, used_smearing, mismatch in problems:
            print(f"  delta={delta}: status={status}, used_smearing={used_smearing}, "
                  f"gap_mismatch={mismatch}", flush=True)
        print("A gap_mismatch=True point means the naive extraction was picking up a "
              "p-derived (or other non-s) state at the k-point that sets the min/max -- "
              "trust s_band_indirect_gap_eV there, not naive_indirect_gap_eV, and consider "
              "promoting this character-resolved extraction into step1_surface_scan.py "
              "itself before running the full grid.", flush=True)
