#!/usr/bin/env python3
"""
Step 1: single-particle (HF) surface over lattice length x dimerization.

Uses the Step 0 recipe directly (small natoms=2 primitive cell, proper k-point
sampling, no supercell/parity concerns): nk=16 along the chain axis,
vacuum=30 A transverse. That combination measured at ~85s/point in Step 0's
joint convergence check (see Step0/README.md) -- accurate enough to use as-is
everywhere, no need to trade accuracy for speed here the way the CASSCF
timing work had to.

Grid: lattice_length in [5.2 .. 6.3] step 0.1 (12 values), delta in
[0.00 .. 0.20] step 0.01 (21 values) -- 252 points total, matching the old
PBE-based surface's grid.

Parallelization: embarrassingly parallel across lattice_length. This script
handles ONE lattice length per invocation (selected by --l-index, or by
SLURM_ARRAY_TASK_ID if that env var is set and --l-index isn't given) and
loops over all 21 deltas for it, checkpointing to its own CSV after every
point. Submit as a 12-way array (submit_step1_array.slurm) rather than one
long serial job.

Robustness: delta=0 is the undimerized chain -- physically metallic (no
single-particle gap, partially-filled band at the Fermi level), which plain
closed-shell RHF is not guaranteed to handle as cleanly as a gapped point. A
quick check (nk=4, small vacuum, delta=0) converged fine in plain RHF, but
that's not the full nk=16/vacuum=30 production setting, and other lattice
lengths near the metallic limit weren't checked at all. So: try plain RHF
first; if it doesn't converge, retry ONCE with Fermi-Dirac smearing
(sigma=0.005 Ha) as a rescue path. Smeared points are flagged
(used_smearing=True) in the output -- their reported energy is a finite-
smearing free energy, not directly on the same footing as the T=0 energies
from the plain path, so don't feed them into the beta(r)/equilibrium fit
without checking that flag first.

Data recorded per point (enough for BOTH equilibrium-finding from the energy
and beta1/beta2 inversion from gap+bandwidth, per the report's own relations
gap=2|beta1-beta2|, bandwidth=2*min(beta1,beta2)):
  e_tot, indirect_gap_eV, gap_at_pi_eV (sanity check vs indirect -- should
  match if the band edge is really at the zone boundary), valence_bandwidth_eV,
  conduction_bandwidth_eV, converged, used_smearing, wall_time_s.
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
from pyscf.pbc import scf as pbcscf
from pyscf import scf as molscf

from ag_chain_lib import build_cell

HARTREE_TO_EV = 27.211386245988

L_VALUES = [round(5.2 + 0.1 * i, 1) for i in range(12)]   # 5.2 .. 6.3
D_VALUES = [round(0.01 * i, 2) for i in range(21)]         # 0.00 .. 0.20

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
NK = 16
VACUUM = 30.0
SMEARING_SIGMA = 0.005   # Ha; rescue path only, see note above -- may need tuning


def run_point(lattice_length, delta):
    row = dict(lattice_length=lattice_length, delta=delta, status='ok', used_smearing=False)
    try:
        cell = build_cell(natoms=2, lattice_length=lattice_length, delta=delta,
                           vacuum=VACUUM, basis=BASIS, pseudo=PSEUDO)
        kpts = cell.make_kpts([NK, 1, 1])

        t0 = time.time()
        mf = pbcscf.KRHF(cell, kpts=kpts).density_fit()
        mf.verbose = 0
        mf.max_cycle = 100
        mf.kernel()

        if not mf.converged:
            # rescue path for near-degenerate/metallic points (small delta)
            mf = pbcscf.KRHF(cell, kpts=kpts).density_fit()
            mf.verbose = 0
            mf.max_cycle = 100
            mf = molscf.addons.smearing_(mf, sigma=SMEARING_SIGMA, method='fermi')
            mf.kernel()
            row['used_smearing'] = True

        wall = time.time() - t0

        nocc = cell.nelectron // 2
        homo_k = [mo[nocc - 1] for mo in mf.mo_energy]
        lumo_k = [mo[nocc] for mo in mf.mo_energy]
        indirect_gap = (min(lumo_k) - max(homo_k)) * HARTREE_TO_EV
        pi_index = NK // 2   # k=pi/a in an unshifted Monkhorst-Pack mesh of NK points
        gap_at_pi = (lumo_k[pi_index] - homo_k[pi_index]) * HARTREE_TO_EV
        valence_bandwidth = (max(homo_k) - min(homo_k)) * HARTREE_TO_EV
        conduction_bandwidth = (max(lumo_k) - min(lumo_k)) * HARTREE_TO_EV

        row.update(
            converged=mf.converged,
            e_tot=mf.e_tot,
            e_tot_per_atom=mf.e_tot / cell.natm,
            indirect_gap_eV=indirect_gap,
            gap_at_pi_eV=gap_at_pi,
            valence_bandwidth_eV=valence_bandwidth,
            conduction_bandwidth_eV=conduction_bandwidth,
            wall_time_s=wall,
            nao=cell.nao,
            nelectron=cell.nelectron,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--l-index', type=int, default=None,
                         help='index into L_VALUES; falls back to SLURM_ARRAY_TASK_ID if unset')
    args = parser.parse_args()

    idx = args.l_index
    if idx is None:
        idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1'))
    if idx < 0 or idx >= len(L_VALUES):
        print(f"No valid lattice-length index given (got {idx}); "
              f"pass --l-index 0..{len(L_VALUES)-1} or set SLURM_ARRAY_TASK_ID.", file=sys.stderr)
        sys.exit(1)

    lattice_length = L_VALUES[idx]
    out_path = f'step1_surface_l{lattice_length:.1f}.csv'
    print(f"=== Step 1 surface: lattice_length={lattice_length} A, "
          f"{len(D_VALUES)} delta points, nk={NK}, vacuum={VACUUM} A ===", flush=True)

    rows = []
    t_start = time.time()
    for delta in D_VALUES:
        print(f"  delta={delta} ...", flush=True)
        row = run_point(lattice_length, delta)
        print(f"    -> {row}", flush=True)
        rows.append(row)
        write_csv(out_path, rows)   # checkpoint after every point

    print(f"=== Done: lattice_length={lattice_length} in {time.time()-t_start:.1f} s. "
          f"See {out_path} ===", flush=True)


if __name__ == '__main__':
    main()
