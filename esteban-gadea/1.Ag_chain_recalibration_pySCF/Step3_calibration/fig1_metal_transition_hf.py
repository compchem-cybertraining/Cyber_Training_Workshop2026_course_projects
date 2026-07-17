#!/usr/bin/env python3
"""
Cheap, illustrative (NOT production-quality) periodic HF gap-vs-delta scan
near delta=0, for Figure 1: demonstrating that HF does not close the gap at
the undimerized/metallic limit, unlike PBE (which gives exactly 0 eV at
delta=0 for every lattice length in DFT_bgap.csv).

Settings here (nk=4, vacuum=15 A) are deliberately much cheaper than the
production Step 0/Step 1 recipe (nk=16, vacuum=30 A, ~85s/point) -- this
figure only needs to make a qualitative point (HF gap doesn't vanish),
already independently confirmed at production settings during the Step 1
preflight check. Checkpointed CSV so it can be re-invoked across multiple
short calls without redoing finished points.
"""
import csv
import os
import time
from ag_chain_lib import build_cell, run_krhf

LATTICE_LENGTH = 6.0
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.0864]
NK = 4
VACUUM = 15.0
OUT_CSV = 'fig1_hf_delta_scan.csv'


def load_done():
    if not os.path.exists(OUT_CSV):
        return {}
    with open(OUT_CSV, newline='') as f:
        return {float(row['delta']): row for row in csv.DictReader(f)}


def main():
    budget_start = time.time()
    done = load_done()
    rows = list(done.values())
    for delta in DELTAS:
        if delta in done:
            continue
        if time.time() - budget_start > 30:
            print("Time budget reached this call; rerun to continue.", flush=True)
            break
        t0 = time.time()
        cell = build_cell(natoms=2, lattice_length=LATTICE_LENGTH, delta=delta, vacuum=VACUUM)
        r = run_krhf(cell, nk=NK)
        row = dict(delta=delta, indirect_gap_eV=r['indirect_gap_eV'],
                   converged=r['converged'], wall_time_s=time.time() - t0)
        print(row, flush=True)
        rows.append(row)
        rows.sort(key=lambda x: float(x['delta']))
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['delta', 'indirect_gap_eV', 'converged', 'wall_time_s'])
            w.writeheader()
            for rr in rows:
                w.writerow(rr)
    else:
        print("All points done.", flush=True)


if __name__ == '__main__':
    main()
