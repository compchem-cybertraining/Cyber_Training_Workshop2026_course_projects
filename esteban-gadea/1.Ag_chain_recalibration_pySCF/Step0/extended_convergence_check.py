#!/usr/bin/env python3
"""
Follow-up to convergence_scan.py. Two things left open from the first round:

1. The k-point scan (vacuum=20) and vacuum scan (nk=8) were each done holding
   the other parameter at a value that wasn't itself fully converged. This
   runs one joint point (nk=16, vacuum=30, natoms=2) to confirm the two
   convergences are independent/additive rather than needing further joint
   refinement.

2. CASSCF/CISD/CASPT2 need a real-space (Gamma-only) cluster, not a k-mesh.
   Gamma-only supercells only sample the true zone-boundary band edge when
   natoms/2 is even (see natoms_scan.csv: natoms=8,12 track the converging
   branch, natoms=2,6,14 do not). This extends that *even* branch to
   natoms=16 to see how close a Gamma-only cluster of feasible size can get
   to the k/vacuum-converged answer -- i.e. how big a cluster you actually
   need for the correlated-method ladder.

Results appended to extended_check.csv (checkpointed after each point, same
pattern as convergence_scan.py).
"""

import time
from ag_chain_lib import build_cell, run_krhf
from convergence_scan import run_one, write_csv

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
LATTICE_LENGTH = 6.0
DELTA = 0.0864


def main():
    rows = []
    t0 = time.time()

    print("=== Joint convergence check: natoms=2, nk=16, vacuum=30 ===", flush=True)
    row = run_one(natoms=2, lattice_length=LATTICE_LENGTH, delta=DELTA,
                   vacuum=30.0, nk=16, basis=BASIS, pseudo=PSEUDO)
    print(f"  -> {row['status']}, wall={row.get('wall_time_s')}, "
          f"indirect_gap={row.get('indirect_gap_eV')}", flush=True)
    rows.append(row)
    write_csv('extended_check.csv', rows)

    print("=== Gamma-only natoms=16 (even branch), vacuum=20 (matches natoms_scan.csv) ===", flush=True)
    row = run_one(natoms=16, lattice_length=LATTICE_LENGTH, delta=DELTA,
                   vacuum=20.0, nk=1, basis=BASIS, pseudo=PSEUDO)
    print(f"  -> {row['status']}, wall={row.get('wall_time_s')}, "
          f"indirect_gap={row.get('indirect_gap_eV')}", flush=True)
    rows.append(row)
    write_csv('extended_check.csv', rows)

    print("=== Gamma-only natoms=16, vacuum=30 (matches the joint-converged target) ===", flush=True)
    row = run_one(natoms=16, lattice_length=LATTICE_LENGTH, delta=DELTA,
                   vacuum=30.0, nk=1, basis=BASIS, pseudo=PSEUDO)
    print(f"  -> {row['status']}, wall={row.get('wall_time_s')}, "
          f"indirect_gap={row.get('indirect_gap_eV')}", flush=True)
    rows.append(row)
    write_csv('extended_check.csv', rows)

    print(f"=== Done in {time.time()-t0:.1f} s ===", flush=True)
    print("Compare extended_check.csv row 1 (the joint-converged reference) against "
          "rows 2-3 (natoms=16 Gamma-only) to see how close a feasible finite cluster "
          "gets to the converged periodic answer.", flush=True)


if __name__ == '__main__':
    main()
