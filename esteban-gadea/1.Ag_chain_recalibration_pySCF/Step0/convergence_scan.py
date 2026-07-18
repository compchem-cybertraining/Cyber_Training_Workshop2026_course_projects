#!/usr/bin/env python3
"""
Convergence / timing scan for periodic Ag chain KRHF calculations in pySCF.

Runs three independent scans and writes one CSV per scan:
  1. k-points along the chain axis, at fixed (generous) vacuum      -> kpoint_scan.csv
  2. transverse vacuum spacing, at fixed moderate k-mesh            -> vacuum_scan.csv
  3. cell size (natoms = 2, 4, 6, ...) at Gamma-only, fixed vacuum  -> natoms_scan.csv

Purpose: get wall-clock timing + convergence behavior (energy, gap) BEFORE
committing to the full production sweep needed for the new (HF/post-HF based)
TB reparametrization. Nothing here is meant to be a final scientific number --
it's a timing/convergence reconnaissance run.

Each run is wrapped in try/except so one failure/non-convergence doesn't
kill the whole sweep -- failures are logged in the CSV with a note.
"""

import csv
import sys
import time

from ag_chain_lib import build_cell, run_krhf


def run_one(natoms, lattice_length, delta, vacuum, nk, basis, pseudo):
    try:
        cell = build_cell(natoms=natoms, lattice_length=lattice_length,
                           delta=delta, vacuum=vacuum, basis=basis, pseudo=pseudo)
        result = run_krhf(cell, nk)
        result['status'] = 'ok'
    except Exception as exc:
        result = dict(natm=natoms, nk=nk, status=f'FAILED: {exc}',
                       e_tot=None, e_tot_per_atom=None, wall_time_s=None,
                       direct_gap_gamma_eV=None, indirect_gap_eV=None,
                       nao=None, nelectron=None, converged=None)
    result['natoms_requested'] = natoms
    result['lattice_length'] = lattice_length
    result['delta'] = delta
    result['vacuum'] = vacuum
    result['basis'] = basis
    result['pseudo'] = pseudo
    return result


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
    basis = 'gth-szv-molopt-sr'
    pseudo = 'gth-pbe'

    # Reference geometry: 6.0 A lattice length, delta=0.0864
    # (bonds 2.74 / 3.26 A) -- matches the single-chain report's primary case.
    lattice_length = 6.0
    delta = 0.0864

    t_start = time.time()

    # ---- Scan 1: k-points along chain, fixed generous vacuum ----
    print("=== Scan 1: k-point convergence (natoms=2, vacuum=20 A) ===", flush=True)
    kpoint_rows = []
    for nk in [1, 2, 4, 8, 16]:
        print(f"  nk={nk} ...", flush=True)
        row = run_one(natoms=2, lattice_length=lattice_length, delta=delta,
                       vacuum=20.0, nk=nk, basis=basis, pseudo=pseudo)
        print(f"    -> {row['status']}, wall={row.get('wall_time_s')}, "
              f"E={row.get('e_tot')}, gap(direct)={row.get('direct_gap_gamma_eV')}", flush=True)
        kpoint_rows.append(row)
        write_csv('kpoint_scan.csv', kpoint_rows)  # checkpoint after each point

    # ---- Scan 2: transverse vacuum spacing, fixed moderate k-mesh ----
    print("=== Scan 2: vacuum-spacing convergence (natoms=2, nk=8) ===", flush=True)
    vacuum_rows = []
    for vac in [8.0, 12.0, 16.0, 20.0, 25.0, 30.0]:
        print(f"  vacuum={vac} A ...", flush=True)
        row = run_one(natoms=2, lattice_length=lattice_length, delta=delta,
                       vacuum=vac, nk=8, basis=basis, pseudo=pseudo)
        print(f"    -> {row['status']}, wall={row.get('wall_time_s')}, "
              f"E={row.get('e_tot')}, gap(direct)={row.get('direct_gap_gamma_eV')}", flush=True)
        vacuum_rows.append(row)
        write_csv('vacuum_scan.csv', vacuum_rows)

    # ---- Scan 3: cell size (natoms), Gamma-only, fixed vacuum ----
    # Gives raw cost-vs-system-size scaling at the HF level, useful for
    # extrapolating to the natoms=2/4/6 points needed for the new model,
    # and as a baseline before adding post-HF (CISD/CASSCF) cost multipliers.
    print("=== Scan 3: cell-size scaling (nk=1, vacuum=20 A) ===", flush=True)
    natoms_rows = []
    for natoms in [2, 4, 6]:
        print(f"  natoms={natoms} ...", flush=True)
        row = run_one(natoms=natoms, lattice_length=lattice_length, delta=delta,
                       vacuum=20.0, nk=1, basis=basis, pseudo=pseudo)
        print(f"    -> {row['status']}, wall={row.get('wall_time_s')}, "
              f"E={row.get('e_tot')}, gap(direct)={row.get('direct_gap_gamma_eV')}", flush=True)
        natoms_rows.append(row)
        write_csv('natoms_scan.csv', natoms_rows)

    print(f"=== Done in {time.time() - t_start:.1f} s total ===", flush=True)
    print("Results: kpoint_scan.csv, vacuum_scan.csv, natoms_scan.csv", flush=True)


if __name__ == '__main__':
    main()
