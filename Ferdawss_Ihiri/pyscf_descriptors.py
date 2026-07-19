#!/usr/bin/env python3
"""Compute inexpensive electronic-structure descriptors for PAH101 monomers.

This script performs a Hartree-Fock single-point calculation with the
STO-3G basis set, followed by a configuration interaction singles (CIS)
calculation of the lowest singlet excited state for each extracted PAH101
monomer using PySCF. The resulting electronic descriptors serve as
low-cost features for subsequent machine learning models that predict
GW/BSE electronic properties.

Run with your pyscf Python environment:
    pyscf_env/bin/python pyscf_descriptors.py
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from pyscf import gto, scf, tddft

import numpy as np

import config as cfg


def read_xyz(path: Path):
    lines = path.read_text().strip().splitlines()
    nat = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + nat]:
        parts = line.split()
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return atoms


FIELDS = [
    "csd",
    "n_atoms",
    "n_basis",
    "scf_converged",
    "scf_energy_Ha",
    "homo_lumo_gap_eV",
    "cis_s1_eV",
    "cis_s1_osc_strength",
    "dipole_debye",
    "wall_time_s",
]


def run_one(xyz_path: Path, basis: str, nstates: int, max_memory: int) -> dict:

    t0 = time.time()
    atoms = read_xyz(xyz_path)
    mol = gto.M(atom=atoms, basis=basis, verbose=0, max_memory=max_memory)
    mf = scf.RHF(mol)
    mf.kernel()

    nocc = mol.nelectron // 2
    mo_e = mf.mo_energy
    gap_ev = float((mo_e[nocc] - mo_e[nocc - 1]) * 27.211386245988)

    s1_ev, s1_f = None, None
    if mf.converged:
        td = tddft.TDA(mf)
        td.nstates = nstates
        td.kernel()
        if len(td.e) > 0:
            s1_ev = float(td.e[0] * 27.211386245988)
            try:
                fosc = td.oscillator_strength(gauge="length")
                s1_f = float(fosc[0])
            except Exception:
                s1_f = None

    dip = mf.dip_moment(unit="Debye", verbose=0)
    dip_mag = float(np.linalg.norm(dip))

    return {
        "csd": xyz_path.name.replace("_monomer.xyz", ""),
        "n_atoms": mol.natm,
        "n_basis": mol.nao,
        "scf_converged": bool(mf.converged),
        "scf_energy_Ha": float(mf.e_tot),
        "homo_lumo_gap_eV": gap_ev,
        "cis_s1_eV": s1_ev,
        "cis_s1_osc_strength": s1_f,
        "dipole_debye": dip_mag,
        "wall_time_s": round(time.time() - t0, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default=cfg.CHEAP_BASIS)
    p.add_argument("--nstates", type=int, default=cfg.CHEAP_NSTATES)
    p.add_argument("--max-memory", type=int, default=cfg.CHEAP_MAX_MEMORY_MB)
    p.add_argument("--max-atoms", type=int, default=cfg.MAX_MONOMER_ATOMS)
    p.add_argument("--limit", type=int, default=None, help="cap number of molecules (debug)")
    args = p.parse_args()

    xyz_files = sorted(cfg.GEOM_DIR.glob("*_monomer.xyz"))
    out_csv = cfg.RESULTS_DIR / "cheap_descriptors.csv"
    failed_csv = cfg.RESULTS_DIR / "failed_pyscf.csv"

    done = set()
    if out_csv.exists():
        with out_csv.open() as f:
            done = {row["csd"] for row in csv.DictReader(f)}
        print(f"Resuming: {len(done)} molecules already done in {out_csv.name}")

    permanently_failed = set()
    if failed_csv.exists():
        with failed_csv.open() as f:
            permanently_failed = {row["csd"] for row in csv.DictReader(f)}
        if permanently_failed:
            print(f"Skipping {len(permanently_failed)} previously-failed molecules")
    done |= permanently_failed

    write_header = not out_csv.exists()
    write_failed_header = not failed_csv.exists()
    n_skipped_size, n_run, n_failed = 0, 0, 0

    failed_f = failed_csv.open("a", newline="")
    failed_writer = csv.DictWriter(failed_f, fieldnames=["csd", "error"])
    if write_failed_header:
        failed_writer.writeheader()

    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        candidates = []
        for xyz in xyz_files:
            csd = xyz.name.replace("_monomer.xyz", "")
            if csd in done:
                continue
            nat = int(xyz.read_text().splitlines()[0])
            if nat > args.max_atoms:
                n_skipped_size += 1
                continue
            candidates.append(xyz)
        if args.limit:
            candidates = candidates[: args.limit]

        print(
            f"{len(xyz_files)} monomers total | {len(done)} already done | "
            f"{n_skipped_size} skipped (> {args.max_atoms} atoms) | "
            f"{len(candidates)} to run now"
        )

        for i, xyz in enumerate(candidates, start=1):
            csd = xyz.name.replace("_monomer.xyz", "")
            try:
                rec = run_one(xyz, args.basis, args.nstates, args.max_memory)
                writer.writerow(rec)
                f.flush()
                n_run += 1
                print(
                    f"[{i}/{len(candidates)}] {csd}: gap={rec['homo_lumo_gap_eV']:.2f} eV "
                    f"S1={rec['cis_s1_eV']}  ({rec['wall_time_s']:.1f} s)"
                )
            except Exception as e:
                n_failed += 1
                failed_writer.writerow({"csd": csd, "error": str(e)})
                failed_f.flush()
                print(f"[{i}/{len(candidates)}] {csd}: FAILED ({e})")

    failed_f.close()
    print(f"\nDone. Ran {n_run}, failed {n_failed}, skipped-for-size {n_skipped_size}.")
    print(f"Results in {out_csv}")


if __name__ == "__main__":
    main()
