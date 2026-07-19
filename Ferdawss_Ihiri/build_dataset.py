#!/usr/bin/env python3
"""Glue the cheap PySCF numbers to the expensive PAH101 GW/BSE labels.

Output: results/dataset.csv -- one row per crystal that has both a working
cheap-PySCF run and at least one real GW/BSE target to predict.
"""
from __future__ import annotations

import csv

import config as cfg


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    with cfg.PAH101_CSV.open() as f:
        pah_rows = {r["CSD Reference Code"]: r for r in csv.DictReader(f, delimiter="\t")}

    cheap_path = cfg.RESULTS_DIR / "cheap_descriptors.csv"
    with cheap_path.open() as f:
        cheap_rows = list(csv.DictReader(f))

    fieldnames = (
        ["csd"]
        + ["n_atoms", "n_basis", "homo_lumo_gap_eV", "cis_s1_eV", "cis_s1_osc_strength", "dipole_debye"]
        + cfg.CHEAP_TABULATED_FEATURES
        + cfg.TARGET_COLUMNS
    )

    n_written, n_skipped = 0, 0
    out_path = cfg.RESULTS_DIR / "dataset.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for crow in cheap_rows:
            csd = crow["csd"]
            prow = pah_rows.get(csd)
            if prow is None:
                n_skipped += 1
                continue
            if crow.get("scf_converged") != "True" or crow.get("cis_s1_eV") in (None, "", "None"):
                n_skipped += 1
                continue

            targets = {t: _f(prow.get(t)) for t in cfg.TARGET_COLUMNS}
            if all(v is None for v in targets.values()):
                n_skipped += 1
                continue

            record = {
                "csd": csd,
                "n_atoms": crow["n_atoms"],
                "n_basis": crow["n_basis"],
                "homo_lumo_gap_eV": crow["homo_lumo_gap_eV"],
                "cis_s1_eV": crow["cis_s1_eV"],
                "cis_s1_osc_strength": crow["cis_s1_osc_strength"],
                "dipole_debye": crow["dipole_debye"],
            }
            for feat in cfg.CHEAP_TABULATED_FEATURES:
                record[feat] = _f(prow.get(feat))
            record.update(targets)
            writer.writerow(record)
            n_written += 1

    print(f"Wrote {n_written} rows to {out_path} ({n_skipped} skipped).")


if __name__ == "__main__":
    main()
