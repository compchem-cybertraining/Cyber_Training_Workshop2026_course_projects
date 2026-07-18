#!/usr/bin/env python3
"""
Reads every beta_rescaling_<label>.csv produced by beta_rescaling_scan.py in
the current directory, fits fundamental_gap_ccsd_eV vs 1/natoms for each
(three ways: all points, largest-4, quadratic-on-largest-5, same convention
used throughout this project's other extrapolations), and computes
k = extrapolated_gap / pbe_gap_eV for each point/fit combination.

Safe to rerun any time after beta_rescaling_scan.py has produced or added to
the per-point CSVs. Writes beta_rescaling_k_summary.csv and prints a table
sorted by delta so the k(delta) trend is easy to read off directly.
"""

import csv
import glob
import re
import numpy as np

MIN_POINTS_FOR_EXTRAPOLATION = 4


def read_point_csv(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'ok':
                rows.append(dict(
                    natoms=int(row['natoms']),
                    lattice_length=float(row['lattice_length']),
                    delta=float(row['delta']),
                    pbe_gap_eV=float(row['pbe_gap_eV']),
                    fundamental_gap_ccsd_eV=float(row['fundamental_gap_ccsd_eV']),
                ))
    return rows


def extrapolate_1_over_n(natoms_list, values):
    x = 1.0 / np.array(natoms_list, dtype=float)
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def summarize_point(label, rows):
    rows = sorted(rows, key=lambda r: r['natoms'])
    natoms_list = [r['natoms'] for r in rows]
    values = [r['fundamental_gap_ccsd_eV'] for r in rows]
    pbe_gap = rows[0]['pbe_gap_eV']
    delta = rows[0]['delta']
    lattice_length = rows[0]['lattice_length']

    result = dict(label=label, lattice_length=lattice_length, delta=delta,
                  pbe_gap_eV=pbe_gap, n_points=len(rows))

    if len(rows) < MIN_POINTS_FOR_EXTRAPOLATION:
        result['note'] = f'only {len(rows)} converged points, need >= {MIN_POINTS_FOR_EXTRAPOLATION}'
        return result

    intercept_all, _ = extrapolate_1_over_n(natoms_list, values)
    n4, v4 = natoms_list[-4:], values[-4:]
    intercept_4, _ = extrapolate_1_over_n(n4, v4)
    quad = np.polyfit(1.0 / np.array(natoms_list[-5:], dtype=float),
                       values[-5:], 2) if len(rows) >= 5 else None
    intercept_quad = float(quad[-1]) if quad is not None else None

    result.update(
        extrap_gap_all_points=intercept_all,
        extrap_gap_largest4=intercept_4,
        extrap_gap_quadratic=intercept_quad,
        k_all_points=intercept_all / pbe_gap,
        k_largest4=intercept_4 / pbe_gap,
        k_quadratic=(intercept_quad / pbe_gap) if intercept_quad is not None else None,
    )
    return result


def main():
    paths = sorted(glob.glob('beta_rescaling_*.csv'))
    paths = [p for p in paths if not re.search(r'k_summary', p)]
    if not paths:
        print("No beta_rescaling_<label>.csv files found in this directory.")
        return

    summaries = []
    for path in paths:
        label = re.sub(r'^beta_rescaling_|\.csv$', '', path)
        rows = read_point_csv(path)
        if not rows:
            print(f"{path}: no converged rows, skipping")
            continue
        summaries.append(summarize_point(label, rows))

    summaries.sort(key=lambda s: s['delta'])

    print(f"{'label':10s} {'delta':>7s} {'pbe_gap':>9s} {'extrap(all)':>12s} "
          f"{'extrap(4)':>10s} {'extrap(quad)':>13s} {'k(all)':>8s} {'k(4)':>8s} {'k(quad)':>8s}")
    for s in summaries:
        if 'note' in s:
            print(f"{s['label']:10s} {s['delta']:7.4f}  -- {s['note']}")
            continue
        print(f"{s['label']:10s} {s['delta']:7.4f} {s['pbe_gap_eV']:9.4f} "
              f"{s['extrap_gap_all_points']:12.4f} {s['extrap_gap_largest4']:10.4f} "
              f"{(s['extrap_gap_quadratic'] or float('nan')):13.4f} "
              f"{s['k_all_points']:8.3f} {s['k_largest4']:8.3f} "
              f"{(s['k_quadratic'] or float('nan')):8.3f}")

    fieldnames = sorted({k for s in summaries for k in s.keys()})
    with open('beta_rescaling_k_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print("\nSaved beta_rescaling_k_summary.csv.")
    print("Read k(delta) across rows above (sorted by delta) -- watch specifically whether "
          "k is leveling off toward the largest-delta points, which is the regime this "
          "rescaling approach is actually trustworthy in (small delta is the near-metallic "
          "regime where every method in this project has broken down, so don't expect or "
          "chase convergence there). If k plateaus at large delta, use that plateau value; "
          "if it's still trending, more/larger-delta points are needed before picking a "
          "final k.")


if __name__ == '__main__':
    main()
