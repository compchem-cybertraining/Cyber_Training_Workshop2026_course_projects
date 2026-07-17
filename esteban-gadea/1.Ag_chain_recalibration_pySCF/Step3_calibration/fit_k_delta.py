#!/usr/bin/env python3
"""
Formalizes the k(delta) = k_inf + A/delta fit that was done ad hoc over the
course of the calibration discussion, using the 8-point scan in
Step2/beta_rescaling_k_summary.csv (produced by Step2/beta_rescaling_scan.py
+ beta_rescaling_summary.py).

Why this fit exists: k = (EOM-CCSD extrapolated fundamental gap) / (TB-model
gap at the same (l, delta)) diverges as delta -> 0. That's not noise -- the
TB-model/PBE gap correctly vanishes at delta=0 (the metal-to-semiconductor
transition), while the EOM-CCSD fundamental gap does NOT (it's built on an
HF reference, and HF/CCSD share the same "doesn't close the gap at the
metallic point" failure documented throughout Step2 -- e.g. the periodic HF
delta=0 gap not vanishing in the very first preflight check). Dividing a
non-vanishing numerator by a vanishing denominator necessarily diverges as
delta -> 0, regardless of the real, delta-independent part of the
correction. Fitting k = k_inf + A/delta separates that divergent artifact
(the A/delta term) from the physically meaningful, delta-independent
rescaling factor (k_inf) that should actually be applied to beta(r).

Fit performed independently for each of the three natoms-extrapolation
flavors (all-points / largest-4 / quadratic-on-largest-5) used throughout
Step 2, since they don't fully agree with each other -- reporting all three
keeps that known source of uncertainty visible rather than hiding it behind
a single number.

Output: Step3_calibration/k_inf_fit.csv (k_inf, A, and parameter standard
errors per flavor) and figures/fig2_k_delta_calibration.png (the 8 points,
all three flavors, plus the fitted curves and k_inf asymptotes).
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

SUMMARY_CSV = '../Step2/beta_rescaling_k_summary.csv'
FLAVORS = ['k_all_points', 'k_largest4', 'k_quadratic']
FLAVOR_LABELS = {'k_all_points': 'all-points', 'k_largest4': 'largest-4', 'k_quadratic': 'quadratic'}
FLAVOR_COLORS = {'k_all_points': 'tab:blue', 'k_largest4': 'tab:orange', 'k_quadratic': 'tab:green'}


def model(delta, kinf, A):
    return kinf + A / delta


def load_summary(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: float(r['delta']))
    return rows


def main():
    rows = load_summary(SUMMARY_CSV)
    delta = np.array([float(r['delta']) for r in rows])

    results = []
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for flavor in FLAVORS:
        k = np.array([float(r[flavor]) for r in rows])
        popt, pcov = curve_fit(model, delta, k, p0=[3.0, 0.3])
        perr = np.sqrt(np.diag(pcov))
        kinf, A = popt
        kinf_err, A_err = perr
        residuals = k - model(delta, *popt)

        results.append(dict(
            flavor=flavor, kinf=kinf, kinf_err=kinf_err, A=A, A_err=A_err,
            max_abs_residual=np.max(np.abs(residuals)),
        ))
        print(f"{FLAVOR_LABELS[flavor]:10s}: k_inf = {kinf:.3f} +/- {kinf_err:.3f}, "
              f"A = {A:.4f} +/- {A_err:.4f}, max|residual| = {np.max(np.abs(residuals)):.4f}")

        color = FLAVOR_COLORS[flavor]
        ax.scatter(delta, k, color=color, marker='o', s=30, label=f'{FLAVOR_LABELS[flavor]} (data)')
        d_fit = np.linspace(delta.min() * 0.9, delta.max() * 1.05, 200)
        ax.plot(d_fit, model(d_fit, kinf, A), color=color, linestyle='-', linewidth=1.5)
        ax.axhline(kinf, color=color, linestyle=':', linewidth=1, alpha=0.6)

    ax.set_xlabel('dimerization, delta')
    ax.set_ylabel('k = ab initio gap / TB-model (PBE) gap')
    ax.set_title('k(delta) = k_inf + A/delta calibration fit')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('../report/figures/fig2_k_delta_calibration.png', dpi=200)
    print("\nSaved ../report/figures/fig2_k_delta_calibration.png")

    with open('k_inf_fit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['flavor', 'kinf', 'kinf_err', 'A', 'A_err', 'max_abs_residual'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("Saved k_inf_fit.csv")

    kinf_values = [r['kinf'] for r in results]
    print(f"\nk_inf across flavors: {min(kinf_values):.2f} - {max(kinf_values):.2f} "
          f"(central estimate ~{np.mean(kinf_values):.2f})")
    print("Recommendation: weight toward the largest-4/quadratic flavors over all-points, "
          "consistent with how natoms-extrapolation reliability has been treated everywhere "
          "else in this project (smallest-natoms points are the least converged).")


if __name__ == '__main__':
    main()
