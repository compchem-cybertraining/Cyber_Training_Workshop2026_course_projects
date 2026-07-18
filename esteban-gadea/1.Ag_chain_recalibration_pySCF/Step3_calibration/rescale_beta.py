#!/usr/bin/env python3
"""
Applies k_inf (from fit_k_delta.py) to the original PBE-based beta(r) to
produce the final, ab-initio-corrected TB hopping parameters.

Original model (Ag_chains_parametrization.pdf, Section 3), verified in this
project against TB_bgap.csv to <0.0002 eV agreement everywhere tested
(4 independent points, see conversation record -- the only issue was a
units slip in the report's stated A, eV/Bohr vs eV/Angstrom, corrected here):

    beta(r) = beta_eq + A * (r - r_eq)^q,   q = 1
    beta_eq = -0.668653 eV      (equilibrium hopping; fixed by matching
                                  average DFT valence/conduction bandwidth)
    A       = 0.371035 eV/Angstrom   (controls how strongly hopping splits
                                       under dimerization; this is the ONLY
                                       parameter this project's k-calibration
                                       targets)
    r_eq    = 2.604816 Angstrom

Why only A is rescaled, not beta_eq: the fundamental gap of this SSH-type
model is gap = 2|beta1 - beta2| = 2*A*|a1 - a2|, where a1, a2 are the two
alternating bond lengths -- beta_eq cancels exactly in the subtraction.
Structurally, beta_eq sets the band CENTER/dispersion width (bandwidth =
4|beta_eq| for the undimerized chain), a completely separate quantity that
this whole project's ab initio work (fundamental gap via EOM-IP/EA-CCSD,
Koopmans, etc.) never targeted or measured. Rescaling A alone is therefore
the change that's actually justified by the calibration; touching beta_eq
would need its own, separate ab initio bandwidth benchmark that hasn't been
done here.

IMPORTANT, per direct confirmation: increasing A changes the electronic
contribution to the total energy-vs-dimerization curve, which means the
repulsive potential V(r) = -beta_eq * B * (r_eq/r)^p (Section 3, fit
separately via the Section 4 Monte Carlo procedure against the DFT PES)
will need to be RE-FIT against the new A before the model's equilibrium
geometry/PES is self-consistent again. That re-fit is explicitly OUT OF
SCOPE for this script -- flagged here and in the report as required
follow-up work, not attempted.

DECISION (final, adopted): k_inf = 3.1, rounded from fit_k_delta.py's
largest-4 flavor (3.090 +/- 0.033) -- preferred over all-points (2.451) and
quadratic (3.417) for the same reason natoms-extrapolation reliability was
judged throughout Step 2: the smallest-natoms points are the least
converged, and all-points weights them equally instead of downweighting
them. The raw fit value is still loaded and printed for transparency, but
ADOPTED_K_INF (not the raw fit value) is what's actually used below.
Recompute k_inf_fit.csv first if the underlying calibration data changes,
and update ADOPTED_K_INF by hand if the decision changes.
"""

import csv
import numpy as np

BETA_EQ_EV = -0.668653
A_OLD_EV_PER_ANGSTROM = 0.371035
Q = 1
R_EQ_ANGSTROM = 2.604816

K_INF_FLAVOR = 'k_largest4'   # 'k_all_points' | 'k_largest4' | 'k_quadratic' -- for reference only
ADOPTED_K_INF = 3.1            # final decision: rounded largest-4 fit value; this is what's used


def load_kinf(path='k_inf_fit.csv', flavor=K_INF_FLAVOR):
    flavor_key = flavor.replace('k_', '')  # k_inf_fit.csv uses 'flavor' column matching FLAVORS names
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['flavor'] == flavor:
                return float(row['kinf']), float(row['kinf_err'])
    raise ValueError(f"flavor {flavor!r} not found in {path}")


def beta(r, A, beta_eq=BETA_EQ_EV, r_eq=R_EQ_ANGSTROM, q=Q):
    return beta_eq + A * (r - r_eq) ** q


def gap(lattice_length, delta, A, **kwargs):
    a1 = lattice_length * (1 + delta) / 2
    a2 = lattice_length * (1 - delta) / 2
    return 2 * abs(beta(a1, A, **kwargs) - beta(a2, A, **kwargs))


def main():
    kinf_raw, kinf_err = load_kinf()
    kinf = ADOPTED_K_INF
    A_new = kinf * A_OLD_EV_PER_ANGSTROM

    print(f"k_inf raw fit ({K_INF_FLAVOR}) = {kinf_raw:.3f} +/- {kinf_err:.3f}")
    print(f"k_inf ADOPTED (rounded, final) = {kinf}")
    print(f"A_old = {A_OLD_EV_PER_ANGSTROM} eV/Angstrom")
    print(f"A_new = {A_new:.6f} eV/Angstrom  (= k_inf * A_old)")
    print(f"beta_eq, r_eq, q UNCHANGED: beta_eq={BETA_EQ_EV} eV, "
          f"r_eq={R_EQ_ANGSTROM} Angstrom, q={Q}")
    print(f"\nFinal beta(r) = {BETA_EQ_EV} + {A_new:.4f} * (r - {R_EQ_ANGSTROM})   [eV, r in Angstrom]")

    print("\n--- Validation: old (PBE-fit) gap vs new (rescaled) gap at the 8 "
          "calibration geometries ---")
    rows = []
    calibration_points = [
        (5.8, 0.037688), (5.9, 0.062842), (6.0, 0.086419), (6.1, 0.107853),
        (6.2, 0.127447), (6.3, 0.145561), (6.4, 0.162456), (6.5, 0.178341),
    ]
    for l, d in calibration_points:
        gap_old = gap(l, d, A_OLD_EV_PER_ANGSTROM)
        gap_new = gap(l, d, A_new)
        print(f"  l={l}, delta={d}: gap_old={gap_old:.4f} eV, gap_new={gap_new:.4f} eV "
              f"(ratio {gap_new/gap_old:.3f})")
        rows.append(dict(lattice_length=l, delta=d, gap_old_eV=gap_old, gap_new_eV=gap_new))

    with open('rescaled_beta_parameters.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value', 'units'])
        w.writerow(['beta_eq', BETA_EQ_EV, 'eV'])
        w.writerow(['A_old', A_OLD_EV_PER_ANGSTROM, 'eV/Angstrom'])
        w.writerow(['A_new', A_new, 'eV/Angstrom'])
        w.writerow(['k_inf_adopted', kinf, 'dimensionless'])
        w.writerow(['k_inf_raw_fit', kinf_raw, 'dimensionless'])
        w.writerow(['k_inf_raw_fit_err', kinf_err, 'dimensionless'])
        w.writerow(['k_inf_flavor', K_INF_FLAVOR, '-'])
        w.writerow(['q', Q, '-'])
        w.writerow(['r_eq', R_EQ_ANGSTROM, 'Angstrom'])
    print("\nSaved rescaled_beta_parameters.csv")

    with open('gap_old_vs_new.csv', 'w', newline='') as f:
        fieldnames = ['lattice_length', 'delta', 'gap_old_eV', 'gap_new_eV']
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)
    print("Saved gap_old_vs_new.csv")

    print("\nNOTE: A changed -> the repulsive potential V(r) (r_eq, B, p from Section 3/4 "
          "of Ag_chains_parametrization.pdf) needs to be re-fit against this new A before the "
          "model's equilibrium PES is self-consistent again. That re-fit is done separately, "
          "not by this script -- see report/objective1_report.md Section 6 for the completed "
          "result (r_eq=3.989152 Ang, B=0.032205, p=8).")


if __name__ == '__main__':
    main()
