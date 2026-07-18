#!/usr/bin/env python3
"""
Consolidated version of the one-off exciton_binding_eom_ccsd_point*.py scripts:
given a list of (lattice_length, delta, pbe_gap) calibration points, computes
the EOM-IP/EA-CCSD fundamental-gap natoms-convergence series for each one, so
`beta_rescaling_summary.py` can extrapolate and compute k = ab_initio_gap /
pbe_gap per point.

Why this scan exists: the plan is to keep the existing PBE-based beta(r)
shape and rescale it by k, calibrated against high-quality ab initio gaps,
rather than redo Step 1's HF surface (which has its own, worse pathology --
HF doesn't get the delta=0 metal-to-semiconductor transition right at all).
Two points computed so far (see beta_rescaling_point1.csv/point2.csv,
carried over from the earlier one-off scripts):

  point1 (l=6.0,     delta=0.0864): pbe_gap=0.4057 eV, k ~ 5.9-6.5  (central ~6.1)
  point2 (l=6.42857, delta=0.167 ): pbe_gap=0.797  eV, k ~ 4.7-4.7  (central ~4.7)

That ~25-30% drop from point1 to point2 is bigger than either point's own
extrapolation uncertainty (~10% and ~2%) -- a real trend, not noise: k drops
as delta increases. The open question (per the most recent conversation) is
NOT what happens at small delta -- that's exactly the near-metallic regime
where every method in this project has broken down (HF's delta=0 gap not
vanishing, CASSCF/UHF multiple minima, etc.), so k there is expected to be
unreliable and isn't the target of this scan. The actual question is whether
k CONVERGES to a stable value as delta increases (i.e. as the system becomes
more clearly dimerized/molecular and single-reference methods, including the
CCSD reference here, become more trustworthy) -- if so, that converged k is
the one to use for the rescaling, since it's evaluated in the regime where
this whole approach is most defensible.

New (label, lattice_length, delta, pbe_gap_eV) entries can be added to
CALIBRATION_POINTS below as new PBE grid values come in (prefer delta values
larger than 0.167, per the above). Already-computed points are skipped
automatically (checked via existing beta_rescaling_<label>.csv), so it's safe
to keep adding points to the same list over time.

Cost: same profile as the earlier point1/point2 scripts -- natoms=12 took
~2.4-3.3 min wall per point in earlier runs. Budget accordingly if adding
several new points at once (SLURM time budget ~ n_points * 10-15 min for the
full natoms=2..12 grid per point).
"""

import os
import csv
import time
from pyscf import gto, scf, cc, tdscf

HARTREE_TO_EV = 27.211386245988

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
NATOMS_VALUES = [2, 4, 6, 8, 10, 12]

# Add new points here as more PBE grid values come in. Larger delta (>0.167)
# preferred -- see docstring for why small delta isn't the useful direction.
CALIBRATION_POINTS = [
    dict(label='point1', lattice_length=6.0,     delta=0.0864, pbe_gap_eV=0.4057),
    dict(label='point2', lattice_length=6.42857, delta=0.167,  pbe_gap_eV=0.797),
]


def build_mole(natoms, lattice_length, delta, verbose=0):
    assert natoms % 2 == 0, "natoms must be even (clean dimer pairs, no dangling end atom)"
    half = lattice_length / 2.0
    d_short = half * (1 - delta)
    d_long = half * (1 + delta)
    positions = [0.0]
    for i in range(natoms - 1):
        bond = d_short if i % 2 == 0 else d_long
        positions.append(positions[-1] + bond)
    atom_str = "; ".join(f"Ag {p:.6f} 0 0" for p in positions)
    return gto.M(atom=atom_str, basis=BASIS, pseudo=PSEUDO, verbose=verbose)


def run_point(natoms, lattice_length, delta):
    row = dict(natoms=natoms, lattice_length=lattice_length, delta=delta, status='ok')
    try:
        t0 = time.time()

        mol = build_mole(natoms, lattice_length, delta)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.max_cycle = 150
        mf.kernel()
        t_hf = time.time() - t0

        nocc = mol.nelectron // 2
        koopmans_gap = (mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_TO_EV

        td = tdscf.TDA(mf)
        td.nstates = 3
        td.kernel()
        optical_gap = td.e[0] * HARTREE_TO_EV

        t1 = time.time()
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        t_ccsd = time.time() - t1

        t2 = time.time()
        eip, _ = mycc.ipccsd(nroots=1)
        eea, _ = mycc.eaccsd(nroots=1)
        t_eom = time.time() - t2

        IP_eV = eip * HARTREE_TO_EV
        EA_eV = -eea * HARTREE_TO_EV   # chemist convention: positive = favorable
        fundamental_gap_ccsd = IP_eV - EA_eV
        binding_energy_eV = fundamental_gap_ccsd - optical_gap

        row.update(
            hf_converged=mf.converged,
            ccsd_converged=mycc.converged,
            e_hf=mf.e_tot,
            e_ccsd=mf.e_tot + mycc.e_corr,
            koopmans_gap_eV=koopmans_gap,
            optical_gap_eV=optical_gap,
            IP_ccsd_eV=IP_eV,
            EA_ccsd_eV=EA_eV,
            fundamental_gap_ccsd_eV=fundamental_gap_ccsd,
            binding_energy_eV=binding_energy_eV,
            gap_closed_by_correlation_eV=koopmans_gap - fundamental_gap_ccsd,
            nao=mol.nao,
            wall_hf_s=t_hf,
            wall_ccsd_s=t_ccsd,
            wall_eom_s=t_eom,
            wall_total_s=time.time() - t0,
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
    for point in CALIBRATION_POINTS:
        label = point['label']
        out_path = f'beta_rescaling_{label}.csv'
        if os.path.exists(out_path):
            print(f"=== {label}: {out_path} already exists, skipping ===", flush=True)
            continue

        print(f"=== {label}: lattice_length={point['lattice_length']}, "
              f"delta={point['delta']}, pbe_gap={point['pbe_gap_eV']} eV ===", flush=True)
        rows = []
        for natoms in NATOMS_VALUES:
            row = run_point(natoms, point['lattice_length'], point['delta'])
            row['label'] = label
            row['pbe_gap_eV'] = point['pbe_gap_eV']
            print(f"  natoms={natoms} -> fundamental_gap_ccsd_eV="
                  f"{row.get('fundamental_gap_ccsd_eV')}, status={row.get('status')}", flush=True)
            rows.append(row)
            write_csv(out_path, rows)  # checkpoint after every point

    print("\nDone with all calibration points not already computed. "
          "Run beta_rescaling_summary.py next to extrapolate and compute k per point.",
          flush=True)


if __name__ == '__main__':
    main()
