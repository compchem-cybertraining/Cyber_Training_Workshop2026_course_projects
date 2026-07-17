#!/usr/bin/env python3
"""
Second calibration point for the beta(r) rescaling factor k.

CORRECTED geometry/PBE value (superseding an earlier, mistaken version of
this script): lattice_length=6.42857, delta=0.167, PBE gap=0.797 eV. The
earlier version used delta=0.1367 (bgap quoted as 0.6338 eV) -- that was a
transcription error on the (l, delta) pair and does not correspond to any
real point on the PBE grid; whatever EOM-CCSD numbers came out of that run
should be discarded, they were answering a question nobody was asking.
lattice_length itself was already right (6.429 rounds from 6.42857), only
delta and the PBE gap value were wrong.

Context: rather than redo Step 1's whole HF surface (which has its own,
worse pathology -- HF doesn't even get the delta=0 metal-to-semiconductor
transition right), the plan is to keep the EXISTING PBE-based beta(r) shape
and rescale it by a single multiplicative constant k = (high-quality ab
initio gap) / (PBE gap), calibrated at well-converged reference geometries.

At point 1 (lattice_length=6.0, delta=0.0864), that gave:
  PBE gap: 0.4057 eV
  EOM-CCSD extrapolated fundamental gap: ~2.4-2.6 eV (intercepts 2.641 / 2.412
  / 2.384 eV for all-points / largest-4 / quadratic 1/natoms fits)
  => k1 ~ 5.9-6.5 (central ~6.1-6.2)

That's a much bigger correction than the "typical" PBE gap-underestimate
(usually 30-50%, not 500-600%) -- plausible for a marginally-dimerized
quasi-1D system (GGA functionals are known to sometimes badly under-open
Peierls/SSH-type gaps for exactly this reason), but big enough to want a
second, independent geometry as a check before trusting one global k for
the whole beta(r) curve. This script is that second point, now with the
corrected geometry/target.

This script repeats the EXACT same EOM-IP/EA-CCSD fundamental-gap
convergence procedure as `exciton_binding_eom_ccsd.py`. Once run, fit
fundamental_gap_ccsd_eV vs 1/natoms the same way as point 1 (all-points,
largest-4, quadratic-on-largest-5), then compute
k2 = (extrapolated fundamental_gap_ccsd_eV) / 0.797, and compare against k1:
  - If they agree reasonably well (say within ~20-30% of each other), a
    single global rescaling of beta(r) is defensible.
  - If they disagree substantially, the PBE-vs-ab-initio discrepancy is
    geometry-dependent and a single constant k isn't the right correction --
    would need at minimum a delta-dependent (or l-dependent) correction
    function instead of one number.

Everything else (methods, integrals, sign conventions, cost profile) is
identical to `exciton_binding_eom_ccsd.py` -- see that script's docstring
for the full methodology writeup (why EOM-CCSD over extended Koopmans', the
IP/EA sign convention check, the CCSD cost warning). Only LATTICE_LENGTH,
DELTA, PBE_GAP_EV, and this docstring differ.

Grid kept the same as point 1: natoms=2..12. Cost should be similar (bond
lengths differ modestly: point 1's short/long bonds are 2.741/3.259 A at
delta=0.0864; this point's are 2.678/3.750 A at delta=0.167,
lattice_length=6.42857 -- a somewhat more dimerized geometry, but the same
order of magnitude, so point 1's timings, natoms=12 in ~2.4 min wall,
should still be a reasonable guide).
"""

import csv
import time
from pyscf import gto, scf, cc, tdscf

HARTREE_TO_EV = 27.211386245988

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
LATTICE_LENGTH = 6.42857
DELTA = 0.167
PBE_GAP_EV = 0.797   # corrected reference value for computing k2
NATOMS_VALUES = [2, 4, 6, 8, 10, 12]


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
    rows = []
    for natoms in NATOMS_VALUES:
        print(f"=== point 2 (corrected): natoms={natoms}, lattice_length={LATTICE_LENGTH}, "
              f"delta={DELTA} ===", flush=True)
        row = run_point(natoms, LATTICE_LENGTH, DELTA)
        print(f"  -> {row}", flush=True)
        rows.append(row)
        write_csv('exciton_binding_eom_ccsd_point2.csv', rows)  # checkpoint after every point

    print("\nDone. See exciton_binding_eom_ccsd_point2.csv.", flush=True)
    print(f"Next: fit fundamental_gap_ccsd_eV vs 1/natoms (same way as point 1) to get the "
          f"extrapolated fundamental gap, then k2 = extrapolated_gap / {PBE_GAP_EV} -- "
          f"compare against k1 ~5.9-6.5 from point 1 to see whether a single global "
          f"beta(r) rescaling factor is defensible.", flush=True)


if __name__ == '__main__':
    main()
