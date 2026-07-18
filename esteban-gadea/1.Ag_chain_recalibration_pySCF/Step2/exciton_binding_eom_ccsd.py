#!/usr/bin/env python3
"""
Fundamental gap via EOM-IP/EA-CCSD instead of Koopmans, to check whether
adding real dynamic correlation (screening) brings the exciton binding
energy down from the ~3 eV the Koopmans+CIS recipe is stabilizing near --
too large to be usable in the TB model (alpha's own instability threshold
is ~0.5 eV, comparable to the hopping).

Why EOM-CCSD and not extended Koopmans' theorem: both avoid ever building
an open-shell wavefunction (no UHF, no multiple-minima risk -- the whole
point of moving off the dSCF/UHF approach). EOM-IP/EA-CCSD is the one with
a direct, well-tested pyscf implementation (cc.CCSD(mf).ipccsd()/.eaccsd()),
built entirely on top of a single closed-shell CCSD reference. Extended
Koopmans' theorem would need a hand-rolled generalized eigenvalue solve
against a correlated density matrix -- more implementation risk for a
"check this in the meantime" script, so left for later if this doesn't
already answer the question.

What this computes, per point:
  - Koopmans gap (eps_LUMO - eps_HOMO from the same RHF used as the CCSD
    reference) -- included for free, so you can see side by side how much
    of the gap correlation actually closes.
  - EOM-CCSD fundamental gap = IP + EA, where IP = ipccsd lowest root
    (E(N-1) - E(N), positive cost to remove an electron) and EA is defined
    here in the chemist sign convention (EA = E(N) - E(N+1), positive if
    forming the anion is favorable) = -(eaccsd lowest root). Verified this
    sign convention directly at natoms=2 before writing this: eaccsd gave
    -0.0122 Ha, i.e. a favorable (positive, +0.33 eV) true EA, and the
    resulting fundamental gap (6.61 eV) came out a bit below the Koopmans
    value (7.10 eV) at the same geometry -- the right direction, if a
    modest one.
  - Optical gap: still plain CIS/TDA, NOT EOM-EE-CCSD. I checked EOM-EE-CCSD
    too, and its lowest singlet root came out surprisingly low (~1.4 eV vs
    CIS's 2.26 eV at natoms=2) -- plausible in principle (correlation
    differentially stabilizing the excited state relative to the ground
    state), but given this system's history of near-degenerate, easily-
    mislabeled states (recall the Gamma-point 5s/5p mixing issue from
    Step 1), I didn't have a way to quickly verify which physical state
    that root actually corresponds to. Safer to keep the already-validated
    CIS/TDA optical gap here and isolate the one change we actually want to
    test (does correlating the FUNDAMENTAL gap help), rather than change
    two things at once. EOM-EE-CCSD is a reasonable follow-up once the
    IP/EA side is trusted.
  - binding_energy_eV = EOM-CCSD fundamental gap - CIS optical gap.

Cost warning: CCSD scales far worse than the plain RHF/CIS version.
natoms=4 (CCSD+EOM-IP+EOM-EA) ran in ~4s here; natoms=8 did NOT finish
within a 44s budget during testing (HF alone was ~2s, so CCSD itself is
the expensive part, consistent with its steep formal scaling). So the grid
here is deliberately just [2, 4, 6, 8] -- do not casually extend to 16/24
the way the Koopmans version could; natoms=10/12 are plausible but expect
minutes, and go up from there fast. Run natoms=8 alone first if you want a
cheaper read before committing to the whole grid.
"""

import csv
import time
from pyscf import gto, scf, cc, tdscf

HARTREE_TO_EV = 27.211386245988

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
LATTICE_LENGTH = 6.0
DELTA = 0.0864
NATOMS_VALUES = [2, 4, 6, 8]   # see cost warning above before extending


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
        fundamental_gap_ccsd = IP_eV + (-EA_eV)  # = (eip + eea) * HARTREE_TO_EV
        # (kept as IP - true_EA for clarity even though algebraically identical)
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
        print(f"=== natoms={natoms}, delta={DELTA} ===", flush=True)
        row = run_point(natoms, LATTICE_LENGTH, DELTA)
        print(f"  -> {row}", flush=True)
        rows.append(row)
        write_csv('exciton_binding_eom_ccsd.csv', rows)  # checkpoint after every point

    print("\nDone. See exciton_binding_eom_ccsd.csv.", flush=True)
    print("Compare binding_energy_eV here against the Koopmans+CIS version at the same "
          "natoms (exciton_binding_convergence.csv): gap_closed_by_correlation_eV shows "
          "how much of the Koopmans gap CCSD correlation actually closes. If binding_energy_eV "
          "is still order-eV rather than dropping toward something comparable to your alpha "
          "instability threshold (~0.5 eV), that's evidence the gap is genuinely that large at "
          "this level of theory/system size, not an artifact of Koopmans specifically -- and "
          "the missing piece would be screening of the electron-hole interaction itself (the "
          "optical-gap side, still CIS-level here), not further correlation of the fundamental "
          "gap alone.", flush=True)


if __name__ == '__main__':
    main()
