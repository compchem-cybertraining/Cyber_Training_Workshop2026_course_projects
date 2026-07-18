#!/usr/bin/env python3
"""
CASSCF timing reconnaissance on Gamma-only Ag-chain clusters, now comparing
three different starting orbital guesses per cluster size: HF canonical
orbitals, MP2 natural orbitals, and DFT (PBE) orbitals.

This is the actual bottleneck test: HF cost is cheap and roughly cubic
(see natoms_scan.csv), but CASSCF/CISD/CASPT2 scale much worse, and that's
what determines whether the "rebuild the TB model from ab initio" plan is
feasible in the time you have. The initial-guess comparison matters because
CASSCF orbital optimization can converge to different local minima (or take
very different numbers of macro iterations) depending on the starting
orbitals -- HF canonical orbitals are the default and simplest choice, but
MP2 natural orbitals (which already carry some correlation-driven orbital
character) or DFT orbitals are commonly used to give CASSCF a better-
conditioned starting point, especially once near-degeneracies show up at
longer chains.

Active space: CAS(N,N) with N=natoms -- one active orbital + one active
electron per Ag atom, matching the one-orbital-per-site TB picture. Orbitals
are selected by pySCF's default energy ordering (frontier orbitals around
the Fermi level); for Ag this should be dominated by 5s-derived states since
the 4d shell sits well below the Fermi level, but that is NOT verified here
-- this script is a timing/robustness test, not a validated active-space
calculation. Check CI vector / orbital composition before trusting any
energies from it.

IMPORTANT things verified interactively before writing this:
- mcscf.CASSCF(mf, ncas, nelecas) does NOT automatically inherit density
  fitting from a DF-based mean field (mc.with_df is unset even though
  mc._scf.with_df is a GDF object). You must call .density_fit() on the
  CASSCF object explicitly. Without it, a CAS(2,2) smoke test hadn't
  finished 1 macro iteration in 40s; with it, the same case converged in
  ~23s. This script always adds .density_fit() to every CASSCF object.
- pyscf.mp.MP2(mf) (the generic, non-pbc module) does automatically pick up
  the mean field's with_df object -- but on a real run at natoms=8 it threw
  "SVD did not converge" inside make_natural_orbitals' MP2-density pipeline.
  I checked the obvious culprit (a badly-conditioned AO overlap matrix from
  basis linear dependency) and ruled it out -- overlap eigenvalues at
  natoms=8 range from 0.069 to 1.98, condition number ~28.5, perfectly
  healthy. So the failure is somewhere inside the generic MP2/GDF pipeline
  itself, not the natural-orbital diagonalization (which is a plain
  generalized eigh against a healthy S). This version switches to the
  periodic-specific pyscf.pbc.mp.MP2 (RMP2) instead, which gave identical
  E_corr to the generic module on the small natoms=2 test case, and may
  handle the auxiliary-basis/integral construction more robustly at larger
  sizes. If it still fails, the full traceback is now captured (see below)
  instead of being swallowed to a bare str(exc), so the actual failure
  point will be visible in cas_timing.csv / the job log.
- mcscf.addons.make_natural_orbitals() works directly on the resulting
  pbc.mp.MP2 object and returns sensible occupation numbers (near 2/0 for
  occupied/virtual, fractional occupation appearing right at the frontier
  orbitals as expected from correlation).
- pbcscf.RKS(cell, xc='pbe').density_fit() builds and converges normally;
  its mo_coeff has the same shape/AO ordering as the RHF mo_coeff (same
  cell/basis), so it can be substituted directly as a CASSCF initial guess
  via mc.kernel(mo_coeff=...).
- On the first (natoms=4, natoms=8) run, the three guesses did NOT land on
  the same energy, and the ranking flipped between the two sizes (DFT-pbe
  found the lowest energy at natoms=4; at natoms=8 the still-unconverged
  HF-guess trajectory was already lower than DFT's "converged" answer).
  That's a real multiple-local-minima issue in this active space, not
  numerical noise -- max_cycle_macro=10 was too tight to tell whether any
  guess had actually reached the true minimum. Raised to 40 here so
  convergence has a real chance to complete before concluding anything
  about which guess is best.

Only even natoms/2 are included -- natoms/2 odd (e.g. 6) misses the true
zone-boundary band edge in a Gamma-only supercell, per the natoms_scan.csv
finding, so it's not a meaningful geometry to spend CASSCF time on.
natoms=12 is deliberately deferred: given the natoms=4->8 cost jump and the
unresolved convergence/multiple-minima questions at the cheaper sizes, it's
not worth committing cluster time to 12 until 4 and 8 are actually settled.

Results checkpointed to cas_timing.csv after every (natoms, guess) point.
"""

import time
import traceback
from pyscf import mcscf
from pyscf.pbc import mp as pbcmp
from pyscf.mcscf import addons
from pyscf.pbc import scf as pbcscf

from ag_chain_lib import build_cell
from convergence_scan import write_csv

BASIS = 'gth-szv-molopt-sr'
PSEUDO = 'gth-pbe'
LATTICE_LENGTH = 6.0
DELTA = 0.0864
VACUUM = 20.0          # matches natoms_scan.csv for direct comparability
MAX_CYCLE_MACRO = 40    # raised from 10 -- see note above; the multiple-minima question
                        # needs a real chance at convergence, not just a speed cap
DFT_XC = 'pbe'          # cheap, PBE-consistent with your existing TB reference


def get_initial_guesses(cell, mf_hf):
    """Return dict of {guess_name: mo_coeff} to try as CASSCF starting points."""
    guesses = {'HF': mf_hf.mo_coeff}

    try:
        mymp = pbcmp.MP2(mf_hf)   # periodic-specific module -- see note above
        mymp.verbose = 0
        mymp.run()
        _noons, natorbs = addons.make_natural_orbitals(mymp)
        guesses['MP2-NO'] = natorbs
    except Exception:
        print(f"  MP2 natural orbital guess failed:\n{traceback.format_exc()}", flush=True)

    try:
        mf_dft = pbcscf.RKS(cell).density_fit()
        mf_dft.xc = DFT_XC
        mf_dft.verbose = 0
        mf_dft.kernel()
        if mf_dft.converged:
            guesses[f'DFT-{DFT_XC}'] = mf_dft.mo_coeff
        else:
            print("  DFT guess SCF did not converge, skipping", flush=True)
    except Exception:
        print(f"  DFT guess failed:\n{traceback.format_exc()}", flush=True)

    return guesses


def run_one_cas(natoms):
    rows = []
    base = dict(natoms=natoms)
    try:
        cell = build_cell(natoms=natoms, lattice_length=LATTICE_LENGTH, delta=DELTA,
                           vacuum=VACUUM, basis=BASIS, pseudo=PSEUDO)
        base['nao'] = cell.nao
        base['nelectron'] = cell.nelectron

        t0 = time.time()
        mf = pbcscf.RHF(cell).density_fit()
        mf.verbose = 0
        mf.kernel()
        t_hf = time.time() - t0
        base['hf_wall_s'] = t_hf
        base['hf_converged'] = mf.converged
        base['hf_e_tot'] = mf.e_tot

        ncas = natoms
        nelecas = natoms
        guesses = get_initial_guesses(cell, mf)

        for guess_name, mo_guess in guesses.items():
            row = dict(base)
            row['guess'] = guess_name
            row['status'] = 'ok'
            try:
                mc = mcscf.CASSCF(mf, ncas, nelecas).density_fit()  # must be explicit -- see note above
                mc.verbose = 4      # stream macro-iteration progress to the log, even if truncated
                mc.max_cycle_macro = MAX_CYCLE_MACRO

                t1 = time.time()
                e_cas = mc.kernel(mo_guess)[0]
                t_cas = time.time() - t1

                row['ncas'] = ncas
                row['nelecas'] = nelecas
                row['ncore'] = mc.ncore
                row['cas_wall_s'] = t_cas
                row['cas_converged'] = mc.converged
                row['e_cas'] = e_cas
                row['total_wall_s'] = t_hf + t_cas
            except Exception:
                tb = traceback.format_exc()
                print(f"  CASSCF failed for guess={guess_name}, natoms={natoms}:\n{tb}", flush=True)
                row['status'] = 'FAILED: ' + tb.strip().splitlines()[-1]  # last line in CSV; full tb in log
            rows.append(row)

    except Exception:
        tb = traceback.format_exc()
        print(f"  Pre-guess stage failed for natoms={natoms}:\n{tb}", flush=True)
        base['status'] = 'FAILED (pre-guess stage): ' + tb.strip().splitlines()[-1]
        rows.append(base)

    return rows


def main():
    all_rows = []
    for natoms in [4, 8]:   # natoms=12 deferred -- see note above
        print(f"=== CASSCF({natoms},{natoms}) on natoms={natoms} cell, all initial guesses ===", flush=True)
        rows = run_one_cas(natoms)
        for row in rows:
            print(f"  -> {row}", flush=True)
        all_rows.extend(rows)
        write_csv('cas_timing.csv', all_rows)  # checkpoint after every cluster size

    print("Done. See cas_timing.csv. For each natoms, compare cas_wall_s and "
          "e_cas across guess in {HF, MP2-NO, DFT-pbe}: a faster guess with the "
          "same converged e_cas is a straightforward win; if converged e_cas "
          "disagrees across guesses beyond ~mHartree, that's a multiple-minimum "
          "or setup issue worth chasing down before trusting any of the numbers.",
          flush=True)


if __name__ == '__main__':
    main()
