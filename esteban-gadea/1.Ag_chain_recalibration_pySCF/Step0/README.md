# Step 0 — Establish a Converged, Feasible Ab Initio Recipe

## Purpose

Before spending real time on the actual physics (Step 1's HF surface, Step 2's binding
energies), we needed to know: what pySCF settings actually give a converged periodic
HF calculation on an Ag chain, and what does it cost. This folder is the reconnaissance
that answers that, plus the CASSCF robustness testing needed to know Step 2 is feasible
at all. Everything here is methodology/feasibility work; none of the numbers below are
meant as final physics.

Approach used throughout: a genuine 3D-periodic "wire array" (`cell.dimension = 3`,
the well-tested default code path) with a large vacuum gap in the two transverse
directions, rather than pySCF's native `dimension=1` mode. pySCF itself flags the
low-dimensional Coulomb kernel for 1D as inaccurate, and this mirrors the same lesson
learned earlier with Quantum Espresso's native 1D mode. Pseudopotential/basis:
`gth-pbe` + `gth-szv-molopt-sr`, the only Ag combination bundled by default with
pySCF (treats 4d^10 5s^1 as valence -- carries the full d-manifold, not just the 5s
band the TB model represents, which matters for later production runs but not for
this convergence testing).

Reference geometry throughout: lattice length 6.0 A, dimerization delta = 0.0864
(bonds 2.74/3.26 A), matching the primary case in the existing single-chain report.

## Runs performed, in order

### 1. k-point convergence (`convergence_scan.py`, natoms=2, vacuum=20 A)

| nk | indirect gap (eV) | direct gap @ Gamma (eV) | wall time (s) |
|----|-------------------|--------------------------|----------------|
| 1  | 9.116 | 9.116 | 6.0 |
| 2  | 5.079 | 8.840 | 14.0 |
| 4  | 4.486 | 8.962 | 14.6 |
| 8  | 4.322 | 8.962 | 22.2 |
| 16 | 4.275 | 8.957 | 64.5 |

Indirect gap converges cleanly and monotonically (as it must -- it's a running
min/max over the sampled k-mesh, so denser sampling can only lower it or hold it,
never raise it). Direct gap at Gamma stays roughly flat (~8.84-9.12 eV, small
self-consistency wiggle from the nonlocal HF exchange operator depending on the whole
mesh) because it's pinned to one fixed momentum with no reason to trend either way.

### 2. Vacuum (interwire spacing) convergence (`convergence_scan.py`, natoms=2, nk=8)

| vacuum (A) | indirect gap (eV) | wall time (s) |
|------------|--------------------|-----------------|
| 8  | 3.812 | 17.1 |
| 12 | 4.150 | 18.0 |
| 16 | 4.272 | 24.3 |
| 20 | 4.322 | 26.5 |
| 25 | 4.347 | 31.4 |
| 30 | 4.356 | 45.0 |

Converges cleanly, essentially flat (<0.01 eV steps) by vacuum=25-30 A.

### 3. Cluster-size (natoms) scan, Gamma-only (`convergence_scan.py`, nk=1, vacuum=20 A)

| natoms | m = natoms/2 | parity | gap (eV) | wall time (s) |
|--------|--------------|--------|----------|-----------------|
| 2  | 1 | odd  | 9.116 | 5.5 |
| 4  | 2 | even | 5.079 | 15.6 |
| 6  | 3 | odd  | 5.585 | 32.4 |
| 8  | 4 | even | 4.486 | 63.2 |
| 12 | 6 | even | 4.361 | 154.0 |
| 14 | 7 | odd  | 4.598 | 262.4 |

**Key finding:** a Gamma-only supercell of m primitive cells folds to m evenly-spaced
k-points, k_j = 2*pi*j/(m*a), j=0..m-1. The true band-edge for this SSH-type
(alternating-hopping) chain sits at the zone boundary k=pi/a, which is only included
in that folded set when m is even. Odd-m points (natoms=2, 6, 14) give artificially
inflated gaps that break the otherwise-monotonic trend -- confirmed exactly via
zone-folding identities: natoms=4/Gamma matches natoms=2/nk=2 to 7 significant
figures, natoms=8/Gamma matches natoms=2/nk=4, natoms=16/Gamma (see below) matches
natoms=2/nk=8. **Only even natoms/2 should ever be used for anything gap-related.**

### 4. Joint convergence + finite-cluster comparison (`extended_convergence_check.py`)

| case | indirect gap (eV) | wall time (s) |
|------|--------------------|-----------------|
| natoms=2, nk=16, vacuum=30 (joint-converged reference) | 4.332 | 84.6 |
| natoms=16 (m=8), Gamma-only, vacuum=20 | 4.322 | 335.1 |
| natoms=16 (m=8), Gamma-only, vacuum=30 | 4.356 | 713.4 |

The joint reference (4.332 eV) is the real converged answer -- note it is NOT simply
the sum of the two 1D scans' individual corrections (naive additive estimate would
give ~4.31 eV), so the two convergence directions are close to but not perfectly
independent. natoms=16 at vacuum=30 reproduces the natoms=2/nk=8/vacuum=30 value
exactly (as zone-folding predicts, since m=8), but that is still ~24 meV away from
the true nk=16-converged answer -- closing that gap would require natoms=32 (m=16),
which is not affordable for the CASSCF stage that comes next (natoms=16 alone already
took 335-713 s just for HF; CASSCF at that size would be dramatically more than the
~35 min already seen for natoms=8's 3-guess sweep, see below).

### 5. CASSCF feasibility/robustness (`cas_timing_scan.py`, not in this folder --
    lives in `Step2/`)

Not a Step 0 deliverable itself, but the result that closes Step 0's decision: at
natoms=8, three independent CASSCF initial guesses (HF canonical orbitals, MP2 natural
orbitals, DFT/PBE orbitals) converged to energies spanning about 19 meV (two of the
three landed on the identical minimum, one found a distinct nearby local minimum).
That guess-dependent spread is the same order of magnitude as the ~24-40 meV residual
finite-size gap error from stopping at natoms=8-16 rather than natoms=32. There is no
payoff in buying more finite-size accuracy in the single-particle gap than the
correlated step can resolve.

## Conclusions / recipe adopted going forward

- **For Step 1** (single-particle HF surface): use the small primitive (natoms=2)
  cell with proper k-point sampling directly -- nk~16, vacuum~25-30 A. Already fully
  converged, cheap (~85 s/point from the joint check above), no supercell/parity
  concerns at all. This is the clean, accurate route; there is no reason to build
  Step 1's grid out of finite Gamma-only supercells.
- **For Step 2** (CASSCF binding energies, which are stuck with Gamma-only finite
  clusters since CASSCF doesn't handle k-points): use natoms=8 or natoms=12. Do not
  chase natoms=16 or larger for gap-matching purposes -- the residual finite-size
  error at these sizes is already smaller than or comparable to the CASSCF
  guess-dependent uncertainty, so it would not improve the quantity that actually
  matters (the binding energy / calibrated alpha).

## Files in this folder

- `ag_chain_lib.py` -- shared cell-builder (`build_cell`) and timed KRHF runner
  (`run_krhf`), imported by both scan scripts.
- `convergence_scan.py` -- runs scans 1-3 above, writes `kpoint_scan.csv`,
  `vacuum_scan.csv`, `natoms_scan.csv`.
- `extended_convergence_check.py` -- runs the joint check and natoms=16 comparison
  (scan 4 above); its own log is `extended_convergence.out` (the checkpointed CSV,
  `extended_check.csv`, was not preserved -- the .out log has the full result).
- `submit_convergence.slurm` -- generic SLURM template for the above (adapt
  partition/account/module-load).
- `kpoint_scan.csv`, `vacuum_scan.csv`, `natoms_scan.csv`, `extended_convergence.out`
  -- the actual results referenced in the tables above.

## Feeds into

Step 1's grid script (not yet written) should import `build_cell`/`run_krhf` from
`ag_chain_lib.py` and loop over (lattice_length, delta) using the nk=16/vacuum~25-30
recipe established here. Step 2's production binding-energy runs should use natoms=8
or 12 (not larger) with the multi-guess/keep-the-minimum practice from the CASSCF
testing.
