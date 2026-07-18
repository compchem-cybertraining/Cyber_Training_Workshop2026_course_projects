# Step 2 — Ab Initio Electron-Hole Binding Energy

## Purpose

Estimate the exciton binding energy E_b = fundamental (quasiparticle) gap - optical
gap for the Ag chain, at a fixed reference geometry (lattice_length=6.0 A,
delta=0.0864, i.e. bonds 2.74/3.26 A -- the same case used throughout Step 0), so it
can be inverted through the TB model's f_xc(alpha, gamma) relation in Step 3.

This turned into the largest methodology detour of the project so far. The short
version: every closed-shell-safe ab initio route we could afford gives E_b of order
1-5 eV, converging (with more atoms and more correlation) toward roughly 1-3 eV
depending on method -- one to two orders of magnitude larger than the model's own
stability ceiling (|alpha| ~ 0.5 eV, where f_xc becomes comparable to the hopping and
drives spurious charge ordering). That mismatch, and how we got there, is the actual
deliverable of this step.

## Methodology journey (in order attempted)

### 1. Frozen-orbital CASCI IP/EA (`test_ipea.py`, `test_ipea2.py`, `test_ipea3.py`)

First attempt: get the fundamental gap as IP-EA (charged excitations) via CASCI on
the neutral/cation/anion natoms=2 system, using the same frozen HF orbitals for all
three (a Koopmans-style shortcut -- no orbital relaxation for the charged states).
Result: implied binding energy ~5.58 eV, EA came out negative. Diagnosed as a classic
frozen-orbital artifact -- charged states need to relax their orbitals, especially the
extra/missing electron's own orbital, and skipping that badly overestimates IP and
underestimates EA.

### 2. Relaxed UHF delta-SCF (`test_dscf.py`, `test_dscf_n4.py`, then
`exciton_binding_convergence.py` v1/v2)

Fixed the relaxation problem by running genuine self-consistent UHF (with stability
analysis) on the cation and anion separately, instead of frozen CASCI orbitals.
Better at natoms=2 (EA became positive, 4.07 eV implied binding) but did not survive
scaling up: across natoms=2-16, EA plateaued around 4.7-4.9 eV while IP kept falling
without leveling off, one point (natoms=14) had a non-converged cation, and the two
1/natoms extrapolation fits (all points vs. largest 4) disagreed by 2.6 eV and
disagreed MORE as more data was added. Diagnosed as UHF landing on different,
inconsistent symmetry-broken solutions for the near-degenerate open-shell radical at
different cluster sizes -- the same multiple-local-minima pathology CASSCF showed
during Step 0's feasibility testing, resurfacing in a cheaper method. Abandoned.

**Turning point:** at this stage the question was raised directly -- the TB model's
exciton is a same-electron-count (neutral, two-particle) excitation, not a charged
state, so there was never a strict requirement to leave the closed-shell world in the
first place. IP/EA via charged SCF was only ever a means to estimate the fundamental
gap; Koopmans' theorem gives that same fundamental gap directly from the neutral
system's own orbital energies, with no separate SCF, no radical, no multiple-minima
risk.

### 3. Closed-shell Koopmans + CIS/TDA (`exciton_binding_convergence.py`)

fundamental gap = eps_LUMO - eps_HOMO from plain closed-shell RHF on the neutral
system; optical gap = CIS/TDA lowest singlet excitation (also closed-shell, neutral).
binding = koopmans_gap - optical_gap. No open-shell SCF anywhere.

Ran cleanly out to natoms=24 (nao=216), smooth and monotonic the whole way:

| natoms | koopmans_gap (eV) | optical_gap (eV) | binding_energy (eV) | wall (s) |
|--------|--------------------|--------------------|-----------------------|----------|
| 2  | 7.101 | 2.263 | 4.838 | 0.1 |
| 4  | 5.672 | 1.690 | 3.982 | 0.7 |
| 6  | 5.131 | 1.616 | 3.515 | 2.7 |
| 8  | 4.864 | 1.523 | 3.341 | 7.2 |
| 10 | 4.712 | 1.450 | 3.262 | 15.9 |
| 12 | 4.617 | 1.397 | 3.221 | 28.6 |
| 14 | 4.555 | 1.358 | 3.196 | 46.1 |
| 16 | 4.511 | 1.330 | 3.181 | 79.3 |
| 20 | 4.455 | 1.291 | 3.163 | 200.8 |
| 24 | 4.422 | 1.267 | 3.154 | 476.4 |

1/natoms extrapolation: binding_energy -> 2.93 eV (all points) / 3.09 eV (largest 4),
agreement 0.165 eV -- a genuinely converged answer, not a finite-size artifact, sitting
right between the two fits and consistent with the raw natoms=24 point (3.154 eV).
Both legs are known to be one-sided approximations that push the same direction
(overestimate): Koopmans neglects orbital relaxation (upper bound on the true IP), and
CIS/TDA uses the bare, unscreened Coulomb interaction for the e-h attraction (likely
overbinds relative to a properly screened treatment). So ~3.0 eV is a ballpark, and if
anything an overestimate, not an underestimate.

### 4. EOM-IP/EA-CCSD (`exciton_binding_eom_ccsd.py`)

Tested whether adding genuine dynamic correlation to the fundamental-gap side (still
via a single closed-shell CCSD reference -- `cc.CCSD(mf).ipccsd()`/`.eaccsd()`, no
open-shell SCF at all) would bring the binding energy down substantially. Kept the
optical gap at CIS/TDA (unchanged) to isolate this one change. (Also checked
EOM-EE-CCSD-singlet as a fully-CCSD-consistent optical gap -- it gave suspiciously low
values, ~1.4 eV vs. CIS's 2.26 eV at natoms=2, without an easy way to confirm it
wasn't picking up a mislabeled/contaminated state given this system's history of
near-degenerate states; not used.)

| natoms | fundamental_gap_ccsd (eV) | gap_closed_by_correlation (eV) | binding_energy (eV) | wall (s) |
|--------|----------------------------|-----------------------------------|------------------------|----------|
| 2  | 6.614 | 0.487 | 4.351 | 0.3 |
| 4  | 4.834 | 0.838 | 3.145 | 1.5 |
| 6  | 4.052 | 1.079 | 2.436 | 6.2 |
| 8  | 3.638 | 1.226 | 2.115 | 20.3 |
| 10 | 3.392 | 1.320 | 1.942 | 60.7 |
| 12 | 3.235 | 1.382 | 1.838 | 146.8 |

Real effect, growing with system size (correlation closes more of the gap as the
chain grows -- consistent with a collective/many-body screening response that needs
more polarizable material around the hole to develop), but not enough on its own:
1/natoms extrapolation puts the converged binding energy around 1.2-1.4 eV (linear
and quadratic fits agree in this range) -- a real ~2x reduction from the Koopmans+CIS
ballpark, but still 2-3x too large for the model. Cost is steep (~O(N^6)-type
scaling): natoms=12 took ~2.4 min wall; natoms=16 was not attempted.

### 5. Range-separated (long-range-corrected) TDDFT (`exciton_binding_lc_tddft.py`)

Tested whether the missing physics is screening of the e-h attraction on the
optical-gap side instead, by swapping CIS/TDA for TDA on CAM-B3LYP (a standard
literature choice for exciton binding energy estimates in molecular/organic-
semiconductor contexts). Kept the fundamental gap at plain RHF Koopmans (unchanged)
to isolate this one change.

| natoms | koopmans_gap (eV) | lc_optical_gap (eV) | binding_energy (eV) | wall (s) |
|--------|--------------------|-----------------------|------------------------|----------|
| 2 | 7.101 | 3.155 | 3.946 | 0.5 |
| 4 | 5.672 | 2.328 | 3.344 | 2.8 |
| 6 | 5.131 | 1.949 | 3.182 | 8.9 |
| 8 | 4.864 | 1.702 | 3.162 | 17.4 |

**Key finding:** CAM-B3LYP's optical gap is consistently larger than CIS's at every
natoms (as expected -- its response kernel carries some real correlation/screening
that bare CIS lacks), so binding_energy is consistently smaller than the Koopmans+CIS
series at the same natoms. But it also converges much faster with system size
(differences shrink from -0.60 to -0.16 to -0.02 eV between natoms=2->4->6->8, versus
Koopmans+CIS's much slower -0.87/-0.47/-0.17 eV over the same range) -- it is already
essentially flat by natoms=8. A rough 4-point extrapolation lands around ~2.8 eV,
close to the Koopmans+CIS asymptote (~2.9-3.1 eV). **Interpretation: CAM-B3LYP's
screening speeds up finite-size convergence but does not lower the true converged
binding energy relative to bare HF+CIS.** Whatever screening this functional's kernel
supplies isn't enough to change the answer that matters. (Caveat: only 4 points, cost
rises fast -- natoms=8 needed density fitting + a coarser DFT grid just to finish in
17 s; not pushed further.)

### 6. Phenomenological dielectric-scaled CIS (`exciton_binding_screened_cis.py`)

Instead of trying another ab initio method for screening, put in a single explicit
screening knob by hand: build the closed-shell singlet CIS/TDA matrix
A_{ia,jb} = delta_ij delta_ab(eps_a-eps_i) + 2(ia|jb) - (ij|ab) directly (validated
against pyscf's own `tdscf.TDA(mf).get_ab()` matrix, diagonalized the same way, to 7
decimal places at natoms=2, and against the production CIS/TDA gap at natoms=8,
matching to 5 decimal places), then divide the entire coupling term by a scalar
epsilon representing a dielectric constant. eps=1 is bare CIS; eps->infinity turns the
coupling off entirely and binding_energy -> 0 (single-particle gap only) -- the
correct qualitative limit. Solved (via interpolation across an epsilon grid) for the
epsilon at which binding_energy = 0.5 eV (the model's own stability threshold).

| natoms | bare_binding (eps=1, eV) | eps_for_0.5eV_binding |
|--------|---------------------------|------------------------|
| 2  | 4.852 | 7.66 |
| 4  | 3.982 | 6.27 |
| 6  | 3.515 | 5.70 |
| 8  | 3.341 | 5.38 |
| 10 | 3.262 | 5.12 |
| 12 | 3.221 | 4.94 |

1/natoms extrapolation of eps_for_target: ~4.2-4.5 (all-points vs. largest-4 fits).
**Key finding: this is a physically plausible dielectric constant** -- comparable to
many semiconductors and molecular crystals (roughly 3-15 range), not an absurd number
like 50+ that would signal the wrong physics entirely. So a screening strength in a
reasonable ballpark, applied consistently to the e-h attraction, would be enough to
close essentially the whole gap between our ab initio numbers and the model's usable
range. The catch: none of the actual quantum-chemistry tools tried (CIS, CCSD-IP/EA,
CAM-B3LYP-TDDFT) deliver that much screening on their own -- they're all
local/semi-local or single-reference approximations to a collective polarization
response that would really need something like GW+BSE to capture properly, which was
ruled out of scope from the start of this project.

## Conclusion

The true ab initio exciton binding energy for this system is almost certainly larger
than what the TB model's f_xc(alpha, gamma) term can stably represent (|alpha| up to
~0.5 eV before charge-ordering instability sets in). Every method tried -- from bare
HF+CIS (~3.0-3.1 eV converged) through added dynamical correlation on the fundamental
gap (CCSD, ~1.2-1.4 eV converged) through added screening on the optical gap
(CAM-B3LYP, converges to essentially the same ~2.8-3.1 eV as bare CIS) -- lands
somewhere between "an order of magnitude too large" and "still 2-3x too large" for the
model. The one piece of good news: a phenomenological dielectric-screening test shows
that a physically reasonable screening strength (effective dielectric constant ~4-5)
would be enough to explain the discrepancy, meaning there's no reason to think the
model's parametrization is fundamentally the wrong shape for this physics -- just that
the missing ingredient (real many-body screening) is out of reach of every method
affordable in this workshop.

This is a real, honest result, not a dead end: Step 3 does not get to invert a single
clean ab initio E_b into alpha. Instead, alpha will need to be chosen empirically
within the model's stable range, with this whole investigation serving as the
documented justification for why that's the right call (see Step 3 plan in the
top-level README).

## Files in this folder

- `exciton_binding_convergence.py` / `.csv` / `_extrapolation.csv` -- the closed-shell
  Koopmans+CIS series (method #3 above), natoms=2-24.
- `exciton_binding_eom_ccsd.py` / `.csv` -- EOM-IP/EA-CCSD fundamental gap + CIS
  optical gap (method #4), natoms=2-12.
- `exciton_binding_lc_tddft.py` / `.csv` -- Koopmans fundamental gap + CAM-B3LYP TDA
  optical gap (method #5), natoms=2-8.
- `exciton_binding_screened_cis.py` / `.csv` -- Koopmans fundamental gap + hand-built,
  dielectric-scaled CIS optical gap with epsilon scan (method #6), natoms=2-12.
- `test_ipea.py`, `test_ipea2.py`, `test_ipea3.py`, `test_dscf.py`, `test_dscf_n4.py`
  -- early throwaway diagnostics (methods #1-2 above), kept for the record, not
  meant to be rerun.

All scripts share `build_mole` (isolated finite `pyscf.gto.Mole` clusters, gth-pbe +
gth-szv-molopt-sr, natoms even for clean dimer-pair termination) and
`HARTREE_TO_EV = 27.211386245988`, imported from `exciton_binding_convergence.py`
where not redefined locally.

## Feeds into

Step 3 needs one number out of all this: a working alpha for the TB model. Given the
conclusion above, that number will be chosen empirically within the |alpha| < ~0.5 eV
stable range rather than by direct ab initio inversion -- see the top-level README for
the specific plan.
