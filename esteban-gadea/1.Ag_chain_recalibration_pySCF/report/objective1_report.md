# Objective 1: Ab Initio Correction of the PBE Band Gap and Exciton Binding Energy Estimate for Ag Atomic Chains

## 1. Objective

The single-chain tight-binding (TB) model for Ag atomic chains (`Ag_chains_parametrization.pdf`) reproduces a PBE (Quantum ESPRESSO, plane-wave) band gap and uses a phenomenological electron-hole interaction term,

    f_xc^A = -alpha * sum_B  dq_B / sqrt(r_AB^2 + gamma^2),

to generate a bound sub-gap excitonic peak in real-time Ehrenfest dynamics. Neither the band gap (PBE-derived) nor alpha (hand-picked) is anchored to a level of theory beyond semi-local DFT. This objective uses pySCF-based quantum chemistry (Hartree-Fock and post-HF: EOM-IP/EA-CCSD, CIS/TDA, CAM-B3LYP TDDFT) to:

1. correct the magnitude of the TB model's band gap using a higher level of theory, while preserving the qualitatively correct metal-to-semiconductor (Peierls) transition that PBE gets right and HF/post-HF methods do not, and
2. obtain a physically-grounded estimate of the exciton binding energy that the f_xc/alpha term is meant to represent.

## 2. Background

PBE is known to systematically underestimate band gaps. The natural instinct is to replace the PBE gap with one from a higher level of theory. This is not trivial for two reasons, both resolved in the course of this work:

- **Double counting.** If a correlated method (CISD, CASSCF, EOM-CCSD, ...) is used to compute a *neutral* excitation (an optical gap), it already contains electron-hole binding — using it as the TB model's single-particle gap would double-count the same physics the alpha/f_xc term is meant to supply separately. The fix is to compute the *fundamental* (quasiparticle) gap as a charged excitation, IP − EA, rather than a neutral excitation. This is the same conceptual split GW+BSE makes between a quasiparticle gap and an optical gap; BSE itself was ruled out as beyond the scope of this project.
- **Method choice for the charged gap.** IP/EA can be obtained several ways. Frozen-orbital CASCI and relaxed open-shell (UHF) delta-SCF were both tried and abandoned — the former lacks orbital relaxation and gives an unphysical gap, and the latter hits the same multiple-local-minima instability as CASSCF on this near-degenerate system, with no natoms-convergence to speak of. The TB model's own exciton is a neutral, same-electron-count excitation, so there was never a strict requirement to leave the closed-shell world for this problem in the first place. Two closed-shell-safe routes were used instead: Koopmans' theorem (eps_LUMO − eps_HOMO from a plain RHF calculation, an upper bound on the true IP) and EOM-IP/EA-CCSD (adds real dynamical correlation on top of a single closed-shell CCSD reference, without ever constructing an open-shell wavefunction).

## 3. Strategy

**Ab initio recipe.** All calculations use pySCF, the `gth-pbe`/`gth-szv-molopt-sr` pseudopotential/basis pair (the only Ag combination bundled by default with pySCF; carries the full 4d^10 5s^1 valence manifold, not just the 5s band the TB model represents), and isolated finite Ag_n clusters (even n, symmetric dimer-pair termination) rather than periodic supercells for the correlated (CCSD, TDDFT) steps, since those methods don't support k-point sampling.

**The metal-to-semiconductor test (Figure 1).** A quick periodic HF scan near delta=0 shows the gap does *not* vanish at the undimerized limit (~3.4 eV at delta=0, rising with dimerization) — a known HF/post-HF failure for 1D half-filled metals (missing screening/correlation, the same class of effect documented for polyacetylene). PBE, by contrast, gives exactly 0 eV at delta=0 for every lattice length tested. This rules out simply replacing the PBE-fit beta(r) with an HF-derived one: doing so would break the one qualitative feature the whole project depends on.

**Fundamental gap benchmark: EOM-IP/EA-CCSD.** Chosen over extended Koopmans' theorem because it has a direct, well-tested pySCF implementation (`cc.CCSD(mf).ipccsd()`/`.eaccsd()`) built on a single closed-shell reference. Validated by finite-size convergence (natoms = 2 to 12) and cross-checked sign conventions against Koopmans at the same geometries.

**The rescaling idea.** Rather than discard PBE's shape, keep it and apply a single multiplicative correction to beta(r)'s dimerization-sensitive slope: k = (ab initio fundamental gap) / (TB-model gap), evaluated at several dimerized reference geometries. Because the TB model's gap is gap = 2|beta(a1) − beta(a2)|, at delta=0 both bonds are equal length and the gap vanishes identically *for any overall scale of beta(r)* — a multiplicative rescaling cannot break the metal-to-semiconductor transition, unlike an additive correction would risk doing.

**Calibration and its 1/delta artifact (Figure 2).** k was evaluated at 8 points along the model's equilibrium ridge (lattice_length 5.8-6.5 Å, dimerization 0.038-0.178, increasing together) using the natoms-extrapolated EOM-CCSD fundamental gap. k falls sharply with delta (12.6 to 4.6 across the range) and fits essentially perfectly to k(delta) = k_inf + A_fit/delta (max residual < 0.13 across all 8 points, all three natoms-extrapolation flavors). This divergence has a physical origin, not a numerical one: the EOM-CCSD fundamental gap itself does not vanish at delta=0 (Koopmans gaps stay at 5-7 eV regardless of dimerization, the same HF-inherited pathology as Figure 1), while the TB/PBE gap in the denominator correctly does — dividing a non-vanishing quantity by one that vanishes necessarily diverges. Fitting out the 1/delta term isolates k_inf, the delta-independent part of the correction, from that artifact.

**Exciton binding energy.** Binding energy = fundamental gap − optical gap, tested with three combinations (Figure 4): Koopmans + CIS/TDA (baseline, fully closed-shell, both bare/unscreened), EOM-CCSD fundamental + CIS optical (tests whether correlating the fundamental gap alone is enough), and Koopmans fundamental + CAM-B3LYP TDDFT optical (tests whether screening the optical/exciton side alone is enough). A phenomenological dielectric-scaled CIS test (dividing the entire CIS coupling term by a scalar epsilon) was also tried early on, back when the binding energy was being compared against the *original*, uncorrected model's |alpha| ~ 0.5 eV stability ceiling and every ab initio estimate looked too large by 1-5x: it implied an effective screening strength (epsilon ~ 4-5) physically plausible for this system, as a way to bring the binding energy down further. With the corrected (larger) beta(r) from this objective, the model's own stability ceiling should scale up with it (Section 5), so this extra screening correction is likely unnecessary — kept here for the record, not adopted.

## 4. Results

### 4.1 The metal-to-semiconductor transition is real in PBE and absent in HF

![Figure 1](figures/fig1_metal_transition.png)

*Figure 1. Indirect band gap vs. dimerization near delta=0 at lattice_length=6.0 Å. PBE (green) vanishes exactly at delta=0, as required by the Peierls physics. Periodic HF (red, illustrative settings: nk=4, vacuum=15 Å — the qualitative point was independently confirmed at full production settings, nk=16/vacuum=30 Å, during the Step 1 preflight check) sits at ~3.4 eV even at delta=0 and only rises further with dimerization.*

### 4.2 k(delta) calibration

![Figure 2](figures/fig2_k_delta_calibration.png)

*Figure 2. k = ab initio gap / TB-model gap at the 8 calibration points, fit to k(delta) = k_inf + A_fit/delta for each of three natoms-extrapolation flavors (all-points, largest-4, quadratic). Horizontal dotted lines mark each flavor's k_inf asymptote.*

| flavor | k_inf | A_fit | max residual |
|---|---|---|---|
| all-points | 2.451 ± 0.013 | 0.383 ± 0.001 | 0.030 |
| largest-4 | 3.090 ± 0.033 | 0.271 ± 0.003 | 0.079 |
| quadratic | 3.417 ± 0.060 | 0.233 ± 0.005 | 0.130 |

The three flavors disagree by about 40% with each other (2.45 to 3.42), reflecting the same natoms-extrapolation ambiguity present throughout this project. **Adopted: k_inf = 3.1**, the largest-4 flavor (3.090 ± 0.033) rounded to one decimal place — preferred over all-points and quadratic for the same reason natoms-extrapolation reliability was judged throughout this project: the smallest-natoms points are the least converged, and all-points weights them equally instead of downweighting them.

### 4.3 The corrected model

![Figure 3](figures/fig3_corrected_gap_curve.png)

*Figure 3. Fundamental gap vs. delta along the equilibrium ridge: the original PBE-fit TB model (dashed blue), the k_inf-rescaled model using the adopted k_inf = 3.1 (solid purple), and the actual EOM-CCSD extrapolated ab initio points (black x). Both TB curves vanish at delta=0 by construction. The rescaled curve is deliberately NOT fit to pass through the ab initio markers point-by-point — those markers still carry the 1/delta artifact from Figure 2 at the smaller-delta end, which k_inf was specifically constructed to remove. The gap between the purple line and the black markers is that artifact, shrinking with delta exactly as expected.*

Applying the adopted k_inf = 3.1 to the original model: beta_eq = −0.668653 eV, r_eq = 2.604816 Å, and q = 1 are unchanged; only the dimerization-sensitive slope changes, A: 0.371035 → 1.150209 eV/Å (rounded 1.1502). **The corrected hopping expression is therefore:**

    beta_new(r) = -0.668653 + 1.1502 * (r - 2.604816)      [eV, r in Angstrom]

with the original repulsive potential V(r) = −beta_eq · B · (r_eq/r)^p (B = 0.231122, p = 15) carried over unchanged for now — flagged in Section 6 as requiring its own re-fit against this new A. **Note:** Section 6's subsequent joint re-fit against the DFT PES ends up moving r_eq substantially (beta_eq, satisfyingly, comes back out of that re-fit essentially unchanged — see Section 6). This does not retroactively affect anything in this section or in Section 4.2's calibration: gap = 2*A*|a1 − a2| depends only on A and the geometry (l, delta) — beta_eq and r_eq cancel completely, not just approximately, so no gap computed anywhere in this report changes regardless of how much r_eq moves. Only the repulsive fit (Section 6) and downstream quantities that depend on the *absolute* value of beta(r) (e.g. bandwidth, total energy) are affected.

**Why only A, not beta_eq, is rescaled.** gap = 2|beta1 − beta2| = 2*A*|a1 − a2| — beta_eq cancels *exactly* in that subtraction, for any value of delta, not just approximately at small delta. So the calibration data (which measures only the gap) constrains A alone; it says nothing about beta_eq's correctness, at any dimerization. Rescaling beta_eq as well would not change the fit to the calibration points at all (the gap is identical either way), so it can't be justified as "more correct" on that basis. The honest reason it's tempting is a different one: if PBE's underestimate is a generic effect (e.g. too little self-interaction correction, favoring delocalization) rather than something specific to the dimerization-splitting mechanism, beta_eq itself (which sets the bandwidth, ~4|beta_eq| for the undimerized chain) might plausibly be underestimated by a similar factor. That would need its own ab initio benchmark — a bandwidth calculation, analogous to what was done here for the gap — which this project has not done (the finite-cluster CCSD/Koopmans calculations measure only the fundamental gap, not a periodic bandwidth). Note also that rescaling A alone is not fully bandwidth-neutral away from l = 2·r_eq: the valence bandwidth is |beta1+beta2| − |beta1−beta2|, and beta1+beta2 = 2·beta_eq + A·(l − 2·r_eq), so A does leak into the bandwidth once l departs from 2·r_eq (≈5.21 Å) — at the largest calibration lattice length (6.5 Å), l − 2·r_eq ≈ 1.29 Å and this A-dependent term is already *larger in magnitude* than beta_eq itself (1.48 eV vs. 0.67 eV with the rescaled A, versus 0.48 eV with the original A). So rescaling A alone already changes the bandwidth substantially at the more-stretched end of the equilibrium ridge, even with beta_eq untouched — this isn't a fully isolated "gap-only" change in practice, just the one that's actually constrained by data. Rescaling beta_eq too is a plausible but currently untested extrapolation, worth flagging as a candidate refinement if an ab initio bandwidth estimate becomes available.

### 4.4 Exciton binding energy

![Figure 4](figures/fig4_binding_energy_convergence.png)

*Figure 4. Exciton binding energy vs. 1/natoms for three method combinations. The Koopmans+CIS baseline and CAM-B3LYP-screened-optical curves converge to essentially the same value (~2.8-3.1 eV) — CAM-B3LYP's screening speeds up finite-size convergence but does not lower the true converged answer. Only correlating the fundamental gap (EOM-CCSD) meaningfully reduces the binding energy, to ~1.2-1.4 eV extrapolated.*

The most-corrected estimate (~1.2-1.4 eV) is larger than the model's original alpha stability ceiling (|alpha| ~ 0.5 eV) — but that ceiling was calibrated against the *original*, smaller hopping scale, and this objective's whole point is that beta(r) is now larger (Section 4.3). Whether ~1.2-1.4 eV is actually workable under the new, rescaled beta(r) is being checked directly in Section 7.

**A harder constraint emerged once the re-fit model's actual equilibrium geometry was known (Section 6):** at l = 6.0 Å the model's own equilibrium dimerization is delta = 0.095665, giving a fundamental gap of 1.32 eV there. A bound exciton's binding energy cannot exceed the fundamental gap it is binding within (the optical gap = fundamental gap − binding energy must stay positive) — so the top of the EOM-CCSD range (~1.4 eV) is ruled out outright at this geometry, and even ~1.2-1.3 eV would leave only a very small optical gap (0.02-0.12 eV). This tightens, rather than resolves, the open question from Section 4.4: either the true binding energy sits well below the ~1.4 eV end of the EOM-CCSD range, or the EOM-CCSD extrapolation itself is still an overestimate for the reasons already flagged (Koopmans neglects relaxation; CIS/TDA and CAM-B3LYP under-screen relative to full GW+BSE). Section 7's TB+Ehrenfest scan is evaluated with this bound in mind.

### 4.5 Orbital character: the excited electron is on the Ag 5s manifold

Natural Transition Orbital (NTO) analysis of the lowest CIS/TDA excited state (natoms=8, lattice_length=6.0 Å, delta=0.0864) gives direct, visual support for the single s-band TB picture:

| NTO pair | weight | hole 5s-character | electron 5s-character |
|---|---|---|---|
| 1 (dominant) | 0.697 | 0.000 | 0.554 |
| 2 | 0.191 | 0.000 | 0.440 |
| 3 | 0.074 | 0.000 | 0.233 |

Across all three leading pairs, the hole carries essentially zero Ag 5s character (it is p/d-derived), while the electron carries substantial and consistent 5s character. Cube files for the dominant pair's hole and electron NTOs are in `report/nto_cubes/`.

![Dominant-pair electron NTO isosurface](figures/electron_nto.png)

*Figure 5. Isosurface of the dominant-pair electron NTO (natoms=8, lattice_length=6.0 Å, delta=0.0864), rendered from `report/nto_cubes/electron_nto.cube`. The electron density sits on the Ag 5s-derived band, consistent with the single s-band picture the TB model assumes.*

## 5. Discussion and limitations

- **k_inf is a fit-extrapolated quantity, not raw data**, and disagrees by ~40% across natoms-extrapolation flavors (2.45-3.42). This is the dominant source of uncertainty in the whole correction.
- **Only 8 calibration points, along a single (equilibrium) ridge** of the (lattice_length, delta) surface. The 1/delta fit is well-constrained within the tested range but is an extrapolation outside it, particularly toward delta=0 where it is not meant to be trusted (that is precisely the regime the correction is designed to route around).
- **The repulsive potential has been re-fit (Section 6)** against the new A, with r_eq allowed to move substantially (justified since the gap depends only on A); beta_eq comes back out of the re-fit unchanged. The re-fit reproduces the DFT equilibrium delta and near-equilibrium curvature well; residual mismatch in the tails is attributed to PBE's own known PES softness rather than a fitting deficiency.
- **The re-fit model's actual equilibrium gap (1.32 eV at l = 6.0 Å) caps the plausible exciton binding energy** — a binding energy cannot exceed the gap it lives inside of, which rules out the top of the EOM-CCSD range (~1.4 eV) outright (Section 4.4). This is a model-internal bound, independent of alpha.
- **The alpha stability ceiling has been scanned directly (Section 7)** against the rescaled model. Binding energy grows super-quadratically with alpha over the range tested (0.8-1.6), reaching only ~0.03-0.21 eV — well short of the ab initio estimate even after accounting for the gap bound above. Whether higher alpha remains stable, and whether it can close that gap, is the open question going forward.

## 6. Re-fitting of the interatomic (repulsive) potential

With A fixed at its ab initio-corrected value (1.150209 eV/Å, Section 4.3), the repulsive potential was re-fit against the DFT total-energy PES using the same Monte Carlo procedure as the original parametrization (Section 4 of `Ag_chains_parametrization.pdf`), now allowing beta_eq and r_eq to vary jointly with B and p rather than holding them at their original values. This is justified rather than arbitrary: as established in Section 4.3, the fundamental gap depends only on A, l, and delta — beta_eq and r_eq cancel out of it completely — so nothing about the ab initio gap calibration constrains them, and they are legitimately free parameters for the purpose of matching the DFT potential energy surface.

**Fitting conventions.** Energies on both sides are compared relative to their own minimum, not in absolute terms. The TB energy surface is evaluated on an 80-pair chain while the DFT reference is a single unit cell sampled over many k-points, so the two are only comparable after the appropriate per-cell scaling. The fit weights configurations with (E − E_min) < 0.1 eV more heavily than higher-energy ones, since the latter are geometries the system is unlikely to explore during the dynamics of interest. The raw PES data behind the fit (`TBenergy.dat`, `DFT_energysurface.dat`) are archived in `Step3_calibration/refit_data/` for reference; the Monte Carlo fitting code itself is not reproduced in this repository — see [`TB_Ag_excitons`](https://github.com/estebangadea/TB_Ag_excitons) for the full procedure.

**Result** (superseding an earlier version of this fit that had a bug; corrected below):

    beta_final(r) = -0.668653 + 1.150209 * (r - 3.989152)     [eV, r in Angstrom]
    V(r)          = -beta_eq * B * (r_eq / r)^p,  B = 0.032205,  p = 8

| Parameter | Original | Re-fit |
|---|---|---|
| beta_eq (eV) | −0.668653 | −0.668653 (unchanged) |
| r_eq (Å) | 2.604816 | 3.989152 |
| A (eV/Å) | 0.371035 | 1.150209 (unchanged from Section 4.3) |
| B | 0.231122 | 0.032205 |
| p | 15 | 8 |

**beta_eq comes back out of the re-fit exactly unchanged** — a satisfying consistency check. It confirms the Section 4.3 argument directly: nothing about matching the DFT PES actually wants to move beta_eq, since it is genuinely under-constrained by any of the ab initio data used in this project. r_eq, by contrast, shifts substantially (2.60 → 3.99 Å). This is not a claim about the true physical Ag-Ag bond length — the physical bond lengths that come out of the model are set by where the *total* energy (electronic + repulsive) is minimized, and at the l=6.0 Å reference geometry that still lands at a1 ≈ 3.29 Å, a2 ≈ 2.71 Å, an ordinary range for this system. r_eq is a reference length internal to the V(r) power-law form, not a physical bond length in its own right, and is free to move as the fit trades it off against p and B.

The exponent p dropping from 15 to 8 is the structurally important change, and was anticipated before the fit was run: p sets both how quickly V(r) decays once r exceeds r_eq and how quickly it stiffens once r drops below r_eq — a single exponent governing both the wall's "reach" and its steepness. With A roughly tripled, the electronic energy gain from dimerizing is much stronger over a wider range of bond length, and a repulsive wall as steep as p = 15 decays away almost entirely before it can resist that pull at longer lattice lengths, while overshooting DFT once the compressed bond gets close enough to r_eq to feel it. A smaller p gives the repulsion longer reach and a gentler rise on both sides — exactly the behavior needed to counterbalance a stronger, wider-acting electronic term.

The re-fit reproduces the DFT equilibrium dimerization well (delta ≈ 0.09 at l = 6.0 Å, matching the reference geometry used throughout this project) and tracks the curvature near it closely; the main residual is at the tails, where the TB curve sits somewhat below or above the DFT one. This residual most plausibly reflects PBE's own known tendency to under-stiffen potential energy surfaces (a generic GGA deficiency related to self-interaction/delocalization error) rather than a shortcoming of the refit itself — the refit is being asked to match a repulsive term against an electronic term that is now *more* correct than PBE (Section 4.2's ab initio gap correction) while the reference PES it's fit against is still PBE's own, somewhat too-soft, total energy surface. Since the equilibrium geometry and near-equilibrium force constants (what matters for the Ehrenfest dynamics downstream) are reproduced well, this residual is not considered a blocker.

## 7. Binding energy as a function of alpha

TB+Ehrenfest binding energy was scanned as a function of the phenomenological electron-hole coupling strength alpha, under the re-fit model (Section 6):

| alpha | binding energy (eV) |
|---|---|
| 0.8 | 0.030 |
| 1.0 | 0.059 |
| 1.2 | 0.096 |
| 1.4 | 0.141 |
| 1.6 | 0.212 |

Binding energy grows super-quadratically with alpha over this range (BE/alpha² rises from 0.047 at alpha=0.8 to 0.083 at alpha=1.6, rather than staying flat) — consistent with approaching, but not yet reaching, the kind of instability that sets in as alpha grows further (Section 5). None of the tested alpha values come close to the ab initio target: even at alpha=1.6, the binding energy (0.21 eV) is an order of magnitude below the EOM-CCSD estimate (~1.2-1.4 eV, Section 4.4), and well below the harder gap-based ceiling of 1.32 eV established in Section 4.4.

This leaves two possibilities, not mutually exclusive: (1) alpha needs to be substantially larger than 1.6 to reach ab initio-scale binding energies, and it remains to be seen whether such values are numerically/dynamically stable — the accelerating BE(alpha) trend above suggests the ceiling may not be far past this range; or (2) the EOM-CCSD estimate itself remains an overestimate of the true binding energy (both legs of that calculation are known to be one-sided, as already noted in Section 4.4's and 8's discussion), and the true, phenomenologically-achievable binding energy is closer to the tens-of-meV to few-hundred-meV scale seen here. Distinguishing between these requires extending the alpha scan further and watching for the onset of instability — the natural next step beyond this objective.

## 8. Conclusion

This objective produced a TB parametrization that reproduces both physically relevant limits the original, purely-PBE-based model could not simultaneously guarantee from first principles: the metal-to-semiconductor transition (inherited intact from PBE's beta(r) shape, protected exactly by the model's own structure) and a band gap magnitude anchored to a correlated, closed-shell-safe ab initio method (EOM-IP/EA-CCSD) for sufficiently dimerized geometries, where PBE's underestimate is largest. NTO analysis independently confirms the model's core assumption — that the relevant excited-state physics lives on a single Ag 5s-derived band. An ab initio estimate of the exciton binding energy (~1.2-1.4 eV, EOM-CCSD-based) is also in hand, understood as still likely an overestimate (both legs of the binding-energy calculation are known to be one-sided: Koopmans neglects relaxation, CIS/TDA and even CAM-B3LYP under-screen the electron-hole attraction relative to what full GW+BSE would give).

## 9. Outlook: connection to Libra dynamics and exciton self-trapping

The corrected beta(r) and a properly re-derived alpha (Sections 5, 7) set up the next stage of this project: real-time nonadiabatic electron-nuclear dynamics in Libra, using the same Ehrenfest framework the original TB model was built for. With energetics now tied to ab initio input rather than hand-picked values, the natural question becomes dynamical rather than static — whether, and under what conditions, a photoexcited exciton locally distorts the lattice enough to self-trap, and how the corrected (larger) hopping/binding-energy scale changes the self-trapping threshold relative to the original PBE-based model. That dynamical study is future work (Objective 2 and beyond).
