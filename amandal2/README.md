# Excited-state dynamics and multireference benchmarking with PySCF

Two workstreams that share a PySCF backend but answer different questions.

**A. `libra_pyscf`** — on-the-fly nonadiabatic molecular dynamics, coupling PySCF
excited-state structure to the Libra dynamics engine. *When a molecule is excited, how
fast and to where does population flow between electronic states?*

**B. Ethylene three-method PES** — a static benchmark of TDDFT vs SA-CASSCF vs NEVPT2
along the ethylene torsion. *Where, and why, do single-reference methods fail for
excited states?*

They are kept separate on purpose: A needs Libra, B needs prism, and nothing needs both
in one interpreter.

## Background

Most chemistry happens on one Born–Oppenheimer surface. Photochemistry breaks this: an
excited molecule sits where several surfaces lie close, and near their crossings
(conical intersections) population transfers between electronic states — the event
behind vision, UV photoprotection, and pyrazine's ~20 fs S₂→S₁ decay. Two hard problems
follow: **propagating** the coupled quantum-electron / classical-nucleus dynamics
(workstream A), and **getting the electronic structure right** where the ground state
itself becomes multiconfigurational and single-reference methods fail (workstream B).

## A. `libra_pyscf`

A PySCF backend that supplies Libra, at each timestep, the adiabatic energies, their
gradients, and the many-body time-overlap `⟨Ψᵢ(t)|Ψⱼ(t+dt)⟩`. The time-overlap is
load-bearing: it is the only route by which population moves between states, so if it is
wrong the results are wrong silently. It is cheap in a Gaussian basis via
`gto.intor_cross` (AO overlaps across two geometries) — a one-liner in PySCF, a real
project in a plane-wave code.

Because mainline PySCF has no analytic NAC vectors for TDDFT, the dynamics uses **local
diabatization**, which propagates directly from time-overlaps and never forms the sharp
coupling spike at a crossing. This makes `ehrenfest_adi_ld` the natural recipe and forces
FSSH to rescale along force differences rather than NAC vectors (the "forces and energies
only" regime of Akimov 2024).

**Files:** `pyscf_libra.py` (backend + Wigner sampling), `run_pyrazine.py` (two-stage
prep/run driver), `submit_prep.sh` / `submit_pyrazine.sh` (SLURM, one trajectory per
array task), `plot_pyrazine.py` (ensemble average + energy-conservation check).

**Caveats worth knowing.** A cluster of hard-won failures live in the code comments —
Libra's C++ needs `CMATRIXList` (not Python lists) and several `dyn_general` overrides
that the stock 2-state recipes get wrong. The recurring physics pitfall is the
**band edge**: the top state(s) in the window lose overlap norm to partners outside it,
and local diabatization then fabricates amplitude to compensate. The fix is a buffer —
choose `nexc` so the states of interest sit a few below the edge. Always read the
energy-conservation panel first; a clean-looking population curve on drifting total
energy means nothing.

## B. Ethylene: TDDFT vs SA-CASSCF vs QD-NEVPT2

`ethylene_three_methods.ipynb` scans the HCCH torsion 0°→90°. At 90° the π bond breaks
into a diradical — a two-configuration ground state that a restricted reference cannot
represent, which is exactly why ethylene illustrates single-reference failure so cleanly
(and why it is unusable for workstream A). The active space is trivially π/π\* at every
angle.

CAS(2,2) has three singlet roots, and they show two TDDFT failures at once: S₀ goes
qualitatively wrong at 90°, and the doubly-excited Z state (π\*²) is absent entirely,
since linear response produces only single excitations. The quantitative argument is the
ground-state natural occupations, which move from ~1.91/0.09 to **1.00/1.00** across the
scan — the numerical signature of the wavefunction becoming multireference, lined up with
where the restricted methods depart. QD-NEVPT2 is included because S₁/S₂ approach at 90°,
where SS-NEVPT2 (which treats states independently) breaks down; prism's oscillator
strengths also let states be identified by character (V bright, Z dark) rather than index.

The notebook bakes in the prism gotchas (`rdm_order=2`, `keep_amplitudes=True`, freezing
cores, saving SS and QD separately). 6-31G omits diffuse functions deliberately, so the V
state's absolute energy is too high — the shapes and cross-method comparison are the
point, not the number.

## Environments

Libra and prism are separate projects, and the clean setup is **two environments**.
Libra links numpy through a compiled Boost.Python extension, so PySCF must never
re-resolve that numpy (an ABI mismatch segfaults rather than erroring) — when PySCF lives
in a user directory it is *appended* to `sys.path`, never inserted. prism is pure Python
(git clone + `$PYTHONPATH`; note `pip install prism` fetches an unrelated package) and is
shipped under the workshop's `SOFTWARE_2026/`. `prism_kernel_launcher.sh` builds a Jupyter
kernel for it, mirroring the cluster's "hard clean" so environments don't leak.

## File map

```
pyscf_libra.py                 backend: PySCFSource, compute_model, Wigner sampling
run_pyrazine.py                two-stage NA-MD driver (prep / run)
submit_prep.sh                 SLURM: one-time prep
submit_pyrazine.sh             SLURM array: one trajectory per task
plot_pyrazine.py               ensemble analysis + energy referee
ethylene_three_methods.ipynb   TDDFT / CASSCF / SS- & QD-NEVPT2 torsion scan
prism_kernel_launcher.sh       Jupyter kernel for the pyscf+prism environment
```
