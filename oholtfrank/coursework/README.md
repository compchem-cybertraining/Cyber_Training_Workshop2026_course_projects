Workshop coursework — Owen Holtfrank

Documented work from the five CyberTraining 2026 sessions. Each folder holds the inputs, scripts, and outputs I generated. 

Several fed directly into the course project (../project/).

01_chemml — ChemML notebooks (Hachmann): AutoML regression, genetic-algorithm feature selection, Pyscript.

02_pyscf_prism — Excited-state benchmarking of s-tetrazine against the QUEST set (Sokolov): CIS/TDHF, TDDFT, ADC(2)/ADC(3), EOM-CCSD, SA-CASSCF, NEVPT2. Motivated the project's choice of SA-CASSCF/XMS-CASPT2 over TDDFT.

03_openmolcas — CASSCF, RASSI, and nonadiabatic-coupling tutorials (Mehmood). Basis for the project's scans. Found a possible bug in the Hessian converter (reads the BFGS optimizer Hessian).

04_libra — Non-equilibrium Fermi golden rule (Akimov). Applied to the project as a documented negative: a harmonic bath cannot represent a dissociative continuum.

05_tenso — HEOM open-system dynamics (Franco): a three-site superexchange model swept over bath temperature (10 K–10⁹ K).

The PySpawn AIMS session (Levine, Day 3 PM) is documented in the project itself, where the ethylene tutorial became the start.py template and required patching get_overlap to run beyond 10 states (../project/patches/).
