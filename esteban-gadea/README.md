# Exciton Self-Trapping in Silver Atomic Chains

**CyberTraining 2026 Workshop on Computational Chemistry — Course Project**
**Author:** Esteban D. Gadea

## Overview

This project takes a tight-binding model for one-dimensional Ag atomic chains, corrects its band gap and exciton binding energy against ab initio quantum chemistry (pySCF), and ports its excited-state dynamics into Libra to observe and directly visualize exciton self-trapping in real time. The original tight-binding model lives in [TB_Ag_excitons](https://github.com/estebangadea/TB_Ag_excitons).

**Start here:** [`report.md`](report.md) is the full write-up.

## Repository layout

- [`report.md`](report.md) — the report (start here).
- [`1.Ag_chain_recalibration_pySCF/`](1.Ag_chain_recalibration_pySCF/) — ab initio re-parametrization of the model's band gap and exciton binding energy (Part I). Has its own [README](1.Ag_chain_recalibration_pySCF/README.md) for reproducing the underlying calculations.
- [`2.Exciton_trapping_Libra/`](2.Exciton_trapping_Libra/) — the Libra port and the three production dynamics runs that show self-trapping (Part II). Has its own [README](2.Exciton_trapping_Libra/README.md).
