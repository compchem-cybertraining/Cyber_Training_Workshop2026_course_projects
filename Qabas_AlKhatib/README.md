# ML Approximation of Hessian Components for CASSCF
**CyberTraining 2026 Capstone Project**

**Author:** Qabas Mohammad Al Khatib  
**University:** Case Western Reserve University, Department of Chemistry  
**Advisor:** Dr. Shane M. Parker  

---

## What is this project about?

My PhD dissertation focuses on making the Resonating Hartree-Fock (ResHF) method 
converge reliably. One of the biggest bottlenecks is computing an expensive Hessian 
matrix at every optimization step. This project asks: can we use machine learning 
to predict this Hessian from cheap molecular features, instead of computing it 
explicitly every time?

As a first step, I used CASSCF as a starting point (since ResHF is still being 
implemented), trained a neural network on 20 INVEST molecules, and got 5.30% 
prediction error on molecules the model had never seen before.

---

## What did I actually do?

**Step 1 — Generate data**  
I ran CASSCF(6,6)/def2-SVP calculations on 23 INVEST molecules from Pollice et al. 
2021 using PySCF on the UB CCR HPC cluster. 20 out of 23 molecules converged.

**Step 2 — Extract features**  
Each molecule has variable-sized arrays (different number of orbitals). I summarized 
them into 12 fixed-size features: gradient norms, energy, CI coefficient statistics, 
and Hessian diagonal statistics.

**Step 3 — Train a neural network**  
I used PyTorch to train a feedforward network (12 → 64 → 64 → 400) to predict the 
CI Hessian diagonal from the 12 features. Trained on 14 molecules, tested on 6.

**Step 4 — Results**  
5.30% overall relative error on the 6 held-out test molecules. The model generalizes 
reasonably well, though performance varies by molecule (see table below).

---

## Results

| | Value |
|--|-------|
| Training molecules | 14 |
| Test molecules | 6 |
| Relative error (S1000) | 2.49% |
| Relative error (S1200) | 4.78% |
| Relative error (S1180) | 4.80% |
| Relative error (S1010) | 5.95% |
| Relative error (S1050) | 5.73% |
| Relative error (S1090) | 7.01% |
| **Overall (PyTorch FFNN, 400-dim target)** | **5.30%** |
| Overall (ChemML MLP, 1-dim proxy baseline) | 21.61% |

Note: the ChemML baseline predicts only the *mean* of the 400-dim `ci_hdiag` vector 
(a limitation of ChemML's single-output regression interface), so its much higher 
error reflects that lossy 1-dim target — not a fair architecture comparison against 
the full PyTorch model.

**Why is S1090 the hardest to predict?** Looking at the spread (`std`) of the 
400-dim CI Hessian diagonal (`ci_hdiag`) across the 6 test molecules, the two 
hardest-to-predict molecules (S1090: 7.01% error, S1050: 5.73% error) also have 
the two highest `std_ci_hdiag` values (0.7130 and 0.6366, respectively) of the six — 
noticeably higher than the other four (0.31–0.49). This suggests molecules with a 
more heterogeneous CI Hessian diagonal are harder for a 12-feature model to 
summarize accurately.

One physically motivated hypothesis is that this heterogeneity reflects the 
energetic spread of different electronic configurations within the CAS(6,6) active 
space (e.g., closely spaced or widely split singlet/triplet character). This has 
**not been verified** — the current pipeline computes only a single CASSCF state 
per molecule, with no explicit multi-root or singlet-triplet gap calculation. A 
natural follow-up would be state-averaged CASSCF or NEVPT2 calculations to test 
this directly.

---

## Honest limitations

- I only predicted the CI Hessian diagonal — not the full coupled orbital-CI Hessian 
  (the H_κc block is not accessible via PySCF's `gen_g_hop` interface)
- CASSCF is not the same as ResHF (orthogonal vs nonorthogonal orbitals) — this is a 
  proof of concept, not a structural stand-in
- 20 of 23 staged molecules are present in the processed dataset; 3 (S1070, S1110, 
  S1120) are missing. The exact cause has not been confirmed — several data-generation 
  SLURM jobs were cancelled (time limit / signal termination) during this project, so 
  it's possible these are convergence failures, but this has not been individually 
  verified per molecule
- Only 20 molecules total — a reasonably sized proof of concept, but not yet enough 
  for a fully rigorous cross-validated error estimate
- The hypothesis connecting prediction difficulty to electronic state spacing 
  (singlet-triplet character) is speculative and not yet tested against actual 
  multi-state calculations

---

## What comes next?

This pipeline is a proof of concept for a larger goal: once the ResHF optimizer is 
implemented in my dissertation, I plan to retrain it on ResHF-specific Hessian data, 
extending to the full coupled Hessian including the orbital-CI cross terms. Along 
the way, I'll also investigate the missing molecules and run state-averaged 
CASSCF/NEVPT2 calculations to test the electronic-state-spacing hypothesis above.

---

## How to reproduce

```bash
git clone https://github.com/Qabas96/ml-hessian-casscf.git
cd ml-hessian-casscf
pip install pyscf torch numpy matplotlib scikit-learn
sbatch scripts/submit.slurm  # generates CASSCF data on HPC
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_chemml_baseline.ipynb
```

---

## Tools used

PySCF 2.13.1, PyTorch 2.8.0, ChemML 1.3, UB CCR HPC (Slurm)

## Citations

This project builds on the following software:

- **PySCF:** Sun, Q. et al. "Recent developments in the PySCF program package." 
  *J. Chem. Phys.* 153, 024109 (2020). https://doi.org/10.1063/5.0006074
- **PyTorch:** Paszke, A. et al. "PyTorch: An Imperative Style, High-Performance 
  Deep Learning Library." *Advances in Neural Information Processing Systems 32* 
  (NeurIPS 2019), 8024–8035.
- **ChemML:** Haghighatlari, M., Vishwakarma, G., Altarawy, D., Subramanian, R., 
  Kota, B. U., Sonpal, A., Setlur, S., & Hachmann, J. "ChemML: A machine learning 
  and informatics program package for the analysis, mining, and modeling of 
  chemical and materials data." *WIREs Computational Molecular Science* 10, e1458 
  (2020). https://doi.org/10.1002/wcms.1458