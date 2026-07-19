# Predicting GW/BSE Properties of PAH Crystals from Inexpensive PySCF Descriptors

## Introduction

Accurate prediction of quasiparticle band gaps and excitonic properties is central 
to the computational design of organic semiconductors. Within many-body perturbation theory, 
the GW approximation combined with the Bethe–Salpeter equation (GW/BSE) provides one of the most 
reliable first-principles frameworks for computing these quantities in molecular crystals. 
However, the computational expense of GW/BSE calculations limits their applicability in large-scale 
screening studies and machine learning workflows.

An attractive alternative is to use inexpensive quantum-chemical calculations as surrogate descriptors 
for machine learning models that approximate GW/BSE predictions. If sufficiently informative low-cost 
electronic structure features can be identified, accurate estimates of excited-state properties may be 
obtained at only a small fraction of the computational cost.

This project investigates that possibility using the PAH101 benchmark dataset of polycyclic aromatic 
hydrocarbon crystals. The central objective is to determine whether inexpensive gas-phase electronic 
structure calculations performed on isolated molecular monomers, combined with supervised machine 
learning, can reproduce GW/BSE quasiparticle band gaps, exciton energies, and singlet exciton 
binding energies.

To construct the low-cost descriptor set, Hartree–Fock calculations with the minimal STO-3G basis 
were performed using PySCF, followed by a single configuration interaction singles (CIS) excited-state 
calculation for each extracted monomer. These calculations deliberately neglect crystal packing effects, 
periodic boundary conditions, and electron correlation, providing an intentionally inexpensive baseline 
from which machine learning models learn corrections to the corresponding GW/BSE reference values.

⸻

## Dataset and Molecular Geometry Preparation

Molecular geometries were obtained from the PAH101 crystal structures using a monomer-extraction workflow. 
The workflow employs pymatgen and ASE to identify the largest bonded connected component within each unit 
cell and exports the resulting molecular geometry as an XYZ file. Approximately 20 structures containing 
crystallographic disorder could not be reliably processed by this heuristic and were excluded from subsequent analysis.

## Computational Workflow

```
geometries/*.xyz  (symlinked from pah_gap_ml)
      |
      v
pyscf_descriptors.py   HF/STO-3G + CIS, one molecule at a time
      |
      v
results/cheap_descriptors.csv
      |
      v                              PAH101_calculation_info_04162024.csv
build_dataset.py  <---------------------------  (GW_QP_BandGap, GWBSE_EsC,
      |                                          Singlet_binding, + tabulated
      v                                          crystal features)
results/dataset.csv
      |
      +----------------------------+
      v                            v
train_ml.py                chemml_automl.py
(Ridge / RF / GB,          (8 sklearn regressors,
 5-fold CV)                 GA-tuned by ChemML)
      |                            |
      v                            v
results/*.png,              results/chemml_rerun/*
results/ml_summary_*.json
      |
      v
results_report.py  -->  results/summary.md
```



## Descriptor Generation and Dataset Construction

After successful data processing, the final machine learning dataset comprised 64 molecular crystals.
The input feature set combined inexpensive electronic descriptors computed with PySCF—including the HOMO–LUMO gap, 
first CIS excitation energy (S₁), S₁ oscillator strength, molecular dipole moment, and atom count—with five
crystal-level descriptors provided by the PAH101 dataset (GapC, AtomNumC, MolWtS, RhoC, and EpsilonC). 
this hybrid feature representation allows the models to leverage both low-cost quantum-mechanical information 
and readily available structural descriptors.

## Results

**sklearn (**`train_ml.py`**, 5-fold CV):**


| Target                         | Best model        | RMSE (eV) | R²   |
| ------------------------------ | ----------------- | --------- | ---- |
| GW quasiparticle band gap      | Ridge             | 0.22      | 0.96 |
| GW+BSE exciton energy          | Ridge             | 0.27      | 0.89 |
| Singlet exciton binding energy | Gradient Boosting | 0.24      | 0.51 |


**ChemML AutoML (**`chemml_automl.py`**, GA-tuned, held-out split):**


| Target                         | Best model        | RMSE (eV) | R²   |
| ------------------------------ | ----------------- | --------- | ---- |
| GW quasiparticle band gap      | Gradient Boosting | 0.25      | 0.94 |
| GW+BSE exciton energy          | Ridge             | 0.36      | 0.80 |
| Singlet exciton binding energy | Ridge             | 0.18      | 0.70 |


Full tables and the actual figures are in
`[results/summary.md](results/summary.md)`.

Both the scikit-learn and ChemML workflows achieved comparable predictive performance 
for the GW quasiparticle band gap and GW/BSE exciton energy. In both cases, Ridge regression 
consistently ranked among the best-performing models, indicating that the relationship between 
the inexpensive PySCF descriptors and the target properties is largely linear. More complex 
ensemble methods provided only modest improvements for these quantities.

Prediction of the singlet exciton binding energy proved substantially more challenging. 
Although the ChemML workflow achieved higher performance than the manually selected baseline models, 
the small dataset (64 crystals) and differing validation strategies preclude a direct comparison of 
the reported metrics. Nevertheless, both workflows consistently identified the binding energy 
as the most difficult target.

This trend is consistent with the underlying physics. Unlike the quasiparticle band gap and exciton excitation energy, 
the exciton binding energy depends strongly on electron–hole interactions and many-body correlation effects, which are 
not explicitly represented in a gas-phase Hartree–Fock calculation. Consequently, while inexpensive single-particle 
descriptors capture much of the information required to predict band gaps and excitation energies, they provide a less 
complete description of exciton binding, resulting in reduced predictive accuracy.

## Repo layout

```
capstone/
  config.py               paths, feature/target lists, all the knobs
  pyscf_descriptors.py    cheap HF/STO-3G + CIS on each monomer
  build_dataset.py        merges cheap descriptors with PAH101 labels
  train_ml.py             sklearn CV baselines + figures
  chemml_automl.py        ChemML GA model screen
  chemml_plots.py         bar charts from the ChemML run
  results_report.py       writes results/summary.md
  geometries/             symlink -> pah_gap_ml/geometries
  submit/                 SLURM scripts for all of the above
  results/
    cheap_descriptors.csv, dataset.csv
    fig_ml_*.png, ml_summary_*.json
    chemml_rerun/          ChemML's own output, kept separate on purpose
    summary.md
```



## Running it

Everything is meant to go through SLURM, not the login/interactive node:

```bash
cd qm_ml_capstone

sbatch submit/submit_pyscf_descriptors.slurm 
sbatch submit/submit_pyscf_descriptors.slurm                       

# dataset + sklearn CV, depends on cheap_descriptors.csv existing
sbatch submit/submit_ml.slurm

sbatch submit/submit_chemml.slurm
sbatch submit/submit_chemml_plots.slurm
```

Alternatively, to run locally:

```bash
/path/to/your/qc_env/bin/python pyscf_descriptors.py
/path/to/your/.venv_chemml/bin/python build_dataset.py
/path/to/your/.venv_chemml/bin/python train_ml.py --target GW_QP_BandGap
/path/to/your/.venv_chemml/bin/python \
    chemml_automl.py --target GW_QP_BandGap --out-dir results/chemml_rerun
```



## Caveats

64 crystals is a small dataset -- CV numbers will move around if you change the random seed or the fold count. This is a screening tool for one molecular family (PAH crystals), not a general-purpose GW/BSE surrogate, and the exciton binding energy result should be read as "weakly predictable," only