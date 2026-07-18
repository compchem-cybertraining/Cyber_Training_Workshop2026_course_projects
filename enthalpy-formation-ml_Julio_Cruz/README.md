# Machine-Learning Models for Standard Enthalpies of Formation

This repository contains a notebook-based workflow for predicting
standard enthalpies of formation,
$\Delta_{\mathrm{f}}H^\circ(298.15\,\mathrm{K})$, using molecular
composition, bond counts, molecular descriptors, and machine-learning models.

The project uses thermochemical values from the Active Thermochemical Tables
(ATcT), assigns molecular structures through PubChem, generates descriptors
with RDKit and Mordred/ChemML, and compares linear and nonlinear regressors.

## Repository structure

```text
enthalpy-formation-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── thermochemical_acta.xlsx
│   ├── ATcT_molecules_features.csv
│   └── ATcT_bonds_corrected.csv
├── figures/
│   └── ATcT_formation_enthalpy.jpg
└── notebooks/
    ├── 01_data_preparation_and_bond_descriptors.ipynb
    ├── 02_exploratory_analysis.ipynb
    ├── 03_model_comparison.ipynb
```

## Notebook order

1. **Data preparation and bond descriptors**  
   Parses the ATcT spreadsheet, separates molecular and atomic entries,
   retrieves PubChem structures, and constructs atom- and bond-count features.

2. **Exploratory analysis**  
   Examines the target distribution, descriptor correlations, sparsity,
   Random Forest feature importance, and the effect of molecular charge.

3. **Model comparison**  
   Compares linear regression, Ridge, Random Forest, gradient boosting,
   Extra Trees, and XGBoost using several molecular representations.

The archived notebook preserves early exploratory work but is not part of the
maintained workflow.

## Included data

- `thermochemical_acta.xlsx` is the source spreadsheet used by the
  preprocessing notebook.
- `ATcT_molecules_features.csv` is an intermediate table retained for
  traceability.
- `ATcT_bonds_corrected.csv` is the processed table used by the analysis and
  modeling notebooks.

The repository copy of `ATcT_bonds_corrected.csv` contains
**3,235 populated molecular records** and **109
columns**. Completely empty trailing rows from the uploaded CSV were removed.

## Reported model result

On the fixed 80/20 random holdout split, the best saved model was XGBoost:

- RMSE: **155.38 kJ mol$^{-1}$**
- MAE: **95.75 kJ mol$^{-1}$**
- $R^2$: **0.92165**

These values describe performance under a random split and should not be
interpreted as a complete measure of transferability to unseen chemical
families.

## Use

Create a Python environment with the packages in `requirements.txt`, launch
Jupyter, and open the notebooks in numerical order. No run-all script is
included.

The PubChem retrieval cells require internet access and may return different
results as external records change. The processed CSV is included so that the
analysis and model notebooks can be inspected without repeating those queries.

## Reproducibility notes

- The main model comparisons use an 80/20 split.
- Randomized procedures use `random_state=42` where specified.
- Notebook outputs are retained to document the completed analysis.
- Expensive descriptor calculations and hyperparameter searches may take
  substantial time.
