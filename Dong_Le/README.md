# 📐 Auto-differentiation Non-Adiabatic Coupling 

This project demonstrates how to compute non-adiabatic couplings using automatic differentiation in **JAX** for simple Holstein and Morse model Hamiltonians implemented in **Libra** package.

## ∂ Motivation

Non-adiabatic molecular dynamics requires accurate coupling between electronic states and nuclear motion. Evaluating derivatives can be challenging, especially when working with new basis sets or unfamiliar models.

## ∂ Models

- Holstein2: sparse matrix with many zeros
- Morse potential: dense matrix with more complex structure

## ∂ Approach

- Compare automatic differentiation with analytical derivatives
- Explore ways to implement a function 

##  ∂ Dependencies

- Python
- JAX and jaxlib 
- Libra 
- NumPy, Jupyter Notebook, Matplotlib


## ∂ Getting Started

1. Copy and overwrite files to /src/libra_py/models
2. Open Jupyter notebook

