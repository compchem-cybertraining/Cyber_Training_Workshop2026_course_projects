"""Minimal runtime for the NQDH S1/S2 beating demo.

A frozen Neural Quasi-Diabatic Hamiltonian drives diabatic mean-field (Ehrenfest)
dynamics. Two engines consume the identical model W(R), dW/dR:

  - ``libra_adapter.make_nqdh_compute_model`` -> Libra (the school software), and
  - ``ehrenfest.run_ehrenfest`` -> a pure-numpy reference engine (no Libra needed).

Only :mod:`nqdh_demo.provider` depends on hippynn + torch (to evaluate the model).
"""
from .provider import NQDHDiabaticProvider

__all__ = ["NQDHDiabaticProvider"]
