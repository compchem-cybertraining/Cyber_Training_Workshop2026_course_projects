"""Libra adapter for the NQDH model: expose W(R), dW/dR as a Libra compute_model.

Bridges the trained NQDH (via :class:`NQDHDiabaticProvider`, atomic units) to
Libra's model-Hamiltonian convention (the ``libra_py.models`` pattern): a callback
``compute_model(q, params, full_id)`` returning ``obj`` with

    ham_dia    CMATRIX(K, K)            diabatic Hamiltonian (Hartree)
    ovlp_dia   CMATRIX(K, K)            identity (orthonormal diabats)
    d1ham_dia  CMATRIXList, ndof items  dW/dR_i (Hartree/bohr), ndof = 3*n_atoms
    dc1_dia    CMATRIXList, ndof items  zeros (strict diabats: no derivative coupling)

Libra hands nuclear DOFs as a flat (ndof, ntraj) MATRIX in bohr; we reshape the
requested trajectory's column to (n_atoms, 3). Everything model-specific stays in
the provider -- this file is molecule-agnostic plumbing.

Usage (inside the ``libra`` conda env):
    from src.dynamics.libra_adapter import make_nqdh_compute_model
    compute_model, K, ndof = make_nqdh_compute_model(run_dir, Z)
    ... tsh.generic_recipe(dyn_params, compute_model, {"model0": 0, "nstates": K}, ...)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_nqdh_compute_model(run_dir: str | Path, Z: np.ndarray, device: str = "cpu"):
    """Build the Libra ``compute_model`` for a trained NQDH run dir.

    Returns ``(compute_model, K, ndof)``. The provider is loaded once and closed
    over; each call costs one forward + K(K+1)/2 autograd passes.
    """
    from liblibra_core import CMATRIX, CMATRIXList, Cpp2Py

    from .provider import NQDHDiabaticProvider

    provider = NQDHDiabaticProvider(run_dir, device=device)
    Z = np.asarray(Z, dtype=np.int64)
    n_atoms = int(Z.shape[0])
    ndof = 3 * n_atoms
    K = provider.K

    class Tmp:
        pass

    def compute_model(q, params, full_id):
        indx = Cpp2Py(full_id)[-1]
        R_bohr = np.array([q.get(i, indx) for i in range(ndof)], dtype=np.float64).reshape(n_atoms, 3)
        es = provider.evaluate_diabatic(Z, R_bohr)                 # W (K,K) Ha, dW (K,K,n_atoms,3) Ha/bohr
        dW = es.dW.reshape(K, K, ndof)

        ham_dia = CMATRIX(K, K)
        ovlp_dia = CMATRIX(K, K)
        ovlp_dia.identity()
        d1ham_dia = CMATRIXList()
        dc1_dia = CMATRIXList()
        for _ in range(ndof):
            d1ham_dia.append(CMATRIX(K, K))
            dc1_dia.append(CMATRIX(K, K))

        for m in range(K):
            for n in range(m, K):
                ham_dia.set(m, n, es.W[m, n] * (1.0 + 0.0j))
                if n != m:
                    ham_dia.set(n, m, es.W[m, n] * (1.0 + 0.0j))
        for i in range(ndof):
            for m in range(K):
                for n in range(m, K):
                    v = dW[m, n, i]
                    if v != 0.0:
                        d1ham_dia[i].set(m, n, v * (1.0 + 0.0j))
                        if n != m:
                            d1ham_dia[i].set(n, m, v * (1.0 + 0.0j))

        obj = Tmp()
        obj.ham_dia = ham_dia
        obj.ovlp_dia = ovlp_dia
        obj.d1ham_dia = d1ham_dia
        obj.dc1_dia = dc1_dia
        return obj

    return compute_model, K, ndof
