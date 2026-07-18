"""Diabatic ``W`` and ``dW/dR`` from a trained NQDH model.

Loads the frozen hippynn NQDH graph and queries its ``W`` node -- the native
learned diabatic Hamiltonian, smooth and gauge-fixed by construction -- plus its
nuclear gradient via one autograd pass per unique W entry. Units are converted at
this boundary: the model is trained in (eV, Angstrom); the dynamics engine and the
Libra adapter consume atomic units (Hartree, bohr).

This is the only module that depends on hippynn + torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import units
from .ehrenfest import DiabaticES, DiabaticProvider


class NQDHDiabaticProvider(DiabaticProvider):
    """Diabatic ``W`` and ``dW/dR`` from a trained NQDH run dir, for diabatic Ehrenfest.

    The GS-decoupled block structure (``W[0, j] = 0``) is preserved, so an S2 initial
    coefficient leaves S0 empty automatically.
    """

    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        import torch
        import hippynn  # noqa: F401  (registers node classes for unpickling)

        self.run_dir = Path(run_dir).resolve()
        cfg = json.loads((self.run_dir / "nqdh_training_config.json").read_text())
        self.K = len(cfg["band_states"])
        # w_graph.pt is a standalone hippynn GraphModule (inputs Z, R -> W). Its
        # pickle references the model's node classes under ``src.*`` (vendored in
        # this repo's ``src/`` package), so the repo root must be importable.
        self.wsub = torch.load(self.run_dir / "w_graph.pt", map_location=device, weights_only=False)
        self.wsub.eval()
        self.input_nodes = list(self.wsub.input_nodes)
        self.device = device
        self._torch = torch

    def evaluate_diabatic(self, Z: np.ndarray, R_bohr: np.ndarray) -> DiabaticES:
        torch = self._torch
        R_ang = units.bohr_to_angstrom(np.asarray(R_bohr, dtype=np.float64))
        n = R_ang.shape[0]
        Zt = torch.as_tensor(np.asarray(Z, dtype=np.int64)[None, :], device=self.device)
        Rt = torch.tensor(R_ang[None, :, :], dtype=torch.float32, device=self.device, requires_grad=True)
        by = {"Z": Zt, "R": Rt}
        out = self.wsub(*[by[node.db_name] for node in self.input_nodes])
        W = (out[0] if isinstance(out, (list, tuple)) else out)[0]          # (K, K) eV
        K = W.shape[0]
        dW = np.zeros((K, K, n, 3))
        for m in range(K):
            for j in range(m, K):                                          # unique entries; W symmetric
                g = torch.autograd.grad(W[m, j], Rt, retain_graph=True)[0][0]  # (n, 3) eV/Angstrom
                dW[m, j] = dW[j, m] = g.detach().cpu().numpy()
        W_au = units.ev_to_hartree(W.detach().cpu().numpy())
        dW_au = units.ev_to_hartree(dW) * units.BOHR_ANGSTROM
        return DiabaticES(W=W_au, dW=dW_au)

    def evaluate_diabatic_batch(self, Z: np.ndarray, R_bohr: np.ndarray):
        """Batched W and dW/dR for an ensemble: one forward + K(K+1)/2 backward passes
        for ALL geometries (grad of the batch-sum is exact -- geometries decouple).

        Z: (n_atoms,), R_bohr: (B, n_atoms, 3). Returns (W (B,K,K) Ha, dW (B,K,K,n,3) Ha/bohr).
        """
        torch = self._torch
        R_ang = units.bohr_to_angstrom(np.asarray(R_bohr, dtype=np.float64))
        B, n, _ = R_ang.shape
        Zt = torch.as_tensor(np.broadcast_to(np.asarray(Z, np.int64)[None, :], (B, len(Z))).copy(),
                             device=self.device)
        Rt = torch.tensor(R_ang, dtype=torch.float32, device=self.device, requires_grad=True)
        by = {"Z": Zt, "R": Rt}
        out = self.wsub(*[by[node.db_name] for node in self.input_nodes])
        W = out[0] if isinstance(out, (list, tuple)) else out               # (B, K, K) eV
        K = W.shape[-1]
        dW = np.zeros((B, K, K, n, 3))
        for m in range(K):
            for j in range(m, K):
                g = torch.autograd.grad(W[:, m, j].sum(), Rt, retain_graph=True, allow_unused=True)[0]
                if g is None:
                    continue                                                # structural zero (decoupled pair)
                gn = g.detach().cpu().numpy()
                dW[:, m, j] = gn
                if j != m:
                    dW[:, j, m] = gn
        W_au = units.ev_to_hartree(W.detach().cpu().numpy())
        dW_au = units.ev_to_hartree(dW) * units.BOHR_ANGSTROM
        return W_au, dW_au
