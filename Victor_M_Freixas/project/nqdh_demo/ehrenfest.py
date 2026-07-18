"""Diabatic mean-field (Ehrenfest) dynamics.

The mean-field complement to the FSSH in ``methods.py``, and a faithful Python
prototype of the Libra diabatic-Ehrenfest target used for the S1/S2 beating. Nuclei
move under the mean-field force ``F = -Tr(rho dW/dR)``; electrons evolve under the
diabatic Hamiltonian ``i hbar c_dot = W c`` -- there is **no** ``v.d`` term because
diabatic states carry no derivative coupling, so nothing diverges at the S1/S2
near-degeneracies where adiabatic Ehrenfest (and FSSH) blow up. The beating is read
off by diagonalising ``W`` along the trajectory and projecting ``c`` onto the
adiabatic eigenvectors -> adiabatic populations.

Atomic units throughout (Hartree, bohr, hbar = 1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import units
from .integrators import kinetic_energy, remove_com_and_rotation, remove_com_velocity


@dataclass
class DiabaticES:
    """Diabatic Hamiltonian and its nuclear gradient at one geometry (atomic units).

    W  : (K, K)              diabatic Hamiltonian (Hartree), real symmetric
    dW : (K, K, n_atoms, 3)  dW_mn/dR (Hartree/bohr), real, symmetric in (m, n)
    """

    W: np.ndarray
    dW: np.ndarray

    @property
    def n_states(self) -> int:
        return self.W.shape[0]


class DiabaticProvider(ABC):
    """Return the diabatic ``W`` and ``dW/dR`` at a geometry (atomic units)."""

    @abstractmethod
    def evaluate_diabatic(self, Z: np.ndarray, R: np.ndarray) -> DiabaticES:
        """Z: (n_atoms,) atomic numbers; R: (n_atoms, 3) bohr."""


def diabatic_force(rho: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """Mean-field force (n_atoms, 3): F = -sum_mn Re(rho_mn) dW_mn/dR (Hellmann-Feynman)."""
    return -np.einsum("mn,mnad->ad", rho.real, dW)


def propagate_diabatic(c: np.ndarray, W_a: np.ndarray, W_b: np.ndarray, dt: float, n_sub: int = 20) -> np.ndarray:
    """Propagate ``i c_dot = W c`` over ``[t, t+dt]`` with W linear from W_a to W_b.

    Exponential-midpoint substeps: on each substep W is frozen at its midpoint value
    and the exact unitary ``exp(-i W_mid dt_sub)`` is applied via the eigendecomposition
    of the (small, real-symmetric) W -- norm-conserving by construction.
    """
    dts = dt / n_sub
    for s in range(n_sub):
        f = (s + 0.5) / n_sub
        Wm = (1.0 - f) * W_a + f * W_b
        lam, Q = np.linalg.eigh(Wm)
        c = Q @ (np.exp(-1j * lam * dts) * (Q.T @ c))
    return c


def adiabatic_populations(c: np.ndarray, W: np.ndarray):
    """(|<phi_i|psi>|^2, eigenvalues) for the eigenstates phi_i of W (ascending)."""
    lam, Q = np.linalg.eigh(W)
    a = Q.T @ c
    return np.abs(a) ** 2, lam


@dataclass
class EhrenfestResult:
    time_fs: np.ndarray
    adiabatic_pops: np.ndarray   # (n_steps+1, K)
    energies: np.ndarray         # (n_steps+1, K) adiabatic eigenvalues (Hartree)
    total_energy: np.ndarray     # (n_steps+1,) KE + <psi|W|psi> (Hartree)
    coeffs: np.ndarray           # (n_steps+1, K) complex diabatic amplitudes

    def energy_drift_eV(self) -> float:
        e = self.total_energy
        return float(np.max(np.abs(e - e[0]))) * units.HARTREE_EV

    @property
    def diabatic_pops(self) -> np.ndarray:
        return np.abs(self.coeffs) ** 2


def propagate_diabatic_batch(c: np.ndarray, W_a: np.ndarray, W_b: np.ndarray, dt: float, n_sub: int = 20) -> np.ndarray:
    """Batched :func:`propagate_diabatic`: c (B, K), W_a/W_b (B, K, K)."""
    dts = dt / n_sub
    for s in range(n_sub):
        f = (s + 0.5) / n_sub
        lam, Q = np.linalg.eigh((1.0 - f) * W_a + f * W_b)          # batched eigh
        a = np.einsum("bkm,bk->bm", Q, c.conj()).conj()             # Q^T c per batch
        c = np.einsum("bkm,bm->bk", Q, np.exp(-1j * lam * dts) * a)
    return c


@dataclass
class EnsembleResult:
    time_fs: np.ndarray          # (n_frames,)
    adiabatic_pops: np.ndarray   # (n_frames, B, K)
    energies: np.ndarray         # (n_frames, B, K) adiabatic eigenvalues (Hartree)
    total_energy: np.ndarray     # (n_frames, B)

    @property
    def mean_pops(self) -> np.ndarray:
        """(n_frames, K) ensemble-averaged adiabatic populations -- the beating observable."""
        return self.adiabatic_pops.mean(axis=1)

    def energy_drift_eV(self) -> np.ndarray:
        e = self.total_energy
        return np.max(np.abs(e - e[0]), axis=0) * units.HARTREE_EV   # (B,)


def run_ehrenfest_ensemble(
    batch_eval,                  # callable (Z, R (B,n,3)) -> (W (B,K,K), dW (B,K,K,n,3)), au
    Z: np.ndarray,
    R0: np.ndarray,              # (B, n_atoms, 3) bohr
    V0: np.ndarray,              # (B, n_atoms, 3) au
    masses: np.ndarray,          # (n_atoms,)
    c0: np.ndarray,              # (B, K) complex diabatic amplitudes
    dt: float,
    n_steps: int,
    *,
    remove_com: bool = True,
    n_sub: int = 20,
    stride: int = 1,             # record every this many steps
    progress=None,               # optional callable(step)
) -> EnsembleResult:
    """Vectorized ensemble Ehrenfest: one batched W/dW evaluation per step for ALL
    trajectories. Physics identical to :func:`run_ehrenfest` (validated against it
    and against Libra); this exists purely for ensemble throughput."""
    R = np.asarray(R0, float).copy()
    V = np.asarray(V0, float).copy()
    masses = np.asarray(masses, float)
    B = R.shape[0]
    c = np.asarray(c0, complex).copy()
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    if remove_com:
        for b in range(B):
            V[b] = remove_com_and_rotation(masses, R[b], V[b])

    W, dW = batch_eval(Z, R)
    rho = np.einsum("bm,bn->bmn", c, c.conj())
    A = -np.einsum("bmn,bmnad->bad", rho.real, dW) / masses[None, :, None]

    times, pops, energies, etot = [], [], [], []

    def record(step: int) -> None:
        lam, Q = np.linalg.eigh(W)
        a = np.einsum("bkm,bk->bm", Q, c.conj()).conj()
        ke = 0.5 * np.sum(masses[None, :, None] * V ** 2, axis=(1, 2))
        pe = np.einsum("bmn,bmn->b", np.einsum("bm,bn->bmn", c, c.conj()).real, W)
        times.append(step * units.autime_to_fs(dt))
        pops.append(np.abs(a) ** 2)
        energies.append(lam)
        etot.append(ke + pe)

    record(0)
    for step in range(1, n_steps + 1):
        R_new = R + V * dt + 0.5 * A * dt ** 2
        V_half = V + 0.5 * A * dt
        W_new, dW_new = batch_eval(Z, R_new)
        c = propagate_diabatic_batch(c, W, W_new, dt, n_sub)
        rho = np.einsum("bm,bn->bmn", c, c.conj())
        A = -np.einsum("bmn,bmnad->bad", rho.real, dW_new) / masses[None, :, None]
        V = V_half + 0.5 * A * dt
        if remove_com:
            V -= (masses[None, :, None] * V).sum(axis=1, keepdims=True) / masses.sum()
        R, W, dW = R_new, W_new, dW_new
        if step % stride == 0 or step == n_steps:
            record(step)
        if progress is not None:
            progress(step)

    return EnsembleResult(np.array(times), np.array(pops), np.array(energies), np.array(etot))


def run_ehrenfest(
    provider: DiabaticProvider,
    Z: np.ndarray,
    R0: np.ndarray,
    V0: np.ndarray,
    masses: np.ndarray,
    c0: np.ndarray,
    dt: float,
    n_steps: int,
    *,
    remove_com: bool = True,
    n_sub: int = 20,
) -> EhrenfestResult:
    """Velocity-Verlet nuclei + diabatic electronic propagation. R0/V0 bohr & au; dt au."""
    R = np.asarray(R0, float).copy()
    V = np.asarray(V0, float).copy()
    masses = np.asarray(masses, float)
    c = np.asarray(c0, complex).copy()
    c /= np.linalg.norm(c)
    if remove_com:
        V = remove_com_and_rotation(masses, R, V)

    es = provider.evaluate_diabatic(Z, R)
    A = diabatic_force(np.outer(c, c.conj()), es.dW) / masses[:, None]

    times, pops, energies, etot, coeffs = [], [], [], [], []

    def record(step: int) -> None:
        p, lam = adiabatic_populations(c, es.W)
        pe = float((np.outer(c, c.conj()).real * es.W).sum())      # Re(c^dag W c)
        times.append(step * units.autime_to_fs(dt))
        pops.append(p)
        energies.append(lam)
        etot.append(kinetic_energy(masses, V) + pe)
        coeffs.append(c.copy())

    record(0)
    for step in range(1, n_steps + 1):
        R_new = R + V * dt + 0.5 * A * dt ** 2                     # verlet drift
        V_half = V + 0.5 * A * dt
        es_new = provider.evaluate_diabatic(Z, R_new)
        c = propagate_diabatic(c, es.W, es_new.W, dt, n_sub)        # electrons over the step
        A = diabatic_force(np.outer(c, c.conj()), es_new.dW) / masses[:, None]
        V = V_half + 0.5 * A * dt                                  # verlet velocity completion
        if remove_com:
            V = remove_com_velocity(masses, V)
        R, es = R_new, es_new
        record(step)

    return EhrenfestResult(np.array(times), np.array(pops), np.array(energies),
                           np.array(etot), np.array(coeffs))
