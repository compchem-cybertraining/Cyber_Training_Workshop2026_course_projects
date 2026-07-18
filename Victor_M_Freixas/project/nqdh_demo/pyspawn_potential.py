"""NQDH potential module for pySpawn17 (full multiple spawning).

pySpawn convention (template: ``pyspawn/potential/test_cone.py``): every function in
this module is grafted onto the ``traj`` class by
``pyspawn.import_methods.into_traj(this_module)``, so the functions below take
``self`` (a pySpawn traj) as first argument.  Three are required
(``compute_elec_struct``, ``init_h5_datasets``, ``potential_specific_traj_copy``);
we additionally define ``rescale_momentum`` so the graft *overrides* pySpawn's
spawn-time momentum adjustment through its own extension mechanism (no source
edits): stock pySpawn rescales the child momentum by a scalar (i.e. along the full
velocity vector), which smears the Pechukas kick over all modes; mode "coupling"
instead adjusts along the S1-S2 derivative-coupling direction (the physical kick
direction, and the one the ensemble beating rides on).

NQDH mapping (all quantities already in atomic units, pySpawn's units):
  energies  = eigvalsh(W)                       (K,)
  forces    = -diag(U^T dW U)  per state        (K, ndims)
  wf        = U^T  (eigenvectors as rows, in the FIXED diabatic basis; length_wf=K)
              -> pySpawn's step overlap prev_wf @ wf.T equals U(t)^T U(t+dt),
              exactly the object its NPI time-derivative coupling wants.
  tdc       = NPI on the (S1,S2) 2x2 block of that overlap; S0 is decoupled by
              construction (zero off-diagonal W row/col), so its couplings are 0.

Call ``configure(run_dir, Z, device=..., rescale=...)`` once before grafting.
"""
from __future__ import annotations

import math

import numpy as np

_STATE = {"provider": None, "Z": None, "rescale": "velocity",
          "n_spawns": 0, "n_frustrated": 0, "coupled_pair": (1, 2),
          "events": []}   # (sim_time_au, accepted) per spawn attempt -- gate timing data


def configure(run_dir, Z, device="cuda", rescale="velocity", coupled_pair=(1, 2)):
    """Load the NQDH provider once (module-level: grafted methods share it)."""
    from .provider import NQDHDiabaticProvider
    if rescale not in ("velocity", "coupling"):
        raise ValueError(f"rescale must be 'velocity' or 'coupling', got {rescale!r}")
    _STATE["provider"] = NQDHDiabaticProvider(run_dir, device=device)
    _STATE["Z"] = np.asarray(Z, dtype=np.int64)
    _STATE["rescale"] = rescale
    _STATE["coupled_pair"] = tuple(coupled_pair)
    _STATE["n_spawns"] = 0
    _STATE["n_frustrated"] = 0
    _STATE["events"] = []


def _eval(positions_flat):
    """Provider call at one geometry (bohr, flat (3N,)) -> lam (K,), U (K,K), M (K,K,N,3)."""
    prov, Z = _STATE["provider"], _STATE["Z"]
    R = np.asarray(positions_flat, float).reshape(1, len(Z), 3)
    W, dW = prov.evaluate_diabatic_batch(Z, R)
    W = np.asarray(W)[0]; dW = np.asarray(dW)[0]
    lam, U = np.linalg.eigh(W)
    M = np.einsum("mi,mnad,nj->ijad", U, dW, U)
    return lam, U, M


def compute_elec_struct(self, zbackprop):
    cbackprop = "backprop_" if zbackprop else ""

    getattr(self, "set_" + cbackprop + "prev_wf")(getattr(self, "get_" + cbackprop + "wf")())

    pos = getattr(self, "get_" + cbackprop + "positions")()
    lam, U, M = _eval(pos)
    K = self.numstates

    getattr(self, "set_" + cbackprop + "energies")(lam[:K].copy())

    f = np.empty((K, self.numdims))
    for i in range(K):
        f[i] = -M[i, i].reshape(-1)
    getattr(self, "set_" + cbackprop + "forces")(f)

    # electronic "wavefunction" = eigenvector rows in the fixed diabatic basis
    wf = U.T[:K, :K].copy()
    prev_wf = getattr(self, "get_" + cbackprop + "prev_wf")()
    if not np.any(prev_wf):                       # very first call: no history yet
        prev_wf = wf.copy()
    S = np.matmul(prev_wf, wf.T)                  # = U(t)^T U(t+dt)
    for i in range(K):                            # phase for continuity (diag > 0)
        if S[i, i] < 0.0:
            wf[i, :] *= -1.0
            S[:, i] *= -1.0

    # NPI tdc on the coupled 2x2 block; S0 decoupled -> tdc stays 0 elsewhere
    lo, hi = _STATE["coupled_pair"]
    tdc = np.zeros(K)
    if self.istate in (lo, hi):
        jstate = hi if self.istate == lo else lo
        Ssub = S[np.ix_([lo, hi], [lo, hi])].copy()
        # the step overlap is orthogonal, so roundoff can leave |S_ij| = 1 + eps;
        # pySpawn's NPI clamps the wrong elements for that case and arcsin returns
        # NaN -- one NaN here poisons the quantum Hamiltonian (la.eig crash).
        np.clip(Ssub, -1.0, 1.0, out=Ssub)
        val = float(self.compute_tdc(Ssub))
        tdc[jstate] = val if math.isfinite(val) else 0.0
    getattr(self, "set_" + cbackprop + "timederivcoups")(tdc)

    getattr(self, "set_" + cbackprop + "wf")(wf)


def rescale_momentum(self, v_parent):
    """Spawn-time momentum adjustment (grafted OVER pySpawn's own).

    mode "velocity": bitwise-identical math to stock pySpawn (scalar rescale of the
    whole momentum vector).  mode "coupling": adjust along the normalized S1-S2
    derivative coupling d12 = M12/(E_hi - E_lo) at the spawn geometry -- solve
    sum_a (p_a + lam*dhat_a)^2 / 2m_a = T_parent + V_parent - V_child for lam and
    take the smaller-|lam| root.  Either mode returns False on a frustrated spawn
    (not enough energy), mirroring stock semantics.
    """
    v_child = self.get_energies()[self.get_istate()]
    p_parent = self.get_momenta()
    m = self.get_masses()
    t_parent = float(np.sum(0.5 * p_parent * p_parent / m))
    _t_now = float(getattr(self, "time", float("nan")))

    if _STATE["rescale"] == "velocity":
        factor = (v_parent + t_parent - v_child) / t_parent
        if factor < 0.0:
            _STATE["n_frustrated"] += 1
            _STATE["events"].append((_t_now, 0))
            print("# Aborting spawn because child does not have")
            print("# enough energy for momentum adjustment (velocity mode)")
            return False
        factor = math.sqrt(factor)
        print("# rescaling momentum by factor ", factor)
        p_child = factor * p_parent
    else:
        lam_e, U, M = _eval(self.get_positions())
        lo, hi = _STATE["coupled_pair"]
        gap = lam_e[hi] - lam_e[lo]
        d = (M[lo, hi].reshape(-1) / gap) if abs(gap) > 1e-10 else M[lo, hi].reshape(-1)
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            print("# coupling direction vanished; falling back to velocity rescale")
            _STATE["rescale"], out = "velocity", rescale_momentum(self, v_parent)
            _STATE["rescale"] = "coupling"
            return out
        dhat = d / nrm
        a = float(np.sum(dhat * dhat / (2.0 * m)))
        b = float(np.sum(p_parent * dhat / m))
        c = v_child - v_parent
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            _STATE["n_frustrated"] += 1
            _STATE["events"].append((_t_now, 0))
            print("# Aborting spawn: frustrated along the coupling direction")
            return False
        r1 = (-b + math.sqrt(disc)) / (2.0 * a)
        r2 = (-b - math.sqrt(disc)) / (2.0 * a)
        lam = r1 if abs(r1) < abs(r2) else r2
        print("# adjusting momentum along d12 by lambda ", lam)
        p_child = p_parent + lam * dhat

    _STATE["n_spawns"] += 1
    _STATE["events"].append((_t_now, 1))
    self.set_momenta(p_child)
    self.set_backprop_momenta(p_child)
    return True


def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    for i in range(self.numstates):
        self.h5_datasets["wf" + str(i)] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates


def potential_specific_traj_copy(self, from_traj):
    return


# h5_output resolves each dataset name to a "get_<name>" method; provide one per
# wf row (test_cone does the same for its two states).
def get_wf0(self):
    return self.wf[0, :].copy()


def get_wf1(self):
    return self.wf[1, :].copy()


def get_wf2(self):
    return self.wf[2, :].copy()


def get_backprop_wf0(self):
    return self.backprop_wf[0, :].copy()


def get_backprop_wf1(self):
    return self.backprop_wf[1, :].copy()


def get_backprop_wf2(self):
    return self.backprop_wf[2, :].copy()
