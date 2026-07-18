"""
pyscf_libra -- a PySCF backend for Libra's on-the-fly nonadiabatic dynamics.

Drop this next to your notebook and:

    from pyscf_libra import PySCFSource, compute_model, make_model_params

    source = PySCFSource(["C", "O"], basis="6-31G*", xc="pbe0", nexc=3)
    model_params = make_model_params(source)
    res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params,
                                      elec_params, nucl_params, rnd)

`compute_model` occupies the same slot as `libra_py.models.Holstein.Holstein2`, but
returns ADIABATIC quantities: PySCF hands us adiabatic states directly and there is
no diabatic basis to transform from.


WHAT LIBRA MUST BE TOLD (post-load overrides, after recipe.load(dyn_general))
============================================================================
The stock recipes are written for 2-state model Hamiltonians. On an on-the-fly path
you must override:

    dyn_general.update({
        "ham_update_method": 2,      # adiabatic properties come FROM the model
        "ham_transform_method": 0,   # no diabatic Ham exists to diagonalize
        "time_overlap_method": 0,    # we supply time_overlap_adi; don't recompute
        "nac_update_method": 2,      # NACs from time-overlaps, not from dc1_adi
        "nac_algo": 1,               # NPI (Meek-Levine), not HST
        "hvib_update_method": 1,
        "decoherence_rates": MATRIX(nstates, nstates),   # recipes hard-code 2x2
        "ave_gaps": MATRIX(nstates, nstates),
    })

Do NOT override `state_tracking_algo`: the LD recipes set it to -1 deliberately,
because local diabatization consumes the time-overlap matrix directly.

For FSSH, also override the rescaling -- `fssh.py` hard-codes NAC-vector rescaling
(`hop_acceptance_algo:20`), which is meaningless when dc1_adi is zeros:

    dyn_general.update({"hop_acceptance_algo": 21, "momenta_rescaling_algo": 200})


FIVE THINGS THAT WILL BITE YOU (each one cost a debugging session)
==================================================================
1. `d1ham_adi` / `dc1_adi` must be `CMATRIXList`, not a Python list. Libra's C++
   extracts them into std::vector<CMATRIX>; a Python list ABORTS the process rather
   than raising, and only once C++ reaches in -- so calling compute_model from
   Python looks perfectly healthy.
2. `Cpp2Py(full_id)` before indexing. full_id is a Boost.Python-wrapped C++ vector;
   `full_id[-1]` is not reliably bounds-checked.
3. `model_params` must contain "model0" (and "model"). generic_recipe does
   `check_input(model_params, {}, ["model0"])`, which sys.exit()s on a miss -- in a
   Jupyter kernel that looks like a silent death. `make_model_params` handles this.
4. `__deepcopy__` returning self. generic_recipe deepcopies model_params on its
   first line; without this the dynamics runs against a clone and your `source`
   handle silently stays empty.
5. Gradients are dH/dR, not forces. PySCF's nuc_grad_method already returns dE/dR.
   Same sign. Do not negate.


LIMITS OF THE PHYSICS
=====================
* No analytic derivative couplings for TDDFT in mainline PySCF, so `dc1_adi` is
  zeros and time-overlaps carry the coupling. This is why local diabatization is the
  natural integrator here, not a workaround.
* State overlaps use the FACTORIZED (neglect-of-orbital-relaxation) CIS form. Exact
  in the dt->0 limit; drops orbital relaxation between geometries. The rigorous route
  is `libra_py.workflows.nbra.mapping.ovlp_arb` over a Slater-determinant basis.
* TDA by default. Full TDDFT can return imaginary excitation energies near S1/S0
  crossings, and linear response gets the S1/S0 conical intersection topology wrong
  regardless. Fine if the crossings that matter are between excited states.
"""

import os
import sys
import time
import copy

import numpy as np

# ---------------------------------------------------------------------------
# PySCF may live in a user-writable directory when the conda env is read-only.
# APPEND, never insert: this keeps the shared env winning every name collision, so
# a stray numpy here can never shadow the one liblibra_core was compiled against.
# Override with:  export LIBRA_EXTRAS=/path/to/dir
# ---------------------------------------------------------------------------
_EXTRAS = os.environ.get("LIBRA_EXTRAS", os.path.expanduser("~/libra-extras"))
if os.path.isdir(_EXTRAS) and _EXTRAS not in sys.path:
    sys.path.append(_EXTRAS)

try:
    from pyscf import gto, scf, dft, tdscf
except ImportError as exc:                                    # pragma: no cover
    raise ImportError(
        f"pyscf not importable (looked in {_EXTRAS}). Install with:\n"
        f"    python -m pip install --no-deps --target={_EXTRAS} pyscf\n"
        f"--no-deps matters: re-resolving numpy breaks liblibra_core's ABI."
    ) from exc

from liblibra_core import CMATRIX, MATRIX, CMATRIXList, Cpp2Py
from libra_py import data_conv

__all__ = ["PySCFSource", "compute_model", "make_model_params",
           "np2cmatrix", "q2R", "report_degeneracies", "tmp", "AMU"]

AMU = 1822.888486209          # electron masses per amu; Libra works in a.u. throughout


# ===========================================================================
#  Libra glue
# ===========================================================================

class tmp:
    """Plain attribute bag -- the same pattern Libra's own model functions use."""
    pass


def np2cmatrix(a):
    """numpy -> Libra CMATRIX, with a fallback if data_conv is laid out differently."""
    a = np.asarray(a, dtype=complex)
    try:
        return data_conv.nparray2CMATRIX(a)
    except Exception:
        m = CMATRIX(a.shape[0], a.shape[1])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                m.set(i, j, complex(a[i, j]))
        return m


def q2R(q, indx, natoms):
    """Libra's flat coordinate MATRIX -> (natoms, 3) numpy array, in Bohr."""
    return np.array([[q.get(3 * a + k, indx) for k in range(3)] for a in range(natoms)])


def make_model_params(source, **extra):
    """
    Build model_params with the keys generic_recipe requires.

    "model" / "model0" are REQUIRED even though compute_model ignores them:

        comn.check_input(model_params, {}, ["model0"])                # no default
        model_params1.update({"model": model_params["model0"], ...})  # unconditional

    Miss them and generic_recipe dies before any dynamics runs. The VALUE is
    irrelevant on our path -- model0 only feeds the diabatic->adiabatic priming
    branch, which ham_update_method:2 skips entirely.
    """
    p = {"source": source,
         "natoms": source.natoms,
         "nstates": source.nstates,
         "model": 0,
         "model0": 0}
    p.update(extra)
    return p


def report_degeneracies(mo_energy, tol=1e-5):
    """
    Diagnostic only -- nothing is 'fixed' here, and nothing needs to be.

    Adjacent MOs with equal energies span a degenerate subspace, inside which the SCF
    eigensolver returns an ARBITRARY unitary rotation. The MO-overlap diagonal between
    two steps is therefore meaningless for these orbitals; it is not a phase problem
    and a sign flip cannot address it. See PySCFSource.step for why it needs no fix.

    Reported because degenerate orbitals mean the TDA roots come in degenerate
    multiplets too -- do not let `nexc` split one across the band edge.
    """
    d = np.where(np.abs(np.diff(mo_energy)) < tol)[0]
    return [(int(i), int(i + 1)) for i in d]


# ===========================================================================
#  The electronic structure source
# ===========================================================================

class PySCFSource:
    """
    Per-geometry electronic structure + per-trajectory history for time-overlaps.

    States are ordered [S0, S1, ..., S_nexc]  ->  nstates = nexc + 1.
    Everything in and out is atomic units (Bohr, Hartree), which is what Libra wants.

    Unlike Libra's analytic model functions, this object is STATEFUL: a time-overlap
    needs the previous step's orbitals. With ntraj > 1 that history is keyed by
    trajectory index, or trajectories silently contaminate each other.

    This is the only PySCF-aware class in the interface. Swapping in a different
    backend (e.g. CasidaPy) means replacing this one class; compute_model and the
    dynamics layer never notice.
    """

    def __init__(self, symbols, basis, xc, nexc, charge=0, spin=0,
                 grid_level=3, conv_tol=1e-9, verbose_timing=True, timing_every=1):
        self.symbols = list(symbols)
        self.basis = basis
        self.xc = xc
        self.nexc = nexc
        self.nstates = nexc + 1
        self.charge = charge
        self.spin = spin
        self.grid_level = grid_level
        self.conv_tol = conv_tol
        self.natoms = len(symbols)

        self.hist = {}                 # traj index -> previous step. Keyed!
        self.ncalls = 0
        self.report_degen = True
        self.report_undet = True
        self.verbose_timing = verbose_timing
        self.timing_every = timing_every
        self._t_es_tot = 0.0
        self._t_grad_tot = 0.0

    def __deepcopy__(self, memo):
        """
        generic_recipe does `model_params = copy.deepcopy(_model_params)` on its very
        first line, which would clone this object -- and with it every cached PySCF
        Mole in self.hist. Two reasons that is wrong:

          1. The dynamics would run against a COPY, so `source.ncalls` and
             `source.hist` in the caller would silently stay empty.
          2. Cloning C-backed PySCF objects is pointless work at best.

        We WANT shared mutable state here, so refuse to be copied.
        """
        return self

    def reset(self):
        """Clear history and counters. Call before every independent run."""
        self.hist.clear()
        self.ncalls = 0
        self._t_es_tot = 0.0
        self._t_grad_tot = 0.0
        self.report_degen = True
        self.report_undet = True

    # ---------------------------------------------------------------- pyscf

    def _mol(self, R):
        mol = gto.Mole()
        mol.atom = [[s, tuple(float(x) for x in r)] for s, r in zip(self.symbols, R)]
        mol.basis = self.basis
        mol.unit = "Bohr"
        mol.charge = self.charge
        mol.spin = self.spin
        mol.verbose = 0
        mol.build()
        return mol

    def _electronic_structure(self, R, prev):
        mol = self._mol(R)
        mf = dft.RKS(mol, xc=self.xc)
        mf.grids.level = self.grid_level
        mf.conv_tol = self.conv_tol

        # Project the previous density in as the SCF guess. Same idea as keeping the
        # QE density resident between steps: at MD timesteps this is a large speedup,
        # and it is free because the object persists.
        dm0 = None
        if prev is not None:
            try:
                dm0 = scf.addons.project_dm_nr2nr(prev["mol"], prev["dm"], mol)
            except Exception:
                dm0 = None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            print("  WARNING: SCF not converged")

        td = tdscf.TDA(mf)
        td.nstates = self.nexc
        td.kernel()
        if not all(np.atleast_1d(td.converged)):
            print("  WARNING: TDA not converged for all roots")
        return mol, mf, td

    # ------------------------------------------------------------ overlaps

    def _amplitudes(self, td):
        """
        TDA amplitudes as a list of (nocc, nvir) arrays, each normalized to 1.

        Normalized numerically rather than by assuming PySCF's spin-adaptation
        convention -- that convention has changed between versions and is not worth
        betting the dynamics on.
        """
        X = []
        for n in range(self.nexc):
            x = np.asarray(td.xy[n][0])
            X.append(x / np.linalg.norm(x))
        return X

    def _state_overlaps(self, cur, prev):
        """
        <Psi_I(t)|Psi_J(t+dt)> in the factorized CIS form.

            S_00 = det S^oo
            S_0J = sum_ia X^J_ia S^ov_ia
            S_I0 = sum_ia X^I_ia S^vo_ai
            S_IJ = sum_{ia,jb} X^I_ia X^J_jb S^oo_ij S^vv_ab

        As dt -> 0 the blocks go to identity and S_IJ -> delta_IJ. Differentiating
        recovers the standard CIS time-derivative coupling.

        gto.intor_cross is the load-bearing call: it evaluates AO integrals with the
        bra basis on one geometry and the ket basis on another. One line in a Gaussian
        code; a genuine project in a plane-wave code.
        """
        S_ao = gto.intor_cross("int1e_ovlp", prev["mol"], cur["mol"])
        S_mo = prev["mo"].T @ S_ao @ cur["mo"]

        o, v = self.occ, self.vir
        S_oo = S_mo[np.ix_(o, o)]
        S_vv = S_mo[np.ix_(v, v)]
        S_ov = S_mo[np.ix_(o, v)]
        S_vo = S_mo[np.ix_(v, o)]

        n = self.nstates
        St = np.zeros((n, n))
        St[0, 0] = np.linalg.det(S_oo)
        for J in range(1, n):
            St[0, J] = np.sum(cur["X"][J - 1] * S_ov)
            St[J, 0] = np.sum(prev["X"][J - 1] * S_vo.T)
        for I in range(1, n):
            for J in range(1, n):
                St[I, J] = np.einsum("ia,jb,ij,ab->",
                                     prev["X"][I - 1], cur["X"][J - 1], S_oo, S_vv)
        return St

    # ---------------------------------------------------------------- main

    def step(self, R, traj=0, want_grads=True, phase_tol=0.5):
        """
        One geometry -> (energies, gradients, time-overlap).

        Returns
        -------
        energies : (nstates,)          Hartree
        grads    : (nstates, natoms, 3)  dE/dR, Hartree/Bohr. None if want_grads=False
        St       : (nstates, nstates)  <Psi_I(t)|Psi_J(t+dt)>; identity on first call
        """
        _t0 = time.time()
        prev = self.hist.get(traj, None)
        mol, mf, td = self._electronic_structure(R, prev)
        self.ncalls += 1
        _t_es = time.time() - _t0

        nocc = int((mf.mo_occ > 0).sum())
        nmo = mf.mo_coeff.shape[1]
        self.occ = np.arange(nocc)
        self.vir = np.arange(nocc, nmo)

        # NO MO phase or sign fixing -- deliberately.
        #
        # mo_coeff and td.xy come from the same diagonalization and are mutually
        # consistent. The factorized overlap is invariant under an orthogonal rotation
        # within the occ or vir block: with C' = CU and X' = XU,
        #
        #   sum_b X'_jb S'_vv,ab = sum_bcd X_jc U_cb S0_vv,ad U_db
        #                        = sum_cd X_jc S0_vv,ad delta_cd    (since U U^T = 1)
        #                        = sum_c X_jc S0_vv,ac
        #
        # so both the sign ambiguity (diagonal U) and the degenerate-subspace rotation
        # cancel in the contraction. Transforming C WITHOUT applying the same transform
        # to td.xy desynchronizes them and corrupts the overlaps -- a far worse bug
        # than the one it appears to fix.
        C = mf.mo_coeff

        if self.report_degen and prev is None:
            degen = report_degeneracies(mf.mo_energy)
            if degen:
                print(f"  degenerate MO pairs (expected for symmetric molecules): {degen}")
            self.report_degen = False

        energies = np.concatenate(([mf.e_tot], mf.e_tot + td.e))

        grads = None
        _t_g0 = time.time()
        if want_grads:
            # dH/dR, NOT forces. nuc_grad_method already returns dE/dR. Do not negate.
            #
            # COST: Ehrenfest (force_method:2) needs every state's gradient for the
            # mean-field force. Each TDA gradient solves a Z-vector equation and costs
            # roughly what the TDA itself did. This loop is usually the bulk of the
            # wall time. FSSH does not help -- Libra picks the active state out of
            # d1ham_adi, so we must supply all of them anyway.
            g = [mf.nuc_grad_method().kernel()]
            tdg = td.nuc_grad_method()
            for n in range(self.nexc):
                g.append(tdg.kernel(state=n + 1))
            grads = np.asarray(g)
        _t_grad = time.time() - _t_g0

        cur = {"mol": mol, "mo": C, "dm": mf.make_rdm1(),
               "X": self._amplitudes(td), "E": energies}

        if prev is None:
            St = np.eye(self.nstates)
        else:
            St = self._state_overlaps(cur, prev)

            # State phase fix -- but ONLY where the sign is actually determined.
            #
            # Within a degenerate multiplet the TDA solver returns an arbitrary
            # ROTATION of the degenerate states, so that block of St is a rotation
            # matrix and its diagonal can be ~0. np.sign() of noise would flip a column
            # at random and inject a discontinuity into the very matrix the LD
            # integrator relies on. Leave those alone: LD consumes St directly and
            # handles orthogonal mixing correctly -- which is exactly why the recipes
            # set state_tracking_algo:-1 for LD.
            d = np.diag(St)
            sgn = np.where(np.abs(d) > phase_tol, np.sign(d), 1.0)
            sgn[sgn == 0.0] = 1.0
            St = St * sgn[None, :]
            cur["X"] = [x * s for x, s in zip(cur["X"], sgn[1:])]

            undet = np.where(np.abs(d) <= phase_tol)[0]
            if len(undet) and self.report_undet:
                print(f"  note: St diagonal ~0 at states {undet} -- phase undetermined "
                      f"(expected for degenerate multiplets); left to LD")
                self.report_undet = False

        self.hist[traj] = cur

        if self.verbose_timing:
            self._t_es_tot += _t_es
            self._t_grad_tot += _t_grad
            if self.ncalls % self.timing_every == 0:
                tot = self._t_es_tot + self._t_grad_tot
                print(f"    [call {self.ncalls:5d}] scf+tda {_t_es:5.1f}s  "
                      f"grads {_t_grad:5.1f}s  |  cumulative {tot/60:6.1f} min  "
                      f"({tot/self.ncalls:4.1f} s/call, "
                      f"{100*self._t_grad_tot/max(tot, 1e-9):3.0f}% in gradients)",
                      flush=True)
        return energies, grads, St

    def masses_au(self, R):
        """
        Length-3N mass list in a.u., in Libra's flat dof ordering.

        Needs a real geometry -- building a Mole with every atom at the origin makes
        PySCF choke.
        """
        m = self._mol(R).atom_mass_list()
        return [float(mi) * AMU for mi in m for _ in range(3)]


# ===========================================================================
#  The model function
# ===========================================================================

def compute_model(q, params, full_id):
    """
    Libra's model-function contract, backed by PySCF.

    Same slot as Holstein.Holstein2, but returns ADIABATIC quantities -- there is no
    diabatic basis here, so Libra transforms nothing (ham_transform_method:0).

    params must carry: source, natoms, nstates, model, model0 (see make_model_params).
    """
    # Cpp2Py first. full_id is a Boost.Python-wrapped C++ vector; indexing it directly
    # with -1 is not reliably bounds-checked and can read off the front of the buffer
    # -> abort, not an exception. This is the idiom Libra's own models use.
    Id = Cpp2Py(full_id)
    indx = Id[-1]

    src = params["source"]
    natoms = params["natoms"]
    nst = params["nstates"]

    R = q2R(q, indx, natoms)
    energies, grads, St = src.step(R, traj=indx)

    obj = tmp()
    obj.ham_adi = np2cmatrix(np.diag(energies))
    obj.time_overlap_adi = np2cmatrix(St)

    # CMATRIXList, NOT a Python list.
    #
    # The single most important line here. Libra's C++ extracts these into a
    # std::vector<CMATRIX>; a plain Python list ABORTS the process rather than raising,
    # and only once C++ reaches in -- so calling compute_model from Python looks
    # perfectly healthy. Holstein2 does exactly this: CMATRIXList() then .append().
    d1ham = CMATRIXList()
    for a in range(natoms):
        for k in range(3):
            d1ham.append(np2cmatrix(np.diag(grads[:, a, k])))
    obj.d1ham_adi = d1ham

    # No analytic derivative couplings. Zeros, not fabricated values -- which is why
    # NAC-vector-based rescaling must be overridden rather than left to the recipe.
    dc1 = CMATRIXList()
    for _ in range(3 * natoms):
        dc1.append(np2cmatrix(np.zeros((nst, nst))))
    obj.dc1_adi = dc1

    return obj


# ===========================================================================
#  Self-test:  python pyscf_libra.py
# ===========================================================================

if __name__ == "__main__":
    from liblibra_core import Py2Cpp_int

    print("pyscf_libra self-test\n" + "=" * 60)
    symbols = ["C", "O"]
    R0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.132]])

    src = PySCFSource(symbols, basis="6-31G*", xc="pbe0", nexc=3, timing_every=10**9)
    mp = make_model_params(src)
    print("model_params keys:", sorted(mp))
    assert "model0" in mp

    t0 = time.time()
    E, G, St = src.step(R0)
    print(f"\nstep 1: {time.time()-t0:.1f} s")
    print("  energies (Ha):   ", np.round(E, 6))
    print("  excitations (eV):", np.round((E[1:] - E[0]) * 27.211386, 3))
    print("  St == I:", np.allclose(St, np.eye(src.nstates)))

    E1, G1, St1 = src.step(R0 + np.array([[0, 0, 0], [0, 0, 0.02]]))
    print("\nstep 2 (displaced):")
    print("  St diagonal:", np.round(np.diag(St1), 4))
    print("  max |off-diag|:", np.round(np.abs(St1 - np.diag(np.diag(St1))).max(), 4))

    # The check that matters: the C++ type contract.
    src.reset()
    q = MATRIX(3 * len(symbols), 1)
    for i, x in enumerate(R0.flatten()):
        q.set(i, 0, float(x))
    obj = compute_model(q, mp, Py2Cpp_int([0, 0]))
    for name in ["d1ham_adi", "dc1_adi"]:
        t = type(getattr(obj, name)).__name__
        assert t == "CMATRIXList", f"obj.{name} is {t}, must be CMATRIXList"
    print("\ncompute_model: d1ham_adi/dc1_adi are CMATRIXList -- OK")
    print(f"  d1ham_adi entries: {len(obj.d1ham_adi)} (expect {3*len(symbols)})")

    assert copy.deepcopy(src) is src, "__deepcopy__ not taking"
    print("__deepcopy__: returns self -- OK")
    print("\nall checks passed")
