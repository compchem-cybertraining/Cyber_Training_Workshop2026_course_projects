# *********************************************************************************
# P1-6: CIS/TDA exciton Hamiltonian + gradients, ported into Libra's on-the-fly
# ham_update_method=2 ("adiabatic") compute_model contract.
#
# TEMPLATE: libra_py.packages.dftbplus.methods.dftb_compute_adi (found in the
# workshop's 4_dftbplus_methods/3_workflow/tutorial.ipynb, cell 37) -- the real
# production function DFTB+ many-electron NA-MD workflows use. Key structural
# findings this module mirrors exactly:
#
#   * obj.ham_adi is diagonal = state energies directly -- Libra does NOT
#     diagonalize anything on its side for this contract (unlike Step 2's
#     ham_dia/AgChain.py, which handed Libra a diabatic matrix to diagonalize).
#   * EVERY state's diagonal entry gets the SAME ground-state total energy added
#     on top of its own (relative) energy -- dftb_compute_adi does this via
#     `obj.ham_adi.add(i, i, e0)` after setting each state's excitation energy.
#     States differ only by what's added on top of that shared baseline. This is
#     why this module adds ground-state electronic + repulsive energy back in --
#     without it the PES would be missing the dominant lattice-restoring forces.
#   * obj.d1ham_adi is diagonal-only per DOF: d1ham_adi[k].set(i,i, dE_i/dR_k).
#     (Note dftb_compute_adi writes `-forces[...]`; force = -dE/dR, so
#     -forces = +dE/dR -- i.e. d1ham_adi stores the energy GRADIENT, not the
#     force. Same convention used here.)
#   * obj.dc1_adi legitimately left all-zero when there's no NAC data -- the
#     reference code does exactly this when NACV.DAT doesn't exist. We do the
#     same: with a single fixed 2-state window (ground / ground+exciton) built
#     fresh from instantaneous eigenvectors every call, there's no coupling
#     pathway available in this simplified model, so this is a legitimate
#     (documented) approximation, not a shortcut -- see README "Open
#     Questions" for the state-tracking caveat this implies.
#   * obj.hvib_adi = obj.ham_adi when dc1_adi/time_overlap contribute no
#     off-diagonal terms (reference builds hvib's off-diagonals purely from
#     the time-overlap-derived NAC estimate, which is zero here for the same
#     reason as dc1_adi).
#   * obj.basis_transform, obj.time_overlap_adi: identity. CAVEAT (documented,
#     not yet resolved): true state-tracking across MD steps needs a real
#     time-overlap between consecutive-step eigenvectors to catch state
#     reordering/degeneracy crossing; this module assumes state identity is
#     stable step-to-step (valid near the dimerized equilibrium geometry where
#     HOMO/LUMO stay non-degenerate band-edge states -- see cis_gradient.py's
#     mo_response docstring). Revisit if P1-7 dynamics shows state-order flips.
#
# PHYSICS REUSED AS-IS (no rederivation): cis_gradient.py's kernel_with_gradient
# and mo_response (both geometry-agnostic already). NEW in this module: a
# general-geometry H0_and_dH0 (P1-1/P1-5's build_H0/H0_with_gradient only build
# the fixed equilibrium-dimerized geometry; MD moves the atoms off of it), plus
# the repulsive potential energy/gradient (never needed before now -- Steps 1-5
# only ever needed the electronic Hamiltonian) and the ground-state total energy/
# gradient assembly described above.
#
# VALIDATION (2026-07-09): see validate() below. Finite-difference check of the
# FULLY ASSEMBLED ham_adi diagonal (ground-state total energy+repulsion AND
# ground+exciton total) at a randomly-JITTERED (not equilibrium) 32-pair
# geometry -- equilibrium geometries have near-zero net force by construction,
# which is a weak test; jittering ensures the check exercises genuinely nonzero
# gradients in every term.
# *********************************************************************************

import sys
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform in ("linux", "linux2"):
    from liblibra_core import *
import util.libutil as comn

import numpy as np
from cis_exciton import ANGSTROM, ELECTRONVOLT
from cis_gradient import kernel_with_gradient, mo_response, contract3
from cis_time_overlap import ground_exciton_time_overlap, multi_state_time_overlap   # P1-17/P1-19,
                                                             # only used when params["enable_coupling"]=True


class tmp:
    pass


def H0_and_dH0(rion, boxl, hop=-0.668653 * ELECTRONVOLT,
               hopslope=1.150209 * ELECTRONVOLT / ANGSTROM,
               req=3.989152 * ANGSTROM):
    """Same construction as AgChain.py's compute_ag_chain / cis_exciton.build_H0,
    but for an ARBITRARY current geometry `rion` (not just the fixed equilibrium-
    dimerized generator) -- needed because MD moves the atoms. Returns (H0, dH0)
    as plain numpy arrays, dH0 shape (n, n, n) = dH0[k] = dH0/dR_k.

    Defaults are the ab-initio-corrected (production) hopping parameters from
    Objective 1's pySCF re-parametrization (esteban-gadea/1.Ag_chain_recalibration_pySCF/
    report/objective1_report.md, Section 6): beta_eq=-0.668653 eV (unchanged), the
    rescaled slope A=1.150209 eV/Ang, and the re-fit r_eq=3.989152 Ang -- NOT the
    original PBE-fit values (hop=-0.0245725447, hopslope=0.007215487659, req=4.922388,
    all in Ha/Bohr) that every P1-1 through P1-19 development/validation run in this
    file's history used. Those legacy values are preserved as-is in AgChain.py's and
    cis_exciton.py's own get_default_params/build_H0 defaults (kept there deliberately,
    since AgChain.validate() checks against a reference eigenvalue computed with them)
    -- do not "fix" those to match production; they validate a different, older thing.
    """
    n = len(rion)
    H0 = np.zeros((n, n))
    dH0 = np.zeros((n, n, n))

    def bond(i, j, dr):
        h = hop + hopslope * (dr - req)
        H0[i, j] = h
        H0[j, i] = h
        dH0[i, i, j] = -hopslope
        dH0[i, j, i] = -hopslope
        dH0[j, i, j] = hopslope
        dH0[j, j, i] = hopslope

    for i in range(n - 1):
        bond(i, i + 1, rion[i + 1] - rion[i])
    bond(n - 1, 0, (rion[0] + boxl) - rion[n - 1])
    return H0, dH0


def repulsion_energy_and_gradient(rion, boxl, hop, pref, p, req):
    """V(r) = -hop*pref*(req/r)^p per bond, matching
    Ag_chains_parametrization.pdf Eq. 3 / td_code/TLS_module.jl's
    construct_potential exactly (pref, p defaults from IOmodule.jl /
    example/inp.in: pref=0.231122, p=15)."""
    n = len(rion)
    E = 0.0
    dE = np.zeros(n)

    def bond(i, j, dr):
        nonlocal E
        Vr = -hop * pref * (req / dr) ** p
        dVr_dr = hop * pref * p * req ** p / dr ** (p + 1)
        E += Vr
        dE[i] += -dVr_dr
        dE[j] += dVr_dr

    for i in range(n - 1):
        bond(i, i + 1, rion[i + 1] - rion[i])
    bond(n - 1, 0, (rion[0] + boxl) - rion[n - 1])
    return E, dE


def ground_state_energy_and_gradient(H0, dH0, rion, boxl, eps, C, nocc,
                                      hop, pref, p, req):
    """E_ground_total = 2*sum_occ(eps_i) + V_repulsion(R), plus its gradient
    (Hellmann-Feynman for the electronic piece, analytic for the repulsive
    piece). This is the shared baseline dftb_compute_adi adds to EVERY state's
    ham_adi diagonal (its `e0`/mermin_energy + `obj.ham_adi.add(i,i,e0)`).

    PERFORMANCE (2026-07-10, P1-9): dE_elec vectorized -- the original had a
    Python loop over n_dof, each iteration summing nocc separate
    C[:,i] @ dH0[k] @ C[:,i] matrix-vector-vector products (O(n_dof*nocc)
    Python-level work). Rewritten using the occupied-space density matrix
    P = C_occ @ C_occ.T (n,n): sum_i C[:,i] . dH0[k] . C[:,i] = trace(dH0[k] @ P),
    computed for ALL k at once as a single matrix-vector product
    dH0.reshape(n_dof,-1) @ P.T.reshape(-1) (BLAS gemv). Confirmed bit-for-bit
    equivalent to the original (max difference ~7e-18, floating-point noise).
    Measured speedup: ~77x (nchain=32) to ~94x (nchain=16)."""
    n_dof, n, _ = dH0.shape
    E_elec = 2.0 * np.sum(eps[:nocc])
    Cocc = C[:, :nocc]
    Pocc = Cocc @ Cocc.T
    dE_elec = 2.0 * (dH0.reshape(n_dof, -1) @ Pocc.T.reshape(-1))

    E_rep, dE_rep = repulsion_energy_and_gradient(rion, boxl, hop, pref, p, req)
    return E_elec + E_rep, dE_elec + dE_rep


def exciton_energy_and_gradient(rion, boxl, H0, dH0, eps, C, homo, lumo,
                                 fxcalpha, fxcgamma, hartreeu):
    """CIS/TDA exciton energy (relative to the bare HOMO->LUMO gap baseline)
    and its gradient, for the single dominant configuration, at an ARBITRARY
    geometry. Same physics as cis_gradient.exciton_gradient_1config, but takes
    H0/dH0/eps/C/rion/boxl directly instead of rebuilding them from
    (nchain, dimer1, lattice_ang) -- exciton_gradient_1config only ever builds
    the fixed equilibrium-dimerized geometry, which MD moves away from."""
    K, dK = kernel_with_gradient(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exch, dK_exch = kernel_with_gradient(rion, boxl, 0.0, fxcgamma, hartreeu)

    dC_homo = mo_response(C, eps, dH0, homo)
    dC_lumo = mo_response(C, eps, dH0, lumo)

    def Dvec(p_, q_):
        return C[:, p_] * C[:, q_]

    d_ia, d_ii, d_aa = Dvec(homo, lumo), Dvec(homo, homo), Dvec(lumo, lumo)
    direct = d_ia @ K @ d_ia
    exch = d_ii @ K_exch @ d_aa
    E = (eps[lumo] - eps[homo]) + 2 * direct - exch

    n_dof = dH0.shape[0]
    dE = np.zeros(n_dof)
    for k in range(n_dof):
        dDia_k = dC_homo[k] * C[:, lumo] + C[:, homo] * dC_lumo[k]
        dDii_k = 2 * dC_homo[k] * C[:, homo]
        dDaa_k = 2 * dC_lumo[k] * C[:, lumo]

        d_direct = dDia_k @ K @ d_ia + d_ia @ dK[k] @ d_ia + d_ia @ K @ dDia_k
        d_exch = dDii_k @ K_exch @ d_aa + d_ii @ dK_exch[k] @ d_aa + d_ii @ K_exch @ dDaa_k
        d_deps = C[:, lumo] @ dH0[k] @ C[:, lumo] - C[:, homo] @ dH0[k] @ C[:, homo]
        dE[k] = d_deps + 2 * d_direct - d_exch

    return E, dE


def cis_windowed_energy_and_gradient(rion, boxl, H0, dH0, eps, C, homo, lumo,
                                      fxcalpha, fxcgamma, hartreeu, n_near,
                                      state_index=0, degeneracy_tol=1e-8):
    """
    P1-8b: general n_near CIS/TDA exciton energy + gradient at an ARBITRARY
    geometry -- the arbitrary-geometry generalization of cis_gradient.py's
    cis_gradient_windowed (P1-8a), exactly the same relationship
    exciton_energy_and_gradient above already has to exciton_gradient_1config.
    Reuses the degenerate-safe mo_response (P1-8a) unchanged -- that's what
    makes n_near>1 possible: for n_near=1 this is byte-for-byte identical to
    exciton_energy_and_gradient (regression-checked in the sandbox dev script,
    exact 0.0 difference), but for n_near>1 the occ/virt window pulls in +-k
    degenerate pairs (e.g. HOMO-1/HOMO-2 for n_near=3) that the old
    single-configuration function could never have handled.

    THIS is what P1-7 was actually missing: with n_near=1, the exciton is a
    single delocalized Bloch-like configuration with no way to spatially
    localize (confirmed by P1-7's uniform breathing-mode result). With
    n_near>1, the returned state (state_index=0, the lowest CIS eigenstate) is
    a genuine superposition of several near-gap configurations -- something
    that CAN be spatially localized, the actual object needed to test whether
    the exciton-lattice coupling produces self-trapping.

    Args:
        n_near (int): occ/virt window half-width (use ODD values -- see
            cis_gradient.py's cis_gradient_windowed docstring for why).
        state_index (int): which CIS eigenstate to return (0 = lowest exciton
            state -- this is what cis_compute_adi uses for its state-1 PES).

    Returns:
        (E_n, dE_n): E_n is the state_index-th CIS eigenvalue (Ha, relative to
        the bare gap baseline, same convention as exciton_energy_and_gradient),
        dE_n[k] is its gradient w.r.t. site DOF k (Ha/Bohr).

    VALIDATED (2026-07-09, numpy sandbox): finite-difference agreement ~1e-7
    at the EXACT equilibrium (degenerate) geometry and ~1e-9-1e-10 at a
    jittered (non-degenerate) geometry, n_near=3 (pulls in the HOMO-1/HOMO-2
    and LUMO+1/LUMO+2 degenerate pairs). n_near=1 regression-checked exactly
    equal to exciton_energy_and_gradient. Real-kernel confirmed 2026-07-09.

    PERFORMANCE (2026-07-10, P1-9): this is the actual per-MD-step hot path
    (called once every Ehrenfest timestep), and the original implementation
    made it "painfully slow" (Esteban's words) -- ~72-326 ms/step depending
    on nchain, i.e. 25-108 minutes for a 20000-step run. Profiling
    (cProfile) identified three redundant/unvectorized pieces, all fixed
    here with NO change in the underlying math -- only correctness-preserving
    rewrites (validated bit-for-bit, ~1e-17-1e-18 max difference, against the
    original at both equilibrium and jittered geometries, nchain=16 and 32):

      1. REDUNDANT Dvec/dDvec RECOMPUTATION: the original recomputed
         Dvec(i,a)/dDvec(i,a,k) from scratch for every (I,J) pair and every k
         -- for n_near=3 (m=9 configs, 81 (I,J) pairs) that's thousands of
         redundant recomputations of the same ~30 unique (p,q) products per
         step. Fixed with a `get_D` memo cache keyed on (p_,q_), computed
         once and reused.
      2. Hm/dHk MATRIX SYMMETRY: Hm and dHk are symmetric (CIS Hamiltonian
         and its derivative), but the original looped over ALL (I,J) pairs
         including both (I,J) and (J,I). Fixed by looping over the upper
         triangle only (J >= I) and mirroring, with the off-diagonal
         contribution to Psi @ dHk @ Psi weighted by 2 (since
         Psi[I]*dHk[I,J]*Psi[J] + Psi[J]*dHk[J,I]*Psi[I] = 2*Psi[I]*Psi[J]*dHk[I,J]
         when dHk is symmetric) -- avoids computing dHk entirely and avoids
         the Psi @ dHk @ Psi matrix product altogether.
      3. UNVECTORIZED k-LOOP: the original had an explicit `for k in
         range(n_dof)` Python loop, rebuilding an (m,m) matrix from scratch
         every iteration. Fixed by using `contract3` (cis_gradient.py) to
         get each (I,J) pair's full (n_dof,)-shaped gradient contribution in
         one BLAS-backed call, accumulated directly into dE_n (no per-k
         matrix ever built).

      Combined with kernel_with_gradient's and mo_response's own
      vectorization (see their docstrings), measured end-to-end speedup:
      ~20-25x at nchain=16 (94 -> 3.7 ms/step; 31 -> 1.2 min for 20000
      steps) and ~12-18x at nchain=32 (326 -> 17.8 ms/step; 108 -> 6.0 min
      for 20000 steps). See README's P1-9 section for the full profiling
      story and all measured numbers.
    """
    occ_idx = list(range(homo - n_near + 1, homo + 1))
    virt_idx = list(range(lumo, lumo + n_near))
    configs = [(i, a) for i in occ_idx for a in virt_idx]
    m = len(configs)
    n_dof = dH0.shape[0]

    K, dK = kernel_with_gradient(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exch, dK_exch = kernel_with_gradient(rion, boxl, 0.0, fxcgamma, hartreeu)

    used = sorted(set(occ_idx) | set(virt_idx))
    dC = {p_: mo_response(C, eps, dH0, p_, degeneracy_tol) for p_ in used}

    Dv, dDv = {}, {}

    def get_D(p_, q_):
        key = (p_, q_)
        if key not in Dv:
            Dv[key] = C[:, p_] * C[:, q_]
            dDv[key] = dC[p_] * C[:, q_][None, :] + C[:, p_][None, :] * dC[q_]
        return Dv[key], dDv[key]

    Hm = np.zeros((m, m))
    for I, (i, a) in enumerate(configs):
        for J in range(I, m):
            j, b = configs[J]
            d_ia, _ = get_D(i, a)
            d_jb, _ = get_D(j, b)
            d_ij, _ = get_D(i, j)
            d_ab, _ = get_D(a, b)
            direct = d_ia @ K @ d_jb
            exch = d_ij @ K_exch @ d_ab
            val = 2.0 * direct - exch
            if i == j and a == b:
                val += (eps[a] - eps[i])
            Hm[I, J] = val
            Hm[J, I] = val
    evals, evecs = np.linalg.eigh(Hm)
    Psi = evecs[:, state_index]
    E_n = evals[state_index]

    dE_n = np.zeros(n_dof)
    for I, (i, a) in enumerate(configs):
        for J in range(I, m):
            j, b = configs[J]
            w = Psi[I] * Psi[J] * (1.0 if I == J else 2.0)
            d_ia, dd_ia = get_D(i, a)
            d_jb, dd_jb = get_D(j, b)
            d_ij, dd_ij = get_D(i, j)
            d_ab, dd_ab = get_D(a, b)
            d_direct = (dd_ia @ K) @ d_jb + contract3(d_ia, dK, d_jb) + (d_ia @ K) @ dd_jb.T
            d_exch = (dd_ij @ K_exch) @ d_ab + contract3(d_ij, dK_exch, d_ab) + (d_ij @ K_exch) @ dd_ab.T
            val = 2.0 * d_direct - d_exch
            if i == j and a == b:
                val = val + contract3(C[:, a], dH0, C[:, a]) - contract3(C[:, i], dH0, C[:, i])
            dE_n += w * val

    return E_n, dE_n


def cis_compute_adi(q, params, full_id):
    """
    Libra ham_update_method=2 compute_model contract: 2-state adiabatic PES
    (state 0 = ground, state 1 = ground+exciton) for the Ag-chain ring, built
    fresh from the instantaneous geometry every call -- no internal Libra
    diagonalization, matching libra_py.packages.dftbplus.methods.dftb_compute_adi's
    contract (see module docstring).

    Args:
        q (MATRIX(ndof, ntraj)): current site positions, Bohr. ndof = 2*nchain.
        params (dict): critical keys nchain, hop, hopslope, req, boxl, pref, p,
            fxcalpha, fxcgamma, hartreeu, n_near (see get_default_params).
            n_near added in P1-8b: window half-width for the CIS configuration
            space feeding state 1's energy/gradient. n_near=1 (the original
            P1-6 behavior) is a single delocalized Bloch-like configuration
            that cannot spatially localize (see P1-7's result); n_near=3 (the
            new default) mixes in the HOMO-1/HOMO-2 and LUMO+1/LUMO+2
            degenerate pairs via cis_windowed_energy_and_gradient, giving a
            genuinely localizable exciton state -- this is the actual point
            of P1-8.

            P1-17 (2026-07-11): optional `enable_coupling` (bool, default
            False -- OFF unless explicitly requested, for full backward
            compatibility with every already-completed/in-flight single-PES
            run). When True, requires `dt` (nuclear timestep, a.u.) also in
            params, and:
              - populates obj.time_overlap_adi with the REAL ground-exciton
                time-overlap S_adi(t-dt, t) (cis_time_overlap.py's validated
                closed-form formula), instead of the identity fallback,
              - populates obj.hvib_adi's off-diagonal via the standard
                Hammes-Schiffer-Tully formula d_01=(S01-S10)/(2dt), matching
                recipes/fssh2.py's nac_algo=0 / recipes/ehrenfest_onthefly.py's
                already-set nac_update_method=2 -- this is the SAME pathway
                4_dftbplus_methods/3_workflow/tutorial.ipynb's dftb_compute_adi
                uses, confirmed by direct comparison against that reference
                implementation before wiring this in.
              - caches the previous step's MO coefficients + CIS eigenvector
                in `params` (keyed by trajectory index, same pattern as
                dftb_compute_adi's MO_prev/data_prev/is_first_time) since a
                time-overlap needs BOTH steps' wavefunctions -- this makes
                cis_compute_adi implicitly STATEFUL across calls when
                enable_coupling=True (was fully stateless before).
            dc1_adi is left at all-zero either way (see its own comment below
            -- legitimate given hop_acceptance_algo/momenta_rescaling_algo
            don't use it in this recipe, matching dftb_compute_adi's usage
            where dc1_adi only matters for NACV-based momentum rescaling).
            Validated in a real dynamics run (P1-18): coupling mechanism
            confirmed correct (no crash, real nonzero time-overlaps computed),
            but ground<->state0 coupling turned out to be suppressed by an
            exact symmetry-driven cancellation specific to state_index=0 (see
            README) -- motivated P1-19 below.

            P1-19 (2026-07-11): optional `n_exciton_states` (int, default 1 --
            preserves the original 2-state ground+state0 model EXACTLY). Set to
            2 to track a THIRD state (ground, exciton state_index=0, exciton
            state_index=1), with enable_coupling=True now populating the FULL
            3x3 time-overlap/hvib_adi (all three edges: ground<->state1,
            ground<->state2, state1<->state2), via
            cis_time_overlap.multi_state_time_overlap. Motivated by a 2026-07-11
            finding: direct ground<->state0 coupling is symmetry-suppressed
            (~1e-16, noise floor, persistent through a full 200fs trajectory),
            but state0<->state1 (within-manifold) coupling is NOT -- it grows to
            ~-2.4e-5 a.u. in the deeply-self-trapped regime, orders of magnitude
            larger than either state's direct ground coupling. The 2-state model
            structurally cannot represent this channel at all.
        full_id: trajectory identifier.

    Returns:
        PyObject obj with obj.ham_adi, obj.d1ham_adi, obj.dc1_adi, obj.hvib_adi,
        obj.basis_transform, obj.time_overlap_adi -- see module docstring for
        exactly what each holds and why.
    """
    critical_params = ["nchain", "hop", "hopslope", "req", "boxl", "pref", "p",
                        "fxcalpha", "fxcgamma", "hartreeu", "n_near"]
    default_params = {}
    comn.check_input(params, default_params, critical_params)
    enable_coupling = params.get("enable_coupling", False)

    nchain = params["nchain"]
    hop = params["hop"]
    hopslope = params["hopslope"]
    req = params["req"]
    boxl = params["boxl"]
    pref = params["pref"]
    p = params["p"]
    fxcalpha = params["fxcalpha"]
    fxcgamma = params["fxcgamma"]
    hartreeu = params["hartreeu"]
    n_near = params["n_near"]

    n = 2 * nchain
    n_exciton_states = params.get("n_exciton_states", 1)   # P1-19: default 1 preserves the
                                                             # original 2-state (ground+state0)
                                                             # behavior EXACTLY. Set to 2 for the
                                                             # 3-state (ground, state0, state1)
                                                             # model, motivated by the 2026-07-11
                                                             # finding that state0<->state1
                                                             # within-manifold coupling is orders
                                                             # of magnitude larger than either
                                                             # state's direct ground coupling in
                                                             # the deeply-self-trapped regime.
    nstates = 1 + n_exciton_states

    Id = Cpp2Py(full_id)
    traj = Id[-1]
    rion = np.array([q.get(i, traj) for i in range(n)])

    H0, dH0 = H0_and_dH0(rion, boxl, hop, hopslope, req)
    eps, C = np.linalg.eigh(H0)
    nocc = n // 2
    homo, lumo = nocc - 1, nocc

    E_ground, dE_ground = ground_state_energy_and_gradient(
        H0, dH0, rion, boxl, eps, C, nocc, hop, pref, p, req)

    state_E = [E_ground]
    state_dE = [dE_ground]   # state_dE[i][k] = dE_i/dR_k
    for state_index in range(n_exciton_states):
        E_exc, dE_exc = cis_windowed_energy_and_gradient(
            rion, boxl, H0, dH0, eps, C, homo, lumo, fxcalpha, fxcgamma, hartreeu,
            n_near, state_index=state_index)
        state_E.append(E_ground + E_exc)
        state_dE.append(dE_ground + dE_exc)

    obj = tmp()
    obj.ham_adi = CMATRIX(nstates, nstates)
    obj.hvib_adi = CMATRIX(nstates, nstates)
    obj.basis_transform = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)
    for i in range(nstates):
        e = state_E[i] * (1.0 + 0.0j)
        obj.ham_adi.set(i, i, e)
        obj.hvib_adi.set(i, i, e)
        obj.basis_transform.set(i, i, 1.0 + 0.0j)
        obj.time_overlap_adi.set(i, i, 1.0 + 0.0j)   # off-diagonal overwritten below if enable_coupling

    obj.d1ham_adi = CMATRIXList()
    for k in range(n):
        m = CMATRIX(nstates, nstates)
        for i in range(nstates):
            m.set(i, i, state_dE[i][k] * (1.0 + 0.0j))
        obj.d1ham_adi.append(m)

    obj.dc1_adi = CMATRIXList()
    for k in range(n):
        obj.dc1_adi.append(CMATRIX(nstates, nstates))
        # left at zero even with enable_coupling=True: this recipe's
        # hop_acceptance_algo/momenta_rescaling_algo don't consume dc1_adi
        # (matching dftb_compute_adi, where dc1_adi is populated only from an
        # external NACV file, purely for momentum-rescaling direction -- not
        # used in the TDSE propagation itself, which runs off hvib_adi/
        # time_overlap_adi via nac_update_method=2/nac_algo=0 instead).

    # ---- P1-17: real ground-exciton time-overlap + off-diagonal hvib_adi ----
    # DIAGNOSTIC HARDENING (2026-07-11, after a boost::python::error_already_set /
    # core-dump crash on the first real-kernel attempt, with no Python traceback
    # surfacing -- the C++/Python boundary is swallowing whatever the actual
    # exception was). Wrapped in try/except with explicit traceback printing +
    # forced flush so the NEXT run attempt reveals exactly what and where, since
    # a raw C++ abort otherwise gives no diagnostic information at all. Also
    # print-bisected into stages so even if the traceback itself is still lost,
    # the last stage-N print visible before the crash narrows down the culprit.
    if enable_coupling:
        import sys as _sys
        try:
            assert "dt" in params, (
                "enable_coupling=True requires params['dt'] (nuclear timestep, a.u.) "
                "for the Hammes-Schiffer-Tully time-derivative-coupling formula.")
            dt = float(params["dt"])

            from exciton_density import cis_windowed_spectrum   # local import: avoids a
            # circular import at module load time (exciton_density.py imports
            # H0_and_dH0 FROM this module) -- safe here since by call time this
            # module has already finished loading.
            evals, evecs, configs, _, _ = cis_windowed_spectrum(
                rion, boxl, fxcalpha, fxcgamma, hartreeu, n_near, hop, hopslope, req)
            Psi_list = [evecs[:, si] for si in range(n_exciton_states)]   # P1-19: one Psi
            # per tracked exciton state (was a single Psi in P1-17's 2-state version).

            params.setdefault("MO_prev", {})
            params.setdefault("Psi_prev", {})
            params.setdefault("is_first_time", {})
            is_first_time = params["is_first_time"].get(traj, True)

            if is_first_time:
                MO_prev = C.copy()                             # first call: no real "previous"
                Psi_prev_list = [p_.copy() for p_ in Psi_list]  # step -- degrades to S_adi=I,
            else:                                               # zero off-diagonal, exactly
                MO_prev = params["MO_prev"][traj]                # matching the enable_coupling=False
                Psi_prev_list = params["Psi_prev"][traj]          # fallback already set above.

            occ = list(range(nocc))
            S_adi = multi_state_time_overlap(MO_prev, C, occ, Psi_prev_list, Psi_list, configs)

            for pp in range(nstates):
                for qq in range(nstates):
                    obj.time_overlap_adi.set(pp, qq, complex(float(S_adi[pp, qq]), 0.0))

            # Hammes-Schiffer-Tully formula on EVERY pair (p,q), p<q -- was hardcoded to
            # just the single (0,1) ground<->state0 edge in P1-17; now covers ALL edges
            # (ground<->state_k for every k, AND state_k<->state_l for every k!=l --
            # this last category is what the 2026-07-11 within-manifold-coupling finding
            # showed can be orders of magnitude LARGER than the direct ground edges in
            # the deeply-self-trapped regime, matching dftb_compute_adi/nac_algo=0).
            for pp in range(nstates):
                for qq in range(pp + 1, nstates):
                    d_pq = (float(S_adi[pp, qq]) - float(S_adi[qq, pp])) / (2.0 * dt)
                    obj.hvib_adi.set(pp, qq, complex(0.0, -d_pq))
                    obj.hvib_adi.set(qq, pp, complex(0.0, d_pq))

            params["MO_prev"][traj] = C.copy()
            params["Psi_prev"][traj] = [p_.copy() for p_ in Psi_list]
            params["is_first_time"][traj] = False

        except Exception:
            import traceback
            print("=" * 70, flush=True)
            print("P1-17/P1-19 enable_coupling block raised an exception -- full traceback:", flush=True)
            traceback.print_exc()
            _sys.stdout.flush()
            _sys.stderr.flush()
            print("=" * 70, flush=True)
            raise

    return obj


def get_default_params(nchain=64, dimer1=0.0868, lattice_ang=6.0,
                        fxcalpha=-2.0 * ELECTRONVOLT, fxcgamma=1.0,
                        hartreeu=0.0, n_near=3):
    """PRODUCTION parameters for the final report's three nchain=32 runs (delocalized
    baseline, self-trapping, ground-state control) -- the ab-initio-corrected
    Hamiltonian from Objective 1's pySCF re-parametrization (esteban-gadea/
    1.Ag_chain_recalibration_pySCF/report/objective1_report.md, Sections 4.3 and 6):
    beta_eq=-0.668653 eV (unchanged), rescaled slope A=1.150209 eV/Ang, re-fit
    r_eq=3.989152 Ang, repulsive potential B=0.032205/p=8 (was 0.231122/15).
    fxcalpha=-2.0 eV, fxcgamma=1 bohr (~0.529177 Ang) are the corrected electron-hole
    coupling parameters for this rerun (was fxcalpha=-0.1 eV, fxcgamma=0.26 Ang);
    hartreeu=0.0 keeps the Hartree/direct-Coulomb channel off, matching every prior
    working run in this project (see README "Note for future reference").

    NOT the same as the original PBE-fit values used for every P1-1 through P1-19
    development/validation run (hop=-0.0245725447 Ha, hopslope=0.007215487659 Ha/Bohr,
    req=4.922388 Bohr, pref=0.231122, p=15) -- those remain the legacy defaults in
    AgChain.py's own get_default_params (deliberately unchanged, see its docstring).
    n_near defaults to 3 -- the smallest odd window that pulls in a full +-k
    degenerate pair on each side of the gap, giving a genuinely localizable exciton
    state instead of a single delocalized configuration."""
    hop = -0.668653 * ELECTRONVOLT
    hopslope = 1.150209 * ELECTRONVOLT / ANGSTROM
    req = 3.989152 * ANGSTROM
    lattice = lattice_ang * ANGSTROM
    boxl = nchain * lattice
    r1 = lattice * (1 - dimer1) / 2
    r2 = lattice * (1 + dimer1) / 2

    return {
        "nchain": nchain, "hop": hop, "hopslope": hopslope, "req": req, "boxl": boxl,
        "r1": r1, "r2": r2,
        "pref": 0.032205, "p": 8,
        "fxcalpha": fxcalpha, "fxcgamma": fxcgamma, "hartreeu": hartreeu,
        "n_near": n_near,
    }


def ring_positions(nchain, r1, r2):
    rvect = [0.0] * (2 * nchain)
    for k in range(nchain):
        rvect[2 * k] = k * (r1 + r2)
        rvect[2 * k + 1] = k * (r1 + r2) + r1
    return rvect


def validate(nchain=32, dimer1=0.0868, lattice_ang=6.0, seed=0, jitter=0.02,
             delta=1e-5, test_dofs=(0, 1, 32, 63), n_near=3):
    """
    Builds cis_compute_adi's full ham_adi (ground+exciton total energy, state 1)
    at a RANDOMLY JITTERED 32-pair geometry (equilibrium has near-zero net force,
    a weak test) and checks obj.d1ham_adi against central finite differences of
    obj.ham_adi's own diagonal entries -- i.e. validates the FULL assembled
    contract (ground-state baseline + exciton piece together), not just the
    exciton piece alone (already validated in cis_gradient.py).

    P1-8b: now defaults to n_near=3 (was hardcoded to the n_near=1 single
    configuration before P1-8). At a JITTERED geometry the degenerate pairs
    are already split (not exactly degenerate), so this specific test doesn't
    exercise mo_response's degenerate-group-exclusion code path -- that's
    what cis_gradient.py's validate_degenerate() is for (run AT the exact
    equilibrium geometry). This test's job is to confirm the n_near=3 window
    is wired into the FULL Libra ham_adi/d1ham_adi contract correctly.

    Run inside the `libra` kernel (needs liblibra_core / util.libutil on the path).
    """
    p = get_default_params(nchain, dimer1, lattice_ang, n_near=n_near)
    n = 2 * nchain
    rion0 = np.array(ring_positions(nchain, p["r1"], p["r2"]))
    rng = np.random.default_rng(seed)
    rion = rion0 + rng.normal(scale=jitter, size=n)

    def make_q(rvec):
        qm = MATRIX(n, 1)
        for i, x in enumerate(rvec):
            qm.set(i, 0, x)
        return qm

    full_id = Py2Cpp_int([0, 0])
    obj = cis_compute_adi(make_q(rion), p, full_id)

    def E_state(rvec, istate):
        obj_ = cis_compute_adi(make_q(rvec), p, full_id)
        return obj_.ham_adi.get(istate, istate).real

    print(f"{'state':>5} {'k':>4} {'analytic dE/dR (eV/bohr)':>26} {'finite-diff (eV/bohr)':>24} {'rel err':>10}")
    for istate in (0, 1):
        for k in test_dofs:
            rp, rm = rion.copy(), rion.copy()
            rp[k] += delta
            rm[k] -= delta
            dE_fd = (E_state(rp, istate) - E_state(rm, istate)) / (2 * delta)
            dE_an = obj.d1ham_adi[k].get(istate, istate).real
            rel = abs(dE_an - dE_fd) / max(abs(dE_fd), 1e-30)
            print(f"{istate:5d} {k:4d} {dE_an / ELECTRONVOLT * ANGSTROM:26.8f} "
                  f"{dE_fd / ELECTRONVOLT * ANGSTROM:24.8f} {rel:10.2e}")


if __name__ == "__main__":
    validate()
    print("\n" + "=" * 70)
    print("P1-8b: same check AT the exact equilibrium (degenerate) geometry --")
    print("this is the actual starting point P1-7's Ehrenfest run uses.")
    print("=" * 70)
    validate(jitter=0.0)
