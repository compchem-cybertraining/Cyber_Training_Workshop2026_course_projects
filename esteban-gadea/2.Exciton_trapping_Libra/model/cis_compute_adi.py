# CIS/TDA exciton Hamiltonian + gradients, ported into Libra's on-the-fly
# ham_update_method=2 ("adiabatic") compute_model contract.
#
# Template: libra_py.packages.dftbplus.methods.dftb_compute_adi, the function
# Libra's own DFTB+ many-electron NA-MD workflows use. Key structural choices
# this module mirrors:
#
#   * obj.ham_adi is diagonal = state energies directly -- Libra does not
#     diagonalize anything on its side for this contract (unlike AgChain.py's
#     ham_dia, which hands Libra a diabatic matrix to diagonalize).
#   * Every state's diagonal entry gets the same ground-state total energy added
#     on top of its own (relative) energy, via `obj.ham_adi.add(i, i, e0)` --
#     states differ only by what's added on top of that shared baseline. This is
#     why this module adds ground-state electronic + repulsive energy back in --
#     without it the PES would be missing the dominant lattice-restoring forces.
#   * obj.d1ham_adi is diagonal-only per DOF: d1ham_adi[k].set(i,i, dE_i/dR_k)
#     (the energy gradient, not the force -- force = -dE/dR).
#   * obj.dc1_adi legitimately left all-zero when there's no NAC data -- the
#     reference code does exactly this when NACV data doesn't exist. With a
#     fixed configuration window built fresh from instantaneous eigenvectors
#     every call, there's no coupling pathway available in the base (enable_
#     coupling=False) model, so this is a documented approximation, not a
#     shortcut.
#   * obj.hvib_adi = obj.ham_adi when dc1_adi/time_overlap contribute no
#     off-diagonal terms (same as the reference: hvib's off-diagonals come
#     purely from the time-overlap-derived NAC estimate, zero here for the
#     same reason as dc1_adi).
#   * obj.basis_transform, obj.time_overlap_adi: identity by default. Caveat:
#     true state-tracking across MD steps needs a real time-overlap between
#     consecutive-step eigenvectors to catch state reordering/degeneracy
#     crossing; this module assumes state identity is stable step-to-step
#     (valid near the dimerized equilibrium geometry where HOMO/LUMO stay
#     non-degenerate band-edge states -- see cis_gradient.py's mo_response
#     docstring). enable_coupling=True (below) replaces this identity fallback
#     with a real time-overlap.
#
# Reuses cis_gradient.py's kernel_with_gradient and mo_response unchanged (both
# already geometry-agnostic). New in this module: a general-geometry H0_and_dH0
# (cis_exciton.py/cis_gradient.py's build_H0/H0_with_gradient only build the
# fixed equilibrium-dimerized geometry; MD moves the atoms off of it), the
# repulsive potential energy/gradient, and the ground-state total energy/
# gradient assembly described above.
#
# Validated by finite-difference check of the fully assembled ham_adi diagonal
# (ground-state total energy+repulsion and ground+exciton total) at a randomly
# jittered (not equilibrium) 32-pair geometry -- see validate() below.
# Equilibrium geometries have near-zero net force by construction, a weak test;
# jittering exercises genuinely nonzero gradients in every term.

import sys
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform in ("linux", "linux2"):
    from liblibra_core import *
import util.libutil as comn

import numpy as np
from cis_exciton import ANGSTROM, ELECTRONVOLT
from cis_gradient import kernel_with_gradient, mo_response, contract3
from cis_time_overlap import ground_exciton_time_overlap, multi_state_time_overlap   # only used
                                                             # when params["enable_coupling"]=True


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
    rescaled slope A=1.150209 eV/Ang, and the re-fit r_eq=3.989152 Ang -- not the
    original PBE-fit values (hop=-0.0245725447, hopslope=0.007215487659, req=4.922388,
    all in Ha/Bohr) used during this module's own development. Those legacy values
    are preserved as-is in AgChain.py's and cis_exciton.py's own get_default_params/
    build_H0 defaults (kept there deliberately, since AgChain.validate() checks
    against a reference eigenvalue computed with them) -- do not "fix" those to match
    production; they validate a different, older thing.
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
    """V(r) = -hop*pref*(req/r)^p per bond, matching Ag_chains_parametrization.pdf
    Eq. 3 / the project's original Julia implementation's construct_potential."""
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
    piece). This is the shared baseline every state's ham_adi diagonal gets on
    top of its own relative energy (see module docstring).

    dE_elec is vectorized via the occupied-space density matrix
    P = C_occ @ C_occ.T (n,n): sum_i C[:,i] . dH0[k] . C[:,i] = trace(dH0[k] @ P),
    computed for all k at once as a single matrix-vector product
    dH0.reshape(n_dof,-1) @ P.T.reshape(-1) (BLAS gemv) rather than a Python loop
    over n_dof -- ~77-94x faster (nchain=32 to 16), since this runs every MD step."""
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
    General n_near CIS/TDA exciton energy + gradient at an arbitrary geometry --
    the arbitrary-geometry generalization of cis_gradient.py's cis_gradient_windowed,
    the same relationship exciton_energy_and_gradient above has to
    exciton_gradient_1config. Reuses the degenerate-safe mo_response unchanged --
    that's what makes n_near>1 possible: for n_near=1 this is byte-for-byte identical
    to exciton_energy_and_gradient, but for n_near>1 the occ/virt window pulls in
    +-k degenerate pairs (e.g. HOMO-1/HOMO-2 for n_near=3) that a single-
    configuration treatment can't handle.

    n_near matters physically: with n_near=1, the exciton is a single delocalized
    Bloch-like configuration with no way to spatially localize. With n_near>1, the
    returned state (state_index=0, the lowest CIS eigenstate) is a genuine
    superposition of several near-gap configurations -- something that can be
    spatially localized, the object needed for self-trapping to be possible at all.

    Args:
        n_near (int): occ/virt window half-width (use ODD values -- see
            cis_gradient.py's cis_gradient_windowed docstring for why).
        state_index (int): which CIS eigenstate to return (0 = lowest exciton
            state -- this is what cis_compute_adi uses for its state-1 PES).

    Returns:
        (E_n, dE_n): E_n is the state_index-th CIS eigenvalue (Ha, relative to
        the bare gap baseline, same convention as exciton_energy_and_gradient),
        dE_n[k] is its gradient w.r.t. site DOF k (Ha/Bohr).

    Validated by finite differences: agreement ~1e-7 at the exact equilibrium
    (degenerate) geometry and ~1e-9-1e-10 at a jittered (non-degenerate) geometry,
    n_near=3. n_near=1 regression-checked exactly equal to exciton_energy_and_gradient.

    This is the per-MD-step hot path (called every Ehrenfest timestep), so it's
    vectorized past the straightforward implementation in three ways, none of which
    change the underlying math (validated bit-for-bit against the unvectorized
    version, ~1e-17-1e-18 max difference):

      1. Memoized Dvec/dDvec: `get_D` caches each unique (p,q) product once per call
         instead of recomputing it for every (I,J) configuration pair.
      2. Hm/dHk matrix symmetry: loops over the upper triangle only (J >= I) and
         mirrors, with the off-diagonal contribution to Psi @ dHk @ Psi weighted by 2
         -- avoids ever building the full dHk matrix or computing Psi @ dHk @ Psi.
      3. Vectorized gradient assembly: uses `contract3` (cis_gradient.py) to get each
         (I,J) pair's full (n_dof,)-shaped gradient contribution in one BLAS-backed
         call, accumulated directly into dE_n, instead of an explicit per-k Python
         loop rebuilding an (m,m) matrix every iteration.

      Combined with kernel_with_gradient's and mo_response's own vectorization,
      end-to-end speedup is ~12-25x depending on nchain.
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
            fxcalpha, fxcgamma, hartreeu, n_near (see get_default_params). n_near
            is the CIS configuration window's half-width feeding state 1's
            energy/gradient (see cis_windowed_energy_and_gradient's docstring for
            why n_near>1 is needed for a spatially localizable exciton).

            Optional `enable_coupling` (bool, default False, so every single-PES
            production run is unaffected). This is part of the ongoing
            nonadiabatic-coupling exploration (report.md Section 2.5), not used
            by the three production runs. When True, requires `dt` (nuclear
            timestep, a.u.) also in params, and:
              - populates obj.time_overlap_adi with the real ground-exciton
                time-overlap S_adi(t-dt, t) (cis_time_overlap.py's closed-form
                formula), instead of the identity fallback,
              - populates obj.hvib_adi's off-diagonal via the standard
                Hammes-Schiffer-Tully formula d_01=(S01-S10)/(2dt), matching
                Libra's DFTB+ workflow's nac_algo=0 pathway (already set by
                recipes/ehrenfest_onthefly.py's nac_update_method=2),
              - caches the previous step's MO coefficients + CIS eigenvector in
                `params` (keyed by trajectory index) since a time-overlap needs
                both steps' wavefunctions -- this makes cis_compute_adi
                implicitly stateful across calls when enable_coupling=True (it
                is fully stateless otherwise).
            dc1_adi is left at all-zero either way -- this recipe's
            hop_acceptance_algo/momenta_rescaling_algo don't consume it, matching
            the reference workflow, where dc1_adi only matters for NACV-based
            momentum rescaling. Exploratory runs found the direct ground<->state0
            coupling is suppressed by an exact symmetry-driven cancellation
            specific to state_index=0 -- see `n_exciton_states` below.

            Optional `n_exciton_states` (int, default 1, preserving the original
            2-state ground+state0 model exactly). Set to 2 to track a third state
            (ground, exciton state_index=0, exciton state_index=1), with
            enable_coupling=True then populating the full 3x3 time-overlap/
            hvib_adi (all three edges) via cis_time_overlap.multi_state_time_overlap.
            Exists because the direct ground<->state0 channel is symmetry-
            suppressed (noise floor) while the state0<->state1 (within-manifold)
            channel is not -- it can be orders of magnitude larger in the deeply
            self-trapped regime, a channel the 2-state model can't represent.
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
    n_exciton_states = params.get("n_exciton_states", 1)   # default 1 preserves the original
                                                             # 2-state (ground+state0) behavior
                                                             # exactly -- see module docstring.
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

    # Real ground-exciton time-overlap + off-diagonal hvib_adi. Wrapped in
    # try/except with explicit traceback printing + forced flush: a raw crash at
    # the C++/Python boundary here otherwise gives no diagnostic information at
    # all (Libra swallows the underlying Python exception).
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
            Psi_list = [evecs[:, si] for si in range(n_exciton_states)]   # one Psi per
                                                                            # tracked exciton state

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

            # Hammes-Schiffer-Tully formula on every pair (p,q), p<q: ground<->state_k
            # for every k, and state_k<->state_l for every k!=l -- this last category
            # can be orders of magnitude larger than the direct ground edges in the
            # deeply self-trapped regime (matching Libra's DFTB+ workflow's nac_algo=0).
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
            print("enable_coupling block raised an exception -- full traceback:", flush=True)
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

    Not the same as the original PBE-fit values used during this module's own
    development (hop=-0.0245725447 Ha, hopslope=0.007215487659 Ha/Bohr,
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

    Defaults to n_near=3. At a jittered geometry the degenerate pairs are already
    split (not exactly degenerate), so this specific test doesn't exercise
    mo_response's degenerate-group-exclusion code path -- that's what
    cis_gradient.py's validate_degenerate() is for (run at the exact equilibrium
    geometry). This test's job is to confirm the n_near=3 window is wired into
    the full Libra ham_adi/d1ham_adi contract correctly.

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
    print("Same check at the exact equilibrium (degenerate) geometry -- the")
    print("actual starting point the production Ehrenfest runs use.")
    print("=" * 70)
    validate(jitter=0.0)
