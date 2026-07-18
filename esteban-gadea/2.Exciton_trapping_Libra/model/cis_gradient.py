# Analytic nuclear gradients of the CIS/TDA exciton Hamiltonian (cis_exciton.py).
#
# Simpler than a textbook CIS gradient: ab initio CIS/TDDFT gradients need CPHF/
# Z-vector machinery because (a) the AO basis functions move with the nuclei (Pulay
# terms) and (b) the reference orbitals come from a self-consistent field, so their
# R-response requires solving a coupled-perturbed equation. Neither applies here: our
# "AO" basis is the site basis, which is R-independent and always orthonormal (S = I
# regardless of geometry -- same as AgChain.py's ovlp_dia), and the reference orbitals
# are just eigenvectors of the fixed one-electron H0(R) (no SCF), so their R-response
# is plain, uncoupled Rayleigh-Schrodinger perturbation theory. No CPHF needed.
#
# Three pieces, combined by the product rule:
#   1. d(eps_a - eps_i)/dR_k       -- Hellmann-Feynman on H0's eigenvalues (exact,
#                                      no eigenvector derivative needed for this piece)
#   2. dC_p/dR_k                   -- first-order perturbation theory,
#                                      dC_p/dR_k = sum_{q != p} <q|dH0/dRk|p>/(eps_p-eps_q) C_q
#                                      dH0/dRk is exactly AgChain.py's d1ham_dia.
#   3. dK(mu,nu)/dR_k               -- analytic derivative of the real-space kernel,
#                                      reused from the project's original Julia
#                                      implementation's ionic-force calculation.
#
# Piece 2's naive formula has a zero-denominator singularity whenever p is part of a
# degenerate pair -- true for every MO in this ring except the HOMO/LUMO band-edge
# states themselves, a direct consequence of the ring's translational symmetry, and a
# real blocker: it caps the gradient pipeline at n_near=1, a single, fully delocalized
# Bloch-like configuration that cannot spontaneously localize. The fix (excluding p's
# whole degenerate group from the response sum, not just q=p) is exact, not an
# approximation -- see mo_response's docstring for the full derivation, and
# cis_gradient_windowed for the n_near>1 generalization it unblocks.
#
# Validation: analytic dE/dR_k for the single HOMO->LUMO configuration
# (exciton_gradient_1config, non-degenerate case), checked against central finite
# differences (delta=1e-5 bohr) at 4 nuclear DOFs, 32-pair reference geometry:
# relative error ~3.6e-6 at every DOF (finite-difference-noise-level agreement). The
# degenerate-case fix (cis_gradient_windowed, n_near=3, pulling in one full +-k
# degenerate pair on each side of the gap, evaluated at the exactly-degenerate
# equilibrium geometry) validates to ~8.7e-6 relative error against an independent
# finite-difference reference (see mo_response's docstring for why the CIS eigenvalue,
# not an individual dC_p/dR_k, is the only thing that can be validated this way at a
# degenerate point). n_near=1 regression-checked bit-for-bit against the original,
# pre-fix formula (max difference ~1e-19, floating-point noise).

import numpy as np
from cis_exciton import build_H0, build_kernel, ANGSTROM, ELECTRONVOLT


def H0_with_gradient(nchain, dimer1, lattice_ang,
                      hop=-0.0245725447, hopslope=0.007215487659, req=4.922388):
    """Same H0 as cis_exciton.build_H0, plus dH0[k] = dH0/dR_k for every site DOF k
    (identical construction to AgChain.py's d1ham_dia)."""
    H0, rion, boxl = build_H0(nchain, dimer1, lattice_ang, hop, hopslope, req)
    n = len(rion)
    dH0 = np.zeros((n, n, n))

    def bond_grad(i, j):
        dH0[i, i, j] = -hopslope
        dH0[i, j, i] = -hopslope
        dH0[j, i, j] = hopslope
        dH0[j, j, i] = hopslope

    for i in range(n - 1):
        bond_grad(i, i + 1)
    bond_grad(n - 1, 0)
    return H0, dH0, rion, boxl


def kernel_with_gradient(rion, boxl, alpha, gamma, U):
    """Same K as cis_exciton.build_kernel, plus dK[k] = dK/dR_k. Reuses the exact
    analytic form of the project's original Julia implementation's ionic-force
    kernel derivative.

    Vectorized over all (mu, nu) pairs using numpy broadcasting for the pairwise
    distance matrix and K, and fancy indexing (np.triu_indices) to scatter the
    O(n^2) nonzero dK entries in one shot rather than a Python double loop -- a
    pure vectorization, bit-for-bit identical to the unvectorized version. ~30-58x
    faster per call (nchain=16 to 32), which matters since this runs every MD step."""
    n = len(rion)
    rion = np.asarray(rion, dtype=float)
    dij_mat = rion[None, :] - rion[:, None]           # dij_mat[mu,nu] = rion[nu]-rion[mu]
    dij_mat = dij_mat - np.trunc(dij_mat / (boxl / 2)) * boxl
    inv_u2 = (1.0 / U**2) if U != 0.0 else np.inf
    s1 = dij_mat**2 + inv_u2
    s2 = dij_mat**2 + gamma**2
    with np.errstate(divide="ignore", invalid="ignore"):
        hartree = 1.0 / np.sqrt(s1)   # -> 0 when U==0 (s1=inf), matches the original branch
    fxc = alpha / np.sqrt(s2)
    K = hartree + fxc
    np.fill_diagonal(K, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dK_dd = np.where(U != 0.0, -dij_mat / (s1 * np.sqrt(s1)), 0.0) - alpha * dij_mat / (s2 * np.sqrt(s2))
    np.fill_diagonal(dK_dd, 0.0)

    dK = np.zeros((n, n, n))
    mu_idx, nu_idx = np.triu_indices(n, k=1)
    vals = dK_dd[mu_idx, nu_idx]
    dK[mu_idx, mu_idx, nu_idx] = -vals
    dK[mu_idx, nu_idx, mu_idx] = -vals
    dK[nu_idx, mu_idx, nu_idx] = vals
    dK[nu_idx, nu_idx, mu_idx] = vals
    return K, dK


def mo_response(C, eps, dH0, p, degeneracy_tol=1e-8):
    """
    dC_p/dR_k for every DOF k -- first-order Rayleigh-Schrodinger perturbation theory,
    generalized to handle p being part of a degenerate group correctly, not just the
    non-degenerate case.

    THE FIX, IN ONE LINE: exclude p's ENTIRE degenerate group from the response sum
    (not just q == p), i.e.

        dC_p/dR_k = sum_{q not in group(p)} <q|dH0/dRk|p> / (eps_p - eps_q) * C_q

    where group(p) = {q : |eps_q - eps_p| < degeneracy_tol}. That's it -- no explicit
    subspace diagonalization needed in the code. For a non-degenerate p, group(p)={p}
    and this is IDENTICAL to the original formula (confirmed by regression test in
    validate(): n_near=1 reproduces the pre-fix exciton_gradient_1config bit-for-bit).

    WHY THIS SIMPLE FIX IS EXACT, NOT AN APPROXIMATION (the non-trivial nuance):

    Textbook degenerate perturbation theory says: within a degenerate group D, you
    first diagonalize the PROJECTED perturbation W = C_D^T (dH0/dRk) C_D to find the
    "good" zeroth-order states {D_1, D_2, ...} (the specific linear combinations of
    the group whose first-order response is well-defined) -- their eigenvalues give
    the correct dε/dRk (generalized Hellmann-Feynman), and a standard, textbook result
    (e.g. Griffiths/Sakurai) is that the good states' first-order eigenVECTOR
    correction has NO component within the degenerate subspace itself -- only mixing
    with states OUTSIDE the group. That "safe external sum" is exactly piece 2's
    original formula, just restricted to q outside the group.

    The question is what happens for p -- an ARBITRARY, not-necessarily-"good" member
    of the group (e.g. whatever numpy.linalg.eigh happened to return, an essentially
    arbitrary orthonormal basis choice within the degenerate eigenspace). Write p in
    terms of the good states: C_p = sum_i U[p,i] D_i (U is the orthogonal matrix that
    diagonalizes W). Since each D_i's first-order response has zero intra-group
    component, D_i(s) = D_i(0) + s * (external sum for D_i) + O(s^2). Substituting back:

        C_p(s) = sum_i U[p,i] D_i(s) = C_p(0) + s * sum_i U[p,i] * (external sum for D_i)

    and sum_i U[p,i] * (external sum for D_i) simplifies EXACTLY (by linearity, and
    because sum_i U[p,i] D_i = C_p identically) to the external sum evaluated directly
    at C_p -- the same formula as if p had been non-degenerate all along, just with the
    partner(s) excluded. The U matrix -- and the whole explicit diagonalization step --
    cancels out of the final formula. (Sanity-checked in an isolated 2-level toy model
    with no external coupling: there, H0(s) = eps0*I + s*W exactly, so the "good"
    states D_1, D_2 don't rotate at ALL as s varies -- meaning ANY fixed combination
    C_p = cos(theta) D_1 + sin(theta) D_2 is also exactly s-independent, confirming
    "zero intra-group response" isn't just a first-order approximation there, it's
    exact to all orders.)

    The energy piece (eps_a - eps_i in the CIS diagonal) needs NO change at all, even
    for degenerate i or a: the existing formula C_p^T (dH0/dRk) C_p (used unchanged in
    exciton_gradient_1config's d_deps line and in cis_gradient_windowed below) is the
    GENERALIZED Hellmann-Feynman theorem applied to the expectation value
    <C_p(s)|H0(s)|C_p(s)> for our specific, fixed C_p(s) -- and by the ordinary product
    rule plus the eigenvalue equation at s=0, this expectation value's first derivative
    equals <C_p|dH0/dRk|C_p> EXACTLY, regardless of whether C_p(s) stays an eigenvector
    of H0(s) away from s=0 (it generally doesn't, for a degenerate/arbitrary p -- see
    above -- but the expectation-value identity doesn't require that it does). This is
    exactly why ground_state_energy_and_gradient in cis_compute_adi.py already worked
    correctly for the full (degenerate-pair-riddled) occupied manifold without any fix
    -- same underlying fact, applied there via a trace/projector argument instead.

    IMPORTANT CONSEQUENCE FOR VALIDATION: because the fix's correctness hinges on the
    OUTER (physically observable, basis-invariant) CIS eigenvalue, not on any single
    "good" choice of individual degenerate eigenvectors, you CANNOT validate this fix
    by finite-differencing an individual dC_p/dR_k directly -- numpy.linalg.eigh's
    arbitrary basis choice within a (near-)degenerate block is itself numerically
    unstable/discontinuous from one perturbed geometry to the next, so a naive
    finite-difference check of the raw eigenvector would just measure that numerical
    noise, not the physics. The only valid check is at the level of a basis-invariant
    quantity built from the whole degenerate group -- i.e. the CIS Hamiltonian's
    eigenvalue itself. See cis_gradient_windowed's validate_degenerate() for exactly
    this test, run AT the exactly-degenerate equilibrium geometry.

    Vectorized over both k and q as two BLAS matmuls rather than a Python double
    loop, since this is a per-MD-step hot path:
        W[k,m] = sum_n dH0[k,m,n] C[n,p]     (np.tensordot, batched matvec)
        G[k,q] = sum_m W[k,m] C[m,q] = (W @ C)[k,q]   (= <q|dH0_k|p> for every k,q at once)
        dC_p[k,:] = sum_{q external} G[k,q]/(eps_p-eps_q) * C[:,q] = (weight @ C.T)[k,:]
    Degenerate q's are masked out of `weight` (set to 0) rather than skipped in a
    loop -- same "exclude p's whole degenerate group" fix, just vectorized. Bit-for-
    bit equivalent to the unvectorized version (max difference ~1e-16, floating-point
    noise), for both non-degenerate and degenerate p. ~22-30x faster per call.
    """
    n = C.shape[0]
    mask = np.abs(eps - eps[p]) >= degeneracy_tol
    denom = np.where(mask, eps[p] - eps, 1.0)          # placeholder denom where masked; zeroed below
    W = np.tensordot(dH0, C[:, p], axes=([2], [0]))    # (n_dof, n): dH0[k] @ C[:,p]
    G = W @ C                                           # (n_dof, n): <q|dH0_k|p> for all k,q
    weight = np.where(mask[None, :], G / denom[None, :], 0.0)
    return weight @ C.T                                 # (n_dof, n)


def contract3(v1, T, v2):
    """result[k] = v1 . T[k] . v2 for a batch of matrices T (shape (n_dof,n,n)).
    Same quantity as np.einsum('m,kmn,n->k', v1, T, v2), but computed as two
    BLAS calls (tensordot + matmul) instead of a generic einsum contraction --
    ~4-5x faster in practice since einsum doesn't pick an optimal contraction
    order for this 3-operand pattern by default. Used throughout
    cis_compute_adi.py's per-MD-step hot path."""
    W = np.tensordot(T, v2, axes=([2], [0]))            # (n_dof, n)
    return W @ v1                                        # (n_dof,)


def exciton_gradient_1config(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma, hartreeu):
    """
    Nuclear gradient of the lowest (and, for a single configuration, only) CIS exciton
    energy for the dominant HOMO->LUMO configuration. Returns (E, dE) where dE[k] is
    dE/dR_k for every site DOF k, atomic units throughout.
    """
    H0, dH0, rion, boxl = H0_with_gradient(nchain, dimer1, lattice_ang)
    eps, C = np.linalg.eigh(H0)
    n = H0.shape[0]
    nocc = n // 2
    homo, lumo = nocc - 1, nocc

    K, dK = kernel_with_gradient(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exch, dK_exch = kernel_with_gradient(rion, boxl, 0.0, fxcgamma, hartreeu)

    dC_homo = mo_response(C, eps, dH0, homo)
    dC_lumo = mo_response(C, eps, dH0, lumo)

    def Dvec(p, q):
        return C[:, p] * C[:, q]

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


def cis_gradient_windowed(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma, hartreeu,
                           n_near, state_index=0, degeneracy_tol=1e-8):
    """
    General n_near CIS/TDA exciton energy + gradient for an arbitrary configuration
    window -- the n_near>1 generalization exciton_gradient_1config can't do (it's
    hardcoded to the single HOMO->LUMO configuration). A spatially localizable exciton
    (needed for self-trapping) requires a superposition of several near-gap
    configurations, not the single delocalized Bloch-like mode n_near=1 gives.

    Use an ODD n_near (1, 3, 5, ...) to keep the window symmetric -- HOMO/LUMO are
    non-degenerate, but every level further out comes in a complete +-k degenerate
    pair (a consequence of the ring's translational symmetry); an even n_near would
    split a degenerate pair across the occ/virt boundary, asymmetrically including one
    partner but not the other, which has no clean physical justification.

    Args:
        n_near (int): occ/virt window half-width, same convention as
            cis_exciton.py's validate() (n_near occupied states at/below HOMO x
            n_near virtual states at/above LUMO).
        state_index (int): which CIS eigenstate's energy/gradient to return
            (0 = lowest exciton state).

    Returns:
        (E_n, dE_n, configs): E_n is the state_index-th CIS eigenvalue (Ha), dE_n[k]
        is its gradient w.r.t. site DOF k (Ha/Bohr), configs is the (i,a) list.

    Cost: O(n_dof * m^2) where m = (2*n_near)^2 is the number of configurations --
    each of the n_dof gradient components rebuilds an m x m matrix from scratch. Fine
    for small windows (n_near=3, m=9); revisit before scaling much further.
    """
    H0, dH0, rion, boxl = H0_with_gradient(nchain, dimer1, lattice_ang)
    n = H0.shape[0]
    nocc = n // 2
    homo, lumo = nocc - 1, nocc
    eps, C = np.linalg.eigh(H0)

    occ_idx = list(range(homo - n_near + 1, homo + 1))
    virt_idx = list(range(lumo, lumo + n_near))
    configs = [(i, a) for i in occ_idx for a in virt_idx]
    m = len(configs)

    K, dK = kernel_with_gradient(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exch, dK_exch = kernel_with_gradient(rion, boxl, 0.0, fxcgamma, hartreeu)

    # dC_p/dR_k for every orbital actually used in the window (each computed once;
    # this is where the degenerate-safe mo_response fix gets exercised for n_near>1).
    used = sorted(set(occ_idx) | set(virt_idx))
    dC = {p: mo_response(C, eps, dH0, p, degeneracy_tol) for p in used}

    def Dvec(p, q):
        return C[:, p] * C[:, q]

    def dDvec(p, q, k):
        return dC[p][k] * C[:, q] + C[:, p] * dC[q][k]

    H = np.zeros((m, m))
    for I, (i, a) in enumerate(configs):
        for J, (j, b) in enumerate(configs):
            direct = Dvec(i, a) @ K @ Dvec(j, b)
            exch = Dvec(i, j) @ K_exch @ Dvec(a, b)
            val = 2.0 * direct - exch
            if i == j and a == b:
                val += (eps[a] - eps[i])
            H[I, J] = val
    evals, evecs = np.linalg.eigh(H)
    Psi = evecs[:, state_index]
    E_n = evals[state_index]

    n_dof = dH0.shape[0]
    dE_n = np.zeros(n_dof)
    for k in range(n_dof):
        dHk = np.zeros((m, m))
        for I, (i, a) in enumerate(configs):
            for J, (j, b) in enumerate(configs):
                d_ia, d_jb = Dvec(i, a), Dvec(j, b)
                dd_ia, dd_jb = dDvec(i, a, k), dDvec(j, b, k)
                d_direct = dd_ia @ K @ d_jb + d_ia @ dK[k] @ d_jb + d_ia @ K @ dd_jb
                d_ij, d_ab = Dvec(i, j), Dvec(a, b)
                dd_ij, dd_ab = dDvec(i, j, k), dDvec(a, b, k)
                d_exch = dd_ij @ K_exch @ d_ab + d_ij @ dK_exch[k] @ d_ab + d_ij @ K_exch @ dd_ab
                val = 2.0 * d_direct - d_exch
                if i == j and a == b:
                    # generalized Hellmann-Feynman -- valid even when i or a is
                    # degenerate, see mo_response's docstring for why.
                    val += (C[:, a] @ dH0[k] @ C[:, a]) - (C[:, i] @ dH0[k] @ C[:, i])
                dHk[I, J] = val
        dE_n[k] = Psi @ dHk @ Psi

    return E_n, dE_n, configs


def validate_degenerate(nchain=32, dimer1=0.0868, lattice_ang=6.0,
                         fxcalpha=-0.1 * ELECTRONVOLT, fxcgamma=0.26 * ANGSTROM,
                         hartreeu=0.115 * ELECTRONVOLT, n_near=3, state_index=0,
                         delta=1e-5, test_dofs=(0, 1, 5, 32, 63)):
    """
    Checks cis_gradient_windowed at the exactly-degenerate equilibrium geometry (not
    a jittered one -- see mo_response's docstring on why this specific geometry is
    the real test). n_near=3 pulls in one complete +-k degenerate pair on each side
    of the gap (HOMO-1/HOMO-2, LUMO+1/LUMO+2), so this directly exercises the
    degenerate-PT fix, not just the already-validated non-degenerate HOMO/LUMO case.

    The finite-difference reference is built independently (fresh H0/CIS-matrix
    construction and diagonalization at each perturbed geometry, no dependence on
    mo_response or any analytic gradient code) -- this is deliberate: since CIS
    eigenvalues are basis-invariant (unaffected by numpy.linalg.eigh's arbitrary
    choice of basis within a degenerate block), this reference is trustworthy
    regardless of the individual-eigenvector labeling ambiguity that makes a
    finite-difference check of raw MO coefficients unreliable at this geometry.

    Also regression-checks n_near=1 against exciton_gradient_1config -- should match
    to floating-point precision, confirming the degenerate-case fix above doesn't
    change the already-validated non-degenerate behavior.
    """
    H0, rion, boxl = build_H0(nchain, dimer1, lattice_ang)
    eps0 = np.linalg.eigvalsh(H0)
    nocc = len(rion) // 2
    homo, lumo = nocc - 1, nocc
    print("Degeneracy check near the gap (eV):")
    for off in range(min(n_near, 3)):
        print(f"  HOMO-{off}: {eps0[homo - off] / ELECTRONVOLT: .8f}   "
              f"LUMO+{off}: {eps0[lumo + off] / ELECTRONVOLT: .8f}")
    if n_near >= 3:
        gap = abs(eps0[homo - 1] - eps0[homo - 2])
        print(f"  |eps[HOMO-1]-eps[HOMO-2]| = {gap:.2e} Ha (should be ~0)")

    E_n, dE_n, configs = cis_gradient_windowed(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma,
                                                hartreeu, n_near, state_index)
    print(f"\nn_near={n_near}, #configs={len(configs)}, E_{state_index} = {E_n / ELECTRONVOLT:.6f} eV")

    def E_n_at(rion_pert):
        n = len(rion_pert)
        hop, hopslope, req = -0.0245725447, 0.007215487659, 4.922388
        Hp = np.zeros((n, n))

        def bond(i, j, dr):
            h = hop + hopslope * (dr - req)
            Hp[i, j] = h
            Hp[j, i] = h

        for i in range(n - 1):
            bond(i, i + 1, rion_pert[i + 1] - rion_pert[i])
        bond(n - 1, 0, (rion_pert[0] + boxl) - rion_pert[n - 1])
        eps_, C_ = np.linalg.eigh(Hp)
        occ_idx = list(range(homo - n_near + 1, homo + 1))
        virt_idx = list(range(lumo, lumo + n_near))
        configs_ = [(i, a) for i in occ_idx for a in virt_idx]
        m = len(configs_)
        K, _ = kernel_with_gradient(rion_pert, boxl, fxcalpha, fxcgamma, hartreeu)
        Ke, _ = kernel_with_gradient(rion_pert, boxl, 0.0, fxcgamma, hartreeu)

        def Dvec(p, q):
            return C_[:, p] * C_[:, q]

        Hm = np.zeros((m, m))
        for I, (i, a) in enumerate(configs_):
            for J, (j, b) in enumerate(configs_):
                direct = Dvec(i, a) @ K @ Dvec(j, b)
                exch = Dvec(i, j) @ Ke @ Dvec(a, b)
                val = 2.0 * direct - exch
                if i == j and a == b:
                    val += eps_[a] - eps_[i]
                Hm[I, J] = val
        return np.linalg.eigvalsh(Hm)[state_index]

    print(f"\n{'k':>4} {'analytic (eV/bohr)':>20} {'finite-diff (eV/bohr)':>22} {'rel err':>10}")
    for k in test_dofs:
        rp, rm = rion.copy(), rion.copy()
        rp[k] += delta
        rm[k] -= delta
        dE_fd = (E_n_at(rp) - E_n_at(rm)) / (2 * delta)
        rel = abs(dE_n[k] - dE_fd) / max(abs(dE_fd), 1e-30)
        print(f"{k:4d} {dE_n[k] / ELECTRONVOLT * ANGSTROM:20.8f} "
              f"{dE_fd / ELECTRONVOLT * ANGSTROM:22.8f} {rel:10.2e}")

    E1_old, dE1_old = exciton_gradient_1config(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma, hartreeu)
    E1_new, dE1_new, _ = cis_gradient_windowed(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma, hartreeu, 1, 0)
    print(f"\nRegression check (n_near=1 vs exciton_gradient_1config): "
          f"|dE|={abs(E1_old - E1_new):.2e} Ha, max|d(dE)|={np.max(np.abs(dE1_old - dE1_new)):.2e} Ha/Bohr")


def validate(nchain=32, dimer1=0.0868, lattice_ang=6.0,
             fxcalpha=-0.1 * ELECTRONVOLT, fxcgamma=0.26 * ANGSTROM, hartreeu=0.115 * ELECTRONVOLT,
             delta=1e-5, test_dofs=(0, 1, 32, 63)):
    """Central-finite-difference check of exciton_gradient_1config against itself
    (rebuilds E at R+/-delta*e_k). Prints relative error per tested DOF."""
    E, dE = exciton_gradient_1config(nchain, dimer1, lattice_ang, fxcalpha, fxcgamma, hartreeu)
    H0, rion, boxl = build_H0(nchain, dimer1, lattice_ang)

    def E_at(rion_pert):
        n = len(rion_pert)
        Hp = np.zeros((n, n))
        hop, hopslope, req = -0.0245725447, 0.007215487659, 4.922388

        def bond(i, j, dr):
            h = hop + hopslope * (dr - req)
            Hp[i, j] = h
            Hp[j, i] = h

        for i in range(n - 1):
            bond(i, i + 1, rion_pert[i + 1] - rion_pert[i])
        bond(n - 1, 0, (rion_pert[0] + boxl) - rion_pert[n - 1])
        eps_, C_ = np.linalg.eigh(Hp)
        nocc = n // 2
        homo, lumo = nocc - 1, nocc
        Kd, _ = kernel_with_gradient(rion_pert, boxl, fxcalpha, fxcgamma, hartreeu)
        Ke, _ = kernel_with_gradient(rion_pert, boxl, 0.0, fxcgamma, hartreeu)
        d_ia = C_[:, homo] * C_[:, lumo]
        d_ii = C_[:, homo] * C_[:, homo]
        d_aa = C_[:, lumo] * C_[:, lumo]
        return (eps_[lumo] - eps_[homo]) + 2 * (d_ia @ Kd @ d_ia) - (d_ii @ Ke @ d_aa)

    print(f"{'k':>4} {'analytic (eV/bohr)':>20} {'finite-diff (eV/bohr)':>22} {'rel err':>10}")
    for k in test_dofs:
        rp, rm = rion.copy(), rion.copy()
        rp[k] += delta
        rm[k] -= delta
        dE_fd = (E_at(rp) - E_at(rm)) / (2 * delta)
        rel = abs(dE[k] - dE_fd) / max(abs(dE_fd), 1e-30)
        print(f"{k:4d} {dE[k] / ELECTRONVOLT * ANGSTROM:20.8f} {dE_fd / ELECTRONVOLT * ANGSTROM:22.8f} {rel:10.2e}")


if __name__ == "__main__":
    validate()
    print("\n" + "=" * 70)
    print("Degenerate-case validation (cis_gradient_windowed, n_near=3)")
    print("=" * 70)
    validate_degenerate()
