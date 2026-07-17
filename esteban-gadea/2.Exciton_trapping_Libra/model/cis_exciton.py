# *********************************************************************************
# Path 1 (single-quasiparticle reformulation): CIS/TDA-with-kernel exciton Hamiltonian
# built on top of the ground-state TB Hamiltonian from AgChain.py.
#
# GOAL: replace "propagate the full many-electron rho(t)" with "propagate amplitudes
# C(t) over a basis of single electron-hole excitation configurations |i->a>", which
# genuinely fits Libra's rho = C.C+ (pure-state) convention -- see README.md's Open
# Questions section for why the full-rho approach doesn't fit Libra's native TSH/
# Ehrenfest engine.
#
# This module builds and diagonalizes the STATIC (rigid-lattice) exciton Hamiltonian
# only -- no nuclear gradients yet (that's P1-5), no Libra plumbing yet (P1-6). Its
# job is to nail down the correct many-body matrix-element formula and validate it
# gives sensible (sub-gap, bound) exciton energies before anything gets wired into
# Libra.
#
# ---------------------------------------------------------------------------------
# KEY DERIVATION NOTE (found empirically 2026-07-09 -- read this before changing
# anything here):
#
# The standard singlet CIS/TDA matrix element for a closed-shell reference is
#     H(ia,jb) = delta_ij delta_ab (eps_a - eps_i) + 2*(ia|jb) - (ij|ab)
# where (pq|rs) = sum_{mu,nu} C[mu,p]*C[mu,q]*K[mu,nu]*C[nu,r]*C[nu,s] is the MO-basis
# two-electron integral built from the real-space density-density kernel K(mu,nu) =
# hartree(mu,nu) + fxc(mu,nu) (the same kernel as TLS_module.jl's buildxc).
#
# Naively plugging the FULL kernel (hartree+fxc) into BOTH the direct (ia|jb) and
# exchange (ij|ab) slots gives the WRONG sign for the fxc-driven sub-gap peak: it was
# numerically verified that pure repulsive hartreeu (K>0) correctly produces bound
# (sub-gap) singlet states dominated by the exchange channel -- confirming the overall
# formula/sign convention is right in general -- but fxcalpha<0 plugged into that SAME
# formula makes states MORE bound... no, wait: it makes them LESS bound / blue-shifted,
# the opposite of what the fxc term is supposed to do (create the sub-gap peak). See
# cis_dev.py in the session scratch history for the full sweep that exposed this.
#
# Resolution: fxc is not a literal second-quantized two-body interaction subject to
# Pauli exchange the way a genuine Coulomb repulsion is -- it plays the role of a
# TDDFT-style exchange-correlation KERNEL (same spirit as long-range-corrected fxc
# kernels used to recover excitonic/BSE-like binding inside TDDFT for solids). In the
# standard Casida/TDDFT-TDA equations, such a kernel enters ONLY the doubled "direct"
# (Coulomb-like) slot, not the exchange slot:
#     H(ia,jb) = delta_ij delta_ab (eps_a-eps_i) + 2*(ia| K_hartree+K_fxc |jb) - (ij| K_hartree |ab)
# i.e. hartreeu contributes to BOTH slots (it's a genuine repulsive interaction), fxc
# contributes to the direct slot ONLY. This was verified numerically: with this split,
# fxcalpha<0 alone (hartreeu=0) gives a positive (sub-gap) binding energy that grows
# linearly with |fxcalpha|, and the combined hartreeu+fxcalpha case (using the general
# default hartreeu=0.115 eV together with example/inp.in's fxcalpha=-0.1 eV) gives a
# converged (with configuration-window size) binding energy of ~0.11 eV against a bare
# gap of 0.386 eV for the 64-pair, dimer1=0.0868 reference geometry -- physically
# reasonable (order the gap magnitude, not absurd), and the formula is well-behaved
# (monotonic, converges as the window grows). Treat this as validated-by-consistency,
# NOT yet cross-checked against the real-time Delta-rho sub-gap peak value itself --
# that comparison is still open (see README).
# ---------------------------------------------------------------------------------

import numpy as np

ANGSTROM = 1.0 / 0.529177210903
ELECTRONVOLT = 1.0 / 27.211386245988


def build_H0(nchain, dimer1, lattice_ang,
             hop=-0.0245725447, hopslope=0.007215487659, req=4.922388):
    """Ground-state (rigid-lattice) TB Hamiltonian -- same construction as AgChain.py's
    compute_ag_chain, but returned as a plain numpy array (no Libra dependency) since
    this module is meant to be developed/validated standalone before Step P1-6 wires
    it into Libra's compute_model contract."""
    n = 2 * nchain
    lattice = lattice_ang * ANGSTROM
    r1 = lattice * (1 - dimer1) / 2
    r2 = lattice * (1 + dimer1) / 2
    boxl = nchain * lattice
    rion = np.zeros(n)
    for k in range(nchain):
        rion[2 * k] = k * (r1 + r2)
        rion[2 * k + 1] = k * (r1 + r2) + r1

    H = np.zeros((n, n))

    def bond(i, j, dr):
        h = hop + hopslope * (dr - req)
        H[i, j] = h
        H[j, i] = h

    for i in range(n - 1):
        bond(i, i + 1, rion[i + 1] - rion[i])
    bond(n - 1, 0, (rion[0] + boxl) - rion[n - 1])

    return H, rion, boxl


def build_kernel(rion, boxl, alpha, gamma, U):
    """Real-space density-density kernel K(mu,nu) = hartree + fxc, matching
    TLS_module.jl's buildxc exactly (same truncating minimum-image convention,
    same on-site/self term, U=0 means the hartree piece is switched off)."""
    n = len(rion)
    K = np.zeros((n, n))
    for ii in range(n):
        for jj in range(n):
            dij = rion[jj] - rion[ii]
            dij = dij - np.trunc(dij / (boxl / 2)) * boxl
            hartree = 0.0 if U == 0.0 else 1.0 / np.sqrt(dij**2 + 1.0 / U**2)
            fxc = alpha / np.sqrt(dij**2 + gamma**2)
            K[ii, jj] = hartree + fxc
    return K


def cis_hamiltonian(C, eps, occ_idx, virt_idx, K_direct, K_exchange):
    """
    Build the singlet CIS/TDA exciton Hamiltonian over a configuration window.

    Args:
        C (ndarray, n x n): ground-state MO coefficients (columns), site basis.
        eps (ndarray, n): MO energies, ascending.
        occ_idx, virt_idx (list of int): occupied / virtual MO indices spanning the
            configuration window (all i in occ_idx paired with all a in virt_idx).
        K_direct (n x n): kernel used in the doubled direct slot -- hartree + fxc.
        K_exchange (n x n): kernel used in the exchange slot -- hartree ONLY (see the
            module docstring for why fxc is excluded here).

    Returns:
        (H, configs): H is the CIS matrix (len(configs) x len(configs)); configs is
        the list of (i, a) MO-index pairs in the same order as H's rows/columns.
    """
    configs = [(i, a) for i in occ_idx for a in virt_idx]
    m = len(configs)

    def Dvec(p, q):
        return C[:, p] * C[:, q]

    H = np.zeros((m, m))
    for I, (i, a) in enumerate(configs):
        for J, (j, b) in enumerate(configs):
            direct = Dvec(i, a) @ K_direct @ Dvec(j, b)      # (ia|jb)
            exch = Dvec(i, j) @ K_exchange @ Dvec(a, b)      # (ij|ab)
            val = 2.0 * direct - exch
            if i == j and a == b:
                val += (eps[a] - eps[i])
            H[I, J] = val
    return H, configs


def validate(n_near=5, nchain=64, dimer1=0.0868, lattice_ang=6.0,
             fxcalpha=-0.1 * ELECTRONVOLT, fxcgamma=0.26 * ANGSTROM, hartreeu=0.115 * ELECTRONVOLT):
    """
    Build H0, the MO basis, and the windowed CIS Hamiltonian for the example/inp.in-like
    geometry, and report the bare gap / lowest exciton eigenvalue / binding energy.
    Defaults use fxcalpha/fxcgamma from example/inp.in and hartreeu from the general
    default in IOmodule.jl (example/inp.in itself sets hartreeu=0, giving a smaller,
    fxc-only binding energy -- pass hartreeu=0.0 to reproduce that case exactly).
    """
    H0, rion, boxl = build_H0(nchain, dimer1, lattice_ang)
    n = H0.shape[0]
    nocc = n // 2
    eps, C = np.linalg.eigh(H0)
    homo, lumo = nocc - 1, nocc
    gap = eps[lumo] - eps[homo]

    K_direct = build_kernel(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exchange = build_kernel(rion, boxl, 0.0, fxcgamma, hartreeu)

    occ_idx = list(range(homo - n_near + 1, homo + 1))
    virt_idx = list(range(lumo, lumo + n_near))
    H, configs = cis_hamiltonian(C, eps, occ_idx, virt_idx, K_direct, K_exchange)
    evals = np.linalg.eigvalsh(H)
    lowest = evals[0]

    print(f"nchain={nchain}, n_near={n_near} (#configs={len(configs)})")
    print(f"bare HOMO-LUMO gap:   {gap / ELECTRONVOLT:.6f} eV")
    print(f"lowest CIS eigenvalue: {lowest / ELECTRONVOLT:.6f} eV")
    print(f"binding energy:        {(gap - lowest) / ELECTRONVOLT:.6f} eV")
    return gap, evals, configs


if __name__ == "__main__":
    validate()
