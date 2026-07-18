# Real-space exciton character from the CIS wavefunction -- the direct diagnostic
# for "is there actually an exciton localizing here, and where."
#
# For a CIS/TDA state Psi = sum_(i,a) Psi_(i,a) |i->a>, the standard "where is the
# hole/where is the excited electron" real-space marginal densities are obtained by
# marginalizing |Psi_(i,a)|^2 over the partner index and projecting onto real space
# via the MO coefficients:
#   rho_hole(site n) = sum_(i,a) |Psi_(i,a)|^2 * |C[n,i]|^2
#   rho_elec(site n) = sum_(i,a) |Psi_(i,a)|^2 * |C[n,a]|^2
# Both are individually normalized to 1 (sum_n rho_hole(n) = sum_n rho_elec(n) = 1)
# since sum_(i,a)|Psi_(i,a)|^2 = 1 (Psi normalized) and each MO |C[:,p]|^2 is
# itself normalized.
#
# This is a post-processing-only module -- never called during the MD itself
# (dc1_adi/hvib_adi/etc. don't need it), only from analysis scripts that recompute
# the CIS state at a handful of saved geometries from a completed trajectory. No
# performance optimization applied here on purpose: this runs at most a few
# thousand times on small matrices, nowhere near the per-MD-step hot path in
# cis_compute_adi.py.
#
# Validated: cis_windowed_state's E_n matches
# cis_compute_adi.cis_windowed_energy_and_gradient's E_n exactly (same Hm
# construction, just also returning Psi/configs/eps/C instead of only the
# gradient). rho_hole/rho_elec confirmed normalized to 1. IPR (inverse
# participation ratio) confirmed = n exactly at the perfectly symmetric
# equilibrium geometry (fully delocalized) and drops under a jittered geometry
# (responds sensibly to broken symmetry).

import numpy as np
from cis_compute_adi import H0_and_dH0
from cis_gradient import kernel_with_gradient


def cis_windowed_spectrum(rion, boxl, fxcalpha, fxcgamma, hartreeu, n_near,
                           hop=-0.0245725447, hopslope=0.007215487659, req=4.922388):
    """
    Recompute the full windowed-CIS eigenvalue/eigenvector spectrum (all m
    states, not just one) at an arbitrary geometry. Same Hm construction as
    cis_compute_adi.cis_windowed_energy_and_gradient / cis_windowed_state below,
    factored out so callers that need more than state_index=0 (e.g. a
    state-crossing/gap diagnostic) don't have to duplicate the Hm build.

    Returns:
        evals (m,), evecs (m, m) [evecs[:, k] is the k-th eigenvector], configs
        (list of (i,a) pairs matching the amplitude-vector ordering), eps (n,)
        MO eigenvalues, C (n,n) MO coefficients.
    """
    H0, _ = H0_and_dH0(rion, boxl, hop, hopslope, req)
    eps, C = np.linalg.eigh(H0)
    n = H0.shape[0]
    nocc = n // 2
    homo, lumo = nocc - 1, nocc

    occ_idx = list(range(homo - n_near + 1, homo + 1))
    virt_idx = list(range(lumo, lumo + n_near))
    configs = [(i, a) for i in occ_idx for a in virt_idx]
    m = len(configs)

    K, _ = kernel_with_gradient(rion, boxl, fxcalpha, fxcgamma, hartreeu)
    K_exch, _ = kernel_with_gradient(rion, boxl, 0.0, fxcgamma, hartreeu)

    def Dvec(p_, q_):
        return C[:, p_] * C[:, q_]

    Hm = np.zeros((m, m))
    for I, (i, a) in enumerate(configs):
        for J, (j, b) in enumerate(configs):
            direct = Dvec(i, a) @ K @ Dvec(j, b)
            exch = Dvec(i, j) @ K_exch @ Dvec(a, b)
            val = 2.0 * direct - exch
            if i == j and a == b:
                val += (eps[a] - eps[i])
            Hm[I, J] = val
    evals, evecs = np.linalg.eigh(Hm)
    return evals, evecs, configs, eps, C


def cis_windowed_state(rion, boxl, fxcalpha, fxcgamma, hartreeu, n_near,
                        hop=-0.0245725447, hopslope=0.007215487659, req=4.922388,
                        state_index=0):
    """
    Recompute the CIS eigenstate (energy + amplitude vector Psi over the
    configuration window -- NOT the gradient) at an arbitrary geometry. Thin
    wrapper over cis_windowed_spectrum (unchanged numerics/output for existing
    callers -- same Hm, same eigh call, just picks out one state_index).

    Returns:
        E_n (Ha), Psi (m,) CIS amplitude vector (m = (2*n_near)^2 configs),
        configs (list of (i,a) index pairs matching Psi's ordering), eps (n,) MO
        eigenvalues, C (n,n) MO coefficient matrix -- everything needed to build
        a real-space density via electron_hole_density() below.
    """
    evals, evecs, configs, eps, C = cis_windowed_spectrum(
        rion, boxl, fxcalpha, fxcgamma, hartreeu, n_near, hop, hopslope, req)
    return evals[state_index], evecs[:, state_index], configs, eps, C


def electron_hole_density(Psi, configs, C):
    """Real-space hole/electron marginal densities from CIS amplitudes (see
    module docstring for the formula). Returns (rho_hole, rho_elec), each shape
    (n,), each individually normalized to sum to 1."""
    n = C.shape[0]
    rho_hole = np.zeros(n)
    rho_elec = np.zeros(n)
    for (i, a), psi_amp in zip(configs, Psi):
        w = psi_amp ** 2
        rho_hole += w * C[:, i] ** 2
        rho_elec += w * C[:, a] ** 2
    return rho_hole, rho_elec


def ipr(rho):
    """Inverse participation ratio: 1/sum(rho^2). Smaller = more localized
    (IPR=1 means density entirely on one site); larger = more delocalized
    (IPR=n means perfectly uniform over n sites). Confirmed IPR=n exactly at the
    symmetric equilibrium geometry (fully delocalized exciton)."""
    return 1.0 / np.sum(rho ** 2)
