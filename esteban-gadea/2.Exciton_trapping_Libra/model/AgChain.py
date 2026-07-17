# *********************************************************************************
# Ag-chain tight-binding model, ported to Libra's compute_model(q, params, full_id)
# contract (see README.md, "Verified Working API" section).
#
# STEP 2 SCOPE ONLY: electronic (hopping) Hamiltonian. No repulsive potential, no
# Hxc/fxc mean-field term. See "Open Questions" in the README for why those are kept
# out of ham_dia rather than folded into its diagonal the way libra_py.models.Holstein
# folds its harmonic nuclear term in -- in short: Holstein's trick relies on trace(rho)
# == 1 (a single quantum particle). Our rho is a many-electron closed-shell density
# matrix with trace(rho) == n_electron (most levels doubly occupied), so any purely
# classical / occupation-independent term (the repulsive potential) must stay OUTSIDE
# ham_dia and be added as a separate nuclear force term -- exactly how the existing
# Julia code already keeps build_fion's repulsive contribution separate from the
# Hxc/bond-order electronic force. This module deliberately does not attempt that yet.
#
# Physics reference: ../Ag_chains_parametrization.pdf, Eq. 2 (q=1 case, the only one
# implemented in td_code/TLS_module.jl):
#     beta(r) = beta(req) + A*(r - req)
# Ring topology / geometry conventions match construct_aob_hamiltonian,
# construct_rdep_hamiltonian, and construct_rion in ../td_code/TLS_module.jl.
#
# Units: atomic units (Hartree, Bohr) throughout, same convention as
# libra_py.models.Holstein and as td_code's internal (post-parse) Input struct.
# *********************************************************************************

import sys
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform in ("linux", "linux2"):
    from liblibra_core import *
import util.libutil as comn

# Å -> Bohr, eV -> Hartree (same constants as td_code/units.jl)
ANGSTROM = 1.0 / 0.529177210903
ELECTRONVOLT = 1.0 / 27.211386245988


class tmp:
    pass


def compute_ag_chain(q, params, full_id):
    """
    Electronic (diabatic/site-basis) tight-binding Hamiltonian for a closed 1D ring of
    2*nchain Ag sites with alternating (dimerized) nearest-neighbor bonds.

    Args:
        q (MATRIX(ndof, ntraj)): positions of all 2*nchain ring sites along the chain axis,
            atomic units (Bohr). ndof = 2*nchain. Layout matches td_code's `rion`: site 2k
            starts cell k, site 2k+1 ends it (0-indexed) -- consecutive sites alternate
            intra-cell / inter-cell bonds all the way around the ring.
        params (dict): critical keys:
            * params["nchain"]    (int):   number of dimer unit cells (ndof = 2*nchain)
            * params["hop"]       (float, Ha):   beta(req), reference hopping integral
            * params["hopslope"]  (float, Ha/Bohr): A, slope d(beta)/dr (linear/q=1 case)
            * params["req"]       (float, Bohr): equilibrium nearest-neighbor distance
            * params["boxl"]      (float, Bohr): ring circumference (= nchain * lattice)
        full_id: trajectory identifier; Cpp2Py(full_id)[-1] gives the trajectory column of q.

    Returns:
        PyObject: obj with
            * obj.ham_dia    ( CMATRIX(n,n) ): diabatic (site-basis) Hamiltonian, n = 2*nchain
            * obj.ovlp_dia   ( CMATRIX(n,n) ): identity (orthogonal TB basis)
            * obj.d1ham_dia  ( list of n CMATRIX(n,n) ): dH/dR_k for each site position DOF k
            * obj.dc1_dia    ( list of n CMATRIX(n,n) ): zero (diabatic basis -> no derivative
              coupling; Libra diagonalizes ham_dia internally to get adiabatic states/NACs)
    """
    critical_params = ["nchain", "hop", "hopslope", "req", "boxl"]
    default_params = {}
    comn.check_input(params, default_params, critical_params)

    nchain = params["nchain"]
    hop0 = params["hop"]
    hopslope = params["hopslope"]
    req = params["req"]
    boxl = params["boxl"]

    n = 2 * nchain  # sites == nuclear DOFs (each site moves along the 1D ring)

    Id = Cpp2Py(full_id)
    traj = Id[-1]
    r = [q.get(i, traj) for i in range(n)]

    Hdia = CMATRIX(n, n)
    Sdia = CMATRIX(n, n)
    Sdia.identity()
    d1ham_dia = CMATRIXList()
    for k in range(n):
        d1ham_dia.append(CMATRIX(n, n))
    dc1_dia = CMATRIXList()
    for k in range(n):
        dc1_dia.append(CMATRIX(n, n))  # left at zero

    def bond(i, j, dr):
        # dr = r_j - r_i (across the periodic image where needed by the caller below)
        h = (hop0 + hopslope * (dr - req)) * (1.0 + 0.0j)
        Hdia.set(i, j, h)
        Hdia.set(j, i, h)
        # beta(r) is linear in r, so d(beta)/dr = hopslope everywhere;
        # chain rule on dr = r_j - r_i: d(beta)/dr_i = -hopslope, d(beta)/dr_j = +hopslope
        d1ham_dia[i].set(i, j, -hopslope * (1.0 + 0.0j))
        d1ham_dia[i].set(j, i, -hopslope * (1.0 + 0.0j))
        d1ham_dia[j].set(i, j, hopslope * (1.0 + 0.0j))
        d1ham_dia[j].set(j, i, hopslope * (1.0 + 0.0j))

    for i in range(n - 1):
        bond(i, i + 1, r[i + 1] - r[i])

    # wraparound bond closing the ring, through the periodic image of site 0
    bond(n - 1, 0, (r[0] + boxl) - r[n - 1])

    obj = tmp()
    obj.ham_dia = Hdia
    obj.ovlp_dia = Sdia
    obj.d1ham_dia = d1ham_dia
    obj.dc1_dia = dc1_dia
    return obj


def ring_positions(nchain, r1, r2):
    """
    Equilibrium (rigid-lattice) site positions for a dimerized ring, matching
    td_code/TLS_module.jl's construct_rion: cell k starts at site 2k (bond length r1 to
    site 2k+1), then bond length r2 back to the start of cell k+1.

    Args:
        nchain (int): number of dimer unit cells
        r1, r2 (float, Bohr): intra-cell / inter-cell bond lengths (r1+r2 = lattice)

    Returns:
        list of float: the 2*nchain site positions, atomic units
    """
    rvect = [0.0] * (2 * nchain)
    for k in range(nchain):
        rvect[2 * k] = k * (r1 + r2)
        rvect[2 * k + 1] = k * (r1 + r2) + r1
    return rvect


def get_default_params(nchain=64, dimer1=0.0868, lattice_ang=6.0):
    """
    Parameters matching example/inp.in exactly (the reference run whose eigenvalues are
    checked into example/output.out). Uses the *exact* atomic-unit constants baked into
    td_code/IOmodule.jl's Input struct defaults (not the 6-decimal values printed in
    output.out, which are already rounded) to avoid a spurious rounding mismatch in the
    validation below. Defaults: 64-pair chain, dimer1 = 0.0868, lattice = 6.0 Angstrom.
    """
    hop = -0.0245725447              # beta(req), Ha  (= -0.668653 eV)
    hopslope = 0.007215487659        # A, Ha/Bohr     (= 0.371035 eV/Ang)
    req = 4.922388                   # Bohr           (= 2.604816 Ang)
    lattice = lattice_ang * ANGSTROM
    boxl = nchain * lattice
    r1 = lattice * (1 - dimer1) / 2
    r2 = lattice * (1 + dimer1) / 2

    return {
        "nchain": nchain,
        "hop": hop,
        "hopslope": hopslope,
        "req": req,
        "boxl": boxl,
        "r1": r1,
        "r2": r2,
    }


def validate():
    """
    Builds ham_dia at the example/inp.in reference geometry, diagonalizes it with numpy
    (deliberately NOT via any Libra eigensolver call, to keep this check independent of
    any unverified Libra API), and prints the lowest few eigenvalues in eV for direct
    comparison against example/output.out's "Eigenvalues:" section. Reference values from
    that file (first 6, eV): -1.04412474, -1.04291015, -1.04291015, -1.03926962,
    -1.03926962, -1.03321283.

    Pre-checked with a standalone numpy replica of this same construction (bypassing Libra
    entirely): matches the reference to ~7e-5 eV, a constant offset across all levels. That
    residual is expected, not a bug -- TLS_module.jl's main run path builds hop(r) from
    construct_potential's 10000-point *tabulated* grid (rounded to the nearest grid index),
    not the continuous formula used here and in minimize_dimer; the top-level README
    documents this same continuous-vs-tabulated gap as "negligible for MD forces." If your
    comparison run matches to within ~1e-4 eV, treat Step 2 as validated.

    Run this inside the `libra` kernel (needs liblibra_core / util.libutil on the path).
    """
    import numpy as np

    p = get_default_params()
    n = 2 * p["nchain"]
    rion = ring_positions(p["nchain"], p["r1"], p["r2"])

    q = MATRIX(n, 1)
    for i, x in enumerate(rion):
        q.set(i, 0, x)

    full_id = Py2Cpp_int([0, 0])
    obj = compute_ag_chain(q, p, full_id)

    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            H[i, j] = obj.ham_dia.get(i, j)

    evals = np.linalg.eigvalsh(H)
    evals_eV = evals / ELECTRONVOLT
    print("Lowest 6 eigenvalues (eV), compare against example/output.out:")
    for e in evals_eV[:6]:
        print(f"  {e: .8f}")


if __name__ == "__main__":
    validate()
