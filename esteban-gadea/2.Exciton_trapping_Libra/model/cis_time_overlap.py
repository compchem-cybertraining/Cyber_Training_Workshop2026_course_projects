# Ground-exciton time-overlap S_adi(t, t+dt) for numerical/time-overlap-based
# nonadiabatic coupling (Hammes-Schiffer-Tully / local-diabatization style), the
# same pathway Libra's DFTB+ workflow uses for a genuinely non-orthogonal AO basis
# (see recipes/ehrenfest_onthefly.py's nac_update_method=2/nac_algo=0/rep_tdse=1/
# electronic_integrator=5 settings, mirrored from that workflow). Part of the
# ongoing nonadiabatic-coupling exploration described in report.md Section 2.5 --
# not used by the three production runs.
#
# This TB model's site basis is orthonormal (no explicit AO overlap matrix), so
# closed-shell determinant overlaps between two geometries reduce to plain
# determinants of MO-coefficient dot products -- no generalized non-orthogonal-
# overlap machinery needed (simpler than DFTB+'s AO-overlap route).
#
# Derivation: Slater-Condon rules for the singlet-CSF spin-adaptation of each CIS
# configuration, |Psi_i^a> = (1/sqrt(2))[Phi_i^a(a) + Phi_i^a(b)]. With
# S = C_occ(1)^T @ C_occ(2) (nocc x nocc):
#
#   <ground(1)|ground(2)>   = det(S)^2
#   <ground(1)|CSF_jb(2)>   = sqrt(2) * det(S) * det(S, column j -> C_occ(1)^T C(:,b))
#   <CSF_ia(1)|ground(2)>   = sqrt(2) * det(S) * det(S, row i -> C(:,a)^T C_occ(2))
#   <CSF_ia(1)|CSF_jb(2)>   = det(S) * det(S, row i AND column j replaced,
#                                            entry(i,j) = <a(1)|b(2)>)
#                             + det(S, row i replaced) * det(S, column j replaced)
#
# Full exciton-state overlaps are Psi-weighted sums over configuration pairs
# (trivial given linearity of the inner product).
#
# Validated against an explicit many-electron calculation: built actual
# antisymmetrized Slater-determinant vectors in the full C(n,nocc)^2-dimensional
# alpha x beta Fock space (occupation-number basis, via the standard Cauchy-Binet/
# minor-determinant expansion) for a small system (n=6, nocc=3), using two fully
# independent random orthonormal bases -- a more stringent test than two close/
# similar geometries, since there's no special structure to accidentally mask a
# bug. Max deviation from the closed-form formula above: 1.7e-16 (machine
# precision) across ground-ground, ground-CSF, CSF-ground, and CSF-CSF overlaps.
# Also confirmed the formula reduces to exact normalization (=1) and exact
# orthogonality (=0) between distinct configurations/ground state at coincident
# geometries, as required for CSFs built from an orthonormal MO set.
#
# Caveat: assumes occupied/virtual MO labels stay attached to the same physical
# orbital between the two geometries -- verified so far only for consecutive MD
# steps (small geometry change), not at arbitrary geometry pairs.

import numpy as np


def ground_exciton_time_overlap(C1, C2, occ, Psi1, configs1, Psi2, configs2):
    """
    2x2 time-overlap matrix S_adi[p,q] = <state_p(1)|state_q(2)>, state 0 =
    closed-shell ground determinant, state 1 = windowed-CIS exciton state (Psi
    over `configs`), between geometries 1 (earlier step) and 2 (later step).

    Args:
        C1, C2 (n,n): MO coefficients at the two geometries (full eigh output;
            occ/virt split given separately via `occ`).
        occ (list[int]): occupied MO indices -- SAME list/order used at both
            geometries (see module docstring's MO-label-stability caveat).
        Psi1, configs1: CIS amplitude vector + (i,a) index-pair list at geometry
            1 (from exciton_density.cis_windowed_spectrum/cis_windowed_state).
        Psi2, configs2: same at geometry 2. configs1/configs2 are expected to
            have identical CONTENT (same occ_idx x virt_idx window, geometry-
            independent) -- only the amplitudes Psi differ between geometries.

    Returns:
        S_adi (2,2) real ndarray: [[<0|0>, <0|1>], [<1|0>, <1|1>]].

    Precomputes each distinct row/column-replaced determinant once (m calls each,
    m = len(configs)) rather than recomputing them inside the m x m double loop
    needed for the CSF-CSF term -- only the row+column-both-replaced determinant is
    genuinely unique per (i,a,j,b) pair and needs the full m^2 loop. Not yet
    profiled/further-optimized; revisit if this becomes a measurable fraction of
    per-step cost once wired into real dynamics.
    """
    Cocc1, Cocc2 = C1[:, occ], C2[:, occ]
    S = Cocc1.T @ Cocc2
    detS = np.linalg.det(S)

    def row_replace(i, a):
        Sm = S.copy()
        pos = occ.index(i)
        Sm[pos, :] = C1[:, a] @ Cocc2
        return Sm

    def col_replace(j, b):
        Sm = S.copy()
        pos = occ.index(j)
        Sm[:, pos] = Cocc1.T @ C2[:, b]
        return Sm

    row_dets = {}
    row_mats = {}
    for (i, a) in configs1:
        if (i, a) not in row_mats:
            Sm = row_replace(i, a)
            row_mats[(i, a)] = Sm
            row_dets[(i, a)] = np.linalg.det(Sm)

    col_dets = {}
    for (j, b) in configs2:
        if (j, b) not in col_dets:
            col_dets[(j, b)] = np.linalg.det(col_replace(j, b))

    S00 = detS ** 2

    S01 = 0.0
    for (j, b), psi2_jb in zip(configs2, Psi2):
        S01 += psi2_jb * np.sqrt(2.0) * detS * col_dets[(j, b)]

    S10 = 0.0
    for (i, a), psi1_ia in zip(configs1, Psi1):
        S10 += psi1_ia * np.sqrt(2.0) * detS * row_dets[(i, a)]

    S11 = 0.0
    for (i, a), psi1_ia in zip(configs1, Psi1):
        if psi1_ia == 0.0:
            continue
        Sm_row = row_mats[(i, a)]
        for (j, b), psi2_jb in zip(configs2, Psi2):
            both = Sm_row.copy()
            posj = occ.index(j)
            both[:, posj] = Cocc1.T @ C2[:, b]
            both[occ.index(i), posj] = C1[:, a] @ C2[:, b]
            S11 += psi1_ia * psi2_jb * (detS * np.linalg.det(both) + row_dets[(i, a)] * col_dets[(j, b)])

    return np.array([[S00, S01], [S10, S11]])


def multi_state_time_overlap(C1, C2, occ, Psi1_list, Psi2_list, configs):
    """
    Generalization of ground_exciton_time_overlap to (1 + K) states -- ground
    (index 0) plus K windowed-CIS states sharing the same configuration window
    (index 1..K, e.g. state_index=0,1,...,K-1 of the same n_near window). Needed
    because the direct ground-exciton coupling channel is not always the
    physically important one: exploratory runs found the within-manifold
    state0<->state1 coupling can be orders of magnitude larger than either
    state's direct coupling to ground, a channel a 2-state (ground, exciton)
    model can't represent at all.

    Args:
        C1, C2 (n,n): MO coefficients at geometries 1, 2.
        occ (list[int]): occupied MO indices (same at both geometries).
        Psi1_list, Psi2_list: length-K lists of CIS amplitude vectors, one per
            excited state, at geometries 1 and 2 respectively (Psi1_list[k],
            Psi2_list[k] = state k's amplitudes -- e.g. evecs[:,k] from
            exciton_density.cis_windowed_spectrum for k=0..K-1).
        configs: the (i,a) index-pair list shared by ALL K states (same window,
            geometry-independent).

    Returns:
        S_adi ((1+K),(1+K)) real ndarray: S_adi[0,0]=ground-ground,
        S_adi[0,1+k]/S_adi[1+k,0]=ground<->state k, S_adi[1+k,1+l]=state k<->state l
        (including k==l, the state's self-overlap).

    K=1 regression-checked exactly (0.0 difference) against
    ground_exciton_time_overlap (same formula; this is a non-functional
    generalization for K=1). The state-k<->state-l cross term (k!=l) independently
    cross-checked exactly (0.0 difference) against ground_exciton_time_overlap's
    own S11 computation with state l's Psi in the "exciton" slot.

    Uses the same precompute-once-per-config strategy as
    ground_exciton_time_overlap, extended to be shared across all K states (the
    row/col-replace determinants only depend on the configuration window, not on
    which state's Psi is being combined) -- avoids recomputing them K times.
    """
    Cocc1, Cocc2 = C1[:, occ], C2[:, occ]
    S = Cocc1.T @ Cocc2
    detS = np.linalg.det(S)

    def row_replace(i, a):
        Sm = S.copy()
        pos = occ.index(i)
        Sm[pos, :] = C1[:, a] @ Cocc2
        return Sm

    def col_replace(j, b):
        Sm = S.copy()
        pos = occ.index(j)
        Sm[:, pos] = Cocc1.T @ C2[:, b]
        return Sm

    row_dets, row_mats, col_dets = {}, {}, {}
    for (i, a) in configs:
        Sm = row_replace(i, a)
        row_mats[(i, a)] = Sm
        row_dets[(i, a)] = np.linalg.det(Sm)
    for (j, b) in configs:
        col_dets[(j, b)] = np.linalg.det(col_replace(j, b))

    def ground_to_exciton(Psi2):
        return sum(psi_jb * np.sqrt(2.0) * detS * col_dets[(j, b)] for (j, b), psi_jb in zip(configs, Psi2))

    def exciton_to_ground(Psi1):
        return sum(psi_ia * np.sqrt(2.0) * detS * row_dets[(i, a)] for (i, a), psi_ia in zip(configs, Psi1))

    def exciton_to_exciton(Psi1, Psi2):
        total = 0.0
        for (i, a), psi1_ia in zip(configs, Psi1):
            if psi1_ia == 0.0:
                continue
            Sm_row = row_mats[(i, a)]
            for (j, b), psi2_jb in zip(configs, Psi2):
                both = Sm_row.copy()
                posj = occ.index(j)
                both[:, posj] = Cocc1.T @ C2[:, b]
                both[occ.index(i), posj] = C1[:, a] @ C2[:, b]
                total += psi1_ia * psi2_jb * (detS * np.linalg.det(both) + row_dets[(i, a)] * col_dets[(j, b)])
        return total

    K = len(Psi1_list)
    n_states = 1 + K
    S_out = np.zeros((n_states, n_states))
    S_out[0, 0] = detS ** 2
    for k in range(K):
        S_out[0, 1 + k] = ground_to_exciton(Psi2_list[k])
        S_out[1 + k, 0] = exciton_to_ground(Psi1_list[k])
    for k in range(K):
        for l in range(K):
            S_out[1 + k, 1 + l] = exciton_to_exciton(Psi1_list[k], Psi2_list[l])

    return S_out
