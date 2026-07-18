# Ehrenfest recipe compatible with on-the-fly compute_model contracts
# (ham_update_method=2, our cis_compute_adi.py / Libra's own dftb_compute_adi),
# NOT the built-in tutorial `ehrenfest_adi_nac` / `ehrenfest_adi_ld` recipes.
#
# Why not reuse the built-in ehrenfest_adi_* recipes: those ship in Libra's
# `recipes` package alongside fssh/gfsh/etc. and are exercised in Libra's own
# tutorials exclusively against model Hamiltonians (Holstein, Esch-Levine, Morse)
# that return `ham_dia` for Libra to diagonalize internally -- i.e. they're
# written assuming the ham_update_method=1 (diabatic) contract, not the
# ham_update_method=2 (adiabatic, on-the-fly) contract cis_compute_adi.py uses.
#
# Instead, this module is `recipes/fssh2.py` (Libra's real, on-the-fly-validated
# recipe, paired with dftb_compute_adi) with only the two settings changed that
# distinguish "run FSSH2 surface hopping" from "run plain mean-field Ehrenfest,
# no hops":
#     tsh_method:    7 -> -1   (adiabatic dynamics, no hops)
#     force_method:  1 -> 2    (Ehrenfest/MMST mean-field force, not state-specific)
# Every other flag (ham_update_method=2, ham_transform_method=0,
# time_overlap_method=0, nac_update_method=2, hvib_update_method=1, rep_force=1,
# rep_tdse=1, electronic_integrator=5, state_tracking_algo=21,
# do_phase_correction=1, decoherence_algo=-1 already off) is preserved verbatim,
# since those are what make the recipe compatible with an on-the-fly
# compute_model, independent of the hopping/force choice.
#
# Caveat: with dc1_adi identically zero (cis_compute_adi's documented fallback --
# no NAC pathway in the base 2-state, no-coupling model), the Ehrenfest mean-field
# force reduces to a population-weighted average of the two decoupled PES
# gradients, and coherent population transfer cannot occur (Hvib is exactly
# diagonal). Starting purely in state 1 (istates=[0,1]) means the nuclei simply
# feel state 1's (ground+exciton) force the whole trajectory -- exactly what the
# three production runs need (does state 1's PES self-trap the lattice
# distortion?), but not a test of exciton<->ground coherent/mixed dynamics. That
# requires a real coupling term (see model/cis_time_overlap.py, report.md
# Section 2.5).


def load(dyn_general):

    # ====== How to update Hamiltonian ======
    # 2: recompute only adiabatic Hamiltonian, use with file-based or on-the-fly
    #    workflows -- our cis_compute_adi.py returns ham_adi directly.
    dyn_general.update({"ham_update_method": 2})

    # ====== How to transform between representations ======
    # 0: don't do any transforms -- on-the-fly workflow, don't override the
    #    adiabatic values cis_compute_adi.py already computed.
    dyn_general.update({"ham_transform_method": 0})

    # ====== Time-overlaps ======
    # 0: don't recompute -- cis_compute_adi.py already sets time_overlap_adi
    #    (identity, see its docstring for the state-tracking caveat this implies).
    dyn_general.update({"time_overlap_method": 0})

    # ====== NAC update ======
    # 2: update according to time-overlaps (time-derivative NACs). With
    #    time_overlap_adi = identity, this correctly yields zero NAC, matching
    #    cis_compute_adi's dc1_adi = 0.
    dyn_general.update({"nac_update_method": 2})
    dyn_general.update({"nac_algo": 0})

    # ====== Vibronic Hamiltonian ======
    # 1: Hvib = Ham - i*hbar*NAC (regular formula); with NAC=0 this reduces to
    #    Hvib = Ham, matching cis_compute_adi's obj.hvib_adi = obj.ham_adi.
    dyn_general.update({"hvib_update_method": 1})

    dyn_general.update({"do_ssy": 0})

    # ============ FORCES: Ehrenfest, not state-specific ============
    # rep_force=1 (adiabatic): matches ham_adi/d1ham_adi being what
    # cis_compute_adi.py actually provides.
    dyn_general.update({"rep_force": 1})
    # force_method=2: Ehrenfest/MMST mean-field force (changed from fssh2.py's
    # force_method=1/state-specific -- this is what makes it Ehrenfest dynamics).
    dyn_general.update({"force_method": 2})
    dyn_general.update({"qtsh_force_option": 0})
    dyn_general.update({"use_xf_force": 0})

    # ============ SURFACE HOPPING: off ============
    # tsh_method=-1: adiabatic dynamics, no hops (CHANGED from fssh2.py's
    # tsh_method=7/FSSH2 -- this is what makes this Ehrenfest, not FSSH2).
    dyn_general.update({"tsh_method": -1})
    dyn_general.update({"use_qtsh": 0})
    dyn_general.update({"hop_acceptance_algo": 0})     # no hops proposed anyway
    dyn_general.update({"momenta_rescaling_algo": 0})  # no hops proposed anyway
    dyn_general.update({"use_Jasper_Truhlar_criterion": 0})

    # ============ DECOHERENCE: off (plain Ehrenfest) ============
    dyn_general.update({"decoherence_algo": -1})
    dyn_general.update({"instantaneous_decoherence_variant": 1})
    dyn_general.update({"decoherence_times_type": -1})
    dyn_general.update({"decoherence_C_param": 1.0, "decoherence_eps_param": 0.1})

    from liblibra_core import MATRIX
    A = MATRIX(1, 1); A.set(0, 0, 1.0)
    dyn_general.update({"schwartz_decoherence_inv_alpha": A})
    B = MATRIX(1, 1); B.set(0, 0, 1.0)
    dyn_general.update({"schwartz_interaction_width": B})
    dyn_general.update({"reorg_eergy": 0.0})
    dyn_general.update({"dephasing_informed": 0})
    dyn_general.update({"decoherence_rates": MATRIX(2, 2), "ave_gaps": MATRIX(2, 2)})

    # ============ INTEGRATORS / TRACKING / PHASES ============
    # rep_tdse=1 (adiabatic), electronic_integrator=5: same LD-based integrator
    # fssh2.py uses -- appropriate regardless of hopping/force choice.
    dyn_general.update({"rep_tdse": 1, "electronic_integrator": 5})
    dyn_general.update({"state_tracking_algo": 21})
    dyn_general.update({"do_phase_correction": 1})
    dyn_general.update({"do_nac_phase_correction": 0})
    dyn_general.update({"assume_always_consistent": 0})
