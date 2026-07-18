"""Trajectory-surface-hopping dynamics on the frozen NQDH through Libra.

Method presets differ ONLY in the "energy gate" and decoherence knobs -- the
controlled comparison behind the report's gate experiment:

  fssh   classic FSSH: hops accepted only if the kinetic energy along the
         derivative coupling can pay the gap (hop_acceptance 20), momentum
         rescaled along that direction (201).  The gate.
  qtsh   quantum-trajectory surface hopping (Martens): hops always accepted,
         no momentum jump; energy bookkeeping via a coherence-weighted
         nonclassical force.  The gate removed.
  shxf   FSSH + exact-factorization decoherence (aux trajectories).  The gate
         kept, decoherence added.

Requires the ``libra`` conda environment (liblibra_core).  All quantities in
atomic units.  Populations are saved from both estimators: ``sh`` (fraction of
trajectories on each active surface -- the physical one for hopping methods)
and ``se`` (coherent electronic populations).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRESETS = {
    "fssh": {"tsh_method": 0, "use_qtsh": 0,
             "hop_acceptance_algo": 20, "momenta_rescaling_algo": 201},
    "qtsh": {"tsh_method": 0, "use_qtsh": 1, "qtsh_force_option": 1,
             "hop_acceptance_algo": 0, "momenta_rescaling_algo": 0},
    "shxf": {"tsh_method": 0, "use_qtsh": 0, "decoherence_algo": 5,
             "hop_acceptance_algo": 20, "momenta_rescaling_algo": 201},
}

# XF aux-wavepacket width parameters (bohr^-2): standard AIMS frozen-Gaussian
# alphas for H/C/N/O; other elements fall back to 20.0 (tunable).
AIMS_ALPHA = {1: 4.7, 6: 22.7, 7: 19.0, 8: 12.2}


def run_tsh(model_dir, ics_path, out_dir, method="qtsh", ic_index=0, ntraj=8,
            n_steps=2000, dt=4.0, istate=None, device="cpu", seed=1):
    """Propagate one initial condition with `ntraj` hop-RNG replicas.

    Returns (time_au, sh_pop_adi, se_pop_adi) and writes tsh_populations.npz
    in ``out_dir``.
    """
    import h5py
    from liblibra_core import MATRIX, Random
    import libra_py.dynamics.tsh.compute as tsh

    from .libra_adapter import make_nqdh_compute_model

    ics = np.load(ics_path)
    Z = np.asarray(ics["Z"], np.int64)
    R0 = np.asarray(ics["R_bohr"], float)[ic_index].reshape(-1)
    V0 = np.asarray(ics["V_au"], float)[ic_index].reshape(-1)
    m = np.repeat(np.asarray(ics["masses_au"], float), 3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / "libra_out")

    compute_model, K, ndof = make_nqdh_compute_model(model_dir, Z, device=device)
    if istate is None:
        istate = K - 1                                  # photoexcite the top adiabatic state

    dyn_params = {
        "rep_tdse": 1, "force_method": 1, "isNBRA": 0,
        "ham_update_method": 1, "ham_transform_method": 1, "time_overlap_method": 1,
        "ntraj": ntraj, "dt": dt, "nsteps": n_steps, "num_electronic_substeps": 20,
        "prefix": prefix, "prefix2": prefix + "_2",
        "mem_output_level": 3, "hdf5_output_level": -1, "txt_output_level": -1,
        "properties_to_save": ["timestep", "time", "se_pop_adi", "sh_pop_adi", "Etot_ave"],
        "progress_frequency": 0.1,
    }
    dyn_params.update(PRESETS[method])
    if method == "shxf":
        sig = np.repeat(np.array([1.0 / np.sqrt(2.0 * AIMS_ALPHA.get(int(z), 20.0)) for z in Z]), 3)
        wp = MATRIX(ndof, 1)
        for i, s in enumerate(sig):
            wp.set(i, 0, float(s))
        dyn_params["wp_width"] = wp

    model_params = {"model0": 0, "nstates": K}
    init_nucl = {"init_type": 0, "ndof": ndof, "q": list(R0), "p": list(m * V0), "mass": list(m)}
    init_elec = {"ndia": K, "nadi": K, "rep": 1, "init_type": 0, "istate": istate, "verbosity": 0}

    rng = Random()
    tsh.generic_recipe(dyn_params, compute_model, model_params, init_elec, init_nucl, rng)

    with h5py.File(prefix + "/mem_data.hdf", "r") as f:
        t = np.array(f["time/data"]).ravel()
        se = np.array(f["se_pop_adi/data"])
        sh = np.array(f["sh_pop_adi/data"])
    np.savez(out_dir / "tsh_populations.npz", time_au=t, sh_pop_adi=sh, se_pop_adi=se,
             method=method, ic_index=ic_index, ntraj=ntraj, istate=istate)
    return t, sh, se
