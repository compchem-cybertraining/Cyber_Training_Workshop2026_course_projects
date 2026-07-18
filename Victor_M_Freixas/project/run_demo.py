#!/usr/bin/env python
"""Run the frozen NQDH model as diabatic Ehrenfest dynamics and plot the S1/S2 beat.

The trained Neural Quasi-Diabatic Hamiltonian outputs a smooth symmetric matrix
W(R); the dynamics move nuclei under the mean-field force F = -Tr(rho dW/dR) while
electrons evolve under i c_dot = W c (diabatic basis -> no 1/dE divergence at the
S1/S2 near-degeneracy). Diagonalising W along the trajectory gives the adiabatic
S1/S2 populations; averaging over the thermal ensemble reveals the vibronic beat.

Two engines consume the *identical* model W, dW/dR (they agree to ~1e-5 on pops):

  --engine builtin   pure-numpy reference engine (default; needs only torch+hippynn)
  --engine libra     Libra's tsh.generic_recipe via the compute_model adapter
                     (the school software; run inside the `libra` conda env)

Examples
--------
    # quick smoke test: one trajectory, ~1 min, no Libra needed
    python run_demo.py

    # one trajectory through Libra (the school-software path)
    python run_demo.py --engine libra

    # reproduce the full 300-trajectory beat (heavy; ~1 h, GPU recommended)
    python run_demo.py --ensemble

The precomputed full result ships in expected_output/ for comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AMU_TO_AU = 1822.888486209
AUTIME_FS = 2.41888e-2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["builtin", "libra"], default="builtin",
                   help="dynamics engine (default builtin = no Libra needed)")
    p.add_argument("--model", type=Path, default=HERE / "model", help="frozen NQDH run dir")
    p.add_argument("--ics", type=Path, default=HERE / "data" / "ics.npz", help="initial conditions")
    p.add_argument("--n-traj", type=int, default=1, help="number of trajectories (ensemble)")
    p.add_argument("--n-steps", type=int, default=1000, help="nuclear steps (dt=4 au ~ 0.097 fs)")
    p.add_argument("--dt", type=float, default=4.0)
    p.add_argument("--ensemble", action="store_true", help="shortcut for --n-traj 300 --n-steps 2000")
    p.add_argument("--out", type=Path, default=HERE / "output", help="where to write results")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def top_adiabatic_c0(W: np.ndarray) -> np.ndarray:
    """Diabatic amplitudes of the highest adiabatic state (S2 for K=3) = the bright IC."""
    _lam, U = np.linalg.eigh(W)
    return U[..., -1].astype(complex)


def run_builtin(prov, Z, R0, V0, masses, dt, n_steps):
    from nqdh_demo.ehrenfest import run_ehrenfest_ensemble
    W0, _ = prov.evaluate_diabatic_batch(Z, R0)
    c0 = top_adiabatic_c0(W0)                                     # (B, K)
    res = run_ehrenfest_ensemble(prov.evaluate_diabatic_batch, Z, R0, V0, masses,
                                 c0, dt, n_steps, remove_com=True, stride=5,
                                 progress=lambda s: None)
    return res.time_fs, res.adiabatic_pops, res.energy_drift_eV()


def run_libra(prov, Z, R0, V0, masses, dt, n_steps):
    """Loop the validated single-trajectory Libra path over the ensemble."""
    import h5py
    from liblibra_core import Random
    import libra_py.dynamics.tsh.compute as tsh
    from nqdh_demo.libra_adapter import make_nqdh_compute_model

    compute_model, K, ndof = make_nqdh_compute_model(str(prov.run_dir), Z, device="cpu")
    B = len(R0)
    pops = None
    drift = []
    for b in range(B):
        c0 = top_adiabatic_c0(prov.evaluate_diabatic(Z, R0[b]).W)
        init_nucl = {"init_type": 0, "ndof": ndof, "q": list(R0[b].reshape(-1)),
                     "p": list((masses[:, None] * V0[b]).reshape(-1)),
                     "mass": list(np.repeat(masses, 3))}
        init_elec = {"ndia": K, "nadi": K, "rep": 1, "init_type": 0, "istate": K - 1, "verbosity": 0}
        prefix = f"/tmp/nqdh_libra_{b}"
        dyn_params = {
            "rep_tdse": 0, "force_method": 2, "tsh_method": -1, "isNBRA": 0,
            "ham_update_method": 1, "ham_transform_method": 1, "time_overlap_method": 1,
            "ntraj": 1, "dt": dt, "nsteps": n_steps, "num_electronic_substeps": 20,
            "prefix": prefix, "prefix2": prefix + "_2",
            "mem_output_level": 3, "hdf5_output_level": -1, "txt_output_level": -1,
            "properties_to_save": ["timestep", "time", "se_pop_adi", "Etot_ave"],
            "progress_frequency": 1.0,
        }
        tsh.generic_recipe(dyn_params, compute_model, {"model0": 0, "nstates": K},
                           init_elec, init_nucl, Random())
        with h5py.File(f"{prefix}/mem_data.hdf", "r") as f:
            pa = np.array(f["se_pop_adi/data"])                  # (nframes, K)
            et = np.array(f["Etot_ave/data"])
        if pops is None:
            pops = np.zeros((pa.shape[0], B, pa.shape[1]))
        pops[:, b, :] = pa
        drift.append(np.max(np.abs(et - et[0])) * 27.2114)       # Hartree -> eV (matches builtin)
        print(f"  libra traj {b + 1}/{B} done")
    time_fs = np.arange(pops.shape[0]) * dt * AUTIME_FS
    return time_fs, pops, np.array(drift)


def main() -> None:
    a = parse_args()
    if a.ensemble:
        a.n_traj, a.n_steps = 300, 2000
    a.out.mkdir(parents=True, exist_ok=True)

    ic = np.load(a.ics)
    Z = np.asarray(ic["Z"], np.int64)
    masses = np.asarray(ic["masses_au"], float)
    R0 = np.asarray(ic["R_bohr"], float)[: a.n_traj]
    V0 = np.asarray(ic["V_au"], float)[: a.n_traj]
    print(f"NQDH model: {a.model} | {len(Z)} atoms | {a.n_traj} trajectory(ies) x {a.n_steps} steps "
          f"= {a.n_steps * a.dt * AUTIME_FS:.0f} fs | engine={a.engine}")

    from nqdh_demo import NQDHDiabaticProvider
    prov = NQDHDiabaticProvider(a.model, device=a.device)

    runner = run_libra if a.engine == "libra" else run_builtin
    time_fs, pops, drift = runner(prov, Z, R0, V0, masses, a.dt, a.n_steps)
    mean_pops = pops.mean(axis=1)

    out_npz = a.out / f"beating_{a.engine}_{a.n_traj}traj.npz"
    np.savez(out_npz, time_fs=time_fs, mean_pops=mean_pops, adiabatic_pops=pops, drift_eV=drift)
    print(f"saved {out_npz} | mean |drift| {np.abs(drift).mean():.4f} eV/traj")

    plot(time_fs, mean_pops, a)


def plot(time_fs, mean_pops, a) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping figure; data saved as .npz)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(time_fs, mean_pops[:, 2], color="#c0392b", lw=2, label="S2")
    ax.plot(time_fs, mean_pops[:, 1], color="#2471a3", lw=2, label="S1")
    ax.plot(time_fs, mean_pops[:, 0], color="#7f8c8d", lw=1.2, ls="--", label="S0")
    if a.n_traj >= 50 and (a.out.parent / "expected_output" / "beating_ensemble.npz").exists():
        ref = np.load(a.out.parent / "expected_output" / "beating_ensemble.npz")
        ax.plot(ref["time_fs"], ref["mean_pops"][:, 2], color="#c0392b", lw=1, ls=":", alpha=0.6,
                label="S2 (reference 300-traj)")
    ttl = "single trajectory" if a.n_traj == 1 else f"{a.n_traj}-trajectory ensemble mean"
    ax.set_title(f"NQDH diabatic Ehrenfest ({a.engine}) - {ttl}", fontsize=10)
    ax.set_xlabel("time (fs)"); ax.set_ylabel("adiabatic population")
    ax.set_ylim(-0.02, 1.02); ax.legend(loc="center right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    png = a.out / f"beating_{a.engine}_{a.n_traj}traj.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
