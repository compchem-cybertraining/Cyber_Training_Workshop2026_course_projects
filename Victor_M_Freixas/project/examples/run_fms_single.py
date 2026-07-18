#!/usr/bin/env python
"""Example 3 (optional): full multiple spawning on the NQDH, one trajectory.

Requires pySpawn17 (python-3 port) in addition to the base environment:
    git clone <your py3 pySpawn17>  &&  pip install --no-deps -e pySpawn17
(--no-deps matters: pySpawn's setup.py pins an ancient numpy.)

The --rescale flag selects the spawn-time momentum adjustment:
    velocity  stock pySpawn (scalar rescale of the whole momentum vector)
    coupling  along the S1-S2 derivative coupling (the Pechukas direction)
Every spawn attempt is recorded with its time and outcome (accepted /
frustrated) -- the raw data behind the report's gate-timing histogram.

RUNTIME WARNING: FMS cost grows with the number of spawned basis functions.
The default --tfinal 250 (au) is a ~6 fs smoke test (a few minutes).  A
physically interesting window is --tfinal 1650 (~40 fs), which takes
~5-8 HOURS per trajectory: run those through a queue (see slurm_fms_array.sh).

    python examples/run_fms_single.py [--rescale coupling] [--tfinal 250]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nqdh_demo.pyspawn_potential as nqdh_pot  # noqa: E402

ELEMENT_WIDTHS = {1: 4.7, 6: 22.7, 7: 19.0, 8: 12.2}   # AIMS alphas (bohr^-2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-index", type=int, default=0)
    p.add_argument("--rescale", choices=("velocity", "coupling"), default="velocity")
    p.add_argument("--tfinal", type=float, default=250.0,
                   help="au of time; 250 = smoke (~min), 1650 = 40 fs (~hours!)")
    p.add_argument("--timestep", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("output/fms_single"))
    a = p.parse_args()

    ics = np.load(ROOT / "data" / "ics.npz")
    Z = np.asarray(ics["Z"], np.int64)
    R0 = np.asarray(ics["R_bohr"], float)[a.ic_index].reshape(-1)
    V0 = np.asarray(ics["V_au"], float)[a.ic_index].reshape(-1)
    m = np.repeat(np.asarray(ics["masses_au"], float), 3)
    widths = np.repeat(np.array([ELEMENT_WIDTHS.get(int(z), 20.0) for z in Z]), 3)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(a.out_dir)                                   # pySpawn writes to cwd

    nqdh_pot.configure(ROOT / "model", Z, device=a.device, rescale=a.rescale)

    import pyspawn
    pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.fulldiag)
    pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
    pyspawn.import_methods.into_traj(nqdh_pot)
    pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    pyspawn.general.check_files()

    lam0, _, _ = nqdh_pot._eval(R0)
    ts = a.timestep
    traj1 = pyspawn.traj(len(m), 3)
    traj1.set_parameters({
        "time": 0.0, "timestep": ts, "maxtime": a.tfinal,
        "spawnthresh": (0.5 * np.pi) / ts / 20.0,
        "istate": 2, "widths": widths, "masses": m,
        "positions": R0, "momenta": m * V0,
    })
    sim = pyspawn.simulation()
    sim.add_traj(traj1)
    sim.set_parameters({
        "quantum_time": 0.0, "timestep": ts, "max_quantum_time": a.tfinal,
        "qm_amplitudes": np.ones(1, dtype=np.complex128),
        "qm_energy_shift": -float(lam0[2]),
    })
    sim.propagate()

    ev = np.array(nqdh_pot._STATE["events"], dtype=float).reshape(-1, 2)
    print("SPAWN SUMMARY: accepted %d | frustrated %d | mode %s"
          % (nqdh_pot._STATE["n_spawns"], nqdh_pot._STATE["n_frustrated"], a.rescale))
    an = pyspawn.fafile("sim.hdf5")
    for meth in ("fill_quantum_times", "fill_electronic_state_populations"):
        try:
            getattr(an, meth)(column_filename=None)
        except TypeError:
            getattr(an, meth)()
    t = np.asarray(an.datasets["quantum_times"]).ravel()
    pop = np.asarray(an.datasets["electronic_state_populations"])
    np.savez("fms_populations.npz", time_au=t, el_pop=pop,
             event_times_au=ev[:, 0], event_accepted=ev[:, 1], rescale=a.rescale)
    print("saved fms_populations.npz | S2(end) = %.3f at %.1f fs" % (pop[-1, 2], t[-1] * 0.0241888))


if __name__ == "__main__":
    main()
