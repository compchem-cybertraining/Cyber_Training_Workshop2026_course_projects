import sys
import numpy as np
import pyspawn
import pyspawn.general
import pyr4_cone

idx = int(sys.argv[1])

clas_prop = "vv"
qm_prop = "fulldiag"
qm_ham = "adiabatic"
t0 = 0.0
ts = 0.25
tfinal = 182.0
numdims = 4
numstates = 2
spawn_scale = 0.4
seed = 20260715 + idx

OMEGA = pyr4_cone.OMEGA
widths = np.asarray([0.5, 0.5, 0.5, 0.5])
masses = 1.0 / OMEGA

np.random.seed(seed)
positions = np.random.normal(0.0, np.sqrt(0.5), numdims)
momenta = np.random.normal(0.0, np.sqrt(0.5), numdims)

traj_params = {
    "time": t0,
    "timestep": ts,
    "maxtime": tfinal,
    "spawnthresh": spawn_scale * (0.5 * np.pi) / ts / 20.0,
    "istate": 1,
    "widths": widths,
    "masses": masses,
    "positions": positions,
    "momenta": momenta,
}
sim_params = {
    "quantum_time": t0,
    "timestep": ts,
    "max_quantum_time": tfinal,
    "qm_amplitudes": np.ones(1, dtype=np.complex128),
    "qm_energy_shift": 0.0,
}

exec ("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec ("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
pyspawn.import_methods.into_traj(pyr4_cone)
exec ("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

pyspawn.general.check_files()
traj1 = pyspawn.traj(numdims, numstates)
traj1.set_parameters(traj_params)
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)
sys.stderr.write("IC %d  seed=%d  spawnthresh=%.4f\n" % (idx, seed, traj_params["spawnthresh"]))
sim.propagate()
print("IC %d DONE" % idx)
