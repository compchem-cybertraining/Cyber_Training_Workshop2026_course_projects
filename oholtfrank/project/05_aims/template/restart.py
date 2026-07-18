import numpy as np
import pyspawn

pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.fulldiag)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.molcas_cas)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)

tfinal = 1040.0   # ~25 fs

sim = pyspawn.simulation()
sim.restart_from_file("sim.json","sim.hdf5")
sim.set_maxtime_all(tfinal)
sim.enable_ssaims(
   epsilon=1e-10,
   ss_seed=527516,
   suspend_during_spawn=True,
   spawn_delay_steps=10,
   min_tbf_to_start=2,
   verbose=True
)
sim.propagate()
