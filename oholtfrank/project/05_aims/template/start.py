# AIMS: H2O monomer launched on the bright superexcited state (~13.3 eV)
# = the VPD acceptor [absorbs the Ar+ 3s->3p virtual photon].
# SA-13-CAS(8,10)/O:aug-cc-pVTZ,H:aug-cc-pVDZ. Adapted from the working
# dimer start.py (ethylene template lineage).
import numpy as np
import pyspawn
import pyspawn.general
import pyspawn.process_geometry as pg
import pyspawn.dictionaries as dicts
import sys

natoms, atoms, _, comment = pg.process_geometry('geometry.xyz')

mass_dict = dicts.get_atomic_masses()
masses = np.asarray([mass_dict[atom.upper()]*1822.0 for atom in atoms for i in range(3)])

widths_dict = {'O': 12.0, 'H': 4.7}
widths = np.asarray([widths_dict[atom.upper()] for atom in atoms for i in range(3)])

wigner_temp = 0
seed = 69884            # CHANGED PER TRAJECTORY by the clone loop
clas_prop = "vv"
qm_prop = "fulldiag"
qm_ham = "adiabatic"
potential = "molcas_cas"

t0 = 0.0
ts = 5.0                # HALVED vs dimer runs: dense manifold, sharp crossings
tfinal = 2000.0         # ~48 fs; extend via restart.py
numdims = natoms*3      # 9
numstates = 13

molcas_options = {
    "method":       'casscf',
    "pt2":          'xms',
    "basis":        'O.aug-cc-pVTZ, H.aug-cc-pVDZ',
    "atoms":        atoms,
    "charge":       0,
    "spinmult":     1,
    "nactel":       8,
    "actorb":       10,
    "inactive":     1,
    "ipea":         0.0,
    "imaginary":    0.25,
    "cassinglets":  numstates,
    "castargetmult": 1,
    "cas_energy_states": [0,1,2,3,4,5,6,7,8,9,10,11,12],
    "cas_energy_mults":  [1,1,1,1,1,1,1,1,1,1,1,1,1],
    "python3": '/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/compat/usr/bin/python',
    "project": 'H2OSE'
    }

traj_params = {
    "time": t0,
    "timestep": ts,
    "maxtime": tfinal,
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    "istate": 11,           # root 12 (0-indexed) -- VERIFY vs sa13_check!
    "widths": widths,
    "atoms": molcas_options["atoms"],
    "masses": masses,
    "molcas_options": molcas_options
    }

sim_params = {
    "quantum_time": traj_params["time"],
    "timestep": traj_params["timestep"],
    "max_quantum_time": traj_params["maxtime"],
    "qm_amplitudes": np.ones(1,dtype=np.complex128),
    "qm_energy_shift": 0.000000,
}

exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

pyspawn.general.check_files()

traj1 = pyspawn.traj(numdims, numstates)
traj1.set_numstates(numstates)
traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)
traj1.initial_wigner(seed)

sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

sim.enable_ssaims(
   epsilon=1e-10,
   ss_seed=527516,
   suspend_during_spawn=True,
   spawn_delay_steps=10,
   min_tbf_to_start=2,
   verbose=True
)
sim.propagate()
