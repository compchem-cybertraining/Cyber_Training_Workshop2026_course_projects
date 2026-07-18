"""
Run 1 of 3: exciton dynamics on the 32-unit-cell Ag ring, starting from the
exactly symmetric equilibrium geometry with zero initial nuclear velocity, on
the ab initio-corrected Hamiltonian from Part I. Report: Section 2.4.1.
"""
import _pathsetup  # noqa: F401 -- sets up sys.path for the imports below
import sys
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform in ("linux", "linux2"):
    from liblibra_core import *
from libra_py import units
import libra_py.dynamics.tsh.compute as tsh_dynamics

from cis_compute_adi import cis_compute_adi, get_default_params, ring_positions
from recipes import ehrenfest_onthefly

# ---------------- System size / geometry ----------------
NCHAIN = 32          # dimer unit cells; ndof = 2*NCHAIN = 64
DIMER1 = 0.095665    # ground-state equilibrium dimerization at this chain length (energy-minimized)
LATTICE_ANG = 6.0

# get_default_params() already returns the ab initio-corrected Hamiltonian and
# the production electron-hole coupling (alpha=-2.0 eV, gamma=1 bohr) -- see
# model/cis_compute_adi.py's docstring for the exact values and their provenance.
model_params = get_default_params(nchain=NCHAIN, dimer1=DIMER1, lattice_ang=LATTICE_ANG, hartreeu=0.0)
model_params.update({"nstates": 2, "model": 0, "model0": 0})

ndof = 2 * NCHAIN
nstates = 2

# ---------------- Nuclear initial conditions: equilibrium geometry, exactly zero velocity ----------------
q0 = ring_positions(NCHAIN, model_params["r1"], model_params["r2"])
MASS_VAL = 198046.0  # atomic units (m_e)

nucl_params = {
    "ndof": ndof,
    "q": q0, "p": [0.0] * ndof,
    "mass": [MASS_VAL] * ndof,
    "force_constant": [0.0] * ndof,
    "q_width": [1e-8] * ndof,
    "p_width": [1e-8] * ndof,
    "init_type": 0,   # exact replay of the array above -- no sampling, exactly zero velocity
}

# ---------------- Electronic initial conditions: pure state 1 (ground+exciton) ----------------
istate = 1
istates = [0.0] * nstates
istates[istate] = 1.0
elec_params = {
    "verbosity": 2, "init_dm_type": 0,
    "ndia": nstates, "nadi": nstates,
    "rep": 1, "init_type": 1,
    "istates": istates, "istate": istate,
}

# ---------------- Dynamics parameters ----------------
DT_FS = 0.001
NSTEPS = 200000   # 200 fs -- long enough to confirm the null (delocalized) result;
                   # extend to match run_2's 1 ps window for a strict side-by-side if desired.

dyn_general = {
    "nsteps": NSTEPS, "ntraj": 1, "nstates": nstates,
    "dt": DT_FS * units.fs2au, "num_electronic_substeps": 1, "isNBRA": 0, "is_nbra": 0,
    "progress_frequency": 0.1, "which_adi_states": range(nstates), "which_dia_states": range(nstates),
    "mem_output_level": 3,
    "properties_to_save": ["timestep", "time", "q", "p", "f", "Cadi", "Epot_ave", "Ekin_ave", "Etot_ave",
                            "se_pop_adi"],
    "prefix": "run1_unseeded", "prefix2": "run1_unseeded",
}
ehrenfest_onthefly.load(dyn_general)

if __name__ == "__main__":
    print(f"Run 1 (unseeded): nchain={NCHAIN} (ndof={ndof}), n_near={model_params['n_near']}, "
          f"nsteps={NSTEPS} ({NSTEPS * DT_FS:.1f} fs), dt={DT_FS} fs, zero initial velocity")
    rnd = Random()
    tsh_dynamics.generic_recipe(dyn_general, cis_compute_adi, model_params, elec_params, nucl_params, rnd)
    print("Done. Output prefix:", dyn_general["prefix"])
