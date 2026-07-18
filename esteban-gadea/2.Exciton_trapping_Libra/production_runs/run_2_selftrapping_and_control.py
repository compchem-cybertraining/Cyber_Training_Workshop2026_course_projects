"""
Runs 2 and 3 of 3: seeded exciton dynamics and the matched ground-state
control, from the exact same drawn (momentum-balanced) nuclear velocities --
only the propagated electronic state differs. Ab initio-corrected Hamiltonian
from Part I. Report: Sections 2.4.2-2.4.3.
"""
import _pathsetup  # noqa: F401
import sys
if sys.platform == "cygwin":
    from cyglibra_core import *
elif sys.platform in ("linux", "linux2"):
    from liblibra_core import *
import numpy as np
from libra_py import units
import libra_py.dynamics.tsh.compute as tsh_dynamics

from cis_compute_adi import cis_compute_adi, get_default_params, ring_positions
from recipes import ehrenfest_onthefly

# ---------------- System size / geometry ----------------
NCHAIN = 32
DIMER1 = 0.095665    # ground-state equilibrium dimerization at this chain length (energy-minimized)
LATTICE_ANG = 6.0

model_params = get_default_params(nchain=NCHAIN, dimer1=DIMER1, lattice_ang=LATTICE_ANG, hartreeu=0.0)
model_params.update({"nstates": 2, "model": 0, "model0": 0})

ndof = 2 * NCHAIN
nstates = 2

# ---------------- Nuclear initial conditions: zero-momentum-balanced thermal seed ----------------
TSEED_K = 10.0                     # seed temperature (K) -- small on purpose, just enough to
                                    # break symmetry, not meant to represent a real sample temperature
KB_HA_PER_K = 3.1668115634556e-6   # Boltzmann constant, Hartree/Kelvin (atomic units)
MASS_VAL = 198046.0                # atomic units (m_e)
SEED = 25091993                       # numpy RNG seed 

q0 = ring_positions(NCHAIN, model_params["r1"], model_params["r2"])
mass = [MASS_VAL] * ndof
p_width_val = (MASS_VAL * KB_HA_PER_K * TSEED_K) ** 0.5

rng = np.random.default_rng(SEED)
p_raw = rng.normal(loc=0.0, scale=p_width_val, size=ndof)
p_balanced = p_raw - p_raw.mean()   # exact zero total momentum (equal masses -> mean-subtraction is exact)

print(f"Seed: T={TSEED_K} K -> p_width={p_width_val:.4f} a.u. per DOF, numpy seed={SEED}")
print(f"  total momentum after balancing: {p_balanced.sum():.3e} a.u. (should be ~0 to float precision)")

nucl_params_template = {
    "ndof": ndof,
    "q": q0,
    "mass": mass,
    "force_constant": [0.0] * ndof,
    "q_width": [1e-8] * ndof,
    "p_width": [p_width_val] * ndof,
    "init_type": 0,   # exact replay of the pre-balanced array above, not Libra's internal sampler
}

# ---------------- Dynamics parameters ----------------
DT_FS = 0.001
NSTEPS = 800000   # 0.8 ps total -- long enough for the self-trapping signature to develop and
                      # for at least one full localization/re-delocalization "beat" to be captured.


def run_one(istate, prefix):
    istates = [0.0] * nstates
    istates[istate] = 1.0
    elec_params = {
        "verbosity": 2, "init_dm_type": 0,
        "ndia": nstates, "nadi": nstates,
        "rep": 1, "init_type": 1,
        "istates": istates, "istate": istate,
    }

    nucl_params = dict(nucl_params_template)
    nucl_params["p"] = list(p_balanced)   # SAME balanced draw for both states -- isolates
                                           # "which state" as the only variable.

    dyn_general = {
        "nsteps": NSTEPS, "ntraj": 1, "nstates": nstates,
        "dt": DT_FS * units.fs2au, "num_electronic_substeps": 1, "isNBRA": 0, "is_nbra": 0,
        "progress_frequency": 0.1, "which_adi_states": range(nstates), "which_dia_states": range(nstates),
        "mem_output_level": 3,
        "properties_to_save": ["timestep", "time", "q", "p", "f", "Cadi", "Epot_ave", "Ekin_ave", "Etot_ave",
                                "se_pop_adi"],
        "prefix": prefix, "prefix2": prefix,
    }
    ehrenfest_onthefly.load(dyn_general)

    print(f"\nRunning: nchain={NCHAIN} (ndof={ndof}), n_near={model_params['n_near']}, "
          f"nsteps={NSTEPS} ({NSTEPS * DT_FS:.1f} fs), dt={DT_FS} fs, TSEED_K={TSEED_K} K, "
          f"istate={istate} ({'ground, no exciton' if istate == 0 else 'ground+exciton'}) -> prefix={prefix}")
    rnd = Random()
    tsh_dynamics.generic_recipe(dyn_general, cis_compute_adi, model_params, elec_params, nucl_params, rnd)
    print(f"  done: {prefix}")


if __name__ == "__main__":
    run_one(istate=1, prefix="run2_selftrapping")   # Run 2: exciton, seeded -> self-trapping
    run_one(istate=0, prefix="run3_control")         # Run 3: ground state, same seed -> control
    print("\nBoth runs done. Use analyze_dynamics.py on each prefix, and compare_runs.py for the "
          "combined RMS-bond-distortion overlay of all three production runs.")
