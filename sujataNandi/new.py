import os, glob, time, h5py, warnings
import multiprocessing as mp
import matplotlib.pyplot as plt   # plots
import matplotlib.ticker as ticker
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv, dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers
import libra_py.workflows.nbra.decoherence_times as decoherence_times
import libra_py.data_visualize

from recipes import dish_rev2023_nbra, fssh_nbra, fssh2_nbra, gfsh_nbra, ida_nbra, mash_nbra, msdm_nbra

#from matplotlib.mlab import griddata
#%matplotlib inline 
#%matplotlib inline
warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]

def read_files_old(data_path, istep, fstep):
    NSTEPS = fstep - istep

    #================== Read energies =====================
    E = []
    for step in range(istep,fstep):
        energy_filename = F"{data_path}/Hvib_ci_{step}_re.npz"
        energy_mat = sp.load_npz(energy_filename)
        # For data conversion we need to turn np.ndarray to np.array so that 
        # we can use data_conv.nparray2CMATRIX    
        E.append( np.array( np.diag( energy_mat.todense() ) ) )
    E = np.array(E)
    NSTATES = E[0].shape[0]
    
    #================== Read time-overlap =====================
    St = []
    for step in range(istep,fstep):        
        St_filename = F"{data_path}/St_ci_{step}_re.npz"
        St_mat = sp.load_npz(St_filename)
        St.append( np.array( St_mat.todense() ) )
    St = np.array(St)
    
    #================ Compute NACs and vibronic Hamiltonians along the trajectory ============    
    NAC, Hvib = [], []
    for c, step in enumerate(range(istep,fstep)):
        nac_filename = F"{data_path}/Hvib_ci_{step}_im.npz"
        nac_mat = sp.load_npz(nac_filename)
        NAC.append( np.array( nac_mat.todense() ) )
        Hvib.append( np.diag(E[c, :])*(1.0+1j*0.0)  - (0.0+1j)*nac_mat[:, :] )

    NAC = np.array(NAC)
    Hvib = np.array(Hvib)

    print('Number of steps:', NSTEPS)
    print('Number of states:', NSTATES)
    print(NAC.shape)
    print(Hvib.shape)
    print(St.shape)

    # Convert the NumPy arrays to lists of CMATRIX objects - so that we won't need to 
    # convert them in the `compute_model` function.
    St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = [], [], [], [], []
    for i in range(istep,fstep):
        timestep = i - istep
    
        ham = data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
        Ham_adi.append(ham)
    
        nac = data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
        NAC_adi.append(nac)
    
        hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
        Hvib_adi.append(hvib_adi)
    
        basis_transform = CMATRIX(NSTATES,NSTATES); basis_transform.identity()  #basis_transform
        Basis_transform.append(basis_transform)
    
        time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep, :, :] )
        St_adi.append(time_overlap_adi)
        
    return NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform

def read_files_new(data_path, istep, fstep, dt, _nstates=None):
    """
    dt - in fs
    """
    
    nsteps = fstep - istep
    NSTEPS = nsteps
    print(F"Number of steps = {NSTEPS}")

    # ================= Reading the data ======================
    # Get the number of states:
    st = np.loadtxt(F"{data_path}/st_adi_{istep}.txt")
    NSTATES = st.shape[0]
    max_nstates = NSTATES
    if _nstates is not None:
        NSTATES = min(_nstates, NSTATES)
        
    rng = Py2Cpp_int(list(range(NSTATES)))

    print(F"NSTATES = {NSTATES}")
    

    #==== Read time-overlap, adiabatic Hamiltonians and vibronic Hamiltonians ==========
    St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = [], [], [], [], []
    for i in range(istep,fstep):
    
        # Time-overlaps
        st = CMATRIX(max_nstates, max_nstates)
        st_red = CMATRIX(NSTATES, NSTATES)
        st.Load_Matrix_From_File(F"{data_path}/st_adi_{i}.txt")
        pop_submatrix(st, st_red, rng, rng)
        St_adi.append(st_red)
        
        # Adiabatic Hamiltonian
        ham_adi = CMATRIX(max_nstates, max_nstates)
        ham_adi_red = CMATRIX(NSTATES, NSTATES)
        ham_adi.Load_Matrix_From_File(F"{data_path}/ham_adi_{i}.txt")
        pop_submatrix(ham_adi, ham_adi_red, rng, rng)
        Ham_adi.append(ham_adi_red)

        # NACs
        #nac_adi =  (ham_adi - hvib_adi)*(0.0+1.0j)
        nac_adi_red = (st_red - st_red.H())/(2.0*dt * units.fs2au)
        NAC_adi.append(nac_adi_red)

        # Adiabatic vibronic Hamiltonian
        hvib_adi_red = CMATRIX(NSTATES,NSTATES)
        #hvib_adi.Load_Matrix_From_File(F"{data_path}/hvib_adi_{i}.txt")
        hvib_adi_red = ham_adi_red - nac_adi_red*(0.0+1.0j)
        Hvib_adi.append(hvib_adi_red)
        
    
        # Basis transform
        bas_trans_red = CMATRIX(NSTATES, NSTATES);  bas_trans_red.identity()
        Basis_transform.append(bas_trans_red)
        
    return NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform

##================ Example 3 (New) ======================
# Uncomment this line, if you need the data files
#!tar -xf data3.tar.bz2

data_path = 'data3/res'
istep = 0    # the first timestep to read
fstep = 400 # the last timestep to read
ISTATE = 1
DT = 0.5

basis_size = 3
#basis_size = None

NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = read_files_new(data_path, istep, fstep, DT, basis_size)

numpy_st_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )
numpy_ham_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )
numpy_nac_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )

for i in range(NSTEPS):
    numpy_st_adi[i, :, :] = data_conv.MATRIX2nparray(St_adi[i]).real
    numpy_ham_adi[i, :, :] = data_conv.MATRIX2nparray(Ham_adi[i]).real
    numpy_nac_adi[i, :, :] = data_conv.MATRIX2nparray(NAC_adi[i]).real


for nact_st in range(0, NSTATES+1):
    min_val = np.min( np.abs( np.linalg.det(numpy_st_adi[:,:nact_st,:nact_st]) ) )
    print(F"nact_st = {nact_st}, min_val = {min_val}")

# ================= Computing the energy gaps and decoherence times
# Average decoherence times and rates
tau, rates = decoherence_times.decoherence_times_ave([Hvib_adi], [0], len(Hvib_adi), 0)

# Computes the energy gaps between all states for all steps
dE = decoherence_times.energy_gaps_ave([Ham_adi], [0], len(Ham_adi))

# Decoherence times in fs
deco_times = data_conv.MATRIX2nparray(tau) * units.au2fs

# Zero all the diagonal elements of the decoherence matrix
np.fill_diagonal(deco_times, 0)

# Saving the average decoherence times
np.savetxt('decoherence_times.txt',deco_times.real)

# Computing the average energy gaps
gaps = MATRIX(NSTATES, NSTATES)
for step in range(NSTEPS):
    gaps += dE[step]
gaps /= NSTEPS

rates.show_matrix("decoherence_rates.txt")
gaps.show_matrix("average_gaps.txt")

class abstr_class:
    pass

def compute_model(q, params, full_id):
    timestep = params["timestep"]

    obj = abstr_class()
    obj.ham_adi = Ham_adi[timestep]
    obj.hvib_adi = Hvib_adi[timestep]
    obj.nac_adi = NAC_adi[timestep]
    obj.basis_transform = Basis_transform[timestep]
    obj.time_overlap_adi = St_adi[timestep]

    return obj

#================== Model parameters ====================
model_params = { "timestep":0, "icond":0,  "model0":0, "nstates":NSTATES }

#=============== Some automatic variables, related to the settings above ===================
dyn_general = { "nsteps":NSTEPS*2, "ntraj":250, "nstates":NSTATES, "dt":DT*units.fs2au, "nfiles": NSTEPS,
                "decoherence_rates":rates, "ave_gaps": gaps,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "mem_output_level":2,
                "properties_to_save":[ "timestep", "time","se_pop_adi", "sh_pop_adi" ],
                "prefix":F"NBRA", "isNBRA":0
              }

# Uncomment one of the options in each of the categories below:
#====== How to update Hamiltonian ===================
#dyn_general.update({"ham_update_method":0}) # don't update any Hamiltonians
#dyn_general.update({"ham_update_method":1})  # recompute only diabatic Hamiltonian, common choice for model Hamiltonians
dyn_general.update({"ham_update_method":2})  # recompute only adiabatic Hamiltonian; use with file-based or on-the-fly workflows


#====== How to transform the Hamiltonians between representations ============
dyn_general.update( {"ham_transform_method":0 }) # don't do any transforms; usually for NBRA or on-the-fly workflows,
                                                 # so you don't override the read values
#dyn_general.update( {"ham_transform_method":1 }) # diabatic->adiabatic according to internal diagonalization
#dyn_general.update( {"ham_transform_method":2 }) # diabatic->adiabatic according to internally stored basis transformation matrix
#dyn_general.update( {"ham_transform_method":3 }) # adiabatic->diabatic according to internally stored basis transformation matrix
#dyn_general.update( {"ham_transform_method":4 }) # adiabatic->diabatic according to local diabatization method

#====== How do get the time-overlaps in the dynamics ========
dyn_general.update( {"time_overlap_method":0 })  # don't update time-overlaps - maybe they are already pre-computed and read
#dyn_general.update( {"time_overlap_method":1 }) # explicitly compute it from the wavefunction info; common for model Hamiltonians

#================== How to compute NACs ===============================
#dyn_general.update({"nac_update_method":0 })  # just read from files
#dyn_general.update({"nac_update_method":1})  # explicit NAC calculations - let's just focus on this one for now
dyn_general.update({"nac_update_method":2, "nac_algo":0})  # HST algo
#dyn_general.update({"nac_update_method":2, "nac_algo":1})  # NPI algo

#============== How to compute vibronic Hamiltonian ==============
dyn_general.update( {"hvib_update_method":0 }) # don't update Hvib; maybe because we read it from files
#dyn_general.update( {"hvib_update_method":1 }) # recompute diabatic and adiabatic Hvib from the Ham and NACs in those reps

#=========== Ehrenfest or state-resolved options ===========
# This is what we use with any of the TSH-based methods - in all cases here, we would
# use "rep_force":1 so that we are guided by the forces derived from the adiabatic surfaces.
# In Ehrenfest cases though, the forces can be computed using only diabatic properties though
dyn_general.update( {"force_method":0, "rep_force":1} ) # don't compute forces
#dyn_general.update( {"force_method":1, "rep_force":1} ) # state-resolved (e.g. TSH) with adiabatic forces
#dyn_general.update( {"force_method":2, "rep_force":1} ) # for Ehrenfest in adiabatic rep
#dyn_general.update( {"force_method":2, "rep_force":0} ) # for Ehrenfest in diabatic rep

#============ Types of surface hopping acceptance and momenta rescaling opntions =================
#dyn_general.update({"hop_acceptance_algo":10, "momenta_rescaling_algo":100 })  # accept and rescale based on total energy, do not reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":10, "momenta_rescaling_algo":101 })  # accept and rescale based on total energy, reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":20, "momenta_rescaling_algo":200 })  # accept and rescale based on NAC vectors, do not reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":20, "momenta_rescaling_algo":201 })  # accept and rescale based on NAC vectors, reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":21, "momenta_rescaling_algo":200 })  # accept and rescale based on force differences, do not reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":21, "momenta_rescaling_algo":201 })  # accept and rescale based on force differences, reverse on frustrated
dyn_general.update({"hop_acceptance_algo":31, "momenta_rescaling_algo":0 })  # accept and rescale based on total energy, reverse on frustrated

#============ Surface hopping opntions =================
#dyn_general.update({"tsh_method":-1 }) # adiabatic, no surface hopping
dyn_general.update({"tsh_method":0 }) # FSSH
#dyn_general.update({"tsh_method":1 }) # GFSH
#dyn_general.update({"tsh_method":2 }) # MSSH
#dyn_general.update({"tsh_method":3, "rep_lz":0 })  # LZ options
#dyn_general.update({"tsh_method":4, "rep_lz":0 }) # ZN
#dyn_general.update({"tsh_method":5 }) # DISH
#dyn_general.update({"tsh_method":6 }) # MASH
#dyn_general.update({"tsh_method":7 }) # FSSH2
#dyn_general.update({"tsh_method":8 }) # GFSH (original)

#=========== Decoherence options =================
dyn_general.update({ "decoherence_algo":-1}) # no (additional) decoherence
#dyn_general.update({ "decoherence_algo":0}) # SDM and alike
#dyn_general.update({ "decoherence_algo":1}) # IDA (ID-S, ID-A, ID-C)
#dyn_general.update({ "decoherence_algo":2}) # A-FSSH, not yet ready
#dyn_general.update({ "decoherence_algo":3}) # BCSH
#dyn_general.update({ "decoherence_algo":4}) # MF-SD
#dyn_general.update({ "decoherence_algo":5}) # SHXF
#dyn_general.update({ "decoherence_algo":6}) # MQCXF
#dyn_general.update({ "decoherence_algo":7}) # DISH, rev2023

#=========== Decoherence times (for decoherence options 0 or 4) ==================
dyn_general.update({"decoherence_times_type":-1 }) # No decoherence times, infinite decoherence times
#dyn_general.update( { "decoherence_times_type":0 } )  # manual decoherence times
#dyn_general.update( { "decoherence_times_type":1, "decoherence_C_param": 1.0, "decoherence_eps_param":0.1 } )  # EDC + default params
#dyn_general.update( { "decoherence_times_type":2, "schwartz_decoherence_inv_alpha":A } ) # Schwartz version 1
#dyn_general.update( { "decoherence_times_type":3, "schwartz_decoherence_inv_alpha":A } ) # Schwartz version 2

#======= Various decoherence-related parameters =====================
dyn_general.update( {"dephasing_informed":0 } ) #, "decoherence_rates":MATRIX(2,2), "ave_gaps":MATRIX(2,2) } )

#======= DISH-specific ======================
dyn_general.update( {"dish_decoherence_event_option":1} )

#=========== Phase correction of SSY =================
dyn_general.update({"do_ssy":0 }) # do no use it - that's the default

#=========== What to integrate ==================
# solve TD-SE in diabatic representation
#dyn_general.update({"rep_tdse":0, "electronic_integrator":-1 })   # no propagation
#dyn_general.update({"rep_tdse":0, "electronic_integrator":0 })    # Lowdin exp_ with 2-point Hvib_dia
#dyn_general.update({"rep_tdse":0, "electronic_integrator":1 })    # based on QTAG propagator
#dyn_general.update({"rep_tdse":0, "electronic_integrator":2 })    # based on modified QTAG propagator (Z at two times)
#dyn_general.update({"rep_tdse":0, "electronic_integrator":3 })    # non-Hermitian integrator with 2-point Hvib_dia

# solve TD-SE in adiabatic representation
#dyn_general.update({"rep_tdse":1, "electronic_integrator":-1 })  # no propagation
#dyn_general.update({"rep_tdse":1, "electronic_integrator":0 })   # ld, with crude splitting,  with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":1 })   # ld, with symmetric splitting, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":2 })   # ld, original, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":4 })   # 2-points, Hvib integration, with exp_
dyn_general.update({"rep_tdse":1, "electronic_integrator":5 })   # 2-points, Hvib, integration with the second-point correction of Hvib, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":6 })   # same as 4, but without projection matrices (T_new = I)
#dyn_general.update({"rep_tdse":1, "electronic_integrator":10 })  # same as 0, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":11 })  # same as 1, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":12 })  # same as 2, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":13 })  # same as 3, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":14 })  # same as 4, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":15 })  # same as 5, but with rotations

# solve QCLE in diabatic representation
#dyn_general.update({"rep_tdse":3, "electronic_integrator":0 })  # mid-point Hvib, using exp_

# solve QCLE in adiabatic representation
#dyn_general.update({"rep_tdse":3, "electronic_integrator":0 })  # mid-point Ham with the second-point correction of Hvib, using exp_
#dyn_general.update({"rep_tdse":3, "electronic_integrator":1 })  # using Zhu Liouvillian THIS IS NOT JUST A DIFFERENT INTEGRATOR!!!!
#dyn_general.update({"rep_tdse":3, "electronic_integrator":10 }) # same as 0 but with rotations

#=========== Disable state tracking and phase corrections explicitly for the LD integrators ===============
#State tracking algorithm:
#  - -1: use LD approach, it includes phase correction too [ default ]
#  - 0: no state tracking
#  - 1: method of Kosuke Sato (may fail by getting trapped into an infinite loop)
#  - 2: Munkres-Kuhn (Hungarian) algorithm
#  - 21: ChatGPT-generated Munkres-Kuhn (Hungarian) algorithm
#  - 3: experimental stochastic algorithm, the original version with elimination (known problems)
#  - 32: experimental stochastic algorithms with all permutations (too expensive)
#  - 33: the improved stochastic algorithm with good scaling and performance, on par with the mincost
#  - 4: new, experimental force-based tracking
#dyn_general.update({"state_tracking_algo":-1 })
dyn_general.update({"state_tracking_algo":21 })


# Choice of the scaling function for the cost matrix in the Munkres-Kuhn (Hungarian) algorithm
# Options:
# - 0 : exp(- alpha^2 * |dE_ij|^2 )  [default]
# - 1 : exp(- alpha * |dE_ij| )
# - 2 : exp(-alpha * max(dE_ij, 0) )
# - anything else:  1  - no scaling
dyn_general.update({"MK_scaling_function":1})

# Controlling the range of states involved in state tracking
# the larger value means narrower range of states is
# considered when determining possible state tracking effects
# Doesn't matter for LD integrators
dyn_general.update({"MK_alpha":1100.0})

#The algorithm to correct phases on adiabatic states
#
#Options:
#  - 0: no phase correction
#  - 1: according to our phase correction algorithm [ default ]
# phase correction doesn't matter, if we use the LD integrator
#dyn_general.update({"do_phase_correction":0 })
dyn_general.update({"do_phase_correction":1 })

#New phase correction, directly applied to NACs. Intended to be used mostly with state_tracking_algo == 4,
#although can be useful with other state treacking algorithms. Should not be used together with
#`do_phase_correction`
#Options:
#  - 0: no correction [ default ]
#  - 1: do this correction
dyn_general.update({"do_nac_phase_correction":0 })
#dyn_general.update({"do_nac_phase_correction":1 })

#=================== Dynamics =======================
# Nuclear DOF - these parameters don't matter much in the NBRA calculations
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

# Amplitudes are sampled
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

elec_params.update( {"init_type":1,  "rep":1,  "istate":ISTATE } )  # how to initialize: random phase, adiabatic representation, given initial state

ICONDS = [0] #list(range(0,200,25))

time
rnd = Random()

for icond in ICONDS:
    print('Running the calculations for icond:', icond)
    model_params.update({"icond": icond})
    dyn_general.update({"prefix":F"FSSH_NBRA_icond_{icond}"})
    res = tsh_dynamics.generic_recipe(dyn_general, compute_model, model_params, elec_params, nucl_params, rnd)


pref = "FSSH_NBRA_icond_0"

nst = model_params["nstates"]
ntraj = dyn_general["ntraj"]

plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":2,
                "which_trajectories":list(range(ntraj)), "which_dofs":[0], "which_adi_states":list(range(nst)), 
                "which_dia_states":list(range(nst)), 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "what_to_plot":["se_pop_adi", "sh_pop_adi" ], 
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":1
              }

tsh_dynamics_plot.plot_dynamics(plot_params)
