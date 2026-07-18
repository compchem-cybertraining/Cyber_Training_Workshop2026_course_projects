#!/usr/bin/env python
# coding: utf-8

# # Running the nonadiabatic molecular dynamics (NA-MD)
# 
# In this tutorial, we conduct NBRA NA-MD calculations using the Libra code. This version implements the generic dynamical approach that is common to model-based NA-MD simulations and the one based on reading files. In this particular case, we setup the interface to pre-computed energy and time-overlap files (as could be produced by any codes) and use it in the generic dynamical workflow.
# 
# ## Table of contents
# <a name="toc"></a>
# 1. [Importing needed libraries](#import)
# 
# 2. [Read the files in a selected basis](#2)
# 
#    * 2.1. [Function for reading old style (.npz) files](#2.1)
#  
#    * 2.2. [Function for reading new style (.txt) files](#2.2)
#    
#    * 2.3. [Make your choice](#2.3)
# 
#    * 2.4. [Time-overlaps diagnostics](#2.4)
# 
# 3. [Computing the average energy gaps and decoherence times](#gap_deco)
# 
# 4. [Define the interface](#def_interface)
# 
# 5. [Define the parameters](#5) 
# 
# 6. [Run NAMD simulations](#run_namd)
# 
#    * 6.1. [Run all initial conditions in a consecutive way](#6.1)
#    
#    * 6.2. [With pooling - using multiprocessing](#6.2)
#    
# 7. [Plot results](#plot_res)
# 
#    * 7.1. [Plot the results using `plot_dynamics`](#plot_res_1)
#    
#    * 7.2. [Average decoherence times map](#plot_res_2)
#    
#    * 7.3. [Computing lifetimes](#plot_res_3)
#    
# 8. [Additional descriptive analysis: Pre-dynamics](#8)
# 
#    * 8.1. [Convert the matrices to NumPy for convenience](#8.1)
#    
#    * 8.2. [Excitation energies vs time](#8.2)
#    
#    * 8.3. [Gap distributions](#8.3)
#    
#    * 8.4. [NAC map](#8.4)
# 
#    * 8.5. [NAC distribution (all NAC pairs)](#8.5)
# 
#    * 8.6. [NAC distribution (only adjacent pairs)](#8.6)
#    
#    * 8.7. [Time-overlaps diagnostics](#8.7)
# 
# ### A. Learning objectives
# 
# * To set up and run file-based NBRA NA-MD calculations with different methods
# * To be able to plot the decoherence times map
# * To be able to plot and analyze NAMD results
# 
# ### B. Use cases
# 
# * [Run NAMD simulations](#run_namd)
# * [Plot results](#plot_res)
# 
# ### C. Functions
# 
# - `libra_py`
#   - `dynamics`
#     - `tsh`
#       - `compute`
#         - [`generic_recipe`](#run_namd)
#       - `plot`
#         - [`plot_dynamics`](#plot_res)
#   - `data_conv`
#     - [`nparray2CMATRIX`](#gap_deco)
#   - `workflows`
#     - `nbra`
#       - `decoherence_times`
#         - [`decoherence_times_ave`](#decoherence_times_ave-1)
#         - [`energy_gaps_ave`](#energy_gaps_ave-1)
#   - `units`
#     - [`au2ev`](#gap_deco)
#     - [`au2fs`](#gap_deco)
#     

# ## 1. Importing needed libraries <a name="import"></a>
# [Back to TOC](#toc)
# 
# First, let's import all necessary libraries and define parameters such as colors

# In[17]:


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
get_ipython().run_line_magic('matplotlib', 'inline')
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


# ## 2. Read the files in a selected basis<a name="2"></a>
# [Back to TOC](#toc)
# 
# Here, we will be reading the files required for NBRA-NAMD simulations. For convenience, we'll define two functions and will be calling them with the appropriate inuts as needed.
# 
# Both functions have the parameters:
# 
# - `istep`: The initial step index
# 
# - `fstep`: The final step index
# 
# - `data_path` : The path to the data folder containing all the needed files 
# 
# > Note: that all the files from `istep` to `fstep-1` should exist. 
# 
# The function returns:
# 
# - `NSTEPS`: Number of steps which is `fstep-istep`
# 
# - `NSTATES` : the number of states  
# 
# - `St_adi`, `Ham_adi`, `Hvib_adi`, `NAC_adi`, `Basis_transform` : list of CMATRIX objects containing data for all timesteps. These are the global storage for later reuse by the TSH workflow.
# 
# 
# ### 2.1. Function for reading old style (.npz) files
# <a name="2.1"></a>[Back to TOC](#toc)

# In[2]:


def read_files_old(data_path, istep, fstep):
    NSTEPS = fstep - istep

    #================== Read energies =====================
    E = []
    for step in range(istep,fstep):
        energy_filename = F"{data_path}/Hvib_sd_{step}_re.npz"
        energy_mat = sp.load_npz(energy_filename)
        # For data conversion we need to turn np.ndarray to np.array so that 
        # we can use data_conv.nparray2CMATRIX    
        E.append( np.array( np.diag( energy_mat.todense() ) ) )
    E = np.array(E)
    NSTATES = E[0].shape[0]
    
    #================== Read time-overlap =====================
    St = []
    for step in range(istep,fstep):        
        St_filename = F"{data_path}/St_sd_{step}_re.npz"
        St_mat = sp.load_npz(St_filename)
        St.append( np.array( St_mat.todense() ) )
    St = np.array(St)
    
    #================ Compute NACs and vibronic Hamiltonians along the trajectory ============    
    NAC, Hvib = [], []
    for c, step in enumerate(range(istep,fstep)):
        nac_filename = F"{data_path}/Hvib_sd_{step}_im.npz"
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


# ### 2.2. Function for reading new style (.txt) files
# <a name="2.2"></a>[Back to TOC](#toc)

# In[3]:


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


# ### 2.3. Make your choice
# <a name="2.3"></a>[Back to TOC](#toc)
# 
# **Example 1:** 
# 
# - prepared in [this tutorial](https://github.com/compchem-cybertraining/Tutorials_Libra/tree/master/6_dynamics/2_nbra_workflows/8_step3/1_DFT)
# - old style data files (.npz)
# - data in "data.tar.bz2"
# - `istep=1200`, `fstep=1399`
# - 11 states
# - `dt = 1.0 fs`
# 
# **Example 2:**
# 
# - (TiO2)2 from Miguel Recio
# - new style data files (.txt)
# - data in `data2.tar.bz2`
# - `istep=0`, `fstep=399`
# - 10 states
# - `dt = 1.0 fs`
# - prepared in [this tutorial](https://github.com/compchem-cybertraining/Tutorials_Libra/blob/master/11_program_specific_methods/3_cp2k_methods/5_namd_workflow/tutorial2-tio2.ipynb)
# 
# 
# **Example 3:**
# 
# - Mg@C60 from Kosar Yasin
# - new style data files (.txt)
# - data in `data3.tar.bz2`
# - `istep=0`, `fstep=49`
# - 4 states
# - `dt = 0.5 fs`
# 

# In[38]:


##================ Example 1 (Old) ======================
# Uncomment this line, if you need the data files
#!tar -xf data1.tar.bz2
"""
data_path = 'data1'
istep = 1200    # the first timestep to read
fstep = 1399 # the last timestep to read
ISTATE = 10
DT = 1.0

basis_size = 11
#basis_size = None

NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = read_files_old(data_path, istep, fstep, DT, basis_size)
"""


# In[39]:


##================ Example 2 (New) ======================
# Uncomment this line, if you need the data files
#!tar -xf data2.tar.bz2
"""
data_path = 'data2'
istep = 0    # the first timestep to read
fstep = 399 # the last timestep to read
ISTATE = 2
DT = 1.0

basis_size = 10
#basis_size = None

NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = read_files_new(data_path, istep, fstep, DT, basis_size)
"""


# In[3]:


##================ Example 3 (New) ======================
# Uncomment this line, if you need the data files
#!tar -xf data3.tar.bz2

data_path = '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/step3/2_xTB_back/res-sd-xTB'
istep = 0    # the first timestep to read
fstep = 200 # the last timestep to read
#ISTATE = 1
DT = 1.0

#basis_size = 3
#basis_size = None

NSTEPS, NSTATES, St_adi, Ham_adi, Hvib_adi, NAC_adi, Basis_transform = read_files_old(data_path, istep, fstep)


# In[4]:


# ============ Pick the initial state: lowest excited SD ============
E_all = []
for step in range(istep, fstep):
    m = sp.load_npz(F"{data_path}/Hvib_sd_{step}_re.npz")
    E_all.append(np.diag(m.todense()))
E_all = np.array(E_all).real
Ebar = E_all.mean(axis=0)          # trajectory-averaged energy of each state
order = np.argsort(Ebar)

print("Lowest 5 states by average energy:")
print("index :  gap to ground state (eV)")
for idx in order[:5]:
    print(F"  {idx}  :  {(Ebar[idx]-Ebar[order[0]])*units.au2ev:.4f}")

ISTATE = int(order[1])             # the lowest excited state
print(F"\nISTATE = {ISTATE}, gap = {(Ebar[ISTATE]-Ebar[0])*units.au2ev:.3f} eV")


# ### 2.4. Time-overlaps diagnostics
# <a name="2.4"></a>[Back to TOC](#toc)
# 
# If the determinant of the time-overlap matrix is close to zero (or just very small) at some points - this indicates that some of the states may be projecting outside of the selected basis of excited states. In these situations, using LD integrators may be dangerous.
# 
# You can vary the `nact_st` below to see for which value the determinant is not close to zero. This may suggest what active space of CI excitations to use
# 
# The parameter `nact_st` is basically the `basis_size` used above. So, before selecting the `basis_size` for your production run, set it to `None` - the full matrices available will be read then. The cell below will iterate over possible basis set sizes to determine for which one the system is best-behaving.
# 
# Generally, the larger number of states will be making the determinant of the `St` matrices increasingly smaller. However, the behavior is non-monotonic. The local maxima of such measures identify the basis set sizes that lead to best completeness of the basis for the duration of time considered in simulations.

# In[5]:


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


# ## 3. Computing the average energy gaps and decoherence times<a name="gap_deco"></a>
# [Back to TOC](#toc)
# 
# Below, we compute the average energy gaps and decoherence times between all states. The average gaps matrix, `gaps`, and average decoherence times and rates, `avg_deco` and `rates`, are required for NAMD with decoherence methods.
# <a name="energy_gaps_ave-1"></a>

# In[6]:


help(decoherence_times.energy_gaps_ave)


# <a name="decoherence_times_ave-1"></a>

# In[7]:


help(decoherence_times.decoherence_times_ave)


# In[8]:


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


# In[9]:


deco = np.loadtxt('decoherence_times.txt')
print(F"GS <-> state {ISTATE}: {deco[0, ISTATE]:.1f} fs")


# ## 4. Define the interface <a name="def_interface"></a>
# [Back to TOC](#toc)
# 
# The interface function should be defined as explained in other tutorials. It should return objects with the specifically-named data members. This is what we defined here:
# <a name="nparray2CMATRIX-1"></a><a name="use_case-3"></a>

# In[10]:


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


# ## 5. Define the parameters <a name="5"></a>
# [Back to TOC](#toc)
# 
# Make sure you define all of these 4 parameters of the `model_params` dictionary. Here, we first initialize the parameter `timestep` to `0`. This parameter will be updated to `icond` which is representative of the initial geomtery. 

# In[11]:


#================== Model parameters ====================
model_params = { "timestep":0, "icond":0,  "model0":0, "nstates":NSTATES }


# **Dynamics control** parameters are as follows:
# 
# - `nfiles`: The total number of files that was read.
# 
# - `nsteps`: The total number of steps for dynamics. Note that this number can be larger than the number of files `nfiles` read. For `nsteps` more than `nfiles`, Libra will repeat the Hamiltonian matrices and do the dynamics.
# 
# - `ntraj`: The number of surface hopping trajectories.
# 
# - `dt`: The time step in atomic units.
# 
# - `decoherence_rates`: The matrix of dephasing rates with a unit of a.u. of time$^{-1}$.
# 
# - `ave_gaps`: A matrix that contains the averaged moduli of the energy gaps
# 
# $$E_{ij}= |E_i - E_j|$$
# 
# It is needed when `dephasing_informed` option is used, like DISH method.
# 
# - `progress_frequency`: At what intervals print out some "progress" messages. For instance, if you have `nsteps = 100` and `progress_frequency = 0.1`, the code will notify you every 10 steps.
# 
# - `which_adi_states`: Indices of the adiabatic states to plot.
# 
# - `which_dia_states`: Indices of the diabatic states to plot.
# 
# - `mem_output_level`: Controls what info to save into HDF5 files (all at the end) Same meaning and output as with `hdf5_output_level`, except all the variables are first stored in memory (while the calcs are running) and then they are written into the HDF5 file at the end of the calculations. This is a much faster version of hdf5 saver.
# 
# - `properties_to_save`: Describes what properties to save to the HDF5 files. Note that if some properties are not listed in this variable, then they are not saved, even if `mem_output_level` suggests they may be saved. You need to BOTH set the appropriate `mem_output_level` AND `properties_to_save`:
# 
# ```
# default:  [ "timestep", "time", "Ekin_ave", "Epot_ave", "Etot_ave", 
#          "dEkin_ave", "dEpot_ave", "dEtot_ave", "states", "SH_pop", "SH_pop_raw",
#          "D_adi", "D_adi_raw", "D_dia", "D_dia_raw", "q", "p", "Cadi", "Cdia", 
#          "hvib_adi", "hvib_dia", "St", "basis_transform", "projector"]
# ```
# 
# - `prefix`: The name of the folder to be created by this function. All the data files will be created in that folder.
# 
# - `isNBRA`: A flag for NBRA type of calculations.
# 
# Other parameters for NBRA-specific parameters are explained in the comments of the cell below.

# In[12]:


#=============== Some automatic variables, related to the settings above ===================
dyn_general = { "nsteps":NSTEPS, "ntraj":250, "nstates":NSTATES, "dt":DT*units.fs2au, "nfiles": NSTEPS,
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


# **Some recipes**
# 
# - `FSSH2-mSDM` with `LD` integrator: the above example defines the recipe for the FSSH2 calculations with the mSDM decoherence correction using the decoherence rates pre-computed above. It relies on the local diabatization (LD) integrator
# 
# - standard `FSSH` with `LD`:
# 
#       dyn_general.update({"tsh_method":0 }) # FSSH
#       dyn_general.update({ "decoherence_algo":-1}) # no (additional) decoherence
#       dyn_general.update({"rep_tdse":1, "electronic_integrator":2 })
#       dyn_general.update({"state_tracking_algo":-1 })
#       dyn_general.update({"do_phase_correction":0 })
#       dyn_general.update({"do_nac_phase_correction":0 })
#       
# - standard `FSSH` but with `NAC-based integrator`:
# 
#       dyn_general.update({"tsh_method":0 }) # FSSH
#       dyn_general.update({ "decoherence_algo":-1}) # no (additional) decoherence
#       dyn_general.update({"rep_tdse":1, "electronic_integrator":5 })
#       dyn_general.update({"state_tracking_algo":21 })
#       dyn_general.update({"do_phase_correction":1 })
#       dyn_general.update({"do_nac_phase_correction":0 })
#       
#   or 
#   
#       dyn_general.update({"tsh_method":0 }) # FSSH
#       dyn_general.update({ "decoherence_algo":-1}) # no (additional) decoherence
#       dyn_general.update({"rep_tdse":1, "electronic_integrator":5 })
#       dyn_general.update({"state_tracking_algo":21 })
#       dyn_general.update({"do_phase_correction":0 })
#       dyn_general.update({"do_nac_phase_correction":1 })
# 

# More conveniently, the recipes may be loaded like this:

# In[13]:


#============== Select the method =====================
# UNCOMMENT AS NEEDED

#dish_rev2023_nbra.load(dyn_general); prf = "DISH"  # DISH
#fssh_nbra.load(dyn_general); prf = "FSSH"  # FSSH
#fssh2_nbra.load(dyn_general); prf = "FSSH2"  # FSSH2
#gfsh_nbra.load(dyn_general); prf = "GFSH"  # GFSH
#ida_nbra.load(dyn_general); prf = "IDA"  # IDA
#mash_nbra.load(dyn_general); prf = "MASH"  # MASH
msdm_nbra.load(dyn_general); prf = "MSDM"  # MSDM


# Now, we define the nuclear and electronic parameters. The nuclear parameters do not matter in the NBRA-NAMD. The initial state is defined in the `elec_params` with the `istate` parameter. `rep` shows in which representation the dynamics should be done where `1` is the adiabatic representation and `0` shows the dynamics in diabatic representation.
# 
# Other parameters in the `elec_params` are used for density matrix and amplitudes initialization.

# In[14]:


#=================== Dynamics =======================
# Nuclear DOF - these parameters don't matter much in the NBRA calculations
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

# Amplitudes are sampled
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

elec_params.update( {"init_type":1,  "rep":1,  "istate":ISTATE } )  # how to initialize: random phase, adiabatic representation, given initial state


# ## 6. Run NAMD simulation <a name="run_namd"></a>
# [Back to TOC](#toc)
# 
# Here, we will conduct NAMD calculations using only one method - the FSSH, to focus on the practice of doing such calculations. We need to define the key parameters in `dyn_general`. We run the calculations for initial conditions in `range(0,200,25)`. For running each calculations we update the `prefix` to `F'{method}_NBRA_icond_{icond}'`. We then will be using these folders and date stored in them to plot the dynamics and compute the timescales.
# 

# In[15]:


ICONDS = [0] #list(range(0,200,25))


# One of the points of the below examples is to show that multithreading execution is indeed much faster than the consecutive one - so pay attention to the timing for each cell

# ### 6.1. Run all initial conditions in a consecutive way
# <a name="6.1"></a>[Back to TOC](#toc)

# In[16]:


get_ipython().run_cell_magic('time', '', 'rnd = Random()\n\nfor icond in ICONDS:\n    print(\'Running the calculations for icond:\', icond)\n    model_params.update({"icond": icond})\n    dyn_general.update({"prefix":F"MSDM_NBRA_icond_{icond}"})\n    res = tsh_dynamics.generic_recipe(dyn_general, compute_model, model_params, elec_params, nucl_params, rnd)\n')


# ### 6.2. With pooling - using multiprocessing
# <a name="6.2"></a>[Back to TOC](#toc)
# 
# We first define the function to be executed

# In[ ]:


def function1(icond):
    prf = "FSSH"
    
    time.sleep(icond * 0.1 )
    rnd = Random()

    mdl = dict(model_params)
    mdl.update({"icond": icond})  #create separate cop
    dyn_gen = dict(dyn_general)
    
    dyn_gen.update({"prefix":F"{prf}_icond_{icond}", "prefix2":F"{prf}_icond_{icond}" })
    res = tsh_dynamics.generic_recipe(dyn_gen, compute_model, mdl, elec_params, nucl_params, rnd)


# And then we run a pool of processes - for various initial conditions

# In[53]:


get_ipython().run_cell_magic('time', '', '\n################################\nnthreads = 4\n################################\npool = mp.Pool(nthreads)\npool.map(function1, ICONDS)\npool.close()\npool.join()\n')


# ## 7. Plot results 
# <a name="plot_res"></a>[Back to TOC](#toc)
# 
# ### 7.1. Plot the results using `plot_dynamics` <a name="plot_res_1"></a>
# <a name="7.1"></a>[Back to TOC](#toc)
# 
# In this cell, we use the function `tsh_dynamics_plot.plot_dynamics` to plot the dynamics for only one initial condition, `icond:0`. The user can change the range of states to be plotted by changing `which_adi_states`. The rest of the parameters are used for plotting and are descriptive. Here, we only plot the dynamics for FSSH and the first initial condition.
# 
# Here, we are going to compare the results printed into different folders:
# 
# - `FSSH_NBRA_icond_0` - from the sequential execution of all initial conditions
# 
# - `FSSH_icond_0` - from the pool-based execution of all initial conditions

# In[ ]:


pref = "MSDM_NBRA_icond_0"

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


# In[55]:


pref = "FSSH_icond_0"

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


# ### 7.2. Average decoherence times map <a name="plot_res_2"></a>
# [Back to TOC](#toc)

# In[56]:


plt.figure()
avg_deco = np.loadtxt('decoherence_times.txt')
nstates = avg_deco.shape[0]
plt.imshow(np.flipud(avg_deco), cmap='hot', extent=(0,nstates,0,nstates))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('fs')
plt.clim(vmin=0, vmax=30)
plt.title(F'Decoherence time')
plt.tight_layout()
plt.show()


# ### 7.3. Computing lifetimes <a name="plot_res_3"></a>
# [Back to TOC](#toc)
# 
# 
# After the dynamic is done, we load all the NA-MD results using `h5py` and start fitting them to a exponential function of the form:
# 
# $$P(t; E_0)=\exp(-(\frac{t}{\tau})^2)$$
# 
# 
# Then, the average time scale is computed for the fits that has an $R^2$ value more than $0.1$. The error bars are computed using the following formula as above:
# 
# $$\epsilon=Z\frac{s}{\sqrt{N}}$$
# 
# where $s$ is the stadard deviaton and $N$ is the number of samples (the ones that have $R^2$ value of more than $0.1$). The $Z$ value is the confidence interval coefficient which for confidence interval of $95\%$, a value of $1.96$ is chosen.
# 
# In above, we will consider the recovery dynamics population of the first and second excited states. Also, please note that this is just an example and we want to show how the fitting works. So, we consider a low $R^2$ value. For better and more accurate results, we need not only longer dynamics but also larger number of surface hopping trajectories.
# 
# Same as above, we are going to compare the results for two sets of results:
# 
# - `FSSH_NBRA_icond_0` - from the sequential execution of all initial conditions
# 
# - `FSSH_icond_0` - from the pool-based execution of all initial conditions

# #### Processing data from the consecutive run:

# In[57]:


# Define function with beta optimization
def exp_funct(t, _x, beta):
    #return 1 - np.exp(-np.power(t / _x, beta)) # for ground state
    return np.exp(-np.power(t / _x, beta)) # for initial state

# Create a single figure
fig, ax = plt.subplots(figsize=(10, 10))  # Single subplot for combined plot

methods = ['FSSH']#,  'MSDM', 'FSSH2']
colors = ['red']#, 'green', 'black']  # Unique colors for each method

# Store timescale strings for both methods
timescale_labels = []

# Loop over methods and apply different exponential functions based on the method
for idx, method in enumerate(methods):
    taus = []
    betas = []

    for icond in ICONDS:
        try:
            # Load data safely
            with h5py.File(f'{method}_NBRA_icond_{icond}/mem_data.hdf', 'r') as F:
                #sh_pop = np.array(F['sh_pop_adi/data'][:, 0])
                sh_pop = np.array(F['sh_pop_adi/data'][:, ISTATE])
                #sh_pop2 = np.array(F['sh_pop_adi/data'][:, ISTATE+1])
                #sh_pop3 = np.array(F['sh_pop_adi/data'][:, ISTATE+2])
                #sh_pop = np.array(F['sh_pop_adi/data'][:, 0])
                md_time = np.array(F['time/data'][:]) * units.au2fs
                
                ax.plot(md_time, sh_pop, color="black")
                #ax.plot(md_time, sh_pop2, color="blue")
                #ax.plot(md_time, sh_pop3, color="green")
                

            # Initial guesses: tau = 100 fs, beta = 1 (exponential)
            p0 = [100.0, 1.0]

            # Fit data using exp_funct (optimizing tau and beta)
            popt, pcov = curve_fit(exp_funct, md_time, sh_pop, p0=p0, bounds=([0.0, 0.0], [np.inf, 5.0]))
            _tau, _beta = popt

            # Compute R-squared
            residuals = sh_pop - exp_funct(md_time, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((sh_pop - np.mean(sh_pop))**2)
            r_squared = 1.0 - (ss_res / ss_tot)

            print(f"Method: {method}, IC {icond}, Tau: {_tau:.2f}, Beta: {_beta:.2f}, R2 = {r_squared}")

            if r_squared > 0.0:
                taus.append(_tau)
                betas.append(_beta)

        except Exception as e:
            print(f"Error processing {method}, IC {icond}: {e}")
            continue
           
    N = 0
    if len(taus) > 0:
        taus = np.array(taus)
        betas = np.array(betas)
        ave_tau = np.average(taus)
        ave_beta = np.average(betas)
        s_tau = np.std(taus)
        s_beta = np.std(betas)
        Z = 1.96
        N = len(taus)
        error_tau = Z * s_tau / np.sqrt(N)
        error_beta = Z * s_beta / np.sqrt(N)
    else:
        ave_tau, error_tau, ave_beta, error_beta = 0, 0, 0, 0  # Default if no data
        
    print(F"The number of successful runs = {N}")

    # Convert timescale to ps
    ave_tau_ps = ave_tau / 1000
    error_tau_ps = error_tau / 1000
    print(f'Timescales for {method}: {ave_tau_ps:.3f} + {error_tau_ps:.3f} ps, Beta: {ave_beta:.3f} + {error_beta:.3f}')

    timescale_labels.append(f"{method}: {ave_tau_ps:.3f} +  {error_tau_ps:.3f} ps, Beta = {ave_beta:.3f} + {error_beta:.3f}")
    timescale_labels = [label.replace("MSDM", "mSDM") for label in timescale_labels]

    if N > 0:
        ax.plot(md_time, exp_funct(md_time, ave_tau - error_tau, ave_beta), ls='--', linewidth=2, color=colors[idx])
        ax.plot(md_time, exp_funct(md_time, ave_tau, ave_beta), ls='-', linewidth=6, color=colors[idx], label=f"{method}")
        ax.plot(md_time, exp_funct(md_time, ave_tau + error_tau, ave_beta), ls='--', linewidth=2, color=colors[idx])
        
ax.set_xlabel("Time (fs)", fontsize=30)
ax.set_ylabel("Initial State Population", fontsize=30)
ax.set_title("Example", fontsize=32)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=28)

# Add timescale text
timescale_text = "\n".join(timescale_labels)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)

# Save and show plot
#plt.savefig('GS_pop_recovery_TiO2_4_300K.png', dpi=600, bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()


# #### Same code - but processing data from the multithreading run:
# 
# The point here is to check whether the multithreading produces the same results and consecutive execution of the calculation. If some race conditions or unintended data sharing across different trajectories is happenining, it may affect the outcomes of the multithreading calculations. However, this doesn't seem to be happening, which is good

# In[58]:


# Define function with beta optimization
def exp_funct(t, _x, beta):
    #return 1 - np.exp(-np.power(t / _x, beta)) # for ground state
    return np.exp(-np.power(t / _x, beta)) # for initial state

# Create a single figure
fig, ax = plt.subplots(figsize=(10, 10))  # Single subplot for combined plot

methods = ['FSSH']#,  'MSDM', 'FSSH2']
colors = ['red']#, 'green', 'black']  # Unique colors for each method

# Store timescale strings for both methods
timescale_labels = []

# Loop over methods and apply different exponential functions based on the method
for idx, method in enumerate(methods):
    taus = []
    betas = []

    for icond in ICONDS:
        try:
            # Load data safely
            with h5py.File(f'{method}_icond_{icond}/mem_data.hdf', 'r') as F:
                #sh_pop = np.array(F['se_pop_adi/data'][:, 0])
                sh_pop = np.array(F['sh_pop_adi/data'][:, ISTATE])
                #sh_pop2 = np.array(F['sh_pop_adi/data'][:, ISTATE+1])
                #sh_pop3 = np.array(F['sh_pop_adi/data'][:, ISTATE+2])
                #sh_pop = np.array(F['sh_pop_adi/data'][:, 0])
                md_time = np.array(F['time/data'][:]) * units.au2fs
                
                ax.plot(md_time, sh_pop, color="black")
                #ax.plot(md_time, sh_pop2, color="blue")
                #ax.plot(md_time, sh_pop3, color="green")
                

            # Initial guesses: tau = 100 fs, beta = 1 (exponential)
            p0 = [100.0, 1.0]

            # Fit data using exp_funct (optimizing tau and beta)
            popt, pcov = curve_fit(exp_funct, md_time, sh_pop, p0=p0, bounds=([0.0, 0.0], [np.inf, 5.0]))
            _tau, _beta = popt

            # Compute R-squared
            residuals = sh_pop - exp_funct(md_time, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((sh_pop - np.mean(sh_pop))**2)
            r_squared = 1.0 - (ss_res / ss_tot)

            print(f"Method: {method}, IC {icond}, Tau: {_tau:.2f}, Beta: {_beta:.2f}, R2 = {r_squared}")

            if r_squared > 0.0:
                taus.append(_tau)
                betas.append(_beta)

        except Exception as e:
            print(f"Error processing {method}, IC {icond}: {e}")
            continue
           
    N = 0
    if len(taus) > 0:
        taus = np.array(taus)
        betas = np.array(betas)
        ave_tau = np.average(taus)
        ave_beta = np.average(betas)
        s_tau = np.std(taus)
        s_beta = np.std(betas)
        Z = 1.96
        N = len(taus)
        error_tau = Z * s_tau / np.sqrt(N)
        error_beta = Z * s_beta / np.sqrt(N)
    else:
        ave_tau, error_tau, ave_beta, error_beta = 0, 0, 0, 0  # Default if no data
        
    print(F"The number of successful runs = {N}")

    # Convert timescale to ps
    ave_tau_ps = ave_tau / 1000
    error_tau_ps = error_tau / 1000
    print(f'Timescales for {method}: {ave_tau_ps:.3f} + {error_tau_ps:.3f} ps, Beta: {ave_beta:.3f} + {error_beta:.3f}')

    timescale_labels.append(f"{method}: {ave_tau_ps:.3f} +  {error_tau_ps:.3f} ps, Beta = {ave_beta:.3f} + {error_beta:.3f}")
    timescale_labels = [label.replace("MSDM", "mSDM") for label in timescale_labels]

    if N > 0:
        ax.plot(md_time, exp_funct(md_time, ave_tau - error_tau, ave_beta), ls='--', linewidth=2, color=colors[idx])
        ax.plot(md_time, exp_funct(md_time, ave_tau, ave_beta), ls='-', linewidth=6, color=colors[idx], label=f"{method}")
        ax.plot(md_time, exp_funct(md_time, ave_tau + error_tau, ave_beta), ls='--', linewidth=2, color=colors[idx])
        
ax.set_xlabel("Time (fs)", fontsize=30)
ax.set_ylabel("Initial State Population", fontsize=30)
ax.set_title("Example", fontsize=32)
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=28)

# Add timescale text
timescale_text = "\n".join(timescale_labels)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)

# Save and show plot
#plt.savefig('GS_pop_recovery_TiO2_4_300K.png', dpi=600, bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()


# ## 8. Additional descriptive analysis: Pre-dynamics
# <a name="8"></a>[Back to TOC](#toc)
# 

# ### 8.1. Convert the matrices to NumPy for convenience
# <a name="8.1"></a>[Back to TOC](#toc)

# In[59]:


numpy_st_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )
numpy_ham_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )
numpy_nac_adi = np.zeros( (NSTEPS, NSTATES, NSTATES) )

for i in range(NSTEPS):
    numpy_st_adi[i, :, :] = data_conv.MATRIX2nparray(St_adi[i]).real
    numpy_ham_adi[i, :, :] = data_conv.MATRIX2nparray(Ham_adi[i]).real
    numpy_nac_adi[i, :, :] = data_conv.MATRIX2nparray(NAC_adi[i]).real


# ### 8.2. Excitation energies vs time
# <a name="8.2"></a>[Back to TOC](#toc)

# In[60]:


# ----------------------------
# Extract excitation energies
# ----------------------------
time = np.arange(0, NSTEPS)

# ----------------------------
# Plot all excitation energies
# ----------------------------
plt.figure(figsize=(8,5))
# Adjust according to your system
#plt.ylim(3.5,5)
plt.ylim(0.0,1.0)
for i in range(NSTATES):
    plt.plot(time, numpy_ham_adi[:, i, i]  * units.au2ev, label=f"$S_{i}$")

plt.xlabel("Time step", fontsize=16)
plt.ylabel("Excitation energy (eV)", fontsize=16)
plt.title("Adiabatic excitation energies", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


# ### 8.3. Gap distributions
# <a name="8.3"></a>[Back to TOC](#toc)

# In[61]:


# Adjacent gaps
all_gaps = []
for i in range(NSTATES-1):
    all_gaps.append(numpy_ham_adi[:, i+1, i+1] - numpy_ham_adi[:, i, i])
all_gaps = np.array(all_gaps)  * units.au2ev


bins = np.linspace(np.min(all_gaps), np.max(all_gaps), 100)

plt.figure(figsize=(9,6))

for i in range(all_gaps.shape[0]):
    hist, edges = np.histogram(all_gaps[i], bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.plot(centers, hist, linewidth=2.5, label=f"$S_{i+1}-S_{i}$")

plt.xlabel("Energy gap (eV)", fontsize=22)
plt.ylabel("Probability density", fontsize=22)
plt.title("Gap distributions", fontsize=24)
#plt.xlim(0,0.25)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(ncol=2, fontsize=14, frameon=False)
plt.tight_layout()
plt.show()


# ### 8.4. NAC map
# <a name="8.4"></a>[Back to TOC](#toc)

# In[62]:


nacs = numpy_nac_adi #(St - np.transpose(St, (0, 2, 1))) / (2.0 * dt)
avg_coupling_meV = np.mean(np.abs(nacs), axis=0) * units.au2ev * 1000.0 # to meV
np.fill_diagonal(avg_coupling_meV, 0.0)

#nstates = avg_coupling_meV.shape[0]

# keep only pairs within |i-j| <= band
band = 8
masked = avg_coupling_meV.copy()
for i in range(NSTATES):
    for j in range(NSTATES):
        if abs(i - j) > band:
            masked[i, j] = 0.0

fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(masked, origin="lower", cmap="hot", interpolation="nearest", aspect="equal")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("meV", fontsize=20, rotation=0, labelpad=25)
cbar.ax.tick_params(labelsize=15)

ax.set_xlabel("State index", fontsize=24)
ax.set_ylabel("State index", fontsize=24)
ax.set_title("Averaged couplings", fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=18, width=1.8, length=7)

for spine in ax.spines.values():
    spine.set_linewidth(1.8)

plt.tight_layout()
plt.show()


# ### 8.5. NAC distribution (all NAC pairs)
# <a name="8.5"></a>[Back to TOC](#toc)

# In[63]:


# ----------------------------
# Inputs
# ----------------------------
# nacs shape: (nsteps, nstates, nstates)

# ----------------------------
# Gather all unique pair couplings
# ----------------------------
all_couplings = []

for i in range(NSTATES):
    for j in range(i + 1, NSTATES):
        dij = np.abs(nacs[:, i, j])          # shape: (nsteps,)
        cij_meV = dij * units.au2ev * 1000.0 # 
        all_couplings.append(cij_meV)

all_couplings = np.concatenate(all_couplings)

# Remove exact zeros for log-scale plotting
all_couplings = all_couplings[all_couplings > 0]

print("Number of samples:", len(all_couplings))
print("min, max (meV):", np.min(all_couplings), np.max(all_couplings))

# ----------------------------
# Log-spaced bins
# ----------------------------
bins = np.logspace(
    np.log10(np.min(all_couplings)),
    np.log10(np.max(all_couplings)),
    80
)

hist, edges = np.histogram(all_couplings, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])   # geometric centers for log bins

# Keep only nonzero histogram values
mask = hist > 0

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,6))
plt.plot(centers[mask], hist[mask], linewidth=3)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$|NAC_{ij}|$, meV", fontsize=22)
plt.ylabel("PD, 1/meV", fontsize=22)
plt.title("Coupling distribution", fontsize=24)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()


# ### 8.6. NAC distribution (only adjacent pairs)
# <a name="8.6"></a>[Back to TOC](#toc)

# In[64]:


# ----------------------------
# Inputs
# ----------------------------
# nacs shape: (nsteps, nstates, nstates)

# ----------------------------
# Gather all unique pair couplings
# ----------------------------
all_couplings = []

for i in range(NSTATES-1):
    dij = np.abs(nacs[:, i, i+1])          # shape: (nsteps,)
    cij_meV = dij * units.au2ev * 1000.0 # 
    all_couplings.append(cij_meV)

all_couplings = np.concatenate(all_couplings)

# Remove exact zeros for log-scale plotting
all_couplings = all_couplings[all_couplings > 0]

print("Number of samples:", len(all_couplings))
print("min, max (meV):", np.min(all_couplings), np.max(all_couplings))

# ----------------------------
# Log-spaced bins
# ----------------------------
bins = np.logspace(
    np.log10(np.min(all_couplings)),
    np.log10(np.max(all_couplings)),
    50
)

hist, edges = np.histogram(all_couplings, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])   # geometric centers for log bins

# Keep only nonzero histogram values
mask = hist > 0

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,6))
plt.plot(centers[mask], hist[mask], linewidth=3)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$|NAC_{ij}|$, meV", fontsize=22)
plt.ylabel("PD, 1/meV", fontsize=22)
plt.title("Coupling distribution", fontsize=24)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()


# ### 8.7. Time-overlaps diagnostics
# <a name="8.7"></a>[Back to TOC](#toc)
# 
# If the determinant of the time-overlap matrix is close to zero (or just very small) at some points - this indicates that some of the states may be projecting outside of the selected basis of excited states. In these situations, using LD integrators may be dangerous.
# 
# You can vary the `nact_st` below to see for which value the determinant is not close to zero. This may suggest what active space of CI excitations to use

# In[65]:


for nact_st in range(0, NSTATES+1):
    min_val = np.min( np.abs( np.linalg.det(numpy_st_adi[:,:nact_st,:nact_st]) ) ) 
    print(F"nact_st = {nact_st}, min_val = {min_val}")


# As we can see, the larger number of excited states stored in the time-overlap matrix generally leads to a decreased determinant of this matrix 

# In[66]:


nact_st = 1

plt.figure(figsize=(3.21*2,2.41*2))
determinants = []
for i in range(NSTEPS):
    determinants.append(np.linalg.det(numpy_st_adi[i][:nact_st,:nact_st]  )) 
plt.plot(np.arange(len(determinants)), np.abs(determinants))
plt.xlabel('Timestep')
plt.ylabel('|Determinant|')
plt.tight_layout()
plt.show()


# In[67]:


plt.figure(figsize=(3.21*2,2.41*2))
x = []
tim = np.linspace(0, NSTEPS)
#for i in range(NSTATES):
#    plt.plot(np.abs( np.linalg.det( numpy_st_adi[:, :, :] ) ) )
plt.plot(np.abs( numpy_st_adi[:, 1, 2] ) )


print(x)
plt.xlabel('Timestep')
plt.ylabel('|Determinant|')
plt.tight_layout()
plt.show()


# In[68]:


x = np.where(np.abs( numpy_st_adi[:, 1, 2] )>0.3)
print(x)


# In[ ]:




