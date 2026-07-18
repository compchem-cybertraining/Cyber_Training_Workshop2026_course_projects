#!/usr/bin/env python
# coding: utf-8

# # Computing the nonadiabatic couplings in Kohn-Sham and excited states bases
# 
# In this tutorial, we will start computing the nonadiabatic couplings (NACs) from the molecular orbital overlap files obtained in [step2](../../7_step2_cp2k/1_DFT). The NACs will be computed in Kohn-Sham states and also in both single-partcile (SP) and many-body (MB) excited state bases. 
# 
# ## Table of contents
# <a name="toc"></a>
# 1. [Importing needed libraries](#import)
# 2. [Overview of required files](#required_files)
# 3. [Computing NACs](#comp_nacs)     \
#    3.1. [Kohn-Sham basis](#KS)\
#    3.2. [Excited state basis](#excited_states)
# 4. [Plot results](#plot_res)\
#    4.1. [Energy vs time](#ene_time)\
#    4.2. [Average density of states](#ave_pdos)\
#    4.3. [NAC map](#nac_map)\
#    4.4. [NAC distribution](#nac_dist)\
#    4.5. [Influence spectrum](#inf_spec)
#    
# 
# ### A. Learning objectives
# 
# * To be able to compute NACs in Kohn-Sham and excited state bases
# * To be able to plot the computed excited states energies vs time
# * To be able to plot the average partial density of states
# * To be able to plot the NAC maps and distributions
# * To be able to compute and plot the influence spectrum
# 
# ### B. Use cases
# 
# * [Computing NACs](#comp_nacs)
# * [Plot results](#plot_res)
# 
# ### C. Functions
# 
# - `libra_py`
#   - `data_stat`
#     - [`cmat_distrib`](#nac_dist)
#   - `influence_spectrum`
#     - [`recipe1`](#inf_spec)
#   - `workflows`
#     - `nbra`
#       - [`step3`](#comp_nacs)
#         - [`run_step3_ks_nacs_libint`](#KS)
#         - [`run_step3_sd_nacs_libint`](#excited_states)
#   - `units`
#     - `au2ev`

# ## 1. Importing needed libraries <a name="import"></a>
# [Back to TOC](#toc)
# 
# Import `step3` modules which is used to compute the NACs in Kohn-Sham basis using `run_step3_ks_nacs_libint` and in excited states basis using `run_step3_sd_nacs_libint`. 

# In[2]:


import os
import glob
import numpy as np
import scipy.sparse as sp
from libra_py import units, data_stat, influence_spectrum
import matplotlib.pyplot as plt
from liblibra_core import *
from libra_py.workflows.nbra import step3
import libra_py.packages.cp2k.methods as CP2K_methods


from libra_py.workflows.nbra import mapping



def _energy_arb_fixed(SD, e):

    """Sum of orbital energies for the spin-orbitals occupied in SD.

    Same as mapping.energy_arb, but correctly unpacks the

    (indices, parity) tuple returned by the new sd2indx()."""

    if isinstance(e, np.ndarray):

        nbasis = e.shape[0]

        sd, _parity = mapping.sd2indx(SD, nbasis)

        res = 0.0

        for i in sd:

            res += e[i, i]

    else:  # CMATRIX / MATRIX

        nbasis = e.num_of_rows

        sd, _parity = mapping.sd2indx(SD, nbasis)

        res = 0.0 + 0.0j

        for i in sd:

            res += e.get(int(i), int(i))

    return res



mapping.energy_arb = _energy_arb_fixed


# ## 2. Overview of required files <a name="required_files"></a>
# [Back to TOC](#toc)
# 
# * `../../7_step2_cp2k/2_xTB/2_hpc/res`
# 
# The MO overlap files are needed and stored in this folder.
# 
# * `../../7_step2_cp2k/2_xTB/2_hpc/all_logfiles`
# 
# All of the logfiles obtained from the electronic structure calculations of CP2K. These files will be needed to find the Kohn-Sham HOMO index.

# ## 3. Computing NACs <a name="comp_nacs"></a>
# [Back to TOC](#toc)
# 
# ### 3.1. Kohn-Sham basis <a name="KS"></a>
# 
# The `libra_py.workflow.nbra.step3.run_step3_ks_nacs_libint(params)` computes the NACs between pairs of Kohn-Sham states using the molecular orbitals time-overlaps. 
# 
# Libra stores the overlap data in 2-spinor format as follows:
# 
# <div>
# <img src="./active_space_0.png" width="500"/>
# </div>
# 
# With no spin-orbit couplings, two blocks of the matrix is zero. Since most of the elements of the overlap matrices are zero, we use the `scipy.sparse` library for storing and loading them. 
# 
# The paramters for this function are as follows:
# 
# `params['lowest_orbital']`: The lowest orbital considered in the computation of the MO overlaps. This value is exactly the same
# as in the `run_template.py` file in step2.
# 
# `params['highest_orbital']`: The highest orbital considered in the computation of the MO overlaps. This value is exactly the same
# as in the `run_template.py` file in step2.
# 
# `params['num_occ_states']`: The number of occupied orbitals to be considered from HOMO to lower occupied states. This value is defined by user.
# 
# `params['num_unocc_states']`: The number of unoccupied orbitals to be considered from LUMO to higher unoccupied states. This value is defined by user.
# 
# The two values above are used to create an active space which then will be used to select the elements from the MO overlap and 
# energy matrices. 
# 
# `params['use_multiprocessing']`: A boolean flag to use the multiprocessing library of Python or not.
# 
# `params['nprocs']`: The number of processors to be used for the calculations. Libra will use this only if the `params['use_multiprocessing']` 
# is set to `True`.
# 
# `params['time_step']`: The time-step used in the calculations in `fs`.
# 
# `params['es_software']`: The name of the software package used to compute the electronic structure calculations. This will be used to generate the HOMO 
# index of that system so it can build the active space.
# 
# `params['path_to_npz_files']`: The full path to the MO overlap files.
# 
# `params['logfile_directory']`: The full path to the folder where all the log files are stored.
# 
# `params['path_to_save_ks_Hvibs']`: The full path to the folder in which the NACs between the Konh-Sham states are stored.
# 
# `params['start_time']`: The start time-step.
# 
# `params['finish_time']`: The finish time-step.
# 
# 
# After setting all the above paramters, the calculations are run using `step3.run_step3_ks_nacs_libint(params)`.
# 

# In[ ]:


params_ks = {
              'lowest_orbital': 42, 'highest_orbital': 63, 'num_occ_states': 10, 'num_unocc_states': 10,
              'use_multiprocessing': True, 'nprocs': 16, 'time_step': 1.0, 'es_software': 'cp2k',
              'path_to_npz_files': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/res',
              'logfile_directory': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/all_logfiles',
              'path_to_save_ks_Hvibs': os.getcwd()+'/res-ks-xTB',
              'start_time': 0, 'finish_time': 3999,
            }

# For KS states
step3.run_step3_ks_nacs_libint(params_ks)


# ### 3.2. Excited state basis <a name="excited_states"></a>
# [Back to TOC](#toc)
# 
# Below, we will be using `step3.run_step3_sd_nacs_libint` function to compute the time-overlaps and nonadiabatic couplings between excited states basis. This can be done either in the many-body or single-particle basis. A schematic of the workflow for selecting new active space is shown below:
# 
# 
# Some parameters are common with the ones used to run `step3.run_step3_ks_nacs_libint(params)` above.
# 
# Other parameters needed to run the `step3.run_step3_sd_nacs_libint(params)` function are as follows:
# 
# `params['isUKS']`: A boolean flag for unrestricted spin calculations.
# 
# `params['is_many_body']`: If set to `True`, the NACs will be computed between pairs of many-body (TD-DFT) states. Also, the NACs between single-particle 
# SDs obtained from the TD-DFT results will be computed as well. Otherwise, only single-particle NACs will be computed only for the SDs obtained from
# `num_occ_states` and `num_unocc_states`. This will be used for xTB calculations in which no TD-DFT was performed.
# 
# `params['number_of_states']`: The number of TD-DFT states to consider. This value should not exceed the number of requested TD-DFT states in the CP2K
# calculations.
# 
# `params['tolerance']`: A lower bound for selection of the excitation with configuration interaction coefficients higher than this value.
# 
# `params['verbosity']`: An integer value showing the printing level. The default is set to 0. Higher values will print more data on the terminal.
# 
# `params['sorting_type']`: After defining the SDs, Libra will sort them either based on `'energy'` or `'identity'`.
# 
# 
# The NACs can also be computed between excited states. These include the single-particle and many-body bases which the latter is obtained from the
# TD-DFT calculations. First, we need to compute the overlap between excited state Slater-determinants (SDs) then they will be used to compute the NACs
# between them. For many-body states, the configuration interaction coefficietns will be used. We will consider both single-particle 
# and many-body for DFT calculations but only single-particle for xTB.
# 
# There are different ways of defining the excited states SDs (the single-particle excited state basis). The first is through 
# defining the `num_occ_states` and `num_unocc_states` in which Libra
# will start making the SDs from all of the occupied states (starting from `HOMO-num_occ_states+1`) to all of the unoccupied states (ends
# to `LUMO+num_unocc_states-1`). Also, if the unrestricted spin calculation flag is set to `True`, the SDs will be made for both alpha and beta spin channels. 
# 
# For example, if you want to build the electron-only excitation basis, you need to set `params['num_occ_states'] = 1` and set `params['num_unocc_states']`
# to a value less than the number of unoccupied orbitals that was considered in the computation of overlaps. This will generate all the electron-only
# excitation from HOMO to unoccupied states.
# 
# If the TD-DFT calculations has been done, then Libra will go over all log files and 
# generate all the SDs used for all the steps and therefore the definition of these SDs is automatic and Libra will replace the `num_occ_states` and
# `num_unocc_states` itself based on the SDs that were generated from the TD-DFT log files. 
# 
# 
# Here, we only compute the NACs for single-particle basis since in our xTB calculations we didn't use any TDDFT calculations.

# In[ ]:


params_sd = {

          'lowest_orbital': 42, 'highest_orbital': 63, 'num_occ_states': 10, 'num_unocc_states': 10,

          'isUKS': 0, 'number_of_states': 10, 'tolerance': 0.01, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': 16,

          'is_many_body': False, 'time_step': 1.0, 'es_software': 'cp2k',

          'path_to_npz_files': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/res',

          'logfile_directory': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/all_logfiles',

          'path_to_save_sd_Hvibs': os.getcwd()+'/res-sd-xTB',

          'start_time': 0, 'finish_time': 3999, 'sorting_type': 'identity',

         }

step3.run_step3_sd_nacs_libint(params_sd)


# ## 4. Plot results <a name="plot_res"></a>
# [Back to TOC](#toc)
# 
# ### 4.1. Energy vs time <a name="ene_time"></a>
# 
# Here, we will plot the energies of single-particle excitation basis.

# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
titles = ['SP basis']
plt.figure()
basis = 'sd'
dt = 1.0 # fs
energies = []
for step in range(1500,1699):
    file = F'res-sd-xTB/Hvib_{basis}_{step}_re.npz'
    energies.append(np.diag(sp.load_npz(file).todense().real))
energies = np.array(energies)*units.au2ev
md_time = np.arange(0,energies.shape[0]*dt,dt)
for i in range(energies.shape[1]):
    plt.plot(md_time, energies[:,i]-energies[:,0])

plt.title('C$_3$N$_4$ SP basis')
plt.ylabel('Excitation energy, eV')
plt.xlabel('Time, fs')
plt.tight_layout()


# ### 4.2. Average density of states <a name="ave_pdos"></a>
# [Back to TOC](#toc)
# 
# In this section, we will plot the average partial density of states (pDOS) over the MD trajectory. There are two ways to take the average of the pDOS:
# 
# 1- Average all the pDOS files and then convolve the average pDOS for each element.
# 2- Convolve the pDOS files and then take the average for each element.
# 
# We choose the first one due to two reasons. First, the computational cost is much lower and we only need one convolution. Second is that averaging over the grid points (using the method 2) is dependent on the number of grid points we use for convolution which again adds to the complexity of the procedure. 
# 
# Here, we will use normalized Gaussian function for weighting the pDOS values and summing them.
# 
# $$f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
# 
# This function is defined in the `CP2K_methods.gaussian_function`.
# 
# Now, we plot the pDOS for all of the angular momentum components of each atom. This is done by using the `orbital_cols`. In fact, the `orbital_cols` is related to `orbitals`. For example, for `s` orbital, we consider the 3rd index and for `p` orbital, we sum the columns from 4 to 6 (`range(4,7)`). Here we want to show how the code works and how the you can modify that based on your project. In the next section, we will show the pDOS only for atoms and sum all the components in each row of the pdos file. Other parameters are as follows:
# 
# `atoms`: The atoms names and numbers as appear in the `pdos` files which will be used in convolution, labeling and plotting. The atoms order should be exactly the same as appear in the `.pdos` files. For example, the `*k1*.pdos` files contain the pDOS data for `C` atom and `*k2*.pdos` files contain the data for the `N` atom. Therefore, we set `"atoms": [[1,2] , ['C', 'N']]`.
# 
# `npoints`: The number of grid points for making the Gaussian functions. Note that, this value should be more than the number of states in the `.pdos` files.
# 
# `sigma`: The standard deviation in eV.
# 
# `shift`: This value shifts the minimum and maximum energy found in the `pdos_ave` and will extend the boundaries from both sides by `shift`eV.
# 
# Finally, we will plot the total density of states. We manually set the HOMO energy level to zero.

# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')
params = {"path_to_all_pdos": os.getcwd()+'/../../7_step2_cp2k/2_xTB/2_hpc/all_pdosfiles', "atoms": [[1,2] , ['C', 'N']],
          "orbitals_cols": [[3], range(4,7), range(7,12), range(12,19)], "orbitals":  ['s','p','d','f'],
          "npoints": 4000, "sigma": 0.05, "shift": 2.0}
ave_energy_grid, homo_energy, ave_pdos_convolved, pdos_labels, ave_pdos_convolved_total = CP2K_methods.pdos(params)
for i in range(len(pdos_labels)):
    pdos_label = pdos_labels[i]
    plt.plot(ave_energy_grid-homo_energy, ave_pdos_convolved[i], label=pdos_label)
plt.plot(ave_energy_grid-homo_energy, ave_pdos_convolved_total, color='black', label='Total')
plt.legend()
plt.xlim(-4,4)
plt.ylabel('pDOS, 1/eV')
plt.xlabel('Energy, eV')
plt.title('C$_3$N$_4$, 300 K')
plt.tight_layout()


# ### 4.3. NAC map <a name="nac_map"></a>
# [Back to TOC](#toc)
# 
# One way of visualizing the NAC values is to plot the average NAC matrix using `plt.imshow`. 

# In[10]:


get_ipyth:on().run_line_magic('matplotlib', 'notebook')
plt.figure()
nac_files = glob.glob(F'res-sd-xTB/Hvib_sd*im*')
for c2, nac_file in enumerate(nac_files):
    nac_mat = sp.load_npz(nac_file).todense().real
    if c2==0:
        nac_ave = np.zeros(nac_mat.shape)
    nac_ave += np.abs(nac_mat)
nac_ave *= 1000*units.au2ev/c2
nstates = nac_ave.shape[0]
plt.imshow(np.flipud(nac_ave), cmap='hot', extent=(0,nstates,0,nstates))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('meV')
# plt.clim(vmin=0, vmax=15)
plt.title('SP NACs')
plt.tight_layout()


# ### 4.4. NAC distribution <a name="nac_dist"></a>
# [Back to TOC](#toc)
# 
# Another intuitive way to visualize the NACs is to plot the distribution of the NACs. Here we plot them for SP and MB excited states. A smoother distribution plot is obtained if more steps are involved.
# 
# For computing the probability distribution of the couplings within the range of $0$ to $50$ meV, we will be using `data_stat.mat_distrib` function.

# In[11]:


help(data_stat.cmat_distrib)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'notebook')

nac = []
nac_files = glob.glob(F'res-sd-xTB/Hvib_sd*im*')
for nac_file in nac_files:
    hvib = sp.load_npz(nac_file)
    hvib_dense = hvib.todense().real
    for i in range(hvib.shape[0]):
        for j in range(hvib.shape[0]):
            if j != i:
                nac_ij = np.abs(hvib_dense[i,j])* 1000.0 * units.au2ev
                x_mb = MATRIX(1,1)
                x_mb.set(0, 0, nac_ij )
                nac.append( x_mb )
bin_supp, dens, cum = data_stat.cmat_distrib( nac, 0, 0, 0, 0, 50, 0.1)
plt.plot( bin_supp, dens, label='SP')
plt.xlabel('|NAC|, meV')
plt.ylabel('PD, 1/meV')
plt.title('NAC distribution of SP states')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
# plt.savefig('nac_dist_1.jpg', dpi=600)


# ### 4.5. Influence spectrum <a name="inf_spec"></a>
# [Back to TOC](#toc)
# 
# 
# In order to compute the influence spectrum for the energy gap fluctuations between two states, `i` and `j`. The parameters that need to be passed to the function `influence_spectrum.recipe1` are as follows:
# 
# `data`: A list of `MATRIX(ndof, 1)` objects i.e. a sequence of real-valued ndof-dimensional vectors
# 
# `dt`: Time step in fs units
#     
# `wspan`: The window of frequencies for the Fourier transform with units cm$^{-1}$
# 
# `dw`: The grid points spacing in the frequency domain with units of cm$^{-1}$
# 
# `do_output`: Whether we print out the data the results into files
# 
# `acf_filename`: The name of the file where to print the ACF
#     
# `spectrum_filename`: The name of the file where to print the spectrum 
# 
# `do_center`: A flag controlling whether to center data (=`True`) or not (=`False`). Centering means we subtract the average value (over all the data points) from all the data points - this way, we convert values into their fluctuations 
# 
# `acf_type`: selector of the convention to to compute ACF:
# ```
#    * 0 : the chemist convention,  (1/(N-h)) Sum_{t=1,N-h} (Y[t]*Y[t+h]) [ default ]
#    * 1 : the statistician convention, (1/N) Sum_{t=1,N-h} (Y[t]*Y[t+h])
# ```
# 
# `data_type`: What is the format of the data?
# ```
#         * 0 : list of MATRIX(ndof, 1) [ default ]
#         * 1 : list of VECTOR
# ```
# 
# The `recipe1` function returns a tuple of (T, norm_acf, raw_acf, W, J, J2), where:
# 
# `T`: List with time axis in `fs` units
# 
# `norm_acf`: List with normalized ACF
# 
# `raw_acf`: List with un-normalized ACF
# 
# `W`: List with frequencies axis with units $cm^{-1}$
# 
# `J`: List with amplitudes of Fourier transform
# 
# `J2`: List with values of $(1/2\pi)|J|^2$
# 
# Select two states of interest to compute the influence spectrum for their energy gap fluctuations. Here we comute that for the ground state, `i=0`, and the first excited state, `j=1`.

# In[13]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# For computing influence spectra
params_inf_cpec = {"dt": 1.0, "wspan": 4000.0, "dw": 1.0, "do_output": False,
          "do_center": True, "acf_type": 1, "data_type": 0}

# Selecting the two states we want to compute the influence spectra for
i = 0; j = 1
# ========== Computing the autocorrelation function and the influence spectra
data_ij = []
for step in range(energies.shape[0]):
    x = MATRIX(1,1)
    if i<j:
        gap = energies[step,j]-energies[step,i]
    else:
        gap = energies[step,i]-energies[step,j]
    x.set(0,0, gap)
    data_ij.append(x)
Tij, ACFij, uACFij, Wij, Jij, J2ij = influence_spectrum.recipe1(data_ij, params_inf_cpec)


plt.plot(Wij, J2ij)
plt.xlabel('Frequency, cm$^{-1}$')
plt.ylabel('Intensity, a.u.')
plt.yticks([])
plt.title(F'Influence spectrum, states: {i},{j}')
plt.tight_layout()

