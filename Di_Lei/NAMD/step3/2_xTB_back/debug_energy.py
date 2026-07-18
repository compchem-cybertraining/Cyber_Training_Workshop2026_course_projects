import os
import numpy as np
import scipy.sparse as sp
from libra_py import units, data_stat, influence_spectrum
from liblibra_core import *
from libra_py.workflows.nbra import step3, mapping
import libra_py.packages.cp2k.methods as CP2K_methods

orig_energy_arb = mapping.energy_arb
def patched(SD, e):
    if isinstance(e, np.ndarray):
        nbasis = e.shape[0]
        sd = mapping.sd2indx(SD, nbasis)
        for i in sd:
            val = e[i, i]
            if np.ndim(val) != 0 or np.size(val) != 1:
                print('BAD SD =', SD, ' sd_indices =', sd, ' bad_i =', i, ' e.shape=', e.shape, ' val=', val)
    return orig_energy_arb(SD, e)

mapping.energy_arb = patched

params_sd = {
        'lowest_orbital': 42, 'highest_orbital': 63, 'num_occ_states': 10, 'num_unocc_states': 10,
        'isUKS': 0, 'number_of_states': 10, 'tolerance': 0.01, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': 12,
        'is_many_body': False, 'time_step': 1.0, 'es_software': 'cp2k',
        'apply_phase_correction': True,
        'apply_orthonormalization': True,
        'do_state_reordering': True,
        'state_reordering_alpha': 0.0,
        'path_to_npz_files': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/res',
        'logfile_directory': '/vscratch/grp-cyberwksp21/dlei/Tutorials_Libra/NAMD/2_hpc/all_logfiles',
        'path_to_save_sd_Hvibs': os.getcwd()+'/res-sd-xTB',
        'start_time': 0, 'finish_time': 1, 'sorting_type': 'identity',
        }

try:
    step3.run_step3_sd_nacs_libint(params_sd)
except Exception as ex:
    print('EXCEPTION:', repr(ex))
