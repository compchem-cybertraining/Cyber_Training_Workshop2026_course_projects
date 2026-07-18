import os
import sys
import libra_py.packages.cp2k.methods as CP2K_methods
from libra_py.workflows.nbra import step2

path = os.getcwd()
params = {}

params['nprocs'] = 8
params['mpi_executable'] = 'mpirun'
params['cp2k_exe'] = 'cp2k.psmp'

params['istep'] = 
params['fstep'] = 

params['lowest_orbital'] = 42
params['highest_orbital'] = 63

params['isxTB'] = True
params['isUKS'] = False

params['is_periodic'] = True
params['A_cell_vector'] = [8.323183, 0.000000, 0.000000]
params['B_cell_vector'] = [0.000000, 8.264698, 0.000000]
params['C_cell_vector'] = [0.000000, 0.000000, 11.895850]
params['periodicity_type'] = 'XYZ'
origin = [0, 0, 0]
params['translational_vectors'] = CP2K_methods.generate_translational_vectors(origin, [1, 1, 1], 'XYZ')

params['is_spherical'] = True
params['remove_molden'] = True
params['cube_visualization'] = False

params['res_dir'] = path + '/../res'
params['all_pdosfiles'] = path + '/../all_pdosfiles'
params['all_logfiles'] = path + '/../all_logfiles'
params['cp2k_ot_input_template'] = path + '/../es_ot_temp.inp'
params['cp2k_diag_input_template'] = path + '/../es_diag_temp.inp'
params['trajectory_xyz_filename'] = path + '/../traj.xyz'

step2.run_cp2k_libint_step2(params)
