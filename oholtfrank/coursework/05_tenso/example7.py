from math import ceil
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath # Propagator
from tenso.prototypes.bath import gen_bcf # Bath correlation function generator

out_name = 'simple_10' # Output name; results are written to 'simple.dat.log'

# Single bath with a Drude-Lorentz spectral density only
bath_simulation = gen_bcf(
    freq_b=[1243], # Central Frequency of the Brownian Spectral density in cm-1
    re_b=[161.6], # Reorganization energy of the Brownian Spectral density in cm-1
    width_b=[10],  # Width of the spectral density in cm-1
    temperature=10, # Temperature in Kelvin
    decomposition_method='Pade', # Decomposition method for the bath correlation function
    n_ltc=3, # Number of low-temperature correction terms
)
# Initialize system Hamiltonian
H = np.array([[0, 0, 0], [0, 1240, 0], [0, 0, 0]], dtype=np.complex128)
H_SB = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128)
H_SB1 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
H_SB2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)

end_time = 200 # End time in fs
dt = 1
wfn = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
# Run a single calculation with the default settings:
# frame_method='tree2', stepwise_method='mix', ps_method='vmf'
propagator = system_multibath(
    fname=out_name,
    init_rdo=np.outer(wfn, wfn.conj()),
    sys_ham=H,
    sys_ops=[H_SB],
    bath_correlations=[bath_simulation],
    dim=20,
    max_auxiliary_rank=45,
    end_time=end_time,
    step_time=dt,
)
progress_bar = tqdm(propagator, total=ceil(end_time / dt))

for _t in (progress_bar):
    progress_bar.set_description(f'@{_t:.2f} fs')
