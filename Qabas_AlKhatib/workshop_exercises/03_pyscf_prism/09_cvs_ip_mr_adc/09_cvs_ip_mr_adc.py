from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
import numpy as np

# Specify geometry and molecular information
mol = gto.Mole()
mol.atom = """
O    0.0000000000     0.0000000000     -0.4477034680
O    0.0000000000     1.0918051524      0.2238517340
O    0.0000000000    -1.0918051524      0.2238517340
"""
mol.basis = "cc-pcvdz"
mol.verbose = 4
mol.charge = 0
mol.max_memory = 10000
mol.build()

# Run RHF calculation as guess for CASSCF
mf = scf.RHF(mol).density_fit()
mf.kernel()

# CASSCF(6e,6o) calculation
mc = mcscf.CASSCF(mf, 6, 6).density_fit()
emc = mc.mc1step()[0]

# CVS-IP-MR-ADC calculation to simulate the ground-state oxyten K-edge XPS
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit()
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)-x"
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 3
mr_adc.nroots = 8

e_diff, intensity, x = mr_adc.kernel()

# For spectrum
from prism.tools.spectrum import plot
plot(e_diff, intensity, broadening = 0.5, omega_min = 540, omega_max = 550, plot = True, x_label = "Energy, eV", y_label = "Intensity", title = "XPS spectrum", filename = "mr_adc")

# Next run state-averaged CASSCF calculation for the two lowest states
n_states = 2
weights = np.ones(n_states)/n_states
mc = mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit()
mc.fix_spin_(ss=0) # Singlet states only
emc = mc.mc1step()[0]

state1_ci = mc.ci[0]
state2_ci = mc.ci[1]
mo = mc.mo_coeff.copy()

# Run reference calculation for the two lowest states
mc = mcscf.CASCI(mf, 6, 6).density_fit()
emc = mc.casci(mo_coeff=mo, ci0=state1_ci)[0]

interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit()
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)-x"
mr_adc.method_type = "cvs-ip"
mr_adc.max_cycle = 100
mr_adc.ncvs = 3
mr_adc.nroots = 8

e_diff, intensity, x = mr_adc.kernel()

plot(e_diff, intensity, broadening = 0.5, omega_min = 540, omega_max = 550, plot = True, x_label = "Energy, eV", y_label = "Intensity", title = "XPS spectrum", filename = "mr_adc_S0")

mc = mcscf.CASCI(mf, 6, 6).density_fit()
emc = mc.casci(mo_coeff=mo, ci0=state2_ci)[0]

interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit()
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)-x"
mr_adc.method_type = "cvs-ip"
mr_adc.max_cycle = 100
mr_adc.ncvs = 3
mr_adc.nroots = 8

e_diff, intensity, x = mr_adc.kernel()

plot(e_diff, intensity, broadening = 0.5, omega_min = 540, omega_max = 550, plot = True, x_label = "Energy, eV", y_label = "Intensity", title = "XPS spectrum", filename = "mr_adc_S1")
