#!/usr/bin/env python3
"""ROUTE (c): NE-FGR population transfer for VPD via Libra fgr_py."""
import sys, os
import numpy as np

try:
    from liblibra_core import *
    from libra_py import units, fgr_py
    HAVE_LIBRA = True
except Exception as e:
    print("!! Libra not importable:", e)
    HAVE_LIBRA = False

CM2AU  = 1.0/219474.63
EV2AU  = 1.0/27.211386245988
FS2AU  = 41.341374575751

P = {}
if os.path.exists('vpd_params.txt'):
    for line in open('vpd_params.txt'):
        if line.startswith('#') or not line.strip(): continue
        k,v = line.split()[:2]; P[k]=float(v)

dE_eV   = P.get('dE_resonance_eV', 0.0)
V_eV    = P.get('V_35A_eV', 9.35e-3)
V_eV_25 = P.get('V_25A_eV', 25.66e-3)

omega_cm = [P.get('omega_bend',1678.32),
            P.get('omega_sym', 3726.78),
            P.get('omega_asym',3857.09)]
omega_au = [w*CM2AU for w in omega_cm]

MU_OH = 1728.0
d_bohr = {'bend':0.10, 'sym':0.45, 'asym':0.45}
req_nm = []
for name,w in zip(['bend','sym','asym'], omega_au):
    req_nm.append(d_bohr[name]*np.sqrt(MU_OH*w))
gamma_nm = [r*w for r,w in zip(req_nm, omega_au)]

print("="*72)
print("NE-FGR MODEL FOR VPD  (Ar-H2O dimer)")
print("="*72)
print("  dE (donor-acceptor gap)  = %.4f eV   (resonance)" % dE_eV)
print("  V  (coupling, R=3.5 A)   = %.4f meV" % (V_eV*1000))
print("  V  (coupling, R=2.5 A)   = %.4f meV" % (V_eV_25*1000))
print("\n  mode      omega(cm^-1)   omega(au)      req(dimless)   gamma(au)")
for n,(wc,wa,r,g) in enumerate(zip(omega_cm,omega_au,req_nm,gamma_nm)):
    print("  %-9s %10.2f   %.6e   %8.4f      %.4e" % (['bend','sym','asym'][n],wc,wa,r,g))

lam = sum(0.5*w*r**2 for w,r in zip(omega_au,req_nm))
print("\n  reorganization energy Er = %.4f au = %.3f eV" % (lam, lam*27.2114))

if not HAVE_LIBRA:
    print("\nStopping: Libra unavailable. Parameters above still valid.")
    sys.exit(0)

ndof = len(omega_au)
omega = Py2Cpp_double(omega_au)
gamma = Py2Cpp_double(gamma_nm)
req   = Py2Cpp_double(req_nm)

s = -1.0
shift_NE = doubleList()
for i in range(ndof):
    shift_NE.append(s*req_nm[i])

for label, Vev in [("R35", V_eV), ("R25", V_eV_25)]:
    sim_params = {"dt": 1.0*units.fs2au, "dtau": 0.01*units.fs2au,
                  "tmax": 2000.0*units.fs2au,
                  "Temperature": 10.0, "do_output": False,
                  "dyn_type": 1, "method": 0, "filename": "nefgr_out"}
    t, rate, pop = fgr_py.run_NEFGRL_populations(
        dE_eV*EV2AU, Vev*EV2AU, omega, gamma, req, shift_NE, sim_params)
    t=np.array(t); pop=np.array(pop); rate=np.array(rate)
    tfs = t/FS2AU
    idx = np.argmax(pop < np.exp(-1.0)) if (pop<np.exp(-1)).any() else -1
    tau = tfs[idx] if idx>0 else float('nan')
    print("\n  %s :  P_donor(1/e) at tau = %.1f fs" % (label, tau))
    np.savetxt('nefgr_%s.dat'%label, np.c_[tfs,pop,rate], header='t_fs P_donor rate_au')

print("\nwrote nefgr_*.dat")
