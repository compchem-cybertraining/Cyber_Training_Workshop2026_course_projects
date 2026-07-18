#!/usr/bin/env python3
"""Classical propagation on the DISSOCIATIVE lower states (A-tilde, B-tilde).
The acceptor (13.3 eV) is bound -- shown by frag_classical.py (0.1%).
Fragmentation happens AFTER the cascade lands here. This gives the
timescale and KER on those surfaces, from the tracked scan data."""
import numpy as np, glob
from scipy.interpolate import CubicSpline
HA=27.2114; ANG=1.8897259886; AU2FS=0.0241888
MU=1728.0; OMEGA=3857.09/219474.63; RE=0.9572; DE=5.11
mH,mOH=1.008,17.007

f = glob.glob("/user/oholtfrank/Tutorials_OpenMolcas/vpd_phase1_dimer/monomer_scan/tracked_states.npz")
if not f:
    print("tracked_states.npz not found -- check path"); raise SystemExit
d = np.load(f[0]); rs, E = d['r'], d['E']    # E[geom, state], excitation energies
print("tracked: %d geoms, %d states" % E.shape)

a_ang = (OMEGA*np.sqrt(MU/(2*DE/HA)))*ANG
def Vg(r): return DE*(1-np.exp(-a_ang*(np.asarray(r,float)-RE)))**2

for st, name in [(1,"A-tilde"), (3,"B-tilde")]:
    Ee = E[:, st]
    spl = CubicSpline(rs, Ee)
    def Vabs(r):
        r = np.clip(np.asarray(r,float), rs[0], rs[-1])
        return Vg(r) + spl(r)
    print("\n=== state %d (%s) ===" % (st, name))
    print("  E_abs:  r=0.96: %.2f   1.5: %.2f   2.5: %.2f   4.0: %.2f eV"
          % (Vabs(RE), Vabs(1.5), Vabs(2.5), Vabs(4.0)))
    def F(rb):
        rA = rb/ANG; h = 1e-4
        return -(((Vabs(rA+h)-Vabs(rA-h))/(2*h))/HA)/ANG
    rng = np.random.default_rng(42); N = 2000
    alpha = MU*OMEGA
    r = rng.normal(RE*ANG, 1/np.sqrt(2*alpha), N)
    p = rng.normal(0, np.sqrt(alpha/2), N)
    dt = 2.0; ft = np.full(N, np.nan); rmx = np.full(N, RE)
    for n in range(8000):
        p += 0.5*dt*F(r); r += dt*p/MU; p += 0.5*dt*F(r)
        rA = r/ANG; rmx = np.maximum(rmx, rA)
        j = (rA > 2.8) & np.isnan(ft); ft[j] = n*dt*AU2FS
        if not np.isnan(ft).any(): break
    nf = int(np.sum(~np.isnan(ft)))
    print("  fragmented: %d/%d (%.1f%%)" % (nf, N, 100.0*nf/N))
    print("  max r: mean %.2f A" % rmx.mean())
    if nf > 20:
        print("  t_frag = %.1f +/- %.1f fs" % (np.nanmean(ft), np.nanstd(ft)))
        KE = (p**2/(2*MU))*HA
        print("  H KE = %.2f +/- %.2f eV  (H carries %.1f%%)"
              % (KE.mean(), KE.std(), 100*mOH/(mH+mOH)))
