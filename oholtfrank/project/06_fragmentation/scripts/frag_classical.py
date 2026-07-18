#!/usr/bin/env python3
"""Classical propagation on the ab initio acceptor surface (1D O-H cut).
Tests directly whether the bright superexcited state dissociates along a
single O-H. Uses acceptor_curve.npz from monomer_scan2."""
import numpy as np, glob
from scipy.interpolate import CubicSpline
HA=27.2114; ANG=1.8897259886; AU2FS=0.0241888
MU=1728.0; OMEGA=3857.09/219474.63; RE=0.9572; DE=5.11

cand = glob.glob("/user/oholtfrank/Tutorials_OpenMolcas/vpd_phase1_dimer/monomer_scan2/acceptor_curve.npz")
if not cand:
    print("acceptor_curve.npz NOT FOUND"); raise SystemExit
d = np.load(cand[0]); rs, Ee = d['r'], d['Eexc']
print("acceptor curve: %d pts, E_exc(re)=%.3f -> E_exc(end)=%.3f eV" % (len(rs), Ee[0], Ee[-1]))

a_ang = (OMEGA*np.sqrt(MU/(2*DE/HA)))*ANG
def Vg(r): return DE*(1-np.exp(-a_ang*(np.asarray(r,float)-RE)))**2
spl = CubicSpline(rs, Ee)
def Vabs_eV(r):
    r = np.clip(np.asarray(r,float), rs[0], rs[-1])
    return Vg(r) + spl(r)

print("\nABSOLUTE acceptor energy along r(O-H):")
for rr in [0.80, RE, 1.20, 1.60, 2.00, 3.00, 4.00]:
    print("  r=%.2f  E_abs=%7.3f eV" % (rr, float(Vabs_eV(rr))))
print("\n(if E_abs RISES, no dissociation along this coordinate)")

def F(rb):
    rA = rb/ANG; h = 1e-4
    return -(((Vabs_eV(rA+h)-Vabs_eV(rA-h))/(2*h))/HA)/ANG

rng = np.random.default_rng(42); N = 2000
alpha = MU*OMEGA
r = rng.normal(RE*ANG, 1/np.sqrt(2*alpha), N)
p = rng.normal(0, np.sqrt(alpha/2), N)
dt = 2.0
frag_t = np.full(N, np.nan); rmax = np.full(N, RE)
for n in range(10000):
    p += 0.5*dt*F(r); r += dt*p/MU; p += 0.5*dt*F(r)
    rA = r/ANG
    rmax = np.maximum(rmax, rA)
    just = (rA > 2.8) & np.isnan(frag_t)
    frag_t[just] = n*dt*AU2FS
    if not np.isnan(frag_t).any(): break
nf = int(np.sum(~np.isnan(frag_t)))
print("\nfragmented: %d/%d (%.1f%%)  after %.0f fs" % (nf, N, 100.0*nf/N, n*dt*AU2FS))
print("max r(O-H) reached: mean %.2f, max %.2f A" % (rmax.mean(), rmax.max()))
if nf > 0:
    print("t_frag = %.1f +/- %.1f fs" % (np.nanmean(frag_t), np.nanstd(frag_t)))
    KE = (p**2/(2*MU))*HA
    print("H KE = %.2f +/- %.2f eV" % (KE.mean(), KE.std()))
