#!/usr/bin/env python3
"""FIGURE: fragmentation dynamics on the A-tilde surface.
Left  : r(O-H) vs time for the classical ensemble (Clifford Fig-3 shape)
Right : t_frag histogram + H-atom KE distribution

Reads the tracked scan (same source as frag_lower.py) so the surface is the
ab initio one, not a model.
"""
import numpy as np, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

HA=27.2114; ANG=1.8897259886; AU2FS=0.0241888
MU=1728.0; OMEGA=3857.09/219474.63; RE=0.9572; DE=5.11
mH,mOH=1.008,17.007

f = glob.glob("/user/oholtfrank/Tutorials_OpenMolcas/vpd_phase1_dimer/monomer_scan/tracked_states.npz")
if not f:
    raise SystemExit("tracked_states.npz not found")
d = np.load(f[0]); rs, E = d['r'], d['E']

a_ang = (OMEGA*np.sqrt(MU/(2*DE/HA)))*ANG
def Vg(r): return DE*(1-np.exp(-a_ang*(np.asarray(r,float)-RE)))**2

ST = 1                                    # A-tilde
spl = CubicSpline(rs, E[:, ST])
def Vabs(r):
    r = np.clip(np.asarray(r,float), rs[0], rs[-1])
    return Vg(r) + spl(r)
def F(rb):
    rA = rb/ANG; h = 1e-4
    return -(((Vabs(rA+h)-Vabs(rA-h))/(2*h))/HA)/ANG

rng = np.random.default_rng(42); N = 2000
alpha = MU*OMEGA
r = rng.normal(RE*ANG, 1/np.sqrt(2*alpha), N)
p = rng.normal(0, np.sqrt(alpha/2), N)
dt = 2.0; nmax = 4000

traj = np.zeros((nmax, N)); tt = np.zeros(nmax)
ft = np.full(N, np.nan); ke_at_frag = np.full(N, np.nan)
for n in range(nmax):
    traj[n] = r/ANG; tt[n] = n*dt*AU2FS
    p += 0.5*dt*F(r); r += dt*p/MU; p += 0.5*dt*F(r)
    rA = r/ANG
    j = (rA > 2.8) & np.isnan(ft)
    ft[j] = n*dt*AU2FS
    ke_at_frag[j] = (p[j]**2/(2*MU))*HA
    if not np.isnan(ft).any(): break
nlast = n+1
nf = int(np.sum(~np.isnan(ft)))
print("fragmented %d/%d (%.1f%%)  t=%.1f+/-%.1f fs" %
      (nf, N, 100.*nf/N, np.nanmean(ft), np.nanstd(ft)))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# --- left: r(t) ---
show = np.random.default_rng(1).choice(N, 120, replace=False)
for k in show:
    frag = not np.isnan(ft[k])
    ax[0].plot(tt[:nlast], traj[:nlast, k], lw=0.5, alpha=0.35,
               color='#c1272d' if frag else '#4a6fa5')
ax[0].axhline(2.8, ls='--', c='k', lw=1.0)
ax[0].text(tt[nlast-1]*0.60, 2.92, 'fragmentation threshold, 2.8 Å', fontsize=8)
ax[0].set_xlabel('time (fs)'); ax[0].set_ylabel('r(O–H)  (Å)')
ax[0].set_ylim(0.5, 4.5); ax[0].set_xlim(0, min(tt[nlast-1], 60))
ax[0].set_title('Dissociation on Ã  (%d/%d = %.1f%%)' % (nf, N, 100.*nf/N),
                fontsize=10)

# --- right: t_frag + KE ---
ax[1].hist(ft[~np.isnan(ft)], bins=28, color='#c1272d', alpha=0.8)
ax[1].set_xlabel('fragmentation time (fs)'); ax[1].set_ylabel('count')
ax[1].set_title('t = %.1f ± %.1f fs' % (np.nanmean(ft), np.nanstd(ft)),
                fontsize=10)
ins = ax[1].inset_axes([0.55, 0.55, 0.42, 0.40])
ke = ke_at_frag[~np.isnan(ke_at_frag)]
ins.hist(ke, bins=24, color='#4a6fa5', alpha=0.85)
ins.set_xlabel('H KE (eV)', fontsize=7); ins.tick_params(labelsize=6)
ins.set_title('%.2f ± %.2f eV' % (ke.mean(), ke.std()), fontsize=7)

plt.tight_layout()
plt.savefig('fig_frag_dynamics.png', dpi=160)
print("wrote fig_frag_dynamics.png")
