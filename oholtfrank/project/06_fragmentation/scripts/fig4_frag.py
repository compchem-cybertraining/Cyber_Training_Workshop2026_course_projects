#!/usr/bin/env python3
"""FIGURE 4: fragmentation dynamics on the A-tilde surface.
KE is measured AT the 2.8 A crossing (asymptotic value), not averaged
over the whole ensemble."""
import numpy as np, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

HA=27.2114; ANG=1.8897259886; AU2FS=0.0241888
MU=1728.0; OMEGA=3857.09/219474.63; RE=0.9572; DE=5.11
mH, mOH = 1.008, 17.007

f = glob.glob("/user/oholtfrank/Tutorials_OpenMolcas/vpd_phase1_dimer/monomer_scan/tracked_states.npz")
if not f: raise SystemExit("tracked_states.npz not found")
d = np.load(f[0]); rs, E = d['r'], d['E']
a_ang = (OMEGA*np.sqrt(MU/(2*DE/HA)))*ANG
def Vg(x): return DE*(1-np.exp(-a_ang*(np.asarray(x,float)-RE)))**2
spl = CubicSpline(rs, E[:, 1])                      # A-tilde
def Vabs(x):
    x = np.clip(np.asarray(x,float), rs[0], rs[-1]); return Vg(x)+spl(x)
def F(rb):
    rA = rb/ANG; h = 1e-4
    return -(((Vabs(rA+h)-Vabs(rA-h))/(2*h))/HA)/ANG

rng = np.random.default_rng(42); N = 2000
alpha = MU*OMEGA
r = rng.normal(RE*ANG, 1/np.sqrt(2*alpha), N)
p = rng.normal(0, np.sqrt(alpha/2), N)
dt = 2.0; nmax = 4000
traj = np.zeros((nmax, N)); tt = np.zeros(nmax)
ft = np.full(N, np.nan); ke = np.full(N, np.nan)
for n in range(nmax):
    traj[n] = r/ANG; tt[n] = n*dt*AU2FS
    p += 0.5*dt*F(r); r += dt*p/MU; p += 0.5*dt*F(r)
    rA = r/ANG
    j = (rA > 2.8) & np.isnan(ft)
    ft[j] = n*dt*AU2FS; ke[j] = (p[j]**2/(2*MU))*HA
    if not np.isnan(ft).any(): break
nlast = n+1
nf = int(np.sum(~np.isnan(ft))); kev = ke[~np.isnan(ke)]
print("fragmented %d/%d (%.1f%%)" % (nf, N, 100.*nf/N))
print("t_frag = %.1f +/- %.1f fs" % (np.nanmean(ft), np.nanstd(ft)))
print("H KE   = %.2f +/- %.2f eV" % (kev.mean(), kev.std()))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0),
                       gridspec_kw={'width_ratios':[1.15, 1], 'wspace':0.28})
NSHOW = 150
show = np.random.default_rng(1).choice(N, NSHOW, replace=False)
nred = int(np.sum(~np.isnan(ft[show])))
for k in show:
    ax[0].plot(tt[:nlast], traj[:nlast, k], lw=0.6, alpha=0.4,
               color='#c1272d' if not np.isnan(ft[k]) else '#4a6fa5')
ax[0].axhline(2.8, ls='--', c='k', lw=1.0)
ax[0].text(30, 2.92, 'dissociation threshold, 2.8 Å', fontsize=8, ha='center')
ax[0].set_xlabel('time (fs)'); ax[0].set_ylabel('r(O–H)  (Å)')
ax[0].set_xlim(0, 60); ax[0].set_ylim(0.5, 4.5)
ax[0].set_title('%d of %d trajectories shown' % (NSHOW, N), fontsize=9)

ax[1].hist(ft[~np.isnan(ft)], bins=26, color='#c1272d', alpha=0.85)
ax[1].set_xlabel('dissociation time (fs)'); ax[1].set_ylabel('count')
ax[1].set_title('%.1f%% dissociate;  t = %.1f ± %.1f fs'
                % (100.*nf/N, np.nanmean(ft), np.nanstd(ft)), fontsize=9)
ins = ax[1].inset_axes([0.55, 0.50, 0.42, 0.44])
ins.hist(kev, bins=22, color='#4a6fa5', alpha=0.9)
ins.set_xlabel('H KE (eV)', fontsize=7); ins.tick_params(labelsize=6)
ins.set_ylabel('count', fontsize=7)
ins.set_title('%.2f ± %.2f eV' % (kev.mean(), kev.std()), fontsize=7)
fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.13)
plt.savefig('fig_frag_dynamics.png', dpi=160)
print("wrote fig_frag_dynamics.png")
