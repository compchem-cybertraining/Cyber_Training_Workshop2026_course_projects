#!/usr/bin/env python3
"""FIGURE 2: the VPD acceptor is bound along r(O-H)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

d = np.load('/user/oholtfrank/Tutorials_OpenMolcas/vpd_phase1_dimer/monomer_scan2/acceptor_curve.npz')
rs, Ee = d['r'], d['Eexc']
HA=27.2114; MU=1728.0; OMEGA=3857.09/219474.63; RE=0.9572; DE=5.11; ANG=1.8897259886
a = (OMEGA*np.sqrt(MU/(2*DE/HA)))*ANG
r = np.linspace(rs[0], min(rs[-1], 3.0), 400)
V = DE*(1-np.exp(-a*(r-RE)))**2 + CubicSpline(rs, Ee)(r)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(r, V, lw=2.2, color='#c1272d')
ax.plot(1.20, 12.82, 'o', c='k', ms=6, zorder=5)
ax.annotate('12.82 eV at 1.20 Å', xy=(1.20, 12.82), xytext=(1.75, 13.05),
            arrowprops=dict(arrowstyle='->', lw=0.9), fontsize=9,
            ha='left', va='center')
ax.axhline(13.30, ls=':', c='gray', lw=0.9)
ax.text(2.98, 13.36, 'vertical, 13.30 eV', fontsize=8, color='gray', ha='right')
ax.set_xlabel('r(O–H)  (Å)')
ax.set_ylabel('absolute energy (eV)')
ax.set_xlim(0.75, 3.0)
ax.set_ylim(12.5, 15.8)
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.13)
plt.savefig('fig_bound_acceptor.png', dpi=160)
print('wrote fig_bound_acceptor.png')
