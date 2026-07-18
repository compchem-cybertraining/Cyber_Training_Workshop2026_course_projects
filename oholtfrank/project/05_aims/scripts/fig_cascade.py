#!/usr/bin/env python3
"""FIGURE 3: the AIMS cascade.
Left  : target-state distribution (the funnel)
Right : the same events against time

NOTE ON INDEXING: PySpawn logs states 0-indexed (istate=11 launches on
root 12, and the first spawn logs as "state 10" = root 11). This script
converts to ROOT numbering (root = log + 1) so the figure matches the
manuscript text.

Root 12 = the bright VPD acceptor (launch state).
Root 3  = B-tilde region, the lowest reached in the propagated window.
Root 2  = A-tilde -- NOT reached.
"""
import glob, re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
AU2FS = 0.0241888

roots, times = [], []
for d in sorted(glob.glob('[0-9]*/'), key=lambda x: int(x.strip('/'))):
    txt = "".join(open(l, errors='ignore').read() for l in glob.glob(d+'*.log'))
    for st, t in re.findall(
            r'entered spawning region for state\s+(\d+)\s+at time\s+([\d.]+)', txt):
        roots.append(int(st) + 1)          # 0-indexed log -> root number
        times.append(float(t)*AU2FS)
roots = np.array(roots); times = np.array(times)

LAUNCH = 12          # root 12 = the acceptor
DISS_LO, DISS_HI = 2, 4    # dissociative valence region (A-tilde=2, B-tilde=3/4)

print("%d spawns" % len(roots))
print("first spawn %.1f +/- %.1f fs" % (times.mean(), times.std()))
print("lowest root reached: %d" % roots.min())
for r in sorted(set(roots), reverse=True):
    print("  root %2d: %3d" % (r, int(np.sum(roots == r))))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2),
                       gridspec_kw={'width_ratios': [1, 1.4], 'wspace': 0.25})

u = np.arange(roots.min(), roots.max()+1)
cnt = [int(np.sum(roots == r)) for r in u]
cols = ['#c1272d' if r <= DISS_HI else '#4a6fa5' for r in u]
ax[0].barh(u, cnt, color=cols)
ax[0].axhspan(DISS_LO-0.5, DISS_HI+0.5, color='#c1272d', alpha=0.07)
ax[0].axhline(LAUNCH, ls='--', c='k', lw=0.8)
ax[0].text(max(cnt)*0.40, LAUNCH+0.28, 'launch (acceptor)', fontsize=8)
ax[0].text(max(cnt)*0.40, 3.2, 'dissociative\nvalence region', fontsize=8,
           color='#c1272d')
ax[0].set_xlabel('spawning events'); ax[0].set_ylabel('target state (root)')
ax[0].set_yticks(u)
for r, n in zip(u, cnt):
    if n: ax[0].text(n+max(cnt)*0.015, r, str(n), va='center', fontsize=7)
ax[0].set_title('%d events, 16 trajectories' % len(roots), fontsize=10)

c2 = ['#c1272d' if r <= DISS_HI else '#4a6fa5' for r in roots]
ax[1].scatter(times, roots, s=28, c=c2, alpha=0.5, edgecolors='none')
ax[1].axhspan(DISS_LO-0.5, DISS_HI+0.5, color='#c1272d', alpha=0.07)
ax[1].axhline(LAUNCH, ls='--', c='k', lw=0.8)
ax[1].set_xlabel('spawn time (fs)'); ax[1].set_ylabel('target state (root)')
ax[1].set_yticks(u)
ax[1].set_title('first spawn at %.1f ± %.1f fs' % (times.mean(), times.std()),
                fontsize=10)

fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.13)
plt.savefig('fig_cascade.png', dpi=160)
print("wrote fig_cascade.png")
