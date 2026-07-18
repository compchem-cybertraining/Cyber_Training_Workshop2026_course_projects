#!/usr/bin/env python3
"""Robust state tracking: energy continuity dominant, with slope extrapolation."""
import re, glob
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HA2EV=27.2114; E_PHOTON=13.48; D0=5.11; RE=0.9572

def parse(fn):
    txt=open(fn).read()
    if 'Happy landing' not in txt: return None
    en=[float(m) for m in re.findall(
        r'RASSCF root number\s+\d+ Total energy:\s+(-?\d+\.\d+)',txt)]
    occ={}
    for a,b in re.findall(
        r'Natural orbitals and occupation numbers for root\s+(\d+)\s*\n\s*sym 1:((?:\s+\d+\.\d+)+)',txt):
        occ[int(a)-1]=np.array([float(x) for x in b.split()])
    return (np.array(en),occ) if en and occ else None

recs={}
for f in sorted(glob.glob('mono_*.out')):
    tag=f[5:-4]; r=float(tag[0]+'.'+tag[1:])
    o=parse(f)
    if o: recs[r]=o
rs=np.array(sorted(recs))
nr=min(len(recs[r][0]) for r in rs)
E=np.array([(recs[r][0][:nr]-recs[r][0][0])*HA2EV for r in rs])
OCC=[recs[r][1] for r in rs]
print("parsed %d geoms, %d roots"%(len(rs),nr))

tracked=np.zeros_like(E); tracked[0]=E[0]
cur=np.arange(nr)
hist=[cur.copy()]
for i in range(1,len(rs)):
    prev=hist[-1]
    # predicted energy: linear extrapolation from last two points
    if i>=2:
        pred=2*tracked[i-1]-tracked[i-2]
    else:
        pred=tracked[i-1]
    C=np.full((nr,nr),1e9)
    for a in range(nr):
        pa=prev[a]
        oa=OCC[i-1].get(pa)
        for b in range(nr):
            ob=OCC[i].get(b)
            dE=abs(E[i,b]-pred[a])
            docc=np.linalg.norm(ob-oa) if (oa is not None and ob is not None and len(oa)==len(ob)) else 2.0
            C[a,b]= dE + 0.5*docc          # ENERGY DOMINANT
    ra,cb=linear_sum_assignment(C)
    nm=np.zeros(nr,dtype=int)
    for a,b in zip(ra,cb): nm[a]=b
    hist.append(nm)
    tracked[i]=E[i,nm]

ire=int(np.argmin(np.abs(rs-RE)))
print("\n%-8s"%"r(A)"+"".join(" st%-2d "%a for a in range(nr)))
for i,r in enumerate(rs):
    print("%-8.4f"%r+"".join("%6.2f"%tracked[i,a] for a in range(nr)))

print("\nmax single-step jump per state (should be < ~1.5 eV):")
for a in range(nr):
    j=np.max(np.abs(np.diff(tracked[:,a])))
    flag=" <-- DISCONTINUOUS" if j>2.0 else ""
    print("  st%-2d  max |dE| = %5.2f eV%s"%(a,j,flag))

print("\n%-6s %-10s %-10s %-9s %-8s"%("state","E(re)","E(4.0A)","drop","near D0?"))
for a in range(nr):
    e0,e1=tracked[ire,a],tracked[-1,a]
    print("%-6d %-10.3f %-10.3f %-9.3f %-8s"
          %(a,e0,e1,e0-e1,"YES" if abs(e1-D0)<2.0 else ""))

print("\nStates with E(re) within 1.5 eV of the virtual photon (%.2f eV):"%E_PHOTON)
for a in range(nr):
    if abs(tracked[ire,a]-E_PHOTON)<1.5:
        print("  st%-2d  E(re)=%.3f  E(4.0)=%.3f  drop=%.3f"
              %(a,tracked[ire,a],tracked[-1,a],tracked[ire,a]-tracked[-1,a]))

np.savez('tracked_states.npz',r=rs,E=tracked)
fig,ax=plt.subplots(figsize=(7.5,5.5))
for a in range(nr):
    ax.plot(rs,tracked[:,a],lw=1.2,marker='o',ms=2.5,label='st%d'%a if a<8 else None)
ax.axhline(D0,ls='--',c='green'); ax.axhline(E_PHOTON,ls=':',c='k')
ax.axvline(RE,ls=':',c='gray')
ax.text(3.1,D0+0.2,'$D_0$=5.11',color='green',fontsize=8)
ax.text(3.1,E_PHOTON+0.2,'13.48 eV',fontsize=8)
ax.set_xlabel(r'$r$(O-H) [$\AA$]'); ax.set_ylabel('E$_{exc}$ [eV]')
ax.set_ylim(0,16); ax.legend(fontsize=6,ncol=2)
ax.set_title('H$_2$O excited states, energy-continuity tracking')
plt.tight_layout(); plt.savefig('fig_tracked.png',dpi=150)
print("\nwrote fig_tracked.png, tracked_states.npz")
