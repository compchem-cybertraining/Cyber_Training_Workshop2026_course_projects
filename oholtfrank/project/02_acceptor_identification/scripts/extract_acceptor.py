#!/usr/bin/env python3
"""Acceptor identification: resonant + bright + dissociative. Parser fixed."""
import re, glob
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HA2EV=27.2114; RE=0.9572
GAP_AB=13.21; GAP_NIST=13.48

def parse(fn):
    txt=open(fn).read()
    if 'Happy landing' not in txt: return None
    en=[float(m) for m in re.findall(
        r'RASSCF root number\s+\d+ Total energy:\s+(-?\d+\.\d+)',txt)]
    occ={}
    for a,b in re.findall(
        r'Natural orbitals and occupation numbers for root\s+(\d+)\s*\n\s*sym 1:((?:\s+\d+\.\d+)+)',txt):
        occ[int(a)-1]=np.array([float(x) for x in b.split()])
    fosc={}
    if 'Dipole transition strengths' in txt:
        sec=txt.split('Dipole transition strengths',1)[1]
        # stop at the next section marker to avoid the velocity-gauge table
        nxt=sec.find('++')
        if nxt>0: sec=sec[:nxt]
        for line in sec.splitlines():
            mm=re.match(r'\s+(\d+)\s+(\d+)\s+([0-9]\.[0-9]+E[+-]\d+)',line)
            if mm:
                i,j,f=int(mm.group(1)),int(mm.group(2)),float(mm.group(3))
                if i==1: fosc[j-1]=f
    return (np.array(en),occ,fosc) if en else None

recs={}
for f in sorted(glob.glob('m2_*.out')):
    tag=f[3:-4]; r=float(tag[0]+'.'+tag[1:])
    o=parse(f)
    if o: recs[r]=o
    else: print("  r=%.4f NOT CONVERGED"%r)
rs=np.array(sorted(recs))
nr=min(len(recs[r][0]) for r in rs)
E=np.array([(recs[r][0][:nr]-recs[r][0][0])*HA2EV for r in rs])
OCC=[recs[r][1] for r in rs]
FOSC=[recs[r][2] for r in rs]
print("parsed %d geometries, %d roots"%(len(rs),nr))

tracked=np.zeros_like(E); tracked[0]=E[0]
hist=[np.arange(nr)]
for i in range(1,len(rs)):
    prev=hist[-1]
    pred = 2*tracked[i-1]-tracked[i-2] if i>=2 else tracked[i-1]
    C=np.full((nr,nr),1e9)
    for a in range(nr):
        oa=OCC[i-1].get(prev[a])
        for b in range(nr):
            ob=OCC[i].get(b)
            d=np.linalg.norm(ob-oa) if (oa is not None and ob is not None
                                        and len(oa)==len(ob)) else 2.0
            C[a,b]=abs(E[i,b]-pred[a]) + 0.5*d
    ra,cb=linear_sum_assignment(C)
    nm=np.zeros(nr,dtype=int)
    for a,b in zip(ra,cb): nm[a]=b
    hist.append(nm); tracked[i]=E[i,nm]

ire=int(np.argmin(np.abs(rs-RE)))
fre=FOSC[ire]
print("\n%-5s %-9s %-9s %-9s %-10s %-8s"%("st","E(re)","E(end)","drop","f_osc(re)","maxjump"))
cands=[]
for a in range(nr):
    lab=int(hist[ire][a]); f=fre.get(lab,0.0)
    e0,e1=tracked[ire,a],tracked[-1,a]; drop=e0-e1
    j=np.max(np.abs(np.diff(tracked[:,a])))
    near=min(abs(e0-GAP_AB),abs(e0-GAP_NIST))<1.2
    ok = near and f>0.005 and drop>2.0
    if ok: cands.append(a)
    print("%-5d %-9.3f %-9.3f %-9.3f %-10.4f %-8.2f%s"
          %(a,e0,e1,drop,f,j,"  <== CANDIDATE" if ok else ""))

if not cands:
    print("\nno candidate -- inspect table above"); raise SystemExit
best=min(cands,key=lambda a: min(abs(tracked[ire,a]-GAP_AB),abs(tracked[ire,a]-GAP_NIST)))
f_w=fre.get(int(hist[ire][best]),0.0)
print("\nACCEPTOR = state %d : E(re)=%.3f eV, f=%.4f, asymptote=%.3f eV"
      %(best,tracked[ire,best],f_w,tracked[-1,best]))

C_AU=137.036; S2AU=2.4188843265e-17
A_ar=1.4e8*S2AU; E_ar=13.48/HA2EV
d_ar=np.sqrt(3*A_ar*C_AU**3/(4*E_ar**3))
E_w=tracked[ire,best]/HA2EV
d_w=np.sqrt(3*f_w/(2*E_w))
kappa=np.sqrt(2.0/3.0)
print("\nAB INITIO COUPLING:")
print("  d_Ar=%.4f au (NIST)   d_W=%.4f au (RASSI f=%.4f)"%(d_ar,d_w,f_w))
for R,ref in [(3.5,9.35),(3.0,14.85),(2.5,25.66)]:
    V=kappa*d_ar*d_w/(R*1.8897259886)**3
    print("  R=%.1f A : V = %6.2f meV   (golden-rule route: %.2f meV)"
          %(R,V*HA2EV*1000,ref))

np.savez('acceptor_curve.npz', r=rs, Eexc=tracked[:,best],
         fosc=f_w, d_w=d_w, d_ar=d_ar, state=best)
print("\nwrote acceptor_curve.npz")

fig,ax=plt.subplots(figsize=(7.5,5.5))
for a in range(nr):
    lw,c=(2.5,'crimson') if a==best else (0.7,'lightsteelblue')
    ax.plot(rs,tracked[:,a],color=c,lw=lw)
ax.axhline(GAP_AB,ls=':',c='k'); ax.axvline(RE,ls=':',c='gray')
ax.set_xlabel(r'$r$(O-H) [$\AA$]'); ax.set_ylabel(r'$E_{exc}$ [eV]')
ax.set_ylim(0,16); ax.set_title('ab initio acceptor (red): resonant, bright, dissociative')
plt.tight_layout(); plt.savefig('fig_acceptor.png',dpi=150)
print("wrote fig_acceptor.png")
