#!/usr/bin/env python3
"""
Build the 2D ground and acceptor surfaces E_g(r1,r2), E_exc(r1,r2).
Acceptor tracked outward from (re,re) by flood-fill continuity of the
ABSOLUTE energy, seeded by the bright resonant state (f>0.05 near 13.3 eV).
"""
import re, glob
import numpy as np

HA2EV=27.2114
GRID=[0.80,0.9572,1.10,1.30,1.60,2.00,2.50,3.00]
NG=len(GRID); IRE=1
GAP=13.48

def parse(fn):
    txt=open(fn).read()
    if 'Happy landing' not in txt: return None
    en=[float(m) for m in re.findall(
        r'RASSCF root number\s+\d+ Total energy:\s+(-?\d+\.\d+)',txt)]
    fosc={}
    if 'Dipole transition strengths' in txt:
        sec=txt.split('Dipole transition strengths',1)[1]
        nxt=sec.find('++')
        if nxt>0: sec=sec[:nxt]
        for line in sec.splitlines():
            mm=re.match(r'\s+(\d+)\s+(\d+)\s+([0-9]\.[0-9]+E[+-]\d+)',line)
            if mm and int(mm.group(1))==1:
                fosc[int(mm.group(2))-1]=float(mm.group(3))
    return np.array(en), fosc

E_all={}; F_all={}
for f in glob.glob('m3_*.out'):
    mm=re.match(r'm3_(\d+)_(\d+)\.out',f)
    i,j=int(mm.group(1)),int(mm.group(2))
    o=parse(f)
    if o is None:
        print("  (%d,%d) NOT CONVERGED"%(i,j)); continue
    E_all[(i,j)]=o[0]; F_all[(i,j)]=o[1]
print("parsed %d / 36 points"%len(E_all))

E0_ref = E_all[(IRE,IRE)][0]           # ground energy at (re,re)
Eg  = np.full((NG,NG),np.nan)          # ground surface (eV, rel to re,re)
Eab = np.full((NG,NG),np.nan)          # acceptor ABSOLUTE surface
for (i,j),en in E_all.items():
    Eg[i,j]=Eg[j,i]=(en[0]-E0_ref)*HA2EV

# seed at (re,re): bright state nearest the gap
en=E_all[(IRE,IRE)]; fo=F_all[(IRE,IRE)]
exc=(en-en[0])*HA2EV
cands=[k for k in range(len(en)) if abs(exc[k]-GAP)<1.5 and fo.get(k,0)>0.05]
seed=min(cands,key=lambda k:abs(exc[k]-GAP))
Eab[IRE,IRE]=Eab[IRE,IRE]=Eg[IRE,IRE]+exc[seed]
print("seed: root %d at (re,re), E_exc=%.3f eV, f=%.4f"
      %(seed+1,exc[seed],fo.get(seed,0)))

# flood-fill outward by distance from (re,re)
order=sorted(((i,j) for (i,j) in E_all if not (i==IRE and j==IRE)),
             key=lambda p:(GRID[p[0]]-GRID[IRE])**2+(GRID[p[1]]-GRID[IRE])**2)
for (i,j) in order:
    # prediction: mean of assigned neighbors (in the symmetrized grid)
    preds=[]
    for di,dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1)]:
        a,b=i+di,j+dj
        if 0<=a<NG and 0<=b<NG and not np.isnan(Eab[a,b]): preds.append(Eab[a,b])
    pred=np.mean(preds) if preds else Eab[IRE,IRE]
    en=E_all[(i,j)]
    Eabs_all=Eg[i,j]+ (en-en[0])*HA2EV
    k=np.argmin(np.abs(Eabs_all-pred))
    Eab[i,j]=Eab[j,i]=Eabs_all[k]

print("\nGROUND surface (eV):")
print("        "+"".join("%7.2f"%r for r in GRID))
for i,r in enumerate(GRID):
    print("%7.2f "%r + "".join("%7.2f"%Eg[i,j] for j in range(NG)))
print("\nACCEPTOR absolute surface (eV):")
print("        "+"".join("%7.2f"%r for r in GRID))
for i,r in enumerate(GRID):
    print("%7.2f "%r + "".join("%7.2f"%Eab[i,j] for j in range(NG)))

print("\nKEY DIAGNOSTIC:")
print("  acceptor at (re,re):        %7.2f eV"%Eab[IRE,IRE])
print("  acceptor at (3.0,3.0):      %7.2f eV  <- three-body direction"%Eab[-1,-1])
print("  acceptor at (re,3.0):       %7.2f eV  <- single O-H direction"%Eab[IRE,-1])
print("  A 13.5 eV packet fragments along any direction where the")
print("  surface falls BELOW ~13.5.")
np.savez('surf2d.npz',grid=np.array(GRID),Eg=Eg,Eab=Eab,gap=GAP)
print("\nwrote surf2d.npz")
