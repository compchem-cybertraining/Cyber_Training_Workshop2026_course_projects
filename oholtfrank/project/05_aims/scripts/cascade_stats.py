#!/usr/bin/env python3
"""Aggregate the AIMS cascade: spawn times, target states, tree depth."""
import glob, re, numpy as np
AU2FS = 0.0241888
rows = []
for d in sorted(glob.glob('[0-9]*/'), key=lambda x: int(x.strip('/'))):
    logs = glob.glob(d + '*.log')
    if not logs: continue
    txt = "".join(open(l, errors='ignore').read() for l in logs)
    sp = re.findall(r'entered spawning region for state\s+(\d+)\s+at time\s+([\d.]+)', txt)
    tm = re.findall(r'time = ([\d.]+)', txt)
    tmax = max([float(x) for x in tm]) if tm else 0.0
    rows.append((d.strip('/'), sp, tmax))

print("%-5s %-8s %-10s %s" % ("traj", "t_max(fs)", "n_spawn", "spawns (state @ fs)"))
allstates = []; alltimes = []
for name, sp, tmax in rows:
    s = ", ".join("%s@%.1f" % (st, float(t)*AU2FS) for st, t in sp)
    print("%-5s %-8.1f %-10d %s" % (name, tmax*AU2FS, len(sp), s if s else "-"))
    for st, t in sp:
        allstates.append(int(st)); alltimes.append(float(t)*AU2FS)

print("\n=== ENSEMBLE ===")
print("trajectories: %d   total spawns: %d" % (len(rows), len(allstates)))
if alltimes:
    print("first-spawn time: %.1f +/- %.1f fs" % (np.mean(alltimes), np.std(alltimes)))
    print("\ntarget-state distribution:")
    for s in sorted(set(allstates), reverse=True):
        n = allstates.count(s)
        print("  state %2d: %2d spawns  %s" % (s, n, "#"*n))
    print("\nlowest state reached: %d" % min(allstates))
