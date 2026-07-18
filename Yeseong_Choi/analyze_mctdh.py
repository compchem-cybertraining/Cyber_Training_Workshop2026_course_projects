import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("P4DATA", HERE)
out = os.path.join(DATA, sys.argv[1] if len(sys.argv) > 1 else "pyr4", "output")

t, P1, P2, ct = [], [], [], None
for ln in open(out):
    m = re.match(r"\s*Time\s*=\s*([0-9.]+)\s*fs", ln)
    if m:
        ct = float(m.group(1)); continue
    m = re.match(r"\s*state\s*=\s*1\s*pop\.:\s*([0-9.]+)", ln)
    if m and ct is not None:
        p1 = float(m.group(1)); continue
    m = re.match(r"\s*state\s*=\s*2\s*pop\.:\s*([0-9.]+)", ln)
    if m and ct is not None:
        t.append(ct); P1.append(p1); P2.append(float(m.group(1))); ct = None

t, P1, P2 = np.array(t), np.array(P1), np.array(P2)
csv = os.path.join(DATA, "mctdh_pop.csv")
np.savetxt(csv, np.column_stack([t, P1, P2]), header="t_fs P_S1 P_S2", comments="")
print("parsed %d time points -> %s | P_S2(0)=%.3f P_S2(end@%.0ffs)=%.3f"
      % (len(t), csv, P2[0], t[-1], P2[-1]))
