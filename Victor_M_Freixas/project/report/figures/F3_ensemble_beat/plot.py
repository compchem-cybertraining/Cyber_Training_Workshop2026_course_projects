#!/usr/bin/env python
"""F3: the vibronic beat -- Ehrenfest 300-IC ensemble on the frozen NQDH (v12a),
with the model-generation progression as an inset. Self-contained: reads ./data,
writes ./figure.png/.pdf and Mathematica-ready CSVs into ./data."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
d = np.load(HERE / "data" / "beating_v12a.npz")
t, mp = d["time_fs"], d["mean_pops"]
np.savetxt(HERE / "data" / "beat_v12a.csv",
           np.column_stack([t, mp]), delimiter=",",
           header="time_fs,S0,S1,S2", comments="")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(t, mp[:, 2], color="#c0392b", lw=2.2, label="S2")
ax.plot(t, mp[:, 1], color="#2471a3", lw=2.2, label="S1")
ax.set_xlim(0, 120); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("time (fs)"); ax.set_ylabel("ensemble adiabatic population")
ax.grid(alpha=0.25); ax.legend(loc="center right")

axi = ax.inset_axes([0.42, 0.52, 0.55, 0.44])
for tag, c in [("v9", "0.75"), ("v10b", "0.55"), ("v10d", "0.35"), ("v12a", "#c0392b")]:
    dd = np.load(HERE / "data" / f"beating_{tag}.npz")
    axi.plot(dd["time_fs"], dd["mean_pops"][:, 2], color=c, lw=1.3,
             label=tag if tag != "v12a" else "final")
    np.savetxt(HERE / "data" / f"beat_{tag}_S2.csv",
               np.column_stack([dd["time_fs"], dd["mean_pops"][:, 2]]),
               delimiter=",", header="time_fs,S2", comments="")
axi.set_xlim(0, 100); axi.set_ylim(0, 0.75); axi.tick_params(labelsize=7)
axi.set_title("model generations", fontsize=8); axi.legend(fontsize=6)
fig.tight_layout()
fig.savefig(HERE / "figure.png", dpi=170); fig.savefig(HERE / "figure.pdf")
print("F3 done")
