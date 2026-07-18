"""Figure 4 (fig:popSH): ensemble S2 across the Libra family + hop density.

Self-contained: reads only data/ in this folder, writes figure.png/.pdf.

data/beating_v12a.npz            300-IC diabatic Ehrenfest reference
    time_fs, mean_pops (401, 3)     columns [S0, S1, S2]
data/beating_{fssh,shxf,qtsh}_v12a.npz   300-IC Libra ensembles (ntraj=2/IC)
    time_fs, mean_pops (2000, 3)    active-surface populations
    se_mean_pops (2000, 3)          coherent electronic populations
data/hop_density.csv             accepted-hop timing, 7.5 fs bins, columns
                                 t_fs,FSSH,SHXF,QTSH (counts over 300 ICs;
                                 derived from replica-averaged surface changes
                                 -- timing shape robust, counts approximate)
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

COL = {"Ehrenfest": "#2471a3", "QTSH": "#c0392b", "SHXF": "#2e8b57",
       "FSSH": "#8e6bab"}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.8, 6.4), sharex=True, height_ratios=[1.5, 1.0]
)

# --- panel a: ensemble-mean S2, four methods, common ~190 fs window ---
files = {"Ehrenfest": "beating_v12a.npz", "QTSH": "beating_qtsh_v12a.npz",
         "SHXF": "beating_shxf_v12a.npz", "FSSH": "beating_fssh_v12a.npz"}
for name, fn in files.items():
    d = np.load(os.path.join(DATA, fn), allow_pickle=True)
    ax1.plot(d["time_fs"], d["mean_pops"][:, 2], color=COL[name], lw=1.8,
             label=name)
ax1.set_ylabel(r"mean $S_2$ population")
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=False)

# --- panel b: accepted-hop density over time ---
with open(os.path.join(DATA, "hop_density.csv")) as f:
    rows = list(csv.reader(f))
head, vals = rows[0], np.array(rows[1:], dtype=float)
tc = vals[:, 0]
binw = tc[1] - tc[0]
for i, name in enumerate(("FSSH", "SHXF", "QTSH"), start=1):
    ax2.stairs(vals[:, i], np.append(tc - binw / 2, tc[-1] + binw / 2),
               color=COL[name], lw=1.8, label=name)
ax2.set_xlabel("time (fs)")
ax2.set_ylabel(f"hops / {binw:g} fs (300 ICs)")
ax2.set_yscale("log")
ax2.set_xlim(0, float(tc[-1] + binw / 2))
ax2.legend(frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "figure.png"), dpi=200)
fig.savefig(os.path.join(HERE, "figure.pdf"))
print("wrote figure.png / figure.pdf")
