"""FMS convergence figure: S2(t) grouped by spawned-basis size (velocity flavor).

Single panel, single message: FMS recovers the vibronic beat once the spawned
basis is large enough. Ensemble-mean S2(t) for small/medium/large basis-size
quintiles of the completed velocity-flavor trajectories, against the 300-IC
Ehrenfest reference.

Self-contained: reads only data/ in this folder; writes figure.png/.pdf and
CSV exports for external replotting.

data/fms_pertraj_s2.npz    completed FMS runs only (salvaged runs carry no
                           spawn count): velocity_s2 (n_ic, nt),
                           velocity_n_spawns (n_ic,), velocity_ic_indices
data/ehrenfest_mean.npz    time_fs, mean_pops (401, 3), 300 ICs
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "data", "fms_pertraj_s2.npz"))
e = np.load(os.path.join(HERE, "data", "ehrenfest_mean.npz"))

t = d["time_fs"]
s2, ns = d["velocity_s2"], d["velocity_n_spawns"]
te, pe = e["time_fs"], e["mean_pops"]
me = te <= float(t.max())

NQ = 5
edges = np.percentile(ns, np.linspace(0, 100, NQ + 1))
edges[0], edges[-1] = -np.inf, np.inf
qmasks = [(ns > edges[i]) & (ns <= edges[i + 1]) for i in range(NQ)]

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(te[me], pe[me, 2], color="#2471a3", lw=2.2, label="Ehrenfest (300 ICs)")
show = {0: ("#c7d9e8", "small basis"), 2: ("#7fa8c9", "medium basis"),
        4: ("#c0392b", "large basis")}
for qi, (col, name) in show.items():
    m = qmasks[qi]
    mean = s2[m].mean(0)
    sem = s2[m].std(0) / np.sqrt(m.sum())
    ax.plot(t, mean, color=col, lw=2.0,
            label=f"FMS, {name} (~{ns[m].mean():.0f} spawns, {m.sum()} ICs)")
    ax.fill_between(t, mean - sem, mean + sem, color=col, alpha=0.3, lw=0)
ax.set_xlabel("time (fs)")
ax.set_ylabel(r"mean $S_2$ population")
ax.set_xlim(0, float(t.max()))
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "figure.png"), dpi=200)
fig.savefig(os.path.join(HERE, "figure.pdf"))
print("wrote figure.png / figure.pdf")

# --- CSV exports for external replotting ---
with open(os.path.join(HERE, "data", "quintile_curves_velocity.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_fs"] + [f"Q{i+1}_{s}" for i in range(NQ) for s in ("mean", "sem")])
    cols = []
    for m in qmasks:
        cols += [s2[m].mean(0), s2[m].std(0) / np.sqrt(m.sum())]
    for i in range(len(t)):
        w.writerow([f"{t[i]:.4f}"] + [f"{c[i]:.5f}" for c in cols])

with open(os.path.join(HERE, "data", "quintile_legend.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["quintile", "mean_spawns", "min_spawns", "max_spawns", "n_ics"])
    for i, m in enumerate(qmasks):
        w.writerow([i + 1, f"{ns[m].mean():.1f}", int(ns[m].min()),
                    int(ns[m].max()), int(m.sum())])

with open(os.path.join(HERE, "data", "ehrenfest_window.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_fs", "S2_mean", "S1_mean"])
    for i in np.nonzero(me)[0]:
        w.writerow([f"{te[i]:.4f}", f"{pe[i, 2]:.5f}", f"{pe[i, 1]:.5f}"])
print("wrote quintile_curves_velocity.csv / quintile_legend.csv / ehrenfest_window.csv")
