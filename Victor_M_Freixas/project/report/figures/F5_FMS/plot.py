"""Figure 5 (fig:fms): FMS ensembles vs Ehrenfest + forbidden-spawn density.

Self-contained: reads only data/ in this folder, writes figure.png/.pdf and
CSV exports (for external replotting).

data/fms_ensemble_agg.npz  aggregate of the Green Planet fms300 campaign
    time_fs                       common time grid (~40 fs window)
    {mode}_mean, {mode}_sem       ensemble mean/SEM of el_pop, columns
                                  [S0, S1, S2, norm]; mode = velocity|coupling
    {mode}_n                      number of initial conditions aggregated
    {mode}_frustrated_times_fs    times of gate-REJECTED spawn attempts
    {mode}_accepted_times_fs      times of accepted spawns
data/ehrenfest_mean.npz       300-IC diabatic Ehrenfest reference
    time_fs, mean_pops (401, 3)   columns [S0, S1, S2]
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "data", "fms_ensemble_agg.npz"), allow_pickle=True)
e = np.load(os.path.join(HERE, "data", "ehrenfest_mean.npz"))

t = d["time_fs"]
tmax = float(t.max())
te, pe = e["time_fs"], e["mean_pops"]
me = te <= tmax  # Ehrenfest restricted to the FMS window

COL = {"ehrenfest": "#2471a3", "velocity": "#c0392b", "coupling": "#2e8b57"}
BINW = 2.0  # fs, frustrated-event histogram

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.8, 6.4), sharex=True, height_ratios=[1.6, 1.0]
)

# --- panel A: S2 populations, FMS (both rescale modes) vs Ehrenfest ---
ax1.plot(te[me], pe[me, 2], color=COL["ehrenfest"], lw=2.0,
         label="Ehrenfest (300 ICs)")
for mode in ("velocity", "coupling"):
    m, s, n = d[f"{mode}_mean"], d[f"{mode}_sem"], int(d[f"{mode}_n"])
    ax1.plot(t, m[:, 2], color=COL[mode], lw=2.0,
             label=f"FMS {mode} rescale ({n} ICs)")
    ax1.fill_between(t, m[:, 2] - s[:, 2], m[:, 2] + s[:, 2],
                     color=COL[mode], alpha=0.25, lw=0)
ax1.set_ylabel(r"mean $S_2$ population")
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=False)

# --- panel B: forbidden (gate-rejected) spawn density over time ---
# event logs exist only for runs that completed in-band (salvaged runs carry
# no event data), so the per-IC normalization uses the completed count
edges = np.arange(0.0, tmax + BINW, BINW)
fr_c = d["coupling_frustrated_times_fs"]
n_c = int(d.get("coupling_n_completed", d["coupling_n"]))
hist, _ = np.histogram(fr_c, bins=edges)
ax2.bar(edges[:-1], hist / n_c, width=BINW, align="edge",
        color=COL["coupling"], alpha=0.85,
        label=f"coupling rescale ({len(fr_c)} events / {n_c} ICs)")
n_v = int(d.get("velocity_n_completed", d["velocity_n"]))
ax2.axhline(0.0, color=COL["velocity"], lw=2.0,
            label=f"velocity rescale (0 events / {n_v} ICs)")
ax2.set_xlabel("time (fs)")
ax2.set_ylabel(f"forbidden spawns\nper IC per {BINW:g} fs")
ax2.set_xlim(0, tmax)
ax2.legend(frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "figure.png"), dpi=200)
fig.savefig(os.path.join(HERE, "figure.pdf"))
print("wrote figure.png / figure.pdf")

# --- CSV exports for external replotting ---
with open(os.path.join(HERE, "data", "populations.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_fs", "S2_vel_mean", "S2_vel_sem", "S2_coup_mean",
                "S2_coup_sem", "S1_vel_mean", "S1_coup_mean"])
    for i in range(len(t)):
        w.writerow([f"{t[i]:.4f}",
                    f"{d['velocity_mean'][i, 2]:.5f}", f"{d['velocity_sem'][i, 2]:.5f}",
                    f"{d['coupling_mean'][i, 2]:.5f}", f"{d['coupling_sem'][i, 2]:.5f}",
                    f"{d['velocity_mean'][i, 1]:.5f}", f"{d['coupling_mean'][i, 1]:.5f}"])

with open(os.path.join(HERE, "data", "ehrenfest_window.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_fs", "S2_mean", "S1_mean"])
    for i in np.nonzero(me)[0]:
        w.writerow([f"{te[i]:.4f}", f"{pe[i, 2]:.5f}", f"{pe[i, 1]:.5f}"])

np.savetxt(os.path.join(HERE, "data", "forbidden_times_coupling.csv"),
           np.sort(fr_c), fmt="%.4f", header="t_fs", comments="")
print("wrote populations.csv / ehrenfest_window.csv / forbidden_times_coupling.csv")
