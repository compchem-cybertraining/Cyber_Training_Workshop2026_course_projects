#!/usr/bin/env python
"""F2: model card -- v12a vs NEXMD on held-out data: the three gaps + signed
diabatic splitting + scaled-NACR components. Self-contained; CSV exports for
external plotting."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

HERE = Path(__file__).parent
d = np.load(HERE / "data" / "parity_v12a.npz")
blues = LinearSegmentedColormap.from_list("b", ["#dbe9f6", "#2471a3", "#0b2e4f"])
panels = [("g01", "S0-S1 (eV)", False), ("g02", "S0-S2 (eV)", False),
          ("g12", "S1-S2 (eV)", True), ("dd", "signed splitting (eV)", True),
          ("q", "scaled NACR (eV/A)", True)]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
for ax, (k, lbl, disp) in zip(axes, panels):
    tt, tp = d[f"thermal_{k}_t"], d[f"thermal_{k}_p"]
    np.savetxt(HERE / "data" / f"parity_thermal_{k}.csv",
               np.column_stack([tt, tp]), delimiter=",", header="true,pred", comments="")
    lo, hi = np.percentile(np.concatenate([tt, tp]), [0.2, 99.8])
    pad = 0.06 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(tt, tp, gridsize=45, cmap=blues, bins="log", extent=(lo, hi, lo, hi), mincnt=1)
    if disp:
        dt_, dp_ = d[f"displaced_{k}_t"], d[f"displaced_{k}_p"]
        np.savetxt(HERE / "data" / f"parity_displaced_{k}.csv",
                   np.column_stack([dt_, dp_]), delimiter=",", header="true,pred", comments="")
        ax.plot(dt_, dp_, ".", color="#e67e22", ms=1.5, alpha=0.4)
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.7)
    mae = np.abs(tp - tt).mean()
    ax.set_title(f"{lbl}\nMAE {mae*1000:.0f} meV" if "eV)" in lbl and "NACR" not in lbl
                 else f"{lbl}\nMAE {mae:.3f}", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.tick_params(labelsize=7)
axes[0].set_ylabel("NQDH prediction", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "figure.png", dpi=170); fig.savefig(HERE / "figure.pdf")
print("F2 done")
