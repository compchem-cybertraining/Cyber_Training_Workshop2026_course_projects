"""
Overlays RMS bond-length distortion vs. time for all three production runs on
one plot. Saves `production_runs_comparison.png`.
"""
import _pathsetup  # noqa: F401
import h5py
import numpy as np
import matplotlib.pyplot as plt

from cis_compute_adi import get_default_params

RUNS = [
    ("run1_unseeded", "Run 1: unseeded exciton (delocalized baseline)", "tab:gray"),
    ("run2_selftrapping", "Run 2: seeded exciton (self-trapping)", "tab:red"),
    ("run3_control", "Run 3: seeded ground state (control)", "tab:blue"),
]


def rms_bond_distortion(prefix):
    h5path = f"{prefix}/mem_data.hdf"
    with h5py.File(h5path, "r") as f:
        q = np.array(f["q/data"])
        t = np.array(f["time/data"]).flatten()
    if q.ndim == 3:
        q = q[:, 0, :]
    nsteps_saved, ndof = q.shape
    nchain = ndof // 2
    p = get_default_params(nchain=nchain, dimer1=0.086419, lattice_ang=6.0, hartreeu=0.0)
    boxl = p["boxl"]

    bonds = np.zeros((nsteps_saved, ndof))
    for k in range(ndof - 1):
        bonds[:, k] = q[:, k + 1] - q[:, k]
    bonds[:, ndof - 1] = (q[:, 0] + boxl) - q[:, ndof - 1]
    bond_change = bonds - bonds[0, :]
    rms = np.sqrt(np.mean(bond_change ** 2, axis=1))
    return t, rms


plt.figure(figsize=(10, 6))
for prefix, label, color in RUNS:
    try:
        t, rms = rms_bond_distortion(prefix)
        plt.plot(t, rms, color=color, label=label, alpha=0.9)
        print(f"{prefix:20s} final RMS bond distortion = {rms[-1]:.4f} bohr")
    except (OSError, KeyError) as e:
        print(f"SKIPPED {prefix} (not found or unreadable -- run it first): {e!r}")

plt.xlabel("time (a.u.)")
plt.ylabel("RMS bond-length change from t=0 (bohr)")
plt.title("Exciton self-trapping: the three production runs side by side")
plt.legend()
plt.tight_layout()
plt.savefig("production_runs_comparison.png", dpi=150)
plt.show()
print("Saved production_runs_comparison.png")
