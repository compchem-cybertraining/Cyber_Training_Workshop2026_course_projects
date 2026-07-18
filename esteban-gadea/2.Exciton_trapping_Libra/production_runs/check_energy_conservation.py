"""
Total-energy conservation check for one production run -- rules out numerical
integration drift as the source of any observed lattice-distortion growth.
Reads `<prefix>/mem_data.hdf`'s saved energy components and saves
`<prefix>_energy_conservation.png`.
"""
import sys
import _pathsetup  # noqa: F401
import h5py
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <run_prefix>")
PREFIX = sys.argv[1]
H5PATH = f"{PREFIX}/mem_data.hdf"

with h5py.File(H5PATH, "r") as f:
    t = np.array(f["time/data"]).flatten()
    epot = np.array(f["Epot_ave/data"]).flatten()
    ekin = np.array(f["Ekin_ave/data"]).flatten()
    etot = np.array(f["Etot_ave/data"]).flatten()

drift_abs = etot - etot[0]
drift_rel = drift_abs / abs(etot[0]) if etot[0] != 0 else drift_abs

print(f"=== {PREFIX} ===")
print(f"E_tot(t=0)   = {etot[0]:.8f} Ha")
print(f"E_tot(t=end) = {etot[-1]:.8f} Ha")
print(f"Absolute drift at t=end = {drift_abs[-1]:.3e} Ha")
print(f"Relative drift at t=end = {drift_rel[-1]:.3e}")
print(f"Max |drift| over the whole run = {np.max(np.abs(drift_abs)):.3e} Ha "
      f"(at t={t[np.argmax(np.abs(drift_abs))]:.1f} a.u.)")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(t, epot, label="E_pot", color="tab:orange")
axes[0].plot(t, ekin, label="E_kin", color="tab:green")
axes[0].plot(t, etot, label="E_tot", color="black", linewidth=2)
axes[0].set_ylabel("Energy (Ha)")
axes[0].legend()
axes[0].set_title(f"{PREFIX}: energy components vs. time")

axes[1].plot(t, drift_abs, color="black")
axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[1].set_xlabel("time (a.u.)")
axes[1].set_ylabel("E_tot(t) - E_tot(0)  (Ha)")
axes[1].set_title("Total energy drift")

plt.tight_layout()
plt.savefig(f"{PREFIX}_energy_conservation.png", dpi=150)
plt.show()
print(f"Saved {PREFIX}_energy_conservation.png")
