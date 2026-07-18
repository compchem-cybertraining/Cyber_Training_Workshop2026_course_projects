"""
Bond-distortion and hole/electron density (IPR) diagnostics for one production
run. Produces `<prefix>_maps.png` (hole/electron density and bond-distortion
heatmaps) and `<prefix>_localization.png` (IPR vs. RMS bond distortion). For
run3_control there's no exciton, so only the bond-distortion map and RMS trace
are produced.
"""
import sys
import _pathsetup  # noqa: F401
import h5py
import numpy as np
import matplotlib.pyplot as plt

from cis_compute_adi import get_default_params
from exciton_density import cis_windowed_state, electron_hole_density, ipr

if len(sys.argv) != 2:
    sys.exit(f"usage: python {sys.argv[0]} <run_prefix>")
PREFIX = sys.argv[1]
H5PATH = f"{PREFIX}/mem_data.hdf"
IS_GROUND_ONLY = PREFIX == "run3_control"
STRIDE = 250   # recompute the CIS state every STRIDE-th saved step (density recompute is the
               # expensive part of this script; adjust down for finer time resolution)

with h5py.File(H5PATH, "r") as f:
    q = np.array(f["q/data"])
    t = np.array(f["time/data"]).flatten()
if q.ndim == 3:
    q = q[:, 0, :]
nsteps_saved, ndof = q.shape
nchain = ndof // 2

p = get_default_params(nchain=nchain, dimer1=0.095665, lattice_ang=6.0, hartreeu=0.0)
boxl = p["boxl"]
print(f"{PREFIX}: nchain={nchain} (ndof={ndof}), {nsteps_saved} saved steps")

# ---------------- bond-length distortion from t=0, every bond, every saved step ----------------
bonds = np.zeros((nsteps_saved, ndof))
for k in range(ndof - 1):
    bonds[:, k] = q[:, k + 1] - q[:, k]
bonds[:, ndof - 1] = (q[:, 0] + boxl) - q[:, ndof - 1]
bond_change = bonds - bonds[0, :]
rms_bond = np.sqrt(np.mean(bond_change ** 2, axis=1))

bond_vmax = np.max(np.abs(bond_change))


def plot_bond_map(ax, fig):
    im = ax.imshow(bond_change.T, aspect="auto", origin="lower",
                    extent=[t[0], t[-1], 0, ndof], cmap="RdBu_r",
                    vmin=-bond_vmax, vmax=bond_vmax)
    ax.set_ylabel("bond index")
    ax.set_title(f"bond-length distortion delta_r(bond, time)")
    fig.colorbar(im, ax=ax, label="delta bond length (bohr)")
    return im


if IS_GROUND_ONLY:
    fig_maps, ax_map = plt.subplots(figsize=(8, 4))
    plot_bond_map(ax_map, fig_maps)
    ax_map.set_xlabel("time (a.u.)")
    plt.tight_layout()
    plt.savefig(f"{PREFIX}_maps.png", dpi=150)
    plt.show()
    print(f"Saved {PREFIX}_maps.png")

    fig_loc, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, rms_bond, color="black")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("RMS bond-length change (bohr)")
    ax.set_title(f"RMS bond distortion")
    plt.tight_layout()
    plt.savefig(f"{PREFIX}_localization.png", dpi=150)
    plt.show()
    print(f"Saved {PREFIX}_localization.png")
    raise SystemExit

# ---------------- hole/electron density + IPR at a subsample of steps ----------------
idxs = list(range(0, nsteps_saved, STRIDE))
if idxs[-1] != nsteps_saved - 1:
    idxs.append(nsteps_saved - 1)
t_sub = t[idxs]

rho_hole_t = np.zeros((len(idxs), ndof))
rho_elec_t = np.zeros((len(idxs), ndof))
ipr_hole_t = np.zeros(len(idxs))
ipr_elec_t = np.zeros(len(idxs))

for k, idx in enumerate(idxs):
    rion = q[idx, :]
    E_n, Psi, configs, eps, C = cis_windowed_state(
        rion, boxl, p["fxcalpha"], p["fxcgamma"], p["hartreeu"], p["n_near"],
        p["hop"], p["hopslope"], p["req"])
    rho_h, rho_e = electron_hole_density(Psi, configs, C)
    rho_hole_t[k] = rho_h
    rho_elec_t[k] = rho_e
    ipr_hole_t[k] = ipr(rho_h)
    ipr_elec_t[k] = ipr(rho_e)
    if k % max(1, len(idxs) // 20) == 0:
        print(f"  {k + 1}/{len(idxs)}  t={t_sub[k]:.1f} a.u.  E_n={E_n:.6f} Ha  "
              f"IPR_hole={ipr_hole_t[k]:.2f}  IPR_elec={ipr_elec_t[k]:.2f}  (n={ndof})")

# ---------------- figure 1: hole/electron density maps + bond-distortion map ----------------
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

im0 = axes[0].imshow(rho_hole_t.T, aspect="auto", origin="lower",
                      extent=[t_sub[0], t_sub[-1], 0, ndof], cmap="inferno")
axes[0].set_ylabel("atom index")
axes[0].set_title(f"hole density rho_hole(atom, time)")
fig.colorbar(im0, ax=axes[0], label="probability")

im1 = axes[1].imshow(rho_elec_t.T, aspect="auto", origin="lower",
                      extent=[t_sub[0], t_sub[-1], 0, ndof], cmap="viridis")
axes[1].set_ylabel("atom index")
axes[1].set_title(f"electron density rho_elec(atom, time)")
fig.colorbar(im1, ax=axes[1], label="probability")

plot_bond_map(axes[2], fig)
axes[2].set_xlabel("time (a.u.)")

plt.tight_layout()
plt.savefig(f"{PREFIX}_maps.png", dpi=150)
plt.show()
print(f"Saved {PREFIX}_maps.png")

# ---------------- figure 2: localization (IPR) vs. lattice distortion ----------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(t_sub, ipr_hole_t, label="IPR(hole)", color="tab:red")
ax2.plot(t_sub, ipr_elec_t, label="IPR(electron)", color="tab:blue")
ax2.axhline(ndof, color="gray", linestyle=":", linewidth=1, label=f"n={ndof} (fully delocalized)")
ax2.set_ylabel("IPR (64=delocalized, 1=fully localized)")
ax2.legend(loc="upper left")
ax2r = ax2.twinx()
ax2r.plot(t, rms_bond, color="black", alpha=0.5, linewidth=1, label="RMS bond distortion")
ax2r.set_ylabel("RMS bond-length change (bohr)")
ax2r.legend(loc="upper right")
ax2.set_xlabel("time (a.u.)")
ax2.set_title(f"localization (IPR) vs. lattice distortion")

plt.tight_layout()
plt.savefig(f"{PREFIX}_localization.png", dpi=150)
plt.show()
print(f"Saved {PREFIX}_localization.png")
