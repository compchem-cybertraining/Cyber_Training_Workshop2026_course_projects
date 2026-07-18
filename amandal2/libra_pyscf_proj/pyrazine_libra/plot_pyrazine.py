#!/usr/bin/env python
"""
plot_pyrazine.py -- ensemble analysis of the pyrazine S2->S1 array runs.

Each SLURM task wrote its own directory:
    runs/pyrazine-m{method}-{suff}-S{bright}-icond{N}/mem_data.hdf

One trajectory is an anecdote. The physics -- the S2->S1 internal-conversion
timescale -- only emerges as an ENSEMBLE AVERAGE over the Wigner initial conditions.
This script gathers every icond directory that matches, averages them, and makes:

    1. population decay, S2 (bright) draining into S1, with the ensemble spread
    2. an energy-conservation panel  (the referee: if Etot drifts, distrust the rest)
    3. a 1/e timescale estimate with a bootstrap error bar

Usage:
    python plot_pyrazine.py                          # auto-detect the run family
    python plot_pyrazine.py --glob 'runs/pyrazine-m2-*'
    python plot_pyrazine.py --outdir runs --method 2 --save figs
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless: works over SSH / in a batch job
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    sys.exit("h5py not importable -- run this on the libra kernel/env")

AU2FS = 0.02418884254               # 1 a.u. of time in fs
AU2EV = 27.211386


# ---------------------------------------------------------------------------
#  Reading
# ---------------------------------------------------------------------------

def _read_dataset(f, name):
    """Return f[name/data] as ndarray, or None if absent. Libra nests under /data."""
    key = f"{name}/data"
    if key in f:
        return np.array(f[key])
    if name in f:                   # tolerate a flatter layout
        return np.array(f[name])
    return None


def load_trajectory(hdf_path, nstates):
    """
    Pull time, populations, and energies from one mem_data.hdf.

    se_pop_adi / sh_pop_adi arrive as (nsteps, nstates) or (nsteps, ntraj, nstates);
    since each run here is ntraj=1 we squeeze the middle axis if present.
    Returns a dict, or None if the file is unreadable/incomplete.
    """
    try:
        with h5py.File(hdf_path, "r") as f:
            t = _read_dataset(f, "time")
            if t is None:
                return None
            t = np.asarray(t, float).reshape(-1) * AU2FS

            out = {"t": t}
            for key in ("se_pop_adi", "sh_pop_adi"):
                p = _read_dataset(f, key)
                if p is not None:
                    p = np.asarray(p, float)
                    if p.ndim == 3:              # (nsteps, ntraj, nstates)
                        p = p.mean(axis=1)
                    out[key] = p.reshape(len(t), -1)[:, :nstates]

            for key in ("Epot_ave", "Ekin_ave", "Etot_ave"):
                e = _read_dataset(f, key)
                if e is not None:
                    out[key] = np.asarray(e, float).reshape(-1)
            return out
    except (OSError, KeyError) as exc:
        print(f"  skip {hdf_path}: {type(exc).__name__}")
        return None


def gather(run_glob, nstates):
    """Load every matching trajectory onto a common time grid. Returns (t, list-of-dicts)."""
    dirs = sorted(glob.glob(run_glob))
    if not dirs:
        sys.exit(f"no directories match {run_glob!r}")

    trajs, t_ref = [], None
    for d in dirs:
        hdf = os.path.join(d, "mem_data.hdf")
        if not os.path.exists(hdf):
            print(f"  {os.path.basename(d)}: no mem_data.hdf (crashed or still running)")
            continue
        tr = load_trajectory(hdf, nstates)
        if tr is None or "se_pop_adi" not in tr:
            continue
        trajs.append(tr)
        if t_ref is None or len(tr["t"]) > len(t_ref):
            t_ref = tr["t"]

    if not trajs:
        sys.exit("no readable trajectories with populations")

    # Trajectories can differ in length (some crashed early). Truncate to the
    # shortest so the average is over the SAME set at every time point -- otherwise
    # the late-time average is silently over fewer, more biased, survivors.
    nmin = min(len(tr["t"]) for tr in trajs)
    print(f"\n{len(trajs)} trajectories, common length {nmin} steps "
          f"({t_ref[nmin-1]:.1f} fs)")
    if nmin < len(t_ref):
        print(f"  (some ran longer -- truncated to the shortest for an honest average)")
    return t_ref[:nmin], trajs, nmin


# ---------------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------------

def stack(trajs, key, nmin, nstates):
    """(ntraj, nmin, nstates) array of one population field, NaN-padded if missing."""
    out = np.full((len(trajs), nmin, nstates), np.nan)
    for i, tr in enumerate(trajs):
        if key in tr:
            out[i] = tr[key][:nmin, :nstates]
    return out


def tau_1e(t, pop_bright):
    """First crossing of 1/e by linear interpolation; NaN if it never falls that far."""
    thr = 1.0 / np.e
    below = np.where(pop_bright < thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    t0, t1 = t[j-1], t[j]
    p0, p1 = pop_bright[j-1], pop_bright[j]
    return t0 + (thr - p0) * (t1 - t0) / (p1 - p0)


def bootstrap_tau(t, pop_stack_bright, n_boot=2000, seed=0):
    """Bootstrap the 1/e time over the trajectory ensemble. Returns (median, lo, hi)."""
    rng = np.random.default_rng(seed)
    ntraj = pop_stack_bright.shape[0]
    taus = []
    for _ in range(n_boot):
        idx = rng.integers(0, ntraj, ntraj)
        tau = tau_1e(t, np.nanmean(pop_stack_bright[idx], axis=0))
        if not np.isnan(tau):
            taus.append(tau)
    if not taus:
        return np.nan, np.nan, np.nan
    return (np.median(taus), np.percentile(taus, 16), np.percentile(taus, 84))


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------

def make_plots(t, trajs, nmin, nstates, bright, method, savedir):
    se = stack(trajs, "se_pop_adi", nmin, nstates)
    ntraj = se.shape[0]
    se_mean = np.nanmean(se, axis=0)
    se_sem = np.nanstd(se, axis=0) / np.sqrt(ntraj)

    cols = plt.cm.viridis(np.linspace(0, 0.9, nstates))

    # ---- Figure 1: populations ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(nstates):
        lab = f"S{s}" + (" (bright, initial)" if s == bright else "")
        ax.plot(t, se_mean[:, s], lw=2.5, color=cols[s], label=lab)
        ax.fill_between(t, se_mean[:, s]-se_sem[:, s], se_mean[:, s]+se_sem[:, s],
                        color=cols[s], alpha=0.2)
    # thin lines: individual trajectories on the bright state, to show the spread
    for i in range(ntraj):
        ax.plot(t, se[i, :, bright], color=cols[bright], lw=0.4, alpha=0.25)

    ax.axhline(1/np.e, ls=":", c="grey", lw=1)
    ax.text(t[-1]*0.98, 1/np.e+0.02, "1/e", ha="right", fontsize=8, color="grey")
    ax.axvspan(20, 30, alpha=0.10, color="k")
    ax.text(25, 1.03, "lit. S2->S1\n~20-30 fs", ha="center", va="bottom", fontsize=8)

    tau, lo, hi = bootstrap_tau(t, se[:, :, bright])
    if not np.isnan(tau):
        ax.axvline(tau, ls="--", c=cols[bright], lw=1.5)
        ax.text(tau, 0.5, f"  tau(1/e) = {tau:.1f} fs\n  [{lo:.1f}, {hi:.1f}]",
                fontsize=8, color=cols[bright])

    ax.set_xlabel("time, fs"); ax.set_ylabel("adiabatic population (SE)")
    ax.set_ylim(-0.02, 1.08); ax.set_xlim(t[0], t[-1])
    ax.set_title(f"pyrazine S{bright} decay, {ntraj} trajectories "
                 f"(band = standard error)")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(savedir, "pyrazine_populations.png")
    fig.savefig(f1, dpi=150); print("wrote", f1)

    # ---- Figure 2: energy conservation (the referee) ----
    if all("Etot_ave" in tr for tr in trajs):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        etot = np.array([tr["Etot_ave"][:nmin] for tr in trajs])
        # drift per trajectory, relative to its own start, in meV
        drift = (etot - etot[:, [0]]) * AU2EV * 1000
        for i in range(ntraj):
            ax[0].plot(t, drift[i], lw=0.6, alpha=0.4)
        ax[0].plot(t, drift.mean(0), lw=2.5, color="k", label="ensemble mean")
        ax[0].axhline(0, ls=":", c="grey")
        ax[0].set_xlabel("time, fs"); ax[0].set_ylabel("Etot drift, meV")
        ax[0].set_title("energy conservation per trajectory"); ax[0].legend(fontsize=8)

        epot = np.array([tr["Epot_ave"][:nmin] for tr in trajs]).mean(0)
        ekin = np.array([tr["Ekin_ave"][:nmin] for tr in trajs]).mean(0)
        ax[1].plot(t, (epot-epot[0])*AU2EV, label="Epot", lw=2)
        ax[1].plot(t, (ekin-ekin[0])*AU2EV, label="Ekin", lw=2)
        ax[1].plot(t, (epot+ekin-epot[0]-ekin[0])*AU2EV, label="Etot", lw=2.5, c="k")
        ax[1].set_xlabel("time, fs"); ax[1].set_ylabel("energy rel. to t=0, eV")
        ax[1].set_title("mean energy flow (Epot -> Ekin as it leaves FC)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        f2 = os.path.join(savedir, "pyrazine_energy.png")
        fig.savefig(f2, dpi=150); print("wrote", f2)

        worst = np.abs(drift[:, -1]).max()
        print(f"\nmax |Etot drift| at final time: {worst:.1f} meV")
        if worst > 50:
            print("  >50 meV: distrust the populations. Tighten grid_level or shorten dt.")
        else:
            print("  under 50 meV: acceptable.")
    else:
        print("\nno Etot_ave in the files -- skipping the energy panel")

    # ---- text summary ----
    print("\n" + "="*56)
    print(f"  trajectories averaged : {ntraj}")
    print(f"  window                : 0 - {t[-1]:.1f} fs")
    final_bright = se_mean[-1, bright]
    print(f"  final S{bright} population : {final_bright:.3f} "
          f"(expect near 0 if IC completed)")
    if not np.isnan(tau):
        print(f"  tau(1/e)              : {tau:.1f} fs  [{lo:.1f}, {hi:.1f}] (68% boot)")
        print(f"  literature            : ~20-30 fs")
    else:
        print(f"  S{bright} never reached 1/e -- either too short a window, or the")
        print(f"    interface is not transferring population (check St off-diagonals)")
    print("="*56)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=None,
                    help="glob for run directories; overrides --outdir/--method")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--method", type=int, default=2, help="2=ehrenfest_adi_ld, 4=fssh")
    ap.add_argument("--nstates", type=int, default=None,
                    help="defaults to reading it from runs/prep.npz (nexc+1)")
    ap.add_argument("--bright", type=int, default=None,
                    help="initial bright state index; default from prep.npz")
    ap.add_argument("--save", default=".", help="directory for the PNGs")
    args = ap.parse_args()

    # infer nstates / bright from prep.npz if available
    nstates, bright = args.nstates, args.bright
    prep = os.path.join(args.outdir, "prep.npz")
    if (nstates is None or bright is None) and os.path.exists(prep):
        d = np.load(prep, allow_pickle=True)
        if nstates is None:
            nstates = int(d["nexc"]) + 1
        if bright is None:
            bright = int(d["bright"])
        print(f"from {prep}: nstates={nstates}, bright=S{bright}")
    if nstates is None:
        nstates = 6
        print(f"prep.npz not found -- assuming nstates={nstates}")
    if bright is None:
        bright = 1
        print(f"prep.npz not found -- assuming bright=S{bright}")

    run_glob = args.glob or os.path.join(args.outdir, f"pyrazine-m{args.method}-*")
    print(f"gathering: {run_glob}")

    os.makedirs(args.save, exist_ok=True)
    t, trajs, nmin = gather(run_glob, nstates)
    make_plots(t, trajs, nmin, nstates, bright, args.method, args.save)


if __name__ == "__main__":
    main()
