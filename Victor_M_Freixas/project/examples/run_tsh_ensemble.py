#!/usr/bin/env python
"""Ensemble surface hopping on the frozen NQDH: the fair 300-IC comparison.

*** RUNTIME WARNING ***********************************************************
* One initial condition = one call to run_tsh (2000 steps x ntraj replicas).  *
* Measured cost: ~2-4 min/IC on a GPU, ~15-30 min/IC on CPU.  The full        *
* 300-IC ensemble is ~10-20 GPU-HOURS per method (we ran it with 6 concurrent *
* workers overnight).  Use --n-ics for a subset trial first (e.g. 10).        *
*******************************************************************************

The script is RESUMABLE: finished ICs (existing tsh_populations.npz) are
skipped, so it can be interrupted and relaunched, or run concurrently from
several shells (workers race benignly on the same output tree, e.g.
    for i in 1 2 3 4 5 6; do python examples/run_tsh_ensemble.py \\
        --method qtsh --device cuda & done ).

After the loop it aggregates whatever exists into an ensemble-mean beat curve
(beating-format npz + a quick plot); rerun with --aggregate-only to just
collect.  Requires the ``libra`` conda environment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AU_FS = 0.0241888


def aggregate(ens_dir: Path, out: Path) -> None:
    files = sorted(ens_dir.glob("ic*/tsh_populations.npz"))
    sh_list, t = [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        sh = np.asarray(d["sh_pop_adi"])
        if t is None:
            t = np.asarray(d["time_au"]).ravel() * AU_FS
        if sh.shape[0] == len(t):
            sh_list.append(sh)
    if not sh_list:
        raise SystemExit(f"no finished ICs under {ens_dir}")
    sh = np.stack(sh_list)
    np.savez(out, time_fs=t, mean_pops=sh.mean(0),
             adiabatic_pops=np.transpose(sh, (1, 0, 2)), n_ic=len(sh_list))
    print(f"aggregated {len(sh_list)} ICs -> {out}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, sh.mean(0)[:, 2], lw=2, label=f"S2 ({len(sh_list)} ICs)")
    ax.plot(t, sh.mean(0)[:, 1], lw=2, label="S1")
    ax.set_xlabel("time (fs)"); ax.set_ylabel("ensemble population")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(str(out).replace(".npz", ".png"), dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("fssh", "qtsh", "shxf"), default="qtsh")
    p.add_argument("--n-ics", type=int, default=300, help="use a small number for a trial!")
    p.add_argument("--ntraj", type=int, default=2, help="hop-RNG replicas per IC")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()

    out_dir = a.out_dir or Path(f"output/tsh_ens_{a.method}")
    out_npz = out_dir / f"beating_{a.method}.npz"

    if not a.aggregate_only:
        from nqdh_demo.tsh_libra import run_tsh
        for ic in range(a.n_ics):
            d = out_dir / f"ic{ic:03d}"
            if (d / "tsh_populations.npz").exists():
                continue
            print(f"[{a.method}] ic {ic:03d}", flush=True)
            try:
                run_tsh(ROOT / "model", ROOT / "data" / "ics.npz", d, method=a.method,
                        ic_index=ic, ntraj=a.ntraj, n_steps=a.n_steps, device=a.device,
                        seed=ic + 7)
            except Exception as exc:      # noqa: BLE001  keep the ensemble going
                print(f"[ic {ic:03d} FAILED: {exc}]", flush=True)

    aggregate(out_dir, out_npz)


if __name__ == "__main__":
    main()
