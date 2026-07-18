#!/usr/bin/env python
"""Example 2: surface hopping with and without the energy gate, one trajectory.

Runs QTSH (gate removed) and FSSH (gate on) on the same initial condition and
plots the S2 populations side by side -- the single-IC version of the report's
gate experiment.  Requires the ``libra`` conda environment.

Runtime: ~10-20 min on a GPU, ~40-80 min on CPU, for the default 500 steps
(~48 fs, enough to see the first two recurrences).  Use --n-steps 2000 for the
full 194 fs window of the report.

    conda activate libra
    python examples/run_tsh_single.py [--device cuda] [--n-steps 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nqdh_demo.tsh_libra import run_tsh  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-index", type=int, default=0)
    p.add_argument("--ntraj", type=int, default=4, help="hop-RNG replicas")
    p.add_argument("--n-steps", type=int, default=500, help="500 = 48 fs; 2000 = full window")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("output/tsh_single"))
    a = p.parse_args()

    results = {}
    for method in ("qtsh", "fssh"):
        print(f"== {method} (ic {a.ic_index}, {a.ntraj} replicas, {a.n_steps} steps) ==", flush=True)
        t, sh, se = run_tsh(ROOT / "model", ROOT / "data" / "ics.npz",
                            a.out_dir / method, method=method, ic_index=a.ic_index,
                            ntraj=a.ntraj, n_steps=a.n_steps, device=a.device)
        results[method] = (t * 0.0241888, sh[:, 2])
        print(f"   final S2 surface population: {sh[-1, 2]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*results["qtsh"], color="#1e8449", lw=2, label="QTSH (gate removed)")
    ax.plot(*results["fssh"], color="#34495e", lw=2, ls=":", label="FSSH (energy gate)")
    ax.set_xlabel("time (fs)"); ax.set_ylabel("S2 population")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25); ax.legend()
    ax.set_title("The energy gate decides whether S2 can repopulate")
    fig.tight_layout()
    out = a.out_dir / "tsh_single.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    print("Expected behaviour: QTSH keeps returning to S2 (recurrences); FSSH decays and stays.")


if __name__ == "__main__":
    main()
