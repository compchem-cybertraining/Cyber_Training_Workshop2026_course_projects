#!/usr/bin/env python3
"""
plot_pes.py  --  Plot the photoelectron spectrum from an OpenMolcas RASSI Dyson run.

Reads the "Dyson amplitudes" table from a RASSI output (.out) file and builds the
spectrum by broadening each (binding energy, Dyson intensity) stick with a Gaussian.

State convention for this example (neutral states listed first in RASSI):
    From = 1  ->  GROUND-STATE spectrum (ionizing S0)   [plotted in BLUE]
    From = 2  ->  EXCITED-STATE spectrum (ionizing S1)  [plotted in RED]

Usage:
    python plot_pes.py Test.out
    python plot_pes.py Test.out --fwhm 0.5 --separate
"""

import argparse
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_dyson(filename):
    """Return dict: {from_state: [(BE_eV, intensity), ...]} parsed from RASSI output."""
    data = {}
    in_block = False
    started_rows = False
    # rows look like:  "   1    3    8.844    9.90263E-01"
    row = re.compile(r"^\s*(\d+)\s+(\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.?\d+[eE][-+]?\d+)\s*$")
    with open(filename) as fh:
        for line in fh:
            if "Dyson amplitudes" in line:
                in_block = True
                continue
            if in_block:
                m = row.match(line)
                if m:
                    started_rows = True
                    frm = int(m.group(1))
                    be = float(m.group(3))
                    inten = float(m.group(4))
                    data.setdefault(frm, []).append((be, inten))
                    continue
                # end the block only once we've seen data rows and hit a non-row line
                if started_rows and (line.strip().startswith("--")
                                     or "Special properties" in line
                                     or line.strip() == ""):
                    break
    return data


def broaden(sticks, fwhm, grid):
    """Sum of Gaussians on `grid` for a list of (center, height) sticks."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y = np.zeros_like(grid)
    for be, inten in sticks:
        y += inten * np.exp(-0.5 * ((grid - be) / sigma) ** 2)
    return y


def main():
    ap = argparse.ArgumentParser(description="Plot RASSI Dyson photoelectron spectra.")
    ap.add_argument("outfile", help="RASSI output file (e.g. Test.out)")
    ap.add_argument("--fwhm", type=float, default=0.5, help="Gaussian FWHM in eV (default 0.5)")
    ap.add_argument("--separate", action="store_true",
                    help="Make two separate panels instead of one overlaid plot")
    ap.add_argument("--imin", type=float, default=1e-5,
                    help="Ignore Dyson intensities below this (default 1e-5)")
    args = ap.parse_args()

    data = parse_dyson(args.outfile)
    if not data:
        sys.exit("No Dyson amplitudes table found in " + args.outfile)

    # filter weak lines
    for k in data:
        data[k] = [(be, i) for be, i in data[k] if i >= args.imin]

    labels = {1: ("Ground state (ionize S0)", "tab:blue"),
              2: ("Excited state (ionize S1)", "tab:red")}

    # energy grid spanning all binding energies
    all_be = [be for sticks in data.values() for be, _ in sticks]
    lo, hi = min(all_be) - 2.0, max(all_be) + 2.0
    grid = np.linspace(lo, hi, 2000)

    if args.separate and len(data) > 1:
        fig, axes = plt.subplots(len(data), 1, figsize=(8, 3 * len(data)), sharex=True)
        for ax, frm in zip(axes, sorted(data)):
            name, color = labels.get(frm, (f"From state {frm}", "k"))
            y = broaden(data[frm], args.fwhm, grid)
            ax.plot(grid, y, color=color, lw=2, label=name)
            ax.vlines([be for be, _ in data[frm]], 0,
                      [i for _, i in data[frm]], color=color, alpha=0.35, lw=1)
            ax.set_ylabel("Intensity")
            ax.legend(frameon=False)
            ax.tick_params(direction="in")
        axes[-1].set_xlabel("Binding energy (eV)")
        fig.suptitle("Uracil photoelectron spectrum (Dyson / RASSI)")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        for frm in sorted(data):
            name, color = labels.get(frm, (f"From state {frm}", "k"))
            y = broaden(data[frm], args.fwhm, grid)
            ax.plot(grid, y, color=color, lw=2, label=name)
            ax.vlines([be for be, _ in data[frm]], 0,
                      [i for _, i in data[frm]], color=color, alpha=0.30, lw=1)
        ax.set_xlabel("Binding energy (eV)")
        ax.set_ylabel("Intensity")
        ax.set_title("Uracil photoelectron spectrum (Dyson / RASSI)")
        ax.legend(frameon=False)
        ax.tick_params(direction="in")

    plt.tight_layout()
    out = "uracil_pes.png"
    fig.savefig(out, dpi=300)
    print("Wrote", out)


if __name__ == "__main__":
    main()
