#!/usr/bin/env python3
"""
Prepare input.dat for the FFT code from an ORCA .scf.log file.

Usage:
    python3 prepare_fft.py geo_opt_qmd.scf.log --dt 0.2

Output:
    input.dat
        time(au)   deltaE(Hartree)   imag
"""

import argparse
import re

FS_TO_AU = 41.3413745758

parser = argparse.ArgumentParser(
    description="Extract QM energies from an ORCA .scf.log file."
)

parser.add_argument(
    "logfile",
    help="ORCA .scf.log file"
)

parser.add_argument(
    "--dt",
    type=float,
    required=True,
    help="MD timestep in fs"
)

parser.add_argument(
    "-o",
    "--output",
    default="input.dat",
    help="Output filename (default: input.dat)"
)

args = parser.parse_args()

pattern = re.compile(
    r"Total energy after final integration\s*:\s*([-\d.Ee+]+)"
)

energies = []

with open(args.logfile) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            energies.append(float(m.group(1)))

if len(energies) == 0:
    raise RuntimeError("No QM energies found!")

print(f"Found {len(energies)} QM energies.")

meanE = sum(energies) / len(energies)
dt = args.dt * FS_TO_AU

with open(args.output, "w") as out:
    for i, E in enumerate(energies):
        time = i * dt
        fluct = E - meanE
        out.write(f"{time:24.16e} {fluct:24.16e} {0.0:24.16e}\n")

print(f"Wrote {args.output}")
