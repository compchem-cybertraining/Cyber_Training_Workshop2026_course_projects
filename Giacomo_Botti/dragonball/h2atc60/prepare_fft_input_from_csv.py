#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

FS_TO_AU = 41.3413745758

def clean(s: str) -> str:
    return s.strip()

def to_float(s: str):
    s = clean(s)
    if s == "":
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def find_column(header, name):
    target = name.strip().lower()
    for i, h in enumerate(header):
        if clean(h).lower() == target:
            return i
    return None

def main():
    p = argparse.ArgumentParser(
        description="Extract a time series from an ORCA semicolon-separated CSV and write input.dat for FFT."
    )
    p.add_argument("csvfile", help="ORCA CSV file")
    p.add_argument("-o", "--out", default="input.dat", help="Output file (default: input.dat)")
    p.add_argument(
        "--time-col",
        default="Sim. Time",
        help='Time column name (default: "Sim. Time")',
    )
    p.add_argument(
        "--value-col",
        default="Cons.Qty",
        help='Value column name (default: "Cons.Qty")',
    )
    p.add_argument(
        "--value-index",
        type=int,
        default=None,
        help="Fallback 1-based column index if names are not found",
    )
    p.add_argument(
        "--demean",
        action="store_true",
        help="Subtract the mean from the selected series before writing it",
    )
    p.add_argument(
        "--skip-blanks",
        action="store_true",
        help="Skip rows where the selected value is blank",
    )
    args = p.parse_args()

    csv_path = Path(args.csvfile)
    if not csv_path.is_file():
        sys.exit(f"File not found: {csv_path}")

    times = []
    values = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    if not rows:
        sys.exit("Empty file")

    header = [clean(x) for x in rows[0]]
    time_idx = find_column(header, args.time_col)
    value_idx = find_column(header, args.value_col)

    if time_idx is None:
        sys.exit(f'Time column "{args.time_col}" not found in header')

    if value_idx is None:
        if args.value_index is None:
            sys.exit(
                f'Value column "{args.value_col}" not found in header. '
                "Use --value-index to specify a 1-based column number."
            )
        value_idx = args.value_index - 1

    for row in rows[1:]:
        if len(row) <= max(time_idx, value_idx):
            continue

        t = to_float(row[time_idx])
        v = to_float(row[value_idx])

        if t is None:
            continue
        if v is None:
            if args.skip_blanks:
                continue
            else:
                # Keep the row out if the selected value is missing.
                continue

        times.append(t)
        values.append(v)

    if not times:
        sys.exit("No valid data rows found")

    if args.demean:
        mean_v = sum(values) / len(values)
        values = [v - mean_v for v in values]

    # Optional sanity check: print time step if it looks uniform
    if len(times) >= 3:
        dt1 = times[1] - times[0]
        uniform = all(abs((times[i] - times[i - 1]) - dt1) < 1e-8 for i in range(2, len(times)))
        if not uniform:
            print("Warning: time grid is not perfectly uniform. FFT assumes uniform spacing.", file=sys.stderr)

    with open(args.out, "w") as out:
        for t, v in zip(times, values):
            t_au = t * FS_TO_AU
            out.write(f"{t_au:.18e} {v:.18e} 0.0\n")

    print(f"Wrote {len(times)} points to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
