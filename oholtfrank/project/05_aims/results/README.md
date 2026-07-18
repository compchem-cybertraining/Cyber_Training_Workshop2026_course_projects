# AIMS trajectory data

`cascade_stats.txt` — extracted spawn statistics for all 16 trajectories.
This is what the cascade figure is built from.

Raw `sim.hdf5` trajectory files (16 files, ~295 MB) are archived separately
[Zenodo DOI: TBD] — they are too large for this repo. Each contains full
positions, momenta, amplitudes, and state assignments for every TBF.

To regenerate `cascade_stats.txt` from the raw data:
    python3 ../scripts/cascade_stats.py
run from a directory containing the numbered trajectory folders.
