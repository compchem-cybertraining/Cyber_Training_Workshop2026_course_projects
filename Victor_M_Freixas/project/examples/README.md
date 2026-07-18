# Examples

Every example consumes the same frozen NQDH model (`model/`) and the same 300
thermal initial conditions (`data/ics.npz`). Single-trajectory examples are
meant to be *run*; ensemble examples are meant to be *submitted* — mind the
runtime column.

| script | what it shows | environment | runtime |
|---|---|---|---|
| `../run_demo.py` | the S1/S2 beat: diabatic Ehrenfest, single IC or 300-IC ensemble | base (torch+hippynn), optional Libra | ~1 min single / **~1 h ensemble (GPU)** |
| `run_tsh_single.py` | QTSH vs FSSH on one IC — the energy-gate experiment | **libra env** | ~10–20 min (GPU), default 48 fs |
| `run_fms_single.py` | full multiple spawning, one IC, spawn-event logging | base + **pySpawn17 (py3 port)** | minutes (smoke) / **~5–8 h at 40 fs** |
| `run_tsh_ensemble.py` | 300-IC hopping ensembles (fair comparison) | **libra env** | **~10–20 GPU-hours per method** |
| `run_fms_ensemble_slurm.sh` | 300-IC × 2-mode FMS campaign as a SLURM array | cluster (see header) | **~4,000 core-hours total** |

Practical notes:

- **Trial first.** Both ensemble scripts accept subsets (`--n-ics 10`; edit the
  `--array` range) and are **resumable** — finished ICs are skipped, so
  interrupting costs nothing.
- **Populations:** for hopping methods the physical populations are the
  active-surface fractions (`sh_pop_adi`); the coherent electronic populations
  (`se_pop_adi`) are also saved.
- **Expected physics** (what you should reproduce): Ehrenfest shows the
  vibronic beat (recurrences every ~18 fs, decohering by ~90 fs). QTSH keeps
  the ensemble S2 population at the Ehrenfest level; FSSH and SHXF decay to
  zero, because the per-trajectory energy gate blocks the upward return
  transfer. FMS damps the beat at affordable basis sizes but progressively
  recovers it as the spawned basis grows (group trajectories by their spawn
  count to see the trend). See the report in `../report/`.
