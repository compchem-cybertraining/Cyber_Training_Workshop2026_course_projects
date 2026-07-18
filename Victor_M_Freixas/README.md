# Vibronic Coherence Across Nonadiabatic Dynamics Methods
### Victor M. Freixas, CyberTraining 2026 project

A machine-learned quasi-diabatic Hamiltonian (NQDH, HIP-NN based) for a
122-atom halofluorescein heterodimer drives five nonadiabatic dynamics
methods through one diabatic interface: a reference Ehrenfest integrator,
FSSH / SHXF / QTSH through Libra, and full multiple spawning through pySpawn,
at 300-initial-condition ensemble scale. Main finding: the survival of the
S1/S2 vibronic beat is decided by the per-trajectory energy gate at
surface-transfer events (surface-hopping family) and by the spawned-basis
size (FMS, which recovers the beat as the basis converges).

## Contents

- [`report.pdf`](report.pdf): the project report.
- [`project/`](project/): the full project, self-contained and runnable.
  - `project/README.md`: how to run the demo (single trajectory or ensemble).
  - `project/model/`: the frozen NQDH (weights + training configuration).
  - `project/nqdh_demo/`: the provider and the Libra / pySpawn interfaces.
  - `project/examples/`: single-trajectory and ensemble examples for every
    method, with runtime guidance.
  - `project/report/`: LaTeX source and one folder per figure, each with its
    raw data and a self-contained `plot.py` that regenerates it.

## Quick start

```bash
cd project
pip install torch hippynn matplotlib   # base demo needs no Libra/pySpawn
python run_demo.py                     # one trajectory, ~1 minute
python report/figures/F4_pop_comparison/plot.py   # regenerate a figure
```

## Note on data size

To respect the course-repo size guidance, the large per-trajectory arrays
inside four ensemble files under `project/report/figures/` were reduced to
their ensemble means (the quantities the figures actually use; every
`plot.py` runs unchanged), and two npz files not used by any figure were
omitted. The complete, unreduced archive is the standalone project
repository, which is the canonical home of this work:

**https://github.com/vmfreixas/nqdh-beating-demo**
