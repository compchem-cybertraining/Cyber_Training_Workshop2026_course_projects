# NQDH beating demo — a learned Hamiltonian drives Libra dynamics

A **frozen, pre-trained Neural Quasi-Diabatic Hamiltonian (NQDH)** drives
**diabatic mean-field (Ehrenfest) dynamics** and reproduces the **S1/S2 vibronic
quantum beat** of a halofluorescein chromophore heterodimer (122 atoms).

The model is a neural network that, from atomic numbers `Z` and positions `R`,
outputs a smooth symmetric matrix `W(R)` (a diabatic Hamiltonian) and its gradient
`dW/dR`. The dynamics move nuclei under the mean-field force `F = -Tr(rho dW/dR)`
while the electronic coefficients evolve under `i c_dot = W c`. Because this runs
in the **diabatic** basis, nothing diverges at the S1/S2 near-degeneracy where
adiabatic methods blow up. Diagonalising `W` along each trajectory gives the
adiabatic S1/S2 populations; averaging over a thermal ensemble reveals the beat.

Two engines consume the **identical** model `W`, `dW/dR` — two ways to propagate
the same physics (the Libra adapter was validated against the builtin engine to
~1e-5 on a controlled zero-velocity test):

- **`--engine libra`** — Libra's `tsh.generic_recipe` via a `compute_model`
  adapter. This is the school software, and the featured path.
- **`--engine builtin`** — a small pure-numpy reference engine (needs no Libra),
  so you can run something immediately.

## Layout

```
run_demo.py            # entry point: run the model as dynamics, plot the beat
nqdh_demo/             # minimal runtime: provider (W, dW), ehrenfest engine, Libra adapter
src/                   # the model's node/loss definitions (needed to deserialize the graph)
model/
  w_graph.pt           #   frozen trained NQDH graph  (Z, R -> W); generation "v12a":
  nqdh_training_config.json    # 2 active-learning rounds + direct gap loss; the ensemble
                               # beat period (~18 fs) and decoherence (~90 fs) are
                               # reproducible across training seeds
data/ics.npz           # INPUT: 300 thermal initial conditions (geometries + velocities)
expected_output/       # OUTPUT to check against: precomputed 300-trajectory beat (.npz + .png)
environment.yml
```

## Install

```bash
conda env create -f environment.yml
conda activate nqdh-demo
```

This covers the **builtin** path. For the **Libra** path, use your Libra conda
env instead (the one whose activation puts `liblibra_core` on `PYTHONPATH`) and
make sure `torch` + `hippynn` are installed in it too — they load the model.

## Running

```bash
# 1) quick smoke test — one trajectory, ~1 min, no Libra needed
python run_demo.py

# 2) the featured path — one trajectory through Libra (in your Libra env)
python run_demo.py --engine libra

# 3) reproduce the full 300-trajectory beat  (heavy: ~1 h; GPU recommended: --device cuda)
python run_demo.py --ensemble
```

Each run writes `output/beating_<engine>_<N>traj.{npz,png}`. For `N >= 50` the plot
overlays the shipped 300-trajectory reference so you can see your run converging to it.

Troubleshooting: if the model evaluation aborts with a `torch._dynamo` /
inductor compile error (machines without a full C++ toolchain), set
`TORCHDYNAMO_DISABLE=1` in the environment; `HIPPYNN_USE_CUSTOM_KERNELS=False`
similarly avoids numba kernel issues. Both leave the physics unchanged.

## What you should see

- **Energy conservation**: `mean |drift|` ~ **0.4 meV/trajectory** for both engines
  (a rough surface would blow this up — it stays tiny, and Libra matches the
  builtin engine).
- **The beat** (in `expected_output/beating_ensemble.png`, or from `--ensemble`):
  fast S2 -> S1 transfer within ~12 fs, then coherent recurrences at ~17, 36, 53 fs
  that persist out to ~70 fs before decohering — the vibronic funnel of the
  Freixas et al. (2020) NEXMD result, reproduced by a neural surrogate that runs
  ~300 trajectories per hour on one GPU.

A single trajectory (the default) shows the transfer and some oscillation but not
the clean *ensemble* beat — that only emerges after averaging (`--ensemble`), which
is why the full result ships precomputed.

## Notes

- The one heavy dependency is **hippynn** (the model is a hippynn graph). The
  builtin engine is otherwise pure numpy.
- Everything is molecule-agnostic plumbing; the only molecule-specific artifact is
  the trained `model/w_graph.pt` and the initial conditions in `data/ics.npz`.
