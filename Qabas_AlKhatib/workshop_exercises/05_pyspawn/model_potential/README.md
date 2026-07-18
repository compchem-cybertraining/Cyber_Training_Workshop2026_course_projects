# Model Conical Intersection Potential

A minimal, exactly-solvable two-state model for testing and teaching Ab Initio Multiple
Spawning (AIMS) dynamics through a conical intersection, without any electronic-structure
calculation. Everything is analytic, so a full spawning simulation runs in seconds and the
result can be checked by hand.

This directory contains the PySpawn a `start.py` to launch the run, and an analysis script 
to process the output.

---

## 1. Why a model potential?

The ethylene AIMS tutorial calls OpenMolcas at every time step to get energies, forces, and
wavefunctions. That is realistic but slow, and it hides the machinery of AIMS behind a quantum
chemistry black box. This model replaces the electronic-structure call with two lines of
algebra. The dynamics engine (classical propagation, spawning, quantum amplitudes) is identical
to a real run, so you can watch **how AIMS behaves near a conical intersection** with instant
turnaround and full analytic control.

It is the recommended first thing to run when learning PySpawn, and a standard sanity check when
developing the code.

---

## 2. The potential, defined

The nuclear coordinate is two-dimensional, `R = (x, y)`. Define polar-like coordinates

```
    r     = sqrt(x^2 + y^2)
    theta = atan2(y, x) / 2
```

The two adiabatic electronic energies are

```
    E0(r) = (r - 1)^2 - 1        (lower state)
    E1(r) = (r + 1)^2 - 1        (upper state)
```

and the adiabatic electronic wavefunctions (the eigenvectors, written in a fixed 2-dimensional
diabatic basis) are

```
    psi_0 = [  sin(theta),  cos(theta) ]
    psi_1 = [  cos(theta), -sin(theta) ]
```

That is the entire model. Both energies depend only on `r`, the couplings and geometric
structure enter only through `theta`.

---

## 3. What the energies tell you

### The gap is a perfect cone

Subtract the two surfaces:

```
    Delta E(r) = E1 - E0 = (r+1)^2 - (r-1)^2 = 4r
```

The gap grows **linearly** with the distance `r` from the origin and vanishes **only at `r = 0`**.
A gap that closes linearly in the two coordinates `(x, y)` is exactly the definition of a
**conical intersection**: near the origin the two surfaces form a double cone (an X-shaped
crossing in any slice through the origin, a pair of nested cones in the full 2-dimensional space).

### Shape of each surface

- **Lower state `E0 = (r-1)^2 - 1`.** Minimum at `r = 1` with `E0 = -1`. Because it depends only
  on `r`, the set of minima is a **circle** of radius 1 (a "moat" or "sombrero" brim) surrounding
  a central bump. The bump maximum at `r = 0` is `E0 = 0`, i.e. the cone tip sits `+1` above the
  moat floor.
- **Upper state `E1 = (r+1)^2 - 1`.** Monotonically increasing in `r`, with its minimum at the
  intersection point `r = 0`, `E1 = 0`. A wavepacket placed on the upper state therefore slides
  **inward** toward `r = 0`, straight at the conical intersection.

This is the classic "sombrero + funnel" arrangement: the upper surface funnels population down to
the tip, where the vanishing gap lets it cross onto the lower surface and spill into the
surrounding moat.

### Geometric (Berry) phase, the topological fingerprint

Follow the lower wavefunction `psi_0 = [sin(theta), cos(theta)]` as the **physical** angle
`phi = atan2(y, x)` sweeps once around the origin, `phi: 0 -> 2*pi`. Since `theta = phi/2`, the
electronic vector only rotates by `pi`:

```
    phi = 0     ->  theta = 0    ->  psi_0 = [ 0,  1]
    phi = 2*pi  ->  theta = pi   ->  psi_0 = [ 0, -1]
```

The wavefunction comes back with a **flipped sign** after a full loop. This sign change (the
**geometric or Berry phase** of pi) is the unmistakable signature of a genuine conical
intersection encircled by the path. The factor of `1/2` in `theta` is precisely what encodes it.

---

## 4. The forces

The classical force on the active state is `F = -dE/dR`. Using `dr/dx = x/r` and `dr/dy = y/r`,

```
    lower state:  F0 = -2 (r - 1) * (x/r, y/r)
    upper state:  F1 = -2 (r + 1) * (x/r, y/r)
```

Both forces point **radially** (along `(x/r, y/r)`). On the upper state the prefactor
`-2(r+1)` is always negative, so `F1` points inward, pushing the trajectory toward `r = 0`. This
is what carries an upper-state wavepacket into the intersection, exactly matching the surface
shape above. These are the `f[0,:]` and `f[1,:]` arrays built in `compute_elec_struct`.

---

## 5. Time-derivative coupling and spawning

Near the intersection the two electronic states change character rapidly with geometry, so the
**time-derivative coupling** (TDC), the quantity `< psi_i | d/dt | psi_j >`, becomes large. AIMS
monitors this coupling and, when it exceeds a threshold, **spawns** a new basis function on the
other electronic state to capture the population transfer.

In the code the wavefunction is first **phase-matched** to the previous step (the `W` overlap
matrix and the sign-flip checks), which is essential precisely because of the geometric phase
above, then `compute_tdc(W)` returns the coupling that drives spawning. The spawning threshold
itself is set in `start.py` via `spawnthresh`.

---

## 6. Running it

`start.py` launches a single initial trajectory on the **upper** state (`istate = 1`) and lets it
propagate into the intersection, where it will spawn.

```bash
source /projects/academic/cyberwksp21/SOFTWARE_2026/miniforge3/etc/profile.d/conda.sh
conda activate $HOME/pyspawn

python start.py
```

Key parameters in `start.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `numdims` | 2 | two nuclear coordinates `(x, y)` |
| `numstates` | 2 | two electronic states |
| `istate` | 1 | start on the **upper** state |
| `positions` | `[0.45, 0.1]` | start off-center, near the cone |
| `momenta` | `[-5.0, 0.0]` | pushed toward the intersection |
| `masses` | `[1822, 1822]` | ~1 amu per coordinate (atomic units) |
| `widths` | `[6.0, 6.0]` | frozen-Gaussian widths of the basis functions |
| `timestep` | 0.1 | classical time step (a.u.) |
| `tfinal` | 200.0 | total propagation time (a.u.) |
| `spawnthresh` | `(0.5*pi)/ts/20` | TDC threshold that triggers spawning |
| `qm_energy_shift` | -5.18 | constant shift applied in the quantum propagation only |

The propagators match the ethylene run: velocity-Verlet classical integrator (`vv`), full-diagonalization
adaptive Runge-Kutta quantum integrator (`fulldiag`), and the adiabatic NPI Hamiltonian
(`adiabatic`).

> Note: `qm_energy_shift` only removes a constant offset in the quantum phase propagation and does
> not change the physics of the surfaces defined above. The surfaces themselves are `E0` and `E1`
> as given in Section 2.

---

## 7. Analysis

Run the analysis script (as in the ethylene example) from a fresh sub-directory so its outputs do
not clutter the run folder:

```bash
mkdir analysis && cd analysis
python ../analysis.py
```

Things worth plotting and checking for this model:

- **Total electronic populations vs time.** Population should start at 1.0 on the upper state and
  transfer to the lower state as the trajectory passes through `r = 0`. This is the headline AIMS
  result.
- **Trajectory in the `(x, y)` plane.** Watch the upper-state trajectory fall inward to the cone,
  then the spawned lower-state child scatter outward toward the `r = 1` moat.
- **The gap `4r` along the trajectory.** It should collapse toward zero as the trajectory
  approaches the intersection, which is where the spawning happens.
- **Energy conservation.** Total (classical + potential) energy of each basis function should be
  conserved away from spawning events, a standard check that the integrator is behaving.

Because every quantity is analytic, you can compare the simulation directly against the formulas
in Sections 2 to 4, which makes this the ideal case for verifying that your PySpawn installation
and your understanding are both correct.
