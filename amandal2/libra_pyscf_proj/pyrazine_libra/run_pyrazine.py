#!/usr/bin/env python
"""
run_pyrazine.py -- pyrazine S2->S1 NA-MD, one trajectory per process.

Converted from libra_pyscf_pyrazine.ipynb. Two stages, because the geometry
optimization and the Hessian are expensive and must NOT be repeated in every array
task:

    # once, interactively or as a single job (~tens of minutes):
    python run_pyrazine.py prep --ntraj 50 --outdir runs

    # then, as a SLURM array -- each task reads the ensemble and runs ONE trajectory:
    python run_pyrazine.py run --icond $SLURM_ARRAY_TASK_ID --outdir runs

`prep` writes runs/prep.npz holding the optimized geometry, the normal modes, the
Wigner ensemble, and the identified bright state. `run` reads it.

WHY ONE TRAJECTORY PER PROCESS
    generic_recipe runs trajectories SERIALLY. ntraj=8 in one process takes 8x as
    long as ntraj=1 and buys nothing -- trajectories are independent. Use the array.

Requires pyscf_libra.py on the path (same directory is fine).
"""

import argparse
import os
import sys
import time

import numpy as np

from liblibra_core import MATRIX, Random
from libra_py import units
import libra_py.dynamics.tsh.compute as tsh_dynamics

from recipes import ehrenfest_adi_ld, fssh

from pyscf_libra import PySCFSource, compute_model, make_model_params
from pyscf import gto, scf, dft, tdscf

AU2CM = 219474.63
KB_AU = 3.166811563e-6            # Hartree / K


# ---------------------------------------------------------------------------
#  Geometry
# ---------------------------------------------------------------------------

SYMBOLS = ["N", "C", "C", "N", "C", "C", "H", "H", "H", "H"]
PYRAZINE_ANG = np.array([
    [0.000000,  1.410994, 0.0],   # N
    [1.132517,  0.698500, 0.0],   # C
    [1.132517, -0.698500, 0.0],   # C
    [0.000000, -1.410994, 0.0],   # N
    [-1.132517, -0.698500, 0.0],  # C
    [-1.132517,  0.698500, 0.0],  # C
    [2.070801,  1.239345, 0.0],   # H
    [2.070801, -1.239345, 0.0],   # H
    [-2.070801, -1.239345, 0.0],  # H
    [-2.070801,  1.239345, 0.0],  # H
])


# ---------------------------------------------------------------------------
#  Wigner sampling
# ---------------------------------------------------------------------------

def normal_modes(source, R0, ntrans_rot=6):
    """
    Mass-weighted normal modes at R0.

    Returns (omega, L, masses, info). omega in a.u.; L columns are mass-weighted
    orthonormal eigenvectors. Imaginary modes are REPORTED, not silently dropped --
    they mean R0 is not a minimum and the Wigner sample would be fiction.
    """
    mol = source._mol(R0)
    mf = dft.RKS(mol, xc=source.xc)
    mf.grids.level = source.grid_level
    mf.conv_tol = source.conv_tol
    mf.kernel()

    n = source.natoms
    # PySCF: hess[i,j,x,y] = d2E/dR_ix dR_jy -> (3N,3N) needs index order (i,x,j,y)
    hess = mf.Hessian().kernel()
    H = hess.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)

    masses = np.asarray(source.masses_au(R0), dtype=float)
    D = H / np.sqrt(np.outer(masses, masses))
    evals, evecs = np.linalg.eigh(D)

    order = np.argsort(np.abs(evals))
    drop, keep = order[:ntrans_rot], np.sort(order[ntrans_rot:])

    w2 = evals[keep]
    imag = np.where(w2 < 0)[0]
    omega = np.sqrt(np.abs(w2))
    info = {"imaginary": imag,
            "freq_cm": omega * AU2CM * np.sign(w2),
            "dropped_cm": np.sqrt(np.abs(evals[drop])) * AU2CM,
            "n_modes": len(keep)}
    return omega, evecs[:, keep], masses, info


def wigner_sample(omega, L, masses, R0, ntraj, temperature=0.0, seed=None):
    """
    Harmonic Wigner distribution, mass-weighted normal coordinates, hbar = 1.

        <Q_k^2> = coth(w_k / 2kT) / (2 w_k)      -> 1/(2 w_k)  at T=0
        <P_k^2> = w_k coth(w_k / 2kT) / 2        -> w_k / 2    at T=0

    Back-transform:  x = R0 + (L Q)/sqrt(m),   p = sqrt(m)(L P)

    COM momentum comes out at zero by construction: the translational eigenvectors
    are proportional to sqrt(m_i) in mass-weighted coordinates, and L's columns are
    orthogonal to them.
    """
    rng = np.random.default_rng(seed)
    nm = len(omega)

    if temperature > 0:
        cth = 1.0 / np.tanh(omega / (2.0 * KB_AU * temperature))
    else:
        cth = np.ones(nm)

    sig_Q = np.sqrt(cth / (2.0 * omega))
    sig_P = np.sqrt(omega * cth / 2.0)

    Q = rng.normal(0.0, 1.0, (ntraj, nm)) * sig_Q[None, :]
    P = rng.normal(0.0, 1.0, (ntraj, nm)) * sig_P[None, :]

    q = R0.flatten()[None, :] + (Q @ L.T) / np.sqrt(masses)[None, :]
    p = (P @ L.T) * np.sqrt(masses)[None, :]
    return q, p, sig_Q


# ---------------------------------------------------------------------------
#  Stage: prep
# ---------------------------------------------------------------------------

def stage_prep(args):
    os.makedirs(args.outdir, exist_ok=True)
    R0 = PYRAZINE_ANG * units.Angst          # -> Bohr

    source = PySCFSource(SYMBOLS, basis=args.basis, xc=args.xc, nexc=args.nexc,
                         verbose_timing=False)

    # ---- optimize -------------------------------------------------------
    if not args.no_opt:
        print("[prep] optimizing S0 geometry ...", flush=True)
        t0 = time.time()
        mol = source._mol(R0)
        mf = dft.RKS(mol, xc=args.xc)
        mf.grids.level = source.grid_level
        mf.conv_tol = 1e-8
        mf.kernel()
        try:
            from pyscf.geomopt.geometric_solver import optimize
            mol_eq = optimize(mf, maxsteps=100)
            R0 = np.asarray(mol_eq.atom_coords(), dtype=float)
            print(f"[prep]   done in {time.time()-t0:.0f} s", flush=True)
        except ImportError:
            print("[prep]   geometric not installed -- using the guess geometry.")
            print("[prep]   pip install --no-deps --target=$LIBRA_EXTRAS geometric")

    # ---- normal modes ---------------------------------------------------
    print("[prep] Hessian + normal modes ...", flush=True)
    t0 = time.time()
    omega, L, masses, info = normal_modes(source, R0)
    print(f"[prep]   {info['n_modes']} modes in {time.time()-t0:.0f} s")
    print(f"[prep]   range: {info['freq_cm'].min():.0f} - {info['freq_cm'].max():.0f} cm^-1")
    print(f"[prep]   discarded trans/rot: {np.round(info['dropped_cm'],1)}")

    if len(info["imaginary"]):
        msg = (f"[prep] {len(info['imaginary'])} IMAGINARY modes "
               f"({np.round(info['freq_cm'][info['imaginary']],1)} cm^-1). R0 is not a "
               f"minimum -- Wigner sampling would be fiction.")
        if not args.allow_imaginary:
            raise RuntimeError(msg + " Optimize first, or pass --allow-imaginary.")
        print(msg + " CONTINUING because --allow-imaginary was given.")
        good = np.ones(len(omega), bool); good[info["imaginary"]] = False
        omega, L = omega[good], L[:, good]

    # ---- Wigner ensemble ------------------------------------------------
    q, p, sig_Q = wigner_sample(omega, L, masses, R0, args.ntraj,
                                temperature=args.temperature, seed=args.seed)

    # Validate rather than assert: recover <Q_k^2> from the sample.
    Qs = ((q - R0.flatten()[None, :]) * np.sqrt(masses)[None, :]) @ L
    rel = np.abs(Qs.var(axis=0) - sig_Q**2) / sig_Q**2
    noise = np.sqrt(2.0 / args.ntraj)
    print(f"[prep] Wigner check: max rel. dev of <Q_k^2> = {rel.max():.1%} "
          f"(sampling noise ~{noise:.1%})")
    E_vib = 0.5 * ((p.reshape(args.ntraj, -1, 3) ** 2).sum(-1)
                   / np.array(masses).reshape(-1, 3)[:, 0]).sum(-1).mean()
    print(f"[prep]   ZPE (sum w/2) = {(omega/2).sum()*units.au2ev:.3f} eV "
          f"-- if <E> is ~2x this, the sigma factors are wrong")
    com = np.linalg.norm(p.reshape(args.ntraj, -1, 3).sum(1), axis=1)
    print(f"[prep]   max |COM momentum| = {com.max():.2e} (should be ~0)")

    # ---- bright state ---------------------------------------------------
    print("[prep] identifying the bright state ...", flush=True)
    mol = source._mol(R0)
    mf = dft.RKS(mol, xc=args.xc); mf.grids.level = source.grid_level
    mf.kernel()
    td = tdscf.TDA(mf); td.nstates = args.nexc; td.kernel()
    osc = td.oscillator_strength()
    print("[prep]   root   E(eV)     f")
    for n in range(args.nexc):
        flag = "  <-- BRIGHT" if n == int(np.argmax(osc)) else ""
        print(f"[prep]    S{n+1}   {td.e[n]*units.au2ev:6.3f}  {osc[n]:7.4f}{flag}")
    bright = int(np.argmax(osc)) + 1
    print(f"[prep]   reference: pyrazine S2(pi-pi*) ~4.8 eV bright, "
          f"S1(n-pi*) ~3.8 eV dark")
    if osc[bright-1] < 0.01:
        print("[prep]   WARNING: no root has appreciable oscillator strength -- "
              "the bright state may be above your nexc window")

    near = np.where(np.abs(np.diff(td.e)) * units.au2ev < 0.05)[0]
    if len(near):
        print(f"[prep]   WARNING: near-degenerate roots "
              f"{[(int(i+1), int(i+2)) for i in near]} -- nexc may split a multiplet")

    path = os.path.join(args.outdir, "prep.npz")
    np.savez(path, R0=R0, q=q, p=p, masses=masses, omega=omega,
             freq_cm=info["freq_cm"], bright=bright, osc=osc,
             e_exc=td.e, symbols=np.array(SYMBOLS), basis=args.basis, xc=args.xc,
             nexc=args.nexc, ntraj=args.ntraj)
    print(f"\n[prep] wrote {path}")
    print(f"[prep] now:  sbatch --array=0-{args.ntraj-1} submit_pyrazine.sh")


# ---------------------------------------------------------------------------
#  Stage: run
# ---------------------------------------------------------------------------

def stage_run(args):
    path = os.path.join(args.outdir, "prep.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run the prep stage first")
    d = np.load(path, allow_pickle=True)

    ntraj_avail = int(d["ntraj"])
    if not (0 <= args.icond < ntraj_avail):
        raise SystemExit(f"icond {args.icond} outside 0..{ntraj_avail-1}")

    symbols = [str(s) for s in d["symbols"]]
    R0, q, p, masses = d["R0"], d["q"], d["p"], d["masses"]
    nexc = int(d["nexc"])
    nstates = nexc + 1
    bright = int(d["bright"]) if args.istate < 0 else args.istate
    natoms, ndof = len(symbols), 3 * len(symbols)

    source = PySCFSource(symbols, basis=str(d["basis"]), xc=str(d["xc"]), nexc=nexc,
                         verbose_timing=True, timing_every=args.timing_every)
    model_params = make_model_params(source)

    print(f"[run] icond {args.icond}/{ntraj_avail-1} | starting in S{bright} "
          f"(f = {d['osc'][bright-1]:.4f}) | {nstates} states", flush=True)

    # ---- dyn_general ----------------------------------------------------
    dyn = {"nsteps": args.nsteps, "ntraj": 1, "nstates": nstates,
           "dt": args.dt, "num_electronic_substeps": args.substeps,
           "isNBRA": 0, "is_nbra": 0, "progress_frequency": 0.05,
           "which_adi_states": range(nstates), "which_dia_states": range(nstates),
           "mem_output_level": 3,
           "properties_to_save": ["timestep", "time", "q", "p", "f", "Cadi",
                                  "Epot_ave", "Ekin_ave", "Etot_ave",
                                  "se_pop_adi", "sh_pop_adi"],
           "prefix": "x", "prefix2": "x"}

    if args.method == 2:
        ehrenfest_adi_ld.load(dyn)
    elif args.method == 4:
        fssh.load(dyn)
    else:
        raise SystemExit("--method must be 2 (ehrenfest_adi_ld) or 4 (fssh)")

    # ---- post-load overrides: the recipes target 2-state model Hamiltonians ----
    dyn.update({"ham_update_method": 2,      # adiabatic properties come FROM the model
                "ham_transform_method": 0,   # no diabatic Ham to diagonalize
                "time_overlap_method": 0,    # we supply time_overlap_adi
                "nac_update_method": 2,      # NACs from time-overlaps, not dc1_adi
                "nac_algo": 1,               # NPI, not HST
                "hvib_update_method": 1})
    dyn.update({"decoherence_rates": MATRIX(nstates, nstates),   # recipes hard-code 2x2
                "ave_gaps": MATRIX(nstates, nstates)})
    # NOT overridden: state_tracking_algo. Both LD recipes set -1 on purpose.

    suff = "na"
    if args.method == 4:
        dyn.update({"hop_acceptance_algo": 21, "momenta_rescaling_algo": 200})
        suff = "g-"
    if dyn.get("hop_acceptance_algo") == 20:
        raise SystemExit("hop_acceptance_algo:20 needs NAC vectors, zeros here")

    pref = os.path.join(args.outdir,
                        f"pyrazine-m{args.method}-{suff}-S{bright}-icond{args.icond}")
    dyn.update({"prefix": pref, "prefix2": pref})

    # ---- initial conditions ---------------------------------------------
    nucl_params = {"ndof": ndof,
                   "q": [float(x) for x in q[args.icond]],
                   "p": [float(x) for x in p[args.icond]],
                   "mass": [float(m) for m in masses],
                   "force_constant": [0.01] * ndof,     # unused for init_type:0
                   "init_type": 0}                      # sampling already happened

    istates = [0.0] * nstates
    istates[bright] = 1.0
    elec_params = {"verbosity": 0, "init_dm_type": 0,
                   "ndia": nstates, "nadi": nstates,
                   "rep": 1, "init_type": 3, "istates": istates}

    total_fs = args.nsteps * args.dt * units.au2fs
    print(f"[run] {args.nsteps} steps x {args.dt} a.u. = {total_fs:.1f} fs", flush=True)
    print(f"[run] output -> {pref}", flush=True)

    t0 = time.time()
    rnd = Random()
    tsh_dynamics.generic_recipe(dyn, compute_model, model_params,
                                elec_params, nucl_params, rnd)
    wall = time.time() - t0
    print(f"\n[run] done: {source.ncalls:,} calls in {wall/60:.1f} min "
          f"({wall/max(source.ncalls,1):.1f} s/call)", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("prep", help="optimize + Hessian + Wigner ensemble (run ONCE)")
    p.add_argument("--ntraj", type=int, default=50, help="ensemble size")
    p.add_argument("--basis", default="6-31G*")
    p.add_argument("--xc", default="pbe0")
    p.add_argument("--nexc", type=int, default=5,
                   help="keep large enough not to split a degenerate multiplet")
    p.add_argument("--temperature", type=float, default=0.0, help="K; 0 = pure ZPE")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-opt", action="store_true")
    p.add_argument("--allow-imaginary", action="store_true",
                   help="proceed despite imaginary modes (you should not)")
    p.add_argument("--outdir", default="runs")

    r = sub.add_parser("run", help="one trajectory from the prepped ensemble")
    r.add_argument("--icond", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    r.add_argument("--method", type=int, default=2, help="2=ehrenfest_adi_ld, 4=fssh")
    r.add_argument("--nsteps", type=int, default=250)
    r.add_argument("--dt", type=float, default=10.0, help="a.u.; 10 = 0.242 fs")
    r.add_argument("--substeps", type=int, default=20)
    r.add_argument("--istate", type=int, default=-1,
                   help="override the auto-detected bright state")
    r.add_argument("--timing-every", type=int, default=10)
    r.add_argument("--outdir", default="runs")

    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep(args)
    else:
        stage_run(args)


if __name__ == "__main__":
    main()
