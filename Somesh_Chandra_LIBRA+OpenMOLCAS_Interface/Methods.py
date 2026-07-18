# This module implements functions for dealing with the outputs from openmolcas package

import os, sys, math, copy, re, subprocess
import numpy as np
import warnings
import json
from liblibra_core import *
import util.libutil as comn
from types import SimpleNamespace

from libra_py import units
from libra_py import scan
from libra_py import regexlib as rgl
from libra_py import data_conv

import libra_py.packages.cp2k.methods as CP2K_methods
import libra_py.workflows.nbra.step3 as step3

import libra_py.citools.slatdet as sd
import libra_py.citools.interfaces as interfaces
import libra_py.citools.ci as ci
import libra_py.orthogonalizations as ortho



"""
How the Data Flows — Step by Step

1️⃣ Libra reads the trajectory
cp2k.read_trajectory_xyz_file() extracts nuclear coordinates q from each frame of your .xyz file.
This is where Libra's CP2K module acts as a geometry reader.

2️⃣ Libra writes + runs OpenMolcas
make_molcas_input() generates a complete OpenMolcas input file using your molcas_run_params.
run_molcas() spawns pymolcas as a subprocess to run the SA-CASSCF calculation.
OpenMolcas produces:
job.out → CASSCF energies and CI vectors
job.RasOrb → MO coefficients

3️⃣ Libra parses the results
read_molcas_orbital_info() reads both output files and returns:
info → orbital space metadata (inactive, active, virtual boundaries)
MO_curr → MO coefficient matrix (shape nbas × nmo) where nbas is Total number of AO basis functions
data_curr → CI vectors and energies for each state

4️⃣ Libra builds determinant cache
build_det_cache() selects the most important Slater determinants (above ci_coeff_thresh), reducing computational cost.
This cache is reused when computing overlaps between consecutive timesteps.

5️⃣ Libra computes overlaps
ci_overlap_general() computes the overlap matrix between CI wavefunctions using:
Slater determinant overlaps (derived from MO overlaps)
CI coefficient products
Two overlap matrices are produced:
time_overlap_adi → ⟨Ψᵢ(t) | Ψⱼ(t+Δt)⟩ (time-overlap for nonadiabatic dynamics)
overlap_adi → ⟨Ψᵢ(t) | Ψⱼ(t)⟩ (instantaneous overlap)

6️⃣ Results saved to disk
ham_adi — adiabatic Hamiltonian (diagonal = CASSCF energies)
hvib_adi — nonadiabatic coupling vector approximation
st_adi — time-overlap between consecutive timesteps
s_adi — state overlap at the same timestep
"""

def _write_gateway_coord(f, labels, coords, nat):
    """
    Internal helper: write the &GATEWAY Coord block into an open file handle.
    Shared between str-mode and dict-mode branches to avoid code duplication.
    """
    f.write("&GATEWAY\n")
    f.write("Coord\n")
    f.write(f"{nat}\n")
    f.write("Molcas geometry\n")
    for i in range(nat):
        x = coords.get(3 * i + 0, 0)
        y = coords.get(3 * i + 1, 0)
        z = coords.get(3 * i + 2, 0)
        f.write(f"{labels[i]:4s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")


def make_molcas_input(molcas_input_filename, molcas_run_params, labels, coords):
    """
    Write an OpenMolcas input file (.in) from parameters and coordinates.

    Parameters
    ----------
    molcas_input_filename : str
        Path to the output .in file.
    molcas_run_params : dict or str
        If dict  — must contain keys: basis, charge, spin, title,
                   nactel, inactive, ras2, ciroot, nac_states.
        If str   — treated as a raw keyword string appended after the
                   Coord block (legacy compatibility mode).
    labels : list of str
        Atom labels, e.g. ["N", "N", "C", "H", ...].
    coords : dict-like with .get(key, default)
        Atomic coordinates in Angstroms, indexed as:
            coords[3*i + 0] = x
            coords[3*i + 1] = y
            coords[3*i + 2] = z
    """
    nat = len(labels)

    with open(molcas_input_filename, "w") as f:

        # --- Shared: write &GATEWAY + Coord block (no duplication) ---
        _write_gateway_coord(f, labels, coords, nat)

        # ── str mode (legacy): append raw keyword string and exit ──
        if isinstance(molcas_run_params, str):
            f.write(f"{molcas_run_params}\n\n")
            return

        # ── dict mode: structured input generation ──
        
        p = molcas_run_params

        # Finish &GATEWAY
        f.write(f"Basis={p.get('basis', 'cc-pVDZ')}\n")
        f.write("Group=NoSym\n\n")

        # &SEWARD
        f.write("&SEWARD\n")
        

        # &SCF
        f.write("&SCF\n")
        f.write(f"Charge={p.get('charge', 0)}\n")
        f.write(f"Spin={p.get('spin', 1)}\n\n")

        # &RASSCF
        spin = p.get("spin", 1)

        f.write("&RASSCF\n")
        f.write(f"Title={p.get('title', 'Molcas Job')}\n")
        f.write("Symmetry=1\n")
        f.write(f"Spin={spin}\n")
        f.write(f"Nactel={p.get('nactel', '8 0 0')}\n")
        f.write(f"Inactive={p.get('inactive', 17)}\n")
        f.write(f"Ras2={p.get('ras2', 7)}\n")
        f.write(f"Ciroot={p.get('ciroot', '3 3 1')}\n")

        # LSHIFT must be written on separate lines
        lshift = p.get("lshift", 0.5)
        if lshift is not None:
            f.write("LSHIFT\n")
            f.write(f"{lshift}\n")

        # ITER must be written on separate lines
        maxiter = p.get("maxiter", 500)
        iter_micro = p.get("iter_micro", 25)
        f.write("ITER\n")
        f.write(f"{maxiter} {iter_micro}\n")

        # Optional threshold only if explicitly requested
        thre = p.get("thre", None)
        if thre is not None:
            f.write(f"Thre={thre}\n")

        f.write(f"PRWF={p.get('prwf', '1.0d-8')}\n")

        if p.get("prsd", True):
            f.write("PRSD\n")

        f.write("\n")

        # &ALASKA — only written when nac_states is provided
        nac_states = p.get("nac_states", None)
        if nac_states is not None:
            i, j = nac_states
            f.write("&ALASKA\n")
            f.write(f"NAC={i} {j}\n")
            f.write("SHOW\n\n")



def _print_output_tail(path, n=80):
    """Print last n lines of a file, then scan for error keywords for debugging"""
    if not os.path.isfile(path):
        print(f"  [Output file missing: {path}]")
        return

    with open(path, "r") as f:
        lines = f.readlines()

    print(f"── Last {n} lines of {path} (total: {len(lines)} lines) ──")
    for line in lines[-n:]:
        print(line, end="")

    # ✅ One keyword per concept; match case-insensitively
    error_kws = ["convergence", "error", "fatal", "abort", "not found", "stop", "--- stop"]
    hits = [
        (i + 1, line.rstrip())
        for i, line in enumerate(lines)
        if any(kw in line.lower() for kw in error_kws)
    ]

    if hits:
        print(f"\n──  Error keyword scan ({len(hits)} hits) ──")
        for lineno, text in hits:
            print(f"  Line {lineno:5d}: {text}")



def run_molcas(coords, params_):
    """
    Run an OpenMolcas single-point + NAC calculation.
    Returns (output_path, rasorb_path).
    """
    params = dict(params_)

    labels        = params["atom_labels"]
    exe           = params.get("exe", "pymolcas")
    molcas_wd     = params.get("working_directory", "molcas_wd")
    molcas_jobid  = params.get("molcas_jobid", "job_0000")
    input_prefix  = params.get("input_prefix", "input_")

    # ── Build molcas_run_params 
    mrp = params.get("molcas_run_params")
    if mrp is not None:
        molcas_run_params = dict(mrp)
        molcas_run_params.setdefault("title",      f"Molcas_{molcas_jobid}")
        molcas_run_params.setdefault("nac_states",  params.get("nac_states"))
    else:
        molcas_run_params = {
            "basis":      params["basis"],
            "charge":     params["charge"],
            "spin":       params["spin"],
            "title":      f"Molcas_{molcas_jobid}",
            "nactel":     params["nactel"],
            "inactive":   params["inactive"],
            "ras2":       params["ras2"],
            "ciroot":     params["ciroot"],
            "maxiter":    params.get("maxiter", 100),      
            "thre":       params.get("thre", "1.0e-6"),
            "nac_states": params.get("nac_states"),
        }

    # ── File names & paths 
    project_name = f"{input_prefix}{molcas_jobid}"

    molcas_input_filename  = f"{project_name}.in"
    molcas_output_filename = f"{project_name}.out"

    molcas_input_path  = os.path.join(molcas_wd, molcas_input_filename)
    molcas_output_path = os.path.join(molcas_wd, molcas_output_filename)
    rasorb_path        = os.path.join(molcas_wd, f"{project_name}.RasOrb")  

    os.makedirs(molcas_wd, exist_ok=True)

    # ── OpenMolcas environment 
    molcas_env = os.environ.copy()
    molcas_env["MOLCAS_PROJECT"]    = project_name
    molcas_env["MOLCAS_SCRATCH"]    = os.path.abspath(molcas_wd)  
    molcas_env["MOLCAS_SUBMIT_DIR"] = os.path.abspath(molcas_wd)

    # ── Write input & run 
    make_molcas_input(molcas_input_path, molcas_run_params, labels, coords)

    print(f"[run_molcas] Running    : {exe} {molcas_input_filename}")
    print(f"[run_molcas] Working dir: {molcas_wd}")
    print(f"[run_molcas] Output file: {molcas_output_path}")

    with open(molcas_output_path, "w") as fout:
        result = subprocess.run(
            [exe, molcas_input_filename],
            cwd=molcas_wd,
            env=molcas_env,
            check=False,
            stdout=fout,
            stderr=subprocess.STDOUT,
        )

    # ── Handle result 
    if result.returncode != 0:
        print(f"\n[run_molcas] ❌ OpenMolcas FAILED (exit code {result.returncode})")
        print(f"[run_molcas] Input : {molcas_input_path}")
        print(f"[run_molcas] Output: {molcas_output_path}\n")
        _print_output_tail(molcas_output_path, n=80)
        raise RuntimeError(
            f"OpenMolcas failed (exit code {result.returncode}).\n"
            f"  Input : {molcas_input_path}\n"
            f"  Output: {molcas_output_path}\n"
            f"See printed tail above for details."
        )

    # ── Verify RasOrb was created 
    if not os.path.isfile(rasorb_path):                            
        raise FileNotFoundError(
            f"OpenMolcas completed but RasOrb not found at:\n"
            f"  {rasorb_path}\n"
            f"Check MOLCAS_SCRATCH is set correctly."
        )

    print(f"[run_molcas] ✅ Completed successfully")
    print(f"[run_molcas] RasOrb    : {rasorb_path}")
    return molcas_output_path, rasorb_path


def make_ref(nelec):
    """
    Create the reference Slater determinant by filling spin orbitals in order.

    Parameters
    ----------
    nelec : int or np.integer
        Number of electrons (must be non-negative).

    Returns
    -------
    list of int
        Occupied spin orbitals. Positive integers = alpha spin,
        negative integers = beta spin.

    Examples
    --------
    >>> make_ref(4)
    [1, -1, 2, -2]
    >>> make_ref(5)
    [1, -1, 2, -2, 3]
    """
    if not isinstance(nelec, (int, np.integer)) or nelec < 0:
        raise ValueError("nelec must be a non-negative integer")

    n_doubly = nelec // 2

    # ✅ Compact list comprehension — no mutable accumulation
    occ_spin_orbs = [val for orb in range(1, n_doubly + 1) for val in (orb, -orb)]

    if nelec % 2 == 1:
        occ_spin_orbs.append(n_doubly + 1)

    return occ_spin_orbs


def make_alpha_excitation(ref_determinant, config):
    """Performs a single alpha-spin excitation on a reference Slater determinant by 
    moving one electron from a source orbital to a target orbital.
    """
    # ── Validate config 
    if not isinstance(config, (list, tuple)) or len(config) != 2:
        raise ValueError(
            "config must be a list or tuple of length 2: [source_orbital, target_orbital]"
        )

    src, trgt = config

    # ── Validate orbital types 
    if not isinstance(src, int) or not isinstance(trgt, int):
        raise ValueError("source and target orbitals must be integers")

    # ── Validate orbital indices 
    if src <= 0 or trgt <= 0:
        raise ValueError(
            "make_alpha_excitation expects positive orbital indices for alpha excitation"
        )

    if src == trgt:
        raise ValueError("source and target orbitals must be different")

    # ── Perform excitation 
    excited_det = list(ref_determinant)

    if src not in excited_det:
        raise ValueError(
            f"Source alpha orbital {src} not found in reference determinant"
        )

    if trgt in excited_det:
        raise ValueError(
            f"Target alpha orbital {trgt} is already occupied"
        )

    i = excited_det.index(src)
    excited_det[i] = trgt

    return excited_det



def read_casscf_energies(out_file):
    """
    Parse CASSCF / RASSCF total energies from an OpenMolcas output file.

    Parameters
    ----------
    out_file : str or pathlib.Path
        Path to the OpenMolcas output file (e.g., 'output_001.log').

    Returns
    -------
    list[float]
        A list of all total energies (in Hartree) found in the file,
        extracted from lines matching patterns like:
            "RASSCF root  1 Total energy : -262.123456789"
            "CASSCF state  2 Total energy = -262.987654321"
        The list preserves the order in which energies appear in the file.

    Notes
    -----
    - Both 'D' and 'E' exponent markers are handled (e.g., '-1.23D+02' → -123.0).
    - The parser is case‑insensitive for keywords.
    - Duplicate entries (e.g., from intermediate print / final print) are
      **not** filtered – the caller may deduplicate if needed.
    - Only the **first** match per line is recorded; multi‑pattern lines
      (rare in practice) produce a single energy entry.

    Example
    -------
    >>> energies = read_casscf_energies("output_001.log")
    >>> print(f"Found {len(energies)} energy entries")
    >>> print(f"Root energies: {energies}")
    Found 3 energy entries
    Root energies: [-262.123456789, -262.098765432, -262.076543210]

    Typical workflow usage
    ---------------------
    After a successful OpenMolcas run::

        e_list = read_casscf_energies("output_001.log")
        # e_list[0] → ground‑state energy of the SA‑CASSCF
        # e_list[1] → first excited state energy
        # ...

    Limitations
    -----------
    - Does **not** read non‑state‑averaged energies from other modules
      (e.g. &SCF, &CASPT2, &MS‑CASPT2). Those require separate parsers.
    - Spurious matches are possible if the word "Total energy" appears in
      an unrelated comment line.
    """
    energies = []

    patterns = [
        re.compile(
            r"(?:CAS|RAS)SCF.*?root.*?Total energy\s*[:=]\s*([-+0-9.EeDd]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:CAS|RAS)SCF.*?state.*?Total energy\s*[:=]\s*([-+0-9.EeDd]+)",
            re.IGNORECASE,
        ),
    ]

    with open(out_file, "r") as f:
        for line in f:
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    energies.append(
                        float(match.group(1).replace("D", "E").replace("d", "e"))
                    )
                    break

    return energies


def read_ci_vectors(out_file, expected_states=None):
    """
    Parse OpenMolcas CI vectors from RASSCF/CASSCF output.
    Keeps only the LAST occurrence of each root's CI block,
    so intermediate CASSCF iteration printouts are discarded.
    """

    OCC_MAP = {'2': 2, '0': 0, 'u': 1, 'a': 1, 'd': -1, 'b': -1}

    root_pat = re.compile(
        r"printout\s+of\s+CI-coefficients.*for\s+root\s+(\d+)",
        re.IGNORECASE
    )

    # ── Use dict to OVERWRITE intermediate iterations 
    root_confs = {}   # {root_idx: [tuples]}
    root_CIs   = {}   # {root_idx: [floats]}

    current_root   = None
    current_confs  = []
    current_CIs    = []

    CI_END = [
        "Natural orbitals",
        "RASSCF results",
        "CASSCF results",
        "--- Stop Module",
        "++  Convergence",
    ]

    with open(out_file, "r") as f:
        for raw in f:
            stripped = raw.strip()

            # ── New root block found 
            m = root_pat.search(stripped)
            if m:
                # Save whatever we had for the previous root
                if current_root is not None:
                    root_confs[current_root] = current_confs
                    root_CIs[current_root]   = current_CIs

                # Start fresh for new root (OVERWRITES old entry if same root)
                current_root  = int(m.group(1)) - 1   # 0-based
                current_confs = []
                current_CIs   = []
                continue

            if current_root is None:
                continue

            # ── End of CI block 
            if any(p in stripped for p in CI_END):
                root_confs[current_root] = current_confs
                root_CIs[current_root]   = current_CIs
                current_root  = None
                current_confs = []
                current_CIs   = []
                continue

            # ── Skip header / metadata lines 
            if not stripped:
                continue
            if any(h in stripped for h in ["energy=", "conf/sym", "Coeff", "Weight"]):
                continue

            # ── Parse a CI coefficient line 
            parts = stripped.split()
            if len(parts) < 3:
                continue

            try:
                int(parts[0])   # First column must be an integer (conf number)
            except ValueError:
                continue

            try:
                coeff      = float(parts[-2].replace('D', 'E').replace('d', 'e'))
                config_str = "".join(parts[1:-2])
                config     = tuple(OCC_MAP[c] for c in config_str if c in OCC_MAP)

                if len(config) > 0:
                    current_confs.append(config)
                    current_CIs.append(coeff)

            except (ValueError, IndexError, KeyError):
                continue

    # ── EOF flush 
    if current_root is not None:
        root_confs[current_root] = current_confs
        root_CIs[current_root]   = current_CIs

    # ── Reconstruct ordered lists 
    if not root_confs:
        n_found = 0
    else:
        n_found = max(root_confs.keys()) + 1

    if expected_states is not None:
        n_found = max(n_found, expected_states)

    all_confs = [root_confs.get(i, []) for i in range(n_found)]
    all_CIs   = [root_CIs.get(i, [])   for i in range(n_found)]

    # ── Debug report 
    n_total = sum(len(c) for c in all_confs)
    print(f"[DEBUG] Found {len(all_confs)} CI vectors with {n_total} total configurations")

    for i, state_confs in enumerate(all_confs):
        if state_confs:
            print(f"[DEBUG] State {i+1}: first config = {state_confs[0]}, len = {len(state_confs[0])}")
        else:
            print(f"[WARNING]  State {i+1}: 0 configurations — "
                  f"check PRWF threshold and Ciroot in Molcas input")

    return all_confs, all_CIs

def read_rasorb(filepath):
    """
    Reads Molecular Orbital (MO) coefficients from an OpenMolcas .RasOrb / .ScfOrb file.
    This parses the standard INPORB format. Assumes C1 symmetry (Group=NoSym).
    
    Returns
    -------
    mos : np.ndarray
        MO coefficient matrix of shape (nbas, nmo).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"RasOrb file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find where the orbitals start
    orb_start = -1
    for i, line in enumerate(lines):
        if line.startswith("#ORB"):
            orb_start = i
            break

    if orb_start == -1:
        raise ValueError(f"Could not find '#ORB' section in {filepath}")

    coeffs = []
    current_orb = []

    # Parse the coefficients
    for line in lines[orb_start+1:]:
        line = line.strip()
        
        # Stop parsing orbitals if we hit the next block (like #OCC)
        if line.startswith("#"):
            break

        if line.startswith("* ORBITAL"):
            # Save the previous orbital if it exists
            if current_orb:
                coeffs.append(current_orb)
                current_orb = []
        elif line.startswith("*"):
            # Skip other comments
            continue
        elif line:
            # Parse numbers, converting Fortran 'D' exponent to Python 'E'
            parts = line.upper().replace('D', 'E').split()
            current_orb.extend([float(x) for x in parts])

    # Append the last orbital
    if current_orb:
        coeffs.append(current_orb)

    # Convert to numpy array. 
    # 'coeffs' is shape (nmo, nbas). Transpose to get (nbas, nmo)
    mos = np.array(coeffs).T
    
    return mos

def read_molcas_orbital_info(params):
    """
    Parse an OpenMolcas output file and RasOrb file to extract
    orbital space information, MO coefficients, energies, and CI vectors.

    Parameters
    ----------
    params : dict
        Must contain 'filename' or 'output_file', and optionally 'rasorb_file'.

    Returns
    -------
    info : dict
        Orbital space metadata.
    mos : np.ndarray
        MO coefficient matrix, shape (nbas, nmo).
    calc_data : dict
        Contains keys 'energies', 'confs', 'CI'.
    """
    params = dict(params)
    out_file = params.get("filename", params.get("output_file", "job.out"))
    rasorb_file = params.get("rasorb_file", "job.RasOrb")

    if not os.path.isfile(out_file):
        raise FileNotFoundError(
            f"Cannot find the OpenMolcas output file: '{out_file}'\n"
            f"Hint: Current working directory is: {os.getcwd()}"
        )

    if not os.path.isfile(rasorb_file):
        raise FileNotFoundError(
            f"Cannot find the RasOrb file: '{rasorb_file}'\n"
            f"Hint: Current working directory is: {os.getcwd()}"
        )

    # ── Parse metadata from output file 
    n_inactive = 0
    n_active_orb = 0
    n_act_elec = 0
    nbas = 0

    with open(out_file, "r") as f:
        for line in f:
            if "Number of inactive orbitals" in line:
                n_inactive = int(line.split()[-1])
            elif "Number of active orbitals" in line:
                n_active_orb = int(line.split()[-1])
            elif "Number of electrons in active shells" in line:
                n_act_elec = int(line.split()[-1])
            elif "Number of basis functions" in line:
                nbas = int(line.split()[-1])

            if all(x > 0 for x in [n_inactive, n_active_orb, n_act_elec, nbas]):
                break

    # ── Validate 
    if nbas == 0:
        raise RuntimeError(
            f"Could not determine 'Number of basis functions' from '{out_file}'. "
            "Is this a valid OpenMolcas output file?"
        )

    # Derived quantities
    nelec = 2 * n_inactive + n_act_elec

    min_occ = 1
    max_occ = n_inactive
    min_act = n_inactive + 1
    max_act = n_inactive + n_active_orb
    min_vir = max_act + 1
    max_vir = nbas

    # ── Read external data 
    mos = read_rasorb(rasorb_file)
    energies = read_casscf_energies(out_file)
    confs, ci = read_ci_vectors(out_file)

    # ── Build info dictionary
    info = {
        "nao": nbas,
        "nmo": nbas,
        "nelec": nelec,
        "nocc": n_inactive,
        "nact": n_active_orb,
        "nact_elec": n_act_elec,

        "min_occ": min_occ,
        "max_occ": max_occ,
        "min_active": min_act,
        "max_active": max_act,
        "min_vir": min_vir,
        "max_vir": max_vir,

        "actual_orbital_space": list(range(min_act, max_act + 1)),

        "boundaries": {
            "inactive_range": (min_occ, max_occ),
            "active_range": (min_act, max_act),
            "virtual_range": (min_vir, max_vir),
        },
    }

    calc_data = {
        "energies": energies,
        "confs": confs,
        "CI": ci,
    }

    print(json.dumps(info, indent=4))

    return info, mos, calc_data



def read_ao_overlap(job_file, nbas):
    """
    This may not be the perfect method to read the overlap matrix for now it is fine.
    Read AO overlap matrix from OpenMolcas.

    Tries in order:
      1. scf.h5  (primary — AO_OVERLAP_MATRIX lives here)
      2. rasscf.h5 (fallback HDF5)
      3. Text output (if 'Overlap' keyword was used in &SEWARD)
      4. Identity matrix fallback (with warning)
    """

    # ── Option 1: HDF5 — try scf.h5 FIRST ──────────────────────────────────
    h5_candidates = [
        job_file.replace(".out", ".scf.h5"),    # ✅ overlap lives here
        job_file.replace(".out", ".rasscf.h5"), # fallback
        job_file.replace(".out", ".h5"),
    ]

    for h5_path in h5_candidates:
        if not os.path.isfile(h5_path):
            continue
        try:
            import h5py
            with h5py.File(h5_path, "r") as fh:
                print(f"[DEBUG] HDF5 keys in {h5_path}: {list(fh.keys())}")

                for key in ["AO_OVERLAP_MATRIX", "OVERLAP", "overlap", "S"]:
                    if key not in fh:
                        continue

                    raw = np.array(fh[key][:], dtype=np.float64).ravel()

                    expected_packed = nbas * (nbas + 1) // 2

                    if raw.size == expected_packed:
                        # ── lower-triangle packed → full symmetric ──────────
                        S = np.zeros((nbas, nbas), dtype=np.float64)
                        idx = 0
                        for row in range(nbas):
                            for col in range(row + 1):
                                S[row, col] = raw[idx]
                                S[col, row] = raw[idx]
                                idx += 1

                    elif raw.size == nbas * nbas:
                        # ── already full matrix ──────────────────────────────
                        S = raw.reshape(nbas, nbas)

                    else:
                        print(
                            f"[DEBUG] Unexpected array size {raw.size} "
                            f"(expected {expected_packed} packed or "
                            f"{nbas*nbas} full) — skipping key '{key}'"
                        )
                        continue

                    # ── sanity check ─────────────────────────────────────────
                    diag = np.diag(S)
                    if not np.allclose(diag, 1.0, atol=1e-3):
                        print(
                            f"[DEBUG] Key '{key}' diagonal not ~1.0 "
                            f"(max={diag.max():.4f}, min={diag.min():.4f}) — skipping"
                        )
                        continue

                    print(
                        f"[DEBUG] Loaded S_ao from HDF5 key '{key}' "
                        f"in {os.path.basename(h5_path)}, shape {S.shape}"
                    )
                    return S

        except ImportError:
            print("[DEBUG] h5py not available — skipping HDF5 read")
        except Exception as e:
            print(f"[DEBUG] HDF5 read failed for {h5_path}: {e}")

    # ── Option 2: Parse text output ──────────────────────────────────────────
    S = _parse_overlap_from_output(job_file, nbas)
    if S is not None:
        return S

    # ── Option 3: Identity fallback ──────────────────────────────────────────
    warnings.warn(
        "[read_ao_overlap] Could not find AO overlap matrix. "
        "Falling back to identity — MO overlaps will be WRONG for non-orthonormal AO basis.\n"
        "Fix: add 'Overlap' keyword to &SEWARD in your OpenMolcas input.",
        UserWarning,
        stacklevel=2,
    )
    return np.eye(nbas, dtype=np.float64)


def _parse_overlap_from_output(out_file, nbas):
    """
    Parse AO overlap matrix from OpenMolcas text output.
    Requires 'Overlap' keyword in &SEWARD.
    """
    S = np.zeros((nbas, nbas), dtype=np.float64)
    filled = np.zeros((nbas, nbas), dtype=bool)

    in_ovlp   = False
    current_cols = []

    # ✅ Fixed regex: last column may have no trailing space
    col_header_re = re.compile(r"^\s*(\d+)(\s+\d+)*\s*$")

    with open(out_file, "r") as f:
        for line in f:
            stripped = line.strip()

            # ── detect start ──────────────────────────────────────────────
            if "OVERLAP MATRIX" in stripped.upper():
                in_ovlp      = True
                current_cols = []
                S[:]         = 0.0
                filled[:]    = False
                continue

            if not in_ovlp:
                continue

            # ── blank / comment lines ─────────────────────────────────────
            if not stripped or stripped.startswith("*"):
                continue

            # ── detect end ────────────────────────────────────────────────
            if stripped.startswith("---"):
                break
            # A line that starts with a letter but is NOT a data row signals end
            if stripped[0].isalpha():
                break

            # ── column header (only integers on the line) ─────────────────
            tokens = stripped.split()
            if all(t.lstrip("-").isdigit() for t in tokens):
                try:
                    current_cols = [int(t) - 1 for t in tokens]  # 1-based → 0-based
                    continue
                except ValueError:
                    pass

            # ── data row ──────────────────────────────────────────────────
            if not current_cols:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                row_idx = int(parts[0]) - 1  # 1-based → 0-based
            except ValueError:
                continue

            if not (0 <= row_idx < nbas):
                continue

            try:
                vals = [
                    float(x.replace("D", "E").replace("d", "e"))
                    for x in parts[1:]
                ]
            except ValueError:
                continue

            n_vals = min(len(vals), len(current_cols))
            for i in range(n_vals):
                col_idx = current_cols[i]
                if 0 <= col_idx < nbas:
                    S[row_idx, col_idx] = vals[i]
                    filled[row_idx, col_idx] = True

    # ── validate ──────────────────────────────────────────────────────────
    if not filled.any():
        print("[DEBUG] _parse_overlap_from_output: no data found — "
              "add 'Overlap' keyword to &SEWARD")
        return None

    diag = np.diag(S)
    if not np.allclose(diag, 1.0, atol=1e-3):
        print(f"[DEBUG] Diagonal not ~1.0: min={diag.min():.4f}, max={diag.max():.4f}")
        return None

    # ── symmetrize (in case only lower triangle was printed) ─────────────
    S = 0.5 * (S + S.T)

    print(f"[DEBUG] Parsed S_ao from text output, shape {S.shape}")
    return S



# External dependencies assumed available in your environment:
#   Cpp2Py, CMATRIX, units         — from Libra
#   run_molcas                     — your Molcas runner
#   read_molcas_orbital_info       — your output parser
#   read_ao_overlap                — your overlap matrix reader
#   _infer_nstates_from_ciroot     — helper defined earlier



#  MRCI helper functions  (replaces all single-excitation conversion code)


def occ_tuple_to_alpha_beta(occ_tuple, active_space):
    """
    Convert a CASSCF occupation tuple to lists of occupied
    alpha and beta 1-based MO indices (active orbitals only).

    Occupation convention from OpenMolcas:
        2  → doubly occupied  (alpha + beta)
        1  → singly occupied, alpha
       -1  → singly occupied, beta
        0  → unoccupied
    """
    alpha_orbs = []
    beta_orbs  = []

    for i, occ in enumerate(occ_tuple):
        orb_idx = active_space[i]   # 1-based MO index
        if occ == 2:
            alpha_orbs.append(orb_idx)
            beta_orbs.append(orb_idx)
        elif occ == 1:
            alpha_orbs.append(orb_idx)
        elif occ == -1:
            beta_orbs.append(orb_idx)
        # occ == 0 → unoccupied, skip

    return alpha_orbs, beta_orbs


def build_full_orbital_lists(occ_tuple, active_space, inactive_orbs):
    """
    Build full alpha and beta occupied MO index lists including
    inactive (always doubly occupied) orbitals.

    Parameters
    ----------
    occ_tuple     : occupation tuple for active orbitals only
    active_space  : list of 1-based MO indices for active orbitals
    inactive_orbs : list of 1-based MO indices for inactive orbitals

    Returns
    -------
    alpha_orbs, beta_orbs : sorted lists of 1-based MO indices
    """
    alpha_orbs = list(inactive_orbs)
    beta_orbs  = list(inactive_orbs)

    act_alpha, act_beta = occ_tuple_to_alpha_beta(occ_tuple, active_space)

    alpha_orbs += act_alpha
    beta_orbs  += act_beta

    return sorted(alpha_orbs), sorted(beta_orbs)


def slater_det_overlap(alpha_K, beta_K, alpha_L, beta_L, S_mo):
    """
    Compute the overlap between two Slater determinants K and L:

        <Phi_K | Phi_L> = det(S_alpha_KL) * det(S_beta_KL)

    Parameters
    ----------
    alpha_K, beta_K : 1-based occupied MO index lists for bra determinant K
    alpha_L, beta_L : 1-based occupied MO index lists for ket determinant L
    S_mo            : (nmo x nmo) MO overlap matrix, 0-based indexing

    Returns
    -------
    overlap : complex scalar
    """
    # Convert to 0-based indices for numpy
    aK = [a - 1 for a in alpha_K]
    aL = [a - 1 for a in alpha_L]
    bK = [b - 1 for b in beta_K]
    bL = [b - 1 for b in beta_L]

    # Electron count mismatch → orthogonal by particle number
    if len(aK) != len(aL) or len(bK) != len(bL):
        return 0.0 + 0.0j

    det_alpha = (
        1.0 + 0.0j if len(aK) == 0
        else np.linalg.det(S_mo[np.ix_(aK, aL)])
    )
    det_beta = (
        1.0 + 0.0j if len(bK) == 0
        else np.linalg.det(S_mo[np.ix_(bK, bL)])
    )

    return det_alpha * det_beta


def ci_overlap_general(
    data_bra,
    data_ket,
    S_mo,
    active_space,
    inactive_orbs,
    nstates,
    coeff_thresh=1e-6,
    verbose=False,
):
    """
    Compute the full MRCI overlap matrix <Psi_I(bra) | Psi_J(ket)>:

        S_CI[I,J] = sum_{K,L} C_I^K * C_J^L * det(S_alpha_KL) * det(S_beta_KL)

    Parameters
    ----------
    data_bra, data_ket : [energies, confs_list, CI_list]
        confs_list[istate] = list of occupation tuples
        CI_list[istate]    = list of CI coefficients
    S_mo               : (nmo x nmo) MO overlap matrix
                         Pass MO_prev.conj().T @ S_ao @ MO_curr for time-overlap
                         Pass MO_curr.conj().T @ S_ao @ MO_curr for same-time
    active_space       : list of 1-based MO indices (active orbitals)
    inactive_orbs      : list of 1-based MO indices (inactive orbitals)
    nstates            : number of electronic states
    coeff_thresh       : skip determinant pairs where |C_K * C_L| < threshold
    verbose            : print screening statistics

    Returns
    -------
    S_ci : np.ndarray, shape (nstates, nstates), dtype complex128
    """
    S_ci = np.zeros((nstates, nstates), dtype=np.complex128)

    # Pre-build alpha/beta orbital lists for all determinants (avoid redundant work)
    cache_bra = {}
    cache_ket = {}

    for I in range(nstates):
        for K, det_K in enumerate(data_bra[1][I]):
            cache_bra[(I, K)] = build_full_orbital_lists(
                det_K, active_space, inactive_orbs
            )
    for J in range(nstates):
        for L, det_L in enumerate(data_ket[1][J]):
            cache_ket[(J, L)] = build_full_orbital_lists(
                det_L, active_space, inactive_orbs
            )

    total_pairs   = 0
    skipped_pairs = 0

    for I in range(nstates):
        coeffs_I = data_bra[2][I]
        for J in range(nstates):
            coeffs_J  = data_ket[2][J]
            overlap_IJ = 0.0 + 0.0j

            for K in range(len(data_bra[1][I])):
                c_K = coeffs_I[K]
                for L in range(len(data_ket[1][J])):
                    c_L = coeffs_J[L]
                    total_pairs += 1

                    if abs(c_K * c_L) < coeff_thresh:
                        skipped_pairs += 1
                        continue

                    aK, bK = cache_bra[(I, K)]
                    aL, bL = cache_ket[(J, L)]

                    overlap_IJ += c_K * c_L * slater_det_overlap(
                        aK, bK, aL, bL, S_mo
                    )

            S_ci[I, J] = overlap_IJ

    if verbose:
        pct = 100.0 * skipped_pairs / max(total_pairs, 1)
        print(
            f"[ci_overlap_general] Total det pairs : {total_pairs} | "
            f"Skipped (|CK*CL| < {coeff_thresh:.0e}): {skipped_pairs} ({pct:.1f}%)"
        )

    return S_ci


def _infer_nstates_from_ciroot(ciroot):
    """
    Extract the number of states from ciroot parameter.
    
    Args:
        ciroot: Number of roots in various formats
        
    Returns:
        int or None: The number of states to compute, or None if cannot infer
    """
    if ciroot is None:
        return None

    if isinstance(ciroot, int):
        return ciroot

    if isinstance(ciroot, (list, tuple)):
        return int(ciroot[0]) if len(ciroot) > 0 else None

    if isinstance(ciroot, str):
        parts = ciroot.replace(",", " ").split()
        ints = [int(x) for x in parts if x.isdigit()]
        return ints[0] if ints else None

    return None


class tmp:
    pass



#  MAIN FUNCTION


def molcas_compute_adi(q, params, full_id):
    """
    Perform a single-time-step electronic structure evaluation using OpenMolcas
    for trajectory-based nonadiabatic dynamics. Computes molecular orbitals (MOs),
    SA-CASSCF CI states, and their overlaps between consecutive time steps,
    constructing the adiabatic Hamiltonian, vibronic Hamiltonian, and derivative
    couplings.

    The function is designed for trajectory-based nonadiabatic methods, such as:
        - FSSH (Fewest Switches Surface Hopping)
        - Ehrenfest dynamics
        - Mapping-based methods
        - Exact factorization / quantum trajectory approaches

    Workflow
    --------
    1. Extract nuclear coordinates for the trajectory from `q`.
    2. Write an OpenMolcas input file using `make_molcas_input()` with
       SA-CASSCF settings (basis, active space, number of roots, etc.).
    3. Run OpenMolcas (via `pymolcas`) in a trajectory-specific directory.
    4. Parse the output files:
        - `job.out` → CASSCF energies, CI vectors, orbital space metadata
        - `job.RasOrb` → MO coefficient matrix
    5. Build a determinant cache from CI vectors (truncated by `ci_coeff_thresh`).
    6. Compute CI overlaps between the previous and current time step using
       `ci_overlap_general()` (MO-transformed Slater determinant overlaps).
    7. Assemble:
        - Time-overlap matrix between consecutive CI states
        - Adiabatic Hamiltonian (diagonal = CASSCF state energies)
        - Vibronic Hamiltonian (including approximate derivative couplings)
    8. Update trajectory-specific previous-state data in `params`.

    Parameters
    ----------
    q : MATRIX
        Nuclear coordinates for all trajectories.
        Shape: (3 * N_atoms, N_trajectories)
        Units: Bohr
        Column `itraj` corresponds to trajectory `itraj`.

    params : dict
        Dictionary of simulation parameters and trajectory state information.
        Keys used include:

        **Required:**
        atom_labels : list of str
            Atomic symbols, e.g., ["O", "H", "H"].
        molcas_run_params : dict
            Parameters for OpenMolcas SA-CASSCF:
                - basis : str, e.g. "ANO-RCC-VDZP"
                - nactel : int, number of active electrons
                - ras2 : list[int], active orbital indices (1-based)
                - ciroot : list[list[int]], e.g. [[2,2],[2,1]] for state averaging
            See OpenMolcas documentation for all available keywords.

        **Optional / Internal (updated in-place):**
        dt : float, default=41.0
            Nuclear time step in atomic units (1 fs ≈ 41.341 a.u.).
        molcas_exe : str, default="pymolcas"
            Command to invoke OpenMolcas.
        working_directory_prefix : str, default="wd"
            Prefix for trajectory-specific directories.
        ci_coeff_thresh : float, default=0.01
            Threshold for truncating CI expansion in determinant cache.
            Determinants with |C_I| < thresh are discarded for overlap computations.
        is_first_time : dict
            Dictionary keyed by trajectory index (`itraj`) with boolean values.
            True indicates that the current step is the first step of this trajectory.
        act_state : dict
            Dictionary keyed by trajectory index (`itraj`) with integer values
            indicating the active electronic state for this trajectory.
        MO_prev : dict
            Previous MO coefficients per trajectory (updated in-place).
            Shape: (nbas, nbas) per entry.
        data_prev : dict
            Previous CI data (energies, CI vectors) per trajectory (updated in-place).
        coordinates_prev : dict
            Previous nuclear coordinates per trajectory (updated in-place).
        verbose : bool, default=False
            Print detailed progress information during execution.
        timestep : int
            Current time-step index (used for file naming and diagnostics).

    full_id : int or object
        Encoded trajectory identifier (decoded to extract `itraj`).

    Returns
    -------
    obj : tmp (Libra temporary object)
        Object containing adiabatic electronic properties for this trajectory.
        Attributes include:

        ham_adi : CMATRIX (nstates, nstates)
            Adiabatic Hamiltonian matrix.
            Diagonal entries are SA-CASSCF state energies (in Hartree).

        hvib_adi : CMATRIX (nstates, nstates)
            Vibronic Hamiltonian including nonadiabatic coupling:
                Hvib_ij = E_i δ_ij - i d_ij
            where d_ij is the approximate derivative coupling.

        time_overlap_adi : CMATRIX (nstates, nstates)
            Time-overlap matrix S_ij(t, t+dt) = ⟨Ψ_i(t) | Ψ_j(t+dt)⟩.
            Computed via `ci_overlap_general()` using the determinant cache
            and MO-transformed Slater determinant overlaps.

        basis_transform : CMATRIX (nstates, nstates)
            Basis transformation matrix (currently set to identity).

        d1ham_adi : CMATRIXList
            List of derivative Hamiltonians with respect to nuclear coordinates.
            (Currently an empty list placeholder.)

        dc1_adi : CMATRIXList
            List of derivative couplings for each nuclear degree of freedom.
            (Currently an empty list placeholder.)

    Notes
    -----
    - All computations are performed in **trajectory-specific directories**
      to ensure thread safety when running multiple trajectories in parallel.
    - The SA-CASSCF calculation uses a **state-averaged** formalism; energies
      are printed for each root included in the averaging.
    - CI overlaps are computed using the method of Plasser et al. (JCP 2016):
      the Slater determinant overlap is factorised into MO overlap contributions,
      and the CI overlap is assembled as:

        ⟨Ψ_I | Ψ_J⟩ = Σ_{pq} C_I^p * C_J^q * det( MO_prev^T * S_AO * MO_curr )

      where S_AO is the atomic orbital overlap matrix (approximated as identity
      in the MO basis following an orthonormalisation step).

    - Derivative couplings are **approximated** from the anti-symmetric part of
      the time-overlap matrix divided by 2 dt:

          d_ij ≈ [ S_ij(t, t+dt) - S_ji(t, t+dt) ] / (2 * dt)

      This is the **finite-difference overlap-based** approximation (the
      "Hammes-Schiffer–Tully" approach), valid when dt is small.

    - Energies are in **Hartree**, time in **atomic units**, coordinates in
      **Bohr**, and overlaps are **dimensionless**.

    - `is_first_time` and `act_state` are dictionaries keyed by trajectory index,
      enabling simultaneous tracking of multiple trajectories.

    - If MO/CI data is missing for the previous step (first timestep), the
      overlap is set to the identity matrix.

    Example
    -------
    >>> params = {
    ...     "atom_labels": ["O", "H", "H"],
    ...     "molcas_run_params": {
    ...         "basis": "ANO-RCC-VDZP",
    ...         "nactel": 4,
    ...         "ras2": [2, 3, 4, 5, 6, 7],
    ...         "ciroot": [[2, 2], [2, 1]],
    ...     },
    ...     "dt": 41.0,
    ...     "ci_coeff_thresh": 0.001,
    ...     "verbose": True,
    ... }
    >>> obj = molcas_compute_adi(q, params, full_id)
    >>> print(obj.ham_adi)
    >>> print(obj.time_overlap_adi)
    """

    # ── 1. Setup Trajectory Data ──────────────────────────────────────────────
    Id    = Cpp2Py(full_id)
    itraj = Id[-1]
    coords = q.col(itraj)

    params.setdefault("MO_prev",       {})
    params.setdefault("data_prev",     {})
    params.setdefault("is_first_time", {})

    atom_labels          = params["atom_labels"]
    timestep             = params.get("timestep", 0)
    wd_prefix            = params.get("working_directory_prefix", "wd")
    molcas_input_prefix  = params.get("molcas_input_prefix", "input_")
    molcas_output_prefix = params.get("molcas_output_prefix", "output_")
    dt                   = params.get("dt", 0.5 * units.fs2au)
    verbose              = params.get("verbose", False)
    ci_coeff_thresh      = params.get("ci_coeff_thresh", 1e-6)

    # ── 2. Configure Molcas Keywords ──────────────────────────────────────────
    molcas_run_params = copy.deepcopy(
        params.get("molcas_run_params", {
            "basis"     : "ANO-RCC-VDZP",
            "charge"    : 0,
            "spin"      : 1,
            "nactel"    : "6 0 0",
            "inactive"  : 5,
            "ras2"      : 6,
            "ciroot"    : "2 2 1",
            "prwf"      : "1.0d-8",  
        })
    )

    nstates = params.get("nstates", 2)
    is_first_time = params["is_first_time"].get(itraj, True)
    wd = f"{wd_prefix}_itraj{itraj}"
    molcas_jobid = f"_timestep_{timestep}_traj_{itraj}"

    # ── 3. Run Molcas ─────────────────────────────────────────────────────────
    run_params = {
        "atom_labels": atom_labels,
        "exe": params.get("exe", "pymolcas"),
        "molcas_run_params": molcas_run_params,
        "working_directory": wd,
        "molcas_jobid": molcas_jobid,
        "input_prefix": molcas_input_prefix,
        "output_prefix": molcas_output_prefix,
    }

    run_result = run_molcas(coords, run_params)
    out_path, rasorb_file = run_result if isinstance(run_result, tuple) else (run_result, params.get("rasorb_file"))

    # ── 4. Parse Results & Handle Dictionary/Tuple Unpacking ─────────────────
    read_params = {
        "filename": out_path,
        "rasorb_file": rasorb_file,
        "nstates": nstates,
    }
    info, MO_curr, data_curr = read_molcas_orbital_info(read_params)

    # ✅ FIX: Safe normalization of data_curr to prevent string-unpacking bug
    if isinstance(data_curr, dict):
        energies = data_curr.get("energies", data_curr.get("E", []))
        confs    = data_curr.get("confs",    data_curr.get("configs", []))
        CI       = data_curr.get("CI",       data_curr.get("ci", []))
    else:
        # Assume it's a tuple (energies, confs, CI)
        energies, confs, CI = data_curr

    # ── 5. Detailed Data Validation ───────────────────────────────────────────
    if len(energies) < nstates:
        raise ValueError(f"Found only {len(energies)} energies, expected {nstates}.")

    # Find the first state that actually has CI configurations
    sample_conf = None
    sample_state_idx = None
    for idx, state_confs in enumerate(confs):
        if state_confs and len(state_confs) > 0:
            sample_conf = state_confs[0]
            sample_state_idx = idx
            break

    if sample_conf is None:
        raise ValueError(
            "All CI states are empty! Check your 'PRWF' threshold in Molcas "
            "input or ensure the parser is hitting the correct output section."
        )

    # ── 6. Orbital Space Logic ────────────────────────────────────────────────
    MO_curr = np.asarray(MO_curr, dtype=np.complex128)
    nbas    = info.get("nao", info.get("nbas", MO_curr.shape[0]))
    S_ao    = np.asarray(read_ao_overlap(out_path, nbas), dtype=np.complex128)

    active_space = info.get("actual_orbital_space")
    if active_space is None:
        active_space = list(range(info["min_active"], info["max_active"] + 1))
    
    inactive_orbs = list(range(1, info["nocc"] + 1))

    # ✅ FIX: Verify occupation length against the non-empty sample_conf
    occ_tuple_len = len(sample_conf)
    act_space_len = len(active_space)

    if verbose:
        print(f"[molcas_compute_adi] Sample State Index  : {sample_state_idx}")
        print(f"[molcas_compute_adi] Sample Config       : {sample_conf}")
        print(f"[molcas_compute_adi] Occupation length   : {occ_tuple_len}")
        print(f"[molcas_compute_adi] Active space length : {act_space_len}")

    if occ_tuple_len != act_space_len:
        raise ValueError(
            f"Occupation tuple length ({occ_tuple_len}) does not match "
            f"active_space length ({act_space_len}).\n"
            f"Parsed config from State {sample_state_idx+1}: {sample_conf}"
        )

    # ── 7. Overlaps & Hamiltonian Construction ────────────────────────────────
    if is_first_time:
        MO_prev, data_prev = copy.deepcopy(MO_curr), copy.deepcopy((energies, confs, CI))
    else:
        MO_prev   = params["MO_prev"].get(itraj, MO_curr)
        data_prev = params["data_prev"].get(itraj, (energies, confs, CI))

    # Standard Libra/Molcas overlap logic
    st_mo_orb = MO_prev.conj().T @ S_ao @ MO_curr
    s_mo_orb  = MO_curr.conj().T @ S_ao @ MO_curr

    mrci_kwargs = {
        "active_space": active_space,
        "inactive_orbs": inactive_orbs,
        "nstates": nstates,
        "coeff_thresh": ci_coeff_thresh,
        "verbose": verbose,
    }

    # Compute CI overlaps (full Slater determinant expansion)
    st_ci = ci_overlap_general(data_prev, (energies, confs, CI), st_mo_orb, **mrci_kwargs)
    s_ci  = ci_overlap_general((energies, confs, CI), (energies, confs, CI), s_mo_orb,  **mrci_kwargs)

    # ── 8. Build Output Object ────────────────────────────────────────────────
    obj = SimpleNamespace()
    obj.ham_adi          = CMATRIX(nstates, nstates)
    obj.nac_adi          = CMATRIX(nstates, nstates)
    obj.hvib_adi         = CMATRIX(nstates, nstates)
    obj.time_overlap_adi = CMATRIX(nstates, nstates)
    obj.overlap_adi      = CMATRIX(nstates, nstates)
    obj.basis_transform  = CMATRIX(nstates, nstates)

    for i in range(nstates):
        obj.ham_adi.set(i, i, complex(energies[i]))
        obj.hvib_adi.set(i, i, complex(energies[i]))
        obj.basis_transform.set(i, i, 1.0 + 0.0j)
        for j in range(nstates):
            obj.time_overlap_adi.set(i, j, complex(st_ci[i, j]))
            obj.overlap_adi.set(i, j, complex(s_ci[i, j]))

    # Compute NACs from time-overlaps (central difference / Hammes-Schiffer type)
    for i in range(nstates):
        for j in range(i + 1, nstates):
            dij = (obj.time_overlap_adi.get(i, j) - obj.time_overlap_adi.get(j, i)) / (2.0 * dt)
            obj.nac_adi.set(i, j, dij)
            obj.nac_adi.set(j, i, -dij.conjugate())
            obj.hvib_adi.set(i, j, -1.0j * dij)
            obj.hvib_adi.set(j, i, 1.0j * dij.conjugate())

    # ── 9. Persistence ────────────────────────────────────────────────────────
    params["MO_prev"][itraj]       = copy.deepcopy(MO_curr)
    params["data_prev"][itraj]     = copy.deepcopy((energies, confs, CI))
    params["is_first_time"][itraj] = False

    return obj