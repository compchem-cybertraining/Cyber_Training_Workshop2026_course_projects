"""Atomic-unit constants and conversions for the NAMD module.

The dynamics engine runs internally in **Hartree atomic units** (energy: Hartree;
length: bohr; time: au-time; mass: electron mass), exactly matching NEXMD's
``src/naesmd/naesmd_constants.F90`` so that model-driven and NEXMD trajectories
are directly comparable (the FSSH-NEXMD vs FSSH-model benchmark).

NEXMD reference constants (verbatim, naesmd_constants.F90)::

    feVmdqt = 27.2116       Hartree -> eV
    convl   = 0.529177249   bohr    -> Angstrom
    convtf  = 2.41888e-2    au-time -> fs
    convm   = 1822.8885     amu     -> electron mass (au)

I/O is in chemistry-practical units (Angstrom, fs, eV, amu); convert at the
boundary with the helpers below and keep everything au inside the integrators.
"""
from __future__ import annotations

# --- NEXMD constants, verbatim, for bit-comparable benchmarking ----------------
HARTREE_EV = 27.2116          # feVmdqt : Hartree -> eV
BOHR_ANGSTROM = 0.529177249   # convl   : bohr    -> Angstrom
AUTIME_FS = 2.41888e-2        # convtf  : au-time -> fs
AMU_AUMASS = 1822.8885        # convm   : amu     -> electron mass (au)

# --- inverse conversions -------------------------------------------------------
EV_HARTREE = 1.0 / HARTREE_EV
ANGSTROM_BOHR = 1.0 / BOHR_ANGSTROM
FS_AUTIME = 1.0 / AUTIME_FS
AMU_AUMASS_INV = 1.0 / AMU_AUMASS

# Boltzmann constant in au (Hartree / Kelvin) -- for thermostats / Wigner ICs.
KB_AU = 3.166811563e-6

# Velocity: NEXMD writes velocities in Angstrom/picosecond; internal au is bohr/au-time.
ANGPS_AUVEL = ANGSTROM_BOHR / (1000.0 * FS_AUTIME)


def angps_to_auvel(v):
    return v * ANGPS_AUVEL


def auvel_to_angps(v):
    return v / ANGPS_AUVEL


# --- convenience converters (scalars or numpy arrays) --------------------------
def angstrom_to_bohr(x):
    return x * ANGSTROM_BOHR


def bohr_to_angstrom(x):
    return x * BOHR_ANGSTROM


def ev_to_hartree(x):
    return x * EV_HARTREE


def hartree_to_ev(x):
    return x * HARTREE_EV


def fs_to_autime(x):
    return x * FS_AUTIME


def autime_to_fs(x):
    return x * AUTIME_FS


def amu_to_aumass(x):
    return x * AMU_AUMASS


def aumass_to_amu(x):
    return x * AMU_AUMASS_INV
