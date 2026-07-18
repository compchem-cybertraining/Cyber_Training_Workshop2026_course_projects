"""Nuclear bookkeeping helpers for the Ehrenfest integrator (atomic units).

Minimal subset vendored from the research code: kinetic energy and the
translation/rotation removal that keeps the ensemble comparable to NEXMD's
``rescaleveloc``.
"""
from __future__ import annotations

import numpy as np


def kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    """KE = 1/2 sum_i m_i |v_i|^2  (au)."""
    return 0.5 * float(np.sum(masses[:, None] * velocities ** 2))


def remove_com_velocity(masses: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Remove the centre-of-mass (translational) velocity."""
    p_com = (masses[:, None] * V).sum(axis=0) / masses.sum()
    return V - p_com[None, :]


def remove_angular_momentum(masses: np.ndarray, R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Project out the total angular momentum about the centre of mass.

    Subtract the rigid-body field ``omega x r`` that carries L, where
    ``omega = I^{-1} L``. Linear (singular I) geometries fall back to a
    pseudo-inverse so the call is always safe.
    """
    com = (masses[:, None] * R).sum(axis=0) / masses.sum()
    r = R - com[None, :]
    L = (masses[:, None] * np.cross(r, V)).sum(axis=0)
    inertia = np.zeros((3, 3))
    for i in range(R.shape[0]):
        ri = r[i]
        inertia += masses[i] * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    omega = np.linalg.pinv(inertia) @ L
    return V - np.cross(omega[None, :], r)


def remove_com_and_rotation(masses: np.ndarray, R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Strip both translation and rotation from V (NEXMD ``rescaleveloc``)."""
    V = remove_com_velocity(masses, V)
    V = remove_angular_momentum(masses, R, V)
    return V
