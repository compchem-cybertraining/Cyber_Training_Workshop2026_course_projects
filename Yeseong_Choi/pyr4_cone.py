import math
import numpy as np

W10A, W6A, W1, W9A = 0.09357, 0.0740, 0.1273, 0.1568
DELTA = 0.46165
LAM = 0.1825
K6A1, K6A2 = -0.0964, 0.1194
K11, K12 = 0.0470, 0.2012
K9A1, K9A2 = 0.1594, 0.0484

OMEGA = np.array([W10A, W6A, W1, W9A])
KAP1 = np.array([0.0, K6A1, K11, K9A1])
KAP2 = np.array([0.0, K6A2, K12, K9A2])
LVEC = np.array([LAM, 0.0, 0.0, 0.0])


def _diabatic(q):
    har = 0.5 * np.sum(OMEGA * q * q)
    dhar = OMEGA * q
    A = har - DELTA + np.dot(KAP1, q)
    B = har + DELTA + np.dot(KAP2, q)
    W = LAM * q[0]
    dA = dhar + KAP1
    dB = dhar + KAP2
    dW = LVEC
    return A, B, W, dA, dB, dW


def compute_elec_struct(self, zbackprop):
    if not zbackprop:
        cbackprop = ""
    else:
        cbackprop = "backprop_"

    exec ("self.set_" + cbackprop + "prev_wf(self.get_" + cbackprop + "wf())")
    exec ("q = self.get_" + cbackprop + "positions()")

    A, B, W, dA, dB, dW = _diabatic(np.asarray(q))
    half = 0.5 * (A + B)
    dd = 0.5 * (B - A)
    r = math.sqrt(dd * dd + W * W)
    if r < 1.0e-12:
        r = 1.0e-12

    e = np.zeros(self.numstates)
    e[0] = half - r
    e[1] = half + r
    exec ("self.set_" + cbackprop + "energies(e)")

    dr = (dd * 0.5 * (dB - dA) + W * dW) / r
    f = np.zeros((self.numstates, self.numdims))
    f[0, :] = -(0.5 * (dA + dB) - dr)
    f[1, :] = -(0.5 * (dA + dB) + dr)
    exec ("self.set_" + cbackprop + "forces(f)")

    ev, U = np.linalg.eigh(np.array([[A, W], [W, B]]))
    wf = np.zeros((self.numstates, self.length_wf))
    wf[0, :] = U[:, 0]
    wf[1, :] = U[:, 1]
    exec ("prev_wf = self.get_" + cbackprop + "prev_wf()")
    Wov = np.matmul(prev_wf, wf.T)
    if Wov[0, 0] < 0.0:
        wf[0, :] = -1.0 * wf[0, :]
        Wov[:, 0] = -1.0 * Wov[:, 0]
    if Wov[1, 1] < 0.0:
        wf[1, :] = -1.0 * wf[1, :]
        Wov[:, 1] = -1.0 * Wov[:, 1]
    tmp = self.compute_tdc(Wov)
    tdc = np.zeros(self.numstates)
    if self.istate == 1:
        jstate = 0
    else:
        jstate = 1
    tdc[jstate] = tmp
    exec ("self.set_" + cbackprop + "timederivcoups(tdc)")

    exec ("self.set_" + cbackprop + "wf(wf)")


def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["wf0"] = self.numstates
    self.h5_datasets["wf1"] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates


def potential_specific_traj_copy(self, from_traj):
    return


def get_wf0(self):
    return self.wf[0, :].copy()


def get_wf1(self):
    return self.wf[1, :].copy()


def get_backprop_wf0(self):
    return self.backprop_wf[0, :].copy()


def get_backprop_wf1(self):
    return self.backprop_wf[1, :].copy()
