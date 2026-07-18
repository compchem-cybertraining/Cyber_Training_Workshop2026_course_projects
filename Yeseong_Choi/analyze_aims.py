import os, glob, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.environ.get("P4DATA", HERE)
import pyr4_cone as M

HB = 0.6582119569
TF = 182.0
ALPHA = 0.5
NDIM = 4


def eigvecs(q):
    A, B, W, _, _, _ = M._diabatic(np.asarray(q))
    ev, U = np.linalg.eigh(np.array([[A, W], [W, B]]))
    return U


def gauss_overlap(qi, pi, qj, pj):
    dq = qi - qj; sp = pi + pj; dp = pi - pj
    ln = -0.5 * ALPHA * dq * dq - dp * dp / (8.0 * ALPHA) + 0.5j * sp * (qj - qi)
    return np.exp(np.sum(ln))


def coh_P2(path):
    h = h5py.File(path, "r"); g = h["sim"]
    qt = np.array(g["quantum_time"]).ravel()
    amp = np.array(g["qm_amplitudes"]); Sall = np.array(g["S"])
    ntraj = np.array(g["num_traj_qm"]).ravel().astype(int)
    ist = np.array(g["istates_this_step"]); lab = np.array(g["labels_this_step"])
    traj = {}
    for k in h.keys():
        if k.startswith("traj_"):
            traj[k[len("traj_"):]] = (np.array(h[k]["time"]).ravel(),
                                      np.array(h[k]["positions"]), np.array(h[k]["momenta"]))
    h.close()

    def _dec(x):
        return x if isinstance(x, str) else (x.decode() if hasattr(x, "decode") else str(x))
    P = np.full(len(qt), np.nan)
    for k in range(len(qt)):
        n = int(ntraj[k])
        if n < 1:
            continue
        c = amp[k, :n]
        states = [int(x) for x in _dec(ist[k]).split(",")[:n]]
        labels = _dec(lab[k]).split(",")[:n]
        q = np.zeros((n, NDIM)); p = np.zeros((n, NDIM)); u = np.zeros((n, 2))
        for i in range(n):
            L = labels[i]
            if L in traj:
                tt, pos, mom = traj[L]
                q[i] = [np.interp(qt[k], tt, pos[:, d]) for d in range(NDIM)]
                p[i] = [np.interp(qt[k], tt, mom[:, d]) for d in range(NDIM)]
            u[i] = eigvecs(q[i])[:, states[i]]
        Sn = np.ones((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Sn[i, j] = gauss_overlap(q[i], p[i], q[j], p[j])
        num = den = 0.0j
        for i in range(n):
            for j in range(n):
                w = np.conj(c[i]) * c[j] * Sn[i, j]
                num += w * u[i, 1] * u[j, 1]
                den += w * (u[i, 0] * u[j, 0] + u[i, 1] * u[j, 1])
        P[k] = (num / den).real if abs(den) > 1e-14 else np.nan
    return qt, P, int(ntraj[-1])


def main():
    tgrid = np.linspace(0.0, TF, 241)
    dirs = sorted(glob.glob(os.path.join(DATA, "ic??")))
    stack = []; ntbf = []; used = []
    for d in dirs:
        f = os.path.join(d, "sim.hdf5")
        if not os.path.exists(f):
            continue
        try:
            t, P, ne = coh_P2(f)
        except Exception as e:
            sys.stderr.write("%s: %s\n" % (d, e)); continue
        m = np.isfinite(P)
        if m.sum() < 2 or t[m][-1] < 0.9 * TF:
            continue
        stack.append(np.interp(tgrid, t[m], P[m], right=np.nan))
        ntbf.append(ne); used.append(os.path.basename(d))
    if not stack:
        print("no usable ICs"); return
    A = np.vstack(stack); Pavg = np.nanmean(A, axis=0); ncov = np.sum(np.isfinite(A), axis=0)
    tfs = tgrid * HB
    fin = np.isfinite(Pavg)
    csv = os.path.join(DATA, "aims_pop.csv")
    np.savetxt(csv, np.column_stack([tfs[fin], Pavg[fin], ncov[fin]]), delimiter=",",
               header="t_fs,P_S2_avg,ncov", comments="")
    np.savez(os.path.join(DATA, "aims_perrun.npz"), tfs=tfs, P=A, ntbf=np.array(ntbf))
    np.savetxt(os.path.join(DATA, "aims_meta.txt"), [[len(used), sum(ntbf)]],
               fmt="%d", header="n_ic total_carriers")
    print("AIMS: %d ICs used (%s)" % (len(used), ",".join(used)))
    print("total carriers (sum end-TBF) = %d  mean %.1f/IC  spawn %d/%d"
          % (sum(ntbf), np.mean(ntbf), sum(1 for x in ntbf if x > 1), len(ntbf)))
    print("P_S2(end)=%.3f  -> %s" % (Pavg[fin][-1], csv))


if __name__ == "__main__":
    main()
