import sys
import numpy as np
from numpy.polynomial.hermite import hermval
from math import factorial

EV2FS = 0.6582119569

NG = 1024; LG = 32.0
xg = np.linspace(-LG / 2, LG / 2, NG, endpoint=False); dxg = xg[1] - xg[0]
kg = 2 * np.pi * np.fft.fftfreq(NG, d=dxg)


def lap(g):
    return np.fft.ifft(-(kg ** 2) * np.fft.fft(g, axis=-1), axis=-1)


_NORM = {}


def _norm(K):
    if K not in _NORM:
        _NORM[K] = (1.0 / np.sqrt(2.0 ** np.arange(K + 1)
                                  * np.array([factorial(n) for n in range(K + 1)], float)))[:, None]
    return _NORM[K]


def mtower(q, p, K, a=0.5):
    xi = xg - q
    base = (2.0 * a / np.pi) ** 0.25 * np.exp(-a * xi ** 2) * np.exp(1j * p * xi)
    Hn = hermval(np.sqrt(2.0 * a) * xi, np.eye(K + 1))
    return base[None, :] * Hn * _norm(K)


def pyrmod4_model():
    F = 4
    omega = np.array([0.09357, 0.0740, 0.1273, 0.1568])
    delta = 0.46165
    lam = 0.1825
    E = {1: -delta, 2: +delta}
    kap = {1: np.array([0.0, -0.0964, 0.0470, 0.1594]),
           2: np.array([0.0,  0.1194, 0.2012, 0.0484])}
    coup = {(1, 2): {'lin': {0: lam}, 'const': 0.0}}
    return dict(F=F, omega=omega, states=[1, 2], E=E, kap=kap, coup=coup)


def qtraj(x, p, om, kap, h):
    def dpos(pp): return om * pp
    def dmom(xx): return -(om * xx + kap)
    k1x, k1p = dpos(p), dmom(x)
    k2x, k2p = dpos(p + 0.5 * h * k1p), dmom(x + 0.5 * h * k1x)
    k3x, k3p = dpos(p + 0.5 * h * k2p), dmom(x + 0.5 * h * k2x)
    k4x, k4p = dpos(p + h * k3p), dmom(x + h * k3x)
    return (x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
            p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p))


def _towers(x, p, Kplist, a, eps):
    D = []; dq = []; dp = []
    for k in range(len(Kplist)):
        K = Kplist[k] - 1
        D.append(mtower(x[k], p[k], K, a))
        dq.append((mtower(x[k] + eps, p[k], K, a) - mtower(x[k] - eps, p[k], K, a)) / (2 * eps))
        dp.append((mtower(x[k], p[k] + eps, K, a) - mtower(x[k], p[k] - eps, K, a)) / (2 * eps))
    return D, dq, dp


def select_configs(Kplist, cutoff):
    F = len(Kplist)
    out = []; cur = np.zeros(F, int)

    def rec(k, acc):
        if k == F:
            out.append(cur.copy()); return
        nk = 0
        while nk < Kplist[k] and acc + nk <= cutoff + 1e-9:
            cur[k] = nk; rec(k + 1, acc + nk); nk += 1
        cur[k] = 0

    rec(0, 0.0)
    return np.array(out, int).reshape(-1, F) if out else np.zeros((0, F), int)


def run_swarm(P, donor, Kp, tf_fs=250.0, dt_fs=0.1, spawn_dfs=5.0, reg=1e-8, merge_thr=0.0,
              a=0.5, eps=1e-5, prune_eps=0.0, prune_age=2.0, spawn_kp=None, cutoff=1e9, spawn_gap=0.0):
    F = P['F']; omega = np.asarray(P['omega'], float); states = P['states']; coup = P['coup']
    kap = {s: np.asarray(P['kap'][s], float) for s in states}
    E = {s: float(P['E'][s]) for s in states}
    Kplist = tuple(int(k) for k in Kp)
    spawn_Kp = Kplist if spawn_kp is None else tuple(int(k) for k in spawn_kp)
    cfg_main = select_configs(Kplist, cutoff)
    cfg_spawn = select_configs(spawn_Kp, cutoff)

    def fdim(f):
        return len(f['cfg'])

    def combine(mats, f, g):
        cf = f['cfg']; cg = g['cfg']
        out = None; scal = 1.0 + 0.0j
        for k in range(F):
            Mk = mats[k]
            if Mk.shape == (1, 1):
                scal = scal * Mk[0, 0]
            else:
                gk = Mk[cf[:, k][:, None], cg[:, k][None, :]]
                out = gk.astype(complex) if out is None else out * gk
        if out is None:
            out = np.ones((len(cf), len(cg)), complex)
        return scal * out

    lam_pairs = {pr: dict(v['lin']) for pr, v in coup.items() if v['lin']}
    donor_idx = states.index(donor)

    def _offsets(flist):
        dims = [fdim(f) for f in flist]
        return np.concatenate([[0], np.cumsum(dims)]).astype(int)

    frames = [dict(st=donor, x=np.zeros(F), p=np.zeros(F), birth=0.0, Kp=Kplist, cfg=cfg_main)]
    zero_idx = int(np.where((cfg_main == 0).all(axis=1))[0][0])
    Reff = 1
    C0 = np.zeros((len(cfg_main), Reff), complex); C0[zero_idx, 0] = 1.0
    C = [C0]

    def build(mid):
        N = len(mid)
        tow = [_towers(f['x'], f['p'], f['Kp'], a, eps) for f in mid]
        off = _offsets(mid); D = int(off[-1])
        S = np.zeros((D, D), complex); Hb = np.zeros((D, D), complex)
        for f in range(N):
            Df, dqf, dpf = tow[f]; sf = mid[f]['st']
            for g in range(N):
                Dg, dqg, dpg = tow[g]; sg = mid[g]['st']
                rf = slice(off[f], off[f + 1]); rg = slice(off[g], off[g + 1])
                if sf == sg:
                    kp = kap[sf]
                    O = [(Df[k].conj() @ Dg[k].T) * dxg for k in range(F)]
                    Sblk = combine(O, mid[f], mid[g]); S[rf, rg] = Sblk
                    hml = []
                    for k in range(F):
                        TVg = (omega[k] * (-0.5 * lap(Dg[k])) + omega[k] * (0.5 * (xg ** 2) * Dg[k])
                               + kp[k] * (xg * Dg[k]))
                        if k == 0:
                            TVg = TVg + E[sf] * Dg[k]
                        Hk = (Df[k].conj() @ TVg.T) * dxg
                        qd = omega[k] * mid[g]['p'][k]; pd = -(omega[k] * mid[g]['x'][k] + kp[k])
                        tk = (Df[k].conj() @ (qd * dqg[k] + pd * dpg[k]).T) * dxg
                        hml.append(Hk - 1j * tk)
                    bath_sc = 0.0 + 0.0j; Hacc = None
                    for k in range(F):
                        if hml[k].shape == (1, 1):
                            bath_sc = bath_sc + hml[k][0, 0] / O[k][0, 0]
                        else:
                            term = combine([hml[kk] if kk == k else O[kk] for kk in range(F)],
                                           mid[f], mid[g])
                            Hacc = term if Hacc is None else Hacc + term
                    Hb[rf, rg] = bath_sc * Sblk if Hacc is None else bath_sc * Sblk + Hacc
                else:
                    pr = tuple(sorted((sf, sg)))
                    if pr in lam_pairs:
                        Oov = [(Df[k].conj() @ Dg[k].T) * dxg for k in range(F)]
                        blk = None
                        for cm, lamv in lam_pairs[pr].items():
                            Qcm = (Df[cm].conj() @ (xg * Dg[cm]).T) * dxg
                            Q = [Qcm if k == cm else Oov[k] for k in range(F)]
                            term = lamv * combine(Q, mid[f], mid[g])
                            blk = term if blk is None else blk + term
                        if blk is not None:
                            Hb[rf, rg] = blk
        return S, Hb, off

    def propagator(mid):
        S, Hb, _ = build(mid)
        w, U = np.linalg.eigh(S)
        winv = np.where(w > reg * w.max(), 1.0 / w, 0.0)
        return -1j * (U * winv) @ (U.conj().T @ Hb)

    def populations():
        S, _, off = build(frames)
        out = {}
        for s in states:
            idx = [i for i in range(len(frames)) if frames[i]['st'] == s]
            tot = 0.0
            for i in idx:
                for j in idx:
                    blk = S[off[i]:off[i + 1], off[j]:off[j + 1]]
                    tot += np.real(np.einsum('ar,ab,br->', C[i].conj(), blk, C[j]))
            out[s] = tot
        Z = sum(out.values()) or 1.0
        return {s: out[s] / Z for s in states}, Z

    def moverlap_full(i, j):
        al_i = (frames[i]['x'] + 1j * frames[i]['p']) / np.sqrt(2.0)
        al_j = (frames[j]['x'] + 1j * frames[j]['p']) / np.sqrt(2.0)
        return np.exp(-0.5 * np.sum(np.abs(al_i) ** 2) - 0.5 * np.sum(np.abs(al_j) ** 2)
                      + np.sum(al_i.conj() * al_j))

    def kick_momentum(xd, pd, sd, sa):
        pr = tuple(sorted((sd, sa)))
        dvec = np.zeros(F)
        for k, lv in lam_pairs.get(pr, {}).items():
            dvec[k] = lv
        nrm = float(np.sum(omega * dvec * dvec))
        if nrm <= 1e-14:
            return pd
        dE = (E[sd] - E[sa]) + float(np.sum((kap[sd] - kap[sa]) * xd))
        A = 0.5 * nrm; B = float(np.sum(omega * pd * dvec)); disc = B * B + 4 * A * dE
        if disc < 0:
            return pd
        alpha = (-B + np.sqrt(disc)) / (2 * A)
        return pd + alpha * dvec

    def diab_gap(xd, sd, sa):
        return (E[sd] - E[sa]) + float(np.sum((kap[sd] - kap[sa]) * xd))

    def do_spawn(tcur):
        nonlocal C
        newf = []
        for i in range(len(frames)):
            if frames[i]['st'] != donor:
                continue
            if spawn_gap > 0.0:
                if not any(abs(diab_gap(frames[i]['x'], donor, s)) < spawn_gap
                           for s in states if s != donor):
                    continue
            for s in states:
                if s != donor:
                    pk = kick_momentum(frames[i]['x'], frames[i]['p'], donor, s)
                    newf.append(dict(st=s, x=frames[i]['x'].copy(), p=pk,
                                     birth=tcur, Kp=spawn_Kp, cfg=cfg_spawn))
        if newf:
            frames.extend(newf)
            C = C + [np.zeros((fdim(f), Reff), complex) for f in newf]

    def do_merge(thr):
        nonlocal C
        while len(frames) > len(states):
            bystate = {}
            for i, f in enumerate(frames):
                bystate.setdefault(f['st'], []).append(i)
            best = None
            for s, idx in bystate.items():
                for a_ in range(len(idx)):
                    for b_ in range(a_ + 1, len(idx)):
                        ov = abs(moverlap_full(idx[a_], idx[b_]))
                        if ov > thr and (best is None or ov > best[0]):
                            best = (ov, idx[a_], idx[b_])
            if best is None:
                break
            _, i, j = best
            ni = float(np.real(np.einsum('ar,ar->', C[i].conj(), C[i])))
            nj = float(np.real(np.einsum('ar,ar->', C[j].conj(), C[j])))
            if nj > ni:
                i, j = j, i
            Di = _towers(frames[i]['x'], frames[i]['p'], frames[i]['Kp'], a, eps)[0]
            Dj = _towers(frames[j]['x'], frames[j]['p'], frames[j]['Kp'], a, eps)[0]
            Omod = [(Di[k].conj() @ Dj[k].T) * dxg for k in range(F)]
            Oij = combine(Omod, frames[i], frames[j])
            C[i] = C[i] + Oij @ C[j]
            frames.pop(j); C.pop(j)

    def do_prune(tnow):
        nonlocal C
        if prune_eps <= 0.0:
            return
        norms = [float(np.real(np.einsum('ar,ar->', C[i].conj(), C[i]))) for i in range(len(frames))]
        drop = set(i for i in range(len(frames))
                   if tnow - frames[i]['birth'] > prune_age and norms[i] < prune_eps)
        for s in set(f['st'] for f in frames):
            idx = [i for i in range(len(frames)) if frames[i]['st'] == s]
            if idx and all(i in drop for i in idx):
                drop.discard(max(idx, key=lambda i: norms[i]))
        if drop:
            keep = [i for i in range(len(frames)) if i not in drop]
            frames[:] = [frames[i] for i in keep]; C = [C[i] for i in keep]

    dt = dt_fs / EV2FS
    tout = 0.5 if tf_fs > 5 else 1.0
    times = np.arange(0, tf_fs + 1e-9, tout)
    nsub = max(1, int(round((times[1] - times[0]) / dt_fs)))
    h = (times[1] - times[0]) / EV2FS / nsub
    tcur = 0.0; last_spawn = -spawn_dfs
    pops = []; nrm = []
    for ti, t_fs in enumerate(times):
        if ti > 0:
            for _ in range(nsub):
                if tcur - last_spawn >= spawn_dfs - 1e-12:
                    do_spawn(tcur)
                    if merge_thr:
                        do_merge(merge_thr)
                    do_prune(tcur)
                    last_spawn = tcur
                mid = []
                for f in frames:
                    xm, pm = qtraj(f['x'], f['p'], omega, kap[f['st']], 0.5 * h)
                    mid.append(dict(st=f['st'], Kp=f['Kp'], cfg=f['cfg'], x=xm, p=pm))
                Cstack = np.vstack(C)
                M = propagator(mid)
                Nd = M.shape[0]; Im = np.eye(Nd)
                Cstack = np.linalg.solve(Im - 0.5 * h * M, (Im + 0.5 * h * M) @ Cstack)
                off = _offsets(frames)
                C = [Cstack[off[i]:off[i + 1]] for i in range(len(frames))]
                for f in frames:
                    f['x'], f['p'] = qtraj(f['x'], f['p'], omega, kap[f['st']], h)
                tcur += h * EV2FS
        pr, Z = populations()
        pops.append([pr[s] for s in states]); nrm.append(Z)
        nb = int(_offsets(frames)[-1])
        if abs(t_fs % 10.0) < 1e-9:
            print("  t=%6.1f fs  P(don=%s)=%.5f  frames=%d  N=%d  Z=%.6f"
                  % (t_fs, donor, pr[donor], len(frames), nb, nrm[-1]), flush=True)
    pops = np.array(pops)
    return times, pops[:, donor_idx], pops, np.array(nrm)


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    args = sys.argv[1:]
    opt = {k: v for k, _, v in (x[2:].partition('=') for x in args if x.startswith('--'))}
    P = pyrmod4_model()
    donor = 2
    Kp = tuple(int(x) for x in opt['Kp'].split(',')) if 'Kp' in opt else (6, 3, 3, 3)
    tf = float(opt.get('tf', 120.0)); dt = float(opt.get('dt', 0.1))
    spawn_dfs = float(opt.get('spawn', 5.0))
    merge_thr = float(opt.get('merge', 0.0))
    reg = float(opt.get('reg', 1e-8))
    prune_eps = float(opt.get('prune', 0.0)); prune_age = float(opt.get('pruneage', 2.0))
    spawn_kp = tuple(int(x) for x in opt['spawnkp'].split(',')) if 'spawnkp' in opt else None
    cutoff = float(opt.get('cutoff', 1e9))
    spawn_gap = float(opt.get('spawngap', 0.0))
    tag = opt.get('tag', 'pyrmod4')
    nsel = len(select_configs(Kp, cutoff))
    print("=== tower_pyr4: pyrmod4  states=%s donor=%s  Kp=%s  cutoff=%g (n_sel=%d / %d full) ===" %
          (P['states'], donor, Kp, cutoff, nsel, int(np.prod(Kp))), flush=True)
    t, Pd, pops, nrm = run_swarm(P, donor, Kp, tf_fs=tf, dt_fs=dt, spawn_dfs=spawn_dfs,
                                 merge_thr=merge_thr, reg=reg, prune_eps=prune_eps, prune_age=prune_age,
                                 spawn_kp=spawn_kp, cutoff=cutoff, spawn_gap=spawn_gap)
    hdr = 't_fs ' + ' '.join('P_%s' % s for s in P['states']) + ' Z'
    out = 'swarm_pop_%s.csv' % tag
    np.savetxt(out, np.column_stack([t, pops, nrm]), header=hdr, comments='')
    print("Z drift = %.2e   -> %s" % (abs(nrm - nrm[0]).max(), out))
