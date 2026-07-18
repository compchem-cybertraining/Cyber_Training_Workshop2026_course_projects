from math import ceil
import numpy as np
from tqdm import tqdm

from tenso.prototypes.heom import system_multibath
from tenso.prototypes.bath import gen_bcf


# ── Bath ──────────────────────────────────────────────────────────────────────
bath = gen_bcf(
    re_d    = [540],
    width_d = [700],
    freq_b  = [650, 1252, 1480],
    re_b    = [330, 330, 330],
    width_b = [100, 50, 50],
    temperature           = 300,
    decomposition_method  = 'Pade',
    n_ltc                 = 1,
)


# ── System ───────────────────────────────────────────────────────────────────
sys_ham = np.array([
    [2138, 0, 35477, 0],
    [0, 6414, 0, 0],
    [35447, 0, 108248, 0],
    [0, 0, 0, 109973],
], dtype=np.complex128)

sys_op0 = np.array([
    [1, 0, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -0.5],
], dtype=np.complex128)

sys_ops = [sys_op0]

init_rdo = np.array([
    [0, 0, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0.5],
], dtype=np.complex128)


# ── Run ──────────────────────────────────────────────────────────────────────
end_time = 100  # fs
dt = 0.1  # fs
out = 'output'  # outputs: {out}.dat.log (data) and {out}.debug.log

propagator = system_multibath(
        fname=out,
        init_rdo=init_rdo,
        sys_ham=sys_ham,
        sys_ops=sys_ops,
        bath_correlations=[bath],
        end_time=end_time,
        step_time=dt,
        dim=5,
        max_auxiliary_rank=32,
        ode_atol=1e-9,
        ode_rtol=1e-5,
        frame_method='train',
        rank=1,
        stepwise_method='simple',
        ps_method='ps1',
    )

progress_bar = tqdm(propagator, total=ceil(end_time / dt))
for _t in (progress_bar):
    progress_bar.set_description(f'@{_t:.2f} fs')
