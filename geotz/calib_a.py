"""Calibrate the site A model (all-human traffic) against the UAV observations.
Free parameters: demand multiplier of the estimated approach volumes, human time gap, and the
speed of the westbound downstream bottleneck. Targets per arterial segment (EB/WB approach and
exit leg, observed windows): mean speed, vehicles per 10 m cell, lane changes per 1000
vehicle-seconds and share of vehicle-seconds with TTC < 3 s."""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore, site_a

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = json.load(open(os.path.join(ROOT, 'data', 'siteA_targets.json')))
GRID = dict(qd=[0.9, 1.0], hv_tau=[1.2, 1.5], v_b=[40.0, 60.0, 80.0], depart=['best', 'random'])
T_END, WARM = 1500.0, 300.0


def seg_stats(r):
    out = {}
    for ax, (o0, o1) in site_a.OBS_WINDOW.items():
        s = o0 + site_a.CELL * (np.arange(len(r['vs_cells'][ax])) + 0.5)
        for seg, m in (('in', s < 0), ('out', s > 40)):
            vs = np.array(r['vs_cells'][ax])[m]; sp = np.array(r['spd_cells'][ax])[m]
            lc = np.array(r['lc_cells'][ax])[m]; te = np.array(r['tet_cells'][ax])[m]
            V = max(vs.sum(), 1)
            out[f'{ax}_{seg}'] = dict(speed=sp.sum() / V, dens=vs.sum() / (T_END - WARM) / m.sum(),
                                      lc_rate=1000 * lc.sum() / V, tet_frac=te.sum() / V)
    return out


def job(args):
    try:
        qd, tau, vb, dep, seed = args
        sg = 1.0
        r = simcore.run(site_a, None, qd, 0.0, 1, seed, f'runs/calibA/w{os.getpid()}', vb, t_end=T_END, warm=WARM,
                        mix=(1.0, 0.0, 0.0), hv_lc=dict(hv_sg=sg, hv_strat=1.0, hv_assert=1.0, HV=tau),
                        route_kw=dict(depart=dep))
        return dict(qd=qd, hv_tau=tau, v_b=vb, depart=dep, seed=seed, seg=seg_stats(r), delay=r['delay_mean'],
                    thr=r['throughput_vph'], backlog=r['backlog_vs'],
                    cells={k: r[k] for k in ('vs_cells', 'spd_cells', 'lc_cells', 'tet_cells')})
    except Exception as e:
        return dict(err=repr(e), args=list(args))


if __name__ == '__main__':
    os.makedirs('runs/calibA', exist_ok=True)
    for vb in GRID['v_b']:
        site_a.build_net('runs/calibA/nets', None, vb)
    jobs = [k + (s,) for k in itertools.product(*GRID.values()) for s in (101, 102)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 36) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open('runs/calibA/results2.json', 'w'))
    print('done', len(res))
