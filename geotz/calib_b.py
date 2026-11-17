"""Calibrate the site B model (all-human traffic) against the aerial observations.
Free parameters: left-branch bottleneck speed v_b and the human lane-change parameters.
Targets (observed window, 1 Hz): mean speed of left and right lane groups, lane changes per
1000 vehicle-seconds, share of vehicle-seconds with TTC < 3 s, flow."""
import os, sys, json, itertools
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore, site_b

OBS = dict(spd_left=4.94, spd_right=13.43, lc_rate=4.39, tet_frac=0.0149)
GRID = dict(v_b=[4.0, 4.5, 5.0, 5.5], hv_tau=[1.2, 1.5], rho=[0.9, 1.0], ramp_l=[0.0, 0.435])


def job(args):
    try:
        return _job(args)
    except Exception as e:
        return dict(err=repr(e), args=list(args))


def _job(args):
    v_b, tau, rho, rl, seed = args
    sg, st = 2.0, 3.0
    r = simcore.run(site_b, None, 1.0, 0.0, 1, seed, f'runs/calibB/w{os.getpid()}', v_b, t_end=1200.0, warm=300.0,
                    mix=(1.0, 0.0, 0.0), hv_lc=dict(hv_sg=sg, hv_strat=st, hv_assert=1.0, HV=tau),
                    route_kw=dict(rho=rho, ramp_l=rl))
    vs = sum(r['vs_cells']['B'])
    lc = sum(r['lc_cells']['B'])
    return dict(v_b=v_b, hv_tau=tau, rho=rho, ramp_l=rl, seed=seed, spd_left=r['spd_left'], spd_right=r['spd_right'],
                lc_rate=1000 * lc / vs, tet_frac=sum(r['tet_cells']['B']) / vs, thr=r['throughput_vph'],
                lc_cells=r['lc_cells']['B'], tet_cells=r['tet_cells']['B'], vs_cells=r['vs_cells']['B'])


if __name__ == '__main__':
    for v in GRID['v_b']:
        site_b.build_net('runs/calibB/nets', None, v)
    jobs = [(a, b, c, r, s) for a, b, c, r in itertools.product(*GRID.values()) for s in (101, 102, 103)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 36) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open('runs/calibB/results2.json', 'w'))
    print('done', len(res))
