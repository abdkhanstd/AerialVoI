"""Fidelity of the information-level models: all-human traffic at observed demand vs aerial targets.
Run with GEOTZ_LEVEL=G|C|B; writes runs/fidelity_<level>.json."""
import os, sys, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore, site_a, site_b, calib_a

LEVEL = os.environ.get('GEOTZ_LEVEL', 'B')
OBS_B = dict(spd_left=4.94, spd_right=13.43, lc_rate=4.39, tet_frac=0.0149)


def job(a):
    site, seed = a
    if site == 'B':
        r = simcore.run(site_b, None, 1.0, 0.0, 1, seed, f'runs/fid{LEVEL}/w{os.getpid()}', site_b.V_B, t_end=1200.0, warm=300.0,
                        mix=(1.0, 0.0, 0.0))
        vs = sum(r['vs_cells']['B'])
        return dict(site='B', seed=seed, spd_left=r['spd_left'], spd_right=r['spd_right'],
                    lc_rate=1000 * sum(r['lc_cells']['B']) / vs, tet_frac=sum(r['tet_cells']['B']) / vs, thr=r['throughput_vph'])
    r = simcore.run(site_a, None, 1.0, 0.0, 1, seed, f'runs/fid{LEVEL}/w{os.getpid()}', site_a.V_BN, t_end=1500.0, warm=300.0,
                    mix=(1.0, 0.0, 0.0))
    return dict(site='A', seed=seed, seg=calib_a.seg_stats(r), thr=r['throughput_vph'])


if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    jobs = [(s, 200 + k) for s in ('A', 'B') for k in range(3)]
    with Pool(6) as p:
        res = p.map(job, jobs)
    out = dict(level=LEVEL, runs=res)
    A = [r['seg'] for r in res if r['site'] == 'A']; B = [r for r in res if r['site'] == 'B']
    obsA = calib_a.OBS
    keys = ['speed', 'dens', 'lc_rate', 'tet_frac']
    out['msre_A'] = float(np.mean([((np.mean([a[s][k] for a in A]) - obsA[s][k]) / obsA[s][k]) ** 2 for s in obsA for k in keys]))
    out['msre_B'] = float(np.mean([((np.mean([b[k] for b in B]) - v) / v) ** 2 for k, v in OBS_B.items()]))
    out['mean_A'] = {s: {k: float(np.mean([a[s][k] for a in A])) for k in keys} for s in obsA}
    out['mean_B'] = {k: float(np.mean([b[k] for b in B])) for k in list(OBS_B) + ['thr']}
    json.dump(out, open(f'runs/fidelity_{LEVEL}.json', 'w'), indent=1)
    print(LEVEL, out['msre_A'], out['msre_B'])
