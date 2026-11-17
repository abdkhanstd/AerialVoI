"""Check of teleports and discarded vehicles in the most congested scenario (q = 1.0, p = 0.1, P = 4),
all layouts, reference model B, four seeds, using SUMO's statistic output."""
import os, sys, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore, grid


def job(a):
    site, lay, seed = a
    mod = __import__('site_b' if site == 'B' else 'site_a')
    vb = mod.V_B if site == 'B' else mod.V_BN
    r = simcore.run(mod, lay, 1.0, 0.1, 4, 1000 + seed, f'runs/tpcheck/w{os.getpid()}', vb)
    return dict(site=site, layout=str(lay), seed=seed, teleports=r['teleports'], stats=r['vehicle_stats'])


if __name__ == '__main__':
    jobs = [(s, lay, sd) for s in ('A', 'B') for lay in grid.layouts(s) for sd in (4, 5, 6, 7)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 40) as p:
        res = p.map(job, jobs, chunksize=1)
    os.makedirs('results', exist_ok=True)
    json.dump(res, open('results/teleport_check.json', 'w'), indent=1)
    tot = sum(r['teleports'] for r in res)
    print('runs', len(res), 'teleports', tot, 'max', max(r['teleports'] for r in res))
