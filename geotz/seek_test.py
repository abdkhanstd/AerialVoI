"""Diagnostic: effect of the DL-seeking rule of CAVs on CAV and HV delay (site A and B, p = 0.5)."""
import os, sys, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore, site_a, site_b


def job(a):
    site, lay, seek, seed = a
    m = site_a if site == 'A' else site_b
    vb = site_a.V_BN if site == 'A' else site_b.V_B
    r = simcore.run(m, lay, 0.85, 0.5, 4, seed, f'runs/seek/w{os.getpid()}', vb, dl_seek=seek)
    return dict(site=site, lay=str(lay), seek=seek, seed=seed, J=r['delay_mean'], by=r['delay_by_class'], dl=r['dl_share'],
                orig=r['delay_by_origin'])


if __name__ == '__main__':
    jobs = [(s, lay, seek, sd) for s, lays in (('A', [None, dict(e=-130.0, g=40.0)]), ('B', [None, 25.0, -325.0]))
            for lay in lays for seek in (True, False) for sd in (1, 2)]
    jobs = [j for j in jobs if not (j[1] is None and j[2] is False)]
    with Pool(12) as p:
        for r in p.imap_unordered(job, jobs):
            print(json.dumps(r), flush=True)
