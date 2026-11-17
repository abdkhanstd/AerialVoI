"""Per-direction analysis of the site A decisions (eastbound and westbound decision units).

Reads results/voi_decisions.csv (written by voi.py) and writes results/direction.json with
  - regret of every strategy by direction and reference,
  - information steps by direction (mean and 95 % bootstrap CI over the 30 scenarios),
  - DL opening rates by direction,
  - the approach demand, red time and red-phase arrivals assumed by each planner, and the
    DL end position each planner chooses for eastbound placement, against the oracle.
Usage: python geotz/direction.py
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
D = pd.read_csv(os.path.join(R, 'voi_decisions.csv'))
D = D[(D.site == 'A') & (D.p <= 0.6 + 1e-9)].copy()
REFS = ['B', 'B2', 'R3', 'R4']
rng = np.random.default_rng(0)

# demand and signal assumed by each planner (site_a.py LEVEL blocks); observed: EB 1885, WB 1520 veh/h,
# 160 s cycle with 57 s EB green and 43 s WB green; G: equal greens of 37 s; G0 variants: 120 s cycle, 27 s greens
OBS_TOT = 1884.9 + 1519.6 + 1344.5 + 1269.3
PLAN_IN = {
    'G0': dict(EB=0.75 * 1620, WB=0.75 * 1620, cycle=120, gEB=27, gWB=27),
    'G0H': dict(EB=0.9 * 1620, WB=0.9 * 1620, cycle=120, gEB=27, gWB=27),
    'G0V': dict(EB=OBS_TOT / 4, WB=OBS_TOT / 4, cycle=120, gEB=27, gWB=27),
    'G0F': dict(EB=1620.0, WB=1620.0, cycle=120, gEB=27, gWB=27),
    'G': dict(EB=OBS_TOT / 4, WB=OBS_TOT / 4, cycle=160, gEB=37, gWB=37),
    'C': dict(EB=1884.9, WB=1519.6, cycle=160, gEB=57, gWB=43),
    'B': dict(EB=1884.9, WB=1519.6, cycle=160, gEB=57, gWB=43),
}
YEL = 3


def boot(x, n=4000):
    x = np.asarray(x, float)
    i = rng.integers(0, len(x), (n, len(x)))
    m = x[i].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def end_pos(layout):
    return np.nan if layout == 'none' else float(layout.split('_')[0][1:])


out = {'regret': {}, 'steps': {}, 'open_rate': {}, 'planner_inputs': {}, 'eb_place_end': {}}
STRATS = ['never', 'threshold', 'georule', 'full', 'G0', 'G0H', 'G0F', 'G0V', 'G0Vq', 'G0Hq', 'G0Vc', 'G0Vk', 'G0Vg', 'G0Vx', 'G0Vo', 'G', 'C', 'B', 'AB1', 'AB1g', 'AB1e', 'AB2']
for ref in REFS:
    for u in ('EB', 'WB'):
        d = D[(D.ref == ref) & (D.unit == u)]
        tab = {}
        for s in STRATS:
            if s in ('never', 'threshold', 'georule', 'full'):
                g = d[d.strategy == s]
                tab[s] = dict(deploy=float(g.regret_deploy.mean()), place=float(g.regret_place.mean()) if s in ('georule', 'full') else None)
            else:
                gd = d[d.strategy == s + '_deploy']; gp = d[d.strategy == s + '_place']
                tab[s] = dict(deploy=float(gd.regret_deploy.mean()), place=float(gp.regret_place.mean()))
        out['regret'][f'{ref}|{u}'] = tab
        # opening rates
        rate = {}
        for s in ['G0', 'G0H', 'G0F', 'G0V', 'G0Vq', 'G0Hq', 'G0Vc', 'G0Vk', 'G0Vg', 'G0Vx', 'G0Vo', 'G', 'C', 'B', 'AB1', 'AB1g', 'AB1e', 'AB2']:
            rate[s] = float((d[d.strategy == s + '_deploy'].layout != 'none').mean())
        rate['oracle'] = float((d[d.strategy == 'oracle_deploy'].layout != 'none').mean())
        out['open_rate'][f'{ref}|{u}'] = rate
        # information steps (first minus second; positive: the second, more informed planner is better)
        st = {}
        for prob, a, b in [('place', 'G0', 'G0V'), ('place', 'G0V', 'G'), ('place', 'G', 'C'), ('place', 'C', 'B'), ('place', 'AB2', 'B'),
                           ('deploy', 'G0', 'G0V'), ('deploy', 'G0V', 'G'), ('deploy', 'G0H', 'G'), ('deploy', 'G', 'C'), ('deploy', 'C', 'B'),
                           ('deploy', 'AB1', 'B'), ('deploy', 'AB1g', 'B'), ('deploy', 'AB1e', 'B'), ('deploy', 'AB2', 'B'),
                           ('deploy', 'G0V', 'G0Vq'), ('deploy', 'G0H', 'G0Hq'), ('place', 'G0V', 'G0Vq'), ('place', 'G0H', 'G0Hq'),
                           ('deploy', 'G0V', 'G0Vc'), ('deploy', 'G0V', 'G0Vk'), ('place', 'G0V', 'G0Vc'), ('place', 'G0V', 'G0Vk'), ('deploy', 'G0V', 'G0Vg'), ('place', 'G0V', 'G0Vg'), ('deploy', 'G0V', 'G0Vx'), ('place', 'G0V', 'G0Vx'), ('deploy', 'G0V', 'G0Vo'), ('place', 'G0V', 'G0Vo')]:
            col = 'regret_' + prob
            xa = d[d.strategy == f'{a}_{prob}'].set_index(['p', 'q', 'P'])[col]
            xb = d[d.strategy == f'{b}_{prob}'].set_index(['p', 'q', 'P'])[col]
            diff = (xa - xb.reindex(xa.index)).values
            m, lo, hi = boot(diff)
            st[f'{prob}:{a}-{b}'] = dict(mean=m, lo=lo, hi=hi)
        out['steps'][f'{ref}|{u}'] = st

# planner inputs: approach demand error and arrivals during red (at the mean demand factor of the scenarios)
qbar = float(D.drop_duplicates(['p', 'q', 'P']).q.mean())
for k, v in PLAN_IN.items():
    r = {}
    for u, obs in (('EB', 1884.9), ('WB', 1519.6)):
        red = v['cycle'] - v['g' + u] - YEL
        r[u] = dict(demand=v[u], error=v[u] / obs - 1.0, red=red, red_arrivals=v[u] * qbar * red / 3600.0)
    out['planner_inputs'][k] = r
obs_red = {'EB': 160 - 57 - YEL, 'WB': 160 - 43 - YEL}
out['observed_red_arrivals'] = {u: v * qbar * obs_red[u] / 3600.0 for u, v in (('EB', 1884.9), ('WB', 1519.6))}

# eastbound placement: chosen DL end (m from the stop line, negative upstream) by planner and oracle
for ref in REFS:
    d = D[(D.ref == ref) & (D.unit == 'EB')]
    r = {}
    for s in ['G0', 'G0H', 'G0V', 'G0F', 'G', 'C', 'B']:
        e = d[d.strategy == s + '_place'].layout.map(end_pos)
        r[s] = dict(mean_end=float(e.mean()), share_far=float((e <= -130).mean()), regret=float(d[d.strategy == s + '_place'].regret_place.mean()))
    e = d[d.strategy == 'oracle_place'].layout.map(end_pos)
    r['oracle'] = dict(mean_end=float(e.mean()), share_far=float((e <= -130).mean()), regret=0.0)
    out['eb_place_end'][ref] = r
# rank correlation between the assumed eastbound red-phase arrivals and the placement regret / chosen end
from scipy.stats import spearmanr
P = ['G0', 'G0H', 'G0V', 'G0F', 'G', 'C', 'B']
qa = [out['planner_inputs'][k]['EB']['red_arrivals'] for k in P]
cor = {}
for ref in REFS:
    reg = [out['eb_place_end'][ref][k]['regret'] for k in P]
    end = [out['eb_place_end'][ref][k]['mean_end'] for k in P]
    cor[ref] = dict(rho_regret=float(spearmanr(qa, reg)[0]), rho_end=float(spearmanr(qa, end)[0]))
out['eb_spearman'] = cor
json.dump(out, open(os.path.join(R, 'direction.json'), 'w'), indent=1)

# console summary
print('mean demand factor', round(qbar, 3), 'observed red arrivals', {k: round(v, 1) for k, v in out['observed_red_arrivals'].items()})
for k in P:
    pi = out['planner_inputs'][k]
    print(f"{k:4s} EB err {pi['EB']['error']:+.2f} red {pi['EB']['red']} arr {pi['EB']['red_arrivals']:.1f} | WB err {pi['WB']['error']:+.2f} red {pi['WB']['red']} arr {pi['WB']['red_arrivals']:.1f}")
for ref in REFS:
    print(ref, 'EB place end:', {k: (round(v['mean_end']), round(v['share_far'], 2), round(v['regret'], 1)) for k, v in out['eb_place_end'][ref].items()}, cor[ref])
for ref in REFS:
    for u in ('EB', 'WB'):
        s = out['steps'][f'{ref}|{u}']
        print(ref, u, {k: f"{v['mean']:.1f} [{v['lo']:.1f},{v['hi']:.1f}]" for k, v in s.items()})
for ref in REFS:
    for u in ('EB', 'WB'):
        print(ref, u, 'open', {k: round(v, 2) for k, v in out['open_rate'][f'{ref}|{u}'].items()})
