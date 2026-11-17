"""Mechanism checks for the site A decisions.

1. Penalty of a westbound (WB) DL in each planner model and each reference model: for every scenario
   (p <= 0.6), the cost J of the best DL layout minus that of no DL, on the planner's selection seeds or
   the reference's evaluation seeds, together with the mean no-DL WB delay and the share of scenarios in
   which a WB DL is better than none.
2. Per-direction information steps on each four-seed half of the selection seeds (strategies '<planner>#1'
   and '<planner>#2' in results/voi_decisions.csv), and the EB placement regret of every planner per half.
3. Rank correlations of the assumed EB red-phase arrivals with the chosen EB DL end and the EB
   placement regret, on each four-seed half.
Writes results/mechanism.json.  Usage: python geotz/mechanism.py
"""
import os, json
import numpy as np
import pandas as pd
from voi import read, PLAN, REFS, EVAL, RES

out = {'wb_penalty': {}, 'dir_halves': {}, 'eb_place_halves': {}}


def penalty(g, seeds):
    g = g[(g.unit == 'WB') & (g.p <= 0.6 + 1e-9) & g.seed.isin(seeds)]
    m = g.groupby(['p', 'q', 'P', 'layout'])[['J', 'delay']].mean().reset_index()
    rows = []
    for k, s in m.groupby(['p', 'q', 'P']):
        none = s[s.layout == 'none']
        dl = s[s.layout != 'none']
        rows.append((float(dl.J.min() - none.J.iloc[0]), float(none.delay.iloc[0])))
    r = np.array(rows)
    return dict(n=len(r), penalty=float(r[:, 0].mean()), share_dl_better=float((r[:, 0] < 0).mean()), nodl_delay=float(r[:, 1].mean()))


for L in ('G0V', 'G0H', 'G0Vq', 'G0Hq', 'G0Vc', 'G0Vk', 'G0Vg', 'G0Vx', 'G0Vo', 'G', 'C', 'B'):
    files, seeds = PLAN[L]['A']
    out['wb_penalty'][L] = penalty(read('A', files), seeds)
for ref, f in REFS.items():
    out['wb_penalty']['ref_' + ref] = penalty(read('A', [f]), EVAL)

D = pd.read_csv(os.path.join(RES, 'voi_decisions.csv'))
D = D[(D.site == 'A') & (D.p <= 0.6 + 1e-9)]
STEPS = [('place', 'G0', 'G0V'), ('place', 'G0V', 'G'), ('place', 'G0V', 'G0Vq'), ('place', 'G0H', 'G0Hq'),
         ('deploy', 'G0V', 'G0Vq'), ('deploy', 'G0H', 'G0Hq'), ('deploy', 'G0V', 'G0Vc'), ('deploy', 'G0V', 'G0Vk'), ('deploy', 'G0V', 'G0Vg'), ('deploy', 'G0V', 'G0Vx'), ('deploy', 'G0V', 'G0Vo')]
for ref in REFS:
    for u in ('EB', 'WB'):
        d = D[(D.ref == ref) & (D.unit == u)]
        for prob, a, b in STEPS:
            col = 'regret_' + prob
            for h in ('#1', '#2'):
                xa = d[d.strategy == f'{a}{h}_{prob}'].set_index(['p', 'q', 'P'])[col]
                xb = d[d.strategy == f'{b}{h}_{prob}'].set_index(['p', 'q', 'P'])[col]
                out['dir_halves'][f'{ref}|{u}|{prob}:{a}-{b}{h}'] = float((xa - xb.reindex(xa.index)).mean())
    d = D[(D.ref == ref) & (D.unit == 'EB')]
    for L in ('G0', 'G0V', 'G', 'C', 'B'):
        for h in ('#1', '#2', ''):
            out['eb_place_halves'][f'{ref}|{L}{h}'] = float(d[d.strategy == f'{L}{h}_place'].regret_place.mean())

# rank correlation of the assumed EB red-phase arrivals with the chosen EB end and the EB placement regret, per half
from scipy.stats import spearmanr
X = json.load(open(os.path.join(RES, 'direction.json')))
P7 = ['G0', 'G0H', 'G0V', 'G0F', 'G', 'C', 'B']
arr = [X['planner_inputs'][L]['EB']['red_arrivals'] for L in P7]
out['eb_spearman_halves'] = {}
for ref in REFS:
    d = D[(D.ref == ref) & (D.unit == 'EB')]
    for h in ('#1', '#2'):
        end = [d[d.strategy == f'{L}{h}_place'].layout.map(lambda x: float(x.split('_')[0][1:])).mean() for L in P7]
        reg = [d[d.strategy == f'{L}{h}_place'].regret_place.mean() for L in P7]
        out['eb_spearman_halves'][f'{ref}{h}'] = dict(rho_end=float(spearmanr(arr, end)[0]), rho_regret=float(spearmanr(arr, reg)[0]))
print(out['eb_spearman_halves'])
json.dump(out, open(os.path.join(RES, 'mechanism.json'), 'w'), indent=1)
for k, v in out['wb_penalty'].items():
    print('wb', k, {a: round(b, 2) for a, b in v.items()})
for k, v in out['dir_halves'].items():
    print('half', k, round(v, 2))
for k, v in out['eb_place_halves'].items():
    print('ebplace', k, round(v, 2))
