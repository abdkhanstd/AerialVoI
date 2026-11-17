"""Share of decision units in which the best DL layout of a reference raises the delay of human drivers.

For every site, reference model and decision unit with p <= 0.6, the best place-only layout of the
reference (oracle_place in results/voi_decisions.csv) is compared with no DL on the evaluation seeds
(mean delay of HV, CHV and heavy-vehicle trips, column d_hv of the grid files).
Writes results/hv_penalty.json.  Usage: python geotz/hv_penalty.py
"""
import os, json
import pandas as pd
from voi import read, REFS, EVAL, RES

D = pd.read_csv(os.path.join(RES, 'voi_decisions.csv'))
out = {}
for site in ('A', 'B'):
    for ref, f in REFS.items():
        g = read(site, [f])
        g = g[g.seed.isin(EVAL)].groupby(['unit', 'p', 'q', 'P', 'layout']).d_hv.mean()
        d = D[(D.site == site) & (D.ref == ref) & (D.p <= 0.6 + 1e-9) & (D.strategy == 'oracle_place')]
        diff = [g.loc[(r.unit, r.p, r.q, r.P, r.layout)] - g.loc[(r.unit, r.p, r.q, r.P, 'none')] for r in d.itertuples()]
        s = pd.Series(diff)
        out[f'{ref}|{site}'] = dict(n=int(len(s)), share_hv_worse=float((s > 0).mean()), mean_change=float(s.mean()))
        print(ref, site, len(s), round(float((s > 0).mean()), 3), round(float(s.mean()), 1))
json.dump(out, open(os.path.join(RES, 'hv_penalty.json'), 'w'), indent=1)
