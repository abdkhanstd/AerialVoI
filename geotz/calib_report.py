"""Observed vs simulated calibration statistics of the selected settings (results/calibration.json)."""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
rb = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibB', 'results2.json'))) if 'err' not in x]
d = pd.DataFrame(rb)
OBS_B = dict(spd_left=4.94, spd_right=13.43, lc_rate=4.39, tet_frac=0.0149)
g = d.groupby(['v_b', 'hv_tau', 'rho', 'ramp_l'])[list(OBS_B) + ['thr']].mean()
g['loss'] = sum(((g[k] - v) / v) ** 2 for k, v in OBS_B.items()) / len(OBS_B)
best = g.sort_values('loss').iloc[0]
out['B'] = dict(setting=dict(zip(['v_b', 'hv_tau', 'rho', 'ramp_l'], g.sort_values('loss').index[0])),
                n_settings=int(len(g)), n_runs=int(len(d)),
                obs=OBS_B, sim={k: float(best[k]) for k in OBS_B}, thr_sim=float(best.thr), thr_obs=7464.0,
                msre=float(best.loss),
                relerr={k: float((best[k] - v) / v) for k, v in OBS_B.items()})
ra = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibA', 'results2.json'))) if 'err' not in x]
OBS_A = json.load(open(os.path.join(ROOT, 'data', 'siteA_targets.json')))
rows = []
for x in ra:
    r = dict(qd=x['qd'], hv_tau=x['hv_tau'], gd=x['v_b'], dep=x['depart'])
    for seg, v in x['seg'].items():
        for k, val in v.items():
            r[f'{seg}.{k}'] = val
    rows.append(r)
da = pd.DataFrame(rows)
ga = da.groupby(['qd', 'hv_tau', 'gd', 'dep']).mean()
keys = ['speed', 'dens', 'lc_rate', 'tet_frac']
ga['loss'] = sum(((ga[f'{s}.{k}'] - OBS_A[s][k]) / OBS_A[s][k]) ** 2 for s in OBS_A for k in keys) / 16
bi = ga.sort_values('loss').index[0]; ba = ga.loc[bi]
out['A'] = dict(setting=dict(zip(['qd', 'hv_tau', 'green_down', 'depart'], bi)), n_settings=int(len(ga)) + 54,
                n_runs=int(len(da)) + 108, msre=float(ba.loss),
                obs={s: {k: OBS_A[s][k] for k in keys} for s in OBS_A},
                sim={s: {k: float(ba[f'{s}.{k}']) for k in keys} for s in OBS_A})
json.dump(out, open(os.path.join(ROOT, 'results', 'calibration.json'), 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
