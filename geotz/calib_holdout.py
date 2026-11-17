"""Held-out check of the calibration: choose the setting on all recordings but one, score it on the
held-out recording, and compare with the error of the setting chosen on that recording itself."""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')


def obs_b():
    d = pd.read_parquet(os.path.join(DATA, 'siteB_tracks.parquet'))
    d = d[d.valid & (d.s >= -325) & (d.s < 225)]
    d1 = d[(d.time * 10).round().astype(int) % 10 == 0]
    c = pd.read_csv(os.path.join(DATA, 'siteB_cells.csv'))
    out = {}
    for r, g in d1.groupby('run_index'):
        cc = c[c.run == r]
        out[int(r)] = dict(spd_left=g[g.lane_kf <= 12].speed_kf.mean(), spd_right=g[g.lane_kf >= 13].speed_kf.mean(),
                           lc_rate=1000 * cc.n_lc.sum() / cc.vehsec.sum(), tet_frac=(g.ttc < 3).mean())
    return out


def obs_a():
    c = pd.read_csv(os.path.join(DATA, 'siteA_cells.csv'))
    win = {'EB': (-130, 210), 'WB': (-160, 170)}
    c = c[[win[a][0] <= s < win[a][1] for a, s in zip(c.ax, c.s_mid)]].copy()
    c['seg'] = np.where(c.s_mid < 0, 'in', np.where(c.s_mid > 40, 'out', 'jct'))
    c = c[c.seg != 'jct']
    T = {0: 582.0}
    out = {}
    for r, g in c.groupby('rec'):
        o = {}
        for (ax, seg), h in g.groupby(['ax', 'seg']):
            V = h.vehsec.sum()
            o[f'{ax}_{seg}'] = dict(speed=(h.speed * h.vehsec).sum() / V, dens=V / T.get(r, 872.0) / h.cell_s.nunique(),
                                   lc_rate=1000 * h.n_lc.sum() / V, tet_frac=h.tet.sum() / V)
        out[int(r)] = o
    return out


def msre(sim, obs):
    return float(np.mean([((sim[k] - v) / v) ** 2 for k, v in obs.items()]))


def main():
    res = {}
    rb = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibB', 'results2.json'))) if 'err' not in x]
    sim = pd.DataFrame(rb).groupby(['v_b', 'hv_tau', 'rho', 'ramp_l'])[['spd_left', 'spd_right', 'lc_rate', 'tet_frac']].mean()
    ob = obs_b()
    rows = []
    for held in ob:
        train = [r for r in ob if r != held]
        tr = {k: np.mean([ob[r][k] for r in train]) for k in ob[held]}
        best = min(sim.index, key=lambda i: msre(sim.loc[i], tr))
        own = min(sim.index, key=lambda i: msre(sim.loc[i], ob[held]))
        rows.append(dict(site='B', held=held, chosen=str(best), err_heldout=msre(sim.loc[best], ob[held]),
                         err_own_best=msre(sim.loc[own], ob[held]), same_as_pooled=best == (4.5, 1.5, 1.0, 0.435)))
    ra = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibA', 'results2.json'))) if 'err' not in x]
    recs = []
    for x in ra:
        r = dict(qd=x['qd'], hv_tau=x['hv_tau'], gd=x['v_b'], dep=x['depart'])
        for seg, v in x['seg'].items():
            for k, val in v.items():
                r[f'{seg}.{k}'] = val
        recs.append(r)
    sa = pd.DataFrame(recs).groupby(['qd', 'hv_tau', 'gd', 'dep']).mean()
    oa = obs_a()
    def flat(o):
        return {f'{s}.{k}': v for s, d in o.items() for k, v in d.items()}
    for held in oa:
        train = [r for r in oa if r != held]
        tr = {k: np.mean([flat(oa[r])[k] for r in train]) for k in flat(oa[held])}
        best = min(sa.index, key=lambda i: msre(sa.loc[i], tr))
        own = min(sa.index, key=lambda i: msre(sa.loc[i], flat(oa[held])))
        rows.append(dict(site='A', held=held, chosen=str(best), err_heldout=msre(sa.loc[best], flat(oa[held])),
                         err_own_best=msre(sa.loc[own], flat(oa[held])), same_as_pooled=best == (1.0, 1.2, 40.0, 'random')))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'results', 'calib_holdout.csv'), index=False)
    print(df.round(3).to_string())


if __name__ == '__main__':
    main()
