"""Check of the 1 s all-red interval that netconvert inserts at the site A junction in the no-DL network and the
layouts ending the DL 90 or 130 m before the stop line (161 s cycle instead of 160 s, 121 instead of 120 s).

Planners G0 (v/c 0.75), G, C and B (selection seeds 0-3) and the four references B, B', R3 and R4 (evaluation seeds
4-7) were rerun with the interval removed (GEOTZ_NOALLRED=1, grids G0n, G3n, C3n, v3n, B23n, R3n, R4n). The
four-seed analysis (p <= 0.6) is repeated on the original and the corrected grids, with Holm correction over the
comparisons of each version. Writes results/allred_check.json.  Usage: python geotz/allred_check.py
"""
import os, json
import numpy as np, pandas as pd
import voi

PLANNERS = {'orig': dict(G0='G0', G='G3', C='C3', B='v3'), 'nar': dict(G0='G0n', G='G3n', C='C3n', B='v3n')}
REFS = {'orig': dict(B='v3', B2='B23', R3='R3', R4='R4'), 'nar': dict(B='v3n', B2='B23n', R3='R3n', R4='R4n')}
SEL, EV = voi.SEL4, voi.EVAL
KEYS = voi.KEYS


def decisions(version, ref):
    truth = voi.read('A', [REFS[version][ref]])
    truth = truth[truth.p <= 0.6 + 1e-9]
    ev = truth[truth.seed.isin(EV)].groupby(KEYS + ['layout']).J.mean()
    sel = {}
    for L, f in PLANNERS[version].items():
        d = voi.read('A', [f])
        sel[L] = d[d.seed.isin(SEL) & (d.p <= 0.6 + 1e-9)].groupby(KEYS + ['layout']).J.mean()
    rows = []
    for k, e in ev.groupby(level=[0, 1, 2, 3, 4]):
        e = e.droplevel([0, 1, 2, 3, 4])
        ch = {}
        for L, s in sel.items():
            sl = s.loc[k]
            ch[f'{L}_deploy'] = sl.idxmin(); ch[f'{L}_place'] = sl.drop('none').idxmin()
        for r, lay in voi.rules('A', k[2]).items():
            ch[r] = lay
        ch['oracle_deploy'] = e.idxmin(); ch['oracle_place'] = e.drop('none').idxmin()
        for strat, lay in ch.items():
            rows.append(dict(zip(KEYS, k), strategy=strat, layout=lay, J=float(e[lay]),
                             regret_deploy=float(e[lay] - e.min()), regret_place=float(e[lay] - e.drop('none').min())))
    return pd.DataFrame(rows), ev


COMPS = [('deploy', 'G0_deploy', 'G_deploy'), ('deploy', 'G_deploy', 'C_deploy'), ('deploy', 'C_deploy', 'B_deploy'),
         ('place', 'G0_place', 'G_place'), ('place', 'G_place', 'C_place'), ('place', 'C_place', 'B_place'),
         ('deploy', 'G_deploy', 'threshold'), ('deploy', 'C_deploy', 'threshold'), ('deploy', 'B_deploy', 'never'),
         ('deploy', 'G_deploy', 'never'), ('place', 'G_place', 'georule'), ('place', 'G0_place', 'georule')]
PLACE_STRATS = ('georule', 'full')

out = {'orig': {}, 'nar': {}}
D, EVS = {}, {}
for version in ('orig', 'nar'):
    for ref in REFS[version]:
        df, ev = decisions(version, ref)
        D[version, ref], EVS[version, ref] = df, ev
        res = {'regret': {}, 'regret_by_unit': {}, 'steps': {}}
        for s, g in df.groupby('strategy'):
            res['regret'][s] = dict(deploy=float(g.regret_deploy.mean()), place=float(g.regret_place.mean()))
            col = 'regret_place' if (s.endswith('_place') or s in PLACE_STRATS) else 'regret_deploy'
            res['regret_by_unit'][s] = {u: float(gu[col].mean()) for u, gu in g.groupby('unit')}
        for prob, a, b in COMPS:
            r = voi.paired(df, a, b, 'regret_' + prob)
            r['floor'] = (ref == 'B' and 'B_' in (a[:2], b[:2]))
            res['steps'][f'{prob}:{a}-{b}'] = r
        out[version][ref] = res
    # Holm over all comparisons of this version (12 comparisons x 4 references)
    cells = [(ref, k) for ref in out[version] for k in out[version][ref]['steps']]
    adj = voi.holm(np.array([out[version][r]['steps'][k]['p'] for r, k in cells]))
    for (r, k), x in zip(cells, adj):
        out[version][r]['steps'][k]['p_holm'] = float(x)

# effect of the interval on the reference costs (evaluation seeds), by reference, layout family and approach
fam = lambda l: 'none' if l == 'none' else ('end 90/130 m' if l.split('_')[0] in ('e-90', 'e-130') else 'end 10/50 m')
out['cost_shift'] = {}
for ref in REFS['orig']:
    e0, e1 = EVS['orig', ref], EVS['nar', ref]
    dj = (e1 - e0.reindex(e1.index)).reset_index()
    dj['family'] = dj.layout.map(fam)
    out['cost_shift'][ref] = {f'{u}|{f}': dict(mean=float(g.J.mean()), min=float(g.J.min()), max=float(g.J.max()))
                              for (u, f), g in dj.groupby(['unit', 'family'])}
# agreement of decisions, by reference and strategy
out['same_decision'] = {}
for ref in REFS['orig']:
    a0 = D['orig', ref].set_index(KEYS + ['strategy']).layout
    a1 = D['nar', ref].set_index(KEYS + ['strategy']).layout
    agree = (a0 == a1.reindex(a0.index))
    out['same_decision'][ref] = {s: float(g.mean()) for s, g in agree.groupby(level='strategy')}
# summary of changes between the versions
summ = []
for ref in REFS['orig']:
    for k, v0 in out['orig'][ref]['steps'].items():
        v1 = out['nar'][ref]['steps'][k]
        summ.append(dict(ref=ref, comp=k, m0=v0['mean'], m1=v1['mean'], d=v1['mean'] - v0['mean'],
                         s0=v0['p_holm'] < 0.05, s1=v1['p_holm'] < 0.05, u0=v0['p'] < 0.05, u1=v1['p'] < 0.05,
                         sign=np.sign(round(v0['mean'], 6)) == np.sign(round(v1['mean'], 6)), floor=v0['floor']))
out['summary'] = summ
json.dump(out, open(os.path.join(voi.RES, 'allred_check.json'), 'w'), indent=1, default=float)
S = pd.DataFrame(summ)
pd.set_option('display.width', 250)
print(S.round(2).to_string())
