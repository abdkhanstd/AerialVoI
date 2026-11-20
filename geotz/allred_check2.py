"""Second all-red check: the downstream-signal planners and the queue given counts, with eight selection seeds.

The first check (allred_check.py) reran G0, G, C and B with four selection seeds. This one reruns, without the
1 s all-red interval (GEOTZ_NOALLRED=1), the planners whose post hoc equivalences the paper reports:
G0 with the observed volume (G0V) and its five downstream-signal variants (+ q, + q_c, + q_g, + q_g', + q_0), and C and
B-b for the step C -> B-b, all on selection seeds 0-3 and 8-11 (grids *n and *ns, chain15). The references are the
no-all-red grids of the first check (v3n, B23n, R3n, R4n, evaluation seeds 4-7). Every comparison is computed in the
original and the corrected version with the same code as voi.py (cluster bootstrap over scenarios, unadjusted
Wilcoxon p, equivalence when the 90% interval lies within +-2 s/veh). Writes results/allred_check2.json.
Usage: python geotz/allred_check2.py
"""
import os, json
import numpy as np, pandas as pd
import voi

SEL8, SEL4, EV, KEYS = voi.SEL8, voi.SEL4, voi.EVAL, voi.KEYS
PLANNERS = {
    'orig': dict(G0V=['G0V', 'G0Vs'], G0Vq=['G0Vq', 'G0Vqs'], G0Vc=['G0Vc', 'G0Vcs'], G0Vg=['G0Vg', 'G0Vgs'],
                 G0Vo=['G0Vo', 'G0Vos'], G0Vx=['G0Vx', 'G0Vxs'], C=['C3', 'C3s'], AB2=['AB2', 'AB2s']),
    'nar': dict(G0V=['G0Vn', 'G0Vns'], G0Vq=['G0Vqn', 'G0Vqns'], G0Vc=['G0Vcn', 'G0Vcns'], G0Vg=['G0Vgn', 'G0Vgns'],
                G0Vo=['G0Von', 'G0Vons'], G0Vx=['G0Vxn', 'G0Vxns'], C=['C3n', 'C3ns'], AB2=['AB2n', 'AB2ns'])}
# G0 at v/c 0.9 with and without the calibrated queue (chain16), used when its grids are present
PLANNERS['orig'].update(G0H=['G0H', 'G0Hs'], G0Hq=['G0Hq', 'G0Hqs'])
PLANNERS['nar'].update(G0H=['G0Hn', 'G0Hns'], G0Hq=['G0Hqn', 'G0Hqns'])
for _L in ('G0H', 'G0Hq'):
    if any(voi.read('A', PLANNERS[v][_L]) is None for v in PLANNERS):
        for v in PLANNERS:
            PLANNERS[v].pop(_L)
REFS = {'orig': dict(B='v3', B2='B23', R3='R3', R4='R4'), 'nar': dict(B='v3n', B2='B23n', R3='R3n', R4='R4n')}
# (problem, less informed, more informed): mean = regret of the first minus regret of the second
COMPS = [(pr, a, b) for pr in ('deploy', 'place') for a, b in
         [('G0V', 'G0Vq'), ('C', 'AB2'), ('G0V', 'G0Vc'), ('G0V', 'G0Vg'), ('G0V', 'G0Vo'), ('G0V', 'G0Vx')]]
COMPS += [('deploy', a, b) for a, b in [('G0Vc', 'G0Vq'), ('G0Vg', 'G0Vq'), ('G0Vo', 'G0Vq'), ('G0Vx', 'G0Vq'),
                                         ('G0Vx', 'G0Vg'), ('G0Vo', 'G0Vg')]]
COMPS += [('deploy', a, r) for a in ('G0Vq', 'G0Vc', 'G0Vg', 'G0Vo', 'G0Vx') for r in ('threshold', 'never')]
COMPS += [('place', a, 'georule') for a in ('G0Vq', 'G0Vc', 'G0Vg', 'G0Vo', 'G0Vx')]
if 'G0H' in PLANNERS['nar']:
    COMPS += [(pr, 'G0H', 'G0Hq') for pr in ('deploy', 'place')]
    COMPS += [('deploy', 'G0H', 'threshold'), ('deploy', 'G0Hq', 'threshold'), ('deploy', 'G0Hq', 'never'),
              ('place', 'G0Hq', 'georule')]
RULES = ('never', 'threshold', 'georule', 'full')


def decisions(version, ref):
    truth = voi.read('A', [REFS[version][ref]])
    truth = truth[truth.p <= 0.6 + 1e-9]
    ev = truth[truth.seed.isin(EV)].groupby(KEYS + ['layout']).J.mean()
    sel = {}
    for L, files in PLANNERS[version].items():
        d = voi.read('A', files)
        if d is None:
            raise SystemExit(f'missing grid for {version} {L}: {files}')
        d = d[d.p <= 0.6 + 1e-9]
        sel[L] = d[d.seed.isin(SEL8)].groupby(KEYS + ['layout']).J.mean()
        sel[L + '#1'] = d[d.seed.isin(SEL4)].groupby(KEYS + ['layout']).J.mean()
        sel[L + '#2'] = d[d.seed.isin([8, 9, 10, 11])].groupby(KEYS + ['layout']).J.mean()
    rows = []
    for k, e in ev.groupby(level=[0, 1, 2, 3, 4]):
        e = e.droplevel([0, 1, 2, 3, 4])
        ch = {}
        for L, s in sel.items():
            sl = s.loc[k]
            ch[f'{L}_deploy'] = sl.idxmin(); ch[f'{L}_place'] = sl.drop('none').idxmin()
        for r, lay in voi.rules('A', k[2]).items():
            ch[r] = lay
        for strat, lay in ch.items():
            rows.append(dict(zip(KEYS, k), strategy=strat, layout=lay, J=float(e[lay]),
                             regret_deploy=float(e[lay] - e.min()), regret_place=float(e[lay] - e.drop('none').min())))
    return pd.DataFrame(rows)


def penalty(g, seeds):
    """Penalty of a WB DL as in mechanism.py: cost of the best WB DL layout minus that of no DL, mean over scenarios."""
    g = g[(g.unit == 'WB') & (g.p <= 0.6 + 1e-9) & g.seed.isin(seeds)]
    m = g.groupby(['p', 'q', 'P', 'layout'])[['J', 'delay']].mean().reset_index()
    r = np.array([(float(s[s.layout != 'none'].J.min() - s[s.layout == 'none'].J.iloc[0]),
                   float(s[s.layout == 'none'].delay.iloc[0])) for _, s in m.groupby(['p', 'q', 'P'])])
    return dict(penalty=float(r[:, 0].mean()), share_dl_better=float((r[:, 0] < 0).mean()), nodl_delay=float(r[:, 1].mean()))


def name(x, prob, h=''):
    return x if x in RULES else f'{x}{h}_{prob}'


VERSIONS = os.environ.get('VERSIONS', 'orig,nar').split(',')   # VERSIONS=orig checks the original side alone
out = {v: {} for v in VERSIONS}
for version in VERSIONS:
    for ref in REFS[version]:
        df = decisions(version, ref)
        res = {'regret': {}, 'wb_dl': {}, 'steps': {}, 'halves': {}}
        for L in PLANNERS[version]:
            g = df[df.strategy == f'{L}_deploy']
            res['regret'][L] = dict(deploy=float(g.regret_deploy.mean()),
                                    place=float(df[df.strategy == f'{L}_place'].regret_place.mean()))
            res['wb_dl'][L] = float((g[g.unit == 'WB'].layout != 'none').mean())
            for h in ('#1', '#2'):
                gh = df[df.strategy == f'{L}{h}_deploy']
                res['wb_dl'][L + h] = float((gh[gh.unit == 'WB'].layout != 'none').mean())
        for prob, a, b in COMPS:
            r = voi.paired(df, name(a, prob), name(b, prob), 'regret_' + prob)
            r['floor'] = voi.is_floor('A', a, ref) or voi.is_floor('A', b, ref)
            res['steps'][f'{prob}:{a}-{b}'] = r
            for h in ('#1', '#2'):
                rh = voi.paired(df, name(a, prob, h), name(b, prob, h), 'regret_' + prob)
                res['halves'][f'{prob}:{a}-{b}{h}'] = dict(mean=rh['mean'], lo=rh['lo'], hi=rh['hi'], eq2=rh['eq2'])
        out[version][ref] = res
    out[version]['wb_penalty'] = {L: penalty(voi.read('A', f), SEL8) for L, f in PLANNERS[version].items()}
    out[version]['wb_penalty'].update({'ref_' + r: penalty(voi.read('A', [f]), EV) for r, f in REFS[version].items()})

if VERSIONS != ['orig', 'nar']:
    for ref in out['orig']:
        for k, v in out['orig'][ref]['steps'].items():
            print(ref, f'{k:28s}', round(v['mean'], 2), [round(v['lo'], 2), round(v['hi'], 2)], 'p', round(v['p'], 3), 'eq2', v['eq2'])
    raise SystemExit
summ = []
for ref in REFS['orig']:
    for k, v0 in out['orig'][ref]['steps'].items():
        if k not in out['nar'][ref]['steps']:
            continue
        v1 = out['nar'][ref]['steps'][k]
        summ.append(dict(ref=ref, comp=k, m0=v0['mean'], m1=v1['mean'], d=v1['mean'] - v0['mean'],
                         lo1=v1['lo'], hi1=v1['hi'], ci90_1=v1['ci90'], p0=v0['p'], p1=v1['p'],
                         eq0=v0['eq2'], eq1=v1['eq2'], u0=v0['p'] < 0.05, u1=v1['p'] < 0.05,
                         sign=bool(np.sign(round(v0['mean'], 6)) == np.sign(round(v1['mean'], 6))), floor=v0['floor']))
out['summary'] = summ
json.dump(out, open(os.path.join(voi.RES, 'allred_check2.json'), 'w'), indent=1, default=float)
S = pd.DataFrame(summ)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(S.drop(columns=['ci90_1']).round(2).to_string())
for v in out:
    if v in ('orig', 'nar'):
        print(v, {ref: {L: round(x, 2) for L, x in out[v][ref]['wb_dl'].items() if '#' not in L} for ref in out[v]})
