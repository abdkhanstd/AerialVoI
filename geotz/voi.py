"""Value of aerial information for dedicated-lane (DL) decisions, across several reference models.

Planner models (choose on their own selection seeds; at site A the models marked * use eight
selection seeds 0-3 and 8-11, all others seeds 0-3):
  G0   imagery only, demand from lane-count capacity at v/c 0.75, generic 120 s signal
  G0V* imagery + observed total volume, generic 120 s signal          (site A)
  G0H* imagery, capacity default at v/c 0.9, generic 120 s signal     (site A)
  G0F  imagery, capacity default at v/c 1.0, generic 120 s signal     (site A)
  G    imagery + observed total volume, generic 160 s equal-split signal
  C*   + movement counts and observed signal plan
  B*   + trajectory layer (queue boundary state and calibrated behaviour)
  AB1  B without queue state; AB1g / AB1e: no queue state and only the calibrated time gap /
       only the random entry lanes (site A)
  AB2* B without calibrated behaviour
Reference models (evaluation seeds 4-7): B, B2, R3, R4.
Regret = cost of the choice minus cost of the best option of the problem in the reference.
One family of paired comparisons per site and reference, Holm-corrected Wilcoxon tests on scenario
means, cluster-bootstrap CIs over scenarios, minimal detectable difference (80% power, alpha 0.05)
from the bootstrap standard error. GEOTZ_METRIC=delay repeats everything without the TTC term.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SEL4, SEL8, EVAL = [0, 1, 2, 3], [0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7]
KEYS = ['site', 'unit', 'p', 'q', 'P']
NB = 10000
METRIC = os.environ.get('GEOTZ_METRIC', 'J')
PLAN = {
    'G0': {'A': (['G0', 'G0s'], SEL8), 'B': (['G0'], SEL4)},
    'G0V': {'A': (['G0V', 'G0Vs'], SEL8)},
    'G0H': {'A': (['G0H', 'G0Hs'], SEL8)},
    'G0F': {'A': (['G0F', 'G0Fs'], SEL8)},
    'G': {'A': (['G3', 'G3s'], SEL8), 'B': (['G3'], SEL4)},
    'C': {'A': (['C3', 'C3s'], SEL8), 'B': (['C3'], SEL4)},
    'B': {'A': (['v3', 'v3s'], SEL8), 'B': (['v3'], SEL4)},
    'AB1': {'A': (['AB1'], SEL4), 'B': (['AB1'], SEL4)},
    'AB1g': {'A': (['AB1g'], SEL4)},
    'AB1e': {'A': (['AB1e'], SEL4)},
    'AB2': {'A': (['AB2', 'AB2s'], SEL8), 'B': (['AB2'], SEL4)},
    'G0Vq': {'A': (['G0Vq', 'G0Vqs'], SEL8)},
    'G0Hq': {'A': (['G0Hq', 'G0Hqs'], SEL8)},
    'G0Vc': {'A': (['G0Vc', 'G0Vcs'], SEL8)},
    'G0Vk': {'A': (['G0Vk', 'G0Vks'], SEL8)},
    'G0Vg': {'A': (['G0Vg', 'G0Vgs'], SEL8)},
    'G0Vx': {'A': (['G0Vx', 'G0Vxs'], SEL8)},
    'G0Vo': {'A': (['G0Vo', 'G0Vos'], SEL8)},
}
REFS = {'B': 'v3', 'B2': 'B23', 'R3': 'R3', 'R4': 'R4'}


def is_floor(site, planner, ref):
    """Planner whose model is identical to the reference model (only seed noise separates them)."""
    return (planner == 'B' and ref == 'B') or (site == 'A' and planner == 'AB2' and ref == 'B2')


def read(site, files):
    paths = [os.path.join(RES, f'grid_{site}_{f}.csv') for f in files]
    if not all(os.path.exists(p) for p in paths):
        return None
    d = pd.concat([pd.read_csv(p) for p in paths])
    if METRIC != 'J':
        d['J'] = d[METRIC]
    return d


def rules(site, p):
    full = 'b-325' if site == 'B' else 'e-10_g40'
    geo = 'b25' if site == 'B' else 'e-50_g160'
    mid = 'b-75' if site == 'B' else 'e-90_g80'
    return {'never': 'none', 'full': full, 'georule': geo, 'threshold': (mid if p >= 0.3 else 'none')}


def decisions(site, ref):
    truth = read(site, [REFS[ref]])
    ev = truth[truth.seed.isin(EVAL)].groupby(KEYS + ['layout']).J.mean()
    sel = {}
    for L, spec in PLAN.items():
        if site not in spec:
            continue
        d = read(site, spec[site][0])
        if d is None:
            continue
        sel[L] = d[d.seed.isin(spec[site][1])].groupby(KEYS + ['layout']).J.mean()
        if spec[site][1] == SEL8:            # the two disjoint four-seed halves, for the seed-variance check
            sel[L + '#1'] = d[d.seed.isin(SEL4)].groupby(KEYS + ['layout']).J.mean()
            sel[L + '#2'] = d[d.seed.isin([8, 9, 10, 11])].groupby(KEYS + ['layout']).J.mean()
    rows = []
    for k, e in ev.groupby(level=[0, 1, 2, 3, 4]):
        e = e.droplevel([0, 1, 2, 3, 4])
        ch = {}
        for L, s in sel.items():
            try:
                sl = s.loc[k]
            except KeyError:
                continue
            ch[f'{L}_deploy'] = sl.idxmin(); ch[f'{L}_place'] = sl.drop('none').idxmin()
        for r, lay in rules(site, k[2]).items():
            ch[r] = lay
        ch['oracle_deploy'] = e.idxmin(); ch['oracle_place'] = e.drop('none').idxmin()
        for strat, lay in ch.items():
            rows.append(dict(zip(KEYS, k), ref=ref, strategy=strat, layout=lay, J=float(e[lay]),
                             regret_deploy=float(e[lay] - e.min()), regret_place=float(e[lay] - e.drop('none').min())))
    return pd.DataFrame(rows)


def paired(df, a, b, metric, seed=0):
    w = df.pivot_table(index=KEYS, columns='strategy', values=metric)
    if a not in w or b not in w:
        return None
    d = (w[a] - w[b]).dropna()
    clusters = [f'{s}|{p}|{q}|{P}' for s, u, p, q, P in d.index]
    cl = pd.Series(d.values).groupby(clusters).mean()
    rng = np.random.default_rng(seed)
    bs = cl.values[rng.integers(0, len(cl), (NB, len(cl)))].mean(1)
    nz = cl[np.abs(cl) > 1e-9]
    p = wilcoxon(nz).pvalue if len(nz) >= 6 else 1.0
    se = float(bs.std())
    lo90, hi90 = np.percentile(bs, 5), np.percentile(bs, 95)
    return dict(eq2=bool(lo90 > -2.0 and hi90 < 2.0), ci90=[float(lo90), float(hi90)], n=int(len(d)), mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p=float(p), se=se, mdd=2.8 * se, win=float((d < -1e-9).mean()), tie=float((np.abs(d) <= 1e-9).mean()))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


D_COMPS = [('G0', 'G0V'), ('G0V', 'G'), ('G0H', 'G'), ('G0F', 'G'), ('G', 'C'), ('C', 'B'), ('AB2', 'B'),
           ('AB1', 'B'), ('AB1', 'C'), ('AB1g', 'B'), ('AB1e', 'B'),
           ('G', 'never'), ('C', 'never'), ('B', 'never'), ('G', 'threshold'), ('C', 'threshold'), ('B', 'threshold'),
           ('AB1', 'threshold'), ('G0H', 'threshold')]
P_COMPS = [('G0', 'G0V'), ('G0V', 'G'), ('G0H', 'G'), ('G0F', 'G'), ('G', 'C'), ('C', 'B'), ('AB2', 'B'), ('AB1', 'B'),
           ('B', 'georule'), ('G0', 'georule'), ('B', 'full')]
RULES = ('never', 'threshold', 'georule', 'full')
RULE_OF = {'never': 'deploy', 'threshold': 'deploy', 'georule': 'place', 'full': 'place'}   # the problem each rule answers
# decomposition of the queue state, reported separately from the Holm families above (unadjusted p):
# C -> B-b adds the downstream queue to the counts model, G0 obs. -> G0 obs. + queue and G0 v/c 0.9 -> + queue
# add it to the generic-signal models, and G0 obs. + queue -> G then isolates the cycle length given the queue
X_COMPS = [('C', 'AB2'), ('G0V', 'G0Vq'), ('G0H', 'G0Hq'), ('G0Vq', 'G'), ('G0Hq', 'G')]
# the generic-signal models with the downstream queue against the rules (also outside the Holm families)
X_COMPS += [('G0V', 'threshold'), ('G0Vq', 'never'), ('G0Hq', 'never'), ('G0Vq', 'threshold'), ('G0Hq', 'threshold'), ('G0Vq', 'georule'), ('G0Hq', 'georule')]
# the downstream queue estimated without calibration, from the movement counts (G0Vc) or from speed x density at the
# edge of the footage (G0Vk), against the planner without a queue, the calibrated queue, G and the rules
X_COMPS += [('G0V', 'G0Vc'), ('G0V', 'G0Vk'), ('G0Vc', 'G0Vq'), ('G0Vk', 'G0Vq'), ('G0Vc', 'G'), ('G0Vk', 'G'),
            ('G0Vc', 'never'), ('G0Vk', 'never'), ('G0Vc', 'threshold'), ('G0Vk', 'threshold'), ('G0Vc', 'georule'), ('G0Vk', 'georule')]
# the counts-derived green placed in a generic 120 s downstream cycle (no observed timing at all)
X_COMPS += [('G0V', 'G0Vg'), ('G0Vg', 'G0Vc'), ('G0Vg', 'G0Vq'), ('G0Vg', 'G'), ('G0Vg', 'never'), ('G0Vg', 'threshold'), ('G0Vg', 'georule')]
# controls: a generic downstream green (27 s in 120 s, no counts) and the counts green at a 60 s offset
X_COMPS += [('G0V', 'G0Vx'), ('G0Vx', 'G0Vg'), ('G0Vx', 'G0Vq'), ('G0Vx', 'threshold'), ('G0Vx', 'georule'), ('G0Vx', 'never'),
            ('G0V', 'G0Vo'), ('G0Vo', 'G0Vg'), ('G0Vo', 'G0Vq'), ('G0Vo', 'threshold'), ('G0Vo', 'georule'), ('G0Vo', 'never')]


def summarise(D):
    out = {}
    for ref in REFS:
        for site in ('A', 'B'):
            d = D[(D.ref == ref) & (D.site == site)]
            tab = d[~d.strategy.str.contains('#')].groupby('strategy')[['J', 'regret_deploy', 'regret_place']].mean()
            comps = {}
            for prob, lst, metric in (('deploy', D_COMPS, 'regret_deploy'), ('place', P_COMPS, 'regret_place')):
                for a, b in lst:
                    aa = a if a in RULES else f'{a}_{prob}'
                    bb = b if b in RULES else f'{b}_{prob}'
                    r = paired(d, aa, bb, metric)
                    if r is not None:
                        r['floor'] = is_floor(site, a, ref) or is_floor(site, b, ref)
                        comps[f'{prob}:{a}-{b}'] = r
            adj = holm(np.array([v['p'] for v in comps.values()]))
            for k, x in zip(comps, adj):
                comps[k]['p_holm'] = float(x)
            extra = {}
            for prob, metric in (('deploy', 'regret_deploy'), ('place', 'regret_place')):
                for a, b in X_COMPS:
                    if RULE_OF.get(b, prob) != prob:
                        continue
                    aa = a if a in RULES else f'{a}_{prob}'
                    bb = b if b in RULES else f'{b}_{prob}'
                    r = paired(d, aa, bb, metric)
                    if r is not None:
                        r['floor'] = is_floor(site, a, ref) or is_floor(site, b, ref)
                        extra[f'{prob}:{a}-{b}'] = r
            halves = {}
            for prob, lst, metric in (('deploy', D_COMPS + X_COMPS, 'regret_deploy'), ('place', P_COMPS + X_COMPS, 'regret_place')):
                for a, b in lst:
                    if RULE_OF.get(b, prob) != prob:
                        continue
                    for h in ('#1', '#2'):
                        aa = a if a in RULES else f'{a}{h}_{prob}'
                        bb = b if b in RULES else f'{b}{h}_{prob}'
                        if aa not in d.strategy.values and a not in RULES:
                            aa = f'{a}_{prob}'
                        if bb not in d.strategy.values and b not in RULES:
                            bb = f'{b}_{prob}'
                        r = paired(d, aa, bb, metric)
                        if r is not None:
                            halves[f'{prob}:{a}-{b}{h}'] = dict(mean=r['mean'], lo=r['lo'], hi=r['hi'], p=r['p'])
            agree = {}
            for L in PLAN:
                for prob in ('deploy', 'place'):
                    x1 = d[d.strategy == f'{L}#1_{prob}'].set_index(KEYS).layout
                    x2 = d[d.strategy == f'{L}#2_{prob}'].set_index(KEYS).layout
                    if len(x1) and len(x2):
                        agree[f'{L}_{prob}'] = float((x1 == x2.reindex(x1.index)).mean())
            w = d.pivot_table(index=KEYS, columns='strategy', values='layout', aggfunc='first')
            dep = {s.split('_')[0]: float((w[s] != 'none').mean()) for s in w if s.endswith('_deploy') and '#' not in s}
            out[f'{ref}|{site}'] = dict(table=tab.round(3).to_dict(orient='index'), paired=comps, extra=extra, halves=halves, half_agreement=agree, deploy_rate=dep,
                                        n_comparisons=len(comps), oracle_J=float(d[d.strategy == 'oracle_deploy'].J.mean()))
    return out


def decomposition(D):
    """Extra cost of the no-queue planner (B-q): metrics of its choice minus those of the oracle choice,
    over the units in which the two differ (evaluation seeds of the reference)."""
    out = {}
    for site in ('A', 'B'):
        for ref, f in REFS.items():
            g = read(site, [f])
            g = g[g.seed.isin(EVAL)].groupby(['unit', 'p', 'q', 'P', 'layout'])[
                ['J', 'delay', 'teth', 'd_hv', 'd_cav', 'thr', 'n_unfinished']].mean()
            d = D[(D.site == site) & (D.ref == ref) & (D.p <= 0.6 + 1e-9)]
            a = d[d.strategy == 'AB1_deploy'].set_index(['unit', 'p', 'q', 'P']).layout
            b = d[d.strategy == 'oracle_deploy'].set_index(['unit', 'p', 'q', 'P']).layout
            diff = [g.loc[k + (a[k],)] - g.loc[k + (b[k],)] for k in a.index if a[k] != b[k]]
            if diff:
                x = pd.DataFrame(diff).mean()
                out[f'{ref}|{site}'] = dict(n=len(diff), **{c: float(v) for c, v in x.items()})
    return out


if __name__ == '__main__':
    D = pd.concat([decisions(s, r) for s in ('A', 'B') for r in REFS])
    tag = '' if METRIC == 'J' else '_' + METRIC
    D.to_csv(os.path.join(RES, f'voi_decisions{tag}.csv'), index=False)
    S = summarise(D[D.p <= 0.6 + 1e-9])
    S['decomposition'] = decomposition(D)
    ext = D[(D.p > 0.6 + 1e-9) & (D.ref == 'B')]
    S['ext'] = ext[['site', 'unit', 'p', 'strategy', 'layout', 'J', 'regret_deploy']].to_dict(orient='records')
    json.dump(S, open(os.path.join(RES, f'voi_summary{tag}.json'), 'w'), indent=1)
    for key in [k for k in S if '|' in k and isinstance(S[k], dict) and 'table' in S[k]]:
        T = pd.DataFrame(S[key]['table']).T
        print('=====', key, 'oracle', round(S[key]['oracle_J'], 1), 'comparisons', S[key]['n_comparisons'])
        print(T[['regret_deploy', 'regret_place']].round(2).T.to_string())
        for k, v in S[key]['paired'].items():
            print(f"  {k:26s} {v['mean']:7.2f} [{v['lo']:6.2f},{v['hi']:6.2f}] holm={v['p_holm']:.3g} mdd={v['mdd']:.2f}{' floor' if v['floor'] else ''}")
        for k, v in S[key]['extra'].items():
            print(f"  extra {k:20s} {v['mean']:7.2f} [{v['lo']:6.2f},{v['hi']:6.2f}] p={v['p']:.3g} eq2={v['eq2']}{' floor' if v['floor'] else ''}")
    print(json.dumps(S['decomposition'], indent=0))
