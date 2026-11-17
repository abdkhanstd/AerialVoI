"""Figures of the value-of-information study (four reference models)."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_data import save, W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
REFS = ['B', 'B2', 'R3', 'R4']
RMK = {'B': 'o', 'B2': 's', 'R3': '^', 'R4': 'D'}
RLAB = {'B': 'ref. B', 'B2': "ref. B$'$", 'R3': 'ref. R3', 'R4': 'ref. R4'}
GREY, BLUE = '#8c8c8c', '#2171b5'


def panel(ax, S, site, prob, strategies, labels, title):
    y = np.arange(len(strategies))
    def key(s):
        return s
    vals = np.array([[S[f'{r}|{site}']['table'][key(s)][prob] for r in REFS] for s in strategies])
    cols = [GREY if not s.split('_')[0] in ('G0', 'G', 'C', 'B', 'AB1', 'AB2') else BLUE for s in strategies]
    ax.barh(y, vals.mean(1), color=cols, alpha=0.55, height=0.62)
    for j, r in enumerate(REFS):
        ax.plot(np.maximum(vals[:, j], 0.05), y + (j - 1.5) * 0.12, RMK[r], ms=2.6, color='k', mfc='white' if j % 2 else 'k',
                mew=0.5, label=RLAB[r], ls='none')
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xscale('symlog', linthresh=2); ax.set_xlim(0, 160)
    ax.set_xlabel('regret (s/veh)'); ax.set_title(title)


def fig_regret():
    S = json.load(open(os.path.join(RES, 'voi_summary.json')))
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45))
    dep = ['never', 'threshold', 'georule', 'full', 'G0_deploy', 'G_deploy', 'C_deploy', 'B_deploy', 'AB1_deploy', 'AB2_deploy']
    dlab = ['Never', 'Threshold', 'Design rule', 'Longest DL', 'G0', 'G', 'C', 'B', 'B-q', 'B-b']
    pl = ['georule', 'full', 'G0_place', 'G_place', 'C_place', 'B_place', 'AB1_place', 'AB2_place']
    plab = ['Design rule', 'Longest DL', 'G0', 'G', 'C', 'B', 'B-q', 'B-b']
    panel(axs[0], S, 'A', 'regret_deploy', dep, dlab, '(a) Site A, deploy and place')
    panel(axs[1], S, 'A', 'regret_place', pl, plab, '(b) Site A, place only')
    panel(axs[2], S, 'B', 'regret_deploy', dep, dlab, '(c) Site B, deploy and place')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.85)
    save(fig, 'fig_regret.pdf')


def fig_levels():
    S = json.load(open(os.path.join(RES, 'voi_summary.json')))
    fid = {L: json.load(open(os.path.join(RES, f'fidelity_{L}.json'))) for L in ['G0', 'G', 'C', 'AB1', 'AB2', 'B', 'R3', 'R4']}
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.95))
    lv = ['G0', 'G', 'C', 'AB1', 'AB2', 'B']
    lab = ['G0', 'G', 'C', 'B-q', 'B-b', 'B']
    x = np.arange(len(lv))
    for k, (site, key) in enumerate((('A', 'msre_A'), ('B', 'msre_B'))):
        axs[0].bar(x + (k - 0.5) * 0.38, [fid[L][key] for L in lv], width=0.36, color=['#2471a3', '#c0392b'][k], label=f'Site {site}')
    axs[0].set_xticks(x); axs[0].set_xticklabels(lab, fontsize=6); axs[0].set_ylabel('MSRE vs aerial data')
    axs[0].set_title('(a) Fidelity of the models'); axs[0].legend(frameon=False, fontsize=6)
    # (b) deployment share at site A vs oracle share in each reference
    D = pd.read_csv(os.path.join(RES, 'voi_decisions.csv'))
    D = D[(D.p <= 0.6 + 1e-9) & (D.site == 'A')]
    pl = ['G0', 'G', 'C', 'AB1', 'AB2', 'B']
    share = [float((D[(D.ref == 'B') & (D.strategy == f'{L}_deploy')].layout != 'none').mean()) for L in pl]
    axs[1].bar(np.arange(len(pl)), share, color=BLUE, alpha=0.6)
    for j, r in enumerate(REFS):
        o = float((D[(D.ref == r) & (D.strategy == 'oracle_deploy')].layout != 'none').mean())
        axs[1].axhline(o, color='k', lw=0.6, ls=['-', '--', ':', '-.'][j], label=f'oracle, {RLAB[r]}')
    axs[1].set_xticks(np.arange(len(pl))); axs[1].set_xticklabels(lab, fontsize=6)
    axs[1].set_ylim(0, 1.15); axs[1].set_ylabel('share of units with a DL'); axs[1].set_title('(b) Site A, deployment'); axs[1].legend(frameon=False, fontsize=5.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    # (c) value of each step at site A (place only), per reference
    steps = [('place:G0-G0V', 'demand'), ('place:G0V-G', 'cycle'), ('place:G-C', 'counts'),
             ('place:C-B', 'traject.'), ('place:AB2-B', 'behav.')]
    for i, (key, nm) in enumerate(steps):
        for j, r in enumerate(REFS):
            v = S[f'{r}|A']['paired'][key]
            axs[2].errorbar(v['mean'], i + (j - 1.5) * 0.15, xerr=[[v['mean'] - v['lo']], [v['hi'] - v['mean']]], fmt=RMK[r],
                            ms=2.6, color='k', mfc='white' if j % 2 else 'k', lw=0.5, mew=0.5)
    axs[2].axvline(0, color='k', lw=0.5)
    axs[2].set_yticks(range(len(steps))); axs[2].set_yticklabels([s[1] for s in steps], fontsize=6); axs[2].invert_yaxis()
    axs[2].set_xlabel('value for placement (s/veh, 95% CI)'); axs[2].set_title('(c) Site A, value of each step')
    fig.subplots_adjust(wspace=0.62)
    save(fig, 'fig_levels.pdf')


def fig_tradeoff():
    rows = []
    dec = pd.read_csv(os.path.join(RES, 'voi_decisions.csv'))
    for site in 'AB':
        g = pd.read_csv(os.path.join(RES, f'grid_{site}_v3.csv'))
        g = g[(g.seed >= 4) & (g.q == 0.85) & (g.P == 4)]
        dd = dec[(dec.site == site) & (dec.ref == 'B') & (dec.strategy == 'oracle_place') & (dec.q == 0.85) & (dec.P == 4)]
        for (p, unit), d in g.groupby(['p', 'unit']):
            sel = dd[(dd.p == p) & (dd.unit == unit)]
            if not len(sel):
                continue
            lay = sel.layout.iloc[0]
            n = d[d.layout == 'none']; b = d[d.layout == lay]
            rows.append(dict(site=site, unit=unit, p=p, cav_none=n.d_cav.mean(), cav_dl=b.d_cav.mean(), hv_none=n.d_hv.mean(), hv_dl=b.d_hv.mean()))
    R = pd.DataFrame(rows).groupby(['site', 'p']).mean(numeric_only=True).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.62, 1.8))
    for ax, site in zip(axs, 'AB'):
        r = R[R.site == site]
        ax.plot(r.p, r.cav_dl - r.cav_none, 'o-', color='#c0392b', ms=2.5, label='CAV')
        ax.plot(r.p, r.hv_dl - r.hv_none, 's--', color='#4d4d4d', ms=2.5, label='HV')
        ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('CAV penetration $p$'); ax.set_title(f'Site {site}')
    axs[0].set_ylabel('delay with best DL minus\nwithout DL (s)'); axs[0].legend(frameon=False, fontsize=6)
    fig.subplots_adjust(wspace=0.4)
    save(fig, 'fig_tradeoff.pdf')
    R.to_csv(os.path.join(RES, 'tradeoff.csv'), index=False)


if __name__ == '__main__':
    fig_regret(); fig_levels(); fig_tradeoff(); print('ok')
