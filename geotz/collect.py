"""Collect grid results (JSON per run, possibly from several servers) into one table.

Per run and per decision unit (site B: the corridor; site A: each arterial direction EB / WB):
  delay   mean delay (time loss + insertion delay) of trips that start in the measurement window
          and use the unit (site A: trips originating on that approach)
  teth    seconds with TTC < 3 s of human-driven followers in the observed window, per trip
  J       delay + teth (s per vehicle), the pre-registered objective
plus throughput, lane changes, DL utilisation and platoon statistics.
"""
import os, sys, json, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lay_id(site, lay):
    if lay is None:
        return 'none'
    if site == 'B':
        return f'b{int(lay)}'
    return f'e{int(lay["e"])}_g{int(lay["g"])}'


def load(site, dirs):
    keep_lay = {lay_id(site, l) for l in grid.layouts(site)}
    keep_sc = set(grid.scenarios(ext=True))
    rows = []
    seen = set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, '*.json')):
            r = json.load(open(f))
            lid = lay_id(site, r['layout'])
            sc = (r['p'], r['q'], r['P'])
            sd = r['seed'] - 1000
            if lid not in keep_lay or sc not in keep_sc or sd not in grid.SEEDS:
                continue
            k = (lid, sc, sd)
            if k in seen:
                continue
            seen.add(k)
            base = dict(site=site, layout=lid, p=r['p'], q=r['q'], P=r['P'], seed=sd,
                        thr=r['throughput_vph'], lc_total=r['lc_total'], dl_share=r['dl_share'],
                        mean_string=r['mean_string'], frac_in_string=r['frac_in_string'],
                        backlog=r['backlog_vs'], wall=r['wall_s'], n_unfinished=r.get('n_unfinished'),
                        d_cav=(r.get('delay_by_class') or {}).get('CAV'), d_hv=(r.get('delay_by_class') or {}).get('HV'))
            if site == 'B':
                n = max(r['n_trips'], 1)
                teth = sum(r['teth_cells']['B']) / n
                rows.append(dict(base, unit='B', delay=r['delay_mean'], teth=teth, n=r['n_trips'],
                                 lc_win=sum(r['lc_cells']['B'])))
            else:
                for ax in ('EB', 'WB'):
                    n = max(r['n_by_origin'].get(ax, 0), 1)
                    teth = sum(r['teth_cells'][ax]) / n
                    rows.append(dict(base, unit=ax, delay=r['delay_by_origin'].get(ax), teth=teth, n=n,
                                     lc_win=sum(r['lc_cells'][ax])))
    df = pd.DataFrame(rows)
    df['J'] = df.delay + df.teth
    return df


if __name__ == '__main__':
    site, tag = sys.argv[1], sys.argv[2]
    dirs = sys.argv[3:]
    df = load(site, dirs)
    out = os.path.join(ROOT, 'results', f'grid_{site}_{tag}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(site, len(df), 'rows', df.groupby('layout').size().describe().to_dict())
