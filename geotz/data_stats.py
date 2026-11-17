"""Numbers quoted in the data section (written to results/data_stats.json)."""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
out = {}
a = pd.read_parquet(os.path.join(DATA, 'siteA_tracks.parquet'))
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_a
allA = extract_a.load()
out['A_tracks_all'] = int(allA.tid.nunique())
out['A_tracks_axis'] = {ax: int(g.tid.nunique()) for ax, g in a.groupby('ax')}
out['A_rec_durations_s'] = [float(x) for x in allA.groupby('rec').t.max().round(0)]
out['A_total_s'] = float(sum(out['A_rec_durations_s']))
out['A_types'] = allA.groupby('tid').Type.first().value_counts().to_dict()
out['A_points'] = int(len(allA))
b = pd.read_parquet(os.path.join(DATA, 'siteB_tracks.parquet'))
out['B_tracks'] = {int(r): int(g.id.nunique()) for r, g in b.groupby('run_index')}
out['B_durations_s'] = {int(r): float(g.time.max() - g.time.min()) for r, g in b.groupby('run_index')}
out['B_points'] = int(len(b))
out['B_types'] = b.groupby(['run_index', 'id']).type_most_common.first().value_counts().to_dict()
out['B_av_ids'] = int(b[b.av != 'no'].groupby(['run_index', 'id']).ngroups) if 'av' in b else None
ev = pd.read_csv(os.path.join(DATA, 'lc_events.csv'))
out['events'] = {f'{s}_{a_}': dict(n=int(len(g)), disruptive=float(g.disruptive.mean()), with_follower=float(g.has_follower.mean()))
                 for (s, a_), g in ev.groupby(['site', 'ax'])}
cA = pd.read_csv(os.path.join(DATA, 'siteA_cells.csv')); cB = pd.read_csv(os.path.join(DATA, 'siteB_cells.csv'))
out['A_lc_total'] = int(cA.n_lc.sum()); out['B_lc_total'] = int(cB.n_lc.sum())
gA = json.load(open(os.path.join(DATA, 'siteA_geom.json')))
out['A_landmarks'] = {k: v['landmarks'] for k, v in gA.items()}
out['A_width_median'] = {k: float(np.median(v['width']['w'])) for k, v in gA.items()}
gB = json.load(open(os.path.join(DATA, 'siteB_geom.json')))
ws = []
for r, G in gB.items():
    ln = G['lane_n']; lanes = sorted(ln, key=int)
    for x in np.arange(-300, 200, 25):
        ws.append(np.mean(np.abs(np.diff([np.interp(x, ln[l][0], ln[l][1]) for l in lanes]))))
out['B_width_median'] = float(np.median(ws))
out['B_kappa_max'] = float(max(np.max(np.abs(G['kappa'])) for G in gB.values()))
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'data_stats.json'), 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
