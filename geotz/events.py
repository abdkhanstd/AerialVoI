"""Lane-change events from the aerial trajectories and whether they disrupt the new follower.

An event is the first sample of a vehicle in a new lane (within one lane group). The new follower
is the vehicle directly behind it in the target lane at that time. The event is disruptive when,
within 3 s, that follower has TTC < 3 s or a time gap < 0.5 s to the lane changer.
Output: data/lc_events.csv with site, axis, recording, station, flag and local traffic state.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
HORIZON, TTC_THR, GAP_THR = 3.0, 3.0, 0.5


def _events(d, key, lane, tcol, scol, vcol, lencol, dt, groups=None):
    d = d.sort_values(key + [tcol]).reset_index(drop=True)
    d['pl'] = d.groupby(key).lane.shift() if lane == 'lane' else d.groupby(key)[lane].shift()
    if groups is not None:
        same = np.zeros(len(d), bool)
        for g in groups:
            same |= d[lane].isin(g).values & d.pl.isin(g).values
    else:
        same = d.pl.notna().values
    ev = d[same & (d.pl != d[lane]).values]
    # snapshot index by (scene, time, lane)
    scene = [k for k in key if k not in ('id', 'tid')]
    snap = d.set_index(scene + [tcol, lane]).sort_index()
    out = []
    for _, e in ev.iterrows():
        sc = tuple(e[k] for k in scene)
        t0 = e[tcol]; L = e[lane]
        try:
            q = snap.loc[sc + (t0, L)]
        except KeyError:
            continue
        if isinstance(q, pd.Series):
            q = q.to_frame().T
        behind = q[q[scol] < e[scol] - 0.1]
        if len(behind) == 0:
            out.append((e, False, np.nan, 0)); continue
        f = behind[scol].idxmax() if False else behind.sort_values(scol).iloc[-1]
        fid = f['vid']
        # follow both vehicles for HORIZON s
        flag = False; min_ttc = np.inf
        for k in range(0, int(round(HORIZON / dt)) + 1):
            t = round(t0 + k * dt, 3)
            try:
                qq = snap.loc[sc + (t, L)]
            except KeyError:
                continue
            if isinstance(qq, pd.Series):
                qq = qq.to_frame().T
            a = qq[qq.vid == e['vid']]; b = qq[qq.vid == fid]
            if len(a) == 0 or len(b) == 0:
                continue
            gap = a[scol].iloc[0] - b[scol].iloc[0] - 0.5 * (a[lencol].iloc[0] + b[lencol].iloc[0])
            dv = b[vcol].iloc[0] - a[vcol].iloc[0]
            ttc = gap / dv if dv > 0.1 and gap > 0 else np.inf
            th = gap / max(b[vcol].iloc[0], 0.5)
            min_ttc = min(min_ttc, ttc)
            if ttc < TTC_THR or th < GAP_THR:
                flag = True
        out.append((e, flag, min_ttc, 1))
    rows = [dict(scene=str(tuple(e[k] for k in scene)), s=e[scol], v=e[vcol], disruptive=bool(f), min_ttc=m, has_follower=h)
            for e, f, m, h in out]
    return pd.DataFrame(rows)


def site_b():
    t = pd.read_parquet(os.path.join(DATA, 'siteB_tracks.parquet'),
                        columns=['run_index', 'id', 'time', 's', 'lane_kf', 'speed_kf', 'length_smoothed', 'valid'])
    t = t[t.valid].copy()
    t['time'] = t.time.round(1)
    t['vid'] = t.id
    t = t.rename(columns={'lane_kf': 'lane'})
    ev = _events(t, ['run_index', 'id'], 'lane', 'time', 's', 'speed_kf', 'length_smoothed', 0.1)
    ev['site'] = 'B'; ev['ax'] = 'B'
    ev['rec'] = ev.scene.str.extract(r'\((\d+)').astype(int)
    return ev


def site_a():
    t = pd.read_parquet(os.path.join(DATA, 'siteA_tracks.parquet'))
    t['vid'] = t.tid
    t = t.rename(columns={'lane_id': 'lane'})
    out = []
    import extract_a
    for ax, cfg in extract_a.AX.items():
        q = t[t.ax == ax].copy()
        groups = cfg['inb'] + [cfg['out']]
        ev = _events(q, ['rec', 'vedio-id', 'tid'], 'lane', 't', 's', 'v', 'len', 1.0, groups)
        ev['site'] = 'A'; ev['ax'] = ax
        ev['rec'] = ev.scene.str.extract(r'\((\d+)').astype(int)
        out.append(ev)
    return pd.concat(out)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    A = site_a(); print('A events', len(A), 'disruptive share', A.disruptive.mean().round(3))
    B = site_b(); print('B events', len(B), 'disruptive share', B.disruptive.mean().round(3))
    pd.concat([A, B]).to_csv(os.path.join(DATA, 'lc_events.csv'), index=False)
