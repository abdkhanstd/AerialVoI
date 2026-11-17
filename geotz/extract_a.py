"""Site A (CityArtery UAV data, Dongguan, signalised arterial junction, 1 Hz).

Two station axes along the east-west arterial, one per direction of travel. Each axis is the
median path of the through trajectories of that direction (binned along x, smoothed).
Station s = 0 at the stop line, negative upstream, positive downstream (outbound leg).
Per 10 m cell: exposure, lane changes inside a lane group, time-exposed TTC (< 3 s, same-lane
leader), lane width (spacing of lane centrelines), curvature and distance to the slip-lane
merge/diverge points.
Lane groups (median lane first) were read from the semantic lane ids of the data set.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), 'data_raw', 'cityartery')
OUT = os.path.join(ROOT, 'data'); os.makedirs(OUT, exist_ok=True)
CELL = 10.0
TTC_THR = 3.0
VLEN = {'Car': 4.8, 'Medium Vehicle': 7.0, 'Heavy Vehicle': 11.0, 'Bus': 12.0, 'Motorcycle': 2.0, 'Tuk-Tuk': 3.0}

# lane groups, median lane first (right-hand traffic)
AX = {
    'EB': dict(inb=[[14, 15, 16, 17]], out=[24, 25, 26, 27], slip_div=(17, 18), slip_merge=(13, 27),
               xdir=+1),
    'WB': dict(inb=[[0, 1, 2, 3], [4, 5, 6, 7]], out=[32, 33, 34, 35], slip_div=(3, 8), slip_merge=(23, 35),
               xdir=-1),
}


def load():
    fs = sorted(f for f in os.listdir(RAW) if f.startswith('190b112d__ssh'))
    L = []
    for k, f in enumerate(fs):
        d = pd.read_csv(os.path.join(RAW, f), skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        d['Type'] = d.Type.str.strip()
        d = d[~d.Type.isin(['Pedestrian', 'Bicycle'])].copy()
        d['rec'] = k
        L.append(d)
    d = pd.concat(L)
    d['tid'] = d.rec * 100000 + d['Track ID-new']
    d['t'] = d['Time [s]-new'].round(0)
    d['v'] = d['Speed [km/h]'] / 3.6
    d = d.rename(columns={'x_sumo': 'x', 'y_sumo': 'y'})
    return d[['rec', 'tid', 't', 'x', 'y', 'v', 'lane_id', 'Type', 'vedio-id']].sort_values(['tid', 't'])


def axis_line(d, ax):
    cfg = AX[ax]
    inl = sum(cfg['inb'], []); outl = cfg['out']
    g = d.groupby('tid').lane_id.agg(lambda s: set(s))
    thr = g[g.apply(lambda s: bool(s & set(inl)) and bool(s & set(outl)))].index
    p = d[d.tid.isin(thr) & d.lane_id.isin(inl + outl + [-1])]
    bins = np.arange(p.x.min(), p.x.max() + 5, 5.0)
    m = p.groupby(pd.cut(p.x, bins)).agg(x=('x', 'median'), y=('y', 'median'), n=('x', 'size')).dropna()
    m = m[m.n >= 20]
    xy = m[['x', 'y']].values
    if cfg['xdir'] < 0:
        xy = xy[::-1]
    y_s = pd.Series(xy[:, 1]).rolling(5, center=True, min_periods=1).mean().values
    xy = np.c_[xy[:, 0], y_s]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    # stop line = downstream end of the inbound lanes (5th percentile of their last positions)
    lastin = d[d.lane_id.isin(cfg['inb'][-1])]
    return xy, s, thr


def project(pts, line, s_line):
    a = line[:-1]; b = line[1:]; ab = b - a; L2 = (ab ** 2).sum(1)
    S = np.empty(len(pts)); N = np.empty(len(pts))
    for k0 in range(0, len(pts), 20000):
        p = pts[k0:k0 + 20000]
        ap = p[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / L2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        dist = np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1))
        j = dist.argmin(1); r = np.arange(len(p))
        S[k0:k0 + 20000] = s_line[j] + t[r, j] * np.sqrt(L2[j])
        cross = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
        N[k0:k0 + 20000] = np.sign(cross) * dist[r, j]
    return S, N


def main():
    d = load()
    geo = {}
    cells_all = []
    tracks = []
    for ax, cfg in AX.items():
        xy, s_line, thr = axis_line(d, ax)
        inl = sum(cfg['inb'], []); outl = cfg['out']
        lanes = inl + outl + [cfg['slip_div'][1], cfg['slip_merge'][0]]
        g = d[d.lane_id.isin(lanes + [-1])].copy()
        g = g[g.tid.isin(set(d[d.lane_id.isin(lanes)].tid))]
        S, N = project(g[['x', 'y']].values, xy, s_line)
        # stop line: 98th percentile of station of the last inbound lane group points
        last_in = cfg['inb'][-1]
        s_stop = np.percentile(S[g.lane_id.isin(last_in).values], 99)
        g['s'] = S - s_stop; g['n'] = N; g['ax'] = ax
        g = g[g.lane_id.isin(lanes)]
        # lane centrelines (lateral offset per 5 m) and widths
        cen = {}
        for l in inl + outl:
            q = g[g.lane_id == l]
            b = q.groupby((q.s // 5) * 5).n.median()
            cen[l] = b
        widths = []
        for grp in cfg['inb'] + [outl]:
            for l1, l2 in zip(grp[:-1], grp[1:]):
                w = (cen[l1] - cen[l2]).abs().dropna()
                widths.append(w.rename(f'{l1}-{l2}'))
        W = pd.concat(widths, axis=1).mean(axis=1)
        # curvature of the axis
        hd = np.unwrap(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])))
        kap = np.gradient(hd, s_line[1:])
        kap = pd.Series(kap).rolling(5, center=True, min_periods=1).mean().values
        # landmark stations
        def trans_s(a, b):
            q = g.sort_values(['tid', 't'])
            prev = q.groupby('tid').lane_id.shift()
            m = (prev == a) & (q.lane_id == b)
            return float(np.median(q.s[m])) if m.sum() else np.nan
        lm = dict(stop=0.0, slip_div=trans_s(*cfg['slip_div']), slip_merge=trans_s(*cfg['slip_merge']))
        if len(cfg['inb']) > 1:
            lm['lane_add'] = float(np.median([trans_s(a, b) for a, b in zip(cfg['inb'][0][:3], cfg['inb'][1][1:4])]))
        # lane changes within a lane group (median-lane index 0)
        g = g.sort_values(['tid', 't'])
        g['pl'] = g.groupby('tid').lane_id.shift()
        same = np.zeros(len(g), bool)
        for grp in cfg['inb'] + [outl]:
            same |= g.lane_id.isin(grp).values & g.pl.isin(grp).values
        g['lc'] = same & (g.pl != g.lane_id).values
        g['lc_dl'] = g.lc & (g.lane_id.isin([grp[0] for grp in cfg['inb'] + [outl]]) | g.pl.isin([grp[0] for grp in cfg['inb'] + [outl]]))
        # TTC with same-lane leader at the same second
        g['len'] = g.Type.map(VLEN).fillna(4.8)
        g = g.sort_values(['rec', 'vedio-id', 't', 'lane_id', 's'])
        k = g.groupby(['rec', 'vedio-id', 't', 'lane_id'])
        g['s_lead'] = k.s.shift(-1); g['v_lead'] = k.v.shift(-1); g['len_lead'] = k.len.shift(-1)
        gap = g.s_lead - g.s - 0.5 * (g.len + g.len_lead)
        dv = g.v - g.v_lead
        g['ttc'] = np.where((dv > 0.1) & (gap > 0), gap / dv, np.inf)
        tracks.append(g)
        bins = np.arange(np.floor(g.s.min() / CELL) * CELL, g.s.max() + CELL, CELL)
        g['cell_s'] = pd.cut(g.s, bins, labels=bins[:-1]).astype(float)
        c = g.groupby(['rec', 'cell_s']).agg(vehsec=('s', 'size'), n_lc=('lc', 'sum'), n_lc_dl=('lc_dl', 'sum'),
                                              tet=('ttc', lambda x: (x < TTC_THR).sum()), speed=('v', 'mean')).reset_index()
        c['ax'] = ax
        c['s_mid'] = c.cell_s + CELL / 2
        c['width'] = np.interp(c.s_mid, W.index.values + 2.5, W.values)
        c['kappa'] = np.interp(c.s_mid + s_stop, s_line[1:], np.abs(kap))
        cells_all.append(c)
        geo[ax] = dict(axis_xy=xy.tolist(), s_stop=float(s_stop), landmarks=lm,
                       width=dict(s=(W.index.values + 2.5).tolist(), w=W.values.tolist()),
                       lane_offsets={int(l): dict(s=(cen[l].index.values + 2.5).tolist(), n=cen[l].values.tolist()) for l in cen},
                       s_range=[float(g.s.min()), float(g.s.max())])
        print(ax, 'stop at', round(s_stop, 1), 'landmarks', {k: round(v, 1) for k, v in lm.items()},
              's range', round(g.s.min()), round(g.s.max()), 'median width', round(W.median(), 2))
    cells = pd.concat(cells_all)
    cells.to_csv(os.path.join(OUT, 'siteA_cells.csv'), index=False)
    pd.concat(tracks).to_parquet(os.path.join(OUT, 'siteA_tracks.parquet'))
    json.dump(geo, open(os.path.join(OUT, 'siteA_geom.json'), 'w'))
    s = cells.groupby(['ax', 'cell_s'])[['vehsec', 'n_lc', 'n_lc_dl', 'tet']].sum()
    s['lc_rate'] = 1000 * s.n_lc / s.vehsec; s['tet_frac'] = s.tet / s.vehsec
    print(s[s.vehsec > 200].round(3).to_string())


if __name__ == '__main__':
    main()
