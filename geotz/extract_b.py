"""Site B (TGSIM I-90/I-94 stationary, helicopter 8K video, runs 2/4/6).

Builds a common station axis for the three runs from the provided lane centrelines
(s = 0 at the end of the weaving section, negative upstream), then computes per
25 m cell: exposure (vehicle-seconds), lane changes by lane pair, conflict events
(onset of an episode with time-to-collision below 1.5 s to the same-lane leader;
the first and last 3 s of every track are ignored because the smoother's
acceleration and speed are unreliable there)
and geometry (curvature, lane width, distance to the weaving gores).
Outputs data/siteB_cells.csv, data/siteB_geom.json, data/siteB_tracks.parquet.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), 'data_raw', 'tgsim_i9094s')
OUT = os.path.join(ROOT, 'data'); os.makedirs(OUT, exist_ok=True)
RUNS = [2, 4, 6]
LANES = [10, 11, 12, 13, 14, 15]          # 10 = median (left) lane, 15 = right lane
CELL = 25.0
TTC_THR = 1.5           # s, near-miss threshold
EDGE = 30               # samples (3 s) dropped at both track ends (Kalman warm-up)


def polyline_arclen(xy):
    seg = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0], np.cumsum(seg)])


def project(xy_pts, line, s_line):
    """Project points on a polyline, return station and signed lateral offset."""
    a = line[:-1]; b = line[1:]; ab = b - a; L2 = (ab ** 2).sum(1)
    best_s = np.empty(len(xy_pts)); best_n = np.empty(len(xy_pts))
    for k0 in range(0, len(xy_pts), 20000):
        p = xy_pts[k0:k0 + 20000]
        ap = p[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / L2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        dist = np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1))
        j = dist.argmin(1); r = np.arange(len(p))
        best_s[k0:k0 + 20000] = s_line[j] + t[r, j] * np.sqrt(L2[j])
        cross = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
        best_n[k0:k0 + 20000] = np.sign(cross) * dist[r, j]
    return best_s, best_n


def run_geometry(run):
    c = pd.read_csv(os.path.join(RAW, f'I-90-stationary-Run_{run}-geometry-with-ramps.csv'))
    lanes = {}
    for l in LANES:
        cc = c[[f'x_{l}', f'y_{l}', f'ramp_{l}']].dropna().values
        lanes[l] = cc
    # reference line = mean of lanes 12 and 13 resampled by index fraction
    ref_lanes = [lanes[12][:, :2], lanes[13][:, :2]]
    n = 400
    def resample(xy, n):
        s = polyline_arclen(xy); t = np.linspace(0, s[-1], n)
        return np.c_[np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]
    ref = np.mean([resample(x, n) for x in ref_lanes], axis=0)
    s_ref = polyline_arclen(ref)
    # landmark: end of weaving flag (3 -> 0) on lanes 12 and 13
    ends = []
    for l in (12, 13):
        f = lanes[l][:, 2]; i = np.where((f[:-1] == 3) & (f[1:] == 0))[0][0]
        s_e, _ = project(lanes[l][i:i + 1, :2], ref, s_ref); ends.append(s_e[0])
    s0 = float(np.mean(ends))
    return lanes, ref, s_ref, s0


def main():
    d = pd.read_parquet(os.path.join(RAW, 'all.parquet'))
    d = d[d.run_index.isin(RUNS)].copy()
    geo = {}
    parts = []
    for run in RUNS:
        lanes, ref, s_ref, s0 = run_geometry(run)
        g = d[d.run_index == run].copy()
        s, nn = project(g[['xloc_kf', 'yloc_kf']].values, ref, s_ref)
        g['s'] = s - s0; g['n'] = nn
        parts.append(g)
        # lane geometry on the common axis: width and curvature of reference
        lane_s = {}
        for l in LANES:
            ls, ln = project(lanes[l][:, :2], ref, s_ref)
            lane_s[l] = (ls - s0, ln)
        # curvature of ref (smoothed heading derivative)
        hd = np.unwrap(np.arctan2(np.diff(ref[:, 1]), np.diff(ref[:, 0])))
        ds = np.diff(s_ref)
        k = np.gradient(hd, s_ref[1:])
        k = pd.Series(k).rolling(15, center=True, min_periods=1).mean().values
        geo[run] = dict(s0=s0, s_ref=(s_ref[1:] - s0).tolist(), kappa=k.tolist(),
                        lane_n={l: [lane_s[l][0].tolist()[::5], lane_s[l][1].tolist()[::5]] for l in LANES})
    d = pd.concat(parts).sort_values(['run_index', 'id', 'time'])
    d['pl'] = d.groupby(['run_index', 'id']).lane_kf.shift()
    d['lc'] = (d.pl.notna()) & (d.pl != d.lane_kf)
    # same-lane leader at each time step
    d = d.sort_values(['run_index', 'time', 'lane_kf', 's'])
    grp = d.groupby(['run_index', 'time', 'lane_kf'])
    d['s_lead'] = grp.s.shift(-1); d['v_lead'] = grp.speed_kf.shift(-1); d['len_lead'] = grp.length_smoothed.shift(-1)
    gap = d.s_lead - d.s - d.len_lead
    dv = d.speed_kf - d.v_lead
    d['ttc'] = np.where((dv > 0.1) & (gap > 0), gap / dv, np.inf)
    d['headway'] = np.where(d.speed_kf > 1, gap / d.speed_kf.clip(lower=0.1), np.nan)
    # conflict onset events (first step of each episode per vehicle)
    d = d.sort_values(['run_index', 'id', 'time'])
    k = d.groupby(['run_index', 'id']).cumcount(); kr = d.groupby(['run_index', 'id']).cumcount(ascending=False)
    d['valid'] = (k >= EDGE) & (kr >= EDGE)
    flag = (d.ttc < TTC_THR) & d.valid
    prev = flag.groupby([d.run_index, d.id]).shift(fill_value=False)
    d['conf'] = flag & ~prev
    d.to_parquet(os.path.join(OUT, 'siteB_tracks.parquet'))
    # cells
    lo, hi = -325.0, 225.0                    # covered by all three runs (checked below)
    bins = np.arange(lo, hi + CELL, CELL)
    d['cell'] = pd.cut(d.s, bins, labels=False)
    dt = 0.1
    rows = []
    for (run, cell), g in d[d.cell.notna() & d.valid].groupby(['run_index', 'cell']):
        row = dict(run=run, cell=int(cell), s_mid=float(bins[int(cell)] + CELL / 2),
                   vehsec=len(g) * dt, n_lc=int(g.lc.sum()), n_conf=int(g.conf.sum()),
                   speed=float(g.speed_kf.mean()), hw_p10=float(np.nanpercentile(g.headway, 10)))
        for l in LANES[:-1]:
            row[f'lc_{l}_{l+1}'] = int(((g.pl == l) & (g.lane_kf == l + 1)).sum() + ((g.pl == l + 1) & (g.lane_kf == l)).sum())
        rows.append(row)
    cells = pd.DataFrame(rows)
    cov = d.groupby('run_index').s.agg(['min', 'max']); print(cov)
    cells.to_csv(os.path.join(OUT, 'siteB_cells.csv'), index=False)
    json.dump(geo, open(os.path.join(OUT, 'siteB_geom.json'), 'w'))
    print(cells.groupby('cell')[['vehsec', 'n_lc', 'n_conf']].sum().assign(
        lc_rate=lambda x: 1000 * x.n_lc / x.vehsec, conf_rate=lambda x: 1000 * x.n_conf / x.vehsec).round(2).to_string())


if __name__ == '__main__':
    main()
