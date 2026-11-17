"""Downstream WB queue of site A estimated without trajectory calibration.

The WB exit leg is queued at the downstream edge of the footage (mean speed about 3 m/s against a
16.7 m/s limit), so the downstream bottleneck discharges at its capacity, and the exit flow
estimates that capacity. With a downstream signal of cycle C, n lanes and a saturation flow s per lane,
the green time is g = C * q / (n * s). Two estimates of the exit flow q are used:
  counts:  the flow into the WB exit from the movement counts (approach volumes times the shares
           of the WB exit destination),
  edge:    mean density times mean speed on the observed WB exit leg (video, no tracking with lane
           assignment needed).
The counts green is given for the observed 160 s cycle and for the generic 120 s cycle (no observed timing).
Also reports the tracked flow across stations of the WB exit leg and where the exit-lane tracks end,
which locate the gap between the two estimates at the ragged edge of the footage.
Writes results/cheap_queue.json.  Usage: python geotz/cheap_queue.py
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM = json.load(open(os.path.join(ROOT, 'data', 'siteA_demand.json')))
OBS = json.load(open(os.path.join(ROOT, 'data', 'siteA_targets.json')))
C, N, S = 160.0, 4, 1800.0            # cycle of the downstream signal (s), lanes at the bottleneck, veh/h per lane
C_GEN = 120.0                         # generic cycle of the planners without observed timing

q_counts = sum(DEM['app_vph'][a] * DEM['od_share'][a]['WBo'] for a in DEM['app_vph'])
k = OBS['WB_out']['dens'] / 10.0      # vehicles per metre over all lanes (density is per 10 m cell)
q_edge = 3600.0 * k * OBS['WB_out']['speed']

# coverage of the exit leg in the footage
t = pd.read_parquet(os.path.join(ROOT, 'data', 'siteA_tracks.parquet'), columns=['rec', 'tid', 't', 's', 'ax', 'lane_id'])
dur = t.groupby('rec').t.agg(lambda x: x.max() - x.min()).sum()
w = t[t.ax == 'WB']
g = w.groupby(['rec', 'tid']).s.agg(['min', 'max'])
station_flow = {str(s0): float(3600.0 * ((g['min'] < s0) & (g['max'] >= s0)).sum() / dur) for s0 in (60, 100, 150)}
x = w[(w.s > 40) & w.lane_id.isin([32, 33, 34, 35])].groupby(['rec', 'tid']).agg(smax=('s', 'max'), lane=('lane_id', 'last'))
track_end_median = {str(int(l)): float(v) for l, v in x.groupby('lane').smax.median().items()}

out = dict(cycle=C, lanes=N, sat=S, exit_speed=OBS['WB_out']['speed'],
           q_counts=q_counts, green_counts=C * q_counts / (N * S), green_counts_120=C_GEN * q_counts / (N * S),
           q_edge=q_edge, green_edge=C * q_edge / (N * S), green_calibrated=40.0,
           capacity_edge=N * S * round(C * q_edge / (N * S)) / C,
           station_flow=station_flow, track_end_median=track_end_median)
json.dump(out, open(os.path.join(ROOT, 'results', 'cheap_queue.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
