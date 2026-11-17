"""One SUMO run of a site with a given dedicated-lane (DL) layout and traffic scenario.

Returns efficiency, lane-change, surrogate-safety and platoon metrics measured after a
warm-up period. libsumo is used when available (server), TraCI otherwise (local tests).
"""
import os, sys, json, time, tempfile, shutil
import xml.etree.ElementTree as ET
import numpy as np
from sumo import SUMO_HOME

try:
    import libsumo as traci
    LIB = True
except ImportError:
    import traci
    LIB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtypes import vtypes_xml, TAU, class_shares

SUMO_BIN = os.path.join(SUMO_HOME, 'bin', 'sumo')
CELL = 25.0
SSM_THR = 3.0
TTC_THR = 3.0       # s, time-exposed TTC threshold (same definition as for the aerial data)


def write_routes(path, routes, shares, truck_share, t_end, hv_lc=None):
    cars = [k for k in ('HV', 'CHV', 'AV', 'CAV') if shares[k] > 1e-9]
    probs = ' '.join(f'{shares[k]:.6f}' for k in cars)
    out = ['<routes>', vtypes_xml(hv_lc),
           f'  <vTypeDistribution id="cars" vTypes="{" ".join(cars)}" probabilities="{probs}"/>']
    for rid, edges, vph, dlane in routes:
        out.append(f'  <route id="{rid}" edges="{" ".join(edges)}"/>')
    for rid, edges, vph, dlane in routes:
        if vph <= 0:
            continue
        out.append(f'  <flow id="{rid}_c" type="cars" route="{rid}" begin="0" end="{t_end}" '
                   f'vehsPerHour="{vph * (1 - truck_share):.1f}" departLane="{dlane}" departSpeed="max"/>')
        if truck_share > 0:
            out.append(f'  <flow id="{rid}_t" type="TRK" route="{rid}" begin="0" end="{t_end}" '
                       f'vehsPerHour="{vph * truck_share:.1f}" departLane="{dlane}" departSpeed="max"/>')
    out.append('</routes>')
    open(path, 'w', newline='\n').write('\n'.join(out) + '\n')


class PlatoonManager:
    """1 Hz supervisor: platoon size cap, mixed connected platoons, DL seeking of CAVs."""

    def __init__(self, edges, max_p, x0map, obs, dl_seek=True, dl_route=None, cell=25.0):
        self.cell = cell
        self.dl_route = dl_route or (lambda r: True)
        self.wants = {}
        self.max_p = max_p
        self.dl = {e['id']: e['dl_idx'] for e in edges if e.get('dl')}
        self.x0 = x0map
        self.dl_seek = dl_seek
        self.cls = {}
        self.stats = dict(cav_vs=0.0, cav_dl_vs=0.0, string_len_sum=0.0, string_cnt=0, cav_in_string=0.0, tet=0.0, tet_h=0.0, spd_L=0.0, n_L=0, spd_R=0.0, n_R=0)
        self.n_left_from = {e['id']: e['nl'] - 3 for e in edges}
        self.obs = obs if isinstance(obs, dict) else {'B': obs}
        self.tet_cells = {a: np.zeros(int(np.ceil((o[1] - o[0]) / cell))) for a, o in self.obs.items()}
        self.vs_cells = {a: np.zeros(len(v)) for a, v in self.tet_cells.items()}
        self.spd_cells = {a: np.zeros(len(v)) for a, v in self.tet_cells.items()}
        self.teth_cells = {a: np.zeros(len(v)) for a, v in self.tet_cells.items()}

    def vclass(self, v):
        c = self.cls.get(v)
        if c is None:
            c = traci.vehicle.getTypeID(v).split('@')[0]
            self.cls[v] = c
        return c

    def step(self, measure):
        ids = traci.vehicle.getIDList()
        rows = []
        for v in ids:
            e = traci.vehicle.getRoadID(v)
            x0 = self.x0.get(e)
            if x0 is None:
                continue
            ax, x0 = x0 if isinstance(x0, tuple) else ('B', x0)
            rows.append((ax, x0 + traci.vehicle.getLanePosition(v), v, self.vclass(v), e))
        rows.sort(reverse=True)                     # downstream first within each axis
        pos = {}
        for ax, x, v, c, e in rows:
            o0, o1 = self.obs.get(ax, (0.0, -1.0))
            inwin = o0 <= x < o1
            lead = traci.vehicle.getLeader(v, 100.0)
            spd = traci.vehicle.getSpeed(v)
            if measure and lead is not None and lead[0] != '':
                dv = spd - traci.vehicle.getSpeed(lead[0])
                if dv > 0.1 and lead[1] / dv < TTC_THR:
                    self.stats['tet'] += 1.0
                    human = c in ('HV', 'CHV', 'TRK')
                    if human:
                        self.stats['tet_h'] += 1.0
                    if inwin:
                        self.tet_cells[ax][int((x - o0) // self.cell)] += 1
                        if human:
                            self.teth_cells[ax][int((x - o0) // self.cell)] += 1
            if measure and inwin:
                self.vs_cells[ax][int((x - o0) // self.cell)] += 1
                self.spd_cells[ax][int((x - o0) // self.cell)] += spd
                side = 'L' if traci.vehicle.getLaneIndex(v) >= self.n_left_from.get(e, 99) else 'R'
                self.stats['spd_' + side] += spd; self.stats['n_' + side] += 1
            if c not in ('CAV', 'CHV'):
                continue
            k = 1
            if lead and lead[0] in pos and lead[1] < 1.5 * max(spd, 2.0) * TAU['CAV_acc'] + 5.0:
                k = pos[lead[0]] + 1
                lc = self.vclass(lead[0])
                if k > self.max_p:
                    k = 1
                    tau = TAU['CAV_acc'] if c == 'CAV' else TAU['CHV_free']
                elif c == 'CAV':
                    tau = TAU['CAV_cacc'] if lc == 'CAV' else TAU['CAV_mixed']
                else:
                    tau = TAU['CHV_conn']
            else:
                tau = TAU['CAV_acc'] if c == 'CAV' else TAU['CHV_free']
            pos[v] = k
            traci.vehicle.setTau(v, tau)
            if c == 'CAV':
                li = traci.vehicle.getLaneIndex(v)
                if measure:
                    self.stats['cav_vs'] += 1.0
                    if e in self.dl and li == self.dl[e]:
                        self.stats['cav_dl_vs'] += 1.0
                w = self.wants.get(v)
                if w is None:
                    w = self.wants[v] = self.dl_route(traci.vehicle.getRouteID(v))
                if self.dl_seek and w and e in self.dl and li != self.dl[e]:
                    traci.vehicle.changeLane(v, self.dl[e], 2.0)
        if measure and pos:
            self.stats['string_cnt'] += sum(1 for k in pos.values() if k == 1)
            self.stats['string_len_sum'] += len(pos)
            self.stats['cav_in_string'] += sum(1 for k in pos.values() if k > 1)


def run(site, b, q, p_cav, max_p, seed, workdir, v_b, t_end=1500.0, warm=300.0, step=0.1, ssm=False,
        dl_seek=False, mix=(0.7, 0.2, 0.1), hv_lc=None, route_kw=None):
    t0 = time.time()
    os.makedirs(workdir, exist_ok=True)
    netdir = os.path.join(os.path.dirname(workdir.rstrip('/\\')), 'nets')
    net, edges = site.build_net(netdir, b, v_b)
    x0map = site.station(edges)
    tmp = tempfile.mkdtemp(dir=workdir)
    rou = os.path.join(tmp, 'r.rou.xml')
    write_routes(rou, site.routes(edges, q, **(route_kw or {})), class_shares(p_cav, mix), site.TRUCK_SHARE, t_end - 100,
                 hv_lc if hv_lc is not None else getattr(site, 'HV_LC', None))
    trip, lcf, ssmf = (os.path.join(tmp, n) for n in ('trip.xml', 'lc.xml', 'ssm.xml'))
    cmd = [SUMO_BIN, '-n', net, '-r', rou, '--step-length', str(step), '--begin', '0', '--end', str(t_end),
           '--seed', str(seed), '--tripinfo-output', trip, '--lanechange-output', lcf,
           '--tripinfo-output.write-unfinished', 'true', '--tripinfo-output.write-undeparted', 'true',
           '--no-step-log', 'true', '--no-warnings', 'true', '--time-to-teleport', '180',
           '--collision.action', 'warn', '--default.action-step-length', str(step),
           '--max-depart-delay', '900']
    statf = os.path.join(tmp, 'stat.xml')
    cmd += ['--statistic-output', statf]
    if ssm:
        cmd += ['--device.ssm.probability', '1', '--device.ssm.measures', 'TTC', '--device.ssm.thresholds', str(SSM_THR),
                '--device.ssm.range', '60', '--device.ssm.file', ssmf, '--device.ssm.trajectories', 'false',
                '--device.ssm.geo', 'false', '--device.ssm.write-positions', 'true']
    if LIB:
        traci.start(cmd)
    else:
        traci.start(cmd, label=f's{os.getpid()}')
    CELLW = getattr(site, 'CELL', CELL)
    pm = PlatoonManager(edges, max_p, x0map, site.OBS_WINDOW, dl_seek, getattr(site, 'dl_route', None), CELLW)
    n_steps_s = int(round(1 / step))
    k = 0
    tts = 0.0
    backlog_s = 0.0
    try:
        while traci.simulation.getTime() < t_end - 1e-6:
            traci.simulationStep()
            k += 1
            t = traci.simulation.getTime()
            if k % n_steps_s == 0:
                meas = t > warm
                pm.step(meas)
                if meas:
                    tts += traci.vehicle.getIDCount()
                    backlog_s += traci.simulation.getMinExpectedNumber() - traci.vehicle.getIDCount()
        n_teleport = -1
    finally:
        traci.close()
    stat = {}
    try:
        root = ET.parse(statf).getroot()
        tp = root.find('teleports'); vh = root.find('vehicles')
        if tp is not None:
            n_teleport = int(tp.get('total', 0))
        if vh is not None:
            stat = {k: int(float(v)) for k, v in vh.attrib.items()}
    except Exception:
        pass
    OBS = site.OBS_WINDOW if isinstance(site.OBS_WINDOW, dict) else {'B': site.OBS_WINDOW}
    # tripinfo
    tt, tl, n_arr = [], [], 0
    by_o = {}
    by_cls = {}
    n_unfin = 0
    for _, el in ET.iterparse(trip):
        if el.tag == 'tripinfo':
            dep = float(el.get('depart'))
            arr = float(el.get('arrival'))
            dd = float(el.get('departDelay'))
            if dep < 0:                       # never inserted: intended departure = end - waiting time
                dep_int = t_end - dd
                tl_ = 0.0
            else:
                dep_int = dep - dd
                tl_ = float(el.get('timeLoss'))
            if arr > warm:
                n_arr += 1
            if warm <= dep_int <= t_end - 300:
                if arr < 0:
                    n_unfin += 1              # unfinished or never inserted: delay accrued until the end of the run
                tt.append(float(el.get('duration')) if dep >= 0 else 0.0)
                dl_ = tl_ + dd
                tl.append(dl_)
                o = el.get('id').split('_')[0]
                by_o.setdefault(o, []).append(dl_)
                c = el.get('vType').split('@')[0]
                by_cls.setdefault(c, []).append(dl_)
            el.clear()
    # lane changes in the observed window, after warm-up
    lc_cells = {a: np.zeros(int(np.ceil((o[1] - o[0]) / CELLW))) for a, o in OBS.items()}; lc_reason = {}
    lc_all = 0
    for _, el in ET.iterparse(lcf):
        if el.tag == 'change' and float(el.get('time')) > warm:
            lane = el.get('from'); e = lane.rsplit('_', 1)[0]
            x0 = x0map.get(e)
            if x0 is None:
                el.clear(); continue
            ax, x0 = x0 if isinstance(x0, tuple) else ('B', x0)
            x = x0 + float(el.get('pos'))
            lc_all += 1
            r = el.get('reason', '').split('|')[0]
            lc_reason[r] = lc_reason.get(r, 0) + 1
            o0, o1 = OBS.get(ax, (0, -1))
            if o0 <= x < o1:
                lc_cells[ax][int((x - o0) // CELLW)] += 1
        el.clear()
    if not os.environ.get("GEOTZ_KEEP"): shutil.rmtree(tmp, ignore_errors=True)
    meas_s = t_end - warm
    st = pm.stats
    return dict(b=b, q=q, p=p_cav, P=max_p, seed=seed, v_b=v_b,
                n_trips=len(tt), tt_mean=float(np.mean(tt)) if tt else None,
                delay_mean=float(np.mean(tl)) if tl else None,
                throughput_vph=n_arr * 3600.0 / meas_s, tts_vs=tts, backlog_vs=backlog_s,
                delay_by_origin={o: float(np.mean(v)) for o, v in by_o.items()},
                n_by_origin={o: len(v) for o, v in by_o.items()}, n_unfinished=n_unfin,
                delay_by_class={c: float(np.mean(v)) for c, v in by_cls.items()},
                lc_total=lc_all, lc_reason=lc_reason, lc_cells={a: v.tolist() for a, v in lc_cells.items()},
                teleports=n_teleport, vehicle_stats=stat,
                spd_left=st['spd_L'] / max(st['n_L'], 1), spd_right=st['spd_R'] / max(st['n_R'], 1),
                tet_vs=st['tet'], teth_vs=st['tet_h'], teth_cells={a: v.tolist() for a, v in pm.teth_cells.items()}, tet_cells={a: v.tolist() for a, v in pm.tet_cells.items()},
                vs_cells={a: v.tolist() for a, v in pm.vs_cells.items()},
                spd_cells={a: v.tolist() for a, v in pm.spd_cells.items()},
                dl_share=st['cav_dl_vs'] / st['cav_vs'] if st['cav_vs'] else None,
                mean_string=st['string_len_sum'] / st['string_cnt'] if st['string_cnt'] else None,
                frac_in_string=st['cav_in_string'] / st['string_len_sum'] if st['string_len_sum'] else None,
                wall_s=time.time() - t0)


if __name__ == '__main__':
    import importlib
    site = importlib.import_module(sys.argv[1])
    b = None if sys.argv[2] == 'none' else float(sys.argv[2])
    r = run(site, b, float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
            sys.argv[7], float(sys.argv[8]), t_end=float(sys.argv[9]) if len(sys.argv) > 9 else 1500.0)
    print(json.dumps({k: v for k, v in r.items() if not k.endswith('cells')}))
