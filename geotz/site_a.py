"""SUMO model of site A (CityArtery UAV data, Dongguan, signalised junction on an arterial).

Geometry comes from the aerial trajectories (extract_a): the east-west arterial follows the two
station axes (eastbound EB, westbound WB) with the measured lane widths and curvature, in the
coordinate frame of the data set. s = 0 at the stop line of the direction, s < 0 approach,
s > 0 exit leg. Observed windows: EB [-130, 210], WB [-160, 175]; beyond them the legs are
extended along their end heading (synthetic, needed for platoon formation and queues).

Features taken from the data:
- four lanes per direction, median lane first: EB in 14-17 / out 24-27, WB in 0-3 then 4-7 / out 32-35,
- WB: a median left-turn lane is added and the curb lane diverges to the right-turn slip at s = -38,
- EB: the right-turn slip diverges at s = -73,
- the right-turn slips of the cross street join the exit legs at s = +87 (EB) and +85 (WB) through
  a 60 m acceleration lane,
- lane use at the stop line (which lanes carry left turns, U-turns and through traffic),
- origin-destination shares and approach volumes, vehicle mix, a 160 s four-phase signal plan
  (one phase per approach) estimated from stop-line discharges.
The CAV dedicated lane (DL) is the median lane of the arterial. In each direction it runs from
the upstream end to s = e (it ends e metres before the stop line, so that left-turning and
U-turning vehicles can use the median lane), and starts again on the exit leg at s = g.
"""
import os, json, subprocess
import numpy as np
from sumo import SUMO_HOME

NETCONVERT = os.path.join(SUMO_HOME, 'bin', 'netconvert')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = json.load(open(os.path.join(ROOT, 'data', 'siteA_geom.json')))
DEM = json.load(open(os.path.join(ROOT, 'data', 'siteA_demand.json')))

S_FAR_IN, S_FAR_OUT = -450.0, 450.0
CELL = 10.0
HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.2)   # calibrated (calib_a)
S_BN = 250.0        # WB exit: downstream bottleneck (next signal of the corridor, outside the footage)
V_BN = 40.0         # green time (s) of the downstream WB signal in a 160 s cycle, calibrated (calib_a)
BN_CYC = 160.0      # cycle (s) of the downstream WB signal
BN_OFF = 0.0        # offset (s) of the downstream WB signal
NOALLRED = os.environ.get('GEOTZ_NOALLRED', '0') == '1'   # drop the 1 s all-red phase netconvert adds to some layouts
DEPART = 'random'   # entry lane of the approach flows, calibrated (calib_a)
OBS_WINDOW = {'EB': (-130.0, 210.0), 'WB': (-160.0, 170.0)}
CAND_E = [-130.0, -110.0, -90.0, -70.0, -50.0, -30.0, -10.0]      # DL end before the stop line
CAND_G = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]  # DL restart on the exit leg
J = (3132.0, 1530.0)
V_ART = 16.7          # 60 km/h arterial limit
V_SLIP = 8.3
ACC_LEN = 60.0
TRUCK_SHARE = 0.129    # medium + heavy vehicles and buses
CYCLE = dict(EB=57, WB=43, NB=25, SB=23, yellow=3)

LM = {ax: GEOM[ax]['landmarks'] for ax in GEOM}


class Axis:
    def __init__(self, ax):
        g = GEOM[ax]
        xy = np.array(g['axis_xy'])
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) - g['s_stop']
        self.xy, self.s = xy, s
        self.t0 = (xy[1] - xy[0]) / np.linalg.norm(xy[1] - xy[0])
        self.t1 = (xy[-1] - xy[-2]) / np.linalg.norm(xy[-1] - xy[-2])
        self.lanes = g['lane_offsets']

    def point(self, s, off=0.0):
        if s < self.s[0]:
            p = self.xy[0] + (s - self.s[0]) * self.t0; t = self.t0
        elif s > self.s[-1]:
            p = self.xy[-1] + (s - self.s[-1]) * self.t1; t = self.t1
        else:
            i = min(np.searchsorted(self.s, s), len(self.s) - 1); i = max(i, 1)
            f = (s - self.s[i - 1]) / (self.s[i] - self.s[i - 1])
            p = self.xy[i - 1] + f * (self.xy[i] - self.xy[i - 1])
            t = (self.xy[i] - self.xy[i - 1]) / np.linalg.norm(self.xy[i] - self.xy[i - 1])
        nl = np.array([-t[1], t[0]])            # unit normal to the left
        return p + off * nl

    def offset(self, lanes, s):
        return float(np.mean([np.interp(s, self.lanes[str(l)]['s'], self.lanes[str(l)]['n']) for l in lanes]))

    def shape(self, s0, s1, lanes, step=10.0):
        ss = np.unique(np.r_[np.arange(s0, s1, step), s1])
        off_lo, off_hi = self.lanes[str(lanes[0])]['s'][0], self.lanes[str(lanes[0])]['s'][-1]
        pts = [self.point(s, self.offset(lanes, min(max(s, off_lo), off_hi))) for s in ss]
        return ' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in pts)


AXES = {ax: Axis(ax) for ax in ('EB', 'WB')}
GROUPS = {  # lanes used for the edge offset (median first)
    'EB_in': [14, 15, 16, 17], 'EB_out': [24, 25, 26, 27],
    'WB_up': [0, 1, 2], 'WB_in': [4, 5, 6, 7], 'WB_out': [32, 33, 34, 35]}
WIDTH = {ax: float(np.median(GEOM[ax]['width']['w'])) for ax in GEOM}


def _pieces(ax, e, g):
    """Edge pieces of one direction: list of dict(id, s0, s1, nl, kind, group)."""
    lm = LM[ax]
    if ax == 'EB':
        inb = sorted({S_FAR_IN, lm['slip_div'], 0.0} | ({e} if e is not None and e < -1 else set()))
    else:
        inb = sorted({S_FAR_IN, lm['lane_add'], 0.0} | ({e} if e is not None and e < -1 else set()))
    m = lm['slip_merge']
    s_exit = 40.0
    outb = sorted({s_exit, m, m + ACC_LEN, S_FAR_OUT} | ({g} if g is not None and g > s_exit + 1 else set())
                  | ({S_BN} if ax == 'WB' else set()))
    P = []
    for a, b in zip(inb[:-1], inb[1:]):
        grp = 'EB_in' if ax == 'EB' else ('WB_up' if b <= lm['lane_add'] + 1e-6 else 'WB_in')
        nl = 4
        dl = (e is not None) and b <= e + 1e-6
        P.append(dict(s0=a, s1=b, nl=nl, kind='in', group=grp, dl=dl))
    for a, b in zip(outb[:-1], outb[1:]):
        nl = 5 if (a >= m - 1e-6 and b <= m + ACC_LEN + 1e-6) else 4
        dl = (g is not None) and a >= g - 1e-6
        P.append(dict(s0=a, s1=b, nl=nl, kind='out', group=f'{ax}_out', dl=dl, bn=(ax == 'WB' and a >= S_BN - 1e-6)))
    for i, p in enumerate(P):
        p['id'] = f'{ax}{p["kind"]}{i}'
        p['dl_idx'] = p['nl'] - 1
        p['axis'] = ax
    return P


def build_net(outdir, layout, v_b=None):
    v_b = V_BN if v_b is None else v_b
    """layout = None (no DL) or dict(e=..., g=...) applied to both directions."""
    os.makedirs(outdir, exist_ok=True)
    e = g = None
    if layout is not None:
        e, g = layout['e'], layout['g']
    tag = ('noDL' if layout is None else f'e{int(e)}_g{int(g)}') + f'_gd{v_b:.0f}' + ('' if BN_CYC == 160.0 else f'c{BN_CYC:.0f}') + ('' if BN_OFF == 0.0 else f'o{BN_OFF:.0f}') + ('_nar' if NOALLRED else '')
    net = os.path.join(outdir, f'A_{tag}.net.xml')
    pieces = {ax: _pieces(ax, e, g) for ax in ('EB', 'WB')}
    edges = []
    nodes = {'J': J}
    nod_type = {'J': 'traffic_light'}
    E = []      # (id, from, to, nl, speed, shape or None, lane allow list, width)
    for ax, P in pieces.items():
        A = AXES[ax]
        for i, p in enumerate(P):
            fr = 'J' if (p['kind'] == 'out' and abs(p['s0'] - 40.0) < 1e-6) else f'{ax}_n{p["s0"]:.0f}'
            to = 'J' if (p['kind'] == 'in' and abs(p['s1']) < 1e-6) else f'{ax}_n{p["s1"]:.0f}'
            lanes_for_shape = GROUPS[p['group']]
            for nm, s in ((fr, p['s0']), (to, p['s1'])):
                if nm != 'J' and nm not in nodes:
                    nodes[nm] = tuple(A.point(s, A.offset(lanes_for_shape, s)))
            shape = A.shape(p['s0'], p['s1'], lanes_for_shape)
            allow = ['custom1' if (p['dl'] and k == p['dl_idx']) else 'passenger truck custom1' for k in range(p['nl'])]
            E.append((p['id'], fr, to, p['nl'], V_ART, shape, allow, WIDTH[ax]))
            if p.get('bn') and fr != 'J' and v_b < 150:
                nod_type[fr] = 'traffic_light'
            edges.append(dict(id=p['id'], axis=ax, x0=p['s0'], s0=p['s0'], x1=p['s1'], nl=p['nl'], dl=p['dl'],
                              dl_idx=p['dl_idx'], kind=p['kind'], group=p['group']))
    # cross street (north and south legs), straight, from the lane extents of the data
    cross = {
        'NBin': ((3143.0, 1230.0), (3143.0, 1500.0)), 'SBout': ((3124.0, 1505.0), (3124.0, 1230.0)),
        'SBin': ((3123.0, 1830.0), (3123.0, 1556.0)), 'NBout': ((3137.0, 1556.0), (3137.0, 1830.0)),
    }
    # split points for the slips on the cross street
    nodes.update({'NBin_far': cross['NBin'][0], 'NBin_div': (3143.0, 1430.0), 'SBout_m': (3124.0, 1440.0),
                  'SBout_far': cross['SBout'][1], 'SBin_far': cross['SBin'][0], 'SBin_div': (3123.0, 1600.0),
                  'NBout_m': (3137.0, 1600.0), 'NBout_far': cross['NBout'][1]})
    W = 3.2
    E += [('NBin0', 'NBin_far', 'NBin_div', 4, V_ART, None, None, W), ('NBin1', 'NBin_div', 'J', 4, V_ART, None, None, W),
          ('SBout0', 'J', 'SBout_m', 4, V_ART, None, None, W), ('SBout1', 'SBout_m', 'SBout_far', 4, V_ART, None, None, W),
          ('SBin0', 'SBin_far', 'SBin_div', 4, V_ART, None, None, W), ('SBin1', 'SBin_div', 'J', 4, V_ART, None, None, W),
          ('NBout0', 'J', 'NBout_m', 4, V_ART, None, None, W), ('NBout1', 'NBout_m', 'NBout_far', 4, V_ART, None, None, W)]
    for c in ('NBin0', 'NBin1', 'SBout0', 'SBout1', 'SBin0', 'SBin1', 'NBout0', 'NBout1'):
        edges.append(dict(id=c, axis=None, x0=None, nl=4, dl=False, dl_idx=3, kind='cross'))
    # slips: EB right -> SBout_m, WB right -> NBout_m, NB right -> EB merge, SB right -> WB merge
    def nid(ax, s):
        return f'{ax}_n{s:.0f}'
    lmE, lmW = LM['EB'], LM['WB']
    E += [('slipEB', nid('EB', lmE['slip_div']), 'SBout_m', 1, V_SLIP, None, None, 3.5),
          ('slipWB', nid('WB', lmW['lane_add']), 'NBout_m', 1, V_SLIP, None, None, 3.5),
          ('slipNB', 'NBin_div', nid('EB', lmE['slip_merge']), 1, V_SLIP, None, None, 3.5),
          ('slipSB', 'SBin_div', nid('WB', lmW['slip_merge']), 1, V_SLIP, None, None, 3.5)]
    for c in ('slipEB', 'slipWB', 'slipNB', 'slipSB'):
        edges.append(dict(id=c, axis=None, x0=None, nl=1, dl=False, dl_idx=0, kind='slip'))
    if not os.path.exists(net):
        _write_and_convert(outdir, tag, net, nodes, nod_type, E, edges, pieces, v_b)
    _attach_stations(net, edges)
    return net, edges


def _connections(edges, pieces):
    con = []
    def c(a, b, i, j):
        con.append(f'  <connection from="{a}" to="{b}" fromLane="{i}" toLane="{j}"/>')
    for ax, P in pieces.items():
        lm = LM[ax]
        for a, b in zip(P[:-1], P[1:]):
            if a['kind'] != b['kind']:
                continue
            if a['nl'] == b['nl']:
                if ax == 'WB' and a['kind'] == 'in' and abs(a['s1'] - lm['lane_add']) < 1e-6:
                    # WB: 4 upstream lanes (idx 3 = lane 0 median ... idx 0 = lane 3 curb) -> approach 4 lanes
                    # with the added median left-turn lane; the curb lane feeds the right-turn slip
                    c(a['id'], b['id'], 3, 3); c(a['id'], b['id'], 3, 2); c(a['id'], b['id'], 2, 1)
                    c(a['id'], b['id'], 1, 0); c(a['id'], 'slipWB', 0, 0); c(a['id'], b['id'], 0, 0)
                elif ax == 'EB' and a['kind'] == 'in' and abs(a['s1'] - lm['slip_div']) < 1e-6:
                    for k in range(4):
                        c(a['id'], b['id'], k, k)
                    c(a['id'], 'slipEB', 0, 0)
                else:
                    for k in range(a['nl']):
                        c(a['id'], b['id'], k, k)
            elif b['nl'] == a['nl'] + 1:          # acceleration lane for the merging slip at index 0
                for k in range(a['nl']):
                    c(a['id'], b['id'], k, k + 1)
            else:
                for k in range(1, a['nl']):
                    c(a['id'], b['id'], k, k - 1)
        acc = [p for p in P if p['nl'] == 5][0]
        c('slipNB' if ax == 'EB' else 'slipSB', acc['id'], 0, 0)
    # cross street pieces
    for a, b in (('NBin0', 'NBin1'), ('SBout0', 'SBout1'), ('SBin0', 'SBin1'), ('NBout0', 'NBout1')):
        for k in range(4):
            c(a, b, k, k)
    c('NBin0', 'slipNB', 0, 0); c('SBin0', 'slipSB', 0, 0)
    c('slipEB', 'SBout1', 0, 0); c('slipWB', 'NBout1', 0, 0)
    # junction J. Inbound lane index 3 = median.
    EBi = [p for p in pieces['EB'] if p['kind'] == 'in'][-1]['id']; EBo = [p for p in pieces['EB'] if p['kind'] == 'out'][0]['id']
    WBi = [p for p in pieces['WB'] if p['kind'] == 'in'][-1]['id']; WBo = [p for p in pieces['WB'] if p['kind'] == 'out'][0]['id']
    # EB: 14 (idx3) left + U-turn + through, 15 (idx2) left + through, 16 through, 17 through
    c(EBi, 'NBout0', 3, 3); c(EBi, WBo, 3, 3); c(EBi, WBo, 3, 2); c(EBi, EBo, 3, 3)
    c(EBi, 'NBout0', 2, 2); c(EBi, EBo, 2, 2); c(EBi, EBo, 1, 1); c(EBi, EBo, 0, 0)
    # WB: 4 (idx3) left + U-turn, 5 (idx2) through + left, 6 through, 7 through
    c(WBi, 'SBout0', 3, 3); c(WBi, EBo, 3, 3); c(WBi, EBo, 3, 2); c(WBi, 'SBout0', 2, 2); c(WBi, WBo, 2, 2)
    c(WBi, WBo, 1, 1); c(WBi, WBo, 0, 0)
    # NB: 9 (idx3) left, 10 left + through, 11 through, 12 through
    c('NBin1', WBo, 3, 3); c('NBin1', WBo, 2, 2); c('NBin1', 'NBout0', 2, 2)
    c('NBin1', 'NBout0', 1, 1); c('NBin1', 'NBout0', 0, 0)
    # SB: 19 (idx3) left, 20 left + through, 21 through, 22 through
    c('SBin1', EBo, 3, 3); c('SBin1', EBo, 2, 2); c('SBin1', 'SBout0', 2, 2)
    c('SBin1', 'SBout0', 1, 1); c('SBin1', 'SBout0', 0, 0)
    return con


def _write_and_convert(outdir, tag, net, nodes, nod_type, E, edges, pieces, v_b):
    base = os.path.join(outdir, f'A_{tag}_{os.getpid()}')
    nod = ['<nodes>']
    for k, (x, y) in nodes.items():
        tp = nod_type.get(k, 'priority')
        nod.append(f'  <node id="{k}" x="{x:.2f}" y="{y:.2f}" type="{tp}"/>')
    nod.append('</nodes>')
    edg = ['<edges>']
    for (i, fr, to, nl, v, shape, allow, w) in E:
        sh = f' shape="{nodes[fr][0]:.2f},{nodes[fr][1]:.2f} {shape} {nodes[to][0]:.2f},{nodes[to][1]:.2f}"' if shape else ''
        if fr == 'J' and shape:
            sh = f' shape="{shape} {nodes[to][0]:.2f},{nodes[to][1]:.2f}"'
        if to == 'J' and shape:
            sh = f' shape="{nodes[fr][0]:.2f},{nodes[fr][1]:.2f} {shape}"'
        edg.append(f'  <edge id="{i}" from="{fr}" to="{to}" numLanes="{nl}" speed="{v:.2f}" spreadType="center"{sh}>')
        for k in range(nl):
            al = allow[k] if allow else 'passenger truck custom1'
            edg.append(f'    <lane index="{k}" allow="{al}" width="{w:.2f}"/>')
        edg.append('  </edge>')
    edg.append('</edges>')
    con = ['<connections>'] + _connections(edges, pieces) + ['</connections>']
    for ext, txt in (('nod', nod), ('edg', edg), ('con', con)):
        open(f'{base}.{ext}.xml', 'w', newline='\n').write('\n'.join(txt) + '\n')
    tmpnet = net + f'.{os.getpid()}.tmp.xml'
    subprocess.run([NETCONVERT, '-n', f'{base}.nod.xml', '-e', f'{base}.edg.xml', '-x', f'{base}.con.xml',
                    '-o', tmpnet, '--no-turnarounds', 'false', '--offset.disable-normalization', 'true',
                    '--tls.layout', 'incoming', '--tls.yellow.time', str(CYCLE['yellow']),
                    '--junctions.corner-detail', '5', '--no-warnings', 'true', '--no-internal-links', 'false'],
                   check=True, stdout=subprocess.DEVNULL)
    _set_signal(tmpnet, edges, v_b)
    os.replace(tmpnet, net)


def _set_signal(netfile, edges, green_d=None):
    """Set green durations of the 'incoming' layout phases to the estimated plan (160 s cycle)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(netfile); root = tree.getroot()
    for tl2 in root.findall('tlLogic'):
        if tl2.get('id') != 'J':
            n = len(tl2.find('phase').get('state'))
            for ph in list(tl2):
                tl2.remove(ph)
            for dur, ch in ((green_d, 'G'), (3, 'y'), (BN_CYC - green_d - 3, 'r')):
                el = ET.SubElement(tl2, 'phase'); el.set('duration', str(dur)); el.set('state', ch * n)
            tl2.set('offset', str(BN_OFF))
    tl = [x for x in root.findall('tlLogic') if x.get('id') == 'J'][0]
    if NOALLRED:                      # netconvert inserts an all-red phase in the no-DL net and the layouts ending 90 or 130 m before the stop line
        for ph in list(tl.findall('phase')):
            if not any(ch in 'Gg' + 'y' for ch in ph.get('state')):
                tl.remove(ph)
    links = {}
    for cn in root.findall('connection'):
        if cn.get('tl') == 'J':
            links[int(cn.get('linkIndex'))] = cn.get('from')
    def approach(eid):
        if eid.startswith('EB'): return 'EB'
        if eid.startswith('WB'): return 'WB'
        return 'NB' if eid.startswith('NB') else 'SB'
    for ph in tl.findall('phase'):
        st = ph.get('state')
        g = {approach(links[i]) for i, ch in enumerate(st) if ch in 'Gg' and i in links}
        if 'y' in st:
            ph.set('duration', str(CYCLE['yellow']))
        elif len(g) == 1:
            ph.set('duration', str(CYCLE[g.pop()]))
        else:
            ph.set('duration', '1')
    tree.write(netfile)


def _attach_stations(netfile, edges):
    """Station of the first point of each arterial edge after netconvert (junction cut-backs)."""
    import sumolib
    n = sumolib.net.readNet(netfile)
    for e in edges:
        if e['axis'] is None:
            continue
        A = AXES[e['axis']]
        p = np.array(n.getEdge(e['id']).getLane(0).getShape()[0])
        q = np.array(n.getEdge(e['id']).getLane(e['nl'] - 1).getShape()[0])
        mid = (p + q) / 2
        a = A.xy[:-1]; b = A.xy[1:]; ab = b - a; L2 = (ab ** 2).sum(1)
        tt = np.clip(((mid - a) * ab).sum(1) / L2, 0, 1)
        proj = a + tt[:, None] * ab
        d = np.hypot(*(proj - mid).T); i = int(d.argmin())
        s_net = float(A.s[i] + tt[i] * np.sqrt(L2[i]))
        if A.s[0] + 1 < s_net < A.s[-1] - 1:
            e['x0'] = s_net                 # inside the measured axis: use the projected start
        e['x0_net'] = s_net


# ---------------------------------------------------------------- demand
FLOW = DEM['app_vph']           # approach volumes (veh/h)
SH = DEM['od_share']            # origin -> destination shares
APPROACH_EDGES = {'EB': 'EB', 'WB': 'WB', 'NB': 'NB', 'SB': 'SB'}


def _path(edges, o, dst):
    art = {ax: [e['id'] for e in edges if e.get('axis') == ax] for ax in ('EB', 'WB')}
    ins = {ax: [e['id'] for e in edges if e.get('axis') == ax and e['kind'] == 'in'] for ax in ('EB', 'WB')}
    outs = {ax: [e['id'] for e in edges if e.get('axis') == ax and e['kind'] == 'out'] for ax in ('EB', 'WB')}
    lmE, lmW = LM['EB'], LM['WB']
    def ins_until(ax, s):
        return [e['id'] for e in edges if e.get('axis') == ax and e['kind'] == 'in' and e['x1'] <= s + 1e-6]
    def outs_from(ax, s):
        return [e['id'] for e in edges if e.get('axis') == ax and e['kind'] == 'out' and e['s0'] >= s - 1e-6]
    if o == 'EB':
        if dst == 'EBo': return ins['EB'] + outs['EB']
        if dst == 'NBo': return ins['EB'] + ['NBout0', 'NBout1']
        if dst == 'WBo': return ins['EB'] + outs['WB']
        if dst == 'SBo': return ins_until('EB', lmE['slip_div']) + ['slipEB', 'SBout1']
    if o == 'WB':
        if dst == 'WBo': return ins['WB'] + outs['WB']
        if dst == 'SBo': return ins['WB'] + ['SBout0', 'SBout1']
        if dst == 'EBo': return ins['WB'] + outs['EB']
        if dst == 'NBo': return ins_until('WB', lmW['lane_add']) + ['slipWB', 'NBout1']
    if o == 'NB':
        if dst == 'NBo': return ['NBin0', 'NBin1', 'NBout0', 'NBout1']
        if dst == 'WBo': return ['NBin0', 'NBin1'] + outs['WB']
        if dst == 'EBo': return ['NBin0', 'slipNB'] + outs_from('EB', lmE['slip_merge'])
        if dst == 'SBo': return None
    if o == 'SB':
        if dst == 'SBo': return ['SBin0', 'SBin1', 'SBout0', 'SBout1']
        if dst == 'EBo': return ['SBin0', 'SBin1'] + outs['EB']
        if dst == 'WBo': return ['SBin0', 'slipSB'] + outs_from('WB', lmW['slip_merge'])
        if dst == 'NBo': return None
    return None


def routes(edges, q, depart=None):
    depart = DEPART if depart is None else depart
    out = []
    for o in ('EB', 'WB', 'NB', 'SB'):
        tot = sum(v for k, v in SH[o].items() if _path(edges, o, k) is not None)
        for dst, sh in SH[o].items():
            p = _path(edges, o, dst)
            if p is None or sh <= 0:
                continue
            out.append((f'{o}_{dst}', p, FLOW[o] * q * sh / tot, depart))
    return out


def dl_route(route_id):
    """The DL helps CAVs that travel along the arterial through the junction (through, left, U-turn)."""
    o, dst = route_id.split('_')[:2]
    return o in ('EB', 'WB') and dst != {'EB': 'SBo', 'WB': 'NBo'}[o]


def station(edges):
    return {e['id']: (e['axis'], e['x0']) for e in edges if e.get('axis')}


# ---------------------------------------------------------------- information levels
# GEOTZ_LEVEL selects which aerial layers the model uses (default B = all layers):
#   G  imagery layer only: measured geometry (axes, lane widths, slip lanes, landmarks), generic
#      demand (equal approach volumes with the observed total, left 25 % / through 55 % / right 20 %),
#      generic signal (equal greens), default driver parameters, best-lane entry, no downstream queue
#   C  + aerial counts: observed approach volumes, origin-destination shares and the signal plan
#      estimated from stop-line discharges
#   B  + aerial behaviour: calibrated driver parameters, entry lanes and the downstream queue
LEVEL = os.environ.get('GEOTZ_LEVEL', 'B')
if LEVEL in ('G', 'C'):
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
    V_BN = 160.0                      # no downstream signal (queue not visible without trajectories)
if LEVEL == 'G':
    _tot = sum(FLOW.values())
    FLOW = {k: _tot / 4.0 for k in FLOW}
    _turn = {'EB': dict(NBo=0.25, EBo=0.55, SBo=0.20, WBo=0.0), 'WB': dict(SBo=0.25, WBo=0.55, NBo=0.20, EBo=0.0),
             'NB': dict(WBo=0.25, NBo=0.55, EBo=0.20, SBo=0.0), 'SB': dict(EBo=0.25, SBo=0.55, WBo=0.20, NBo=0.0)}
    SH = _turn
    CYCLE = dict(EB=37, WB=37, NB=37, SB=37, yellow=3)
if LEVEL == 'B2':                     # alternative near-optimal calibration (second best of calib_a)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
if LEVEL == 'G0':                     # imagery only: demand from lane-count capacity, generic 120 s signal
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
    V_BN = 160.0
    # capacity of an approach = 4 lanes x 1800 veh/h x 27/120 green ratio = 1620 veh/h, demand at v/c = 0.75
    FLOW = {k: 0.75 * 1620.0 for k in FLOW}
    SH = {'EB': dict(NBo=0.25, EBo=0.55, SBo=0.20, WBo=0.0), 'WB': dict(SBo=0.25, WBo=0.55, NBo=0.20, EBo=0.0),
          'NB': dict(WBo=0.25, NBo=0.55, EBo=0.20, SBo=0.0), 'SB': dict(EBo=0.25, SBo=0.55, WBo=0.20, NBo=0.0)}
    CYCLE = dict(EB=27, WB=27, NB=27, SB=27, yellow=3)
if LEVEL == 'AB1':                    # layer B without the boundary condition (no downstream signal)
    V_BN = 160.0
if LEVEL == 'AB2':                    # layer B without the calibrated behaviour (default driver gap, best-lane entry)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
if LEVEL == 'R3':                     # reference: setting chosen when recordings 0 and 4 are held out
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    V_BN = 60.0
if LEVEL == 'R4':                     # reference: setting chosen when recordings 2 and 3 are held out
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    V_BN = 60.0
    DEPART = 'best'
if LEVEL in ('G0V', 'G0H', 'G0F'):    # G0 variants (site A)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
    V_BN = 160.0
    SH = {'EB': dict(NBo=0.25, EBo=0.55, SBo=0.20, WBo=0.0), 'WB': dict(SBo=0.25, WBo=0.55, NBo=0.20, EBo=0.0),
          'NB': dict(WBo=0.25, NBo=0.55, EBo=0.20, SBo=0.0), 'SB': dict(EBo=0.25, SBo=0.55, WBo=0.20, NBo=0.0)}
    CYCLE = dict(EB=27, WB=27, NB=27, SB=27, yellow=3)
    if LEVEL == 'G0V':                # observed total volume, equal split, generic 120 s cycle
        _tot = sum(DEM['app_vph'].values())
        FLOW = {k: _tot / 4.0 for k in DEM['app_vph']}
    else:                             # capacity default at v/c 0.9 or 1.0
        FLOW = {k: (0.9 if LEVEL == 'G0H' else 1.0) * 1620.0 for k in DEM['app_vph']}
if LEVEL == 'G0Vx':                   # control: downstream signal assumed with the generic green of the planner's own signal (27 s in 120 s)
    V_BN, BN_CYC = 27.0, 120.0
    LEVEL = 'G0Vq'
if LEVEL == 'G0Vo':                   # G0Vg with the downstream signal shifted by half a cycle
    BN_OFF = 60.0
    LEVEL = 'G0Vg'
if LEVEL == 'G0Vg':                   # as G0Vc, but the downstream signal has the generic 120 s cycle: green 120 x 1880 / (4 x 1800) = 31.3 s
    V_BN, BN_CYC = 31.0, 120.0
    LEVEL = 'G0Vq'
if LEVEL in ('G0Vc', 'G0Vk'):         # G0 with observed volume and a downstream queue estimated without calibration
    V_BN = 42.0 if LEVEL == 'G0Vc' else 30.0     # cheap_queue.py: exit flow from counts (41.8 s) or from edge density x speed (30.3 s)
    LEVEL = 'G0Vq'
if LEVEL in ('G0Vq', 'G0Hq'):         # G0 variants with the calibrated downstream queue (WB exit signal, 40 s green)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    DEPART = 'best'
    SH = {'EB': dict(NBo=0.25, EBo=0.55, SBo=0.20, WBo=0.0), 'WB': dict(SBo=0.25, WBo=0.55, NBo=0.20, EBo=0.0),
          'NB': dict(WBo=0.25, NBo=0.55, EBo=0.20, SBo=0.0), 'SB': dict(EBo=0.25, SBo=0.55, WBo=0.20, NBo=0.0)}
    CYCLE = dict(EB=27, WB=27, NB=27, SB=27, yellow=3)
    if LEVEL == 'G0Vq':
        _tot = sum(DEM['app_vph'].values())
        FLOW = {k: _tot / 4.0 for k in DEM['app_vph']}
    else:
        FLOW = {k: 0.9 * 1620.0 for k in DEM['app_vph']}
if LEVEL == 'AB1g':                   # no downstream queue, calibrated time gap only (best-lane entry)
    V_BN = 160.0
    DEPART = 'best'
if LEVEL == 'AB1e':                   # no downstream queue, random entry lanes only (default time gap)
    V_BN = 160.0
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
