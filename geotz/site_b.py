"""SUMO model of site B (TGSIM I-90/I-94 northbound, helicopter 8K video, runs 2/4/6).

Station x (m) on a straight axis, x = 0 at the end of the marked weaving section (extract_b).
Observed window [-325, 225]. Upstream of -325 and downstream of 225 the model is synthetic.

What the aerial data show and the model reproduces:
- 6 lanes (SUMO index 0 = right lane = TGSIM 15, index 5 = median lane = TGSIM 10),
- an on-ramp joining on the right at x = -250 (acceleration lane to x = -100),
- strong lane asymmetry: the three left lanes queue (4-6 m/s) while the right lanes flow
  (12-14 m/s). Downstream, the carriageway therefore separates into a left branch L (3 lanes,
  queued, bottleneck speed v_b) and a right branch R (3 lanes, free); 43.5 % of the vehicles
  leaving the window are in the left three lanes,
- arrivals at the upstream edge of the window are spread almost evenly over the lanes, so
  vehicles sort themselves (weave) inside the window.
The CAV dedicated lane (DL) is the median lane from x = b through the left branch.
"""
import os, subprocess
import numpy as np
from sumo import SUMO_HOME

NETCONVERT = os.path.join(SUMO_HOME, 'bin', 'netconvert')
X_UP, X_MERGE, X_ACC_END, X_SPLIT, X_BN, X_END = -700.0, -250.0, -100.0, 300.0, 500.0, 900.0
OBS_WINDOW = (-325.0, 225.0)
V_MAIN = 24.6               # 88 km/h posted limit
LANE_W = 3.6
N_LANES = 6
SHARE_L = 0.435             # share of flow leaving in the three left lanes (measured)
TRUCK_SHARE = 0.033         # large-vehicle share (measured)
FLOW_MAIN = 6800.0          # veh/h crossing x = -200 (runs 2/4/6 mean 6762)
FLOW_RAMP = 600.0           # veh/h, chosen so that flow at x = 100 matches the observed 7464
CAND = [float(x) for x in np.arange(-325, 201, 25)]   # DL start candidates (observed window)


def _main_breaks(b):
    pts = [X_UP, X_MERGE, X_ACC_END, X_SPLIT]
    if b is not None and X_UP < b < X_SPLIT and all(abs(b - p) > 1.0 for p in pts):
        pts.append(b)
    return sorted(pts)


def build_net(outdir, b, v_b):
    os.makedirs(outdir, exist_ok=True)
    tag = ('noDL' if b is None else f'b{int(round(b))}') + f'_vb{v_b:.1f}'
    net = os.path.join(outdir, f'B_{tag}.net.xml')
    pts = _main_breaks(b)
    edges = []
    for i in range(len(pts) - 1):
        x0, x1 = pts[i], pts[i + 1]
        nl = N_LANES + (1 if (x0 >= X_MERGE - 1e-6 and x1 <= X_ACC_END + 1e-6) else 0)
        dl = (b is not None) and x0 >= b - 1e-6
        edges.append(dict(id=f'm{i}', fr=f'n{i}', to=f'n{i+1}', x0=x0, x1=x1, nl=nl, dl=dl, v=V_MAIN,
                          dl_idx=nl - 1, branch='M'))
    ns = len(pts) - 1     # split node index
    dlL = b is not None
    edges += [dict(id='L0', fr=f'n{ns}', to='l1', x0=X_SPLIT, x1=X_BN, nl=3, dl=dlL, v=V_MAIN, dl_idx=2, branch='L'),
              dict(id='L1', fr='l1', to='l2', x0=X_BN, x1=X_END, nl=3, dl=dlL, v=v_b, dl_idx=2, branch='L'),
              dict(id='R0', fr=f'n{ns}', to='r2', x0=X_SPLIT, x1=X_END, nl=3, dl=False, v=V_MAIN, dl_idx=2, branch='R')]
    if os.path.exists(net):
        return net, edges
    nod = ['<nodes>'] + [f'  <node id="n{i}" x="{p:.2f}" y="0"/>' for i, p in enumerate(pts)]
    nod += [f'  <node id="l1" x="{X_BN:.2f}" y="8"/>', f'  <node id="l2" x="{X_END:.2f}" y="8"/>',
            f'  <node id="r2" x="{X_END:.2f}" y="-30"/>', f'  <node id="q0" x="{X_MERGE - 250:.2f}" y="-60"/>', '</nodes>']
    edg = ['<edges>']
    for e in edges:
        edg.append(f'  <edge id="{e["id"]}" from="{e["fr"]}" to="{e["to"]}" numLanes="{e["nl"]}" speed="{e["v"]:.2f}" spreadType="center">')
        for k in range(e['nl']):
            allow = 'custom1' if (e['dl'] and k == e['dl_idx']) else 'passenger truck custom1'
            edg.append(f'    <lane index="{k}" allow="{allow}" width="{LANE_W}"/>')
        edg.append('  </edge>')
    i_merge = [i for i, e in enumerate(edges) if e['branch'] == 'M' and abs(e['x0'] - X_MERGE) < 1e-6][0]
    edg.append(f'  <edge id="ramp" from="q0" to="n{i_merge}" numLanes="1" speed="16.7" allow="passenger truck custom1"/>')
    edg.append('</edges>')
    con = ['<connections>']
    mains = [e for e in edges if e['branch'] == 'M']
    for a, c in zip(mains[:-1], mains[1:]):
        if a['nl'] == c['nl']:
            pairs = [(k, k) for k in range(a['nl'])]
        elif c['nl'] == a['nl'] + 1:
            pairs = [(k, k + 1) for k in range(a['nl'])]
        else:
            pairs = [(k, k - 1) for k in range(1, a['nl'])]
        con += [f'  <connection from="{a["id"]}" to="{c["id"]}" fromLane="{i}" toLane="{j}"/>' for i, j in pairs]
    last = mains[-1]
    con += [f'  <connection from="{last["id"]}" to="L0" fromLane="{k}" toLane="{k-3}"/>' for k in (3, 4, 5)]
    con += [f'  <connection from="{last["id"]}" to="R0" fromLane="{k}" toLane="{k}"/>' for k in (0, 1, 2)]
    con += [f'  <connection from="L0" to="L1" fromLane="{k}" toLane="{k}"/>' for k in (0, 1, 2)]
    acc = [e for e in mains if e['nl'] == N_LANES + 1][0]
    con.append(f'  <connection from="ramp" to="{acc["id"]}" fromLane="0" toLane="0"/>')
    con.append('</connections>')
    base = os.path.join(outdir, f'B_{tag}_{os.getpid()}')
    for ext, txt in (('nod', nod), ('edg', edg), ('con', con)):
        open(f'{base}.{ext}.xml', 'w', newline='\n').write('\n'.join(txt) + '\n')
    tmpnet = net + f'.{os.getpid()}.tmp.xml'
    subprocess.run([NETCONVERT, '-n', f'{base}.nod.xml', '-e', f'{base}.edg.xml', '-x', f'{base}.con.xml',
                    '-o', tmpnet, '--no-turnarounds', 'true', '--offset.disable-normalization', 'true',
                    '--junctions.corner-detail', '0', '--no-warnings', 'true'], check=True,
                   stdout=subprocess.DEVNULL)
    os.replace(tmpnet, net)
    return net, edges


RHO = 1.0                   # share of mainline vehicles already in their destination lane group (calibrated)
V_B = 4.5                   # left-branch bottleneck speed (calibrated, calib_b)
HV_LC = dict(hv_sg=2.0, hv_strat=3.0, hv_assert=1.0, HV=1.5)   # human lane-change / gap parameters (calibrated)


def routes(edges, q, rho=None, ramp_l=None):
    """(route id, edges, veh/h, departLane). Destination L = left branch (where the DL leads).
    A share rho of the mainline vehicles enters in a lane of its destination group, the rest in a
    random lane, which reproduces the lane sorting (weaving) seen inside the observed window."""
    rho = RHO if rho is None else rho
    ramp_l = (globals().get('RAMP_L_OVERRIDE', SHARE_L)) if ramp_l is None else ramp_l
    mains = [e['id'] for e in edges if e['branch'] == 'M']
    ramp = ['ramp'] + [e['id'] for e in edges if e['branch'] == 'M' and e['x0'] >= X_MERGE - 1e-6]
    out = []
    for dst, tail, sh, grp in (('L', ['L0', 'L1'], SHARE_L, (3, 4, 5)), ('R', ['R0'], 1 - SHARE_L, (0, 1, 2))):
        f = FLOW_MAIN * q * sh
        for k in grp:
            out.append((f'main{dst}{k}', mains + tail, f * rho / 3, str(k)))
        out.append((f'main{dst}x', mains + tail, f * (1 - rho), 'random'))
        rs = ramp_l if dst == 'L' else 1 - ramp_l
        if rs > 0:
            out.append((f'ramp{dst}', ramp + tail, FLOW_RAMP * q * rs, 'best'))
    return out


def dl_route(route_id):
    """True when the dedicated lane is useful for this route (left branch)."""
    return 'L' in route_id[4:]


def station(edges):
    d = {e['id']: e['x0'] for e in edges}
    d['ramp'] = None
    return d


# ---------------------------------------------------------------- information levels
# GEOTZ_LEVEL selects which aerial layers the model uses (default B = all layers):
#   G  imagery layer only: lanes, on-ramp, acceleration lane and branch split as seen in the image,
#      generic demand (observed total, 10 % from the on-ramp, 50/50 destination split), random entry
#      lanes, default driver parameters, no downstream bottleneck
#   C  + aerial counts: observed mainline and ramp volumes and destination split
#   B  + aerial behaviour: calibrated bottleneck, lane pre-sorting and driver parameters
LEVEL = os.environ.get('GEOTZ_LEVEL', 'B')
if LEVEL in ('G', 'C'):
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    RHO = 0.0
    V_B = V_MAIN
if LEVEL == 'G':
    FLOW_MAIN, FLOW_RAMP, SHARE_L = 6660.0, 740.0, 0.5
if LEVEL == 'B2':                     # alternative near-optimal calibration (second best of calib_b)
    V_B = 4.0
if LEVEL == 'G0':                     # imagery only: demand from lane-count capacity (6 x 2000 veh/h at v/c 0.75)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    RHO = 0.0
    V_B = V_MAIN
    FLOW_MAIN, FLOW_RAMP, SHARE_L = 0.9 * 9000.0, 0.1 * 9000.0, 0.5
if LEVEL == 'AB1':                    # layer B without the boundary condition (no left-branch bottleneck)
    V_B = V_MAIN
if LEVEL == 'AB2':                    # layer B without calibrated behaviour (no pre-sorting, default lane changing)
    HV_LC = dict(hv_sg=1.0, hv_strat=1.0, hv_assert=1.0, HV=1.5)
    RHO = 0.0
if LEVEL == 'R3':                     # reference: third-best calibration (4.0 m/s, 90 % pre-sorted)
    V_B = 4.0
    RHO = 0.9
if LEVEL == 'R4':                     # reference: fourth-best calibration (4.0 m/s, on-ramp traffic all to the right branch)
    V_B = 4.0
    RAMP_L_OVERRIDE = 0.0
