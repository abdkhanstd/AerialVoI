"""Figures that depend only on the aerial data, the site models and the calibration runs (written to figs/)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figs')
RAW = os.path.join(os.path.dirname(ROOT), 'data_raw')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0})
W = 6.6     # text width (in)
C = dict(obs='#4d4d4d', full='#c0392b', geom='#2471a3', beh='#7d3c98', lc='#bcbcbc', lm='#7f8c8d')


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close(fig)


def fig_sites():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[0.85, 1.25]))
    # (a) site A: UAV trajectories in the data's coordinate frame, coloured by lane group
    t = pd.read_parquet(os.path.join(DATA, 'siteA_tracks.parquet'), columns=['x', 'y', 'ax', 'lane_id', 's'])
    geo = json.load(open(os.path.join(DATA, 'siteA_geom.json')))
    col = {'EB': '#2471a3', 'WB': '#c0392b'}
    for a, g in t.groupby('ax'):
        g = g.sample(min(len(g), 60000), random_state=0)
        ax[0].scatter(g.x, g.y, s=0.05, c=col[a], alpha=0.25, rasterized=True, linewidths=0)
    for a in ('EB', 'WB'):
        xy = np.array(geo[a]['axis_xy'])
        ax[0].plot(xy[:, 0], xy[:, 1], color='k', lw=0.6, ls='--')
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) - geo[a]['s_stop']
        for k, v in geo[a]['landmarks'].items():
            i = int(np.argmin(np.abs(s - v)))
            mk = {'stop': 's', 'slip_div': 'v', 'slip_merge': '^', 'lane_add': 'D'}[k]
            ax[0].plot(*xy[i], marker=mk, ms=3.5, mfc='white', mec='k', mew=0.6, ls='none')
    ax[0].set_aspect('equal'); ax[0].set_xlabel('$x$ (m)'); ax[0].set_ylabel('$y$ (m)')
    ax[0].set_title('(a) Site A, UAV trajectories (EB blue, WB red)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, ls='none', mfc='white', mec='k', ms=3.5, label=l) for m, l in
         (('s', 'stop line'), ('v', 'slip diverge'), ('^', 'slip merge'), ('D', 'lane added'))]
    ax[0].legend(handles=h, loc='lower right', frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.6)
    # (b) site B: helicopter frame (TGSIM run 2), crop around the tracked lanes
    im = Image.open(os.path.join(RAW, 'tgsim_i9094s', 'ann', 'Annotation on Regions', 'I90_94_Stationary_Run_2_ref_image_annotated.jpg')).convert('RGB')
    w, h_ = im.size
    crop = im.crop((int(0.55 * w), int(0.315 * h_), int(0.78 * w), int(0.50 * h_)))
    crop.thumbnail((1800, 1800))
    a = np.asarray(crop).astype(float) / 255.0
    a = np.clip(a ** 0.55 * 1.05, 0, 1)                 # brighten the dusk footage for print
    ax[1].imshow(a); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].annotate('traffic', xy=(0.12, 0.30), xytext=(0.32, 0.30), xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='w', lw=0.8), color='w', fontsize=7, va='center')
    ax[1].set_title('(b) Site B, helicopter frame (TGSIM I-90/I-94)')
    save(fig, 'fig_sites.pdf')


def fig_calibration():
    """Observed vs simulated profiles for the selected calibration settings."""
    rb = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibB', 'results2.json'))) if 'err' not in x]
    best_b = [x for x in rb if x['v_b'] == 4.5 and x['hv_tau'] == 1.5 and x['rho'] == 1.0 and x['ramp_l'] == 0.435]
    cb = pd.read_csv(os.path.join(DATA, 'siteB_cells.csv')).groupby('s_mid')[['vehsec', 'n_lc']].sum()
    ra = [x for x in json.load(open(os.path.join(ROOT, 'runs', 'calibA', 'results2.json'))) if 'err' not in x]
    best_a = [x for x in ra if x['qd'] == 1.0 and x['hv_tau'] == 1.2 and x['v_b'] == 40.0 and x['depart'] == 'random']
    ca = pd.read_csv(os.path.join(DATA, 'siteA_cells.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.8))
    # (a) site B lane-change rate per cell
    sB = -325 + 25 * (np.arange(22) + 0.5)
    sim = np.mean([1000 * np.array(x['lc_cells']) / np.maximum(np.array(x['vs_cells']), 1) for x in best_b], axis=0)
    axs[0].plot(cb.index, 1000 * cb.n_lc / cb.vehsec, 'o-', ms=2, color=C['obs'], label='observed')
    axs[0].plot(sB, sim, 's--', ms=2, color=C['full'], label='simulated')
    axs[0].set_xlabel('station $s$ (m)'); axs[0].set_ylabel('LC per 1000 veh-s'); axs[0].set_title('(a) Site B lane changes')
    axs[0].legend(frameon=False)
    for k, (u, title) in enumerate((('EB', '(b) Site A eastbound speed'), ('WB', '(c) Site A westbound speed'))):
        ax = axs[k + 1]
        o0 = {'EB': -130.0, 'WB': -160.0}[u]
        n = len(best_a[0]['cells']['vs_cells'][u])
        s = o0 + 10 * (np.arange(n) + 0.5)
        sp = np.mean([np.array(x['cells']['spd_cells'][u]) / np.maximum(np.array(x['cells']['vs_cells'][u]), 1) for x in best_a], axis=0)
        vs = np.mean([np.array(x['cells']['vs_cells'][u]) for x in best_a], axis=0)
        sp[vs < 50] = np.nan
        q = ca[ca.ax == u].groupby('s_mid').apply(lambda g: np.average(g.speed, weights=g.vehsec))
        q = q[(q.index > o0) & ((q.index < 0) | (q.index > 40))]
        ax.plot(q.index, q.values, 'o-', ms=2, color=C['obs'], label='observed')
        ax.plot(s, sp, 's--', ms=2, color=C['full'], label='simulated')
        ax.axvspan(0, 40, color='#eeeeee', zorder=-1)
        ax.set_xlabel('station $s$ (m)'); ax.set_title(title)
        if k == 0:
            ax.set_ylabel('mean speed (m/s)')
    fig.subplots_adjust(wspace=0.35)
    save(fig, 'fig_calibration.pdf')


def fig_thumbs():
    """Small input thumbnails of the two sites for the study-design diagram (Fig. 2)."""
    # site A: 1 Hz UAV trajectory samples, EB blue and WB red, on a grey road-like background
    t = pd.read_parquet(os.path.join(DATA, 'siteA_tracks.parquet'), columns=['x', 'y', 'ax'])
    fig, ax = plt.subplots(figsize=(1.6, 1.15))
    fig.patch.set_facecolor('#5f6366'); ax.set_facecolor('#5f6366')
    col = {'EB': '#7fb3e6', 'WB': '#f1948a'}
    for a, g in t.groupby('ax'):
        g = g.sample(min(len(g), 40000), random_state=0)
        ax.scatter(g.x, g.y, s=0.04, c=col[a], alpha=0.5, rasterized=True, linewidths=0)
    ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(os.path.join(FIGS, 'thumb_A.png'), dpi=300, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    # site B: helicopter frame of TGSIM run 2, crop around the tracked lanes, brightened
    im = Image.open(os.path.join(RAW, 'tgsim_i9094s', 'ann', 'Annotation on Regions', 'I90_94_Stationary_Run_2_ref_image_annotated.jpg')).convert('RGB')
    w, h_ = im.size
    crop = im.crop((int(0.55 * w), int(0.30 * h_), int(0.78 * w), int(0.50 * h_)))
    crop.thumbnail((700, 700))
    a = np.clip((np.asarray(crop).astype(float) / 255.0) ** 0.55 * 1.05, 0, 1)
    a = a[:, :int(0.5 * a.shape[1])]                    # left half, without the lane annotations
    Image.fromarray((a * 255).astype('uint8')).save(os.path.join(FIGS, 'thumb_B.png'))


def fig_networks():
    """Reconstructed SUMO networks with one DL layout and the candidate boundaries."""
    import sumolib, site_a, site_b
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    netA, edgesA = site_a.build_net(os.path.join(ROOT, 'runs', 'fignets'), dict(e=-50.0, g=120.0))
    netB, edgesB = site_b.build_net(os.path.join(ROOT, 'runs', 'fignets'), -125.0, site_b.V_B)
    for ax, net, edges, title in ((axs[0], netA, edgesA, '(a) Site A, layout $e=-50$ m, $g=120$ m'),
                                  (axs[1], netB, edgesB, '(b) Site B, layout $b=-125$ m')):
        n = sumolib.net.readNet(net)
        dl = {e['id']: e['dl_idx'] for e in edges if e.get('dl')}
        for ed in n.getEdges():
            for ln in ed.getLanes():
                sh = np.array(ln.getShape())
                isdl = ed.getID() in dl and ln.getIndex() == dl[ed.getID()]
                ax.plot(sh[:, 0], sh[:, 1], color='#c0392b' if isdl else '#9a9a9a', lw=1.4 if isdl else 0.5,
                        solid_capstyle='butt')
        ax.set_aspect('equal'); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_xlim(2940, 3350); axs[0].set_ylim(1390, 1640)
    axs[1].set_aspect('auto'); axs[1].set_xlim(-720, 920); axs[1].set_ylim(-45, 30)
    for x in site_b.CAND[::2]:
        axs[1].plot([x, x], [6, 11], color='k', lw=0.6)
    axs[1].text(-420, 14, 'candidate DL starts (every 50 m)', fontsize=6.5)
    axs[1].text(-560, -40, 'on-ramp', fontsize=6.5)
    axs[1].text(600, -8, 'left branch', fontsize=6.5)
    axs[1].text(560, -40, 'right branch', fontsize=6.5)
    axs[1].text(-700, 22, 'vertical scale exaggerated', fontsize=6, style='italic')
    save(fig, 'fig_networks.pdf')


if __name__ == '__main__':
    fig_sites(); fig_calibration(); fig_networks(); fig_thumbs()
    print('ok')
