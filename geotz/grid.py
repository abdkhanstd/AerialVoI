"""Full-factorial simulation grid: every dedicated-lane layout x scenario x seed.

Every planner choice, rule and oracle is later a lookup in this table, so all of them are compared
on the same seeds (common random numbers).
Resumable: one JSON file per run in runs/grid_<site>/res, finished runs are skipped.
Usage: python grid.py <site A|B> <n_procs> [shard k/K]
"""
import os, sys, json, itertools, hashlib, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore

PEN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SEEDS = [int(s) for s in os.environ.get('GEOTZ_SEEDS', '0,1,2,3,4,5,6,7').split(',')]   # 0-3 selection, 4-7 evaluation
T_END, WARM = 1500.0, 300.0


EXT = [(0.7, 0.85, 4), (0.8, 0.85, 4)]   # high-penetration extension, reported separately


def scenarios(ext=False):
    sc = [(p, q, 4) for p in PEN for q in (0.7, 0.85, 1.0)]      # demand relative to the observed level
    sc += [(p, 0.85, P) for p in PEN for P in (2, 8)]
    return sc + (EXT if ext else [])


def layouts(site):
    if site == 'B':
        return [None] + [float(x) for x in range(-325, 176, 50)]               # DL start, 50 m grid
    return [None] + [dict(e=float(e), g=float(g)) for e in (-130, -90, -50, -10) for g in (40, 80, 120, 160, 200)]


def key(site, lay, p, q, P, seed):
    return f'{site}|{json.dumps(lay, sort_keys=True)}|{p}|{q}|{P}|{seed}'


def fname(k):
    return hashlib.md5(k.encode()).hexdigest()[:16] + '.json'


def job(a):
    site, lay, p, q, P, seed, outdir = a
    k = key(site, lay, p, q, P, seed)
    f = os.path.join(outdir, fname(k))
    if os.path.exists(f):
        return 0
    try:
        mod = __import__('site_b' if site == 'B' else 'site_a')
        vb = mod.V_B if site == 'B' else mod.V_BN
        r = simcore.run(mod, lay, q, p, P, 1000 + seed, os.path.join(os.path.dirname(outdir), f'w{os.getpid()}'), vb,
                        t_end=T_END, warm=WARM)
        r['key'] = k; r['layout'] = lay
        tmp = f + '.tmp'
        json.dump(r, open(tmp, 'w')); os.replace(tmp, f)
        return 1
    except Exception as e:
        open(os.path.join(os.path.dirname(outdir), 'errors.txt'), 'a').write(f'{k}\t{e!r}\n')
        return -1


if __name__ == '__main__':
    site, nproc = sys.argv[1], int(sys.argv[2])
    shard = sys.argv[3] if len(sys.argv) > 3 else '0/1'
    only_ext = 'ext' in sys.argv[4:]
    reverse = 'rev' in sys.argv[4:]
    k_, K_ = map(int, shard.split('/'))
    base = f'runs/grid_{site}' + os.environ.get('GEOTZ_GRID_SUFFIX', '')
    outdir = os.path.join(base, 'res'); os.makedirs(outdir, exist_ok=True)
    mod = __import__('site_b' if site == 'B' else 'site_a')
    vb = mod.V_B if site == 'B' else mod.V_BN
    for lay in layouts(site):                      # build all networks once (no races)
        mod.build_net(os.path.join(base, 'nets'), lay, vb)
    scs = EXT if only_ext else scenarios()
    jobs = [(site, lay, p, q, P, s, outdir) for (p, q, P) in scs for s in SEEDS for lay in layouts(site)]
    jobs = [j for i, j in enumerate(jobs) if i % K_ == k_]
    random.Random(0).shuffle(jobs)
    if reverse:
        jobs = jobs[::-1]          # a second server can take a shard from the other end
    with Pool(nproc) as pool:
        n = 0
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            n += 1
            if n % 200 == 0:
                print(n, len(jobs), flush=True)
    print('done', flush=True)
