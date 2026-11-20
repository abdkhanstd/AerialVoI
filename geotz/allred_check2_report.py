"""Prints the results of allred_check2.py in compact form: regrets, WB DL shares, steps and equivalences."""
import os, json
import voi

o = json.load(open(os.path.join(voi.RES, 'allred_check2.json')))
refs = ['B', 'B2', 'R3', 'R4']
print('regret deploy / place (orig -> nar)')
for L in o['orig']['B']['regret']:
    print(f'{L:5s}', '  '.join(f"{r}: {o['orig'][r]['regret'][L]['deploy']:5.1f}->{o['nar'][r]['regret'][L]['deploy']:5.1f} / "
                               f"{o['orig'][r]['regret'][L]['place']:5.1f}->{o['nar'][r]['regret'][L]['place']:5.1f}" for r in refs))
print('WB DL share of planner (8 seeds; halves)')
for L in o['orig']['B']['wb_dl']:
    if '#' in L:
        continue
    print(f'{L:5s}', f"orig {o['orig']['B']['wb_dl'][L]:.2f} ({o['orig']['B']['wb_dl'][L + '#1']:.2f}, {o['orig']['B']['wb_dl'][L + '#2']:.2f})",
          f"nar {o['nar']['B']['wb_dl'][L]:.2f} ({o['nar']['B']['wb_dl'][L + '#1']:.2f}, {o['nar']['B']['wb_dl'][L + '#2']:.2f})")
print('steps: orig mean [95%] p eq | nar mean [95%] 90% p eq')
for k in o['orig']['B']['steps']:
    if 'threshold' in k or 'never' in k or 'georule' in k:
        continue
    print(k)
    for r in refs:
        a, b = o['orig'][r]['steps'][k], o['nar'][r]['steps'][k]
        print(f"   {r:3s} {a['mean']:6.2f} [{a['lo']:6.2f},{a['hi']:6.2f}] p={a['p']:.3f} eq={a['eq2']!s:5s} | "
              f"{b['mean']:6.2f} [{b['lo']:6.2f},{b['hi']:6.2f}] ({b['ci90'][0]:5.2f},{b['ci90'][1]:5.2f}) p={b['p']:.3f} eq={b['eq2']!s:5s}"
              f"{' floor' if b['floor'] else ''}")
print('rules (nar): mean range over refs')
for k in o['nar']['B']['steps']:
    if any(x in k for x in ('threshold', 'never', 'georule')):
        v = [o['nar'][r]['steps'][k] for r in refs]
        print(f'  {k:24s}', ' '.join(f"{x['mean']:6.1f}(p={x['p']:.3f})" for x in v))
