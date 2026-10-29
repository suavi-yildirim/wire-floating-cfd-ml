# Script from the 2026-10-19 thesis session (transcript command s041).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import os, json
def load(p):
    rows=[]
    for ln in open(p, encoding='utf-8', errors='ignore'):
        parts=ln.split()
        if len(parts)==2:
            try: rows.append((int(float(parts[0])), float(parts[1])))
            except: pass
    return rows
out=[]
for d in sorted(os.listdir('.')):
    n=os.path.join(d,'net-mass-flow-rfile.out'); m=os.path.join(d,'mass-flow-inlet-rfile.out')
    if not (os.path.isdir(d) and os.path.exists(n) and os.path.exists(m)): continue
    net=load(n); mf=load(m)
    if not net or not mf: continue
    it,last=net[-1]; mref=mf[-1][1]
    grp='helix' if d.startswith('Helix') else d[:4]
    out.append(dict(case=d, group=grp, iter=it, net=last, mdot=mref, imb=100*last/mref))
json.dump(out, open('/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/imbalance.json','w'), indent=1)
import collections
for g in ['WC08','WC10','helix']:
    v=[abs(r['imb']) for r in out if r['group']==g]
    print(g, len(v), 'below1%:', sum(x<1 for x in v), 'below3%:', sum(x<3 for x in v), 'median abs:', round(sorted(v)[len(v)//2],2))