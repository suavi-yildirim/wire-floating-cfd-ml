# Script from the 2026-10-19 thesis session (transcript command s051).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import zipfile, os, json, re
def parse(txt):
    out=[]
    for l in txt.splitlines():
        q=l.split()
        if len(q)==2 and q[0].isdigit():
            out.append((int(q[0]),float(q[1])))
    return out
def case_metrics(net_txt, mf_txt):
    net=parse(net_txt); mf=parse(mf_txt)
    if not net or not mf: return None
    it=net[-1][0]; last=net[-1][1]; mref=mf[-1][1]
    v=[x[1] for x in mf[-20:]]
    return dict(iter=it, net=last, mdot=mref, imb=100*last/mref if mref else float('nan'),
                mdot_chg20=100*(max(v)-min(v))/mref if mref else float('nan'))
z=zipfile.ZipFile('/Users/Suavi/Downloads/Report_Data.zip')
new={}
for n in z.namelist():
    if n.startswith('__MACOSX') or not n.endswith('net-mass-flow-rfile.out'): continue
    parts=n.split('/')
    if len(parts)!=3: continue   # top-level case folders only
    case=parts[1]
    mf_name=n.replace('net-mass-flow-rfile.out','mass-flow-inlet-rfile.out')
    if mf_name not in z.namelist(): continue
    m=case_metrics(z.read(n).decode('utf-8','ignore'), z.read(mf_name).decode('utf-8','ignore'))
    if m: new[case]=m
# helix top-level folders are under Report_Data/Helix_...
for n in z.namelist():
    if n.startswith('Report_Data/Helix_') and n.count('/')==2 and n.endswith('net-mass-flow-rfile.out'):
        case=n.split('/')[1]
        mf_name=n.replace('net-mass-flow-rfile.out','mass-flow-inlet-rfile.out')
        if mf_name in z.namelist():
            new[case]=case_metrics(z.read(n).decode('utf-8','ignore'), z.read(mf_name).decode('utf-8','ignore'))
old={r['case']:r for r in json.load(open('imbalance.json'))}
print('new cases', len(new), 'old cases', len(old))
both=sorted(set(new)&set(old))
print('in both', len(both), 'only new', sorted(set(new)-set(old))[:20], 'only old', sorted(set(old)-set(new))[:20])
print(f"{'case':28s} {'old imb':>9s} {'new imb':>9s} {'old it':>6s} {'new it':>6s} {'new chg20%':>10s}")
for c in both:
    o=old[c]; nn=new[c]
    flag='' if abs(o['imb']-nn['imb'])<0.05 and o['iter']==nn['iter'] else '  <-'
    print(f"{c:28s} {o['imb']:+9.2f} {nn['imb']:+9.2f} {o['iter']:6d} {nn['iter']:6d} {nn['mdot_chg20']:10.3f}{flag}")
json.dump(new, open('new_metrics.json','w'), indent=1)