# Script from the 2026-10-19 thesis session (transcript command s052).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,
                     'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7})
new=json.load(open('new_metrics.json'))
data=[]
for c,m in sorted(new.items()):
    grp='helix' if c.startswith('Helix') else c[:4]
    data.append(dict(case=c, group=grp, iter=m['iter'], net=m['net'], mdot=m['mdot'], imb=m['imb'], chg20=m['mdot_chg20']))
json.dump(data, open('imbalance_final.json','w'), indent=1)
with open('imbalance_final.csv','w') as f:
    f.write('case,group,iterations,net_mass_flow_kg_s,inlet_mass_flow_kg_s,imbalance_pct,inlet_mdot_change_last20_pct\n')
    for r in data: f.write(f"{r['case']},{r['group']},{r['iter']},{r['net']:.6e},{r['mdot']:.6f},{r['imb']:.4f},{r['chg20']:.4f}\n")
for g in ['WC08','WC10','helix']:
    v=np.array([r['imb'] for r in data if r['group']==g])
    print(g, len(v), 'within1', int(np.sum(abs(v)<=1)), 'within10', int(np.sum(abs(v)<=10)), 'median', round(float(np.median(v)),2))
groups=[('WC08','WC08 straight','#2F5D8C',8.0),('WC10','WC10 straight','#4E8A5A',10.0),('helix','WC10 helical (UID18)','#C2703D',10.0)]
fig,ax=plt.subplots(figsize=(6.3,4.0))
ax.axhspan(-10,10,color='0.92',zorder=0,lw=0); ax.axhline(0,color='0.45',lw=0.6,zorder=1)
labels=[]
for i,(g,lab,col,dia) in enumerate(groups):
    vals=np.array([r['imb'] for r in data if r['group']==g]); n=len(vals)
    k=int(np.sum(np.abs(vals)<=10)); s=(dia*1.1)**2
    rng=np.random.default_rng(3); x=i+rng.uniform(-0.2,0.2,n); ins=np.abs(vals)<=10
    ax.scatter(x[~ins], vals[~ins], s=s, color=col, edgecolor='white', lw=0.5, zorder=3)
    ax.scatter(x[ins], vals[ins], s=s, facecolor='white', edgecolor=col, lw=1.0, zorder=3)
    med=float(np.median(vals))
    ax.hlines(med, i-0.3, i+0.3, color='k', lw=1.2, zorder=4)
    ax.text(i+0.33, med, f'median {med:+.1f} %', va='center', ha='left', fontsize=8.5)
    labels.append(f'{lab}\n$n={n}$, {k} within $\\pm10\\,\\%$')
ax.set_yscale('symlog', linthresh=10, linscale=0.6); ax.set_ylim(-40,40)
ax.set_yticks([-30,-10,-5,0,5,10,30]); ax.set_yticklabels(['$-30$','$-10$','$-5$','$0$','$5$','$10$','$30$'])
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlim(-0.5,2.75)
ax.set_ylabel('Net mass imbalance at last iteration [%]\n(symmetric log scale)')
ax.text(2.7,-38,'filled: outside $\\pm10\\,\\%$;  open: inside;  bar: median;\nmarker size: tube diameter (8 mm, 10 mm)',ha='right',va='bottom',fontsize=7.5,color='0.3')
ax.grid(axis='y',color='0.9',lw=0.5,zorder=0)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('imbalance_final.pdf'); print('saved')