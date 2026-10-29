# Script from the 2026-10-19 thesis session (transcript command s057).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,
                     'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7})
data=json.load(open('imbalance_final.json'))
d8=[r for r in data if r['case']=='WC10_UID08'][0]
d8.update(iter=166, net=1.228713047116072e-04, mdot=0.002457426094232144, imb=100*1.228713047116072e-04/0.002457426094232144, chg20=9.327, unconverged=True)
json.dump(data, open('imbalance_with_uid08.json','w'), indent=1)
groups=[('WC08','WC08 straight','#2F5D8C',8.0),('WC10','WC10 straight','#4E8A5A',10.0),('helix','WC10 helical (UID18)','#C2703D',10.0)]
fig,ax=plt.subplots(figsize=(6.3,4.0))
ax.axhspan(-10,10,color='0.92',zorder=0,lw=0); ax.axhline(0,color='0.45',lw=0.6,zorder=1)
labels=[]
for i,(g,lab,col,dia) in enumerate(groups):
    rows=[r for r in data if r['group']==g]
    vals=np.array([r['imb'] for r in rows]); n=len(vals)
    k=int(np.sum(np.abs(vals)<=10)); s=(dia*1.1)**2
    rng=np.random.default_rng(3); x=i+rng.uniform(-0.2,0.2,n)
    for j,r in enumerate(rows):
        if r.get('unconverged'):
            ax.scatter([x[j]], [vals[j]], s=s, marker='^', facecolor='white', edgecolor=col, lw=1.2, zorder=4)
            ax.annotate('UID08: not converged\n(166 it.)', (x[j],vals[j]), xytext=(10,-2), textcoords='offset points', fontsize=7.5, va='center', color='0.25')
            continue
        ins=abs(vals[j])<=10
        ax.scatter([x[j]], [vals[j]], s=s, color=col if not ins else 'white', edgecolor=col, lw=1.0, zorder=3)
    med=float(np.median(vals))
    ax.hlines(med, i-0.3, i+0.3, color='k', lw=1.2, zorder=5)
    ax.text(i+0.33, med, f'median {med:+.1f} %', va='center', ha='left', fontsize=8.5)
    labels.append(f'{lab}\n$n={n}$, {k} within $\\pm10\\,\\%$')
ax.set_yscale('symlog', linthresh=10, linscale=0.6); ax.set_ylim(-40,40)
ax.set_yticks([-30,-10,-5,0,5,10,30]); ax.set_yticklabels(['$-30$','$-10$','$-5$','$0$','$5$','$10$','$30$'])
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlim(-0.5,2.85)
ax.set_ylabel('Net mass imbalance at last iteration [%]\n(symmetric log scale)')
ax.text(2.8,-38,'filled: outside $\\pm10\\,\\%$;  open: inside;  bar: median;\ntriangle: run not converged at last iteration',ha='right',va='bottom',fontsize=7.5,color='0.3')
ax.grid(axis='y',color='0.9',lw=0.5,zorder=0)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('imbalance_with_uid08.pdf')
import pymupdf
pymupdf.open('imbalance_with_uid08.pdf')[0].get_pixmap(dpi=150).save('imb_uid08_preview.png')
for g in ['WC08','WC10','helix']:
    v=np.array([r['imb'] for r in data if r['group']==g])
    print(g, len(v), 'within1', int(np.sum(abs(v)<=1)), 'within10', int(np.sum(abs(v)<=10)), 'median', round(float(np.median(v)),2))