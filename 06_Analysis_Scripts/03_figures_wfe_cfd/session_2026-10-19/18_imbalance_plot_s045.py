# Script from the 2026-10-19 thesis session (transcript command s045).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,
                     'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7})
data=json.load(open('imbalance.json'))
groups=[('WC08','WC08 straight'),('WC10','WC10 straight'),('helix','WC10 helical (UID18)')]
colors=['#2F5D8C','#C2703D','#4E8A5A']
fig,ax=plt.subplots(figsize=(6.3,4.0))
ax.axhspan(-10,10,color='0.92',zorder=0,lw=0)
ax.axhline(0,color='0.45',lw=0.6,zorder=1)
labels=[]
for i,(g,lab) in enumerate(groups):
    vals=np.array([r['imb'] for r in data if r['group']==g]); n=len(vals)
    k=int(np.sum(np.abs(vals)<=10))
    rng=np.random.default_rng(3)
    x=i+rng.uniform(-0.12,0.12,n)
    ins=np.abs(vals)<=10
    ax.scatter(x[~ins], vals[~ins], s=24, color=colors[i], edgecolor='white', lw=0.5, zorder=3)
    ax.scatter(x[ins], vals[ins], s=24, facecolor='white', edgecolor=colors[i], lw=1.0, zorder=3)
    med=float(np.median(vals))
    ax.hlines(med, i-0.28, i+0.28, color='k', lw=1.2, zorder=4)
    ax.text(i+0.31, med, f'median {med:+.1f} %', va='center', ha='left', fontsize=8.5)
    labels.append(f'{lab}\n$n={n}$, {k} within $\\pm10\\,\\%$')
ax.set_yscale('symlog', linthresh=10, linscale=0.6)
ax.set_ylim(-40,40)
ax.set_yticks([-30,-10,-5,0,5,10,30])
ax.set_yticklabels(['$-30$','$-10$','$-5$','$0$','$5$','$10$','$30$'])
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_xlim(-0.5,2.7)
ax.set_ylabel('Net mass imbalance at last iteration [%]\n(symmetric log scale)')
ax.text(2.65,-38,'filled: outside $\\pm10\\,\\%$;  open: inside;  bar: median',ha='right',va='bottom',fontsize=7.5,color='0.3')
ax.grid(axis='y',color='0.9',lw=0.5,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('imbalance_improved2.pdf')
import pymupdf
pymupdf.open('imbalance_improved2.pdf')[0].get_pixmap(dpi=150).save('imb_preview2.png'); print('saved')