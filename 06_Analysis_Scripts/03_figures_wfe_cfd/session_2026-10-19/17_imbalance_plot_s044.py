# Script from the 2026-10-19 thesis session (transcript command s044).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,
                     'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7})
data=json.load(open('imbalance.json'))
groups=[('WC08','WC08 straight'),('WC10','WC10 straight'),('helix','WC10 helical\n(UID18)')]
colors=['#2F5D8C','#C2703D','#4E8A5A']
fig,ax=plt.subplots(figsize=(6.3,3.9))
ax.axhspan(-10,10,color='0.92',zorder=0,lw=0)
ax.axhline(0,color='0.45',lw=0.6,zorder=1)
for i,(g,lab) in enumerate(groups):
    vals=np.array([r['imb'] for r in data if r['group']==g])
    rng=np.random.default_rng(3)
    x=i+rng.uniform(-0.12,0.12,len(vals))
    inside=np.abs(vals)<=10
    ax.scatter(x[~inside], vals[~inside], s=22, color=colors[i], edgecolor='white', lw=0.5, zorder=3)
    ax.scatter(x[inside], vals[inside], s=22, facecolor='white', edgecolor=colors[i], lw=1.0, zorder=3)
    med=np.median(vals)
    ax.hlines(med, i-0.28, i+0.28, color='k', lw=1.1, zorder=4)
    ax.text(i+0.3, med, f'{med:+.1f}\\,\\%', va='center', ha='left', fontsize=8.5, color='k')
ax.set_yscale('symlog', linthresh=10, linscale=0.6)
ax.set_ylim(-40,40)
ax.set_yticks([-30,-10,-5,0,5,10,30])
ax.set_yticklabels(['$-30$','$-10$','$-5$','$0$','$5$','$10$','$30$'])
ax.set_xticks(range(3))
ax.set_xticklabels([f'{g[1]}\n$n={sum(1 for r in data if r["group"]==g[0])}$' for g in groups])
ax.set_xlim(-0.5,2.6)
ax.set_ylabel('Net mass imbalance at last iteration (\\%)')
ax.text(2.55,-38,'filled: outside $\\pm 10\\,\\%$;  open: inside;  bar: median',ha='right',va='bottom',fontsize=7.5,color='0.3')
ax.grid(axis='y',color='0.9',lw=0.5,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('imbalance_improved.pdf'); print('saved')