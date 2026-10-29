# Script from the 2026-10-19 thesis session (transcript command s042).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
data=json.load(open('imbalance.json'))
groups=[('WC08','WC08 (straight)'),('WC10','WC10 (straight)'),('helix','WC10 helical (UID18)')]
fig,ax=plt.subplots(figsize=(6.3,3.6))
plt.rcParams.update({'font.family':'serif','font.size':9})
for i,(g,lab) in enumerate(groups):
    vals=[r['imb'] for r in data if r['group']==g]
    rng=np.random.default_rng(1)
    x=i+rng.uniform(-0.18,0.18,len(vals))
    ax.scatter(x, vals, s=18, color=['#4C72B0','#DD8452','#55A868'][i], zorder=3)
    ax.hlines(np.median(vals), i-0.3, i+0.3, color='k', lw=1.2, zorder=4)
ax.axhspan(-1,1,color='0.85',zorder=0)
ax.axhline(0,color='0.5',lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in groups])
ax.set_ylabel('Net mass imbalance at last iteration [\\%]')
ax.set_ylim(-35,18)
ax.text(2.45,0.0,'shaded: $\\pm1\\,\\%$',va='center',ha='left',fontsize=8, color='0.3')
ax.grid(axis='y',color='0.9',lw=0.5,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('imbalance_all_runs.pdf')
print('saved')