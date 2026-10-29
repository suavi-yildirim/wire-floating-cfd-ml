import sys, os, glob, pandas as pd, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
B = sys.argv[1]
J = pd.read_csv(os.path.join(B, "twoplane_with_pressure.csv"))
COL = {1: "Orange", 5: "Blue", 9: "Yellow", 17: "Purple", 21: "Red"}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                     "grid.linewidth": 0.4, "axes.linewidth": 0.6, "pdf.fonttype": 42})
n = len(J); nc = 3; nr = int(np.ceil(n / nc))
fig, axs = plt.subplots(nr, nc, figsize=(7.16, 2.0 * nr + 0.35), sharex=True, sharey=True, squeeze=False)
for ax, r in zip(axs.ravel(), J.itertuples()):
    d = pd.read_csv(os.path.join(B, f"twoplane_group{r.group}.csv"))
    ax.plot(d.s_mm, d.e_y_mm, color="#002F6C", lw=1.2, label="Plane A")
    ax.plot(d.s_mm, d.e_z_mm, color="#F5821F", lw=1.2, ls="--", label="Plane B (+90°)")
    ax.plot(d.s_mm, d.r_mm, color="#555555", lw=0.9, ls=":", label=r"Resultant $r$")
    ax.axhline(0, color="#888888", lw=0.5)
    flag = "\n(partial span, see text)" if r.status != "ok" else "\n"
    ax.set_title(f"UID {int(r.uid)}, {r.section:g} mm², inst. {int(r.instance)}{flag}", fontsize=8)
for ax in axs[-1]: ax.set_xlabel(r"Position on rule $s$ (mm)")
for ax in axs[:, 0]: ax.set_ylabel(r"Deviation $e$ (mm)")
for ax in axs.ravel()[n:]: ax.axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8)
fig.savefig(os.path.join(B, "fig_twoplane_profiles.pdf")); fig.savefig(os.path.join(B, "fig_twoplane_profiles.png"), dpi=200)
