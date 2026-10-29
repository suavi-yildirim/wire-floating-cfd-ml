"""Compact true-scale profile figure: 0.5 m and 1.0 m marked wires, two columns, equal axis scale, shared limits."""
import sys, os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
A, OUT = sys.argv[1:3]
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                     "grid.linewidth": 0.4, "axes.linewidth": 0.6, "pdf.fonttype": 42})
W = pd.read_csv(f"{A}/wire_metrics.csv"); W = W[(W.status == "ok") & (W.uid.isin([1, 5, 9, 17, 21]))]   # 0.5 m wires only (1.0 m wires leave the rule range; 2.0 m wires are partial)
def section(u): return 0.5 if 1 <= u <= 4 else 1.0 if 5 <= u <= 8 else 1.5 if 9 <= u <= 12 else 4.0 if 17 <= u <= 20 else 6.0
def curve_mm(sheet):
    c = pd.read_csv(f"{A}/{sheet}_curve.csv"); ppm = float(W.loc[W.sheet == sheet, "px_per_mm"].iloc[0]); x, y = c.x.values, c.y.values
    ux, uy = x[-1] - x[0], y[-1] - y[0]; L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    return ((x - x[0]) * ux + (y - y[0]) * uy) / ppm, (-(x - x[0]) * uy + (y - y[0]) * ux) / ppm
items = []
for (u, i), g in W.groupby(["uid", "instance"]):
    pl = dict(zip(g.plane, g.sheet))
    if {"X", "Y"} <= set(pl):
        s1, e1 = curve_mm(pl["X"]); s2, e2 = curve_mm(pl["Y"]); lo, hi = max(s1.min(), s2.min()), min(s1.max(), s2.max()); sg = np.linspace(lo, hi, 300)
        items.append((int(u), int(i), sg, np.interp(sg, s1, e1), np.interp(sg, s2, e2), ""))
    else:
        k = list(pl)[0]; s, e = curve_mm(pl[k]); z = np.full_like(s, np.nan)
        items.append((int(u), int(i), s, e if k == "X" else z, e if k == "Y" else z, f" ({k} only)"))
emax = max(np.nanmax(np.abs(np.r_[it[3], it[4]])) for it in items) * 1.2; smax = 560
n = len(items); nr = int(np.ceil(n / 2))
fig, axs = plt.subplots(nr, 2, figsize=(7.16, 0.55 * nr + 0.7), squeeze=False)
for k, it in enumerate(items):
    ax = axs[k % nr, k // nr]; u, i, sg, ex, ey, tag = it
    ax.plot(sg, ex, color="#002F6C", lw=0.9, label="Plane X"); ax.plot(sg, ey, color="#F5821F", lw=0.9, ls="--", label="Plane Y (+90°)")
    if not tag: ax.plot(sg, np.hypot(ex, ey), color="#555555", lw=0.7, ls=":", label=r"Resultant $r$")
    ax.axhline(0, color="#888888", lw=0.5); ax.set_xlim(-10, smax); ax.set_ylim(-emax, emax); ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"UID {u}, {section(u):g} mm², inst. {i}{tag}", fontsize=6.5, loc="left", pad=1); ax.tick_params(labelsize=5.5)
    ax.set_yticks([-10, 0, 10]); 
for k in range(n, nr * 2): axs[k % nr, k // nr].axis("off")
for c in range(2):
    for r in range(nr - 1):
        if axs[r, c].axison: axs[r, c].set_xticklabels([])
    last = max(r for r in range(nr) if axs[r, c].axison); axs[last, c].set_xlabel(r"$s$ (mm)", fontsize=6.5)
    for r in range(nr):
        if axs[r, c].axison and r != last: axs[r, c].set_xticklabels([])
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
fig.text(0.99, 0.01, "equal scale in length and deviation; ticks ±10 mm", ha="right", fontsize=5.5, color="#555555")
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.15, w_pad=0.6)
fig.savefig(f"{OUT}.pdf"); fig.savefig(f"{OUT}.png", dpi=220)
print(n, "wires, size", fig.get_size_inches())
