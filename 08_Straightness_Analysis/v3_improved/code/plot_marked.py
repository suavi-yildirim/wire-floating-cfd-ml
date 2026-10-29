"""Thesis-style figures for the hand-marked wires (more/analysis). Style follows the earlier figures (plot_twoplane.py,
straightness_pressure_analysis.py). usage: plot_marked.py <analysis folder> <out folder>"""
import sys, os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
A, FIG = sys.argv[1:3]; os.makedirs(FIG, exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                     "grid.linewidth": 0.4, "axes.linewidth": 0.6, "pdf.fonttype": 42})
W = pd.read_csv(os.path.join(A, "wire_metrics.csv"))
def section(uid):   # wire-type cross-section from UID (1-4: 0.5, 5-8: 1.0, 9-12: 1.5, 17-20: 4.0, 21-24: 6.0 mm^2)
    return {0: None}.get(0) if False else (0.5 if 1 <= uid <= 4 else 1.0 if 5 <= uid <= 8 else 1.5 if 9 <= uid <= 12 else 4.0 if 17 <= uid <= 20 else 6.0 if 21 <= uid <= 24 else np.nan)
def length(uid):   # UID position in the 4-block gives the length: 1 -> 0.5 m, 2 -> 1.0 m, 3 -> 2.0 m, 0 -> 3.5 m
    return {1: 0.5, 2: 1.0, 3: 2.0, 0: 3.5}[uid % 4]
def curve_mm(sheet):
    c = pd.read_csv(os.path.join(A, f"{sheet}_curve.csv"))
    ppm = float(W.loc[W.sheet == sheet, "px_per_mm"].iloc[0])
    x, y = c.x.values, c.y.values
    ux, uy = x[-1] - x[0], y[-1] - y[0]; L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    s = ((x - x[0]) * ux + (y - y[0]) * uy) / ppm
    e = (-(x - x[0]) * uy + (y - y[0]) * ux) / ppm
    return s, e
# ---- Figure 1: two-plane profiles per wire, drawn at TRUE proportions (equal scale on s and e, shared limits)
pairs = []
for (u, i), g in W[W.status == "ok"].groupby(["uid", "instance"]):
    if set(g.plane) == {"X", "Y"}:
        pairs.append((int(u), int(i), g[g.plane == "X"].sheet.iloc[0], g[g.plane == "Y"].sheet.iloc[0]))
data = []
for (u, i, sx, sy) in pairs:
    s1, e1 = curve_mm(sx); s2, e2 = curve_mm(sy)
    lo, hi = max(s1.min(), s2.min()), min(s1.max(), s2.max()); sg = np.linspace(lo, hi, 400)
    ex = np.interp(sg, s1, e1); ey = np.interp(sg, s2, e2)
    data.append((u, i, sg, ex, ey))
n = len(data); emax = max(np.abs(np.r_[d[3], d[4]]).max() for d in data) * 1.25
smax = max(d[2].max() for d in data) * 1.02
nr = n
fig, axs = plt.subplots(nr, 1, figsize=(7.16, 0.95 * nr + 0.6), squeeze=False)
for ax, (u, i, sg, ex, ey) in zip(axs.ravel(), data):
    ax.plot(sg, ex, color="#002F6C", lw=1.2, label="Plane X")
    ax.plot(sg, ey, color="#F5821F", lw=1.2, ls="--", label="Plane Y (+90°)")
    ax.plot(sg, np.hypot(ex, ey), color="#555555", lw=0.9, ls=":", label=r"Resultant $r$")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_xlim(0, smax); ax.set_ylim(-emax, emax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"UID {u}, {section(u):g} mm², inst. {i}", fontsize=8, loc="left")
    ax.set_ylabel(r"$e$ (mm)", fontsize=8)
for ax in axs.ravel(): ax.tick_params(labelsize=7)
axs[-1, 0].set_xlabel(r"Position on rule $s$ (mm)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.4)
fig.savefig(os.path.join(FIG, "fig_marked_twoplane_profiles.pdf")); fig.savefig(os.path.join(FIG, "fig_marked_twoplane_profiles.png"), dpi=200)
# ---- Figure 1b: same profiles, narrower and taller panels; the vertical exaggeration factor is printed on the figure
PANEL_W, PANEL_H = 4.0, 1.25                                  # inches per panel
VE = (PANEL_H / (2 * emax)) / (PANEL_W / smax)                 # vertical exaggeration relative to true proportions
fig, axs = plt.subplots(nr, 1, figsize=(PANEL_W + 1.0, PANEL_H * nr + 0.9), squeeze=False)
for ax, (u, i, sg, ex, ey) in zip(axs.ravel(), data):
    ax.plot(sg, ex, color="#002F6C", lw=1.4, label="Plane X")
    ax.plot(sg, ey, color="#F5821F", lw=1.4, ls="--", label="Plane Y (+90°)")
    ax.plot(sg, np.hypot(ex, ey), color="#555555", lw=1.0, ls=":", label=r"Resultant $r$")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_xlim(0, smax); ax.set_ylim(-emax, emax)
    ax.set_title(f"UID {u}, {section(u):g} mm², inst. {i}", fontsize=8, loc="left")
    ax.set_ylabel(r"$e$ (mm)", fontsize=8)
for ax in axs.ravel(): ax.tick_params(labelsize=7)
axs[-1, 0].set_xlabel(rf"Position on rule $s$ (mm)   [vertical exaggeration ×{VE:.1f}]", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6)
fig.savefig(os.path.join(FIG, "fig_marked_twoplane_profiles_exaggerated.pdf")); fig.savefig(os.path.join(FIG, "fig_marked_twoplane_profiles_exaggerated.png"), dpi=200)
print(f"vertical exaggeration x{VE:.2f}")

# ---- Figure 1c: one wire per row, full width, deviation and length on the SAME physical scale (true shape)
fig, axs = plt.subplots(nr, 1, figsize=(7.16, 0.75 * n + 1.1), squeeze=False)
for ax, (u, i, sg, ex, ey) in zip(axs.ravel(), data):
    ax.plot(sg, ex, color="#002F6C", lw=1.0, label="Plane X")
    ax.plot(sg, ey, color="#F5821F", lw=1.0, ls="--", label="Plane Y (+90°)")
    ax.plot(sg, np.hypot(ex, ey), color="#555555", lw=0.8, ls=":", label=r"Resultant $r$")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_xlim(-10, smax); ax.set_ylim(-emax, emax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"UID {u}, {section(u):g} mm², inst. {i}", fontsize=7, loc="left", pad=1)
    ax.set_ylabel(r"$e$ (mm)", fontsize=6)
    ax.tick_params(labelsize=6)
for ax in axs.ravel()[:-1]: ax.set_xticklabels([])
axs.ravel()[-1].set_xlabel(r"$s$ (mm)", fontsize=7)
# scale bar (100 mm along s) below the last panel, clear of the curves
bx = axs.ravel()[-1]
bx.annotate("", xy=(100, -emax * 1.9), xytext=(0, -emax * 1.9), xycoords="data", annotation_clip=False,
            arrowprops=dict(arrowstyle="|-|", lw=0.8, color="black", mutation_scale=4))
bx.text(50, -emax * 2.25, "100 mm (true scale in both directions)", ha="center", va="top", fontsize=6, clip_on=False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.5)
fig.savefig(os.path.join(FIG, "fig_marked_twoplane_truescale.pdf")); fig.savefig(os.path.join(FIG, "fig_marked_twoplane_truescale.png"), dpi=200)

# ---- Figure 4: every marked wire (two-plane or single view) at true scale, one row each
allw = []
for (u, i), g in W[W.status == "ok"].groupby(["uid", "instance"]):
    pl = dict(zip(g.plane, g.sheet))
    if {"X", "Y"} <= set(pl):
        s1, e1 = curve_mm(pl["X"]); s2, e2 = curve_mm(pl["Y"])
        lo, hi = max(s1.min(), s2.min()), min(s1.max(), s2.max()); sg = np.linspace(lo, hi, 400)
        allw.append((int(u), int(i), sg, np.interp(sg, s1, e1), np.interp(sg, s2, e2)))
    else:
        k = list(pl)[0]; sx, ex = curve_mm(pl[k])
        allw.append((int(u), int(i), sx, ex if k == "X" else np.full_like(sx, np.nan), ex if k == "Y" else np.full_like(sx, np.nan), k))
emax_all = max(np.nanmax(np.abs(np.r_[w[3], w[4]])) for w in allw) * 1.25
smax_all = max(w[2].max() for w in allw) * 1.02
na = len(allw)
fig, axs = plt.subplots(na, 1, figsize=(7.16, 0.7 * na + 1.1), squeeze=False)
for ax, w in zip(axs.ravel(), allw):
    u, i, sg, ex, ey = w[:5]
    if len(w) == 5:
        ax.plot(sg, ex, color="#002F6C", lw=1.0, label="Plane X")
        ax.plot(sg, ey, color="#F5821F", lw=1.0, ls="--", label="Plane Y (+90°)")
        ax.plot(sg, np.hypot(ex, ey), color="#555555", lw=0.8, ls=":", label=r"Resultant $r$")
        tag = ""
    else:
        ax.plot(sg, ex if w[5] == "X" else ey, color="#002F6C" if w[5] == "X" else "#F5821F", lw=1.0)
        tag = f" (plane {w[5]} only)"
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_xlim(-10, smax_all); ax.set_ylim(-emax_all, emax_all); ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"UID {u}, {section(u):g} mm², inst. {i}{tag}", fontsize=7, loc="left", pad=1)
    ax.set_ylabel(r"$e$ (mm)", fontsize=6); ax.tick_params(labelsize=6)
for ax in axs.ravel()[:-1]: ax.set_xticklabels([])
axs.ravel()[-1].set_xlabel(r"$s$ (mm)", fontsize=7)
bx = axs.ravel()[-1]
bx.annotate("", xy=(100, -emax_all * 1.9), xytext=(0, -emax_all * 1.9), xycoords="data", annotation_clip=False,
            arrowprops=dict(arrowstyle="|-|", lw=0.8, color="black", mutation_scale=4))
bx.text(50, -emax_all * 2.25, "100 mm (true scale in both directions)", ha="center", va="top", fontsize=6, clip_on=False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.5)
fig.savefig(os.path.join(FIG, "fig_all_marked_truescale.pdf")); fig.savefig(os.path.join(FIG, "fig_all_marked_truescale.png"), dpi=200)

# ---- Figure 5: deviation and bend by wire length (every marked wire, one point per plane)
L = W[W.status == "ok"].copy()
L["length_label"] = L.length_mm.round(0)
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6))
lens = sorted(L.sheet.map(lambda sh: {1: 0.5, 2: 1.0, 3: 2.0, 0: 3.5}[int(sh.split("_")[2]) % 4]).unique())
rng = np.random.default_rng(0)
Lx = L.assign(length=L.sheet.map(lambda sh: {1: 0.5, 2: 1.0, 3: 2.0, 0: 3.5}[int(sh.split("_")[2]) % 4]))
for ax, col, ylab in [(axs[0], "dev_max_mm", r"Max deviation from chord $e_{\max}$ (mm)"), (axs[1], "max_bend_20mm_deg", r"Max bend over 20 mm (°)")]:
    for k, Lv in enumerate(lens):
        d = Lx[Lx.length == Lv][col].dropna()
        ax.scatter(k + rng.uniform(-0.08, 0.08, len(d)), d, s=22, color="#002F6C", edgecolor="white", lw=0.5, zorder=3)
        if len(d): ax.hlines(d.median(), k - 0.2, k + 0.2, color="#F5821F", lw=1.5, zorder=4)
    ax.set_xticks(range(len(lens))); ax.set_xticklabels([f"{x:g} m\n(n={int((Lx.length == x).sum())})" for x in lens], fontsize=7)
    ax.set_ylabel(ylab, fontsize=8)
axs[1].axhline(15, color="#888888", lw=0.6, ls="--"); axs[1].text(-0.3, 15.5, "15° limit", fontsize=6, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_marked_by_length.pdf")); fig.savefig(os.path.join(FIG, "fig_marked_by_length.png"), dpi=200)

# ---- Figure 2: deviation vs stable pressure (relative to the mean of the same UID)
T = pd.read_csv(os.path.join(A, "twoplane_marked.csv"))
P = pd.read_csv(os.path.join(os.path.dirname(A), "out", "twoplane.csv"))[["uid", "instance", "p_stable_mbar"]]
J = T.merge(P, on=["uid", "instance"], how="inner").dropna(subset=["p_stable_mbar"])
J = J[J.groupby("uid").uid.transform("size") >= 2]   # relative pressure needs >=2 matched instances per UID
J["p_rel_pct"] = 100 * (J.p_stable_mbar - J.groupby("uid").p_stable_mbar.transform("mean")) / J.groupby("uid").p_stable_mbar.transform("mean")
SEC = {0.5: ("#F5821F", "o"), 1.0: ("#002F6C", "s"), 1.5: ("#6A8A22", "^"), 4.0: ("#34677D", "D"), 6.0: ("#8C1D40", "v")}
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for sec, g in J.assign(sec=J.uid.map(section)).groupby("sec"):
    c, m = SEC.get(sec, ("#555555", "o"))
    ax.scatter(g.rmax_mm, g.p_rel_pct, color=c, marker=m, s=34, edgecolor="white", lw=0.6, zorder=3, label=rf"$A$ = {sec:g} mm$^2$")
    for r in g.itertuples(): ax.annotate(f"{r.uid}/{r.instance}", (r.rmax_mm, r.p_rel_pct), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color="#555555")
ax.axhline(0, color="#888888", lw=0.5)
if len(J) >= 4:
    rho, p = stats.spearmanr(J.rmax_mm, J.p_rel_pct)
    ax.text(0.98, 0.97, rf"Spearman $\rho$ = {rho:.2f}, $p$ = {p:.2f}, $n$ = {len(J)}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
ax.set_xlabel(r"Max. resultant deviation $r_{\max}$ (mm)")
ax.set_ylabel(r"$(\bar{p}_i-\bar{p}_{\mathrm{UID}})/\bar{p}_{\mathrm{UID}}$ (%)")
ax.legend(loc="upper left", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_marked_dev_vs_pressure.pdf")); fig.savefig(os.path.join(FIG, "fig_marked_dev_vs_pressure.png"), dpi=200)
print("profiles:", n, "| pressure pairs (>=2 per UID):", len(J))
if len(J)>=4: print("Spearman rho=%.2f p=%.2f n=%d" % (stats.spearmanr(J.rmax_mm, J.p_rel_pct)[0], stats.spearmanr(J.rmax_mm, J.p_rel_pct)[1], len(J)))
