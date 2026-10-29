"""Straightness (two-plane photo metrics) vs WFE stable levitation pressure, all matched wires.

Input : <B>/twoplane_with_pressure.csv  (join_pressure.py: columns group, photos, uid, instance, section, length, tube,
        rmax_mm, rrms_mm, status, n_stable, p_mean, p_sd)
        <B>/twoplane_fullspan_metrics.csv (full free span, 1.0 m wires; scale extrapolated beyond the 500 mm rule)
Output: <B>/straightness_pressure_table.csv, <B>/straightness_pressure_stats.csv,
        <B>/figures/fig_pressure_dev_vs_straightness_all.pdf/.png, fig_twoplane_profiles_<len>.pdf/.png
Relative pressure deviation: p_rel = (p_mean - mean over instances of the same UID/length/tube) / that mean.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl, matplotlib.pyplot as plt

B = sys.argv[1]; FIG = os.path.join(B, "figures"); os.makedirs(FIG, exist_ok=True)
J = pd.read_csv(os.path.join(B, "twoplane_with_pressure.csv"))
F = pd.read_csv(os.path.join(B, "twoplane_fullspan_metrics.csv"))
J = J.merge(F[["group", "rmax_mm", "rrms_mm", "span_mm", "s_to_mm"]].rename(columns=lambda c: c if c == "group" else "full_" + c),
            on="group", how="left")
J = J[(J.n_stable > 0) & J.p_mean.notna()].copy()
J_all = J.copy()
J = J[J.status_final == "ok"].copy()          # QC: both views tracked over the 90-490 mm window without track jumps
key = ["uid", "length", "tube"]
J["n_inst_photo"] = J.groupby(key).instance.transform("nunique")
J["p_ref"] = J.groupby(key).p_mean.transform("mean")
J["p_rel_pct"] = 100 * (J.p_mean - J.p_ref) / J.p_ref
J["r_rel"] = J.rmax_mm - J.groupby(key).rmax_mm.transform("mean")          # within-type centring of straightness
J_all.merge(J[["group", "p_ref", "p_rel_pct", "n_inst_photo"]], on="group", how="left").to_csv(
    os.path.join(B, "straightness_pressure_table.csv"), index=False)

rows = []
def corr(D, x, y, label):
    D = D.dropna(subset=[x, y])
    if len(D) < 4: return
    rs, ps = stats.spearmanr(D[x], D[y]); rp, pp = stats.pearsonr(D[x], D[y])
    rows.append(dict(subset=label, x=x, y=y, n=len(D), spearman_rho=rs, spearman_p=ps, pearson_r=rp, pearson_p=pp))
U = J[J.n_inst_photo >= 3]                       # only wire types with >= 3 photographed instances carry within-type information
for x in ["rmax_mm", "rrms_mm", "r_rel"]:
    corr(U, x, "p_rel_pct", "all types, >=3 instances")
    for L in sorted(U.length.unique()):
        corr(U[U.length == L], x, "p_rel_pct", f"length {L:g} m")
corr(U[U.length == 1.0], "full_rmax_mm", "p_rel_pct", "length 1 m, full span")
corr(U[U.length == 1.0], "full_rrms_mm", "p_rel_pct", "length 1 m, full span")
# within-type rank agreement: per UID, Spearman between rmax and p
for (u, L, t), g in U.groupby(key):
    if len(g) >= 3:
        r, p = stats.spearmanr(g.rmax_mm, g.p_mean)
        rows.append(dict(subset=f"UID {int(u)} {L:g} m WC{int(t):02d}", x="rmax_mm", y="p_mean", n=len(g), spearman_rho=r, spearman_p=p))
S = pd.DataFrame(rows); S.to_csv(os.path.join(B, "straightness_pressure_stats.csv"), index=False)
pd.set_option("display.width", 220)
print(J_all[["photos", "uid", "instance", "section", "length", "rmax_mm", "rrms_mm", "full_rmax_mm", "status_final", "n_stable", "p_mean", "p_sd"]].round(2).to_string(index=False))
print(S.round(3).to_string(index=False))

# ---------------- figures
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                     "grid.linewidth": 0.4, "axes.linewidth": 0.6, "pdf.fonttype": 42})
SEC = {0.5: ("#F5821F", "o"), 1.0: ("#002F6C", "s"), 1.5: ("#6A8A22", "^"), 4.0: ("#34677D", "D"), 6.0: ("#8C1D40", "v")}
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
for ax, L in zip(axs, [0.5, 1.0]):
    D = J[(J.length == L) & (J.n_inst_photo >= 2)]       # figure: types with >= 2 QC-ok instances (stats need >= 3)
    for s, g in D.groupby("section"):
        c, m = SEC.get(s, ("#555555", "o"))
        ax.scatter(g.rmax_mm, g.p_rel_pct, color=c, marker=m, s=34, edgecolor="white", lw=0.6, zorder=3,
                   label=rf"$A$ = {s:g} mm$^2$", alpha=1.0)
        for r in g.itertuples():
            ax.annotate(str(int(r.instance)), (r.rmax_mm, r.p_rel_pct), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color="#555555")
    ax.axhline(0, color="#888888", lw=0.5)
    c = S[(S.subset == f"length {L:g} m") & (S.x == "rmax_mm")]
    if len(c):
        c = c.iloc[0]; ax.text(0.98, 0.04, rf"Spearman $\rho$ = {c.spearman_rho:.2f}, $p$ = {c.spearman_p:.2f}, $n$ = {c.n}",
                                transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.set_title(f"Wire length {L:g} m (WC08), {D.uid.nunique()} wire types", fontsize=9); ax.set_xlabel(r"Max. resultant deviation $r_{\max}$ (mm)")
    ax.legend(loc="upper right", frameon=False)
axs[0].set_ylabel(r"$(\bar{p}_i-\bar{p}_{\mathrm{UID}})/\bar{p}_{\mathrm{UID}}$ (%)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_pressure_dev_vs_straightness_all.pdf")); fig.savefig(os.path.join(FIG, "fig_pressure_dev_vs_straightness_all.png"), dpi=200)

for L in sorted(J_all.length.unique()):
    D = J_all[J_all.length == L].sort_values(["uid", "instance"]); n = len(D)
    if n == 0: continue
    full = L > 0.5
    nc = 4; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.16, 1.65 * nr + 0.4), sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axs.ravel(), D.itertuples()):
        fn = os.path.join(B, f"twoplane_full_group{r.group}.csv" if full else f"twoplane_group{r.group}.csv")
        if not os.path.exists(fn): ax.axis("off"); continue
        d = pd.read_csv(fn)
        ax.plot(d.s_mm, d.e_y_mm, color="#002F6C", lw=1.0, label="View A")
        ax.plot(d.s_mm, d.e_z_mm, color="#F5821F", lw=1.0, ls="--", label="View B (rotated 90°)")
        ax.plot(d.s_mm, d.r_mm, color="#555555", lw=0.8, ls=":", label=r"Resultant $r$")
        if full: ax.axvspan(500, max(510, d.s_mm.max()), color="#eeeeee", zorder=0, lw=0)
        ax.axhline(0, color="#888888", lw=0.5)
        ax.set_title(f"UID {int(r.uid)} / inst. {int(r.instance)}" + ("" if r.status_final == "ok" else " (QC fail)"), fontsize=7.5)
    for ax in axs.ravel()[n:]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel(r"$s$ (mm)")
    for ax in axs[:, 0]: ax.set_ylabel(r"$e$ (mm)")
    h, l = axs[0, 0].get_legend_handles_labels()
    if full:
        h.append(mpl.patches.Patch(color="#eeeeee")); l.append("Beyond rule (scale extrapolated)")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06 if nr < 3 else 0.035, 1, 1), h_pad=0.6)
    tag = f"{L:.1f}m".replace(".", "p")
    fig.savefig(os.path.join(FIG, f"fig_twoplane_profiles_{tag}.pdf")); fig.savefig(os.path.join(FIG, f"fig_twoplane_profiles_{tag}.png"), dpi=200)
