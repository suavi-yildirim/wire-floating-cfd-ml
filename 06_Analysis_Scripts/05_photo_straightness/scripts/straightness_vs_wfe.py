"""Link photo-based straightness metrics to WFE levitation pressure (0.5 m wires).

Inputs
  argv[1]  straightness_metrics.csv   (extract_straightness.py)
  argv[2]  centrelines.csv            (extract_straightness.py)
  argv[3]  02_WFE_Processed/Stable_Phase_Summary.csv
           columns used: WIRE_section, WIRE_lenght (m), WIRE_instance, TUBE_diameter (mm),
                         System_Stable, Pressure_Stable_Average (mbar), Flow_Stable_Average (Nm^3/h)
  argv[4]  output folder
Assumption: photo instance _01.._03 == WIRE_instance 1..3 of the WC10 campaign (all 0.5 m WC10 runs on 2026-03-25).
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt

met = pd.read_csv(sys.argv[1]); cl = pd.read_csv(sys.argv[2]); wfe = pd.read_csv(sys.argv[3]); OUT = sys.argv[4]
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4})
# FAPS colour guide + marker shape as secondary encoding (protan pair green/orange is not CVD-safe on colour alone)
SEC = {1.0: ("#002F6C", "o"), 1.5: ("#F5821F", "s"), 4.0: ("#34677D", "^"), 6.0: ("#6A8A22", "D")}
SLAB = {s: rf"$A = {s:g}$ mm$^2$" for s in SEC}
INST_LS = {1: "-", 2: "--", 3: ":"}

# ---------- WFE instance statistics (stable runs only) ----------
w = wfe[(wfe.System_Stable == True) & (wfe["WIRE_lenght (m)"] == 0.5) & (wfe.WIRE_section.isin(SEC))]
P, Q = "Pressure_Stable_Average (mbar)", "Flow_Stable_Average (Nm^3/h)"
inst = (w.groupby(["TUBE_diameter (mm)", "WIRE_section", "WIRE_instance"])
          .agg(n=(P, "size"), p_mean=(P, "mean"), p_sd=(P, "std"), q_mean=(Q, "mean"), q_sd=(Q, "std")).reset_index())
inst = inst[inst.WIRE_instance.isin([1, 2, 3])]
for c in ["p", "q"]:
    secm = inst.groupby(["TUBE_diameter (mm)", "WIRE_section"])[f"{c}_mean"].transform("mean")
    inst[f"{c}_rel_pct"] = 100 * (inst[f"{c}_mean"] - secm) / secm

# Kruskal-Wallis: do the three instances differ beyond run-to-run scatter?
kw = []
for (d, s), g in w[w.WIRE_instance.isin([1, 2, 3])].groupby(["TUBE_diameter (mm)", "WIRE_section"]):
    groups = [gg[P].values for _, gg in g.groupby("WIRE_instance")]
    if len(groups) == 3 and all(len(x) >= 2 for x in groups):
        H, pv = stats.kruskal(*groups)
        kw.append(dict(tube_mm=d, section_mm2=s, H=H, p_value=pv, n_runs=sum(map(len, groups))))
kw = pd.DataFrame(kw)

met["dmax_mm"] = met.dmax_mm; met["A1_abs_mm"] = met.A1_mm.abs()
M = met.merge(inst, left_on=["section_mm2", "instance"], right_on=["WIRE_section", "WIRE_instance"])
M10 = M[M["TUBE_diameter (mm)"] == 10].copy(); M08 = M[M["TUBE_diameter (mm)"] == 8].copy()

corr = []
for tube, D in [(10, M10), (8, M08)]:
    for x in ["dmax_mm", "dpp_mm", "erms_mm", "A1_abs_mm", "wavy_rms_mm"]:
        for y in ["p_rel_pct", "q_rel_pct"]:
            r, pv = stats.spearmanr(D[x], D[y]); rp, pp = stats.pearsonr(D[x], D[y])
            corr.append(dict(tube_mm=tube, x=x, y=y, n=len(D), spearman_rho=r, spearman_p=pv, pearson_r=rp, pearson_p=pp))
corr = pd.DataFrame(corr)

M.to_csv(os.path.join(OUT, "straightness_wfe_merged.csv"), index=False)
kw.to_csv(os.path.join(OUT, "kruskal_instances.csv"), index=False)
corr.to_csv(os.path.join(OUT, "straightness_correlations.csv"), index=False)
pd.set_option("display.width", 220)
print(M10[["label", "dmax_mm", "dpp_mm", "erms_mm", "A1_abs_mm", "n", "p_mean", "p_sd", "p_rel_pct", "q_rel_pct"]].round(2).to_string(index=False))
print(kw.round(4).to_string(index=False))
print(corr[corr.y == "p_rel_pct"].round(3).to_string(index=False))

# ---------- Fig 1: centreline deviation profiles ----------
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.3), sharey=True)
for ax, s in zip(axs, SEC):
    for _, r in met[met.section_mm2 == s].sort_values("instance").iterrows():
        c = cl[cl.label == r.label].sort_values("s_mm")
        sg = np.arange(0, c.s_mm.max(), 1.0); eg = np.interp(sg, c.s_mm, c.e_mm)
        eg = savgol_filter(eg, 21, 2)          # 21 mm window, display smoothing only (metrics use unsmoothed data)
        ax.plot(sg, eg, color=SEC[s][0], ls=INST_LS[r.instance], lw=1.2, label=f"Instance {r.instance}")
    ax.axhline(0, color="#888888", lw=0.5)
    ax.set_title(SLAB[s]); ax.set_xlim(0, 390); ax.set_xticks([0, 100, 200, 300]); ax.set_xlabel(r"$s$ (mm)")
axs[0].set_ylabel(r"Lateral deviation $e$ (mm)")
h = [mpl.lines.Line2D([], [], color="#444444", ls=INST_LS[i], lw=1.2) for i in (1, 2, 3)]
fig.legend(h, ["Instance 1", "Instance 2", "Instance 3"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig_straightness_profiles.pdf")); fig.savefig(os.path.join(OUT, "fig_straightness_profiles.png"), dpi=200)

# ---------- Fig 2: instance pressures (mean +/- SD) annotated with dmax, WC10 ----------
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.3))
for ax, s in zip(axs, SEC):
    g = M10[M10.section_mm2 == s].sort_values("instance")
    raw = w[(w["TUBE_diameter (mm)"] == 10) & (w.WIRE_section == s) & (w.WIRE_instance.isin([1, 2, 3]))]
    ax.scatter(raw.WIRE_instance + np.random.default_rng(0).uniform(-0.08, 0.08, len(raw)), raw[P], s=8, color="#bbbbbb", zorder=1, label="Single run")
    ax.errorbar(g.instance, g.p_mean, yerr=g.p_sd, fmt=SEC[s][1], color=SEC[s][0], ms=6, capsize=3, lw=1, zorder=3, label="Mean $\\pm$ SD")
    for _, r in g.iterrows():
        ax.annotate(f"{r.dmax_mm:.1f}", (r.instance, r.p_mean + (r.p_sd if np.isfinite(r.p_sd) else 0)),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5, color="#333333")
    sub = kw[(kw.tube_mm == 10) & (kw.section_mm2 == s)]
    pv = sub.p_value.iloc[0] if len(sub) else np.nan
    t = SLAB[s] + ("\n$p_{\\mathrm{KW}}$ < 0.001" if pv < 0.001 else f"\n$p_{{\\mathrm{{KW}}}}$ = {pv:.3f}")
    ax.set_title(t); ax.set_xticks([1, 2, 3]); ax.set_xlim(0.5, 3.5); ax.set_xlabel("Wire instance")
    ax.margins(y=0.25)
axs[0].set_ylabel(r"Levitation pressure $\bar{p}$ (mbar)")
axs[0].legend(loc="lower left", frameon=False, fontsize=7)
fig.text(0.99, 0.01, r"Labels: $\delta_{\max}$ in mm", ha="right", fontsize=7, color="#333333")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(OUT, "fig_instance_pressure_WC10.pdf")); fig.savefig(os.path.join(OUT, "fig_instance_pressure_WC10.png"), dpi=200)

# ---------- Fig 3: pooled relative pressure deviation vs straightness ----------
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
for ax, (x, xl) in zip(axs, [("dmax_mm", r"Max. lateral deviation $\delta_{\max}$ (mm)"),
                              ("A1_abs_mm", r"Bow amplitude $|A_1|$ (mm)")]):
    for s in SEC:
        g = M10[M10.section_mm2 == s]
        ax.scatter(g[x], g.p_rel_pct, marker=SEC[s][1], color=SEC[s][0], s=36, edgecolor="white", linewidth=0.6, label=SLAB[s], zorder=3)
        for _, r in g.iterrows():
            ax.annotate(str(r.instance), (r[x], r.p_rel_pct), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="#555555")
    c = corr[(corr.tube_mm == 10) & (corr.x == x) & (corr.y == "p_rel_pct")].iloc[0]
    ax.text(0.98, 0.04, rf"Spearman $\rho$ = {c.spearman_rho:.2f}, $p$ = {c.spearman_p:.2f}, $n$ = {c.n}",
            transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.axhline(0, color="#888888", lw=0.5); ax.set_xlabel(xl)
axs[0].set_ylabel(r"$(\bar{p}_i-\bar{p}_A)/\bar{p}_A$ (%)")
axs[1].legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pressure_dev_vs_straightness_WC10.pdf")); fig.savefig(os.path.join(OUT, "fig_pressure_dev_vs_straightness_WC10.png"), dpi=200)
