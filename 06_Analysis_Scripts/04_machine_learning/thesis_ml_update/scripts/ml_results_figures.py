"""Figures, tables and key numbers for the ML results section (WFE v8 dataset).

Inputs
  data_csv      MAT_Dataset_v8/ML_Data_All_v8.csv
  oof_matlab    MAT_Dataset_v8/CV_OutOfFold_v8.csv   (MATLAB pipeline, GPR+physics, LOSO/LOUO/LOLO)
  abl_metrics   ablation_metrics.csv                 (piml_ablation.py, same folds, Python replication)
  abl_oof       ablation_oof.csv
Outputs (out_dir): fig_ml_parity.pdf, fig_ml_loso_sections.pdf, fig_ml_ablation.pdf,
                   tab_ml_main.tex, tab_ml_ablation.tex, tab_ml_loso.tex, key_numbers.json
Usage: python3 -I ml_results_figures.py data_csv oof_matlab abl_metrics abl_oof out_dir
"""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

data_csv, oof_csv, abl_csv, ablo_csv, out = sys.argv[1:6]
D = pd.read_csv(data_csv)
O = pd.read_csv(oof_csv)
A = pd.read_csv(abl_csv)
AO = pd.read_csv(ablo_csv)
assert (O.Source_File == D.Source_File).all() and (AO.Source_File == D.Source_File).all()
D["Tube"] = D.Tube_D_mm
K = {}

# ---------------------------------------------------------------- style
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
                     "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.7, "lines.linewidth": 1.0})
SEC = {0.5: "#F5821F", 1.0: "#1F77B4", 1.5: "#E6AF00", 4.0: "#8232A0", 6.0: "#D72323"}  # as in the MATLAB figures
GREY = "#5F5F5F"; GREEN = "#6A8A22"; BLUE = "#002F6C"


def metr(y, p):
    e = y - p
    return dict(R2=1 - np.sum(e**2) / np.sum((y - y.mean())**2), RMSE=np.sqrt(np.mean(e**2)),
                MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e / y)), bias=np.mean(p - y))


yQ, yP = D.Flow_Nm3h.values, D.Pressure_Pa.values

# ---------------------------------------------------------------- noise floor
for tgt, y in (("Q", yQ), ("P", yP)):
    m = D.groupby(["WIRE_uid", "Tube"])[tgt == "Q" and "Flow_Nm3h" or "Pressure_Pa"].transform("mean").values
    K[f"floor_MAPE_{tgt}"] = 100 * np.mean(np.abs((y - m) / y))
    K[f"floor_R2_{tgt}"] = 1 - np.sum((y - m)**2) / np.sum((y - y.mean())**2)
cv = D.groupby(["WIRE_uid", "Tube"]).agg(Pm=("Pressure_Pa", "mean"), Ps=("Pressure_Pa", "std"),
                                         Qm=("Flow_Nm3h", "mean"), Qs=("Flow_Nm3h", "std"), n=("Pressure_Pa", "size"))
K["cv_P_mean"] = float((100 * cv.Ps / cv.Pm).mean()); K["cv_Q_mean"] = float((100 * cv.Qs / cv.Qm).mean())
K["n_runs"] = int(len(D)); K["n_types"] = int(len(cv)); K["n_runs_per_type_min"] = int(cv.n.min()); K["n_runs_per_type_max"] = int(cv.n.max())

# ---------------------------------------------------------------- main table (MATLAB pipeline)
schemes = ["LOUO", "LOLO", "LOSO"]
lab = {"LOUO": "LOUO (unseen wire type)", "LOLO": "LOLO (unseen length)", "LOSO": "LOSO (unseen cross-section)"}
rows = []
for s in schemes:
    for tgt, y, gp, ph in (("Q", yQ, f"{s}_Pred_Flow_GPR", f"{s}_Phys_Flow_cal"),
                           ("P", yP, f"{s}_Pred_Press_GPR", f"{s}_Phys_Press_cal")):
        r = metr(y, O[gp].values); rows.append(dict(scheme=s, target=tgt, model="GPR+phys", **r))
        if ph in O:
            # Pressure_Pa is a gauge pressure (Build_MAT_Dataset.m, l.184); the MATLAB baseline adds P_amb, so remove it
            ph_val = O[ph].values - (101325.0 if tgt == "P" else 0.0)
            r = metr(y, ph_val); rows.append(dict(scheme=s, target=tgt, model="Physics", **r))
M = pd.DataFrame(rows)
K["main"] = M.round(4).to_dict(orient="records")

# cluster bootstrap over the 20 wire types (LOUO only; LOLO and LOSO have 4 and 5 clusters)
rng = np.random.default_rng(1)
uids = np.sort(D.WIRE_uid.unique()); idx = {u: np.where(D.WIRE_uid.values == u)[0] for u in uids}
ci = {}
for tgt, y, c in (("Q", yQ, "LOUO_Pred_Flow_GPR"), ("P", yP, "LOUO_Pred_Press_GPR")):
    p = O[c].values; R2, MAPE = [], []
    for _ in range(4000):
        sel = np.concatenate([idx[u] for u in rng.choice(uids, len(uids))])
        m_ = metr(y[sel], p[sel]); R2.append(m_["R2"]); MAPE.append(m_["MAPE"])
    ci[tgt] = dict(R2=np.percentile(R2, [2.5, 97.5]).tolist(), MAPE=np.percentile(MAPE, [2.5, 97.5]).tolist())
K["ci_LOUO"] = ci

# ---------------------------------------------------------------- hysteresis residuals
res = yP - O["LOUO_Pred_Press_GPR"].values; resQ = yQ - O["LOUO_Pred_Flow_GPR"].values
h = D.Trajectory_History.values
for nm, r_ in (("P", res), ("Q", resQ)):
    a, b = r_[h == "Rises"], r_[h == "Drops"]
    K[f"hyst_{nm}"] = dict(n_rises=int(len(a)), n_drops=int(len(b)), med_rises=float(np.median(a)), med_drops=float(np.median(b)),
                           p=float(mannwhitneyu(a, b).pvalue))

# ---------------------------------------------------------------- LOSO per section x tube
L = D.copy(); L["Qp"] = O.LOSO_Pred_Flow_GPR.values; L["Pp"] = O.LOSO_Pred_Press_GPR.values
g = L.groupby(["Section_mm2", "Tube"]).apply(lambda x: pd.Series(dict(
    n=len(x), beta=x.Blockage_Ratio.iloc[0], Qm=x.Flow_Nm3h.mean(), Qp=x.Qp.mean(),
    Qerr=100 * (x.Qp.mean() / x.Flow_Nm3h.mean() - 1), Pm=x.Pressure_Pa.mean() / 1e3, Pp=x.Pp.mean() / 1e3,
    Perr=100 * (x.Pp.mean() / x.Pressure_Pa.mean() - 1),
    MAPEq=100 * np.mean(np.abs((x.Qp - x.Flow_Nm3h) / x.Flow_Nm3h)), MAPEp=100 * np.mean(np.abs((x.Pp - x.Pressure_Pa) / x.Pressure_Pa))))).reset_index()
K["loso_cells"] = g.round(3).to_dict(orient="records")
sec = L.groupby("Section_mm2").apply(lambda x: pd.Series(dict(
    n=len(x), R2q=metr(x.Flow_Nm3h.values, x.Qp.values)["R2"], MAPEq=metr(x.Flow_Nm3h.values, x.Qp.values)["MAPE"],
    MAPEp=metr(x.Pressure_Pa.values, x.Pp.values)["MAPE"]))).reset_index()
K["loso_sections"] = sec.round(3).to_dict(orient="records")

# ---------------------------------------------------------------- fitted coefficients (python replication)
try:
    fc = pd.read_csv(sys.argv[6]) if len(sys.argv) > 6 else None
except Exception:
    fc = None
if fc is not None:
    K["coef"] = {f"{a}_{b}": v for (a, b), v in fc.groupby("scheme")[["cd", "cf"]].agg(["min", "median", "max"]).round(3).T.to_dict().items()} if False else {sc: {c: [float(fc[fc.scheme==sc][c].min()), float(fc[fc.scheme==sc][c].median()), float(fc[fc.scheme==sc][c].max())] for c in ("cd","cf")} for sc in fc.scheme.unique()}
    K["cf_at_bound_share"] = float((fc.cf <= 0.8001).mean())

# ================================================================ FIGURE 1: parity
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.4))
mk = {8: "o", 10: "s"}
for j, s in enumerate(["LOUO", "LOLO", "LOSO"]):
    for i, (tgt, y, c, unit, f) in enumerate((("Flow", yQ, f"{s}_Pred_Flow_GPR", r"$Q_\mathrm{N}$ (m$^3$/h)", 1.0),
                                              ("Pressure", yP, f"{s}_Pred_Press_GPR", r"$p_\mathrm{in}$ (kPa)", 1e-3))):
        ax = axs[i, j]; p = O[c].values
        lim = [0, max((y * f).max(), (p * f).max()) * 1.05]
        ax.plot(lim, lim, color=GREY, lw=0.8)
        ax.fill_between(lim, [l * 0.9 for l in lim], [l * 1.1 for l in lim], color=GREY, alpha=0.12, lw=0)
        for sct, col in SEC.items():
            for t in (8, 10):
                m_ = ((D.Section_mm2 == sct) & (D.Tube == t)).values
                ax.scatter(y[m_] * f, p[m_] * f, s=7, marker=mk[t], color=col, alpha=0.75, lw=0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        mm = metr(y, p)
        ax.text(0.04, 0.96, f"$R^2$ = {mm['R2']:.2f}\nMAPE = {mm['MAPE']:.0f} %", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel("measured " + unit if i == 1 else "measured"); ax.set_ylabel("predicted " + unit if j == 0 else "")
        if i == 0: ax.set_title(s)
        ax.set_aspect("equal")
h1 = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{s} mm$^2$") for s, c in SEC.items()]
h2 = [plt.Line2D([], [], marker=mk[t], ls="", color=GREY, label=f"WC{t:02d}") for t in (8, 10)]
h3 = [plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.2, label=r"$\pm$10 %")]
fig.legend(handles=h1 + h2 + h3, loc="lower center", ncol=8, frameon=False, handletextpad=0.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{out}/fig_ml_parity.pdf"); plt.close(fig)

# ================================================================ FIGURE 2: LOSO cells
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=False)
secs = [0.5, 1.0, 1.5, 4.0, 6.0]; w = 0.38
for ax, col, ttl in ((axs[0], "Qerr", "Flow"), (axs[1], "Perr", "Pressure")):
    for k, t in enumerate((8, 10)):
        v = [g[(g.Section_mm2 == s_) & (g.Tube == t)][col].iloc[0] for s_ in secs]
        ax.bar(np.arange(5) + (k - 0.5) * w, v, w, color=[SEC[s_] for s_ in secs], alpha=1.0 if t == 8 else 0.55,
               edgecolor="white", lw=0.4, hatch="" if t == 8 else "///")
    ax.axhline(0, color="k", lw=0.6); ax.axhspan(-10, 10, color=GREY, alpha=0.12, lw=0)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{s_:g}" for s_ in secs]); ax.set_xlabel(r"held-out cross-section (mm$^2$)")
    ax.set_ylabel(f"mean signed {ttl.lower()} error (%)"); ax.set_title(ttl)
axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=GREY, label="WC08"), plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.5, hatch="///", ec="white", label="WC10")],
              frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{out}/fig_ml_loso_sections.pdf"); plt.close(fig)

# ================================================================ FIGURE 3: ablation
order_Q = ["physics (calibrated)", "GPR geom", "GPR geom+W", "GPR geom+phys (script)", "lin geom+phys"]
order_P = ["physics (Darcy)", "GPR geom", "GPR geom+W", "GPR geom+phys (script)", "GPR geom+Qphys only", "lin geom+phys"]
nice = {"physics (calibrated)": "physics only", "physics (Darcy)": "Darcy only", "GPR geom": "GPR, geometry",
        "GPR geom+W": "GPR, geometry + $W$", "GPR geom+phys (script)": "GPR, geometry + physics",
        "GPR geom+Qphys only": r"GPR, geometry + $Q_\mathrm{N,phys}$", "lin geom+phys": "linear, geometry + physics"}
cols = {"physics (calibrated)": "#B8B8B8", "physics (Darcy)": "#B8B8B8", "GPR geom": "#98B7D8", "GPR geom+W": "#5F8FC0",
        "GPR geom+phys (script)": BLUE, "GPR geom+Qphys only": "#3C5E8F", "lin geom+phys": GREEN}
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.5), sharey=False)
for i, (tgt, order) in enumerate((("Q", order_Q), ("P", order_P))):
    for j, s in enumerate(["LOUO", "LOLO", "LOSO"]):
        ax = axs[i, j]; sub = A[(A.target == tgt) & (A.scheme == s)].set_index("model")
        v = [sub.loc[m_, "MAPE"] for m_ in order]
        ax.barh(range(len(order)), v, color=[cols[m_] for m_ in order], height=0.7)
        for k, x in enumerate(v): ax.text(x + 1, k, f"{x:.0f}", va="center", fontsize=6.5)
        ax.axvline(K[f"floor_MAPE_{tgt}"], color="k", ls="--", lw=0.8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([nice[m_] for m_ in order] if j == 0 else [], fontsize=6.5)
        if j > 0: ax.tick_params(axis="y", length=0)
        ax.invert_yaxis(); ax.set_xlim(0, max(110, 1.2 * max(v)) if s != "LOSO" else None)
        ax.set_xlim(0, 135 if s == "LOSO" and tgt == "Q" else (140 if s == "LOSO" else 85))
        if i == 0: ax.set_title(s)
        if i == 1: ax.set_xlabel("MAPE (%)")
    axs[i, 0].annotate("flow" if tgt == "Q" else "pressure", xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 18), textcoords="offset points", fontweight="bold", fontsize=8, ha="left")
fig.tight_layout(); fig.savefig(f"{out}/fig_ml_ablation.pdf"); plt.close(fig)

# ================================================================ TABLES
def f3(x): return f"{x:.2f}"
def kpa(x): return f"{x/1e3:.1f}"

lines = [r"\begin{tabular}{@{}l l l S[table-format=-1.2] r r S[table-format=2.0]@{}}", r"\toprule",
         r"Scheme & Target & Model & {$R^2$} & {RMSE} & {MAE} & {MAPE (\si{\percent})} \\", r"\midrule"]
for s in schemes:
    for tgt in ("Q", "P"):
        for mod in ("Physics", "GPR+phys"):
            r = M[(M.scheme == s) & (M.target == tgt) & (M.model == mod)]
            if r.empty: continue
            r = r.iloc[0]
            sch = lab[s] if (tgt == "Q" and mod == ("Physics" if s != "LOSO" else "GPR+phys")) else ""
            tg = (r"$Q_\mathrm{N}$ (\si{\cubic\metre\per\hour})" if tgt == "Q" else r"$p_\mathrm{in}$ (\si{\kilo\pascal})") if mod == ("Physics" if s != "LOSO" else "GPR+phys") else ""
            rm = f"{r.RMSE:.2f}" if tgt == "Q" else f"{r.RMSE/1e3:.1f}"; ma = f"{r.MAE:.2f}" if tgt == "Q" else f"{r.MAE/1e3:.1f}"
            name = ("calibrated physics" if tgt == "Q" else "Darcy drop, annulus only") if mod == "Physics" else "GPR + physics"
            lines.append(f"{sch} & {tg} & {name} & {r.R2:.2f} & {rm} & {ma} & {r.MAPE:.0f} \\\\")
        if tgt == "Q": lines.append(r"\addlinespace[2pt]")
    if s != schemes[-1]: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{out}/tab_ml_main.tex", "w").write("\n".join(lines) + "\n")

# ablation table
nameQ = {"physics (calibrated)": "calibrated physics only", "GPR geom": "GPR, geometry only", "GPR geom+W": r"GPR, geometry $+\,W_\mathrm{wire}$",
         "GPR geom+phys (script)": r"GPR, geometry $+\,Q_\mathrm{N,phys}$", "lin geom": "linear, geometry only", "lin geom+phys": r"linear, geometry $+\,Q_\mathrm{N,phys}$"}
nameP = {"physics (Darcy)": "Darcy baseline only", "GPR geom": "GPR, geometry only", "GPR geom+W": r"GPR, geometry $+\,W_\mathrm{wire}$",
         "GPR geom+phys (script)": r"GPR, geometry $+\,p_\mathrm{Darcy}+p_\mathrm{dyn}$", "GPR geom+Qphys only": r"GPR, geometry $+\,Q_\mathrm{N,phys}$",
         "lin geom": "linear, geometry only", "lin geom+phys": r"linear, geometry $+\,p_\mathrm{Darcy}+p_\mathrm{dyn}$"}
lines = [r"\begin{tabular}{@{}l S[table-format=-1.2] S[table-format=3.0] S[table-format=-1.2] S[table-format=3.0] S[table-format=-1.2] S[table-format=3.0]@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{LOUO} & \multicolumn{2}{c}{LOLO} & \multicolumn{2}{c}{LOSO} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"Predictors & {$R^2$} & {MAPE} & {$R^2$} & {MAPE} & {$R^2$} & {MAPE} \\", r"\midrule",
         r"\multicolumn{7}{@{}l}{\textit{Flow rate} $Q_\mathrm{N}$} \\"]
for k, nm in nameQ.items():
    cells = []
    for s in ["LOUO", "LOLO", "LOSO"]:
        r = A[(A.target == "Q") & (A.scheme == s) & (A.model == k)].iloc[0]; cells += [f"{r.R2:.2f}", f"{r.MAPE:.0f}"]
    lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Inlet pressure} $p_\mathrm{in}$} \\"]
for k, nm in nameP.items():
    cells = []
    for s in ["LOUO", "LOLO", "LOSO"]:
        r = A[(A.target == "P") & (A.scheme == s) & (A.model == k)].iloc[0]; cells += [f"{r.R2:.2f}", f"{r.MAPE:.0f}"]
    lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{out}/tab_ml_ablation.tex", "w").write("\n".join(lines) + "\n")

# loso cell table
lines = [r"\begin{tabular}{@{}S[table-format=1.1] S[table-format=2.0] S[table-format=1.2] S[table-format=2.1] S[table-format=2.1] S[table-format=+3.0] S[table-format=3.0] S[table-format=3.0] S[table-format=+3.0]@{}}", r"\toprule",
         r"{$A_\mathrm{cs}$} & {$D_\mathrm{tube}$} & {$\beta$} & {$Q_\mathrm{N}$ meas.} & {$Q_\mathrm{N}$ pred.} & {err.} & {$p_\mathrm{in}$ meas.} & {$p_\mathrm{in}$ pred.} & {err.} \\",
         r"{(\si{\milli\metre\squared})} & {(\si{\milli\metre})} & {(--)} & {(\si{\cubic\metre\per\hour})} & {(\si{\cubic\metre\per\hour})} & {(\si{\percent})} & {(\si{\kilo\pascal})} & {(\si{\kilo\pascal})} & {(\si{\percent})} \\", r"\midrule"]
prev = None
for _, r in g.iterrows():
    if prev is not None and r.Section_mm2 != prev: lines.append(r"\addlinespace[2pt]")
    prev = r.Section_mm2
    lines.append(f"{r.Section_mm2:.1f} & {int(r.Tube)} & {r.beta:.2f} & {r.Qm:.1f} & {r.Qp:.1f} & {r.Qerr:+.0f} & {r.Pm:.0f} & {r.Pp:.0f} & {r.Perr:+.0f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{out}/tab_ml_loso.tex", "w").write("\n".join(lines) + "\n")

json.dump(K, open(f"{out}/key_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in K.items() if k not in ("main", "loso_cells")}, indent=1, default=float))
print(M.round(3).to_string())
