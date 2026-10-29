"""
Helical wire-deformation study (WC10 / UID18) - clean re-analysis from raw Fluent .out files.

Fixes vs. Helix_Plots.py:
  * columns are read by header name, not by index (skin-friction headers are shuffled in some runs)
  * corrupted last row of the baseline wire-drag file is dropped (0.2753 -> 0.2600 jump)
  * every metric is a quasi-steady window mean (not the last row) with a std / drift uncertainty
  * baseline is flagged as provisional (mass flow reached target only at the final iteration)
  * correlation with mesh size and analytical arc-length estimate are reported
Outputs: Thesis_Data/Report_Data/Analysis_Output_Helical/*.csv|tex  and  Thesis_Figures/Helical/*.pdf|png
"""
import os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = "/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis"
RD = f"{ROOT}/03_Experimental_Data/Report_Data"
OUT_T = f"{RD}/Analysis_Output_Helical"
OUT_F = f"{ROOT}/09_Figures/Helical"
os.makedirs(OUT_T, exist_ok=True); os.makedirs(OUT_F, exist_ok=True)

MDOT_TARGET = 0.008576243953761728   # kg/s, value all runs end at
WIN = 30                              # quasi-steady averaging window [iterations]
WIRE_L = 0.96                         # m, Fusion 360 coil height (wire body 1.0 m incl. ferrules)
FERRULE_OD = 5.4                      # mm
WIRE_OD, TUBE_ID = 3.95, 10.0         # mm


def rd(path):
    """Fluent report-file -> DataFrame indexed by iteration, columns by header name."""
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    h = next(i for i, l in enumerate(lines) if l.startswith("("))
    cols = re.findall(r'"([^"]+)"', lines[h])
    rows = [[float(x) for x in l.split()] for l in lines[h + 1:] if l[0].isdigit()]
    return pd.DataFrame(rows, columns=cols).set_index("Iteration")


def short(c):
    m = re.search(r"\((.*)\)", c)
    return m.group(1) if m else c


def load_run(name):
    p = f"{RD}/{name}"
    g = lambda f: rd(f"{p}/{f}-rfile.out")
    d = {}
    drag = g("wire-drag-force"); drag.columns = [short(c) for c in drag.columns]
    cd = g("wire-drag-coefficient"); cd.columns = [short(c) for c in cd.columns]
    if name == "WC10_UID18":              # last row is corrupted (wall_wire jumps 0.2753 -> 0.2600)
        drag, cd = drag.iloc[:-1], cd.iloc[:-1]
    d["F_wire"], d["F_fer_b"], d["F_fer_t"] = drag["wall_wire"], drag["wall_ferrule_bottom"], drag["wall_ferrule_top"]
    d["Cd_wire"] = cd["wall_wire"]
    sf_path = f"{p}/wire_skin_friction-rfile.out"
    if os.path.exists(sf_path):
        sf = g("wire_skin_friction"); vals = sf.values
        # column order is not stable inside some files (final row is written in a different order) -> wire = row-wise max
        d["F_visc"] = pd.Series(vals.max(1), index=sf.index)
    d["p_in"] = g("pressure-inlet")["pressure-inlet"]
    d["mdot"] = g("mass-flow-inlet")["mass-flow-inlet"]
    d["mdot_net"] = g("net-mass-flow")["net-mass-flow"]
    d["dp_tube"] = g("tube-pressure-drop").iloc[:, 0]
    d["dp_silencer"] = g("silencer-pressure-drop").iloc[:, 0]
    d["dp_launch"] = (g("launching_station_inlet_pressure").iloc[:, 0]
                      - g("launching-station-outlet-pressure").iloc[:, 0])   # recomputed: monitor sign is inconsistent
    cc = g("cell-count"); cc.columns = [short(c) for c in cc.columns]
    d["cells_tube"] = float(cc["tube_wc10"].iloc[-1])
    d["cells_total"] = float(cc.iloc[-1].sum())
    return d


def window_stats(s, mask):
    """mean / std% / drift% over the last WIN iterations that satisfy mask."""
    if len(s) < 3:                       # baseline section monitors only hold the final row
        return dict(mean=s.iloc[-1], std_pct=np.nan, drift_pct=np.nan, n=len(s))
    s = s[mask.reindex(s.index, fill_value=False)]
    w = s.iloc[-WIN:]
    if len(w) < 3:
        return dict(mean=np.nan, std_pct=np.nan, drift_pct=np.nan, n=len(w))
    slope = np.polyfit(w.index, w.values, 1)[0]
    return dict(mean=w.mean(), std_pct=100 * w.std() / abs(w.mean()),
                drift_pct=100 * slope * len(w) / abs(w.mean()), n=len(w))


runs = ["WC10_UID18"] + sorted(d for d in os.listdir(RD) if d.startswith("Helix_Dia") and os.path.isdir(f"{RD}/{d}"))
rows, series = [], {}
for r in runs:
    d = load_run(r); series[r] = d
    # iterations where inlet mass flow is at target (quasi-steady operating point)
    at_target = (d["mdot"] / MDOT_TARGET - 1).abs() < 1e-3
    n_target = int(at_target.sum())
    mask = at_target
    rec = dict(run=r)
    m = re.match(r"Helix_Dia([\d.]+)_Rev([\d.]+)", r)
    rec["dia_mm"], rec["rev"] = (float(m[1]), float(m[2])) if m else (0.0, 0.0)
    rec["iters"], rec["iters_at_target"] = len(d["p_in"]), n_target
    rec["net_mdot_pct"] = 100 * d["mdot_net"][mask].iloc[-10:].mean() / MDOT_TARGET if n_target else np.nan
    rec["cells_tube"], rec["cells_total"] = d["cells_tube"], d["cells_total"]
    for k in ["F_wire", "F_fer_b", "F_fer_t", "Cd_wire", "p_in", "dp_tube", "dp_silencer", "dp_launch"] + (["F_visc"] if "F_visc" in d else []):
        st = window_stats(d[k], mask)
        rec[k] = st["mean"]
        if k in ("F_wire", "p_in"):
            rec[k + "_std_pct"], rec[k + "_drift_pct"], rec[k + "_nwin"] = st["std_pct"], st["drift_pct"], st["n"]
    rec["F_total"] = rec["F_wire"] + rec["F_fer_b"] + rec["F_fer_t"]
    if "F_visc" in rec:
        rec["F_press"] = rec["F_wire"] - rec["F_visc"]
    rows.append(rec)
df = pd.DataFrame(rows)

# ---- provisional baseline: extrapolate the still-evolving baseline run to its asymptote ----
b = series["WC10_UID18"]["F_wire"]; t = b.index.values; y = b.values
w = t >= t[-1] - 50
try:
    from scipy.optimize import curve_fit
    t0 = t[w][0]
    f = lambda tt, a, A, tau: a - A * np.exp(-(tt - t0) / tau)
    popt, _ = curve_fit(f, t[w], y[w], p0=[y[-1] * 1.02, 0.02, 40],
                        bounds=([y[-1], 0, 5], [y[-1] * 1.2, 0.5, 500]), maxfev=20000)
    F_inf = popt[0]
except Exception as e:
    print("extrapolation fit failed:", e); F_inf = np.nan
# (B) quasi-steady scaling: F ~ mdot^n, exponent from the baseline's own last 40 iterations
mb = series["WC10_UID18"]["mdot"].reindex(b.index)
sel = b.index >= b.index[-1] - 40
n_exp = np.polyfit(np.log(mb[sel]), np.log(b[sel]), 1)[0]
F_scale = y[-1] * (MDOT_TARGET / mb.iloc[-1]) ** n_exp
print(f"extrapolation B (F~mdot^n, n={n_exp:.2f}): {F_scale:.4f} N  (mdot at last valid it = {mb.iloc[-1]:.6f})")
base_idx = df.index[df.run == "WC10_UID18"][0]
df.loc[base_idx, "F_wire_last"] = y[-1]
df.loc[base_idx, "F_wire_extrap"] = F_inf
print(f"baseline wall-wire drag: last valid it = {y[-1]:.4f} N, exponential-asymptote extrapolation = {F_inf:.4f} N")

hel = df[df.run != "WC10_UID18"].copy()
base = df.loc[base_idx]

# ---- statistics on the 9 helix runs ----
def lin(x, y):
    r = stats.linregress(x, y)
    return dict(slope=r.slope, r=r.rvalue, p=r.pvalue)

report = {}
for k in ["F_wire", "p_in", "Cd_wire", "dp_tube"]:
    report[k] = dict(mean=hel[k].mean(), std_pct=100 * hel[k].std() / hel[k].mean(),
                     rng_pct=100 * (hel[k].max() - hel[k].min()) / hel[k].mean(),
                     vs_rev=lin(hel.rev, hel[k]), vs_dia=lin(hel.dia_mm, hel[k]),
                     vs_cells=lin(hel.cells_tube, hel[k]))
# joint OLS: F ~ 1 + rev + dia + cells(scaled)
X = np.column_stack([np.ones(len(hel)), hel.rev, hel.dia_mm, (hel.cells_tube - hel.cells_tube.mean()) / 1e4])
yv = hel.F_wire.values
beta, res, *_ = np.linalg.lstsq(X, yv, rcond=None)
dof = len(yv) - X.shape[1]; s2 = ((yv - X @ beta) ** 2).sum() / dof
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); pv = 2 * stats.t.sf(np.abs(beta / se), dof)
joint = pd.DataFrame(dict(term=["const", "rev", "dia_mm", "cells_tube/1e4"], coef=beta, se=se, p=pv))

# ---- analytical expectation: helix arc-length (wetted-area) increase ----
def arc_ratio(amp_mm, rev, L=WIRE_L):
    return np.sqrt(1 + (2 * np.pi * amp_mm * 1e-3 * rev / L) ** 2) - 1
hel["arc_pct_if_Dia_is_diameter"] = 100 * arc_ratio(hel.dia_mm / 2, hel.rev)
hel["arc_pct_if_Dia_is_radius"] = 100 * arc_ratio(hel.dia_mm, hel.rev)
gap = (TUBE_ID - WIRE_OD) / 2
hel["eccentricity"] = (hel.dia_mm / 2) / gap
hel["pitch_mm"] = WIRE_L * 1e3 / hel.rev
hel["max_slope_deg"] = np.degrees(np.arctan(2 * np.pi * (hel.dia_mm / 2) / hel.pitch_mm))

# ---- annulus Reynolds number (order of magnitude, inlet density) ----
A_ann = np.pi / 4 * (TUBE_ID ** 2 - WIRE_OD ** 2) * 1e-6
G = MDOT_TARGET / A_ann; Dh = (TUBE_ID - WIRE_OD) * 1e-3
Re = G * Dh / 1.85e-5
print(f"annulus mass flux {G:.0f} kg/m2s, Re_Dh ~ {Re:.2e}")

# ---- relative change tables (vs provisional baseline = last valid value) ----
ref = dict(F_wire=y[-1], p_in=base.p_in, Cd_wire=base.Cd_wire, dp_tube=base.dp_tube, dp_silencer=base.dp_silencer)
for k, v in ref.items():
    hel[f"d_{k}_pct"] = 100 * (hel[k] / v - 1)
hel["d_F_wire_vs_extrap_pct"] = 100 * (hel.F_wire / F_inf - 1)

df.to_csv(f"{OUT_T}/helix_all_runs.csv", index=False)
hel.to_csv(f"{OUT_T}/helix_runs_vs_baseline.csv", index=False)
joint.to_csv(f"{OUT_T}/helix_joint_regression.csv", index=False)

print("\n=== run table ===")
cols = ["run", "iters", "iters_at_target", "net_mdot_pct", "F_wire", "F_wire_std_pct", "F_wire_drift_pct", "p_in", "cells_tube"]
print(df[cols].round(4).to_string(index=False))
print("\n=== helix vs provisional baseline [%] ===")
print(hel[["dia_mm", "rev", "d_F_wire_pct", "d_F_wire_vs_extrap_pct", "d_p_in_pct", "d_Cd_wire_pct", "d_dp_tube_pct",
           "arc_pct_if_Dia_is_diameter", "eccentricity", "pitch_mm", "max_slope_deg"]].round(3).to_string(index=False))
print("\n=== statistics (9 helix runs) ===")
for k, v in report.items():
    print(k, {a: (np.round(b, 4) if not isinstance(b, dict) else {c: np.round(e, 4) for c, e in b.items()}) for a, b in v.items()})
print("\njoint OLS F_wire:\n", joint.round(5).to_string(index=False))
print("\nviscous share of wire drag [%]:", (100 * hel.F_visc / hel.F_wire).round(2).tolist())
print("launch-station dP (recomputed) [Pa]:", hel.dp_launch.round(0).tolist(), "baseline", round(base.dp_launch))

# ================= figures =================
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 6.5, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})
col = {1.0: "#E4572E", 1.5: "#A8325E", 2.0: "#4B2A7B"}
mk = {1.0: "o", 1.5: "s", 2.0: "^"}

# Fig 1: convergence / mass-flow ramp
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6))
for r, d in series.items():
    isb = r == "WC10_UID18"
    kw = dict(color="k", lw=1.6, zorder=5) if isb else dict(lw=0.8, alpha=0.8)
    ax[0].plot(d["mdot"].index, d["mdot"] * 1e3, label="Straight (baseline)" if isb else None, **kw)
    ax[1].plot(d["F_wire"].index, d["F_wire"], **kw)
ax[0].axhline(MDOT_TARGET * 1e3, color="gray", ls=":", lw=0.8)
ax[0].set(xlabel="Iteration", ylabel="Inlet mass flow [g/s]", title="(a) Inlet mass flow")
ax[1].set(xlabel="Iteration", ylabel="Wire-wall drag [N]", title="(b) Wire-wall drag", ylim=(0.2, 0.32))
ax[0].legend(frameon=False, loc="lower right")
plt.tight_layout(); fig.savefig(f"{OUT_F}/helix_convergence.pdf"); fig.savefig(f"{OUT_F}/helix_convergence.png", dpi=300); plt.close(fig)

# Fig 2: relative change vs revolutions
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5))
for a, (k, lab) in zip(ax, [("d_F_wire_pct", r"$\Delta F_{wire}$ [%]"), ("d_p_in_pct", r"$\Delta p_{in}$ [%]"), ("d_dp_tube_pct", r"$\Delta (\Delta p_{tube})$ [%]")]):
    for dia, g in hel.groupby("dia_mm"):
        a.plot(g.rev, g[k], marker=mk[dia], color=col[dia], label=f"{dia} mm", ms=4)
    a.axhline(0, color="k", lw=0.8)
    a.set(xlabel="Revolutions", ylabel=lab, xticks=[1, 3, 5])
ax[0].axhspan(-2, 2, color="gray", alpha=0.15, lw=0)
ax[0].text(1.0, 2.2, r"$\pm2\,\%$ run-to-run scatter", fontsize=6.5, color="gray")
ax[0].legend(title="Helix dia.", frameon=False)
plt.tight_layout(); fig.savefig(f"{OUT_F}/helix_relative_change.pdf"); fig.savefig(f"{OUT_F}/helix_relative_change.png", dpi=300); plt.close(fig)

# Fig 3: mesh-size dependence
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for dia, g in hel.groupby("dia_mm"):
    ax.scatter(g.cells_tube / 1e6, g.F_wire, c=col[dia], marker=mk[dia], s=28, label=f"{dia} mm")
for _, rr in hel.iterrows():
    ax.annotate(f"{int(rr.rev)}", (rr.cells_tube / 1e6, rr.F_wire), textcoords="offset points", xytext=(4, 3), fontsize=6)
lr = stats.linregress(hel.cells_tube / 1e6, hel.F_wire); xx = np.linspace(hel.cells_tube.min(), hel.cells_tube.max(), 10) / 1e6
ax.plot(xx, lr.intercept + lr.slope * xx, "k--", lw=0.8)
ax.set(xlabel=r"Tube cells $N_{tube}$ [$10^6$]", ylabel="Wire-wall drag [N]")
ax.text(0.04, 0.93, f"$r$ = {lr.rvalue:.2f}, $p$ = {lr.pvalue:.3f}", transform=ax.transAxes, fontsize=7)
ax.legend(title="Helix dia. (label: rev.)", frameon=False, loc="lower right")
plt.tight_layout(); fig.savefig(f"{OUT_F}/helix_mesh_dependence.pdf"); fig.savefig(f"{OUT_F}/helix_mesh_dependence.png", dpi=300); plt.close(fig)

# Fig 4: heatmaps dia x rev
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.1))
for a, (k, lab) in zip(ax, [("d_F_wire_pct", r"$\Delta F_{wire}$ [%]"), ("d_p_in_pct", r"$\Delta p_{in}$ [%]"), ("d_Cd_wire_pct", r"$\Delta C_{D,wire}$ [%]")]):
    pv_ = hel.pivot(index="dia_mm", columns="rev", values=k)
    im = a.imshow(pv_.values, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto", origin="lower")
    a.set_xticks(range(3), [int(c) for c in pv_.columns]); a.set_yticks(range(3), pv_.index)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{pv_.values[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    a.set(xlabel="Revolutions", title=lab)
ax[0].set_ylabel("Helix dia. [mm]")
plt.tight_layout(); fig.savefig(f"{OUT_F}/helix_heatmaps.pdf"); fig.savefig(f"{OUT_F}/helix_heatmaps.png", dpi=300); plt.close(fig)

# LaTeX table
with open(f"{OUT_T}/helix_table.tex", "w") as fh:
    fh.write("\\begin{tabular}{rrrrrrr}\n\\toprule\nDia. [mm] & Rev. & $p_{in}$ [kPa] & $F_{wire}$ [N] & $C_{D,wire}$ & $\\Delta p_{tube}$ [kPa] & $N_{tube}$ [$10^6$]\\\\\n\\midrule\n")
    fh.write(f"0 (straight)$^{{*}}$ & 0 & {base.p_in/1e3:.1f} & {y[-1]:.4f} & {base.Cd_wire:.3f} & {base.dp_tube/1e3:.1f} & {base.cells_tube/1e6:.3f}\\\\\n")
    for _, r_ in hel.iterrows():
        fh.write(f"{r_.dia_mm:.1f} & {int(r_.rev)} & {r_.p_in/1e3:.1f} & {r_.F_wire:.4f} & {r_.Cd_wire:.3f} & {r_.dp_tube/1e3:.1f} & {r_.cells_tube/1e6:.3f}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print("\nfigures ->", OUT_F, "\ntables  ->", OUT_T)
