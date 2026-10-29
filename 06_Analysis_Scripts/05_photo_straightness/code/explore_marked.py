"""Exploratory test of shape features of the marked wires against run responses, within wire type.
Features per wire instance: dev_max, dev_rms (chord), bend (max angle change/20 mm), curvature rms/max/mean (1/m, curve smoothed over 8 mm),
bending energy (integral kappa^2 ds, 1/m), arc excess (length/chord-1, %), inflections per plane (S-shape count), resultant rmax (two-plane only).
Responses (per instance, over the runs of the matched session): p_stable, q_stable, p_lift, t_lift, t_stable, t_lift_to_stable, n_incr,
p_overshoot, p_sd_settled, lift_ratio. All values are made relative to the mean of the same UID (needs >= 2 instances).
Test: Spearman; permutation p by shuffling the response within UID; Benjamini-Hochberg q over all tests."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
A = "more/analysis"
W = pd.read_csv(f"{A}/wire_metrics.csv"); W = W[W.status == "ok"].copy()
M = pd.read_csv("more/out/photo_metrics.csv", parse_dates=["time"]).dropna(subset=["time"])
RF = pd.read_csv("run_features.csv"); RF["lift_ratio"] = RF.p_lift / RF.p_stable
rl = pd.read_csv("results_runlog.csv", parse_dates=["start_local", "end_local"]); S = wp.sessions(rl)
sp = pd.read_csv("../../03_Experimental_Data/Stable_Phase_Summary.csv"); sp = sp[sp.System_Stable == True]
tmap = M.groupby(["uid", "instance"]).time.min()
def curve_mm(sheet, ppm):
    c = pd.read_csv(f"{A}/{sheet}_curve.csv"); x, y = c.x.values, c.y.values
    ux, uy = x[-1] - x[0], y[-1] - y[0]; L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    return ((x - x[0]) * ux + (y - y[0]) * uy) / ppm, (-(x - x[0]) * uy + (y - y[0]) * ux) / ppm
feat = []
for (u, i), g in W.groupby(["uid", "instance"]):
    per = {}
    for r in g.itertuples():
        s, e = curve_mm(r.sheet, r.px_per_mm)
        gr = np.arange(0, s[-1], 1.0); eg = np.interp(gr, s, e)
        sm = gaussian_filter1d(eg, 8 / 2.355); k = np.gradient(np.gradient(sm, 1.0), 1.0) * 1000      # 1/m
        per[r.plane] = dict(s=gr, e=eg, k=k, infl=int(np.sum(np.diff(np.sign(k[np.abs(k) > 0.15 * np.abs(k).max()])) != 0)),
                            length=r.length_mm, chord=r.chord_mm)
    kk = np.max([np.abs(v["k"]) for v in per.values()], axis=0) if len(per) == 1 else None
    if len(per) == 2:
        n_ = min(len(per["X"]["k"]), len(per["Y"]["k"])); kk = np.hypot(per["X"]["k"][:n_], per["Y"]["k"][:n_])
        n2 = min(len(per["X"]["e"]), len(per["Y"]["e"])); rr = np.hypot(per["X"]["e"][:n2], per["Y"]["e"][:n2]); rmax = rr.max()
    else:
        rmax = np.nan
    v0 = list(per.values())[0]
    feat.append(dict(uid=int(u), instance=int(i), planes="".join(sorted(per)), dev_max=g.dev_max_mm.max(), dev_rms=g.dev_rms_mm.mean(),
                     bend=g.max_bend_20mm_deg.max(), kappa_rms=float(np.sqrt(np.mean(kk ** 2))), kappa_max=float(kk.max()), kappa_mean=float(kk.mean()),
                     bend_energy=float(np.sum(kk ** 2) * 1e-3), arc_excess_pct=100 * (v0["length"] / v0["chord"] - 1),
                     inflections=float(np.mean([v["infl"] for v in per.values()])), rmax=rmax,
                     partial=bool(False)))
F = pd.DataFrame(feat)
# responses per instance from the matched session
rows = []
for r in F.itertuples():
    tm = tmap.get((r.uid, r.instance))
    if tm is None or pd.isna(tm): continue
    s_ = S[(S.uid == r.uid) & (S.instance == r.instance)].copy(); lead = (s_.start - tm).dt.total_seconds() / 60
    c = s_[lead.abs() <= 60]
    if not len(c): continue
    best = c.loc[lead[c.index].abs().idxmin()]; files = best.files.split(";")
    x = RF[RF.file.isin(files)]; st = sp[sp.File.isin(files)]
    if not len(x): continue
    rows.append(dict(uid=r.uid, instance=r.instance, p_stable=st["Pressure_Stable_Average (mbar)"].mean(), q_stable=st["Flow_Stable_Average (Nm^3/h)"].mean(),
                     p_lift=x.p_lift.mean(), t_lift=x.t_lift_s.median(), t_stable=x.t_stable_s.median(), t_lift_to_stable=x.t_lift_to_stable_s.median(),
                     n_incr=x.n_incr.median(), p_overshoot=x.p_overshoot.mean(), p_sd_settled=x.p_sd_settled.mean(), lift_ratio=x.lift_ratio.mean(), n_runs=len(x)))
R = pd.DataFrame(rows); D = F.merge(R, on=["uid", "instance"])
D = D[D.groupby("uid").uid.transform("size") >= 2].copy()
shape = ["dev_max", "dev_rms", "bend", "kappa_rms", "kappa_max", "kappa_mean", "bend_energy", "arc_excess_pct", "inflections"]
resp = ["p_stable", "q_stable", "p_lift", "t_lift", "t_stable", "t_lift_to_stable", "n_incr", "p_overshoot", "p_sd_settled", "lift_ratio"]
for c in shape + resp: D[c + "_rel"] = D[c] - D.groupby("uid")[c].transform("mean")
D.to_csv(f"{A}/explore_table.csv", index=False)
print("instances:", len(D), "UIDs:", D.uid.nunique())
rng = np.random.default_rng(0); out = []
g = D.uid.values
for sh in shape:
    for rs in resp:
        a, b = D[sh + "_rel"].values, D[rs + "_rel"].values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 6: continue
        rho = stats.spearmanr(a[ok], b[ok])[0]; cnt = 0; N = 4000; gg = g[ok]; aa, bb = a[ok], b[ok]
        for _ in range(N):
            bp = bb.copy()
            for u in np.unique(gg): idx = np.where(gg == u)[0]; bp[idx] = rng.permutation(bb[idx])
            if abs(stats.spearmanr(aa, bp)[0]) >= abs(rho): cnt += 1
        out.append(dict(shape=sh, response=rs, n=int(ok.sum()), rho=rho, p_perm=(cnt + 1) / (N + 1)))
T = pd.DataFrame(out).sort_values("p_perm"); m = len(T)
T["q_BH"] = np.minimum.accumulate((T.p_perm * m / np.arange(1, m + 1))[::-1])[::-1].clip(upper=1)
T.round(3).to_csv(f"{A}/explore_tests.csv", index=False)
pd.set_option("display.width", 200)
print(f"{m} tests"); print(T.round(3).head(15).to_string(index=False))
print("\\nminimum q:", round(float(T.q_BH.min()), 3))
