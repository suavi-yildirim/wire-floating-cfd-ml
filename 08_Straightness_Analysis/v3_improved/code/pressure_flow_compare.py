"""Compare marked-wire shape metrics with the stable pressure and flow of their matched WFE session.
Partial wires (curve reaches the photo edge) are included and flagged. Output: more/analysis/shape_pressure_flow.csv
and the Spearman tests printed to the screen."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
A = "more/analysis"
W = pd.read_csv(f"{A}/wire_metrics.csv")
M = pd.read_csv("more/out/photo_metrics.csv")
rl = pd.read_csv("results_runlog.csv", parse_dates=["start_local", "end_local"]); S = wp.sessions(rl)
sp = pd.read_csv("../../Thesis_Data/Stable_Phase_Summary.csv"); sp = sp[sp.System_Stable == True]
P, Q = "Pressure_Stable_Average (mbar)", "Flow_Stable_Average (Nm^3/h)"
# partial flag from the curves
part = {}
for sh in W.sheet:
    c = pd.read_csv(f"{A}/{sh}_curve.csv")
    part[sh] = bool((c.x.min() < 15) | (c.x.max() > 1649 - 15) | (c.y.min() < 15) | (c.y.max() > 1237 - 15))
W["partial"] = W.sheet.map(part)
# session per (uid, instance): nearest run-log session to the photo time, within 60 min (same rule as process_more)
photo_time = pd.read_csv("more/out/photo_metrics.csv", parse_dates=["time"]).dropna(subset=["time"])
tmap = photo_time.groupby(["uid", "instance"]).time.min()
rows = []
for sh_name in W.sheet:
    p_ = sh_name.split("_"); uid, inst = int(p_[2]), int(p_[3])
    if (uid, inst) in [(r["uid"], r["instance"]) for r in rows]: continue
    tm = tmap.get((uid, inst))
    if tm is None or pd.isna(tm): continue
    s_ = S[(S.uid == uid) & (S.instance == inst)].copy()
    lead = (s_.start - tm).dt.total_seconds() / 60
    c = s_[lead.abs() <= 60]
    if not len(c): continue
    best = c.loc[lead[c.index].abs().idxmin()]
    files = best.files.split(";"); x = sp[sp.File.isin(files)]
    rows.append(dict(uid=uid, instance=inst, session=str(best.start)[:16], n_stable=len(x), p_mbar=x[P].mean(), q_nm3h=x[Q].mean()))
PF = pd.DataFrame(rows)
print('instances with a matched session:', len(PF))
# shape metrics per wire: chord deviation and bend, both planes pooled as the worst plane
sh = W.groupby(["uid", "instance"]).agg(dev_max_mm=("dev_max_mm", "max"), dev_rms_mm=("dev_rms_mm", "mean"),
                                       bend_deg=("max_bend_20mm_deg", "max"), partial=("partial", "any"),
                                       planes=("plane", lambda v: "".join(sorted(set(v))))).reset_index()
J = sh.merge(PF, on=["uid", "instance"], how="inner")
J["length_m"] = J.uid.map(lambda u: {1: 0.5, 2: 1.0, 3: 2.0, 0: 3.5}[u % 4])
J = J[J.groupby("uid").uid.transform("size") >= 2].copy()          # relative values need >= 2 matched instances per UID
J["p_rel_pct"] = 100 * (J.p_mbar - J.groupby("uid").p_mbar.transform("mean")) / J.groupby("uid").p_mbar.transform("mean")
J["q_rel_pct"] = 100 * (J.q_nm3h - J.groupby("uid").q_nm3h.transform("mean")) / J.groupby("uid").q_nm3h.transform("mean")
J.round(2).to_csv(f"{A}/shape_pressure_flow.csv", index=False)
pd.set_option("display.width", 240)
print(J[["uid", "instance", "length_m", "planes", "partial", "dev_max_mm", "dev_rms_mm", "bend_deg", "p_mbar", "q_nm3h", "p_rel_pct", "q_rel_pct"]].round(2).to_string(index=False))
def test(df, x, y, lab):
    if len(df) < 4: print(f"{lab}: n={len(df)} too few"); return
    rho, p = stats.spearmanr(df[x], df[y]); print(f"{lab:48s} n={len(df):2d}  rho={rho:+.2f}  p={p:.2f}")
print()
for name, df in [("all wires", J), ("full-frame wires only", J[~J.partial])]:
    for x in ["dev_max_mm", "dev_rms_mm", "bend_deg"]:
        test(df, x, "p_rel_pct", f"{name}: {x} vs relative pressure")
        test(df, x, "q_rel_pct", f"{name}: {x} vs relative flow")
