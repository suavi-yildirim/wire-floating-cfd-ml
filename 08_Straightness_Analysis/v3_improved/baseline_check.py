import sys, numpy as np, pandas as pd
sys.path.insert(0, "../WFE_Photo_Pipeline_v2 2/code")
import wire_pipeline as wp
from scipy import stats
R = "../WFE_Photo_Pipeline_v2 2/results/"
rl = pd.read_csv(R + "wfe_runlog_local_time.csv", parse_dates=["start_local", "end_local"]); S = wp.sessions(rl)
F = pd.read_csv("run_features.csv"); T = pd.read_csv(R + "straightness_pressure_table.csv")
T = T[T.status_final == "ok"]
resp = ["p_stable", "p_lift", "t_stable_s", "t_lift_s", "t_lift_to_stable_s", "n_incr", "p_overshoot", "p_sd_settled"]
rows = []
for t in T.itertuples():
    s = S[(S.uid == t.uid) & (S.instance == t.instance) & (S.start == pd.Timestamp(t.session_start))].iloc[0]
    f = F[F.file.isin(s.files.split(";"))]
    d = dict(group=t.group, uid=t.uid, length=t.length, rmax=t.rmax_mm, rrms=t.rrms_mm, n=len(f))
    for r in resp: d[r] = f[r].median() if r.startswith(("t_", "n_")) else f[r].mean()
    rows.append(d)
D = pd.DataFrame(rows)
for r in resp:
    D[r + "_rel"] = D[r] / D.groupby(["uid", "length"])[r].transform("mean") - 1
D = D[D.groupby(["uid", "length"]).uid.transform("size") >= 3]
print(len(D))
for r in resp:
    x = D.dropna(subset=[r + "_rel"]); rs, p = stats.spearmanr(x.rmax, x[r + "_rel"]); print(f"{r:22s} n={len(x)} rho={rs:+.2f} p={p:.2f}")
