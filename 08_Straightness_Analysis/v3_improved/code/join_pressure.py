"""Join two-plane metrics to WFE stable pressure of the session that followed each photo group.
argv: batch_dir  runlog.csv  Stable_Phase_Summary.csv  [matched file list is taken from the pipeline's matching]"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
B, RL, SP = sys.argv[1:4]
T = pd.read_csv(os.path.join(B, "twoplane_metrics.csv")); M = pd.read_csv(os.path.join(B, "photo_metrics.csv"), parse_dates=["time"])
rl = pd.read_csv(RL, parse_dates=["start_local", "end_local"]); sess = wp.sessions(rl)
sp = pd.read_csv(SP); sp = sp[sp.System_Stable == True]
P, Q = "Pressure_Stable_Average (mbar)", "Flow_Stable_Average (Nm^3/h)"
rows = []
T = T.dropna(subset=['uid']).copy()
for t in T.itertuples():
    s = sess[(sess.uid == t.uid) & (sess.instance == t.instance) & (sess.start == pd.Timestamp(t.run_start))].iloc[0]
    files = s.files.split(";"); x = sp[sp.File.isin(files)]
    rows.append(dict(group=t.group, uid=t.uid, instance=t.instance, section=t.section, length=t.length, tube=t.tube, session_start=s.start,
                     n_runs_session=s.n_runs, n_stable=len(x), p_mean=x[P].mean(), p_sd=x[P].std(), q_mean=x[Q].mean(),
                     p_all_instance=sp[(sp.WIRE_uid == t.uid) & (sp.WIRE_instance == t.instance) & (sp["WIRE_lenght (m)"] == t.length)
                                       & (sp[[c for c in sp.columns if c.lower().startswith("tube_diam")][0]] == t.tube)][P].mean()))
J = T.merge(pd.DataFrame(rows), on=["group", "uid", "instance", "section", "length", "tube"])
J.to_csv(os.path.join(B, "twoplane_with_pressure.csv"), index=False)
pd.set_option("display.width", 250)
print(J[["photos", "uid", "instance", "length", "tube", "rmax_mm", "rrms_mm", "status", "n_stable", "p_mean", "p_sd", "p_all_instance"]].round(1).to_string(index=False))
