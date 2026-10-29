"""Automatic plausibility check of tracked centrelines: a physical wire on the board cannot move laterally by more
than JUMP_MM within DS_MM along the rule. Such jumps mean the tracker left the wire (e.g. wire crossing the steel
rule, track snapping to the board edge). Writes <B>/photo_track_qc.csv and updates status in photo_metrics.csv
(column status_auto) and the two-plane tables (column qc_track).
Usage: python3 qc_track_jumps.py <pipeline_out_folder>"""
import os, sys
import numpy as np, pandas as pd

B = sys.argv[1]; DS_MM, JUMP_MM = 3.0, 2.5

def max_jump(d):
    d = d.sort_values("s_mm"); s, e = d.s_mm.values, d.e_mm.values
    if len(s) < 20: return np.nan
    g = np.arange(s.min(), s.max(), 0.5); eg = np.interp(g, s, e)
    k = int(DS_MM / 0.5)
    return float(np.abs(eg[k:] - eg[:-k]).max())

C = pd.read_csv(os.path.join(B, "photo_centrelines.csv"))
CF = pd.read_csv(os.path.join(B, "photo_centrelines_fullspan.csv"))
q = pd.DataFrame(dict(photo=C.photo.unique()))
q["jump_win_mm"] = [max_jump(C[C.photo == p]) for p in q.photo]
q["jump_full_mm"] = [max_jump(CF[CF.photo == p]) if (CF.photo == p).any() else np.nan for p in q.photo]
q["track_ok_win"] = q.jump_win_mm <= JUMP_MM
q["track_ok_full"] = q.jump_full_mm <= JUMP_MM
q.to_csv(os.path.join(B, "photo_track_qc.csv"), index=False)

M = pd.read_csv(os.path.join(B, "photo_metrics.csv"))
M = M.drop(columns=[c for c in M.columns if c in ("jump_win_mm", "jump_full_mm", "track_ok_win", "track_ok_full", "status_auto")])
M = M.merge(q, on="photo", how="left")
M["status_auto"] = np.where(M.status == "ok", np.where(M.track_ok_win == True, "ok", "CHECK (track jump)"), M.status)
M.to_csv(os.path.join(B, "photo_metrics.csv"), index=False)

for fn, col in [("twoplane_metrics.csv", "track_ok_win"), ("twoplane_fullspan_metrics.csv", "track_ok_full")]:
    T = pd.read_csv(os.path.join(B, fn))
    ok = M.groupby("group")[col].agg(lambda x: bool(x.fillna(False).all()))
    st = M.groupby("group").status_auto.agg(lambda x: (x == "ok").all())
    T["qc_track"] = T.group.map(ok).map({True: "ok", False: "jump"})
    T["status_final"] = np.where(T.group.map(st) & (T.qc_track == "ok"), "ok", "CHECK")
    T.to_csv(os.path.join(B, fn), index=False)
pd.set_option("display.width", 200)
print(M[["photo", "group", "status", "jump_win_mm", "jump_full_mm", "status_auto"]].round(2).to_string(index=False))
