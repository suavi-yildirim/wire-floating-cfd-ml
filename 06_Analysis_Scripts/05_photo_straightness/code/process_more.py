"""Labelled photos (UID, instance, plane X/Y from more/labels_more.csv): match each (UID, instance) to its run session
(nearest start within +-60 min of the photos), track both planes, build the two-plane curve and join the stable pressure.
usage: python3 process_more.py <photo folder> <jpg cache> <out dir> <runlog> <stable summary csv>"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
import cv2
PH, CACHE, OUT, RL, SP = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
QC = os.path.join(OUT, "qc"); os.makedirs(QC, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
lab = pd.read_csv(os.path.join(os.path.dirname(OUT), "labels_more.csv"))
lab["t"] = [wp.capture_time(os.path.join(PH, p + ".HEIC")) for p in lab.photo]
rl = pd.read_csv(RL, parse_dates=["start_local", "end_local"]); S = wp.sessions(rl)
sp = pd.read_csv(SP); sp = sp[sp.System_Stable == True]
P = "Pressure_Stable_Average (mbar)"
# 1) (uid, instance) -> session
grp = {}
for (u, i), g in lab.groupby(["uid", "instance"]):
    s = S[(S.uid == u) & (S.instance == i)].copy()
    tm = g.t.median()
    lead = (s.start - tm).dt.total_seconds() / 60
    c = s[(lead.abs() <= 60)]
    if len(c):
        best = c.loc[(lead[c.index]).abs().idxmin()]
        grp[(u, i)] = dict(session_start=best.start, n_runs=int(best.n_runs), files=best.files, lead_min=float((best.start - tm).total_seconds() / 60))
    else:
        grp[(u, i)] = None
# 2) track every labelled photo
curves, rows = [], []
for r in lab.itertuples():
    im = wp.load_landscape(os.path.join(PH, r.photo + ".HEIC"), CACHE)
    try:
        p = wp.profile_any(im)
    except Exception as ex:
        p = None; print(r.photo, ex, flush=True)
    key = (r.uid, r.instance)
    info = dict(photo=r.photo, uid=r.uid, instance=r.instance, plane=r.plane, time=str(r.t))
    if p is None:
        rows.append({**info, "status": "no wire found"}); print(r.photo, "no wire", flush=True); continue
    g = grp.get(key)
    info.update(colour=p["colour"], expected=wp.UID_COLOUR.get(int(r.uid)), cal_rms=round(p["cal_rms"], 2),
                span_mm=round(p["span_mm"]), clamp=p["clamp_src"], n_labels=p["n_labels"], upscale=p["upscale"],
                session_start=None if g is None else g["session_start"], lead_min=None if g is None else round(g["lead_min"], 1))
    rows.append({**info, "status": "ok"})
    curves.append(pd.DataFrame(dict(photo=r.photo, uid=r.uid, instance=r.instance, plane=r.plane, s_mm=p["s"], e_mm=p["e"])))
    print(r.photo, "ok", p["clamp_src"], round(p["span_mm"]), flush=True)
    vis = p["im"].copy()
    if p.get("full") is not None:      # whole free span (light green beyond 500 mm: scale extrapolated)
        for (x, y), sv in zip(p["full"]["img_pts"].astype(int), p["full"]["s"]):
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0) if sv <= 500 else (0, 255, 160), -1)
    for x, y in p["img_pts"].astype(int):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    for L in p["labels"].itertuples():
        cv2.circle(vis, (int(L.x_px), int(L.y_px)), 7, (0, 0, 255) if L.score > 0.6 else (0, 255, 255), 2)
    for c in p["clamp_pts"]:
        cv2.drawMarker(vis, tuple(c.astype(int)), (255, 0, 255), cv2.MARKER_TRIANGLE_DOWN, 22, 3)
    cv2.putText(vis, f"{r.photo} UID{r.uid} inst{r.instance} plane {r.plane}  {p['clamp_src']}  span {p['sL']:.0f}-{p['sR']:.0f} mm",
                (20, 45), 0, 1.0, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(QC, f"{r.photo}_{r.uid}_{r.instance}_{r.plane}.jpg"), vis)
M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT, "photo_metrics.csv"), index=False)
C = pd.concat(curves) if curves else pd.DataFrame(); C.to_csv(os.path.join(OUT, "photo_centrelines.csv"), index=False)
C = C.rename(columns={"plane": "view"}) if len(C) else C
# 3) two-plane per (uid, instance)
two = []
if len(C):
    for (u, i), g in C.groupby(["uid", "instance"]):
        if set(g.view) != {"X", "Y"}: continue
        gx = g[g.view == "X"].sort_values("s_mm"); gy = g[g.view == "Y"].sort_values("s_mm")
        lo = max(gx.s_mm.min(), gy.s_mm.min()); hi = min(gx.s_mm.max(), gy.s_mm.max())
        sg = np.linspace(lo, hi, 400)
        ey = np.interp(sg, gx.s_mm, gx.e_mm); ez = np.interp(sg, gy.s_mm, gy.e_mm); rr = np.hypot(ey, ez)
        gg = g.groupby("view").photo.first()
        s_ = grp.get((u, i))
        pr = np.nan
        if s_ is not None:
            x = sp[sp.File.isin(s_["files"].split(";"))]
            pr = x[P].mean() if len(x) else np.nan
        two.append(dict(uid=u, instance=i, photos=f"{gx.photo.iloc[0]}+{gy.photo.iloc[0]}", s_from_mm=lo, s_to_mm=hi,
                        dmax_x_mm=np.abs(ey).max(), dmax_z_mm=np.abs(ez).max(), rmax_mm=rr.max(), rrms_mm=np.sqrt(np.mean(rr ** 2)),
                        session_start=None if s_ is None else s_["session_start"], p_stable_mbar=pr))
T = pd.DataFrame(two); T.to_csv(os.path.join(OUT, "twoplane.csv"), index=False)
print(f"done: {sum(M.status=='ok')} tracked of {len(M)}, two-plane wires {len(T)}, matched instances {sum(v is not None for v in grp.values())}/{len(grp)}", flush=True)
