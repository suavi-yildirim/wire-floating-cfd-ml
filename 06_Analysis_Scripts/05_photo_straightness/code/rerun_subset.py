"""Rerun the current tracker on a subset of WC08 photos (v2 grouping kept) and compare with the v2 two-plane results.
usage: rerun_subset.py <photo folder> <v2 photo_metrics.csv> <v2 twoplane_metrics.csv> <out dir> <cache> <uid list>"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
PH, MET, TWO, OUT, CACHE, UIDS = sys.argv[1:7]
UIDS = [int(u) for u in UIDS.split(",")]
os.makedirs(OUT, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
M = pd.read_csv(MET); M = M[M.uid.isin(UIDS)]
T = pd.read_csv(TWO); T = T[T.uid.isin(UIDS)]
curves = {}; rows = []
for r in M.itertuples():
    im = wp.load_landscape(os.path.join(PH, r.photo + ".HEIC"), CACHE)
    try:
        p = wp.profile_any(im)
    except Exception as ex:
        p = None; print(r.photo, ex)
    if p is None:
        rows.append(dict(photo=r.photo, group=r.group, view=r.view, status="no wire")); continue
    curves[(int(r.group), r.view)] = (p["s"], p["e"])
    rows.append(dict(photo=r.photo, group=r.group, view=r.view, status="ok", span=round(p["span_mm"]), clamp=p["clamp_src"], cal_rms=round(p["cal_rms"], 2)))
    print(r.photo, "ok", p["clamp_src"], round(p["span_mm"]), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "photo_status.csv"), index=False)
new = []
for g in sorted({k[0] for k in curves}):
    if (g, "A") not in curves or (g, "B") not in curves: continue
    sa, ea = curves[(g, "A")]; sb, eb = curves[(g, "B")]
    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
    if hi - lo < 100: continue
    sg = np.linspace(lo, hi, 400)
    ey = np.interp(sg, sa, ea); ez = np.interp(sg, sb, eb); rr = np.hypot(ey, ez)
    new.append(dict(group=g, rmax_new=round(float(rr.max()), 2), rrms_new=round(float(np.sqrt(np.mean(rr ** 2))), 2),
                    dmax_y_new=round(float(np.abs(ey).max()), 2), dmax_z_new=round(float(np.abs(ez).max()), 2),
                    s_from=round(lo), s_to=round(hi)))
N = pd.DataFrame(new)
C = T[["group", "uid", "instance", "rmax_mm", "rrms_mm"]].rename(columns={"rmax_mm": "rmax_v2", "rrms_mm": "rrms_v2"}).merge(N, on="group", how="left")
C.round(2).to_csv(os.path.join(OUT, "compare_v2_vs_current.csv"), index=False)
pd.set_option("display.width", 220); print(C.round(2).to_string(index=False))
