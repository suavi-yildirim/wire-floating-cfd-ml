"""Analysis of hand-marked wires (sheets_clean): spline through magenta dots (1649-px frame), mm scale from the two
cyan rule dots (300 mm apart) or from the printed rule labels (fallback), deviation from the chord, bend angle, and
two-plane resultant for X/Y pairs. usage: analyse_marked.py <sheet folder> <out folder> <photo folder> <jpg cache>"""
import sys, os, glob, re, cv2, numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
SRC, OUT, PH, CACHE = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
# rule-dot spacing per sheet where it was not 300 mm (checked by hand)
CAL_SPAN_MM = {"IMG_6089_23_3_X": 400.0}
lab = pd.read_csv(os.path.join(os.path.dirname(OUT), "labels_more.csv")).set_index("photo")

def blobs(mask, lo=8, hi=14, min_area=8):   # magenta dots ~ 6 px across; cyan dots need a larger minimum (see below)
    n, _, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return np.array([cen[i] for i in range(1, n) if st[i][4] >= min_area and st[i][2] <= hi and st[i][3] <= hi])

def label_scale(photo):
    """px per mm from the printed rule labels (wire_pipeline.profile_any finds them); None if not found."""
    im = wp.load_landscape(os.path.join(PH, photo + ".HEIC"), CACHE)
    for f in (1.0, 1.75, 1.5, 2.0):
        imf = im if f == 1.0 else cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(imf, cv2.COLOR_BGR2GRAY)
        H = g.shape[0]
        try:
            L = wp.ruler_labels_band(g, int(0.30 * H), int(0.92 * H)).sort_values("value_mm")
        except Exception:
            continue
        good = L[L.score > 0.6]
        if len(good) >= 3:
            d = np.hypot(np.diff(good.x_px.values), np.diff(good.y_px.values))
            dv = np.diff(good.value_mm.values.astype(float))
            ok = dv > 0
            return float(np.median(d[ok] / dv[ok])) if ok.any() else None, f
    return None, None

rows, curves = [], {}
for fpath in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    name = os.path.splitext(os.path.basename(fpath))[0]
    photo = name.split("_")[0] + "_" + name.split("_")[1]
    im = cv2.imread(fpath); b, g, r = [im[..., i].astype(int) for i in range(3)]
    mag = blobs(((r > 200) & (b > 200) & (g < 120)))
    cy = blobs(((b > 200) & (g > 200) & (r < 120)), min_area=18)   # cyan user dots ~24-26 px; wire highlights ~12 px
    if len(mag) < 3: continue
    pts = mag[np.argsort(mag[:, 0])]
    pts = pts[np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 4]]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 2.0, k=min(3, len(pts) - 1))
    u = np.linspace(0, 1, 4000); x, y = splev(u, tck)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    # scale
    ppm, src = None, None
    if len(cy) >= 2:
        cy = cy[np.argsort(cy[:, 0])]
        span_mm = CAL_SPAN_MM.get(name, 300.0)              # hand-checked spacing of the two rule dots (default 300 mm)
        ppm = float(np.hypot(*(cy[-1] - cy[0])) / span_mm); src = f"cyan dots ({span_mm:.0f} mm)"
    else:
        ls, f = label_scale(photo)
        if ls: ppm = ls; src = f"printed labels (x{f})"
    rec = dict(sheet=name, uid=int(lab.loc[photo, "uid"]), instance=int(lab.loc[photo, "instance"]), plane=name.split("_")[-1],
               dots=len(pts), cyan=len(cy), scale_src=src or "none", px_per_mm=round(ppm, 3) if ppm else np.nan)
    rows.append(rec)
    curves[name] = dict(x=x, y=y, arc=arc, pts=pts, ppm=ppm)
    pd.DataFrame(dict(x=x.round(1), y=y.round(1), arc_px=arc.round(1))).to_csv(os.path.join(OUT, f"{name}_curve.csv"), index=False)
R = pd.DataFrame(rows)
def chord_metrics(c):
    x, y, arc, ppm = c["x"], c["y"], c["arc"], c["ppm"]
    ux, uy = x[-1] - x[0], y[-1] - y[0]; L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    s = ((x - x[0]) * ux + (y - y[0]) * uy)          # px along chord
    e = (-(x - x[0]) * uy + (y - y[0]) * ux)         # px lateral
    return s, e
M = []
for name, c in curves.items():
    if not c["ppm"]: 
        M.append(dict(sheet=name, status="no scale")); continue
    s, e = chord_metrics(c)
    e_mm = e / c["ppm"]; s_mm = s / c["ppm"]
    # bend angle over 20 mm, from the chord-referenced curve
    g = np.arange(0, s_mm[-1], 1.0); eg = np.interp(g, s_mm, e_mm)
    th = np.degrees(np.arctan(np.gradient(eg, 1.0)))
    dth = np.abs(th[20:] - th[:-20]) if len(th) > 21 else np.array([np.nan])
    M.append(dict(sheet=name, length_mm=round(c["arc"][-1] / c["ppm"], 1), chord_mm=round(s_mm[-1], 1),
                  dev_max_mm=round(float(np.abs(e_mm).max()), 2), dev_rms_mm=round(float(np.sqrt(np.mean(e_mm ** 2))), 2),
                  max_bend_20mm_deg=round(float(np.nanmax(dth)), 1), status="ok"))
    curves[name]["s_mm"] = s_mm; curves[name]["e_mm"] = e_mm
Mt = pd.DataFrame(M)
A = R.merge(Mt, on="sheet", how="left")
A.to_csv(os.path.join(OUT, "wire_metrics.csv"), index=False)
# two-plane (X + Y of the same UID/instance)
T = []
for (u, i), g_ in A[A.status == "ok"].groupby(["uid", "instance"]):
    if set(g_.plane) != {"X", "Y"}: continue
    cx = curves[g_[g_.plane == "X"].sheet.iloc[0]]; cyv = curves[g_[g_.plane == "Y"].sheet.iloc[0]]
    lo = max(cx["s_mm"].min(), cyv["s_mm"].min()); hi = min(cx["s_mm"].max(), cyv["s_mm"].max())
    sg = np.linspace(lo, hi, 400)
    ex = np.interp(sg, cx["s_mm"], cx["e_mm"]); ey = np.interp(sg, cyv["s_mm"], cyv["e_mm"]); rr = np.hypot(ex, ey)
    T.append(dict(uid=u, instance=i, length_mm=g_.length_mm.max(), rmax_mm=round(rr.max(), 2), rrms_mm=round(float(np.sqrt(np.mean(rr ** 2))), 2),
                  scale_check=f"{g_.scale_src.iloc[0]} / {g_.scale_src.iloc[1]}"))
pd.DataFrame(T).to_csv(os.path.join(OUT, "twoplane_marked.csv"), index=False)
pd.set_option("display.width", 220)
print(A[["sheet", "dots", "cyan", "scale_src", "px_per_mm", "length_mm", "dev_max_mm", "dev_rms_mm", "max_bend_20mm_deg", "status"]].to_string(index=False))
print(); print(pd.DataFrame(T).to_string(index=False) if T else "no two-plane pairs yet")
