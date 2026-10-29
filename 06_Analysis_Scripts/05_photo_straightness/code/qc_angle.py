"""Angle QC of tracked centrelines. For each photo, the local slope angle theta(s) = atan(de/ds) is measured along the
centreline (both the 90-490 mm window and the full span). A bending change |theta(s+W/2) - theta(s-W/2)| above ANGLE_MAX
(deg) is a violation: real wires bend ~10-15 deg at most over 20 mm, so larger values mean the tracker left the wire.
The usable wire is cut at the first violation found when walking outward from the window centre (290 mm).
Output: <out>/angle_qc.csv with max |dtheta|, the violation position, the usable range and a status.
usage: python3 qc_angle.py <pipeline out dir> [ANGLE_MAX_DEG=15] [W_MM=20]"""
import sys, os
import numpy as np, pandas as pd
D = sys.argv[1]
AMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
W = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
CENTRE = 290.0
GAP_MM = 30.0           # gaps between detected segments up to this length are bridged by a straight line
SMOOTH_MM = 8.0          # smoothing of the centreline (FWHM) before measuring its slope angle

def analyse(d):
    d = d.sort_values("s_mm"); s, e = d.s_mm.values, d.e_mm.values
    g = np.arange(s.min(), s.max(), 0.5); eg = np.interp(g, s, e)
    # grid points inside a data gap (tape / dropped columns) are interpolated, not measured: exclude them
    # gaps between detected segments: bridged with a straight line between the segment ends when shorter than GAP_MM;
    # longer gaps stay undefined (the wire is not measured there)
    k = np.clip(np.searchsorted(s, g), 1, len(s) - 1)
    near = np.minimum(np.abs(g - s[k - 1]), np.abs(s[k] - g))
    brg = np.zeros(len(g), bool)
    gi = np.where(np.diff(s) > 1.0)[0]
    for i in gi:
        gl = s[i + 1] - s[i]
        if gl <= GAP_MM:
            m = (g > s[i]) & (g < s[i + 1])
            eg[m] = e[i] + (e[i + 1] - e[i]) * (g[m] - s[i]) / gl
            brg[m] = True
        else:
            eg[(g > s[i]) & (g < s[i + 1])] = np.nan
    bridged_mm = float(brg.sum() * 0.5)
    h = int(round(W / 2 / 0.5))
    # the track moves in whole pixels (steps ~0.45 mm every few mm): smooth over SMOOTH_MM before differentiating
    from scipy.ndimage import gaussian_filter1d
    valid = np.isfinite(eg)
    filled = np.interp(g, g[valid], eg[valid])
    sm = gaussian_filter1d(filled, SMOOTH_MM / 0.5 / 2.355)
    sm[~valid] = np.nan
    th = np.degrees(np.arctan(np.gradient(sm, 0.5)))
    dth = np.full_like(th, np.nan)
    dth[h:-h] = th[2 * h:] - th[:-2 * h]                  # theta(s + W/2) - theta(s - W/2)
    bad = np.abs(dth) > AMAX
    ok = np.isfinite(dth)
    bad = bad & ok
    out = dict(bridged_mm=bridged_mm, s_from=float(g[0]), s_to=float(g[-1]), max_dtheta_deg=float(np.nanmax(np.abs(dth))),
               s_at_max=float(g[np.nanargmax(np.abs(dth))]))
    ci = int(np.argmin(np.abs(g - CENTRE)))
    left = np.where(bad[:ci] & ok[:ci])[0]; right = np.where(bad[ci:] & ok[ci:])[0]
    lo_v = g[left[-1]] + W / 2 if len(left) else g[0]
    hi_v = g[ci + right[0]] - W / 2 if len(right) else g[-1]
    out.update(usable_from=float(max(lo_v, g[0])), usable_to=float(min(hi_v, g[-1])),
               first_violation_mm=float(g[ci + right[0]]) if len(right) else np.nan,
               n_violations=int(np.sum(bad & ok)))
    return out

rows = []
for fn, tag in [("photo_centrelines.csv", "window"), ("photo_centrelines_fullspan.csv", "full")]:
    p = os.path.join(D, fn)
    if not os.path.exists(p): continue
    C = pd.read_csv(p)
    for ph, d in C.groupby("photo"):
        if len(d) < 60: continue
        r = analyse(d); r.update(photo=ph, span=tag)
        fv = r["first_violation_mm"]
        r["status"] = ("OK" if not np.isfinite(fv) else
                       ("FLAG - violation inside 90-490 mm: improve detection" if 90 <= fv <= 490 else
                        f"ENOUGH - stop at {fv:.0f} mm, use the usable range"))
        rows.append(r)
R = pd.DataFrame(rows)[["photo", "span", "bridged_mm", "max_dtheta_deg", "s_at_max", "first_violation_mm", "n_violations",
                        "usable_from", "usable_to", "status"]]
R.round(1).to_csv(os.path.join(D, "angle_qc.csv"), index=False)
pd.set_option("display.width", 220)
print(f"angle limit {AMAX} deg over {W} mm")
print(R.round(1).to_string(index=False))
