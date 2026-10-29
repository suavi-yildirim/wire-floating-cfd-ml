"""Fit a smooth curve through the magenta dots on hand-marked sheets (1649 px frame).
usage: python3 fit_marked_wires.py <sheet folder> <out folder>
Writes <out>/<name>_curve.csv (x, y, arc length) and <out>/<name>_check.jpg (dots + curve + residuals)."""
import sys, os, glob, cv2, numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
SRC, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
summary = []
for f in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    name = os.path.splitext(os.path.basename(f))[0]
    im = cv2.imread(f); b, g, r = [im[..., i].astype(int) for i in range(3)]
    m = ((r > 200) & (b > 200) & (g < 120)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    pts = np.array([cen[i] for i in range(1, n) if st[i][4] >= 8 and st[i][2] <= 14 and st[i][3] <= 14])
    if len(pts) < 3:
        summary.append(dict(sheet=name, points=len(pts), status="not marked")); continue
    pts = pts[np.argsort(pts[:, 0])]
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 4]          # drop duplicate dots
    pts = pts[keep]
    if len(pts) < 3:
        summary.append(dict(sheet=name, points=len(pts), status="too few distinct points")); continue
    k = min(3, len(pts) - 1)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 2.0, k=k)
    u = np.linspace(0, 1, 2000)
    x, y = splev(u, tck)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    pd.DataFrame(dict(x=x.round(1), y=y.round(1), arc_px=arc.round(1))).to_csv(os.path.join(OUT, f"{name}_curve.csv"), index=False)
    xs, ys = splev(np.linspace(0, 1, 2000), tck)
    res = [float(np.min(np.hypot(xs - px, ys - py))) for px, py in pts]
    vis = im.copy()
    for X, Y in zip(xs.astype(int), ys.astype(int)):
        cv2.circle(vis, (X, Y), 1, (255, 255, 0), -1)
    for px, py in pts:
        cv2.circle(vis, (int(px), int(py)), 5, (255, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, f"{name}_check.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary.append(dict(sheet=name, points=len(pts), curve_len_px=round(float(arc[-1])), max_dot_to_curve_px=round(max(res), 1), status="fitted"))
S = pd.DataFrame(summary); S.to_csv(os.path.join(OUT, "fit_summary.csv"), index=False)
print(S.to_string(index=False))
