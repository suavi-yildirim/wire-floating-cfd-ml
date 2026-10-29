"""Contact sheets of the far half of each photo's rule-aligned strip (upright), with the tracked path, for marking the
wire tip by hand. Columns are strip columns (ticks every 100 px); the strip is the same one the tracker uses.
usage: python3 tip_sheet.py <photo folder> <cache dir> <out dir> <csv out>"""
import sys, os, glob, re, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
P, CACHE, OUT, CSV = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
files = sorted(f for f in glob.glob(os.path.join(P, "*")) if f.lower().endswith((".heic", ".jpg", ".jpeg", ".png"))
               and (m := re.search(r"IMG_(\d+)", os.path.basename(f))) and 5393 <= int(m.group(1)) <= 5511)
tiles, meta = [], []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    im = wp.load_landscape(f, CACHE)
    try:
        p = wp.profile_any(im); D = wp.DBG
    except Exception as ex:
        p = None
    if p is None:
        meta.append(dict(photo=name, status="no wire found", tip_col=None)); continue
    # re-run the chosen scale to get its strip (profile_any returns the image used)
    imf = p["im"]; p2 = wp.profile(imf); D = wp.DBG
    strip = D["strip"]; path = D["path"]; vis = D["vis"]
    H, W = strip.shape[:2]
    x0 = int(W * 0.45)
    rows = path[vis] if vis.any() else path
    y_lo = max(0, int(np.percentile(rows, 2)) - 70); y_hi = min(H, int(np.percentile(rows, 98)) + 70)
    crop = strip[y_lo:y_hi, x0:][::-1].copy()          # upright
    yy = (H - 1 - path - y_lo)
    for c in range(0, W - x0, 1):
        r = int(np.clip(yy[x0 + c], 0, crop.shape[0] - 1)) if (x0 + c) < len(path) else None
        if r is not None and vis[x0 + c]:
            crop[r, c] = (0, 255, 0)
    sc = 1.3
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for t in range(x0 - x0 % 100 + 100, W, 100):
        xt = int((t - x0) * sc)
        if 0 <= xt < crop.shape[1]:
            cv2.line(crop, (xt, 0), (xt, 12), (0, 0, 255), 2)
            cv2.putText(crop, str(t), (xt + 2, 24), 0, 0.5, (0, 0, 255), 1)
    cv2.putText(crop, f"{name}  strip {W}px  view upright, cols from {x0}", (6, crop.shape[0] - 8), 0, 0.6, (0, 0, 255), 2)
    tiles.append(crop); meta.append(dict(photo=name, status="ok", strip_w=W, x0=x0, scale=sc, tip_col=None))
pd.DataFrame(meta).to_csv(CSV, index=False)
Wt = 1100
rs = []
for t in tiles:
    h = int(t.shape[0] * Wt / t.shape[1]); rs.append(cv2.resize(t, (Wt, h)))
per = 8
for k in range(0, len(rs), per):
    sheet = np.vstack([np.pad(r, ((4, 12), (0, 0), (0, 0)), constant_values=255) for r in rs[k:k + per]])
    cv2.imwrite(os.path.join(OUT, f"tips_{k // per + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(tiles), "tiles")
