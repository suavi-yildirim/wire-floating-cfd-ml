"""Large-label tiles for the enlarged (1.5-2x) photos: two overlapping half-width crops per photo, full strip height,
upright. Columns are the tracker's strip columns (ticks every 100). usage: tip_sheet_large.py <photo folder> <cache> <out> <index csv>"""
import sys, os, glob, re, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
P, CACHE, OUT, IDX = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
idx = pd.read_csv(IDX)
big = set(idx.loc[idx.strip_w > 2000, "photo"])
files = sorted(f for f in glob.glob(os.path.join(P, "*")) if os.path.splitext(os.path.basename(f))[0] in big)
tiles = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    im = wp.load_landscape(f, CACHE)
    try:
        p = wp.profile_any(im)
        if p is None: continue
        p = wp.profile(p["im"])
    except Exception as ex:
        print(name, ex); continue
    if p is None: continue
    D = wp.DBG; strip = D["strip"]; H, W = strip.shape[:2]
    up = strip[::-1].copy()                                  # upright, full height
    btop = int(D["btop"]); top_crop = max(0, H - (btop + 90))
    up = up[top_crop:]
    for x0, x1 in ((1100, min(W, 2000)), (min(W - 1, 1900), W)):
        if x1 - x0 < 300: continue
        c = up[:, x0:x1].copy()
        c = cv2.resize(c, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        for t in range(((x0 // 100) + 1) * 100, x1, 100):
            xt = int((t - x0) * 1.2)
            cv2.line(c, (xt, 0), (xt, 26), (0, 0, 255), 3)
            cv2.putText(c, str(t), (xt + 4, 34), 0, 0.9, (0, 0, 255), 2)
        cv2.putText(c, f"{name}  cols {x0}-{x1}", (8, c.shape[0] - 10), 0, 0.9, (0, 0, 255), 2)
        tiles.append(c)
    print(name, "ok", flush=True)
Wt = 1100; per = 6
rs = [cv2.resize(t, (Wt, int(t.shape[0] * Wt / t.shape[1]))) for t in tiles]
for k in range(0, len(rs), per):
    sheet = np.vstack([np.pad(r, ((4, 14), (0, 0), (0, 0)), constant_values=255) for r in rs[k:k + per]])
    cv2.imwrite(os.path.join(OUT, f"large_{k // per + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(tiles), "tiles")
