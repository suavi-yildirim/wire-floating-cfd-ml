"""Left-section tiles (strip columns 0-800) for wires whose left part is not in the tip sheet. Same upright strip,
same 1.2 px/column display scale and tick labels as tip_sheet. usage: left_tiles.py <photo folder> <cache> <out> <photo ids>"""
import sys, os, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
PH, CACHE, OUT, IDS = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
SC = 1.2; COLS = 800
tiles = []
for pid in IDS.split(","):
    im = wp.load_landscape(glob.glob(os.path.join(PH, pid + ".*"))[0], CACHE)
    p = wp.profile_any(im)
    if p is None: print(pid, "no wire"); continue
    p = wp.profile(p["im"]); D = wp.DBG
    strip = D["strip"]; path = D["path"]; vis = D["vis"]; H = strip.shape[0]
    rows = path[vis] if vis.any() else path
    y_lo = max(0, int(np.percentile(rows, 2)) - 70); y_hi = min(H, int(np.percentile(rows, 98)) + 70)
    crop = strip[y_lo:y_hi, :COLS][::-1].copy()
    crop = cv2.resize(crop, None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
    for t in range(0, COLS, 100):
        xt = int(t * SC)
        cv2.line(crop, (xt, 0), (xt, 12), (0, 0, 255), 2)
        cv2.putText(crop, str(t), (xt + 2, 24), 0, 0.5, (0, 0, 255), 1)
    cv2.putText(crop, f"{pid}  left section, cols 0-{COLS}, {SC} px/col", (6, crop.shape[0] - 8), 0, 0.6, (0, 0, 255), 2)
    tiles.append(crop); print(pid, "tile", crop.shape, flush=True)
sheet = np.vstack([np.pad(t, ((4, 14), (0, 0), (0, 0)), constant_values=255) for t in tiles])
cv2.imwrite(os.path.join(OUT, "left_tiles_01.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("saved", len(tiles), "tiles")
