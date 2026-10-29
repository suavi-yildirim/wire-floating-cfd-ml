"""Photo figures for the thesis from a wire_pipeline.py output folder.

  fig_photo_contact_sheet : all photos, labelled with image number, capture time (local) and matched UID/instance
  fig_qc_montage_<len>    : tracked centrelines (QC overlays) cropped to the board, view A | view B per wire

Usage: python3 make_photo_figures.py <pipeline_out_folder> [photos_per_row]
Inputs : <out>/photo_metrics.csv (columns photo, time, group, view, uid, instance, length, status), <out>/_jpg/*.jpg,
         <out>/qc/qc_<photo>.jpg
"""
import os, sys
import cv2, numpy as np, pandas as pd

OUT = sys.argv[1]; NROW = int(sys.argv[2]) if len(sys.argv) > 2 else 12
M = pd.read_csv(os.path.join(OUT, "photo_metrics.csv"), parse_dates=["time"])
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def label(r):
    t = r.time.strftime("%d.%m. %H:%M:%S") if pd.notna(r.time) else ""
    u = f"UID {int(r.uid)}/{int(r.instance)}" if pd.notna(r.uid) else "unmatched"
    return r.photo.replace("IMG_", ""), t, u

# ---------- contact sheet (portrait thumbnails, original orientation)
W = 300; tiles = []
for r in M.sort_values("photo").itertuples():
    p = os.path.join(OUT, "_jpg", r.photo + ".jpg")
    if not os.path.exists(p): continue
    im = cv2.imread(p); s = W / im.shape[1]; im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    head = np.full((64, W, 3), 255, np.uint8)
    a, b, c = label(r)
    cv2.putText(head, f"{a}  {b}", (6, 26), FONT, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(head, c, (6, 54), FONT, 0.62, (108, 47, 0) if r.status == "ok" else (31, 130, 245), 2, cv2.LINE_AA)
    tiles.append(np.vstack([head, im]))
h = max(t.shape[0] for t in tiles)
tiles = [np.vstack([t, np.full((h - t.shape[0], W, 3), 255, np.uint8)]) for t in tiles]
pad = lambda t: cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
tiles = [pad(t) for t in tiles]
while len(tiles) % NROW: tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + NROW]) for i in range(0, len(tiles), NROW)])
cv2.imwrite(os.path.join(FIG, "fig_photo_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---------- QC montage: board crop of each QC overlay, A | B per group, one figure per wire length
def board_crop(r):
    p = os.path.join(OUT, "qc", f"qc_{r.photo}.jpg")
    if not os.path.exists(p): return None
    im = cv2.imread(p)
    # crop around the drawn centreline (pure green pixels) with margin
    g = (im[..., 1] > 240) & (im[..., 0] < 30) & ((im[..., 2] < 30) | ((im[..., 2] > 130) & (im[..., 2] < 190)))
    ys, xs = np.nonzero(g)
    if len(xs) < 50: return None
    x0, x1 = max(0, xs.min() - int(0.12 * (xs.max() - xs.min()))), min(im.shape[1], xs.max() + int(0.06 * (xs.max() - xs.min())))
    yc = int(np.median(ys)); hh = int(0.16 * (x1 - x0))
    c = im[max(0, yc - hh // 2 - 20):min(im.shape[0], yc + hh // 2 + 60), x0:x1]
    c = cv2.resize(c, (1100, int(c.shape[0] * 1100 / c.shape[1])), interpolation=cv2.INTER_AREA)
    a, b, u = label(r)
    cv2.putText(c, f"{a} ({r.view})  {u}", (10, 34), FONT, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(c, f"{a} ({r.view})  {u}", (10, 34), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return c

for L, sub in M.dropna(subset=["group"]).groupby(M.length.fillna(-1)):
    rows = []
    for gid, g in sub.sort_values(["time"]).groupby("group", sort=False):
        cs = [board_crop(r) for r in g.sort_values("view").itertuples()][:2]
        cs = [c for c in cs if c is not None]
        if not cs: continue
        hh = max(c.shape[0] for c in cs)
        cs = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cs]
        while len(cs) < 2: cs.append(np.full_like(cs[0], 255))
        rows.append(cv2.copyMakeBorder(np.hstack([cs[0], np.full((hh, 12, 3), 255, np.uint8), cs[1]]), 0, 12, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    if not rows: continue
    tag = "unmatched" if L < 0 else f"{L:.1f}m".replace(".", "p")
    for k in range(0, len(rows), 8):          # max 8 wires per page
        cv2.imwrite(os.path.join(FIG, f"fig_qc_montage_{tag}_p{k // 8 + 1}.jpg"), np.vstack(rows[k:k + 8]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
print(sorted(os.listdir(FIG)))
