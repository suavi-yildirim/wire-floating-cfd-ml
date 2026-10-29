"""One gridded image per labelled photo for hand-marking wire points. Coordinates are in the tracker's standard frame
(landscape, 1649 px wide): x to the right, y downwards, grid every 100 px. usage: make_wire_sheets.py <photo folder> <cache> <out> <labels csv>"""
import sys, os, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
PH, CACHE, OUT, LAB = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
lab = pd.read_csv(LAB)
for r in lab.itertuples():
    im = wp.load_landscape(os.path.join(PH, r.photo + ".HEIC"), CACHE).copy()
    H, W = im.shape[:2]
    for x in range(0, W, 100):
        cv2.line(im, (x, 0), (x, H), (0, 0, 255), 1)
        cv2.putText(im, str(x), (x + 3, 18), 0, 0.5, (0, 0, 255), 1)
    for y in range(0, H, 100):
        cv2.line(im, (0, y), (W, y), (0, 0, 255), 1)
        cv2.putText(im, str(y), (3, y - 4), 0, 0.5, (0, 0, 255), 1)
    cv2.putText(im, f"{r.photo} UID {r.uid} inst {r.instance} plane {r.plane}", (8, H - 12), 0, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, f"{r.photo}_{r.uid}_{r.instance}_{r.plane}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(lab), "sheets")
