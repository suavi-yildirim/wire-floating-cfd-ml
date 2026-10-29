"""Locate the bold 100/200/300/400 labels of the steel rule by multi-scale template matching.
Templates are cut from img-010.jpg (BLU_05_02) at manually verified label centres."""
import sys, os, cv2, numpy as np, pandas as pd
SRC, OUT = sys.argv[1], sys.argv[2]
IM = {"BLU_05_01":"img-005.jpg","BLU_05_02":"img-010.jpg","BLU_05_03":"img-011.jpg","YEL_05_01":"img-012.jpg","YEL_05_02":"img-013.jpg","YEL_05_03":"img-014.jpg","PUR_05_01":"img-019.jpg","PUR_05_02":"img-020.jpg","PUR_05_03":"img-021.jpg","RED_05_01":"img-022.jpg","RED_05_02":"img-023.jpg","RED_05_03":"img-024.jpg"}
ref = cv2.cvtColor(cv2.imread(os.path.join(SRC, "img-010.jpg")), cv2.COLOR_BGR2GRAY)
CENT = {100: (517, 690), 200: (804, 690), 300: (1072, 690), 400: (1329, 690)}
T = {v: ref[y - 8:y + 8, x - 15:x + 15] for v, (x, y) in CENT.items()}
rows = []
for lab, fn in IM.items():
    g = cv2.cvtColor(cv2.imread(os.path.join(SRC, fn)), cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    y0, y1 = (0, H) if H < 400 else (520, 880)
    band = g[y0:y1]
    for v, t in T.items():
        best = None
        for s in np.arange(0.75, 1.35, 0.025):
            for ang in np.arange(-5, 5.1, 1.0):
                ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                M = cv2.getRotationMatrix2D((ts.shape[1] / 2, ts.shape[0] / 2), ang, 1)
                tr = cv2.warpAffine(ts, M, (ts.shape[1], ts.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                if tr.shape[0] >= band.shape[0] or tr.shape[1] >= band.shape[1]: continue
                r = cv2.matchTemplate(band, tr, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if best is None or mx > best[0]:
                    best = (mx, loc[0] + tr.shape[1] / 2, y0 + loc[1] + tr.shape[0] / 2, s, ang)
        rows.append(dict(label=lab, value_mm=v, score=best[0], x_px=best[1], y_px=best[2], scale=best[3], angle=best[4]))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "ruler_tmatch.csv"), index=False)
print(d.round(3).to_string(index=False))
