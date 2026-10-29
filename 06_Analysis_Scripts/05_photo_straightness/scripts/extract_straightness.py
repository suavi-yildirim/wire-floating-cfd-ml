"""Extract wire centrelines from WFE_Straightness.pdf photos and compute straightness metrics.

Input : JPEGs extracted with `pdfimages -j WFE_Straightness.pdf img` (raw, un-rotated camera frames).
Output: straightness_metrics.csv, centrelines.csv, QC overlay images.
"""
import sys, os, json
import cv2
import numpy as np
import pandas as pd

SRC = sys.argv[1]          # folder with img-XXX.jpg
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# PDF page -> image index -> label (page text)
IMAGES = {
    "BLU_05_01": "img-005.jpg",   # cropped strip on page 2
    "BLU_05_02": "img-010.jpg", "BLU_05_03": "img-011.jpg",
    "YEL_05_01": "img-012.jpg", "YEL_05_02": "img-013.jpg", "YEL_05_03": "img-014.jpg",
    "PUR_05_01": "img-019.jpg", "PUR_05_02": "img-020.jpg", "PUR_05_03": "img-021.jpg",
    "RED_05_01": "img-022.jpg", "RED_05_02": "img-023.jpg", "RED_05_03": "img-024.jpg",
}
CAL = pd.read_csv(sys.argv[3])
S_WIN = (90.0, 490.0)   # mm on the steel rule
SECTION = {"BLU": 1.0, "YEL": 1.5, "PUR": 4.0, "RED": 6.0}
TRIM = 0.04    # fraction trimmed at each end (tape / ferrule transitions)


def wire_mask(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    bg = cv2.medianBlur(lab, 31).astype(np.float32)
    d = lab.astype(np.float32) - bg
    score = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2) + 0.3 * np.abs(d[..., 0])
    m = (score > 18).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H = im.shape[0]
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        yc = y + h / 2
        if w > 0.55 * im.shape[1] and h < w / 6 and a < w * 25 and 0.25 * H < yc < 0.8 * H:
            cands.append((w, i))
    if not cands:  # strip images: relax vertical window
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if w > 0.55 * im.shape[1] and h < w / 6 and a < w * 25:
                cands.append((w, i))
    cands.sort(reverse=True)
    return (cc == cands[0][1])


def centreline(mask):
    ys, xs = np.nonzero(mask)
    cols = np.unique(xs)
    yc = np.array([ys[xs == c].mean() for c in cols])
    th = np.array([(xs == c).sum() for c in cols])
    # reject columns with abnormal thickness (tape edges, ferrules, blobs)
    med = np.median(th)
    ok = th < 2.5 * med
    cols, yc = cols[ok], yc[ok]
    # trim ends
    L = cols.max() - cols.min()
    keep = (cols > cols.min() + TRIM * L) & (cols < cols.max() - TRIM * L)
    cols, yc = cols[keep].astype(float), yc[keep]
    # light smoothing (5 px running median) to remove JPEG/edge noise
    yc = pd.Series(yc).rolling(9, center=True, min_periods=1).median().values
    return cols, yc


def ruler_scale(im, cols, yc):
    """px per mm from the cm-tick periodicity of the steel rule below the wire.
    Scans rows below the wire; in sliding windows takes the FFT peak in the 10 mm band."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = np.polyfit(cols, yc, 1)
    x0, x1 = int(cols.min()), int(cols.max())
    best = None
    for off in range(25, min(260, im.shape[0] - int(yc.max()) - 2), 2):
        xs = np.arange(x0, x1)
        ys = np.clip((np.polyval(p, xs) + off).astype(int), 0, im.shape[0] - 1)
        prof = g[ys, xs] - np.convolve(g[ys, xs], np.ones(61) / 61, mode="same")
        prof = prof[40:-40]
        n = len(prof)
        f = np.fft.rfftfreq(n * 4)
        P = np.abs(np.fft.rfft(prof * np.hanning(n), n * 4))
        band = (f > 1 / 45) & (f < 1 / 18)  # 10 mm period expected 20-40 px
        k = np.argmax(P[band]); peak = P[band][k]
        snr = peak / np.median(P[band])
        if best is None or snr > best[0]:
            best = (snr, off, 1 / f[band][k])
    snr, off, period = best
    # local scale along x (keystone): windows of 300 px
    xs_all = np.arange(x0, x1)
    ys_all = np.clip((np.polyval(p, xs_all) + off).astype(int), 0, im.shape[0] - 1)
    prof = g[ys_all, xs_all] - np.convolve(g[ys_all, xs_all], np.ones(61) / 61, mode="same")
    loc = []
    W = 300
    for s in range(40, len(prof) - W - 40, 100):
        seg = prof[s:s + W]
        f = np.fft.rfftfreq(W * 8)
        P = np.abs(np.fft.rfft(seg * np.hanning(W), W * 8))
        band = (f > 1 / (period * 1.25)) & (f < 1 / (period * 0.8))
        loc.append((x0 + s + W / 2, 1 / f[band][np.argmax(P[band])] / 10.0))
    loc = np.array(loc)
    return snr, off, period / 10.0, loc


rows, curves, qc = [], [], []
for label, fn in IMAGES.items():
    im = cv2.imread(os.path.join(SRC, fn))
    m = wire_mask(im)
    cols, yc = centreline(m)
    # --- ruler calibration (template-matched 100/200/300/400 mm labels) ---
    rc = CAL[CAL.label == label].sort_values("value_mm")
    lx, ly, lv = rc.x_px.values, rc.y_px.values, rc.value_mm.values.astype(float)
    pr = np.polyfit(lx, ly, 1)                       # ruler axis in image
    ax = np.array([1.0, pr[0]]); ax /= np.linalg.norm(ax)
    O = np.array([lx[0], ly[0]])
    lpos = (np.stack([lx, ly], 1) - O) @ ax           # label positions along axis [px]
    A = np.stack([lpos, np.ones_like(lpos), -lpos * lv], 1)
    pp = np.linalg.lstsq(A, lv, rcond=None)[0]       # 1-D projective map  mm = (a q + b)/(c q + 1)
    to_mm = lambda q: (pp[0] * q + pp[1]) / (pp[2] * q + 1)
    cal_rms = np.sqrt(np.mean((to_mm(lpos) - lv) ** 2))
    # wire points -> along-axis coordinate q and perpendicular offset w (px)
    P = np.stack([cols, yc], 1) - O
    q = P @ ax; w = P @ np.array([-ax[1], ax[0]])
    s_mm = to_mm(q)                                   # along-axis position in mm (ruler reading)
    k = (to_mm(q + 0.5) - to_mm(q - 0.5))             # local mm per px
    w_mm = w * k                                       # lateral position in mm (isotropic local scale)
    # common evaluation window on the rule (free span between the tapes, identical for all wires)
    win = (s_mm >= S_WIN[0]) & (s_mm <= S_WIN[1])
    s_mm, w_mm, cols, yc, k = s_mm[win], w_mm[win], cols[win], yc[win], k[win]
    o = np.argsort(s_mm); s_mm, w_mm, k = s_mm[o], w_mm[o], k[o]
    # robust chord through mean of first / last 3 % of points
    nE = max(5, int(0.03 * len(s_mm)))
    P0 = np.array([s_mm[:nE].mean(), w_mm[:nE].mean()]); P1 = np.array([s_mm[-nE:].mean(), w_mm[-nE:].mean()])
    t = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-t[1], t[0]])
    rel = np.stack([s_mm, w_mm], 1) - P0
    u_mm = rel @ t; e_mm = rel @ nrm
    o = np.argsort(u_mm); u_mm, e_mm = u_mm[o], e_mm[o]
    u_mm = u_mm - u_mm[0]
    Lc = np.linalg.norm(P1 - P0)
    snr, s_x = cal_rms, 1 / k
    off = 0
    # arc length
    seg = np.hypot(np.diff(u_mm), np.diff(e_mm)); arc = seg.sum()
    # low-order fits on normalised coordinate
    xi = np.clip(u_mm / u_mm[-1], 0, 1)
    c2 = np.polyfit(xi, e_mm, 2); bow = -c2[0] / 4.0       # parabola through ends: e = 4*bow*xi(1-xi) approx
    sin_fit = np.linalg.lstsq(np.stack([np.sin(np.pi * xi), np.sin(2 * np.pi * xi), np.sin(3 * np.pi * xi)], 1), e_mm, rcond=None)[0]
    lowfit = np.stack([np.sin(np.pi * xi), np.sin(2 * np.pi * xi), np.sin(3 * np.pi * xi)], 1) @ sin_fit
    resid = e_mm - lowfit
    col = label[:3]
    rows.append(dict(
        label=label, colour=col, section_mm2=SECTION[col], instance=int(label[-2:]), length_m=0.5,
        image=fn, chord_mm=Lc, cal_rms_mm=cal_rms, ruler_s0_mm=float(s_mm.min()), ruler_s1_mm=float(s_mm.max()), px_per_mm=float(np.mean(s_x)),
        dmax_mm=np.abs(e_mm).max(), dpp_mm=e_mm.max() - e_mm.min(), erms_mm=np.sqrt(np.mean(e_mm ** 2)),
        dmax_over_L=np.abs(e_mm).max() / Lc, 
        A1_mm=sin_fit[0], A2_mm=sin_fit[1], A3_mm=sin_fit[2],
        R_bow_m=(Lc / 1000) ** 2 / (8 * abs(sin_fit[0]) / 1000) if abs(sin_fit[0]) > 1e-6 else np.inf,
        wavy_rms_mm=np.sqrt(np.mean(resid ** 2)),
    ))
    curves.append(pd.DataFrame(dict(label=label, s_mm=u_mm, xi=xi, e_mm=e_mm)))
    # QC overlay
    vis = im.copy()
    for x, y in zip(cols.astype(int), yc.astype(int)):
        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    for x, y in zip(lx, ly):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), 2)
    cv2.putText(vis, label, (20, 40), 0, 1.2, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(OUT, f"qc_{label}.jpg"), vis)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "straightness_metrics.csv"), index=False)
pd.concat(curves).to_csv(os.path.join(OUT, "centrelines.csv"), index=False)
pd.set_option("display.width", 250)
print(df.drop(columns=["image"]).round(3).to_string(index=False))
