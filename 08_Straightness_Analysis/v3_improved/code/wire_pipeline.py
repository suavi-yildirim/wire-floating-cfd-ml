"""Batch pipeline: iPhone HEIC/JPG wire photos -> matched WFE run -> 2-plane centreline -> straightness metrics.

Usage:
  python3 -I wire_pipeline.py --photos DIR --runlog runlog.csv --template img-010.jpg --out OUTDIR

Steps
 1. Read EXIF capture time (local, UTC+01:00 in March) straight from the file bytes (works for HEIC/JPG).
 2. Group photos taken < PAIR_GAP_S apart -> same wire, view A (first) / view B (second, rotated 90 deg about axis).
 3. Match each group to the next WFE "session" (consecutive runs of one wire instance, split at gaps > SESSION_GAP_MIN)
    that starts after the photo and is not yet claimed.   Run times come from raw-log headers converted to local time.
 4. Decode (HEIC grid -> JPEG via heic2jpg.py), rotate to landscape (rule below wire), resize to 1649 px width.
 5. Segment wire (chroma vs median background), centreline per column, verify wire hue against the UID colour.
 6. Calibrate px->mm with template-matched 100/200/300/400 mm rule labels (1-D projective map along rule axis).
 7. Lateral deviation e(s) from the end-point chord over the common window 90-490 mm on the rule.
 8. Combine view A (e_y) and view B (e_z) on a common s-grid: r(s) = sqrt(e_y^2 + e_z^2).
"""
import argparse, os, re, sys, glob, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heic2jpg import convert as heic_convert

VIS_FRAC = 0.2
PAIR_GAP_S = 120
SESSION_GAP_MIN = 10
S_WIN = (90.0, 490.0)
W_STD = 1649
UID_COLOUR = {1: "orange", 2: "orange", 3: "orange", 4: "orange", 5: "blue", 6: "blue", 7: "blue", 8: "blue",
              9: "yellow", 10: "yellow", 11: "yellow", 12: "yellow", 17: "purple", 18: "purple", 19: "purple",
              20: "purple", 21: "red", 22: "red", 23: "red", 24: "red"}

# ---------------------------------------------------------------- EXIF time
def capture_time(path):
    with open(path, "rb") as f:
        head = f.read(4_000_000)
    m = re.search(rb"(20\d\d):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)", head)
    if not m:
        return pd.NaT
    return pd.Timestamp(*[int(x) for x in m.groups()])

# ---------------------------------------------------------------- matching
def sessions(runlog):
    r = runlog.sort_values("start_local").copy()
    key = list(zip(r.uid, r.instance, r.length))
    sid, prev, last_end, out = -1, None, None, []
    for k, st, en in zip(key, r.start_local, r.end_local):
        if k != prev or (st - last_end).total_seconds() / 60 > SESSION_GAP_MIN:
            sid += 1
        out.append(sid); prev, last_end = k, en
    r["session"] = out
    return (r.groupby("session").agg(uid=("uid", "first"), instance=("instance", "first"), section=("section", "first"),
                                    length=("length", "first"), tube=("tube", "max"), start=("start_local", "min"),
                                    end=("end_local", "max"), n_runs=("file", "size"), files=("file", lambda x: ";".join(x)))
              .reset_index())

MAX_LEAD_MIN = 60        # photos are taken shortly before the wire's first run
UNMATCHED_COST = 25.0    # min; a group is left unmatched rather than paired with a session further away than this
LENGTH_PENALTY = 60.0    # min; added when the camera-distance length class contradicts the session's wire length
COLOUR_PENALTY = 60.0    # min; added when the photo colour is incompatible with the session's wire colour

def colour_compatible(seen, expected):
    """Colour seen in the photo vs colour of the wire type. Red and purple insulation photograph dark/low-saturation,
    orange can read as red, so only clear contradictions count as incompatible."""
    seen = {c for c in seen if c}
    if not seen or expected is None: return True
    if expected in seen: return True
    ok = {"orange": {"red"}, "red": {"orange", "dark", "purple"}, "purple": {"dark", "blue", "red"},
          "blue": {"dark", "purple"}, "yellow": set()}
    return bool(seen & ok.get(expected, set()))

def match(groups, sess):
    """Globally optimal one-to-one assignment of photo groups to run sessions (Hungarian method).
    Cost = lead time (min) from photos to session start (session must start after the photos and within MAX_LEAD_MIN)
    + COLOUR_PENALTY if colours are incompatible; leaving a group unmatched costs UNMATCHED_COST."""
    from scipy.optimize import linear_sum_assignment
    G, Sn = len(groups), len(sess)
    BIG = 1e6
    C = np.full((G, Sn + G), BIG)
    for i, g in enumerate(groups.itertuples()):
        lead = (sess.start - g.t_first).dt.total_seconds().values / 60
        ok = (lead > 0) & (lead <= MAX_LEAD_MIN)
        col = np.array([0.0 if colour_compatible(g.colours, UID_COLOUR.get(int(u))) else COLOUR_PENALTY for u in sess.uid])
        # wire length from camera distance: 0.5 m wires are photographed close (no upscaling), longer wires from further away
        if g.len_class is not None:
            col = col + np.array([0.0 if (L <= 0.5) == (g.len_class == "short") else LENGTH_PENALTY for L in sess.length])
        C[i, :Sn] = np.where(ok, lead + col, BIG)
        C[i, Sn + i] = UNMATCHED_COST
    rr, cc = linear_sum_assignment(C)
    res = []
    for i, j in zip(rr, cc):
        g = groups.iloc[i]
        if j >= Sn or C[i, j] >= BIG:
            res.append(dict(group=g.group)); continue
        c = sess.iloc[j]
        res.append(dict(group=g.group, session=c.session, uid=c.uid, instance=c.instance, section=c.section,
                        length=c.length, tube=c.tube, run_start=c.start, lead_min=(c.start - g.t_first).total_seconds() / 60,
                        colour_check="ok" if C[i, j] - (c.start - g.t_first).total_seconds() / 60 < 1 else "MISMATCH",
                        n_runs=c.n_runs, files=c.files))
    return pd.DataFrame(res)

PROFILE_CACHE = {}

def group_colour(path, cache):
    """Wire colour of one photo from the full tracker (result cached and reused later)."""
    try:
        im = load_landscape(path, cache)
        p = profile_any(im); PROFILE_CACHE[path] = p
        return p["colour"] if p is not None else None
    except Exception:
        PROFILE_CACHE[path] = None
        return None

# ---------------------------------------------------------------- image
def load_landscape(path, cache):
    out = os.path.join(cache, os.path.splitext(os.path.basename(path))[0] + ".jpg")
    if not os.path.exists(out):
        if path.lower().endswith((".heic", ".heif")):
            heic_convert(path, out, 2700)
        else:
            cv2.imwrite(out, cv2.imread(path))
    im = cv2.imread(out)
    if im.shape[0] > im.shape[1]:
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    s = W_STD / im.shape[1]
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

def wire_mask(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    bg = cv2.medianBlur(lab, 31).astype(np.float32)
    d = lab.astype(np.float32) - bg
    score = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2) + 0.3 * np.abs(d[..., 0])
    m = (score > 18).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = im.shape[:2]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w > 0.5 * W and h < w / 6 and area < w * 25 and 0.2 * H < y + h / 2 < 0.85 * H:
            cands.append((float(chroma[cc == i].mean()), w, i))   # coloured wire beats grey edges
    if not cands:
        return None
    cands.sort(reverse=True)
    return cc == cands[0][2]

def centreline(mask):
    ys, xs = np.nonzero(mask)
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    cols, idx, cnt = np.unique(xs, return_index=True, return_counts=True)
    yc = np.add.reduceat(ys, idx) / cnt
    ok = cnt < 2.5 * np.median(cnt)
    cols, yc = cols[ok].astype(float), yc[ok]
    yc = pd.Series(yc).rolling(9, center=True, min_periods=1).median().values
    return cols, yc

def wire_hue(im, mask):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = mask & (hsv[..., 1] > 60)
    if m.sum() < 50:
        return np.nan, np.nan, "dark"
    h = hsv[..., 0][m].astype(float); sat = float(np.median(hsv[..., 1][m]))
    ang = np.angle(np.mean(np.exp(1j * h / 180 * 2 * np.pi))) / (2 * np.pi) * 180 % 180
    if ang < 7 or ang >= 165: c = "red"
    elif ang < 19: c = "orange"
    elif ang < 40: c = "yellow"
    elif ang < 130: c = "blue"
    else: c = "purple"
    return ang, sat, c

TEMPL = {}
def load_templates(path):
    ref = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    cent = {100: (517, 690), 200: (804, 690), 300: (1072, 690), 400: (1329, 690)}
    for v, (x, y) in cent.items():
        TEMPL[v] = ref[y - 8:y + 8, x - 15:x + 15]

def ruler_labels(g, ywire):
    H = g.shape[0]
    y0, y1 = int(max(0, ywire + 15)), int(min(H, ywire + 330))
    band = g[y0:y1]; rows = []
    for v, t in TEMPL.items():
        best = None
        for s in np.arange(0.75, 1.40, 0.025):
            ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            for ang in (-3.0, -1.5, 0.0, 1.5, 3.0):
                M = cv2.getRotationMatrix2D((ts.shape[1] / 2, ts.shape[0] / 2), ang, 1)
                tr = cv2.warpAffine(ts, M, (ts.shape[1], ts.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                r = cv2.matchTemplate(band, tr, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if best is None or mx > best[0]:
                    best = (mx, loc[0] + tr.shape[1] / 2, y0 + loc[1] + tr.shape[0] / 2)
        rows.append((v, *best))
    return pd.DataFrame(rows, columns=["value_mm", "score", "x_px", "y_px"])

def profile_old(im):
    mask = wire_mask(im)
    if mask is None:
        return None
    cols, yc = centreline(mask)
    hue, sat, cname = wire_hue(im, mask)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lab = ruler_labels(g, np.median(yc)).sort_values("value_mm")
    good = lab[lab.score > 0.6]
    lx, ly, lv = good.x_px.values, good.y_px.values, good.value_mm.values.astype(float)
    pr = np.polyfit(lx, ly, 1)
    ax = np.array([1.0, pr[0]]); ax /= np.linalg.norm(ax)
    O = np.array([lx[0], ly[0]])
    lpos = (np.stack([lx, ly], 1) - O) @ ax
    pp = np.linalg.lstsq(np.stack([lpos, np.ones_like(lpos), -lpos * lv], 1), lv, rcond=None)[0]
    to_mm = lambda q: (pp[0] * q + pp[1]) / (pp[2] * q + 1)
    cal_rms = float(np.sqrt(np.mean((to_mm(lpos) - lv) ** 2)))
    P = np.stack([cols, yc], 1) - O
    q = P @ ax; w = P @ np.array([-ax[1], ax[0]])
    s_mm = to_mm(q); k = to_mm(q + 0.5) - to_mm(q - 0.5); w_mm = w * k
    win = (s_mm >= S_WIN[0]) & (s_mm <= S_WIN[1])
    s_mm, w_mm = s_mm[win], w_mm[win]
    o = np.argsort(s_mm); s_mm, w_mm = s_mm[o], w_mm[o]
    nE = max(5, int(0.03 * len(s_mm)))
    P0 = np.array([s_mm[:nE].mean(), w_mm[:nE].mean()]); P1 = np.array([s_mm[-nE:].mean(), w_mm[-nE:].mean()])
    t = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-t[1], t[0]])
    rel = np.stack([s_mm, w_mm], 1) - P0
    u, e = rel @ t, rel @ nrm
    return dict(u=u - u.min(), e=e, s_rule=s_mm, cols=cols, yc=yc, labels=lab, cal_rms=cal_rms,
                n_labels=len(good), hue=hue, sat=sat, colour=cname, chord=float(np.linalg.norm(P1 - P0)),
                cover=float((s_mm.max() - s_mm.min()) / (S_WIN[1] - S_WIN[0])))

def _tpl(t, s, ang):
    ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((ts.shape[1] / 2, ts.shape[0] / 2), ang, 1)
    return cv2.warpAffine(ts, M, (ts.shape[1], ts.shape[0]), borderMode=cv2.BORDER_REPLICATE)

def _best(img, t, scales, angs, f, excl=()):
    best = None
    for s in scales:
        for ang in angs:
            tr = _tpl(t, s * f, ang)
            if tr.shape[0] >= img.shape[0] or tr.shape[1] >= img.shape[1]: continue
            r = cv2.matchTemplate(img, tr, cv2.TM_CCOEFF_NORMED)
            for ex in excl:                                  # already-claimed label positions (x in img coords)
                lo = int(max(0, ex - tr.shape[1] / 2 - 0.6 * f * 250)); hi = int(ex - tr.shape[1] / 2 + 0.6 * f * 250)
                r[:, lo:max(lo, hi)] = -1
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, loc[0] + tr.shape[1] / 2, loc[1] + tr.shape[0] / 2, s, ang)
    return best

def ruler_labels_band(g, y0, y1, K=10):
    """Joint coarse-to-fine matching of the bold 100/200/300/400 mm labels (~1 s per photo).
    Per template: max response over scale/angle (half resolution) -> top-K peaks; then an ordered chain
    (spacing 180-340 px, |dy| < 45 px) maximising the summed score; finally full-resolution refinement."""
    band = g[y0:y1]; f = 0.5
    small = cv2.resize(band, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    hh, ww = small.shape
    scales = np.arange(0.75, 1.40, 0.05); angs = (-4.0, -2.0, 0.0, 2.0, 4.0)
    cand = {}
    for v in (100, 200, 300, 400):
        R = np.full((hh, ww), -1, np.float32); Ss = np.zeros((hh, ww), np.float32); Aa = np.zeros((hh, ww), np.float32)
        for sc in scales:
            for ang in angs:
                tr = _tpl(TEMPL[v], sc * f, ang)
                if tr.shape[0] >= hh or tr.shape[1] >= ww: continue
                r = cv2.matchTemplate(small, tr, cv2.TM_CCOEFF_NORMED)
                oy, ox = tr.shape[0] // 2, tr.shape[1] // 2
                sl = (slice(oy, oy + r.shape[0]), slice(ox, ox + r.shape[1]))
                upd = r > R[sl]
                R[sl][upd] = r[upd]; Ss[sl][upd] = sc; Aa[sl][upd] = ang
        mx = cv2.dilate(R, np.ones((31, 31), np.uint8))
        ys, xs = np.where((R >= mx) & (R > 0.45))
        order = np.argsort(-R[ys, xs])[:K]
        cand[v] = [(float(R[ys[i], xs[i]]), xs[i] / f, ys[i] / f, float(Ss[ys[i], xs[i]]), float(Aa[ys[i], xs[i]])) for i in order]
    # DP over ordered labels
    vs = (100, 200, 300, 400)
    if any(len(cand[v]) == 0 for v in vs):
        return pd.DataFrame([(v, 0.0, np.nan, np.nan) for v in vs], columns=["value_mm", "score", "x_px", "y_px"])
    best = {i: (c[0], [i]) for i, c in enumerate(cand[100])}
    for v_prev, v in zip(vs[:-1], vs[1:]):
        nxt = {}
        for j, cj in enumerate(cand[v]):
            opts = []
            for i, (tot, path) in best.items():
                ci = cand[v_prev][path[-1]] if False else cand[v_prev][i if len(path) == 1 else path[-1]]
                dx = cj[1] - ci[1]; dy = cj[2] - ci[2]
                if 180 <= dx <= 340 and abs(dy) < 45:
                    opts.append((tot + cj[0], path + [j]))
            if opts: nxt[j] = max(opts, key=lambda o: o[0])
        if not nxt:
            return pd.DataFrame([(v_, 0.0, np.nan, np.nan) for v_ in vs], columns=["value_mm", "score", "x_px", "y_px"])
        best = nxt
    # best maps last-label index -> (total, path of indices); re-key for next iteration handled via path[-1]
    tot, path = max(best.values(), key=lambda o: o[0])
    rows = []
    for v, k in zip(vs, path):
        sc0, cx, cy, s_, a_ = cand[v][k]
        x0, y0w = int(max(0, cx - 60)), int(max(0, cy - 40))
        win = band[y0w:int(cy + 40), x0:int(cx + 60)]
        r = _best(win, TEMPL[v], s_ + np.array([-0.025, 0.0, 0.025]), (a_ - 1, a_, a_ + 1), 1.0)
        if r is None: r = (sc0, cx - x0, cy - y0w, s_, a_)
        rows.append((v, r[0], x0 + r[1], y0 + y0w + r[2]))
    return pd.DataFrame(rows, columns=["value_mm", "score", "x_px", "y_px"])

CLAMP_OFFSET_MM = 15.0   # start of the evaluated free span below the clamp tape (user: 1-2 cm offset)
TIP_MARGIN_MM = 10.0     # cut at the free tip (ferrule region)
RIDGE_GAIN = 1.0
TIPS = {}
CUR_TIP = {}                    # hand-labelled tip for the photo being processed: {'col', 'w'}
TIP_MARGIN_COLS = 10            # stop this many strip columns before the labelled tip (wire side, clear of the ferrule)
FERRULE_V, FERRULE_S = 90, 90   # HSV value / saturation below which a pixel is dark metal (ferrule)
FERRULE_MIN_COLS, FERRULE_MAX_COLS = 6, 40   # ferrule length in strip columns (~2-15 mm)
CLAMP_SEARCH_COLS = 250   # clamp tape must end within this many columns of the wire's first visible column
FLOOR_PX = 4             # path rows this close to the lower search bound are treated as 'no wire'
SPIKE_PX = 6             # lateral excursion (px) from the 181-column running median treated as an outlier
MAX_STEP_PX = 1          # max lateral wire movement per column in the gated re-track (wire slope is small)
BRIDGE_FRAC = 0.15       # weak-score gaps up to this fraction of the wire length are bridged by the reference curve
RULE_BAND_MM = 6.0       # how far below the rule top edge the wire search reaches
GATE_MM = 4.0       # half width of the gating band around the robust reference curve
def robust_ref(path, ok, deg=7, iters=6, k=3.0, tail=120):
    """Huber-like iteratively reweighted Legendre fit of the tracked path over its confident columns.
    Beyond the supported range the curve continues LINEARLY with the end slope (perspective makes the far wire drift in
    rule-aligned coordinates). Returns (curve, dist) with dist = pixels beyond the support, or None if too little support."""
    from numpy.polynomial import legendre as Lg
    idx = np.where(ok)[0]
    if len(idx) < 150: return None
    a, b = idx.min(), idx.max()
    n = len(path); cols = np.arange(n)
    x = (np.clip(cols, a, b) - a) / max(b - a, 1) * 2 - 1
    w = ok.astype(float)
    for _ in range(iters):
        c = Lg.legfit(x[ok], path[ok], deg, w=w[ok])
        r = path - Lg.legval(x, c)
        sc = 1.4826 * np.median(np.abs(r[w > 0])) + 1.0
        w = ok * (np.abs(r) < k * sc)
        if w.sum() < 100: return None
    cur = Lg.legval(x, c)
    for edge, sl in ((a, slice(a, a + tail)), (b, slice(max(a, b - tail), b + 1))):
        cc = np.arange(n)[sl]
        if len(cc) < 10: continue
        m = float(np.clip(np.polyfit(cc, cur[sl], 1)[0], -0.12, 0.12))
        if edge == a: cur[:a] = cur[a] + m * (cols[:a] - a)
        else: cur[b + 1:] = cur[b] + m * (cols[b + 1:] - b)
    dist = np.maximum(a - cols, 0) + np.maximum(cols - b, 0)
    return cur, dist

def find_tip(strip, hsv, path, vis, cL, hi, back=90, fwd=30):
    """Column of the end ferrule (start of the metal block) or None. The continuous visible track starting at cL defines
    where the wire ends; a dark metal run (short, on the track) within [end-back, end+fwd] is the ferrule."""
    r = runs(vis)
    r = [x for x in r if x[1] - x[0] > 0]
    if not r: return None
    cont = [x for x in r if x[0] <= cL + 120 or x[1] >= cL]
    end = max((x[1] for x in r if x[0] >= cL - 5 and x[1] - x[0] > 30), default=None)
    if end is None: return None
    ys = np.arange(strip.shape[0])[:, None]
    metal = (hsv[..., 2] < FERRULE_V) & (hsv[..., 1] < FERRULE_S) & (np.abs(ys - path[None, :]) <= 6)
    cnt = metal.sum(0)
    cand = [(a_, b_) for a_, b_ in runs(cnt >= 3) if FERRULE_MIN_COLS <= b_ - a_ <= FERRULE_MAX_COLS
            and end - back <= a_ <= end + fwd]
    return cand[0][0] - 2 if cand else None

def viterbi(score, max_step=3, lam=0.15):
    """Max-score path through score[h, q] (one row per column), |dh| <= max_step per column."""
    Hh, Q = score.shape
    acc = score[:, 0].copy(); back = np.zeros((Hh, Q), np.int8)
    steps = np.arange(-max_step, max_step + 1)
    for c in range(1, Q):
        cand = np.full((len(steps), Hh), -1e9)
        for i, d in enumerate(steps):
            sh = np.roll(acc, d)
            if d > 0: sh[:d] = -1e9
            elif d < 0: sh[d:] = -1e9
            cand[i] = sh - lam * abs(d)
        k = np.argmax(cand, 0); acc = cand[k, np.arange(Hh)] + score[:, c]; back[:, c] = steps[k]
    path = np.zeros(Q, int); path[-1] = int(np.argmax(acc))
    for c in range(Q - 1, 0, -1):
        path[c - 1] = path[c] - back[path[c], c]
    return path

def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

DBG = {}
def profile(im, margin_mm=4.0):
    H, W = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 1) rule
    y0, y1 = (0, H) if H < 500 else (int(0.30 * H), int(0.92 * H))
    lab = ruler_labels_band(g, y0, y1).sort_values("value_mm")
    good = lab[lab.score > 0.6]
    if len(good) < 3:
        DBG["why"] = "rule labels"; return None
    lx, ly, lv = good.x_px.values, good.y_px.values, good.value_mm.values.astype(float)
    pr = np.polyfit(lx, ly, 1)
    ax = np.array([1.0, pr[0]]); ax /= np.linalg.norm(ax)
    up = np.array([ax[1], -ax[0]])                       # perpendicular pointing up (towards wire)
    O = np.array([lx[0], ly[0]])
    lpos = (np.stack([lx, ly], 1) - O) @ ax
    pp = np.linalg.lstsq(np.stack([lpos, np.ones_like(lpos), -lpos * lv], 1), lv, rcond=None)[0]
    to_mm = lambda q: (pp[0] * q + pp[1]) / (pp[2] * q + 1)
    cal_rms = float(np.sqrt(np.mean((to_mm(lpos) - lv) ** 2)))
    # strip origin is moved DOWN_MM below the label line, because wires can lie across the rule
    mpp0 = float(np.median(np.abs(to_mm(lpos[-1]) - to_mm(lpos[0])) / max(1e-6, abs(lpos[-1] - lpos[0]))))
    D = int(round(DOWN_MM / mpp0))
    O = O - D * up                                       # perpendicular shift: along-rule coordinate q unchanged
    # 2) resample strip above the rule: rows h = 0..HMAX px above the (shifted) origin, columns q along rule
    qs = np.arange(-O[0] - 5, W - O[0] + 5, 1.0)
    HMAX = int(min(420 * max(1.0, W / W_STD) + D, O[1] - 2))  # 420 px at the standard scale (~155 mm) + shift
    hs = np.arange(0, HMAX, 1.0)
    QQ, HH = np.meshgrid(qs, hs)
    mapx = (O[0] + QQ * ax[0] + HH * up[0]).astype(np.float32)
    mapy = (O[1] + QQ * ax[1] + HH * up[1]).astype(np.float32)
    strip = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.remap(np.ones((H, W), np.uint8), mapx, mapy, cv2.INTER_NEAREST, borderValue=0) > 0
    sg = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 3) board top edge: strongest step (band means 4..14 px apart), searched >= 55 px above the label line
    a_ = np.array([sg[max(0, r - 14):max(1, r - 4)].mean(0) if r >= 14 else sg[0] for r in range(len(hs))])
    b_ = np.array([sg[r + 4:r + 14].mean(0) if r + 14 < len(hs) else sg[-1] for r in range(len(hs))])
    step = np.median(np.abs(a_ - b_), axis=1)
    # scale-independent geometry (mm -> px): steel-rule top edge = strongest step 5..25 mm above the label line;
    # the wire is searched only above it (dark wires otherwise lock onto the rule graduations)
    mpp0 = float(np.median(np.abs(to_mm(lpos[-1]) - to_mm(lpos[0])) / max(1e-6, abs(lpos[-1] - lpos[0]))))
    px = lambda mm: D + int(round(mm / mpp0))          # row index of a height (mm) above the label line
    # measured on well-tracked photos of both set-ups: rule top 6-9 mm, board top 45-55 mm above the label line.
    # Search windows are restricted to these ranges so that a thick dark wire's own edge is not taken as an edge.
    lo_, hi_ = px(4), min(len(step) - 1, px(12))
    rule_top = lo_ + int(np.argmax(step[lo_:hi_])) if hi_ > lo_ else 12
    b0, b1 = px(38), min(len(step) - 1, px(62))
    btop = b0 + int(np.argmax(step[b0:b1])) if b1 > b0 else HMAX - 1
    # 4) wire score inside board strip (rows 12 .. btop-5)
    lab_s = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.medianBlur(cv2.cvtColor(strip, cv2.COLOR_BGR2LAB), 31).astype(np.float32)
    d = lab_s - bg
    chroma = np.hypot(d[..., 1], d[..., 2])
    score = chroma + 0.5 * np.abs(d[..., 0])
    # thin-wire ridge term: dark line across the (rule-aligned, ~horizontal) wire, d2I/dy2 at two scales; the rule graduations
    # are vertical lines and give no response. Scaled to the colour score so that thin low-chroma wires (yellow on cream) stand out.
    ridge = np.zeros_like(sg)
    for sig in (max(1.0, 0.5 / mpp0), max(1.8, 1.1 / mpp0)):
        gb = cv2.GaussianBlur(sg, (0, 0), sig)
        ridge = np.maximum(ridge, sig ** 2 * cv2.Sobel(gb, cv2.CV_32F, 0, 2, ksize=3) / 8.0)
    ridge = np.maximum(ridge, 0)
    ridge_s = RIDGE_GAIN * ridge * (np.percentile(score[valid], 99) / max(np.percentile(ridge[valid], 99), 1e-6))
    score = score + ridge_s
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    # wires can lie across the steel rule: the rule band is searched too, but its score is halved so that the
    # graduations do not attract dark wires; the band right at the board top edge is excluded
    r_rule = rule_top + px(1.5) - D                     # first row above the steel rule
    r1 = max(rule_top + px(5) - D, btop - px(8) + D)
    r0 = max(2, rule_top - int(round(RULE_BAND_MM / mpp0)))   # search extends down onto the rule (rule_top already includes D)
    board_sat = float(np.median(hsv[r_rule:r1, ..., 1][valid[r_rule:r1]]))   # cream vs white boards differ
    tape = ((hsv[..., 0] >= 12) & (hsv[..., 0] <= 34) & (hsv[..., 1] >= board_sat + 22)
            & (hsv[..., 1] <= 170) & (hsv[..., 2] >= 110))
    tape[:r_rule] = False                                # wire colours on the rule must not be mistaken for tape
    score_r = score.copy()
    rb = slice(0, r_rule)                                # rule band: only colour counts (grey rule, black graduations)
    score_r[rb] = 0.8 * chroma[rb] + ridge_s[rb]
    for lx_, ly_ in zip(lx, ly):                         # printed red label discs would attract red/orange wires
        qd = (np.array([lx_, ly_]) - O) @ ax
        score_r[max(0, D - 16):D + px(4) - D + 14 + D // 4, max(0, int(qd - qs[0]) - 16):int(qd - qs[0]) + 16] = 0
    S = score_r[r0:r1].copy(); S[~valid[r0:r1]] = 0
    S = np.minimum(S, np.percentile(S[max(0, r_rule - r0):], 99.7))
    S[:max(0, r_rule - r0)] *= 0.6
    path = viterbi(S) + r0
    ps = score_r[path, np.arange(len(qs))]
    ps_s = pd.Series(ps).rolling(15, center=True, min_periods=1).median().values
    centre_lvl = np.median(ps_s[len(qs) // 4: 3 * len(qs) // 4])
    vis = ps_s > VIS_FRAC * centre_lvl
    tape_frac = tape[r0:r1].mean(0)
    is_tape = pd.Series(tape_frac > 0.15).rolling(5, center=True, min_periods=1).max().values.astype(bool)
    # --- gated re-tracking: a wire is smooth, so fit a robust low-order curve to the confident part of the track and
    # track again inside a narrow band around it. This stops jumps onto the rule edge / board edge / graduations.
    band_px = max(4, int(round(GATE_MM / mpp0)))
    for _ in range(5):
        strict = (ps_s > 0.45 * centre_lvl) & ~is_tape
        gr = robust_ref(path.astype(float), strict)
        if gr is None: break
        gref, gdist = gr
        rows_ = np.arange(r1 - r0)[:, None] + r0
        Sg = np.where(np.abs(rows_ - gref[None, :]) <= (band_px + 0.025 * gdist)[None, :], S, -10.0)
        path = viterbi(Sg, max_step=MAX_STEP_PX) + r0
        ps = score_r[path, np.arange(len(qs))]
        ps_s = pd.Series(ps).rolling(15, center=True, min_periods=1).median().values
        centre_lvl = np.median(ps_s[len(qs) // 4: 3 * len(qs) // 4])
        vis = ps_s > VIS_FRAC * centre_lvl
        # columns where the best path lies outside the gate have no wire support: pinned to the band edge, not visible
        in_gate = Sg[np.clip(path - r0, 0, Sg.shape[0] - 1), np.arange(len(qs))] > -9.0
        vis = vis & in_gate & (path - r0 > FLOOR_PX)      # pinned to the search floor = no wire
    # outlier suppression: a physical wire is smooth over ~180 columns (~6 cm); a narrow excursion that deviates from the
    # running median by more than SPIKE_PX is a feature of the set-up (tape, board edge, graduation), not the wire
    pm = pd.Series(path.astype(float)).rolling(181, center=True, min_periods=1).median().values
    spike = np.abs(path - pm) > SPIKE_PX
    path = np.where(spike, np.round(pm), path).astype(path.dtype)
    # visible extent = longest run of visible, non-tape columns plus neighbours merged across tapes
    vr = [r for r in runs(vis & ~is_tape) if r[1] - r[0] > 30]
    # the tape strip at the far end of the board is only a marker, not a clamp: the wire continues past it as a free tip.
    # Extend the visible extent over a tape-width gap to the tip, if wire is seen again behind the strip.
    if vr:
        hi0 = max(r[1] for r in vr)
        for a_, b_ in runs(vis & ~is_tape):
            if b_ - a_ >= 6 and 0 < a_ - hi0 <= 120 and is_tape[hi0:a_].mean() > 0.5:
                vr.append((a_, b_)); hi0 = b_
    DBG.update(mpp0=mpp0, rule_top=rule_top, strip=strip, path=path, vis=vis, is_tape=is_tape, btop=btop, ps_s=ps_s, centre_lvl=centre_lvl, r0=r0)
    if not vr:
        DBG["why"] = "no visible run"; return None
    lo, hi = min(r[0] for r in vr), max(r[1] for r in vr)
    # bridge short weak-score gaps (wire dipping onto the rule, low contrast stretches) with the smooth reference curve
    vis_raw = vis.copy()
    gr = robust_ref(path.astype(float), (ps_s > 0.45 * centre_lvl) & ~is_tape)
    if gr is not None:
        gref = gr[0]
        for a_, b_ in runs(~vis[lo:hi]):
            if b_ - a_ <= BRIDGE_FRAC * (hi - lo) and a_ > 0 and b_ < hi - lo:
                # straight line between the two ends of the gap (a polynomial overshoots across long gaps)
                x0, x1 = lo + a_ - 1, lo + b_
                y0, y1 = path[x0], path[x1]
                xs_ = np.arange(x0 + 1, x1)
                vis[x0 + 1:x1] = True
                path[x0 + 1:x1] = np.round(y0 + (y1 - y0) * (xs_ - x0) / (x1 - x0)).astype(path.dtype)
    cmid = (lo + hi) // 2
    tr_ = [r for r in runs(is_tape) if r[1] - r[0] >= 12]
    # ONE clamp (tape at the top end, the tape nearest the wire start); the far end is the free tip. Tapes further along
    # are markers on the rule or cross over the wire and must not cut the span.
    left = [r[1] for r in tr_ if lo - 40 < r[1] <= lo + CLAMP_SEARCH_COLS]
    right = []
    cL = max(left) if left else lo
    cR = hi
    clamp_src = ("tape" if left else "wire-end") + "/free-tip"
    # end ferrule: the wire ends in a short dark metal block (ferrule). Search only where the visible track ends, so that
    # rule graduations and grit far from the wire are not mistaken for it.
    tip = None
    if CUR_TIP and CUR_TIP.get("w") == len(qs):          # hand-labelled tip (wire side) for this photo at this scale
        tip = CUR_TIP["col"] - TIP_MARGIN_COLS
        cR = min(cR, tip)
        clamp_src += "/labelled"
    else:
        tip = find_tip(strip, hsv, path, vis & ~is_tape, cL, hi)
        if tip is not None:
            cR = min(cR, tip)
            clamp_src += "/ferrule"
    DBG.update(tip=tip)
    q_mm = to_mm(qs)
    mpp = np.gradient(q_mm)                                # mm per px along q
    sL, sR = q_mm[cL] + margin_mm, q_mm[cR - 1] - margin_mm
    clamped = (sL, sR)
    sL, sR = max(sL, S_WIN[0]), min(sR, S_WIN[1])      # common 90-490 mm window (same as first PDF set)
    sel = (q_mm >= sL) & (q_mm <= sR) & vis & ~is_tape
    DBG.update(cL=cL, cR=cR, sL=sL, sR=sR, sel=sel)
    if sel.sum() < 100:
        DBG["why"] = f"sel {sel.sum()}"; return None
    s_mm = q_mm[sel]; w_mm = path[sel] * mpp[sel]          # isotropic local scale
    # sub-pixel refinement: score-weighted centroid +-4 px around the path
    ref = []
    for qi, hp in zip(np.where(sel)[0], path[sel]):
        lo_, hi_ = max(0, hp - 4), min(len(hs), hp + 5)
        wgt = score_r[lo_:hi_, qi]
        # no score mass around the path (e.g. a faint column): keep the tracked row instead of a centroid at row 0
        ref.append((np.arange(lo_, hi_) * wgt).sum() / wgt.sum() if wgt.sum() > 1.0 else float(hp))
    w_mm = np.array(ref) * mpp[sel]
    w_mm = pd.Series(w_mm).rolling(7, center=True, min_periods=1).median().values
    nE = max(5, int(0.03 * len(s_mm)))
    P0 = np.array([s_mm[:nE].mean(), w_mm[:nE].mean()]); P1 = np.array([s_mm[-nE:].mean(), w_mm[-nE:].mean()])
    t = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-t[1], t[0]])
    rel = np.stack([s_mm, w_mm], 1) - P0
    u, e = rel @ t, rel @ nrm
    # full free span between the clamps (needed for 1.0 m wires; beyond the 500 mm rule the scale is EXTRAPOLATED)
    # outermost tapes within the wire extent (wire passes under intermediate tapes, e.g. the rule-end tape at ~490 mm)
    # Only the top end is clamped (one tape). The free span starts CLAMP_OFFSET_MM below the clamp tape and ends at the
    # free wire tip (minus the metal ferrule region).
    clamped = (q_mm[cL] + CLAMP_OFFSET_MM, q_mm[min(cR, len(qs)) - 1] - TIP_MARGIN_MM)
    sel_f = (q_mm >= clamped[0]) & (q_mm <= clamped[1]) & vis & ~is_tape
    # stop where the track jumps off the wire (e.g. onto the converging board edge far from the rule):
    # walking outwards from the 90-490 mm window, cut at the first lateral jump > 3 mm within 10 columns
    pth = pd.Series(path.astype(float)).rolling(5, center=True, min_periods=1).median().values
    jump = np.zeros(len(qs), bool); jump[10:] = np.abs(pth[10:] - pth[:-10]) > px(4.0) - D
    iw = np.where(sel)[0]
    if len(iw):
        right_j = np.where(jump & (np.arange(len(qs)) > iw.max()))[0]
        left_j = np.where(np.roll(jump, -10) & (np.arange(len(qs)) < iw.min()))[0]
        if len(right_j): sel_f[right_j[0] - 10:] = False
        if len(left_j): sel_f[:left_j[-1] + 10] = False
        if sel_f.any():
            clamped = (max(clamped[0], float(q_mm[np.where(sel_f)[0][0]])), min(clamped[1], float(q_mm[np.where(sel_f)[0][-1]])))
    full = None
    if sel_f.sum() > 100:
        sf = q_mm[sel_f]; wf = pd.Series(path[sel_f] * mpp[sel_f]).rolling(7, center=True, min_periods=1).median().values
        nF = max(5, int(0.03 * len(sf)))
        F0 = np.array([sf[:nF].mean(), wf[:nF].mean()]); F1 = np.array([sf[-nF:].mean(), wf[-nF:].mean()])
        tf = (F1 - F0) / np.linalg.norm(F1 - F0); nf = np.array([-tf[1], tf[0]])
        rf = np.stack([sf, wf], 1) - F0
        full = dict(s=sf, u=rf @ tf - (rf @ tf).min(), e=rf @ nf, sL=float(clamped[0]), sR=float(clamped[1]),
                    img_pts=O + np.outer(qs[sel_f], ax) + np.outer(path[sel_f].astype(float), up),
                    gap=float(1 - sel_f.sum() / max(1, ((q_mm >= clamped[0]) & (q_mm <= clamped[1])).sum())))
    # colour along the visible path
    hp_ = path[sel]; px = strip[hp_, np.where(sel)[0]]
    hv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    sat = float(np.median(hv[:, 1]))
    if sat < 60: cname, hue = "dark", np.nan
    else:
        hue = float(np.angle(np.mean(np.exp(1j * hv[:, 0] / 180 * 2 * np.pi))) / (2 * np.pi) * 180 % 180)
        cname = ("red" if (hue < 7 or hue >= 165) else "orange" if hue < 19 else "yellow" if hue < 40
                 else "blue" if hue < 130 else "purple")
    # image coordinates for QC
    qq = qs[sel]; img_pts = O + np.outer(qq, ax) + np.outer(np.array(ref), up)
    clamp_pts = [O + qs[c] * ax + 30 * up for c in (cL, min(cR, len(qs) - 1))]
    tape_pts = [(O + qs[a0] * ax + r0 * up, O + qs[a1 - 1] * ax + r0 * up) for a0, a1 in tr_]
    btop_line = (O + qs[0] * ax + btop * up, O + qs[-1] * ax + btop * up)
    bridged = float((vis & ~vis_raw)[sel].mean()) if sel.any() else np.nan
    return dict(bridged_frac=bridged, mpp0=mpp0, u=u - u.min(), s=s_mm, e=e, labels=lab, cal_rms=cal_rms, n_labels=len(good), hue=hue, sat=sat, colour=cname,
                chord=float(np.linalg.norm(P1 - P0)), span_mm=float(sR - sL), sL=float(sL), sR=float(sR),
                clamp_src=clamp_src, tape_span=f"{clamped[0]:.0f}-{clamped[1]:.0f}", gap_frac=float(1 - sel.sum() / max(1, ((q_mm >= sL) & (q_mm <= sR)).sum())),
                img_pts=img_pts, clamp_pts=clamp_pts, tape_pts=tape_pts, btop_line=btop_line, full=full)

DOWN_MM = 18.0       # strip offset below the label line (12 mm tested: rule graduations then attract dark wires; kept at 0)
UPSCALE = (1.0, 1.75, 1.5, 2.0)     # photos taken from further away (1.0 m wire set-up): rule appears smaller
def profile_any(im):
    """Run profile() at the standard scale; if the rule labels are not found, retry on an upscaled image
    so that the 100 mm label spacing falls into the template range again. Returns the result with the image used."""
    best = None
    for f in UPSCALE:
        imf = im if f == 1.0 else cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        try:
            p = profile(imf)
        except Exception:
            p = None
        if p is not None:
            p["im"], p["upscale"] = imf, f
            if p["n_labels"] >= 4 and p["cal_rms"] < 1.0 and p["span_mm"] > 0.97 * (S_WIN[1] - S_WIN[0]):
                return p
            key = lambda d: (d["cal_rms"] < 1.0, d["n_labels"], d["span_mm"])
            if best is None or key(p) > key(best):
                best = p
    return best

def plane_metrics(u, e):
    xi = np.clip(u / u.max(), 0, 1)
    B = np.stack([np.sin(np.pi * xi), np.sin(2 * np.pi * xi), np.sin(3 * np.pi * xi)], 1)
    a = np.linalg.lstsq(B, e, rcond=None)[0]
    return dict(dmax=float(np.abs(e).max()), dpp=float(e.max() - e.min()), erms=float(np.sqrt(np.mean(e ** 2))),
                A1=float(a[0]), A2=float(a[1]), A3=float(a[2]), wavy_rms=float(np.sqrt(np.mean((e - B @ a) ** 2))))

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True); ap.add_argument("--runlog", required=True)
    ap.add_argument("--template", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tips", default=None, help="tip_index.csv with hand-labelled tip columns (photo, strip_w, tip_col)")
    ap.add_argument("--imgrange", nargs=2, type=int, default=None, help="only IMG_<a>..IMG_<b> (excludes straightener / HMI-screen / floor photos)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); cache = os.path.join(a.out, "_jpg"); os.makedirs(cache, exist_ok=True)
    qc = os.path.join(a.out, "qc"); os.makedirs(qc, exist_ok=True)
    load_templates(a.template)

    files = sorted(f for f in glob.glob(os.path.join(a.photos, "*")) if f.lower().endswith((".heic", ".heif", ".jpg", ".jpeg")))
    TIPS.clear()
    if a.tips:
        tdf = pd.read_csv(a.tips, dtype=str).fillna("")
        for t in tdf.itertuples():
            if t.tip_col.strip().isdigit() and t.strip_w.strip():
                TIPS[t.photo] = dict(col=int(t.tip_col), w=int(float(t.strip_w)))
        print("labelled tips loaded:", len(TIPS))
    if a.imgrange:
        files = [f for f in files if (m_ := re.search(r"IMG_(\d+)", os.path.basename(f))) and a.imgrange[0] <= int(m_.group(1)) <= a.imgrange[1]]
    ph = pd.DataFrame(dict(path=files))
    ph["photo"] = [re.sub(r"^[0-9a-f]{8}-", "", os.path.splitext(os.path.basename(f))[0]) for f in files]
    ph = ph.drop_duplicates("photo")
    ph["t"] = [capture_time(f) for f in ph.path]
    ph = ph.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    gap = ph.t.diff().dt.total_seconds().fillna(1e9)
    ph["group"] = (gap > PAIR_GAP_S).cumsum()
    ph["view"] = ph.groupby("group").cumcount().map(lambda i: "ABCDEFG"[i])
    groups = ph.groupby("group").agg(t_first=("t", "min"), n=("photo", "size"), paths=("path", list)).reset_index()
    groups["colours"] = [[group_colour(pth, cache) for pth in ps] for ps in groups.paths]
    def _lc(ps):
        up = [PROFILE_CACHE[q]["upscale"] for q in ps if PROFILE_CACHE.get(q) is not None]
        return None if not up else ("short" if np.median(up) <= 1.0 else "long")
    groups["len_class"] = [_lc(ps) for ps in groups.paths]
    groups["colour"] = ["/".join(sorted({c for c in cs if c})) or None for cs in groups.colours]

    rl = pd.read_csv(a.runlog, parse_dates=["start_local", "end_local"])
    m = match(groups, sessions(rl))
    ph = ph.merge(m, on="group", how="left")
    um = groups[~groups.group.isin(m.dropna(subset=["uid"]).group)]
    print(f"{len(ph)} photos, {len(groups)} groups, {len(groups) - len(um)} matched to a run session, {len(um)} unmatched")
    for g in um.itertuples():
        print(f"  UNMATCHED group {g.group}: {g.t_first}  colour={g.colour}  n_photos={g.n}")
    if (groups.n != 2).any():
        print("  note: groups with != 2 photos:", groups[groups.n != 2][["group", "t_first", "n"]].values.tolist())

    rows, curves, curves_full = [], [], []
    for r in ph.itertuples():
        im = load_landscape(r.path, cache)
        CUR_TIP.clear()
        if a.tips and r.photo in TIPS:
            CUR_TIP.update(TIPS[r.photo])
        if r.path in PROFILE_CACHE and not CUR_TIP:
            p = PROFILE_CACHE[r.path]
        else:
            try:
                p = profile_any(im)
            except Exception as ex:
                print(f"{r.photo}: {ex}"); p = None
        if p is None:
            rows.append(dict(photo=r.photo, status="no wire found")); continue
        exp = UID_COLOUR.get(int(r.uid)) if pd.notna(r.uid) else None
        ok_col = colour_compatible({p["colour"]}, exp)
        met = plane_metrics(p["u"], p["e"])
        fm = {}
        if p.get("full") is not None:
            ff = p["full"]; mf = plane_metrics(ff["u"], ff["e"])
            fm = dict(full_s_from_mm=ff["sL"], full_s_to_mm=ff["sR"], full_span_mm=ff["sR"] - ff["sL"],
                      full_extrapolated=ff["sR"] > 510, full_gap_frac=ff["gap"],
                      **{f"full_{k}_mm": v for k, v in mf.items()})
        ok = ok_col and p["n_labels"] >= 3 and p["cal_rms"] < 1.0 and p["span_mm"] > 0.97 * (S_WIN[1] - S_WIN[0]) and p["gap_frac"] < 0.1
        rows.append(dict(photo=r.photo, time=r.t, group=r.group, view=r.view, uid=r.uid, instance=r.instance,
                         section=r.section, length=r.length, tube=r.tube, run_start=r.run_start, lead_min=r.lead_min,
                         colour_detected=p["colour"], colour_expected=exp, colour_ok=ok_col, hue=p["hue"],
                         n_labels=p["n_labels"], cal_rms_mm=p["cal_rms"], clamp=p["clamp_src"], tape_span_mm=p["tape_span"], span_rule_mm=f"{p['sL']:.0f}-{p['sR']:.0f}",
                         span_mm=p["span_mm"], gap_frac=p["gap_frac"], bridged_frac=p["bridged_frac"],
                         **{f"{k}_mm" if k != "wavy_rms" else "wavy_rms_mm": v for k, v in met.items()},
                         dmax_over_L=met["dmax"] / p["span_mm"], upscale=p["upscale"], **fm, status="ok" if ok else "CHECK"))
        curves.append(pd.DataFrame(dict(photo=r.photo, group=r.group, view=r.view, u_mm=p["u"], s_mm=p["s"], e_mm=p["e"])))
        if p.get("full") is not None:
            curves_full.append(pd.DataFrame(dict(photo=r.photo, group=r.group, view=r.view, length=r.length,
                                                 u_mm=p["full"]["u"], s_mm=p["full"]["s"], e_mm=p["full"]["e"])))
        vis = p["im"].copy()
        if p.get("full") is not None:      # whole free span between the outer clamps (beyond 500 mm: scale extrapolated)
            for (x, y), sv in zip(p["full"]["img_pts"].astype(int), p["full"]["s"]):
                cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0) if sv <= 500 else (0, 255, 160), -1)
        for x, y in p["img_pts"].astype(int):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        for L in p["labels"].itertuples():
            cv2.circle(vis, (int(L.x_px), int(L.y_px)), 7, (0, 0, 255) if L.score > 0.6 else (0, 255, 255), 2)
        for a0, a1 in p["tape_pts"]:
            cv2.line(vis, tuple(a0.astype(int)), tuple(a1.astype(int)), (0, 200, 255), 4)
        for c in p["clamp_pts"]:
            cv2.drawMarker(vis, tuple(c.astype(int)), (255, 0, 255), cv2.MARKER_TRIANGLE_DOWN, 22, 3)
        b0, b1 = p["btop_line"]; cv2.line(vis, tuple(b0.astype(int)), tuple(b1.astype(int)), (255, 128, 0), 1)
        cv2.putText(vis, f"{r.photo} {r.view}  UID{int(r.uid) if pd.notna(r.uid) else '?'} inst{int(r.instance) if pd.notna(r.instance) else '?'}  {p['colour']}  span {p['sL']:.0f}-{p['sR']:.0f} mm",
                    (20, 45), 0, 1.1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(qc, f"qc_{r.photo}.jpg"), vis)

    M = pd.DataFrame(rows); C = pd.concat(curves)
    M.to_csv(os.path.join(a.out, "photo_metrics.csv"), index=False)
    C.to_csv(os.path.join(a.out, "photo_centrelines.csv"), index=False)

    # ---- combine views A/B per group (2-plane)
    two = []
    for gid, g in C.groupby("group"):
        views = sorted(g.view.unique())
        if len(views) < 2: continue
        ga, gb = g[g.view == views[0]], g[g.view == views[1]]
        lo_, hi_ = max(ga.s_mm.min(), gb.s_mm.min()), min(ga.s_mm.max(), gb.s_mm.max())     # common ruler range
        sg = np.linspace(lo_, hi_, 400)
        ga, gb = ga.sort_values("s_mm"), gb.sort_values("s_mm")
        ey = np.interp(sg, ga.s_mm, ga.e_mm); ez = np.interp(sg, gb.s_mm, gb.e_mm)
        r_ = np.hypot(ey, ez)
        info = M[(M.group == gid) & (M.view == views[0])].iloc[0]
        two.append(dict(group=gid, photos=f"{ga.photo.iloc[0]}+{gb.photo.iloc[0]}", uid=info.uid, instance=info.instance,
                        section=info.section, length=info.length, tube=info.tube, run_start=info.run_start,
                        dmax_y_mm=np.abs(ey).max(), dmax_z_mm=np.abs(ez).max(), rmax_mm=r_.max(), s_from_mm=lo_, s_to_mm=hi_, status=('ok' if (M[M.group == gid].status == 'ok').all() else 'CHECK'),
                        rrms_mm=np.sqrt(np.mean(r_ ** 2)), s_at_rmax_mm=sg[np.argmax(r_)],
                        plane_angle_deg=np.degrees(np.arctan2(ez[np.argmax(r_)], ey[np.argmax(r_)]))))
        pd.DataFrame(dict(group=gid, s_mm=sg, e_y_mm=ey, e_z_mm=ez, r_mm=r_)).to_csv(
            os.path.join(a.out, f"twoplane_group{gid}.csv"), index=False)
    T = pd.DataFrame(two); T.to_csv(os.path.join(a.out, "twoplane_metrics.csv"), index=False)

    # ---- full free span (1.0 m wires: whole wire between the outer tapes; scale extrapolated beyond the 500 mm rule)
    CF = pd.concat(curves_full) if curves_full else pd.DataFrame()
    CF.to_csv(os.path.join(a.out, "photo_centrelines_fullspan.csv"), index=False)
    twof = []
    for gid, g in (CF.groupby("group") if len(CF) else []):
        views = sorted(g.view.unique())
        if len(views) < 2: continue
        ga, gb = g[g.view == views[0]].sort_values("s_mm"), g[g.view == views[1]].sort_values("s_mm")
        lo_, hi_ = max(ga.s_mm.min(), gb.s_mm.min()), min(ga.s_mm.max(), gb.s_mm.max())
        if hi_ - lo_ < 100: continue
        sg = np.linspace(lo_, hi_, 800)
        # re-reference both views to the chord of the common range
        def rechord(df):
            e = np.interp(sg, df.s_mm, df.e_mm); return e - np.interp(sg, [sg[0], sg[-1]], [e[0], e[-1]])
        ey, ez = rechord(ga), rechord(gb); r_ = np.hypot(ey, ez)
        info = M[(M.group == gid) & (M.view == views[0])].iloc[0]
        twof.append(dict(group=gid, photos=f"{ga.photo.iloc[0]}+{gb.photo.iloc[0]}", uid=info.uid, instance=info.instance,
                         section=info.section, length=info.length, tube=info.tube, s_from_mm=lo_, s_to_mm=hi_,
                         span_mm=hi_ - lo_, extrapolated=hi_ > 510, dmax_y_mm=np.abs(ey).max(), dmax_z_mm=np.abs(ez).max(),
                         rmax_mm=r_.max(), rrms_mm=np.sqrt(np.mean(r_ ** 2)), s_at_rmax_mm=sg[np.argmax(r_)],
                         status=('ok' if (M[M.group == gid].status == 'ok').all() else 'CHECK')))
        pd.DataFrame(dict(group=gid, s_mm=sg, e_y_mm=ey, e_z_mm=ez, r_mm=r_)).to_csv(
            os.path.join(a.out, f"twoplane_full_group{gid}.csv"), index=False)
    pd.DataFrame(twof).to_csv(os.path.join(a.out, "twoplane_fullspan_metrics.csv"), index=False)
    pd.set_option("display.width", 250)
    print(M[["photo", "time", "view", "uid", "instance", "tube", "lead_min", "colour_detected", "colour_ok",
             "n_labels", "cal_rms_mm", "clamp", "span_rule_mm", "gap_frac", "dmax_mm", "A1_mm", "status"]].round(2).to_string(index=False))
    print(); print(T.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
