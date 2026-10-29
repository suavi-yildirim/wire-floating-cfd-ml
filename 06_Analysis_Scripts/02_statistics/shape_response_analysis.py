"""Shape features (curvature based, from the two-view centrelines) vs run-response features, within wire type.
Usage: python shape_response_analysis.py <pipeline out dir> <run_features.csv> <runlog.csv> <outdir>
Shape features per photo group (window 90-490 mm, both views, QC ok):
  rmax, rrms      max / rms resultant chord deviation (as in v2)
  kap_rms, kap_max, kap_mean   curvature magnitude kappa = sqrt(eA''^2 + eB''^2) (1/m) from a smooth degree-8 Legendre fit
  bend_E          bending energy  integral kappa^2 ds  (1/m)
  kap_cv          kappa_rms / kappa_mean (localised vs uniform bending)
  n_infl          sign changes of the smoothed curvature vector projection (S-shape count)
  sag             sagitta of the view with larger chord deviation
Responses per photographed session: p_stable, p_lift, t_lift, t_stable, t_lift_to_stable, n_incr, p_overshoot, p_sd_settled,
  lift_ratio = p_lift / p_stable.   Within-type centring: per (uid,length,tube) group, values -> within-group z score.
Test: pooled Spearman of centred values, permutation p (shuffling within type), Benjamini-Hochberg over all tests."""
import sys, os
import numpy as np, pandas as pd
from numpy.polynomial import legendre as Lg
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp

OUT, RF, RL, DST = sys.argv[1:5]; os.makedirs(DST, exist_ok=True)
rng = np.random.default_rng(1)

def feats(sg, ey, ez, deg=8):
    x = 2 * (sg - sg[0]) / (sg[-1] - sg[0]) - 1; Lm = (sg[-1] - sg[0]) / 1000
    out = {}
    d2 = []
    for e in (ey, ez):
        c = Lg.legfit(x, e, deg); d = Lg.legder(c, 2)
        d2.append(Lg.legval(x, d) * (2 / (sg[-1] - sg[0])) ** 2 * 1e6 / 1e3)   # mm/mm^2 -> 1/m   (e in mm, s in mm -> 1/mm -> *1000)
    ky, kz = d2; kap = np.hypot(ky, kz); ds = (sg[1] - sg[0]) / 1000
    out.update(kap_rms=np.sqrt(np.mean(kap ** 2)), kap_max=kap.max(), kap_mean=kap.mean(), bend_E=np.sum(kap ** 2) * ds,
               kap_cv=np.sqrt(np.mean(kap ** 2)) / max(kap.mean(), 1e-9))
    p = ky * np.sign(np.mean(ky) or 1)
    m = np.abs(ky) > 0.1 * np.abs(ky).max()
    out["n_infl"] = int(np.sum(np.diff(np.sign(ky[m])) != 0)) + int(np.sum(np.diff(np.sign(kz[np.abs(kz) > 0.1 * np.abs(kz).max()])) != 0))
    out["sag"] = max(np.abs(ey).max(), np.abs(ez).max())
    return out

M = pd.read_csv(os.path.join(OUT, "twoplane_metrics.csv"))
rl = pd.read_csv(RL, parse_dates=["start_local", "end_local"]); S = wp.sessions(rl)
RFt = pd.read_csv(RF)
resp = ["p_stable", "p_lift", "t_lift_s", "t_stable_s", "t_lift_to_stable_s", "n_incr", "p_overshoot", "p_sd_settled", "lift_ratio"]
RFt["lift_ratio"] = RFt.p_lift / RFt.p_stable
rows = []
for t in M.itertuples():
    if pd.isna(t.uid): continue
    fn = os.path.join(OUT, f"twoplane_group{t.group}.csv")
    if not os.path.exists(fn): fn = os.path.join(OUT, "twoplane", f"twoplane_group{t.group}.csv")
    c = pd.read_csv(fn)
    d = dict(group=t.group, uid=int(t.uid), instance=int(t.instance), length=t.length, tube=t.tube, section=t.section, status=t.status_final,
             rmax=t.rmax_mm, rrms=t.rrms_mm, **feats(c.s_mm.values, c.e_y_mm.values, c.e_z_mm.values))
    s = S[(S.uid == t.uid) & (S.instance == t.instance) & (S.start == pd.Timestamp(t.run_start))]
    if len(s):
        f = RFt[RFt.file.isin(s.iloc[0].files.split(";"))]; d["n_runs"] = len(f)
        for r in resp: d[r] = f[r].median() if r.startswith(("t_", "n_")) else f[r].mean()
    rows.append(d)
D = pd.DataFrame(rows); D.to_csv(os.path.join(DST, "shape_response_table.csv"), index=False)
D = D[D.status == "ok"].copy(); key = ["uid", "length", "tube"]
D["ntype"] = D.groupby(key).uid.transform("size"); D = D[D.ntype >= 3]
shape = ["rmax", "rrms", "kap_rms", "kap_max", "kap_mean", "bend_E", "kap_cv", "n_infl", "sag"]
print("n wires:", len(D), " types:", D.groupby(key).ngroups)
def cz(df, c):
    return df.groupby(key)[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) if v.std(ddof=0) > 0 else 1))
res = []
for r in resp:
    sub = D.dropna(subset=[r]); 
    for sh in shape:
        a, b = cz(sub, sh).values, cz(sub, r).values
        rho = stats.spearmanr(a, b)[0]
        g = sub.groupby(key).ngroup().values; perm = []
        for _ in range(5000):
            bb = b.copy()
            for k in np.unique(g): i = np.where(g == k)[0]; bb[i] = rng.permutation(b[i])
            perm.append(stats.spearmanr(a, bb)[0])
        p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / 5001
        res.append(dict(shape=sh, response=r, n=len(sub), rho=rho, p_perm=p))
R = pd.DataFrame(res).sort_values("p_perm")
m = len(R); R["q_bh"] = np.minimum.accumulate((R.p_perm * m / np.arange(1, m + 1))[::-1])[::-1].clip(upper=1)
R.to_csv(os.path.join(DST, "shape_response_stats.csv"), index=False)
pd.set_option("display.width", 200); print(R.round(3).head(25).to_string(index=False))
print(R.pivot(index="shape", columns="response", values="rho").round(2).to_string())
