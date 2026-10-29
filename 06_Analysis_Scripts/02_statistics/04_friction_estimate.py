"""Exploratory friction estimate from the rising-minus-dropping difference.

1. Per instance (tube, UID, instance) that was tested in both directions:
   d = median(rises) - median(drops) in mbar. Wilcoxon test over instances.
2. Calibration per tube: median stable pressure of each wire type against the
   wire weight from CFD_Experimental_Outcome.xlsx; linear fit, slope in mbar/N.
3. Half-band friction force: F_f = (median d / 2) / slope, in N, and as a share
   of the mean wire weight of the wire types in that tube.
Caveats are in the thesis (Section 7.4.6). Output: output/friction_estimate.json
and output/friction_method.png (figure used in Section 7.4.6).
"""
import json
import statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from wfe_common import (load_stable_rows, to_float, tube_of, uid_of, P, OUT_DIR,
                        load_cfd_weights)

rows = load_stable_rows()
weights = load_cfd_weights()

# 1. per-instance paired differences
inst = defaultdict(lambda: {"wire_rises": [], "wire_drops": []})
pres_by_uid = defaultdict(lambda: defaultdict(list))
for r in rows:
    tube, uid = tube_of(r), uid_of(r)
    p = to_float(r[P])
    if p is None:
        continue
    pres_by_uid[tube][uid].append(p)
    if r["Final_Trend"] in ("wire_rises", "wire_drops"):
        inst[(tube, uid, r["WIRE_instance"])][r["Final_Trend"]].append(p)

result = {}
for tube in (8, 10):
    diffs = [st.median(d["wire_rises"]) - st.median(d["wire_drops"])
             for (t, _, _), d in inst.items()
             if t == tube and d["wire_rises"] and d["wire_drops"]]
    wil = stats.wilcoxon(diffs)
    # 2. calibration over wire types of this tube
    W, M, labels = [], [], []
    for uid, vals in pres_by_uid[tube].items():
        if (uid, tube) in weights:
            W.append(weights[(uid, tube)])
            M.append(st.median(vals))
            labels.append(uid)
    lr = stats.linregress(W, M)
    med = st.median(diffs)
    f_half = (med / 2) / lr.slope                       # N
    mean_w = float(np.mean(W))
    result[tube] = {
        "n_instances": len(diffs),
        "median_diff_mbar": med,
        "iqr_mbar": [float(np.percentile(diffs, 25)), float(np.percentile(diffs, 75))],
        "n_positive": sum(x > 0 for x in diffs),
        "wilcoxon_p": float(wil.pvalue),
        "n_wire_types_calibration": len(W),
        "slope_mbar_per_N": lr.slope,
        "R2": lr.rvalue ** 2,
        "friction_half_band_N": f_half,
        "mean_weight_N": mean_w,
        "friction_share_of_weight": f_half / mean_w,
        "_diffs": diffs, "_W": W, "_M": M, "_labels": labels,
        "_intercept": lr.intercept,
    }
    print(f"WC{tube:02d}: n={len(diffs)} median d={med:.0f} mbar, positive {result[tube]['n_positive']}, "
          f"p={wil.pvalue:.2g}; slope {lr.slope:.0f} mbar/N, R2={lr.rvalue**2:.2f}; "
          f"F_f={f_half:.3f} N ({f_half/mean_w*100:.0f}% of mean weight {mean_w:.3f} N)")

# Figure: calibration (top) and per-instance differences (bottom)
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(2, 2, figsize=(7, 6.2))
for col, tube in enumerate((8, 10)):
    r = result[tube]
    ax = axes[0, col]
    ax.scatter(r["_W"], r["_M"], s=18, color="#1f5f8b")
    xx = np.linspace(min(r["_W"]), max(r["_W"]), 10)
    ax.plot(xx, r["_intercept"] + r["slope_mbar_per_N"] * xx, color="#9a9a9a", lw=1)
    ax.set_title(f"WC{tube:02d}: calibration, R$^2$={r['R2']:.2f}", fontsize=9)
    ax.set_xlabel("Wire weight (N)")
    if col == 0:
        ax.set_ylabel("Median stable pressure (mbar)")
    ax.text(0.04, 0.93, f"slope {r['slope_mbar_per_N']:.0f} mbar/N", transform=ax.transAxes, fontsize=8)
    ax2 = axes[1, col]
    jitter = np.random.default_rng(1).normal(0, 0.05, len(r["_diffs"]))
    ax2.scatter(jitter, r["_diffs"], s=16, color="#555555", alpha=0.8)
    ax2.hlines(r["median_diff_mbar"], -0.2, 0.2, color="#d9822b", lw=2)
    ax2.axhline(0, color="#cccccc", lw=0.8)
    ax2.set_xlim(-0.3, 0.3)
    ax2.set_xticks([])
    ax2.set_title(f"WC{tube:02d}: rising minus dropping, per instance (n={len(r['_diffs'])})", fontsize=9)
    if col == 0:
        ax2.set_ylabel("Pressure difference (mbar)")
    ax2.text(0.22, r["median_diff_mbar"], f"median {r['median_diff_mbar']:.0f}", fontsize=8, va="center")
fig.tight_layout()
fig.savefig(OUT_DIR / "friction_method.png", dpi=130)

clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in result.items()}
with open(OUT_DIR / "friction_estimate.json", "w", encoding="utf-8") as fh:
    json.dump(clean, fh, indent=1)
print("saved", OUT_DIR / "friction_method.png")
