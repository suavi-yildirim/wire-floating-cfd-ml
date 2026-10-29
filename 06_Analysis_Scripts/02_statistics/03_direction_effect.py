"""Within-wire effect of the approach direction (rising vs dropping).

For each wire type (UID) and tube, the median stable pressure of the runs that
enter the target window from below (wire_rises) and from above (wire_drops) is
computed. Only wire types tested in both directions are used. A paired Wilcoxon
test compares the two medians across wire types; the pooled flow rates per
direction are compared with a Mann-Whitney test.
Output: output/direction_pairs.json, output/direction_within_wire.csv and
output/direction_effect_within_wire.pdf (figure used in Section 7.4.6).
"""
import csv
import json
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from wfe_common import load_stable_rows, direction_groups, to_float, P, F, OUT_DIR

rows = load_stable_rows()
groups = direction_groups(rows)

pairs = {}
table = []
for tube in (8, 10):
    items = []
    for (t, uid), d in groups.items():
        if t != tube or not d["wire_rises"] or not d["wire_drops"]:
            continue
        mr, md = st.median(d["wire_rises"]), st.median(d["wire_drops"])
        items.append((uid, md, mr))
        table.append({"tube": tube, "uid": uid, "median_drops": md, "median_rises": mr,
                      "diff_rises_minus_drops": mr - md,
                      "n_drops": len(d["wire_drops"]), "n_rises": len(d["wire_rises"])})
    pairs[tube] = items
    diffs = [mr - md for _, md, mr in items]
    pos = sum(x > 0 for x in diffs)
    w = stats.wilcoxon([mr for _, _, mr in items], [md for _, md, _ in items])
    print(f"WC{tube:02d}: {len(items)} wire types, {pos} with rises > drops, "
          f"median difference {st.median(diffs):.0f} mbar, Wilcoxon p={w.pvalue:.2g}")

# Pooled flow per direction (all stable runs with a known direction)
for tube in (8, 10):
    fr, fd = [], []
    for r in rows:
        if int(float(r["TUBE_diameter (mm)"])) != tube:
            continue
        q = to_float(r[F])
        if q is None:
            continue
        if r["Final_Trend"] == "wire_rises":
            fr.append(q)
        elif r["Final_Trend"] == "wire_drops":
            fd.append(q)
    mw = stats.mannwhitneyu(fr, fd)
    print(f"WC{tube:02d} pooled flow: rises {st.median(fr):.1f} vs drops {st.median(fd):.1f} "
          f"Nm3/h, Mann-Whitney p={mw.pvalue:.2g}")

with open(OUT_DIR / "direction_pairs.json", "w", encoding="utf-8") as fh:
    json.dump({str(k): v for k, v in pairs.items()}, fh, indent=1)
with open(OUT_DIR / "direction_within_wire.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

# Slope chart: one line per wire type, drops (orange) to rises (blue)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.8), sharey=False)
for ax, tube in zip(axes, (8, 10)):
    for uid, md, mr in pairs[tube]:
        ax.plot([0, 1], [md, mr], color="#999999", lw=0.8, zorder=1)
    ax.scatter([0] * len(pairs[tube]), [md for _, md, _ in pairs[tube]], color="#d9822b", s=16, zorder=2)
    ax.scatter([1] * len(pairs[tube]), [mr for _, _, mr in pairs[tube]], color="#1f5f8b", s=16, zorder=2)
    ax.set_xticks([0, 1], ["Enters from above", "Enters from below"])
    ax.set_title(f"WC{tube:02d} (n={len(pairs[tube])} wire types)", fontsize=9)
    ax.set_ylabel("Median stable pressure (mbar)")
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "direction_effect_within_wire.pdf")
print("saved figure and tables to", OUT_DIR)
