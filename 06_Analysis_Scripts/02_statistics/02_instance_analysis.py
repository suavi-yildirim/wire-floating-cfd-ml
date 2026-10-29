"""Instance-based differences in stable pressure and flow.

For every wire type (UID) and tube with at least two instances, a Kruskal-Wallis
test compares the stable pressure (and flow) between instances. Counts how many
combinations are significant at p < 0.05. Only controller-stable runs are used.
Output: output/instance_kruskal_wallis.csv and printed summary.
"""
import csv
from collections import defaultdict
from scipy import stats
from wfe_common import load_stable_rows, tube_of, uid_of, to_float, P, F, OUT_DIR

rows = load_stable_rows()
groups = defaultdict(lambda: defaultdict(lambda: {"p": [], "q": []}))
for r in rows:
    key = (tube_of(r), uid_of(r))
    inst = r["WIRE_instance"]
    p, q = to_float(r[P]), to_float(r[F])
    if p is not None:
        groups[key][inst]["p"].append(p)
    if q is not None:
        groups[key][inst]["q"].append(q)

out = []
for (tube, uid), insts in sorted(groups.items()):
    if len(insts) < 2:
        continue
    pres = [v["p"] for v in insts.values() if len(v["p"]) > 0]
    flow = [v["q"] for v in insts.values() if len(v["q"]) > 0]
    if len(pres) < 2:
        continue
    hp = stats.kruskal(*pres).pvalue
    hq = stats.kruskal(*flow).pvalue if len(flow) >= 2 else float("nan")
    out.append({"tube": tube, "uid": uid, "n_instances": len(insts),
                "p_pressure": hp, "p_flow": hq})

with open(OUT_DIR / "instance_kruskal_wallis.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

for tube in (8, 10):
    sub = [o for o in out if o["tube"] == tube]
    sig = sum(o["p_pressure"] < 0.05 for o in sub)
    print(f"WC{tube:02d}: {sig}/{len(sub)} wire types with significant instance differences (pressure)")
