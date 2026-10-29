"""Run inventory: number of run files per tube, per wire UID and first/last run per tube.

Reads the raw run files in Thesis_Data (WFE_<UID>_<instance>_<YYYYMMDD>_<HHMM>.csv).
The tube diameter is read from the header line TUBE_diameter.
Output: output/run_inventory.csv and printed summary.
"""
import csv
import re
from collections import Counter, defaultdict
from wfe_common import DATA_DIR, OUT_DIR

NAME = re.compile(r"^WFE_(\d+)_\d+_(\d{8})_(\d{4})")


def header_value(path, key):
    with open(path, "rb") as fh:
        for line in fh:
            text = line.decode("utf-8", errors="replace").replace("\r", "").replace(";", "")
            if text.startswith(key + ":"):
                return text.split(":", 1)[1].strip()
    return None


runs = []
for f in sorted(DATA_DIR.glob("WFE_*.csv")):
    m = NAME.match(f.name)
    if not m:
        continue
    uid, date, hhmm = int(m.group(1)), m.group(2), m.group(3)
    dia = header_value(f, "TUBE_diameter")
    runs.append({"file": f.name, "uid": uid, "tube": dia,
                 "stamp": f"{date}{hhmm}"})

with open(OUT_DIR / "run_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "uid", "tube", "stamp"])
    w.writeheader()
    w.writerows(runs)

print("run files:", len(runs))
print("per tube:", Counter(r["tube"] for r in runs))
by_tube = defaultdict(list)
for r in runs:
    by_tube[r["tube"]].append(r["stamp"])
for tube, stamps in sorted(by_tube.items()):
    print(f"tube {tube}: first {min(stamps)}  last {max(stamps)}  n={len(stamps)}")
per_uid = Counter((r["uid"], r["tube"]) for r in runs)
low = sorted(per_uid.items(), key=lambda kv: kv[1])[:5]
print("lowest run counts (uid, tube): n:", low)
