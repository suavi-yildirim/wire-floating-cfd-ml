"""Build a run timeline (local CET) from raw WFE headers.
Experiment_Start/End_Time are UTC (Date: field = local file time, ~1 h later)."""
import sys, glob, os, re, pandas as pd
from datetime import timedelta
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "WFE_*.csv"))):
    h = {}
    with open(f, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            if i > 120: break
            m = re.match(r"([A-Za-z_]+):(.*?);?\s*$", line)
            if m: h[m.group(1)] = m.group(2)
    st = pd.to_datetime(h["Experiment_Start_Time"].split(",")[0], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    en = pd.to_datetime(h["Experiment_End_Time"].split(",")[0], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    dt = pd.to_datetime(h["Date"], format="%Y%m%d_%H%M%S")
    if pd.isna(en) or en.year < 2000:                       # placeholder end time -> use file time
        en = dt - timedelta(hours=1)
    if pd.isna(st) or st.year < 2000:
        st = en
    off = round((dt - en).total_seconds() / 3600)          # hours UTC->local
    rows.append(dict(file=os.path.basename(f), start_local=st + timedelta(hours=off), end_local=en + timedelta(hours=off),
                     utc_offset_h=off, uid=int(h["WIRE_uid"]), instance=int(h["WIRE_instance"]),
                     section=float(h["WIRE_section"].replace(",", ".")), length=float(h["WIRE_lenght"].replace(",", ".")),
                     tube=int(h["TUBE_diameter"])))
d = pd.DataFrame(rows).sort_values("start_local")
d.to_csv(sys.argv[2], index=False)
print(d.utc_offset_h.value_counts().to_string())
print(d.drop(columns=["utc_offset_h"]).to_string(index=False))
