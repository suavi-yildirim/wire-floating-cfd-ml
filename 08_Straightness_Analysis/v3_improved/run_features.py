"""Per-run response features from raw WFE logs (Thesis_Data/WFE_*.csv).
Responses: p_stable (duration-weighted), p_lift (pressure when wire first leaves position 0), t_lift, t_stable (start of the
accepted stable phase, s from experiment start), n_incr, n_decr, p_overshoot (max Pressure_IS before settling), p_rms_settle.
Usage: python run_features.py <raw dir> <out csv>"""
import sys, glob, os, re, io
import numpy as np, pandas as pd

def num(s): return float(str(s).replace(",", "."))
def ts(s):
    s = s.strip().rstrip(";")
    return pd.to_datetime(s.replace(",", "."), format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")

def parse(f):
    L = open(f, encoding="latin-1").read().splitlines()
    h, sec, cur = {}, {}, None
    for i, l in enumerate(L):
        m = re.match(r"^([A-Za-z_0-9]+):\s*(.*?);?\s*$", l)
        if m and cur is None: h[m.group(1)] = m.group(2)
        if re.match(r"^(STABLE PHASES|INCREASES|DECREASES|ALL SENSOR CHANGES|SYSTEM FLUID RESPONSE)", l): sec[l.split(" - ")[0].split(" count")[0].strip()] = i; cur = l
    return L, h, sec

def block(L, start, stop):
    rows = [l for l in L[start + 2:stop] if l.strip() and ";" in l]
    return [r.rstrip(";").split(";") for r in rows]

def features(f):
    L, h, sec = parse(f)
    try:
        t0 = ts(h["Experiment_Start_Time"].split(",")[0] + "," + (h["Experiment_Start_Time"].split(",")[1] if "," in h["Experiment_Start_Time"] else "0"))
    except Exception: t0 = pd.NaT
    keys = sorted(sec.items(), key=lambda kv: kv[1]); idx = {k: v for k, v in keys}
    nxt = lambda k: min([v for _, v in keys if v > idx[k]] + [len(L)])
    out = dict(file=os.path.basename(f), uid=int(h["WIRE_uid"]), instance=int(h["WIRE_instance"]), section=num(h["WIRE_section"]),
               length=num(h["WIRE_lenght"]), tube=int(h["TUBE_diameter"]), target_pos=num(h.get("Target_Position", "nan")),
               gain=num(h.get("PID_Compact_WFE_Retain_CtrlParams_Gain", "nan")), ti=num(h.get("PID_Compact_WFE_Retain_CtrlParams_Ti", "nan")))
    st = None
    for l in L:
        if l.startswith("stable:"): out["stable"] = "True" in l
        if l.startswith("Experiment_Duration:"): out["duration_ms"] = num(l.split(":")[1].strip(" ;"))
        if l.startswith("increases:"): out["n_incr"] = int(l.split(":")[1])
        if l.startswith("decreases:"): out["n_decr"] = int(l.split(":")[1])
    # stable phases
    if "STABLE PHASES" in idx:
        ph = block(L, idx["STABLE PHASES"], nxt("STABLE PHASES"))
        P = []
        for r in ph:
            try: P.append(dict(start=ts(r[1]), dur=num(r[3]), pmean=num(r[7]), q=num(r[10])))
            except Exception: pass
        P = [p for p in P if p["dur"] >= 1000]
        if P:
            w = np.array([p["dur"] for p in P]); out["p_stable"] = np.average([p["pmean"] for p in P], weights=w)
            out["q_stable"] = np.average([p["q"] for p in P], weights=w)
            last = max(P, key=lambda p: p["dur"])
            if pd.notna(t0) and pd.notna(last["start"]): out["t_stable_s"] = (last["start"] - t0).total_seconds()
            out["n_phases"] = len(P)
    # time series
    if "SYSTEM FLUID RESPONSE" in idx:
        a = idx["SYSTEM FLUID RESPONSE"]; per = float(re.search(r"period:(\d+)", L[a]).group(1)) / 1000
        d = pd.read_csv(io.StringIO("\n".join(L[a + 2:])), sep=";", header=None, decimal=",", usecols=range(5),
                        names=["pset", "pis", "q", "temp", "pos"], on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
        t = np.arange(len(d)) * per
        run = d.pset.values > 0
        mv = np.where(d.pos.values > 0)[0]
        if len(mv):
            i = mv[0]; out["t_lift_s"] = t[i]; out["p_lift"] = float(d.pis.values[i]); out["q_lift"] = float(d.q.values[i])
            out["t_lift_to_stable_s"] = out.get("t_stable_s", np.nan) - t[i] if "t_stable_s" in out else np.nan
        out["t_total_s"] = t[-1]
        if "t_stable_s" in out and np.isfinite(out["t_stable_s"]):
            k0 = int(out["t_stable_s"] / per); seg = d.pis.values[max(k0, 0):]
            pre = d.pis.values[:max(k0, 1)]
            out["p_peak_before_stable"] = float(pre.max()); out["p_overshoot"] = float(pre.max() - seg.mean()) if len(seg) else np.nan
            out["p_sd_settled"] = float(seg.std()) if len(seg) else np.nan
        out["frac_pos0"] = float((d.pos.values == 0).mean())
    return out

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "WFE_*.csv"))):
        try: rows.append(features(f))
        except Exception as e: print("FAIL", f, repr(e))
    D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False); print(D.shape); print(D.describe().T.round(2).to_string())
