"""
Raw-data loaders for the Wire Floating Experiment (WFE) thesis.

Formats handled
---------------
* WFE raw logs   (01_WFE_Raw/WFE_<uid>_<p>_<date>_<time>.csv)  - PLC export, ';' separated,
                  German decimal comma, sectioned by upper-case headers.
* TFA raw logs   (03_TFA_Raw/TFA_<date>_<time>.csv)             - tube-flow-analysis ramps.
* Fluent reports (02_Report_Data/WCxx_UIDyy/*.out)               - iteration histories.

Unit conventions found in the raw data (cross-checked against the CFD database):
    pressure   : mbar (gauge)          -> Pa  = mbar * 100
    flow       : 0.01 Nm^3/h counts     -> Nm^3/h = raw / 100
    temperature: 0.1 degC counts        -> degC  = raw / 10
    position   : m (sensor-array coordinate)
    sample period of the fluid response: 50 ms (stated in the header)
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Data root = the folder that contains 01_WFE_Raw, 02_Report_Data, 03_TFA_Raw, 06_CFD_Processed.
# Set it with the environment variable WFE_DATA, e.g.  WFE_DATA=~/Desktop/Thesis_Data python3 make_figures.py
ROOT = Path(os.environ.get("WFE_DATA", Path(__file__).resolve().parent.parent / "data")).expanduser()


def _dir(name: str) -> Path:
    """Accept both  ROOT/name/*.csv  and the zip-extracted  ROOT/name/name/*.csv."""
    nested = ROOT / name / name
    return nested if nested.is_dir() else ROOT / name


WFE_DIR = _dir("01_WFE_Raw")
TFA_DIR = _dir("03_TFA_Raw")
REP_DIR = _dir("02_Report_Data")
CFD_XLSX = _dir("06_CFD_Processed") / "Master_CFD_Database.xlsx"
CFD_OUTCOME = _dir("06_CFD_Processed") / "CFD_Experimental_Outcome.csv"

FLOW_OVERFLOW = 30000  # raw counts; the flow meter saturates at 32760 (int16) - anything above is invalid
Q_FULLSCALE = 65.0    # Nm^3/h - flow-meter full-scale; partially saturated hold averages lie above

# Wire catalogue: UID -> (cross-section mm^2, length m)
SECTION_OF_BLOCK = {0: 0.5, 1: 1.0, 2: 1.5, 4: 4.0, 5: 6.0}
LENGTHS = [0.5, 1.0, 2.0, 3.5]


def uid_to_geom(uid: int) -> tuple[float, float]:
    block, k = divmod(uid - 1, 4)
    return SECTION_OF_BLOCK[block], LENGTHS[k]


def _num(s: str):
    s = str(s).strip().rstrip(";").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return s


# --------------------------------------------------------------------------------------
# WFE
# --------------------------------------------------------------------------------------
_SECTION_HDRS = ("SENSOR INFORMATION", "SYSTEM RESPONSE", "STABLE PHASES", "INCREASES",
                 "DECREASES", "ALL SENSOR CHANGES", "SYSTEM FLUID RESPONSE")


@dataclass
class WFERun:
    file: str
    meta: dict = field(default_factory=dict)
    stable: pd.DataFrame | None = None
    increases: pd.DataFrame | None = None
    decreases: pd.DataFrame | None = None
    sensors: pd.DataFrame | None = None
    ts: pd.DataFrame | None = None
    period_ms: float = 50.0


def _table(lines: list[str]) -> pd.DataFrame | None:
    lines = [l for l in lines if l.strip()]
    if len(lines) < 2:
        return None
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=";", decimal=",", engine="python")


def parse_wfe(path: Path, with_ts: bool = True) -> WFERun:
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    run = WFERun(file=path.name)
    blocks: dict[str, list[str]] = {}
    cur = None
    for raw in txt:
        line = raw.strip()
        if line == "eof":
            break
        hdr = next((h for h in _SECTION_HDRS if line.startswith(h)), None)
        if hdr:
            cur = hdr
            blocks[cur] = []
            m = re.search(r"period:\s*(\d+)", line)
            if m:
                run.period_ms = float(m.group(1))
            continue
        if cur is None or cur == "SYSTEM RESPONSE":
            m = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
            if m:
                run.meta[m.group(1)] = _num(m.group(2))
            if cur is None:
                continue
        blocks[cur].append(line)

    run.stable = _table(blocks.get("STABLE PHASES", []))
    run.increases = _table(blocks.get("INCREASES", []))
    run.decreases = _table(blocks.get("DECREASES", []))
    run.sensors = _table(blocks.get("ALL SENSOR CHANGES", []))
    if with_ts and "SYSTEM FLUID RESPONSE" in blocks:
        ts = _table(blocks["SYSTEM FLUID RESPONSE"])
        if ts is not None:
            ts = ts.apply(pd.to_numeric, errors="coerce")
            ts["t_s"] = np.arange(len(ts)) * run.period_ms / 1000.0
            ts["p_mbar"] = ts["Pressure_IS"]
            ts["p_set_mbar"] = ts["Pressure_SET"]
            ts["Q_Nm3h"] = ts["Flow_IS"] / 100.0
            ts["T_C"] = ts["Temperature_IS"] / 10.0
            ts["pos_m"] = ts["Position_WIRE"]
            run.ts = ts
    return run


def wfe_tube_diameter(meta: dict) -> float | None:
    d = meta.get("TUBE_diameter")
    return float(d) if isinstance(d, (int, float)) else None


def wfe_run_table(min_phase_ms: float = 1000.0) -> pd.DataFrame:
    """One row per raw WFE run with the duration-weighted stable-phase averages
    (same rule as the author's process_data.py: phases >= 1 s, duration-weighted)."""
    rows = []
    for f in sorted(WFE_DIR.glob("WFE_*.csv")):
        r = parse_wfe(f, with_ts=False)
        m = r.meta
        uid = int(m.get("WIRE_uid", -1))
        row = dict(file=f.name, uid=uid, instance=m.get("WIRE_instance"),
                   section=m.get("WIRE_section"), length=m.get("WIRE_lenght"),
                   D=wfe_tube_diameter(m), stable=str(m.get("stable", "")).strip() == "True",
                   duration_ms=m.get("Experiment_Duration"), n_inc=m.get("increases"),
                   n_dec=m.get("decreases"), pid_gain=m.get("PID_Compact_WFE_Retain_CtrlParams_Gain"),
                   p_start=m.get("Pressure_Start"))
        st = r.stable
        if st is not None and "Duration" in st:
            st = st.copy()
            for c in ["Duration", "Pressure_Stable_Average", "Flow_Stable_Average", "Temp_Stable_Average",
                      "Pressure_Stable_Min", "Pressure_Stable_Max"]:
                st[c] = pd.to_numeric(st[c], errors="coerce")
            st = st[st["Duration"] >= min_phase_ms]
            if len(st):
                w = st["Duration"]
                row.update(
                    p_mbar=(st["Pressure_Stable_Average"] * w).sum() / w.sum(),
                    Q_Nm3h=(st["Flow_Stable_Average"] * w).sum() / w.sum() / 100,
                    T_C=(st["Temp_Stable_Average"] * w).sum() / w.sum() / 10,
                    stable_ms=w.sum(), n_phases=len(st))
        if r.increases is not None and len(r.increases):
            row["p_liftoff_mbar"] = float(pd.to_numeric(r.increases["Increase_Pressure"], errors="coerce").iloc[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------------------
# TFA
# --------------------------------------------------------------------------------------
def parse_tfa(path: Path):
    """Returns (steps, ts, meta). steps = quasi-static ramp table, ts = 50 ms fluid response."""
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta, steps, ts, cur = {}, [], [], None
    period = 50.0
    for raw in txt:
        line = raw.strip()
        if line == "eof":
            break
        if line.startswith("SYSTEM RESPONSE"):
            cur = "steps"; continue
        if line.startswith("SYSTEM FLUID RESPONSE"):
            cur = "ts"
            m = re.search(r"period:\s*(\d+)", line)
            period = float(m.group(1)) if m else 50.0
            continue
        if cur is None:
            m = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", line)
            if m:
                meta[m.group(1)] = _num(m.group(2))
            continue
        if not line:
            continue
        (steps if cur == "steps" else ts).append(line.rstrip(";"))
    s = pd.read_csv(io.StringIO("\n".join(steps)), sep=";", decimal=",") if steps else pd.DataFrame()
    t = pd.read_csv(io.StringIO("\n".join(ts)), sep=";", decimal=",") if ts else pd.DataFrame()
    if len(s):
        s.columns = ["p_set", "p_mbar", "Q_raw"][: s.shape[1]] + list(s.columns[3:])
        s["Q_Nm3h"] = s["Q_raw"] / 100
        # flow meter full-scale ~64 Nm3/h; above it the hold-average is partially saturated
        s["valid"] = (s["Q_raw"] < FLOW_OVERFLOW) & (s["Q_Nm3h"] <= Q_FULLSCALE)
        s["file"] = path.name
    if len(t):
        t = t.apply(pd.to_numeric, errors="coerce")
        t["t_s"] = np.arange(len(t)) * period / 1000
        t["Q_Nm3h"] = t["Flow_IS"] / 100
    return s, t, meta


def tfa_points(files: list[str]) -> pd.DataFrame:
    """Valid quasi-static (Q, p) points from a set of TFA files (both ramp resolutions)."""
    out = []
    for f in files:
        s, _, _ = parse_tfa(TFA_DIR / f"{f}.csv" if not f.endswith(".csv") else TFA_DIR / f)
        out.append(s)
    d = pd.concat(out, ignore_index=True)
    return d


# Configuration catalogue (from TFA_WFE_Description.txt)
TFA_CONFIGS = {
    "VF": ["TFA_20250902_1514", "TFA_20250902_1004"],
    # WC08 build-up
    "WC08:+FT": ["TFA_20250902_1405", "TFA_20250902_1033"],
    "WC08:+LS": ["TFA_20250902_1326", "TFA_20250902_1103"],
    "WC08:+T": ["TFA_20250902_1316", "TFA_20250902_1219"],
    "WC08:+Da": ["TFA_20250911_1152", "TFA_20250911_1144"],   # Da-C10-7 = WFE operating setting
    # WC10 build-up
    "WC10:+FT": ["TFA_20260723_1417", "TFA_20260723_1426"],
    "WC10:+LS": ["TFA_20260723_1242", "TFA_20260723_1248"],
    "WC10:+T": ["TFA_20260723_1213", "TFA_20260723_1227"],
    "WC10:+Da": ["TFA_20260723_1309", "TFA_20260723_1256"],   # Da-B12-7
}

# Silencer C (Da-C10) mounted on the full WC08 system, throttle position -> files
DA_C10_FULL = {
    0: ["TFA_20250911_0928", "TFA_20250911_0941"], 1: ["TFA_20250911_1007", "TFA_20250911_0959"],
    3: ["TFA_20250911_1017", "TFA_20250911_1030"], 5: ["TFA_20250911_1055", "TFA_20250911_1044"],
    6: ["TFA_20250911_1112", "TFA_20250911_1125"], 7: ["TFA_20250911_1152", "TFA_20250911_1144"],
    8: ["TFA_20250911_1202", "TFA_20250911_1215"], 9: ["TFA_20250911_1241", "TFA_20250911_1233"],
    10: ["TFA_20250911_1300", "TFA_20250911_1321"], 11: ["TFA_20250911_1343", "TFA_20250911_1335"],
    12: ["TFA_20250911_1352", "TFA_20250911_1405"], 13: ["TFA_20250911_1429", "TFA_20250911_1421"],
    14: ["TFA_20250911_1439", "TFA_20250911_1452"], 15: ["TFA_20250911_1514", "TFA_20250911_1506"],
}


# --------------------------------------------------------------------------------------
# Fluent report files
# --------------------------------------------------------------------------------------
def parse_out(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    cols = None
    data = []
    for l in lines:
        l = l.strip()
        if l.startswith("("):
            cols = re.findall(r'"([^"]+)"', l)
            continue
        if not l or l.startswith('"'):
            continue
        parts = l.split()
        try:
            data.append([float(x) for x in parts])
        except ValueError:
            continue
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if cols and len(cols) == df.shape[1]:
        df.columns = cols
    # Fluent appends on restart -> keep the last occurrence of every iteration
    df = df.drop_duplicates(subset=df.columns[0], keep="last").sort_values(df.columns[0])
    return df.reset_index(drop=True)


def cfd_history(case: str) -> pd.DataFrame:
    """Merged iteration history for one case (e.g. 'WC10_UID19')."""
    d = REP_DIR / case
    frames = []
    for f in sorted(d.glob("*-rfile.out")):
        df = parse_out(f)
        if df.empty:
            continue
        df = df.rename(columns={df.columns[0]: "iter"}).set_index("iter")
        frames.append(df)
    h = pd.concat(frames, axis=1)
    h = h.loc[:, ~h.columns.duplicated()].sort_index()
    h.index.name = "iter"
    # some report files of a case end one iteration earlier/later -> keep common iterations
    h = h.dropna(subset=[c for c in h.columns if c.startswith(("mass-flow-inlet", "wire-drag-force", "net-mass-flow"))])
    return h.reset_index()


def cfd_cases() -> list[str]:
    return sorted(p.name for p in REP_DIR.iterdir() if p.is_dir() and p.name.startswith("WC"))


def cfd_outcome() -> pd.DataFrame:
    c = pd.read_csv(CFD_OUTCOME, sep=None, engine="python", decimal=",")
    c = c.dropna(subset=["ID"])
    c = c[c["ID"].astype(str).str.startswith("WC")].copy()
    for col in c.columns:
        if col not in ("ID", "Colour"):
            c[col] = pd.to_numeric(c[col].astype(str).str.replace("%", "").str.replace(",", "."), errors="coerce")
    return c
