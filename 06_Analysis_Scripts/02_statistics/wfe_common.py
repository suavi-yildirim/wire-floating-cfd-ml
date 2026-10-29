"""Shared paths and loaders for the WFE thesis analysis scripts.

Expected layout (the scripts are run from anywhere):
    Masters Thesis/
        03_Experimental_Data/        run CSVs, Stable_Phase_Summary.csv, CFD_Experimental_Outcome.xlsx
        06_Analysis_Scripts/
            02_statistics/    this file and the statistics scripts
            output/           results are written here
"""
import csv
import collections
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTERS_THESIS = HERE.parents[1]
DATA_DIR = MASTERS_THESIS / "03_Experimental_Data"
OUT_DIR = HERE.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

P = "Pressure_Stable_Average (mbar)"
F = "Flow_Stable_Average (Nm^3/h)"


def to_float(x):
    try:
        return float(x.replace(",", "."))
    except (AttributeError, ValueError):
        return None


def load_stable_rows():
    """Stable-phase summary, restricted to runs flagged as stable by the controller."""
    path = DATA_DIR / "Stable_Phase_Summary.csv"
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [r for r in csv.DictReader(fh) if r["System_Stable"] == "True"]


def tube_of(row):
    return int(float(row["TUBE_diameter (mm)"]))


def uid_of(row):
    return int(row["WIRE_uid"])


def direction_groups(rows):
    """{(tube, uid): {'wire_rises': [p...], 'wire_drops': [p...]}} using stable pressures."""
    g = collections.defaultdict(lambda: {"wire_rises": [], "wire_drops": []})
    for r in rows:
        if r["Final_Trend"] not in ("wire_rises", "wire_drops"):
            continue
        p = to_float(r[P])
        if p is not None:
            g[(tube_of(r), uid_of(r))][r["Final_Trend"]].append(p)
    return g


def load_cfd_weights():
    """{(uid, tube): wire_weight_N} from CFD_Experimental_Outcome.xlsx."""
    import openpyxl
    wb = openpyxl.load_workbook(DATA_DIR / "CFD_Experimental_Outcome.xlsx",
                                read_only=True, data_only=True)
    ws = wb.worksheets[0]
    hdr, info = None, {}
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            hdr = row
            continue
        d = dict(zip(hdr, row))
        if d["Tube_Diameter[mm]"] is None:
            continue
        info[(int(d["Wire_UID"]), int(d["Tube_Diameter[mm]"]))] = d["Wire_Weight[N]"]
    return info
