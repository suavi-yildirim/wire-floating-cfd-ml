"""Settling analysis of WFE runs (scaffold).

Parses the system-response block of each run CSV into a time series.
"""
import csv
import glob
import os

DATA_DIR = "/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis/03_Experimental_Data"
HEADER = "Pressure_SET;Pressure_IS;Flow_IS;Temperature_IS;Position_WIRE"
SAMPLE_MS = 50  # from "period:50" in the file header


def load_response(path):
    """Return a list of rows (set, pressure, flow, temperature, position) for one run.

    Position is a float (decimal comma in the file); the other columns are integers.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == HEADER) + 1
    for line in lines[start:]:
        parts = line.strip().split(";")
        if len(parts) != 5:
            break
        try:
            # Four integer columns; the wire position uses a decimal comma.
            values = [int(p) for p in parts[:4]] + [float(parts[4].replace(",", "."))]
        except ValueError:
            break
        rows.append(tuple(values))
    return rows


def time_s(index):
    return index * SAMPLE_MS / 1000.0


def header_value(path, key):
    """Return a header value such as 'Target_Min:0,0725;' as a string with a decimal point."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip().rstrip(";")
            if text.startswith(key + ":"):
                return text.split(":", 1)[1].replace(",", ".")
    return None


def settling_time(path, rows, tolerance_s=1.0):
    """Settled: the start of the last in-window hold that lasts the specified time.

    Window: Target_Min..Target_Max (m). Hold: Timer_WFE_Stable_Time (ms), converted to samples.
    Dropouts of up to tolerance_s (1 s) outside the window do not interrupt a hold.
    Earlier in-window fluctuations do not count: the settling time is the start of the
    final hold of the required length. Returns None if no hold of that length occurs.
    """
    low = header_value(path, "Target_Min")
    high = header_value(path, "Target_Max")
    hold_ms = header_value(path, "Timer_WFE_Stable_Time")
    if low is None or high is None or hold_ms is None:
        return None
    low, high = float(low), float(high)
    hold_samples = round(float(hold_ms) / SAMPLE_MS)
    gap_samples = round(tolerance_s * 1000.0 / SAMPLE_MS)
    segments = []  # [start, end] of in-window spans, merging short dropouts
    for i, row in enumerate(rows):
        if low <= row[4] <= high:
            if segments and i - segments[-1][1] - 1 <= gap_samples:
                segments[-1][1] = i
            else:
                segments.append([i, i])
    qualifying = [seg for seg in segments if seg[1] - seg[0] + 1 >= hold_samples]
    return None if not qualifying else time_s(qualifying[-1][0])


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA_DIR, "WFE_*.csv")))
    settled = 0
    times = []
    for path in files[:30]:
        t = settling_time(path, load_response(path))
        if t is None:
            continue
        settled += 1
        times.append(t)
    print("tested:", min(30, len(files)), "settled:", settled)
    print("settling times (s):", sorted(round(t, 1) for t in times))
