#!/usr/bin/env python3
import os
import glob
import re
from io import StringIO

import pandas as pd

FOLDER_PATH = "/Users/Suavi/Desktop/Thesis_Data"
OUTPUT_CSV = os.path.join(FOLDER_PATH, "Stable_Phase_Summary.csv")
OUTPUT_XLSX = os.path.join(FOLDER_PATH, "Stable_Phase_Summary.xlsx")

OUTPUT_TRUE_CSV = os.path.join(FOLDER_PATH, "Stable_Phase_Summary__SystemStable_TRUE.csv")
OUTPUT_FALSE_CSV = os.path.join(FOLDER_PATH, "Stable_Phase_Summary__SystemStable_FALSE.csv")

# Settings
MIN_PHASE_DURATION_MS = 1000

NEEDED_COLS = [
    "Pressure_Stable_Min",
    "Pressure_Stable_Max",
    "Pressure_Stable_Average",
    "Flow_Stable_Average",
    "Temp_Stable_Average",
]

WIRE_KEYS = ["WIRE_uid", "WIRE_instance", "WIRE_section", "WIRE_lenght"]
TUBE_KEYS = ["TUBE_uid", "TUBE_diameter", "TUBE_length", "TUBE_channel"]


def _to_number(val: str):
    if val is None:
        return None
    v = str(val).strip().replace(",", ".")
    try:
        if re.fullmatch(r"[-+]?\d+", v):
            return int(v)
        return float(v)
    except Exception:
        return val.strip()


def _to_bool(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    return None


def parse_dt(text):
    if text is None:
        return pd.NaT
    return pd.to_datetime(str(text).strip(), format="%d.%m.%Y %H:%M:%S,%f", errors="coerce")


def safe_read_lines(filepath):
    lines = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines.append(line.rstrip("\n\r"))
    return lines


def is_generated_summary_file(path):
    name = os.path.basename(path)
    return name.startswith("Stable_Phase_Summary")


def extract_wire_metadata(lines):
    meta = {k: None for k in WIRE_KEYS}

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("WIRE CONFIGURATION"):
            start_idx = i
            break

    if start_idx is None:
        return meta

    for j in range(start_idx + 1, len(lines)):
        row = lines[j].strip()
        if not row:
            continue

        up = row.upper()
        if (
            up.startswith("SENSOR INFORMATION")
            or up.startswith("SYSTEM RESPONSE")
            or up.startswith("STABLE PHASES")
            or up.startswith("INCREASES")
            or up.startswith("DECREASES")
            or up.startswith("ALL SENSOR CHANGES")
            or up.startswith("SYSTEM CONFIGURATION")
            or up.startswith("PID CONFIGURATION")
        ):
            break

        m = re.match(r"^\s*(WIRE_[A-Za-z0-9_]+)\s*:\s*([^;]+)\s*;\s*$", row)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key in meta:
                meta[key] = _to_number(val)

    return meta


def extract_tube_metadata(lines):
    meta = {k: None for k in TUBE_KEYS}

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("SYSTEM CONFIGURATION"):
            start_idx = i
            break

    if start_idx is None:
        return meta

    for j in range(start_idx + 1, len(lines)):
        row = lines[j].strip()
        if not row:
            continue

        up = row.upper()
        if (
            up.startswith("WIRE CONFIGURATION")
            or up.startswith("SENSOR INFORMATION")
            or up.startswith("SYSTEM RESPONSE")
            or up.startswith("STABLE PHASES")
            or up.startswith("INCREASES")
            or up.startswith("DECREASES")
            or up.startswith("ALL SENSOR CHANGES")
            or up.startswith("PID CONFIGURATION")
        ):
            break

        m = re.match(r"^\s*(TUBE_[A-Za-z0-9_]+)\s*:\s*([^;]+)\s*;\s*$", row)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key in meta:
                meta[key] = _to_number(val)

    return meta


def extract_system_response(lines):
    resp = {
        "System_Stable": None,
        "System_Above": None,
        "Experiment_Duration_ms": None,
        "Increase_Count": None,
        "Decrease_Count": None,
        "Change_Count": None,
    }

    for line in lines:
        row = line.strip()

        m = re.match(r"^\s*stable\s*:\s*(True|False)\s*$", row, re.IGNORECASE)
        if m:
            resp["System_Stable"] = _to_bool(m.group(1))
            continue

        m = re.match(r"^\s*above\s*:\s*(True|False)\s*$", row, re.IGNORECASE)
        if m:
            resp["System_Above"] = _to_bool(m.group(1))
            continue

        m = re.match(r"^\s*Experiment_Duration\s*:\s*([0-9]+)\s*;?\s*$", row, re.IGNORECASE)
        if m:
            resp["Experiment_Duration_ms"] = int(m.group(1))
            continue

        m = re.match(r"^\s*increases\s*:\s*([0-9]+)\s*;?\s*$", row, re.IGNORECASE)
        if m:
            resp["Increase_Count"] = int(m.group(1))
            continue

        m = re.match(r"^\s*decreases\s*:\s*([0-9]+)\s*;?\s*$", row, re.IGNORECASE)
        if m:
            resp["Decrease_Count"] = int(m.group(1))
            continue

    if resp["Increase_Count"] is not None:
        resp["Change_Count"] = resp["Increase_Count"]
    elif resp["Decrease_Count"] is not None:
        resp["Change_Count"] = resp["Decrease_Count"]

    return resp


def extract_total_sensor_changes(lines):
    for line in lines:
        m = re.search(r"ALL SENSOR CHANGES\s*-\s*count\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_stable_table(lines):
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Phase;Start_Time;End_Time;"):
            header_idx = i
            break

    if header_idx is None:
        return None

    table_lines = [lines[header_idx].strip()]

    for j in range(header_idx + 1, len(lines)):
        row = lines[j].strip()
        if not row:
            break

        up = row.upper()
        if (
            up.startswith("INCREASES")
            or up.startswith("DECREASES")
            or up.startswith("ALL SENSOR CHANGES")
            or up.startswith("SYSTEM RESPONSE")
            or up.startswith("SENSOR INFORMATION")
            or up.startswith("WIRE CONFIGURATION")
            or up.startswith("SYSTEM CONFIGURATION")
            or up.startswith("PID CONFIGURATION")
        ):
            break

        table_lines.append(row)

    if len(table_lines) < 2:
        return None

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep=";", decimal=",", engine="python")

    if "Duration" not in df.columns:
        return None
    if not all(col in df.columns for col in NEEDED_COLS):
        return None

    return df


def extract_sensor_changes_table(lines):
    """
    Parse ALL SENSOR CHANGES table.
    Keep:
    - Sensor_Number
    - Changed_Time
    - Changed_Result
    """
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("ALL SENSOR CHANGES"):
            start_idx = i
            break

    if start_idx is None:
        return pd.DataFrame(columns=["Sensor_Label", "Sensor_Number", "Changed_Time", "Changed_Result"])

    rows = []

    for j in range(start_idx + 1, len(lines)):
        row = lines[j].strip()
        if not row:
            continue

        up = row.upper()
        if (
            up.startswith("WIRE CONFIGURATION")
            or up.startswith("SYSTEM CONFIGURATION")
            or up.startswith("PID CONFIGURATION")
            or up.startswith("SENSOR INFORMATION")
            or up.startswith("SYSTEM RESPONSE")
            or up.startswith("STABLE PHASES")
            or up.startswith("INCREASES")
            or up.startswith("DECREASES")
        ):
            break

        if row.startswith("S;Changed_Time;"):
            continue

        parts = row.split(";")
        if len(parts) < 3:
            continue

        sensor_label = parts[0].strip()
        changed_time = parts[1].strip()
        changed_result_raw = parts[2].strip()

        m = re.match(r"^S(\d+)$", sensor_label, re.IGNORECASE)
        if not m:
            continue

        changed_dt = parse_dt(changed_time)
        if pd.isna(changed_dt):
            continue

        rows.append({
            "Sensor_Label": sensor_label,
            "Sensor_Number": int(m.group(1)),
            "Changed_Time": changed_dt,
            "Changed_Result": _to_bool(changed_result_raw),
        })

    if not rows:
        return pd.DataFrame(columns=["Sensor_Label", "Sensor_Number", "Changed_Time", "Changed_Result"])

    return pd.DataFrame(rows).sort_values("Changed_Time").reset_index(drop=True)


def summarize_stable_phases(df: pd.DataFrame):
    df = df.copy()

    for c in NEEDED_COLS + ["Duration", "Phase"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Duration"]).copy()
    df = df[df["Duration"] >= MIN_PHASE_DURATION_MS].copy()

    if df.empty:
        return None, None

    total_duration = df["Duration"].sum()
    if total_duration <= 0:
        return None, None

    pressure_avg = (df["Pressure_Stable_Average"] * df["Duration"]).sum() / total_duration
    flow_avg = (df["Flow_Stable_Average"] * df["Duration"]).sum() / total_duration
    temp_avg = (df["Temp_Stable_Average"] * df["Duration"]).sum() / total_duration

    summary = {
        "Pressure_Stable_Min (mbar)": df["Pressure_Stable_Min"].min(),
        "Pressure_Stable_Max (mbar)": df["Pressure_Stable_Max"].max(),
        "Pressure_Stable_Average (mbar)": pressure_avg,
        "Flow_Stable_Average (Nm^3/h)": flow_avg / 100,
        "Temp_Stable_Average (C)": temp_avg / 10,
        "Stable_Phase_Rows": int(df.shape[0]),
        "Total_Stable_Duration_ms": float(total_duration),
        "Min_Phase_Duration_ms_Filter": MIN_PHASE_DURATION_MS,
    }

    return summary, df


def replay_sensor_states_until(df_sensor_changes, ref_time):
    """
    Başlangıçta tüm sensörler True kabul edilir.
    ref_time anına kadar event'leri replay eder ve current state döndürür.
    """
    if df_sensor_changes is None or df_sensor_changes.empty or pd.isna(ref_time):
        return {}

    all_sensors = sorted(df_sensor_changes["Sensor_Number"].dropna().astype(int).unique().tolist())
    sensor_states = {s: True for s in all_sensors}

    relevant = df_sensor_changes[df_sensor_changes["Changed_Time"] <= ref_time].copy()
    if relevant.empty:
        return sensor_states

    relevant = relevant.sort_values("Changed_Time")

    for _, row in relevant.iterrows():
        sensor_num = int(row["Sensor_Number"])
        changed_result = row["Changed_Result"]
        if changed_result is None:
            continue
        sensor_states[sensor_num] = bool(changed_result)

    return sensor_states


def get_active_anytime_in_phase(df_sensor_changes, phase_start, phase_end):
    """
    Stable phase boyunca herhangi bir anda aktif olan sensörleri bulur.
    Başlangıç state'i phase_start anında reconstruct edilir,
    sonra phase boyunca event'ler replay edilerek union alınır.
    """
    if df_sensor_changes is None or df_sensor_changes.empty:
        return []

    if pd.isna(phase_start) or pd.isna(phase_end):
        return []

    sensor_states = replay_sensor_states_until(df_sensor_changes, phase_start)

    active_anytime = {s for s, state in sensor_states.items() if state}

    phase_events = df_sensor_changes[
        (df_sensor_changes["Changed_Time"] >= phase_start) &
        (df_sensor_changes["Changed_Time"] <= phase_end)
    ].copy()

    if not phase_events.empty:
        phase_events = phase_events.sort_values("Changed_Time")
        for _, row in phase_events.iterrows():
            sensor_num = int(row["Sensor_Number"])
            changed_result = row["Changed_Result"]
            if changed_result is None:
                continue
            sensor_states[sensor_num] = bool(changed_result)
            if sensor_states[sensor_num]:
                active_anytime.add(sensor_num)

    return sorted(active_anytime)


def get_previous_distinct_sensor(df_sensor_changes, ref_time, stable_phase_sensor):
    """
    stable phase başlamadan önceki en son,
    stable_phase_sensor'dan farklı sensor event'ini bulur.
    """
    if df_sensor_changes is None or df_sensor_changes.empty or pd.isna(ref_time):
        return None

    before = df_sensor_changes[df_sensor_changes["Changed_Time"] < ref_time].copy()
    if before.empty:
        return None

    before = before.sort_values("Changed_Time", ascending=False)

    for _, row in before.iterrows():
        sensor_num = int(row["Sensor_Number"])
        if sensor_num != stable_phase_sensor:
            return sensor_num

    return None


def classify_motion_from_sensors(df_stable_filtered, df_sensor_changes):
    """
    Yeni doğru algoritma:
    1. Son stable phase'i al (largest Phase).
    2. O phase'in gerçek start/end zamanını bul.
    3. Stable phase boyunca herhangi bir anda aktif olan sensörleri reconstruct et.
    4. Stable_Phase_Sensor = min(active_anytime_in_phase)
    5. Previous_Sensor = stable phase başlamadan önceki son distinct sensor
    6. Compare:
         previous < stable -> wire_rises
         previous > stable -> wire_drops
         else -> unknown
    """
    if df_stable_filtered is None or df_stable_filtered.empty:
        return None, None, None, None, None

    if df_sensor_changes is None or df_sensor_changes.empty:
        return None, None, None, False, None

    work = df_stable_filtered.copy()
    work["Phase"] = pd.to_numeric(work["Phase"], errors="coerce")
    work["Duration"] = pd.to_numeric(work["Duration"], errors="coerce")
    work["Stable_Start_DT"] = work["Start_Time"].apply(parse_dt)
    work["Stable_End_DT"] = work["End_Time"].apply(parse_dt)

    work = work.dropna(subset=["Phase", "Duration", "Stable_Start_DT"]).copy()
    if work.empty:
        return None, None, None, False, None

    rep_phase = work.sort_values("Phase", ascending=True).iloc[-1]

    phase_start = rep_phase["Stable_Start_DT"]
    phase_end = rep_phase["Stable_End_DT"]

    if pd.isna(phase_end) or phase_end.year < 2000:
        phase_end = phase_start + pd.to_timedelta(rep_phase["Duration"], unit="ms")

    active_anytime = get_active_anytime_in_phase(df_sensor_changes, phase_start, phase_end)

    if not active_anytime:
        return "unknown", None, None, False, None

    stable_phase_sensor = min(active_anytime)

    # senin fiziksel validation kuralın
    allowed_stable_sensors = {18, 19, 20}
    stable_sensor_valid = set(active_anytime).issubset(allowed_stable_sensors)

    stable_active_sensors_text = ",".join(f"S{s}" for s in active_anytime)

    previous_sensor = get_previous_distinct_sensor(
        df_sensor_changes=df_sensor_changes,
        ref_time=phase_start,
        stable_phase_sensor=stable_phase_sensor,
    )

    if previous_sensor is None:
        return "unknown", stable_phase_sensor, None, stable_sensor_valid, stable_active_sensors_text

    if previous_sensor < stable_phase_sensor:
        final_trend = "wire_rises"
    elif previous_sensor > stable_phase_sensor:
        final_trend = "wire_drops"
    else:
        final_trend = "unknown"

    return (
        final_trend,
        stable_phase_sensor,
        previous_sensor,
        stable_sensor_valid,
        stable_active_sensors_text,
    )


def extract_all_from_file(filepath: str):
    lines = safe_read_lines(filepath)

    wire_meta = extract_wire_metadata(lines)
    tube_meta = extract_tube_metadata(lines)
    sys_resp = extract_system_response(lines)
    total_sensor_changes = extract_total_sensor_changes(lines)
    df_stable = extract_stable_table(lines)
    df_sensor_changes = extract_sensor_changes_table(lines)

    return wire_meta, tube_meta, sys_resp, total_sensor_changes, df_stable, df_sensor_changes


def main():
    csv_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.csv")))
    csv_files = [fp for fp in csv_files if not is_generated_summary_file(fp)]

    print(f"Found {len(csv_files)} raw CSV files.")

    rows = []
    skipped = []
    filtered_out = []

    for idx, fp in enumerate(csv_files, start=1):
        filename = os.path.basename(fp)
        print(f"[{idx}/{len(csv_files)}] Processing {filename}")

        wire_meta, tube_meta, sys_resp, total_sensor_changes, df_stable, df_sensor_changes = extract_all_from_file(fp)

        if df_stable is None:
            skipped.append(filename)
            continue

        summary, df_stable_filtered = summarize_stable_phases(df_stable)
        if summary is None:
            filtered_out.append(filename)
            continue

        (
            final_trend,
            stable_phase_sensor,
            previous_sensor_num,
            stable_sensor_valid,
            stable_active_sensors_text,
        ) = classify_motion_from_sensors(
            df_stable_filtered=df_stable_filtered,
            df_sensor_changes=df_sensor_changes,
        )

        summary["File"] = filename
        summary["WIRE_uid"] = wire_meta.get("WIRE_uid")
        summary["WIRE_instance"] = wire_meta.get("WIRE_instance")
        summary["WIRE_section"] = wire_meta.get("WIRE_section")
        summary["WIRE_lenght (m)"] = wire_meta.get("WIRE_lenght")
        summary["TUBE_diameter (mm)"] = tube_meta.get("TUBE_diameter")

        summary["System_Stable"] = sys_resp.get("System_Stable")
        summary["System_Above"] = sys_resp.get("System_Above")
        summary["Change_Count"] = sys_resp.get("Change_Count")

        summary["Final_Trend"] = final_trend
        summary["Previous_Sensor"] = previous_sensor_num
        summary["Stable_Phase_Sensor"] = stable_phase_sensor
        summary["Stable_Sensor_Valid"] = stable_sensor_valid
        summary["Stable_Active_Sensors"] = stable_active_sensors_text

        summary["Total_Sensor_Changes"] = total_sensor_changes
        summary["Experiment_Duration_ms"] = sys_resp.get("Experiment_Duration_ms")

        rows.append(summary)

    out_df = pd.DataFrame(rows)

    column_order = [
        "File",
        "WIRE_uid",
        "WIRE_instance",
        "WIRE_section",
        "WIRE_lenght (m)",
        "TUBE_diameter (mm)",
        "System_Stable",
        "System_Above",
        "Change_Count",
        "Final_Trend",
        "Previous_Sensor",
        "Stable_Phase_Sensor",
        "Stable_Sensor_Valid",
        "Stable_Active_Sensors",
        "Total_Sensor_Changes",
        "Experiment_Duration_ms",
        "Stable_Phase_Rows",
        "Total_Stable_Duration_ms",
        "Min_Phase_Duration_ms_Filter",
        "Pressure_Stable_Min (mbar)",
        "Pressure_Stable_Max (mbar)",
        "Pressure_Stable_Average (mbar)",
        "Flow_Stable_Average (Nm^3/h)",
        "Temp_Stable_Average (C)",
    ]

    column_order = [c for c in column_order if c in out_df.columns]
    out_df = out_df[column_order]

    sort_cols = [c for c in ["WIRE_uid", "WIRE_instance", "WIRE_lenght (m)", "File"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, na_position="last")

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote summary CSV: {OUTPUT_CSV}")

    true_df = out_df[out_df["System_Stable"] == True].copy() if "System_Stable" in out_df.columns else pd.DataFrame()
    false_df = out_df[out_df["System_Stable"] == False].copy() if "System_Stable" in out_df.columns else pd.DataFrame()

    if not true_df.empty:
        true_df.to_csv(OUTPUT_TRUE_CSV, index=False)
        print(f"✅ Wrote TRUE subset CSV: {OUTPUT_TRUE_CSV}")

    if not false_df.empty:
        false_df.to_csv(OUTPUT_FALSE_CSV, index=False)
        print(f"✅ Wrote FALSE subset CSV: {OUTPUT_FALSE_CSV}")

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            out_df.to_excel(writer, sheet_name="All", index=False)
            if not true_df.empty:
                true_df.to_excel(writer, sheet_name="SystemStable_TRUE", index=False)
            if not false_df.empty:
                false_df.to_excel(writer, sheet_name="SystemStable_FALSE", index=False)
        print(f"✅ Wrote summary Excel (multi-sheet): {OUTPUT_XLSX}")
    except Exception as e:
        print(f"Excel export skipped: {e}")

    if skipped:
        print("\n⚠️ Skipped files:")
        for name in skipped:
            print(" -", name)

    if filtered_out:
        print(f"\nℹ️ Files filtered out because no stable phase survived Duration >= {MIN_PHASE_DURATION_MS} ms:")
        for name in filtered_out:
            print(" -", name)

    if "Final_Trend" in out_df.columns:
        unknown_trend = out_df[out_df["Final_Trend"].isna() | (out_df["Final_Trend"] == "unknown")]["File"].tolist()
        if unknown_trend:
            print("\nℹ️ Files where sensor-based Final_Trend is unknown:")
            for name in unknown_trend:
                print(" -", name)


if __name__ == "__main__":
    main()