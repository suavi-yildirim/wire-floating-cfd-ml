#!/usr/bin/env python3
"""
Tube Flow Analysis (TFA) batch summary + comparison at same pressure setpoints.

What it does:
- Searches a folder (recursively) for the given TFA file IDs (e.g., "TFA_20250911_0928")
- Parses each TFA *.csv (the Siemens-style file with header sections + "SYSTEM RESPONSE")
- Extracts:
  - TUBE_Description (your configuration string)
  - Response table: pressure_SET / pressure_IS / flow_IS / temp_IS
- Builds comparison tables:
  1) Per-file table (long format)
  2) Per-configuration mean flow at each pressure_SET (and std across repetitions)
  3) Wide table: rows=pressure_SET, columns=configs (mean flow)
- Writes outputs to ./tfa_summary_out/

Usage:
  python tfa_batch_summary.py --root /path/to/data

Optional:
  python tfa_batch_summary.py --root . --pattern ".csv"
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# 1) Put your configuration -> file IDs here (no extension needed).
CONFIG_TO_FILEIDS: Dict[str, List[str]] = {
    "Da-C10-0":  ["TFA_20250911_0928", "TFA_20250911_0941"],
    "Da-C10-1":  ["TFA_20250911_1007", "TFA_20250911_0959"],
    "Da-C10-3":  ["TFA_20250911_1017", "TFA_20250911_1030"],
    "Da-C10-5":  ["TFA_20250911_1055", "TFA_20250911_1044"],
    "Da-C10-6":  ["TFA_20250911_1112", "TFA_20250911_1125"],
    "Da-C10-7":  ["TFA_20250911_1152", "TFA_20250911_1144"],
    "Da-C10-8":  ["TFA_20250911_1202", "TFA_20250911_1215"],
    "Da-C10-9":  ["TFA_20250911_1241", "TFA_20250911_1233"],
    "Da-C10-10": ["TFA_20250911_1300", "TFA_20250911_1321"],
    "Da-C10-11": ["TFA_20250911_1343", "TFA_20250911_1335"],
    "Da-C10-12": ["TFA_20250911_1352", "TFA_20250911_1405"],
    "Da-C10-13": ["TFA_20250911_1429", "TFA_20250911_1421"],
    "Da-C10-14": ["TFA_20250911_1439", "TFA_20250911_1452"],
    "Da-C10-15": ["TFA_20250911_1514", "TFA_20250911_1506"],
}


def find_files_by_ids(root: Path, file_ids: List[str], extension_pattern: str = r"\.csv$") -> Dict[str, Path]:
    """
    Searches root recursively for files containing each file_id.
    Returns mapping file_id -> path.
    """
    found: Dict[str, Path] = {}
    # Build a quick lookup set for speed
    wanted = set(file_ids)

    # Search all files once
    ext_re = re.compile(extension_pattern, re.IGNORECASE)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not ext_re.search(p.name):
            continue
        # Typical names: TFA_YYYYMMDD_HHMM.csv
        for fid in list(wanted):
            if fid in p.name:
                found[fid] = p
                wanted.remove(fid)
        if not wanted:
            break
    return found


def parse_tfa_file(file_path: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Parses your TFA file format:
    - reads header lines to capture metadata like TUBE_Description
    - finds 'SYSTEM RESPONSE' section and parses the semicolon table after it

    Returns:
      meta: dict
      data: DataFrame with columns:
        TFA_pressure_SET, TFA_pressure_IS, TFA_flow_IS, TFA_temp_IS
    """
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    meta: Dict[str, str] = {"file": str(file_path)}
    # Extract selected metadata lines (key:value; style)
    # Example: "TUBE_Description:VF-1 + FT-08-2.4 + ...;"
    kv_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*?);?\s*$")

    for ln in lines:
        m = kv_re.match(ln)
        if m:
            k, v = m.group(1), m.group(2)
            meta[k] = v

    # Find the table header line after "SYSTEM RESPONSE"
    # In your example:
    # SYSTEM RESPONSE
    # TFA_pressure_SET;TFA_pressure_IS;TFA_flow_IS;TFA_temp_IS
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "SYSTEM RESPONSE":
            header_idx = i + 1
            break
    if header_idx is None or header_idx >= len(lines):
        raise ValueError(f"Could not find 'SYSTEM RESPONSE' section in {file_path.name}")

    # The header row itself is at header_idx; data starts at header_idx+1
    # Parse with pandas from those lines only
    table_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(
        pd.io.common.StringIO(table_text),
        sep=";",
        decimal=",",          # important: German decimal commas
        engine="python",
    )

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric
    for c in ["TFA_pressure_SET", "TFA_pressure_IS", "TFA_flow_IS", "TFA_temp_IS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    return meta, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root folder containing TFA csv files (searched recursively).")
    ap.add_argument("--out", type=str, default="tfa_summary_out", help="Output folder.")
    ap.add_argument("--extension_regex", type=str, default=r"\.csv$", help="Regex to match data files.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all desired file IDs
    all_file_ids: List[str] = []
    for cfg, fids in CONFIG_TO_FILEIDS.items():
        all_file_ids.extend(fids)

    found_map = find_files_by_ids(root, all_file_ids, extension_pattern=args.extension_regex)

    missing = [fid for fid in all_file_ids if fid not in found_map]
    if missing:
        print("\nWARNING: Missing files (not found under root):")
        for fid in missing:
            print("  -", fid)
        print("\n(Outputs will be created from the files that were found.)\n")

    # Parse all found files and build a long table
    rows = []
    meta_rows = []
    for cfg, fids in CONFIG_TO_FILEIDS.items():
        for fid in fids:
            if fid not in found_map:
                continue
            path = found_map[fid]
            meta, df = parse_tfa_file(path)

            tube_desc = meta.get("TUBE_Description", "")
            tube_name = meta.get("TUBE_name", "")
            tube_uid = meta.get("TUBE_uid", "")
            date_str = meta.get("Date", meta.get("Date_Time", meta.get("DateTime", "")))

            meta_rows.append({
                "configuration": cfg,
                "file_id": fid,
                "path": str(path),
                "Date": date_str,
                "TUBE_uid": tube_uid,
                "TUBE_name": tube_name,
                "TUBE_Description": tube_desc,
            })

            for _, r in df.iterrows():
                rows.append({
                    "configuration": cfg,
                    "file_id": fid,
                    "TFA_pressure_SET": r.get("TFA_pressure_SET"),
                    "TFA_pressure_IS": r.get("TFA_pressure_IS"),
                    "TFA_flow_IS": r.get("TFA_flow_IS"),
                    "TFA_temp_IS": r.get("TFA_temp_IS"),
                    "TUBE_Description": tube_desc,
                })

    long_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta_rows)

    # Save raw extracts
    meta_df.to_csv(out_dir / "01_files_metadata.csv", index=False)
    long_df.to_csv(out_dir / "02_tfa_long_all_points.csv", index=False)

    if long_df.empty:
        print("No data extracted. Check --root and filenames.")
        return

    # 2) Compare flows at same pressure_SET across configurations
    # Aggregate per configuration and pressure_SET across repetitions:
    agg = (
        long_df
        .groupby(["configuration", "TFA_pressure_SET"], dropna=False)
        .agg(
            flow_mean=("TFA_flow_IS", "mean"),
            flow_std=("TFA_flow_IS", "std"),
            pressure_is_mean=("TFA_pressure_IS", "mean"),
            temp_mean=("TFA_temp_IS", "mean"),
            n_points=("TFA_flow_IS", "count"),
            n_files=("file_id", "nunique"),
        )
        .reset_index()
        .sort_values(["TFA_pressure_SET", "configuration"])
    )
    agg.to_csv(out_dir / "03_flow_vs_pressure_by_config_mean_std.csv", index=False)

    # Wide table: rows pressure_SET, columns configs -> mean flow
    wide_mean = agg.pivot(index="TFA_pressure_SET", columns="configuration", values="flow_mean").sort_index()
    wide_std = agg.pivot(index="TFA_pressure_SET", columns="configuration", values="flow_std").sort_index()

    wide_mean.to_csv(out_dir / "04_wide_mean_flow.csv")
    wide_std.to_csv(out_dir / "05_wide_std_flow.csv")

    # Optional: simple "relative to baseline" table (choose Da-C10-0 as baseline if present)
    baseline = "Da-C10-0"
    if baseline in wide_mean.columns:
        rel = wide_mean.divide(wide_mean[baseline], axis=0) - 1.0
        rel.to_csv(out_dir / "06_relative_flow_vs_baseline_Da-C10-0.csv")

    print("Done ✅")
    print(f"Outputs written to: {out_dir}")
    print("Key outputs:")
    print("  03_flow_vs_pressure_by_config_mean_std.csv  (main comparison table)")
    print("  04_wide_mean_flow.csv                      (pressure rows, config columns)")
    print("  06_relative_flow_vs_baseline_Da-C10-0.csv   (optional, if baseline exists)")


if __name__ == "__main__":
    main()