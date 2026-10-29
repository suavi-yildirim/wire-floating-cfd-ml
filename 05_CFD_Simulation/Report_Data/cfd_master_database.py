import os
import pandas as pd
from pathlib import Path

MAIN_FOLDER = "." 
OUTPUT_EXCEL = "Master_CFD_Database.xlsx"

IGNORE_FILES = ["cell-count-rfile.out", "face-count-rfile.out", "report-file-0.out", "volume.out", "pressure-all-rfile.out"]

def parse_fluent_out(filepath):
    """Parses an ANSYS Fluent .out file into a Pandas DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_line = None
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('('):
            header_line = line
            data_start_idx = i + 1
            break

    if not header_line:
        return pd.DataFrame() 

    headers = header_line.strip().strip('()').replace('"', '').split()

    data = []
    for line in lines[data_start_idx:]:
        if line.strip(): 
            data.append(line.strip().split())

    df = pd.DataFrame(data, columns=headers)
    df = df.apply(pd.to_numeric, errors='coerce') 
    return df

def compile_cfd_database():
    main_path = Path(MAIN_FOLDER)
    
    summary_data = []
    history_data = []

    # Iterate through all simulation folders
    for folder_path in main_path.iterdir():
        if not folder_path.is_dir():
            continue
            
        folder_name = folder_path.name
        
        # Parse the folder name based on the naming convention
        name_parts = folder_name.split('_')
        wire_channel = name_parts[0] if len(name_parts) > 0 else "Unknown"
        uid_type = name_parts[1] if len(name_parts) > 1 else "Unknown"

        folder_df = pd.DataFrame()

        # Iterate through all .out files inside the folder
        for file_path in folder_path.glob("*.out"):
            if file_path.name in IGNORE_FILES:
                continue
                
            metric_df = parse_fluent_out(file_path)
            if metric_df.empty:
                continue

            # Merge metrics together based on the 'Iteration' column
            if folder_df.empty:
                folder_df = metric_df
            else:
                folder_df = pd.merge(folder_df, metric_df, on="Iteration", how="outer")

        if folder_df.empty:
            continue

        # --- CALCULATE TOTAL WIRE DRAG FORCE ---
        # Target the specific wall drag force columns
        drag_columns = [
            "wire-drag-force(wall_ferrule_bottom)", 
            "wire-drag-force(wall_ferrule_top)", 
            "wire-drag-force(wall_wire)"
        ]
        
        # Filter to only the columns that actually exist in the merged dataframe
        present_drag_cols = [col for col in drag_columns if col in folder_df.columns]
        
        if present_drag_cols:
            # Calculate the sum row-by-row
            total_drag = folder_df[present_drag_cols].sum(axis=1)
            
            # Find the position of 'wire-drag-force(wall_wire)' and insert right after it
            if "wire-drag-force(wall_wire)" in folder_df.columns:
                insert_loc = folder_df.columns.get_loc("wire-drag-force(wall_wire)") + 1
                folder_df.insert(insert_loc, "total-wire-drag-force", total_drag)
            else:
                # Fallback if that specific column is missing for some reason
                folder_df["total-wire-drag-force"] = total_drag

        # Add Metadata columns for the history sheet
        folder_df.insert(0, "UID", uid_type)
        folder_df.insert(0, "Wire_Channel", wire_channel)
        folder_df.insert(0, "Simulation_ID", folder_name)
        
        # Append full transient history
        history_data.append(folder_df)

        # Extract only the final converged iteration for the summary sheet
        final_row = folder_df.iloc[-1].to_dict()
        summary_data.append(final_row)

    # Combine lists into master DataFrames
    if not summary_data:
        print("No valid data found to process.")
        return
        
    df_summary = pd.DataFrame(summary_data)
    df_history = pd.concat(history_data, ignore_index=True)

    # --- SORT DATA BY UID NUMBER ---
    # Extract the numeric part of the UID string (e.g., "17" from "UID17") for proper sorting
    df_summary['UID_num'] = df_summary['UID'].str.extract(r'(\d+)').astype(float)
    df_history['UID_num'] = df_history['UID'].str.extract(r'(\d+)').astype(float)

    # Sort the Summary by Channel then UID number
    df_summary = df_summary.sort_values(by=['Wire_Channel', 'UID_num']).drop(columns=['UID_num'])
    
    # Sort the History by Channel, UID number, then Iteration step
    df_history = df_history.sort_values(by=['Wire_Channel', 'UID_num', 'Iteration']).drop(columns=['UID_num'])

    # Export to Excel with multiple sheets
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, sheet_name='Summary_Results', index=False)
        df_history.to_excel(writer, sheet_name='Iteration_History', index=False)

    print(f"Success! Master database saved and sorted to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    compile_cfd_database()