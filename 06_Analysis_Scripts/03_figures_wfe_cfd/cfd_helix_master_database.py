import os
import pandas as pd
from pathlib import Path

MAIN_FOLDER = "Report_Data"
OUTPUT_EXCEL = "Report_Data/Master_Helix_CFD_Database.xlsx"
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
    data = [line.strip().split() for line in lines[data_start_idx:] if line.strip()]

    df = pd.DataFrame(data, columns=headers)
    return df.apply(pd.to_numeric, errors='coerce') 

def compile_cfd_database():
    main_path = Path(MAIN_FOLDER)
    summary_data = []
    history_data = []

    for folder_path in main_path.iterdir():
        if not folder_path.is_dir() or not folder_path.name.startswith("Helix"):
            continue
            
        folder_name = folder_path.name
        
        # Parse the folder name: Helix_Dia2.5_Rev1.0_WC10_UID18
        name_parts = folder_name.split('_')
        try:
            helix_dia = float(name_parts[1].replace("Dia", ""))
            helix_rev = float(name_parts[2].replace("Rev", ""))
            wire_channel = name_parts[3]
            uid_type = name_parts[4]
        except IndexError:
            print(f"Skipping {folder_name}: Does not match naming convention.")
            continue

        folder_df = pd.DataFrame()

        for file_path in folder_path.glob("*.out"):
            if file_path.name in IGNORE_FILES:
                continue
                
            metric_df = parse_fluent_out(file_path)
            if metric_df.empty:
                continue

            if folder_df.empty:
                folder_df = metric_df
            else:
                folder_df = pd.merge(folder_df, metric_df, on="Iteration", how="outer")

        if folder_df.empty:
            continue

        # Calculate total wire drag force
        drag_columns = [
            "wire-drag-force(wall_ferrule_bottom)", 
            "wire-drag-force(wall_ferrule_top)", 
            "wire-drag-force(wall_wire)"
        ]
        present_drag_cols = [col for col in drag_columns if col in folder_df.columns]
        
        if present_drag_cols:
            total_drag = folder_df[present_drag_cols].sum(axis=1)
            if "wire-drag-force(wall_wire)" in folder_df.columns:
                insert_loc = folder_df.columns.get_loc("wire-drag-force(wall_wire)") + 1
                folder_df.insert(insert_loc, "total-wire-drag-force", total_drag)
            else:
                folder_df["total-wire-drag-force"] = total_drag

        # Calculate total wire skin friction (Viscous Drag)
        friction_columns = [
            "wire_skin_friction(wall_ferrule_bottom)", 
            "wire_skin_friction(wall_ferrule_top)", 
            "wire_skin_friction(wall_wire)"
        ]
        present_friction_cols = [col for col in friction_columns if col in folder_df.columns]
        
        if present_friction_cols:
            total_friction = folder_df[present_friction_cols].sum(axis=1)
            if "wire_skin_friction(wall_wire)" in folder_df.columns:
                insert_loc = folder_df.columns.get_loc("wire_skin_friction(wall_wire)") + 1
                folder_df.insert(insert_loc, "total-wire-skin-friction", total_friction)
            else:
                folder_df["total-wire-skin-friction"] = total_friction

        # Insert Helical Metadata
        folder_df.insert(0, "Helix_Rev", helix_rev)
        folder_df.insert(0, "Helix_Dia", helix_dia)
        folder_df.insert(0, "UID", uid_type)
        folder_df.insert(0, "Wire_Channel", wire_channel)
        folder_df.insert(0, "Simulation_ID", folder_name)
        
        history_data.append(folder_df)
        summary_data.append(folder_df.iloc[-1].to_dict())

    if not summary_data:
        print("No valid data found to process.")
        return
        
    df_summary = pd.DataFrame(summary_data)
    df_history = pd.concat(history_data, ignore_index=True)

    # Sort primarily by Helix variables
    df_summary = df_summary.sort_values(by=['Wire_Channel', 'Helix_Dia', 'Helix_Rev'])
    df_history = df_history.sort_values(by=['Wire_Channel', 'Helix_Dia', 'Helix_Rev', 'Iteration'])

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, sheet_name='Summary_Results', index=False)
        df_history.to_excel(writer, sheet_name='Iteration_History', index=False)

    print(f"Success! Helix master database saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    compile_cfd_database()