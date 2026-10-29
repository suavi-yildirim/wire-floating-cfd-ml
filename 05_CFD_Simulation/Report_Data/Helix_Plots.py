import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# ==========================================
# 1. LOAD SUMMARY DATA
# ==========================================
df_helix = pd.read_excel("Report_Data/Master_Helix_CFD_Database.xlsx", sheet_name='Summary_Results')
df_straight = pd.read_excel("Report_Data/Master_CFD_Database.xlsx", sheet_name='Summary_Results')

# Integrate the Straight Baseline (Rev 0.0)
baseline_row = df_straight[df_straight['Simulation_ID'] == 'WC10_UID18'].iloc[0]
baseline_entries = []
unique_diameters = df_helix['Helix_Dia'].unique()

for dia in unique_diameters:
    new_row = baseline_row.copy()
    new_row['Helix_Dia'] = dia
    new_row['Helix_Rev'] = 0.0
    new_row['Simulation_ID'] = f"Baseline_Dia{dia}_Rev0.0"
    baseline_entries.append(new_row)
    
df_baselines = pd.DataFrame(baseline_entries)
df_summary = pd.concat([df_baselines, df_helix], ignore_index=True)

df_summary['Mass_Flow_Abs'] = df_summary['mass-flow-inlet'].abs() 
df_summary['Helix_Diameter_Label'] = df_summary['Helix_Dia'].apply(lambda x: f"Dia: {x} mm")
df_summary = df_summary.sort_values(by=['Helix_Dia', 'Helix_Rev'])

# ==========================================
# 2. BULLETPROOF DATA EXTRACTION
# ==========================================
base_dir = "Report_Data"

# Helper function to extract specific columns from Fluent .out files
def extract_last_value(filepath, col_index):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as file:
                lines = [line.strip() for line in file if line.strip()]
            last_line = lines[-1].split()
            if len(last_line) > col_index:
                return float(last_line[col_index])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return np.nan

cd_wire_vals = []
inlet_pressure_vals = []
viscous_drag_vals = []
total_drag_wire_vals = []

for index, row in df_summary.iterrows():
    dia_str = f"{float(row['Helix_Dia']):.1f}"
    rev_str = f"{float(row['Helix_Rev']):.1f}"
    
    if row['Helix_Rev'] == 0.0:
        folder_name = "WC10_UID18"
    else:
        folder_name = f"Helix_Dia{dia_str}_Rev{rev_str}_WC10_UID18"
        
    path_cd       = os.path.join(base_dir, folder_name, "wire-drag-coefficient-rfile.out")
    path_pressure = os.path.join(base_dir, folder_name, "pressure-inlet-rfile.out")
    path_drag     = os.path.join(base_dir, folder_name, "wire-drag-force-rfile.out")
    path_viscous  = os.path.join(base_dir, folder_name, "wire_skin_friction-rfile.out")
    
    # Extract targeted metrics (Wire components are Index 3, Pressure is Index 1)
    cd_wire_vals.append(extract_last_value(path_cd, 3))
    inlet_pressure_vals.append(extract_last_value(path_pressure, 1))
    total_drag_wire_vals.append(extract_last_value(path_drag, 3))
    viscous_drag_vals.append(extract_last_value(path_viscous, 3))

# Assign to DataFrame
df_summary['Final_Cd_Wire'] = pd.to_numeric(cd_wire_vals, errors='coerce')
df_summary['Pressure_Inlet_Pa'] = pd.to_numeric(inlet_pressure_vals, errors='coerce')
df_summary['Total_Drag_Wire_Only'] = pd.to_numeric(total_drag_wire_vals, errors='coerce')
df_summary['Viscous_Drag_Wire'] = pd.to_numeric(viscous_drag_vals, errors='coerce')

# Calculate Pressure Drag directly
df_summary['Pressure_Drag_Wire'] = df_summary['Total_Drag_Wire_Only'] - df_summary['Viscous_Drag_Wire']



# ==========================================
# 3. IEEE & THESIS STYLE CONFIGURATION
# ==========================================
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,           
    'axes.labelsize': 8,      
    'axes.titlesize': 9,      
    'xtick.labelsize': 8,     
    'ytick.labelsize': 8,     
    'legend.fontsize': 6.5,     
    'legend.title_fontsize': 7.0, 
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
})

palette = sns.color_palette("flare", n_colors=len(unique_diameters))

# ==========================================
# 4. GENERATE PLOTS
# ==========================================

# --- PLOT 1: System Drag Force vs. Revolutions ---
fig1, ax1 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Total_Drag_Wire_Only', 
    hue='Helix_Diameter_Label', palette=palette, marker='o', markersize=5, ax=ax1
)
ax1.set_title('Total System Aerodynamic Drag vs. Helical Deformation')
ax1.set_xlabel('Number of Revolutions')
ax1.set_ylabel('Total CFD Drag Force [N]')
ax1.grid(True, linestyle=':', color='lightgray')
ax1.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig1.savefig('Report_Data/Helical_Deformation_Drag.png', dpi=600, bbox_inches='tight')
plt.close(fig1)

# --- PLOT 2: Mass Flow Rate vs. Revolutions ---
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Mass_Flow_Abs', 
    hue='Helix_Diameter_Label', palette=palette, marker='s', markersize=5, ax=ax2
)
ax2.set_title('Mass Flow Rate vs. Helical Deformation')
ax2.set_xlabel('Number of Revolutions')
ax2.set_ylabel('Inlet Mass Flow Rate [kg/s]')
ax2.grid(True, linestyle=':', color='lightgray')
ax2.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig2.savefig('Report_Data/Helical_Deformation_MassFlow.png', dpi=600, bbox_inches='tight')
plt.close(fig2)

# --- PLOT 3: Wire Drag Coefficient vs. Revolutions ---
fig3, ax3 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Final_Cd_Wire', 
    hue='Helix_Diameter_Label', palette=palette, marker='^', markersize=5, ax=ax3
)
ax3.set_title('Wire Body Drag Coefficient vs. Helical Deformation')
ax3.set_xlabel('Number of Revolutions')
ax3.set_ylabel('Wire Drag Coefficient ($C_{D, wire}$)')
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.grid(True, linestyle=':', color='lightgray')
ax3.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig3.savefig('Report_Data/Helical_Deformation_Cd.png', dpi=600, bbox_inches='tight')
plt.close(fig3)

# --- PLOT 4: Pressure Inlet vs. Revolutions ---
fig4, ax4 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Pressure_Inlet_Pa', 
    hue='Helix_Diameter_Label', palette=palette, marker='d', markersize=5, ax=ax4
)
ax4.set_title('Inlet Pressure vs. Helical Deformation')
ax4.set_xlabel('Number of Revolutions')
ax4.set_ylabel('CFD Inlet Pressure [Pa]')
ax4.grid(True, linestyle=':', color='lightgray')
ax4.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig4.savefig('Report_Data/Helical_Deformation_Pressure.png', dpi=600, bbox_inches='tight')
plt.close(fig4)

# --- PLOT 5: Wire Skin Friction (Viscous Drag) ---
fig5, ax5 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Viscous_Drag_Wire', 
    hue='Helix_Diameter_Label', palette=palette, marker='v', markersize=5, ax=ax5
)
ax5.set_title('Wire Body Viscous Drag (Skin Friction) vs. Deformation')
ax5.set_xlabel('Number of Revolutions')
ax5.set_ylabel('Viscous Drag Force [N]')
ax5.grid(True, linestyle=':', color='lightgray')
ax5.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig5.savefig('Report_Data/Helical_Deformation_ViscousDrag.png', dpi=600, bbox_inches='tight')
plt.close(fig5)

# --- PLOT 6: Wire Pressure Drag (Form Drag) ---
fig6, ax6 = plt.subplots(figsize=(5, 3.5))
sns.lineplot(
    data=df_summary, x='Helix_Rev', y='Pressure_Drag_Wire', 
    hue='Helix_Diameter_Label', palette=palette, marker='p', markersize=5, ax=ax6
)
ax6.set_title('Wire Body Pressure Drag (Form Drag) vs. Deformation')
ax6.set_xlabel('Number of Revolutions')
ax6.set_ylabel('Pressure Drag Force [N]')
ax6.grid(True, linestyle=':', color='lightgray')
ax6.legend(title='Helix Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fig6.savefig('Report_Data/Helical_Deformation_PressureDrag.png', dpi=600, bbox_inches='tight')
plt.close(fig6)

print("Plotting complete. Calculated separated pressure and viscous drag metrics.")