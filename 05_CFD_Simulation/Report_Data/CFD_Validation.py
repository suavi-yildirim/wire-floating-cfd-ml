import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Load the Data ---
master_df = pd.read_excel("Master_CFD_Database.xlsx", sheet_name='Summary_Results')
exp_df = pd.read_csv("CFD_Experimental_Outcome.csv", sep=';', decimal=',')
merged_df = pd.merge(master_df, exp_df, left_on='Simulation_ID', right_on='ID', how='inner')

# --- 2. Clean and Prepare Data ---
clean_data = merged_df.dropna(subset=[
    'pressure-inlet', 'mass-flow-inlet', 'Stable_Average_Pressure[mbar]', 
    'Stable_Average_Flow[Nm^3/h]', 'total-wire-drag-force', 'Wire_Weight[N]', 
    'Wire_Length[m]', 'Wire_Channel'
]).copy()

# Uncomment this line if you want to filter out the WC10 anomaly again
# clean_data = clean_data[clean_data['Simulation_ID'] != 'WC10_UID07']

clean_data['CFD_Pressure_mbar'] = clean_data['pressure-inlet'] / 100.0
clean_data['Exp_Flow_m3_h'] = clean_data['Stable_Average_Flow[Nm^3/h]'].astype(str).str.replace(',', '.').astype(float)
clean_data['CFD_Flow_m3_h'] = (clean_data['mass-flow-inlet'] / clean_data['Air_Density']) * 3600.0

clean_data['Cross_Section_Label'] = clean_data['Wire _Cross_Section[mm^2]'].apply(lambda x: f"{x} mm²")
clean_data = clean_data.sort_values(['Wire_Channel', 'Wire _Cross_Section[mm^2]', 'Wire_Length[m]'])

# --- 3. Exact Reference Color Mapping ---
reference_colors = {
    '0.5 mm²': "#db7625",  
    '1.0 mm²': '#2787bd',  
    '1.5 mm²': '#9cbe3f',  
    '4.0 mm²': '#7557aa',  
    '6.0 mm²': '#bd4242'   
}

# --- 4. IEEE & Thesis Style Configuration ---
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,           
    'axes.labelsize': 8,      
    'axes.titlesize': 9,      
    'xtick.labelsize': 8,     
    'ytick.labelsize': 8,     
    'legend.fontsize': 5.5,     
    'legend.title_fontsize': 6.0, 
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Prepare tidy data for the scaling trend plots
tidy_data = pd.melt(
    clean_data,
    id_vars=['Wire_Length[m]', 'Cross_Section_Label', 'Wire_Channel'],
    value_vars=['total-wire-drag-force', 'Wire_Weight[N]'],
    var_name='Force_Type',
    value_name='Force_Value'
)
tidy_data['Force_Type'] = tidy_data['Force_Type'].replace({
    'total-wire-drag-force': 'CFD Drag Force',
    'Wire_Weight[N]': 'Wire Weight'
})

# --- 5. Generate Separate Plots for Each Channel ---
channels = ['WC08', 'WC10']

for channel in channels:
    subset = clean_data[clean_data['Wire_Channel'] == channel]
    tidy_subset = tidy_data[tidy_data['Wire_Channel'] == channel]
    
    # =====================================================================
    # FIGURE 1: Validation Panel (1x2 Layout)
    # =====================================================================
    # Increased width to 8.5 to accommodate external legends gracefully
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    
    # ---- Subplot A: Pressure Validation ----
    sns.scatterplot(
        data=subset, x='Stable_Average_Pressure[mbar]', y='CFD_Pressure_mbar', 
        hue='Cross_Section_Label', palette=reference_colors, style='Wire_Length[m]', 
        edgecolor='black', linewidth=0.5, s=35, ax=ax1
    )
    min_p = min(subset['Stable_Average_Pressure[mbar]'].min(), subset['CFD_Pressure_mbar'].min())
    max_p = max(subset['Stable_Average_Pressure[mbar]'].max(), subset['CFD_Pressure_mbar'].max())
    p_range = np.linspace(min_p, max_p, 100)
    ax1.plot(p_range, p_range, color='black', linestyle='--', linewidth=0.8, label='Ideal Agreement')
    ax1.plot(p_range, p_range * 1.1, color='gray', linestyle=':', linewidth=0.6, label='±10% Error Bounds')
    ax1.plot(p_range, p_range * 0.9, color='gray', linestyle=':', linewidth=0.6)
    
    ax1.set_title(f'(a) {channel} Inlet Pressure Validation')
    ax1.set_xlabel('Experimental Pressure [mbar]')
    ax1.set_ylabel('CFD Inlet Pressure [mbar]')
    ax1.grid(True, linestyle=':', color='lightgray')
    
    # Move legend outside
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    # ---- Subplot B: Flow Validation ----
    sns.scatterplot(
        data=subset, x='Exp_Flow_m3_h', y='CFD_Flow_m3_h', 
        hue='Cross_Section_Label', palette=reference_colors, style='Wire_Length[m]', 
        edgecolor='black', linewidth=0.5, s=35, ax=ax2
    )
    min_m = min(subset['Exp_Flow_m3_h'].min(), subset['CFD_Flow_m3_h'].min())
    max_m = max(subset['Exp_Flow_m3_h'].max(), subset['CFD_Flow_m3_h'].max())
    m_range = np.linspace(min_m, max_m, 100)
    ax2.plot(m_range, m_range, color='black', linestyle='--', linewidth=0.8, label='Ideal Agreement')
    ax2.plot(m_range, m_range * 1.1, color='gray', linestyle=':', linewidth=0.6, label='±10% Error Bounds')
    ax2.plot(m_range, m_range * 0.9, color='gray', linestyle=':', linewidth=0.6)
    
    ax2.set_title(f'(b) {channel} Volumetric Flow Validation')
    ax2.set_xlabel('Experimental Flow [m³/h]')
    ax2.set_ylabel('CFD Volumetric Flow [m³/h]')
    ax2.grid(True, linestyle=':', color='lightgray')
    
    # Move legend outside
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f'{channel}_Validation_Pressure_Flow.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    # =====================================================================
    # FIGURE 2: Force Comparison Panel (1x2 Layout)
    # =====================================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    
    # ---- Subplot A: Force Parity (Log-Log) ----
    sns.scatterplot(
        data=subset, x='Wire_Weight[N]', y='total-wire-drag-force', 
        hue='Cross_Section_Label', palette=reference_colors, style='Wire_Length[m]', 
        edgecolor='black', linewidth=0.5, s=35, ax=ax3
    )
    min_f = min(subset['Wire_Weight[N]'].min(), subset['total-wire-drag-force'].min())
    max_f = max(subset['Wire_Weight[N]'].max(), subset['total-wire-drag-force'].max())
    if min_f <= 0: min_f = 0.01 
    
    f_range = np.linspace(min_f, max_f, 100)
    ax3.plot(f_range, f_range, color='black', linestyle='--', linewidth=0.8, label='Drag = Weight + Friction')
    ax3.plot(f_range, f_range * 1.1, color='gray', linestyle=':', linewidth=0.6, label='±10% Error Bounds')
    ax3.plot(f_range, f_range * 0.9, color='gray', linestyle=':', linewidth=0.6)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_title(f'(a) {channel} Drag vs. Weight (Log-Log)')
    ax3.set_xlabel('Total Wire Weight [N] (Log)')
    ax3.set_ylabel('CFD Total Drag Force [N] (Log)')
    ax3.grid(True, which="both", linestyle=':', color='lightgray')
    
    # Move legend outside
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    # ---- Subplot B: Force Scaling (Log-Y) ----
    sns.lineplot(
        data=tidy_subset, x='Wire_Length[m]', y='Force_Value', 
        hue='Cross_Section_Label', style='Force_Type', palette=reference_colors, 
        markers=True, dashes={'CFD Drag Force': '', 'Wire Weight': (4, 2)}, 
        markersize=5, ax=ax4
    )
    
    ax4.set_yscale('log')
    ax4.set_title(f'(b) {channel} Force Scaling (Log-Y)')
    ax4.set_xlabel('Wire Length [m]')
    ax4.set_ylabel('Drag Force / Wire Weight [N] (Log)')
    ax4.grid(True, which="both", linestyle=':', color='lightgray')
    
    # Move legend outside
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f'{channel}_Comparison_Drag_Force_Wire_Weight_log.png', dpi=600, bbox_inches='tight')
    plt.close()

print("All individual channel plots generated and saved successfully!")