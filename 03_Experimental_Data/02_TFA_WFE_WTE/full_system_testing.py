import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_tfa_data(filepath):
    """
    Loads, cleans, and converts raw test rig data into SI/Standard units.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return pd.DataFrame()
    
    skip_idx = 0
    with open(filepath, "r", encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if 'TFA_pressure_SET' in line:
                skip_idx = i
                break
                
    df = pd.read_csv(filepath, sep=";", skiprows=skip_idx, dtype=str)
    df = df.dropna(subset=['TFA_pressure_SET'])
    df = df[df['TFA_pressure_SET'] != 'eof']
    df = df[~df['TFA_pressure_SET'].str.contains('SYSTEM|Pressure_SET', na=False, case=False)]
    
    for col in ['TFA_pressure_IS', 'TFA_flow_IS']:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.').astype(float)
            
    df['P_mbar'] = df['TFA_pressure_IS']
    df['Q_Nm3h'] = df['TFA_flow_IS'] / 100.0
    return df[df['Q_Nm3h'] > 0.1].sort_values('Q_Nm3h')

def main():
    # -------------------------------------------------------------
    # IEEE Publication Standard Matplotlib Settings
    # -------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,           # General font size
        'axes.labelsize': 10,      # X and Y label size
        'axes.titlesize': 10,      # Title size (IEEE rarely uses figure titles, but kept for clarity)
        'xtick.labelsize': 8,      # Tick labels
        'ytick.labelsize': 8,      # Tick labels
        'legend.fontsize': 8,      # Legend text
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'axes.linewidth': 0.8,     # Standard border weight
        'xtick.direction': 'in',   # Inward facing ticks
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.autolayout': True
    })
    
    # Complete dataset map for Schalldämpfer-C Full WFE Testing
    boundaries = {
        "Da-C10-0 (Fully Open)": ["TFA_20250911_0928.csv", "TFA_20250911_0941.csv"],
        "Da-C10-1": ["TFA_20250911_1007.csv", "TFA_20250911_0959.csv"],
        "Da-C10-3": ["TFA_20250911_1017.csv", "TFA_20250911_1030.csv"],
        "Da-C10-5": ["TFA_20250911_1055.csv", "TFA_20250911_1044.csv"],
        "Da-C10-6": ["TFA_20250911_1112.csv", "TFA_20250911_1125.csv"],
        "Da-C10-7": ["TFA_20250911_1152.csv", "TFA_20250911_1144.csv"],
        "Da-C10-8": ["TFA_20250911_1202.csv", "TFA_20250911_1215.csv"],
        "Da-C10-9": ["TFA_20250911_1241.csv", "TFA_20250911_1233.csv"],
        "Da-C10-10": ["TFA_20250911_1300.csv", "TFA_20250911_1321.csv"],
        "Da-C10-11": ["TFA_20250911_1343.csv", "TFA_20250911_1335.csv"],
        "Da-C10-12": ["TFA_20250911_1352.csv", "TFA_20250911_1405.csv"],
        "Da-C10-13": ["TFA_20250911_1429.csv", "TFA_20250911_1421.csv"],
        "Da-C10-14": ["TFA_20250911_1439.csv", "TFA_20250911_1452.csv"],
        "Da-C10-15 (Fully Closed)": ["TFA_20250911_1514.csv", "TFA_20250911_1506.csv"]
    }

    # Standard CFD color gradient (Blue -> Cyan -> Green -> Yellow -> Red)
    cmap = plt.cm.turbo
    color_array = [cmap(i) for i in np.linspace(0, 1, len(boundaries))]

    # 7.16 inches is the exact maximum width for an IEEE double-column span
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    
    # Iterate through all files and plot them with the color gradient
    for idx, (label, files) in enumerate(boundaries.items()):
        df_1 = load_tfa_data(files[0])
        df_2 = load_tfa_data(files[1])
        
        # Combine the coarse and fine resolution datasets, sort by Flow
        df_combined = pd.concat([df_1, df_2]).sort_values('Q_Nm3h')
        
        if not df_combined.empty:
            # IEEE standard line width for data is generally 1.0 to 1.5 pt
            ax.plot(df_combined['Q_Nm3h'], df_combined['P_mbar'], 
                     label=label, color=color_array[idx], linewidth=1.2, alpha=0.95)
            
            # Fill the area under the "Fully Open" curve (the baseline)
            if idx == 0:
                ax.fill_between(df_combined['Q_Nm3h'], 0, df_combined['P_mbar'], color=color_array[idx], alpha=0.05)

    ax.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    ax.set_ylabel('Total Pressure Drop [mbar]')
    
    # IEEE prefers clean, unobtrusive grids if used at all
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    
    # Place legend outside the plot area, styled with a thin black border
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Exhaust Configuration", 
              frameon=True, edgecolor='black', fancybox=False)
    
    # Optional: Remove title if you plan to rely solely on the LaTeX \caption{}
    # ax.set_title('Complete Pneumatic Operating Envelope (Schalldämpfer-C)', weight='bold')

    # Save as highly uncompressed, 600 DPI image required for IEEE line art
    plt.savefig('ieee_full_system_envelope_cfd_colors.png', dpi=600, bbox_inches='tight')
    print("Saved 'ieee_full_system_envelope_cfd_colors.png'")
    plt.show()

if __name__ == "__main__":
    main()