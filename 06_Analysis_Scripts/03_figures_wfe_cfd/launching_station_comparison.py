import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_tfa_data(filepath):
    """Robust parser that isolates the 'SYSTEM RESPONSE' table and ignores bad lines."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return pd.DataFrame()
    
    data_rows = []
    header = []
    with open(filepath, "r", encoding='utf-8', errors='replace') as f:
        in_system_response = False
        for line in f:
            line = line.strip()
            
            if 'TFA_pressure_SET' in line:
                in_system_response = True
                header = line.split(';')
                continue
            
            if in_system_response:
                if line == "" or "SYSTEM" in line or line == "eof":
                    break
                data_rows.append(line.split(';'))

    if not data_rows: return pd.DataFrame()

    df = pd.DataFrame(data_rows)
    if len(header) == df.shape[1]:
        df.columns = header
    else:
        df.columns = (header + [f"Col_{i}" for i in range(len(header), df.shape[1])])[:df.shape[1]]

    df = df.rename(columns={df.columns[0]: 'TFA_pressure_SET', 
                            df.columns[1]: 'TFA_pressure_IS', 
                            df.columns[2]: 'TFA_flow_IS'})

    for col in ['TFA_pressure_IS', 'TFA_flow_IS']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df['P_mbar'] = df['TFA_pressure_IS']
    df['Q_Nm3h'] = df['TFA_flow_IS'] / 100.0
    
    return df[df['Q_Nm3h'] > 0.1].copy().sort_values(by='Q_Nm3h')

def porous_fit(q, a, b):
    return a * q**2 + b * q

def main():
    # -------------------------------------------------------------
    # IEEE Publication Standard Matplotlib Settings
    # -------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.frameon': True, 'legend.edgecolor': 'black', 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.autolayout': True
    })

    # -------------------------------------------------------------
    # 1. Load Data for 8mm Channel (2025 Tests)
    # -------------------------------------------------------------
    df_base_08 = pd.concat([
        load_tfa_data("TFA_20250902_1405.csv"), load_tfa_data("TFA_20250902_1033.csv")
        #load_tfa_data("TFA_20260723_1417.csv"), load_tfa_data("TFA_20260723_1426.csv")
    ]).sort_values(by='Q_Nm3h')

    df_tot_08 = pd.concat([
        load_tfa_data("TFA_20250902_1326.csv"), load_tfa_data("TFA_20250902_1103.csv")
    ]).sort_values(by='Q_Nm3h')

    # -------------------------------------------------------------
    # 2. Load Data for 10mm Channel (2026 Tests)
    # -------------------------------------------------------------
    df_base_10 = pd.concat([
         load_tfa_data("TFA_20250902_1405.csv"), load_tfa_data("TFA_20250902_1033.csv")
        #load_tfa_data("TFA_20260723_1417.csv"), load_tfa_data("TFA_20260723_1426.csv")
    ]).sort_values(by='Q_Nm3h')

    df_tot_10 = pd.concat([
        load_tfa_data("TFA_20260723_1242.csv"), load_tfa_data("TFA_20260723_1248.csv")
    ]).sort_values(by='Q_Nm3h')

    if df_base_08.empty or df_tot_08.empty or df_base_10.empty or df_tot_10.empty:
        print("Data files missing. Ensure all 8 CSV files are in the directory.")
        return

    # -------------------------------------------------------------
    # 3. Independent Domain Matching
    # -------------------------------------------------------------
    max_Q_08 = min(df_base_08['Q_Nm3h'].max(), df_tot_08['Q_Nm3h'].max())
    max_Q_10 = min(df_base_10['Q_Nm3h'].max(), df_tot_10['Q_Nm3h'].max())
    
    df_base_08 = df_base_08[df_base_08['Q_Nm3h'] <= max_Q_08]
    df_tot_08 = df_tot_08[df_tot_08['Q_Nm3h'] <= max_Q_08]
    
    df_base_10 = df_base_10[df_base_10['Q_Nm3h'] <= max_Q_10]
    df_tot_10 = df_tot_10[df_tot_10['Q_Nm3h'] <= max_Q_10]

    # -------------------------------------------------------------
    # 4. Curve Fitting & Differential Math
    # -------------------------------------------------------------
    # 8mm Analysis
    popt_base_08, _ = curve_fit(porous_fit, df_base_08['Q_Nm3h'], df_base_08['P_mbar'])
    popt_tot_08, _ = curve_fit(porous_fit, df_tot_08['Q_Nm3h'], df_tot_08['P_mbar'])
    
    a_08 = popt_tot_08[0] - popt_base_08[0]
    b_08 = popt_tot_08[1] - popt_base_08[1]

    # 10mm Analysis
    popt_base_10, _ = curve_fit(porous_fit, df_base_10['Q_Nm3h'], df_base_10['P_mbar'])
    popt_tot_10, _ = curve_fit(porous_fit, df_tot_10['Q_Nm3h'], df_tot_10['P_mbar'])
    
    a_10 = popt_tot_10[0] - popt_base_10[0]
    b_10 = popt_tot_10[1] - popt_base_10[1]

    print(f"Isolated LS-08-A Polynomial: P = {a_08:.4f} * Q^2 + {b_08:.4f} * Q")
    print(f"Isolated LS-10-B Polynomial: P = {a_10:.4f} * Q^2 + {b_10:.4f} * Q")

    # -------------------------------------------------------------
    # 5. Plotting Validation
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    
    # Ensure the plot only covers the maximum valid physical range of both experiments
    global_max_Q = min(max_Q_08, max_Q_10)
    Q_range = np.linspace(0, global_max_Q, 100)
    
    # Plot Isolated LS-08
    P_08_iso = (a_08 * Q_range**2 + b_08 * Q_range)
    ax.plot(Q_range, P_08_iso, label='Isolated LS-08-A (8mm Channel)', color='#F5821F', linewidth=1.5)
    ax.fill_between(Q_range, 0, P_08_iso, color='#F5821F', alpha=0.1)

    # Plot Isolated LS-10
    P_10_iso = (a_10 * Q_range**2 + b_10 * Q_range)
    ax.plot(Q_range, P_10_iso, label='Isolated LS-10-B (10mm Channel)', color='#6A8A22', linewidth=1.5)
    ax.fill_between(Q_range, 0, P_10_iso, color='#6A8A22', alpha=0.1)

    ax.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    ax.set_ylabel('Pure Component Pressure Drop $\Delta$P [mbar]')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')

    plt.savefig('ieee_dual_baseline_comparison.png', dpi=600, bbox_inches='tight')
    print("Saved 'ieee_dual_baseline_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()