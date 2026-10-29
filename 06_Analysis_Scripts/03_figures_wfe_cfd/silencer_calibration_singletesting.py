import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_tfa_data(filepath):
    if not os.path.exists(filepath):
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
            
    df['P_Pa'] = df['TFA_pressure_IS'] * 100.0
    df['Q_Nm3h'] = df['TFA_flow_IS'] / 100.0
    return df[df['Q_Nm3h'] > 0.1]

def main():
    # -------------------------------------------------------------
    # IEEE Publication Standard Matplotlib Settings
    # -------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,           
        'axes.labelsize': 10,      
        'axes.titlesize': 10,      
        'xtick.labelsize': 8,      
        'ytick.labelsize': 8,      
        'legend.fontsize': 8,      
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'axes.linewidth': 0.8,     
        'xtick.direction': 'in',   
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.autolayout': True
    })
    
    # 1. Load Baseline Data (Single Test Base: Valve + 10cm Tube + Connector)
    df_base_1 = load_tfa_data("TFA_20250903_1622.csv")
    df_base_2 = load_tfa_data("TFA_20250903_1635.csv")
    df_base = pd.concat([df_base_1, df_base_2]).sort_values(by='Q_Nm3h')

    # 2. Load Single Silencer Data (Da-C10-7)
    df_sil_1 = load_tfa_data("TFA_20250909_1537.csv")
    df_sil_2 = load_tfa_data("TFA_20250909_1551.csv")
    df_sil = pd.concat([df_sil_1, df_sil_2]).sort_values(by='Q_Nm3h')

    if df_base.empty or df_sil.empty:
        print("Data files missing.")
        return

    # 3. Fit Polynomials
    def porous_fit(q, a, b):
        return a * q**2 + b * q

    max_Q = min(df_base['Q_Nm3h'].max(), df_sil['Q_Nm3h'].max())
    df_b_fit = df_base[df_base['Q_Nm3h'] <= max_Q]
    df_s_fit = df_sil[df_sil['Q_Nm3h'] <= max_Q]

    popt_base, _ = curve_fit(porous_fit, df_b_fit['Q_Nm3h'], df_b_fit['P_Pa'])
    popt_sil, _ = curve_fit(porous_fit, df_s_fit['Q_Nm3h'], df_s_fit['P_Pa'])

    # Isolate pure silencer math
    a_pure = popt_sil[0] - popt_base[0]
    b_pure = popt_sil[1] - popt_base[1]

    # 4. Extract Darcy-Forchheimer Coefficients
    D, dx, rho, mu = 0.020, 0.050, 1.225, 1.789e-5
    A = np.pi * (D / 2)**2
    C2 = (2 * a_pure * (3600 * A)**2) / (rho * dx)
    inv_alpha = (b_pure * (3600 * A)) / (mu * dx)

    print(f"Isolated Silencer Polynomial: P = {a_pure:.2f} * Q^2 + {b_pure:.2f} * Q")
    print(f"Inertial Resistance (C2)      = {C2:,.2f} m^-1")
    print(f"Viscous Resistance (1/alpha)  = {inv_alpha:.2e} m^-2")

    # 5. Plotting
    # IEEE Double Column Width (7.16 inches)
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    
    Q_range = np.linspace(0, max_Q, 100)
    
    # Plotting RAW data with tiny dot markers so it looks like a scatter plot, not a thick line
    ax.scatter(df_s_fit['Q_Nm3h'], df_s_fit['P_Pa']/100, 
               label='Raw Data: Silencer Attached', color='#002F6C', s=4, alpha=0.3, marker='.')
    ax.scatter(df_b_fit['Q_Nm3h'], df_b_fit['P_Pa']/100, 
               label='Raw Data: Baseline (No Silencer)', color='#6A8A22', s=4, alpha=0.3, marker='.')
    
    # Plotting the mathematical fit on top with a distinct red dashed line
    ax.plot(Q_range, (a_pure * Q_range**2 + b_pure * Q_range)/100, 
             label='Derived Mathematical Fit (Pure Silencer)', color='#DC1E26', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    ax.set_ylabel('Pressure Drop [mbar]')
    
    # Subtle grid
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    
    # Place legend in an optimal location
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)

    plt.savefig('ieee_silencer_single_calibration.png', dpi=600, bbox_inches='tight')
    print("Saved 'ieee_silencer_single_calibration.png'")
    plt.show()

if __name__ == "__main__":
    main()