import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_tfa_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return pd.DataFrame()
    
    data_rows, header = [], []
    with open(filepath, "r", encoding='utf-8', errors='replace') as f:
        in_system = False
        for line in f:
            line = line.strip()
            if 'TFA_pressure_SET' in line:
                in_system = True
                header = line.split(';')
                continue
            if in_system:
                if line == "" or "SYSTEM" in line or line == "eof": break
                data_rows.append(line.split(';'))

    if not data_rows: return pd.DataFrame()
    df = pd.DataFrame(data_rows)
    df.columns = (header + [f"Col_{i}" for i in range(len(header), df.shape[1])])[:df.shape[1]]
    df = df.rename(columns={df.columns[0]: 'P_SET', df.columns[1]: 'P_IS', df.columns[2]: 'Q_IS'})
    for col in ['P_IS', 'Q_IS']: 
        if col in df.columns: df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df['P_mbar'] = df['P_IS']
    df['Q_Nm3h'] = df['Q_IS'] / 100.0
    return df[df['Q_Nm3h'] > 0.1].sort_values('Q_Nm3h')

def porous_fit(q, a, b):
    return a * q**2 + b * q

def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.frameon': True, 'legend.edgecolor': 'black', 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.autolayout': True
    })

    # Load LS-10 Without Tube
    df_open = pd.concat([
        load_tfa_data("TFA_20260723_1242.csv"), load_tfa_data("TFA_20260723_1248.csv")
    ]).sort_values(by='Q_Nm3h')

    # Load LS-10 WITH 4.3m Tube
    df_tube = pd.concat([
        load_tfa_data("TFA_20260723_1213.csv"), load_tfa_data("TFA_20260723_1227.csv")
    ]).sort_values(by='Q_Nm3h')

    if df_open.empty or df_tube.empty:
        print("Missing data files for leak detection.")
        return

    # Match Domain
    max_Q = min(df_open['Q_Nm3h'].max(), df_tube['Q_Nm3h'].max())
    df_open = df_open[df_open['Q_Nm3h'] <= max_Q]
    df_tube = df_tube[df_tube['Q_Nm3h'] <= max_Q]

    # Fit Curves
    popt_open, _ = curve_fit(porous_fit, df_open['Q_Nm3h'], df_open['P_mbar'])
    popt_tube, _ = curve_fit(porous_fit, df_tube['Q_Nm3h'], df_tube['P_mbar'])

    # Plot
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    Q_range = np.linspace(0, max_Q, 100)
    
    P_open = porous_fit(Q_range, *popt_open)
    P_tube = porous_fit(Q_range, *popt_tube)

    ax.plot(Q_range, P_open, label='LS-10 Open to Atmosphere', color='#6A8A22', linewidth=2)
    ax.plot(Q_range, P_tube, label='LS-10 Connected to 4.3m Tube', color='#002F6C', linewidth=2, linestyle='--')
    
    # Highlight the difference
    ax.fill_between(Q_range, P_open, P_tube, color='red', alpha=0.1, label='Downstream Tube Resistance')

    ax.set_title('Leak Detection Validation: Launching Station 10', weight='bold')
    ax.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    ax.set_ylabel('Total System Pressure [mbar]')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')

    plt.savefig('ieee_ls10_leak_detection.png', dpi=600, bbox_inches='tight')
    print("Saved 'ieee_ls10_leak_detection.png'. If the red shaded area is very small, you have a leak!")
    plt.show()

if __name__ == "__main__":
    main()