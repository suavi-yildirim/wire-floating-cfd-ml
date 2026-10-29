import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_stage_data(file_list):
    """Robust parser that loads and concatenates multiple TFA CSV files."""
    data_rows = []
    header = []
    
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found.")
            continue
            
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            in_system = False
            for line in f:
                line = line.strip()
                if 'TFA_pressure_SET' in line:
                    in_system = True
                    if not header:
                        header = line.split(';')
                    continue
                if in_system:
                    if line == "" or "SYSTEM" in line or line == "eof":
                        break
                    data_rows.append(line.split(';'))

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)
    if len(header) == df.shape[1]:
        df.columns = header
    else:
        df.columns = (header + [f"Col_{i}" for i in range(len(header), df.shape[1])])[:df.shape[1]]

    df = df.rename(columns={df.columns[0]: 'P_SET', df.columns[1]: 'P_IS', df.columns[2]: 'Q_IS'})
    for col in ['P_IS', 'Q_IS']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df['P_mbar'] = df['P_IS']
    df['Q_Nm3h'] = df['Q_IS'] / 100.0
    return df[df['Q_Nm3h'] > 0.1].copy().sort_values(by='Q_Nm3h')

def porous_fit(q, a, b):
    return a * q**2 + b * q

def main():
    # -------------------------------------------------------------
    # IEEE Publication Standard Matplotlib Settings
    # -------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
        'legend.frameon': True, 'legend.edgecolor': 'black', 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.autolayout': True
    })

    # -------------------------------------------------------------
    # Define the 5 Progressive Stages of the 10mm Setup
    # -------------------------------------------------------------
    stages = [
        {
            'label': '1. Base Valve/Flowmeter (VF)',
            'files': ['TFA_20250902_1514.csv', 'TFA_20250902_1004.csv'],
            'line_col': '#666666', 'fill_col': '#E0E0E0',
            'comp_name': 'Base Rig Resistance'
        },
        {
            'label': '2. Add 2.2m Feed Tube (FT-08-2.2)',
            'files': ['TFA_20260723_1417.csv', 'TFA_20260723_1426.csv'],
            'line_col': '#4A90E2', 'fill_col': '#D0E3F5',
            'comp_name': '+ Feed Tube'
        },
        {
            'label': '3. Add Launching Station (LS-10-B)',
            'files': ['TFA_20260723_1242.csv', 'TFA_20260723_1248.csv'],
            'line_col': '#6A8A22', 'fill_col': '#D8E2C3',
            'comp_name': '+ Launching Station (10mm)'
        },
        {
            'label': '4. Add 4.3m Downstream Tube',
            'files': ['TFA_20260723_1213.csv', 'TFA_20260723_1227.csv'],
            'line_col': '#F5821F', 'fill_col': '#FDE2CA',
            'comp_name': '+ Downstream Tube'
        },
        {
            'label': '5. Add Silencer (Da-B12-7)',
            'files': ['TFA_20260723_1309.csv', 'TFA_20260723_1256.csv'],
            'line_col': '#002F6C', 'fill_col': '#B3C1D2',
            'comp_name': '+ Silencer'
        }
    ]

    # Load all data and find global max overlapping flow rate
    dfs = []
    max_Q = float('inf')
    
    for stage in stages:
        df = load_stage_data(stage['files'])
        if not df.empty:
            dfs.append(df)
            max_Q = min(max_Q, df['Q_Nm3h'].max())
        else:
            dfs.append(None)

    if not any(df is not None for df in dfs):
        print("No valid CSV files found in the directory.")
        return

    # -------------------------------------------------------------
    # Plotting: Cumulative Resistance Stack
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    Q_range = np.linspace(0, max_Q, 100)
    
    P_previous = np.zeros_like(Q_range)
    
    for i, stage in enumerate(stages):
        df = dfs[i]
        if df is None: continue
            
        # Match domain strictly
        df_fit = df[df['Q_Nm3h'] <= max_Q]
        popt, _ = curve_fit(porous_fit, df_fit['Q_Nm3h'], df_fit['P_mbar'])
        
        # Calculate current total system curve
        P_current = porous_fit(Q_range, *popt)
        
        # Plot the top line for this stage
        ax.plot(Q_range, P_current, label=stage['label'], color=stage['line_col'], linewidth=1.5, linestyle='-')
        
        # Fill the gap between the previous stage and this stage (shows pure component resistance)
        ax.fill_between(Q_range, P_previous, P_current, color=stage['fill_col'], alpha=0.8, label=stage['comp_name'])
        
        # Update previous curve for the next iteration
        P_previous = P_current

    ax.set_title('Cumulative Pneumatic System Resistance (10mm Configuration)', weight='bold')
    ax.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    ax.set_ylabel('Total System Pressure [mbar]')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    
    # Optional: Format legend into two columns if it gets too tall
    ax.legend(loc='upper left', frameon=True, edgecolor='black', ncol=2, title="Total Lines & Isolated Contributions")

    plt.savefig('cumulative_system_resistance.png', dpi=600, bbox_inches='tight')
    print("Saved 'cumulative_system_resistance.png'")
    plt.show()

if __name__ == "__main__":
    main()