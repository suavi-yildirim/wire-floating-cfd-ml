import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_tfa_data(filepath):
    """
    Parses the TFA CSV files, skips metadata, handles European number formats, 
    and converts units to standard SI based on custom test-rig outputs.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Please check the path.")
        return pd.DataFrame()

    # Find where the actual data table starts
    skip_idx = 0
    with open(filepath, "r", encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if 'TFA_pressure_SET' in line:
                skip_idx = i
                break
    
    # Read the CSV as strings to safely handle commas
    df = pd.read_csv(filepath, sep=";", skiprows=skip_idx, dtype=str)
    
    # Clean up non-data rows (eof, SYSTEM RESPONSE headers, etc.)
    df = df.dropna(subset=['TFA_pressure_SET'])
    df = df[df['TFA_pressure_SET'] != 'eof']
    df = df[~df['TFA_pressure_SET'].str.contains('SYSTEM', na=False)]
    df = df[~df['TFA_pressure_SET'].str.contains('Pressure_SET', na=False)]
    
    # Convert European commas to dots and cast to float
    for col in ['TFA_pressure_SET', 'TFA_pressure_IS', 'TFA_flow_IS', 'TFA_temp_IS']:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.').astype(float)
            
    # Apply correct units based on test-rig scaling
    # Pressure: mbar to Pascal (1 mbar = 100 Pa)
    df['P_Pa'] = df['TFA_pressure_IS'] * 100.0
    # Flow: Test rig output requires division by 100 for Nm^3/h
    df['Q_Nm3h'] = df['TFA_flow_IS'] / 100.0
    # Temperature: Test rig output requires division by 10 for degrees Celsius
    if 'TFA_temp_IS' in df.columns:
        df['Temp_C'] = df['TFA_temp_IS'] / 10.0
    
    # Filter out zero or near-zero flows to prevent curve fitting errors
    df = df[df['Q_Nm3h'] > 0.1]
    return df

def main():
    # 1. Load Baseline Data (Tube System without Silencer)
    print("Loading Baseline Data...")
    df_base_1 = load_tfa_data("TFA_20250902_1316.csv")
    df_base_2 = load_tfa_data("TFA_20250902_1219.csv")
    df_base = pd.concat([df_base_1, df_base_2]).sort_values(by='Q_Nm3h')

    # 2. Load Total System Data (With Da-C10-7 Silencer)
    print("Loading Total System Data...")
    df_tot_1 = load_tfa_data("TFA_20250911_1152.csv")
    df_tot_2 = load_tfa_data("TFA_20250911_1144.csv")
    df_tot = pd.concat([df_tot_1, df_tot_2]).sort_values(by='Q_Nm3h')

    if df_base.empty or df_tot.empty:
        print("Error: Missing data. Please ensure the CSV files are in the same folder as this script.")
        return

    # 3. Domain Matching
    # Filter baseline data to match the maximum flow rate of the total system
    max_Q_tot = df_tot['Q_Nm3h'].max()
    df_base_filtered = df_base[df_base['Q_Nm3h'] <= max_Q_tot * 1.1]

    # 4. Polynomial Curve Fitting (Delta P = a*Q^2 + b*Q)
    # Using scipy.optimize.curve_fit to minimize actual pressure error
    def porous_fit(q, a, b):
        return a * q**2 + b * q

    # Fit Baseline
    popt_base, _ = curve_fit(porous_fit, df_base_filtered['Q_Nm3h'], df_base_filtered['P_Pa'])
    a_base_f, b_base_f = popt_base

    # Fit Total System
    popt_tot, _ = curve_fit(porous_fit, df_tot['Q_Nm3h'], df_tot['P_Pa'])
    a_tot, b_tot = popt_tot

    # 5. The Differential Method (Isolating the Silencer)
    a_sil_f = a_tot - a_base_f
    b_sil_f = b_tot - b_base_f

    print("\n--- Polynomial Curve Fits ---")
    print(f"Tube System without Silencer: P = {a_base_f:.2f} * Q^2 + {b_base_f:.2f} * Q")
    print(f"Total System with Silencer:   P = {a_tot:.2f} * Q^2 + {b_tot:.2f} * Q")
    print(f"Isolated Silencer:            P = {a_sil_f:.2f} * Q^2 + {b_sil_f:.2f} * Q")

    # 6. Physical Geometry and Fluid Parameters for Fluent
    D = 0.020  # m (20mm diameter)
    A = np.pi * (D / 2)**2  # m^2 (Cross-sectional Area)
    dx = 0.050  # m (40mm length/thickness)
    rho = 1.225  # kg/m^3 (Standard air density)
    mu = 1.789e-5  # kg/(m*s) (Standard air dynamic viscosity)

    # 7. Extract Darcy-Forchheimer Coefficients
    # Inertial Resistance (C2)
    C2_f = (2 * a_sil_f * (3600 * A)**2) / (rho * dx)
    
    # Viscous Resistance (1/alpha)
    inv_alpha_f = (b_sil_f * (3600 * A)) / (mu * dx)

    print("\n--- Final Ansys Fluent Coefficients ---")
    print(f"Inertial Resistance (C2)      = {C2_f:,.2f} m^-1")
    print(f"Viscous Resistance (1/alpha)  = {inv_alpha_f:.2e} m^-2")

    # 8. Visualization
    Q_range_f = np.linspace(0, max_Q_tot, 100)
    P_base_fit_f = a_base_f * Q_range_f**2 + b_base_f * Q_range_f
    P_tot_fit = a_tot * Q_range_f**2 + b_tot * Q_range_f
    P_sil_fit_f = a_sil_f * Q_range_f**2 + b_sil_f * Q_range_f

    # Plot settings based on user request
    plt.figure(figsize=(9, 6))
    
    plt.scatter(df_tot['Q_Nm3h'], df_tot['P_Pa']/100, label='Data: Total System with the Silencer', s=10, alpha=0.5, color='blue')
    plt.scatter(df_base_filtered['Q_Nm3h'], df_base_filtered['P_Pa']/100, label='Data: Tube System without the Silencer', s=10, alpha=0.5, color='green')

    plt.plot(Q_range_f, P_tot_fit/100, color='darkblue', linewidth=2, label='Polynomial Fit: Total System with the Silencer')
    plt.plot(Q_range_f, P_base_fit_f/100, color='darkgreen', linewidth=2, label='Polynomial Fit: Tube System without the Silencer')
    plt.plot(Q_range_f, P_sil_fit_f/100, color='red', linestyle='--', linewidth=2, label='Derived: Isolated Silencer')

    plt.title('Pressure Drop Isolation')
    plt.xlabel('Volumetric Flow Rate [Nm$^3$/h]')
    plt.ylabel('Pressure Drop [mbar]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('silencer_isolation_final.png')
    
    print("\nPlot saved successfully as 'silencer_isolation_final.png'.")
    plt.show()

if __name__ == "__main__":
    main()