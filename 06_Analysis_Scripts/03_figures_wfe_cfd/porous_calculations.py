import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import io

def load_stage_data(file_list):
    """
    Extracts the SYSTEM RESPONSE section from the text files, converts units, 
    and returns a clean DataFrame with both SI units (for physics) and 
    standard units (for plotting).
    """
    dfs = []
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"Warning: File '{filepath}' not found.")
            continue
            
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        block_lines = []
        in_system_response = False
        
        for line in lines:
            line_stripped = line.strip()
            if 'SYSTEM RESPONSE' in line_stripped:
                in_system_response = True
                continue
            if in_system_response:
                if line_stripped == "" or "eof" in line_stripped.lower() or "CONFIGURATION" in line_stripped:
                    break
                line_stripped = line_stripped.rstrip(';')
                block_lines.append(line_stripped)

        if block_lines:
            block_str = "\n".join(block_lines)
            df = pd.read_csv(io.StringIO(block_str), sep=';', decimal=',')
            
            if 'TFA_pressure_IS' in df.columns and 'TFA_flow_IS' in df.columns:
                # Core SI units for CFD calculations
                df['Q_m3s'] = (df['TFA_flow_IS'] / 100.0) / 3600.0
                df['P_Pa'] = df['TFA_pressure_IS'] * 100.0
                
                # Standard units for plotting
                df['Q_Nm3h'] = df['TFA_flow_IS'] / 100.0
                df['P_mbar'] = df['TFA_pressure_IS']
                
                dfs.append(df[['Q_m3s', 'P_Pa', 'Q_Nm3h', 'P_mbar']])

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # CRITICAL: Group by exact flow rates and take the mean 
    # This prevents duplicate X-values and smooths perfectly overlapping points
    combined_df = combined_df.groupby('Q_m3s', as_index=False).mean()
    
    # Filter out near-zero sensor noise and sort
    return combined_df[combined_df['Q_m3s'] > 0.00001].sort_values(by='Q_m3s')

def porous_fit(q, a, b):
    return a * q**2 + b * q

def process_and_plot_config(config_name, stages, colors, axes):
    print(f"\n{'='*65}")
    print(f" PROCESSING CONFIGURATION: {config_name}")
    print(f"{'='*65}")
    
    rho = 1.225
    mu = 1.789e-5
    ax1, ax2 = axes
    
    # 1. Pre-load all data files for this configuration
    loaded_data = []
    max_Q_Nm3h = float('inf')
    
    print("\n--- DATASET VERIFICATION ---")
    for stage in stages:
        df = load_stage_data(stage['files'])
        loaded_data.append(df)
        if not df.empty:
            print(f"[{stage['comp_name']:<35}] Loaded {len(df):>3} points | Range: {df['Q_m3s'].min():.6f} -> {df['Q_m3s'].max():.6f} m^3/s")
            # Track the maximum shared flow domain for visual plotting
            max_Q_Nm3h = min(max_Q_Nm3h, df['Q_Nm3h'].max())
        else:
            print(f"[{stage['comp_name']:<35}] EMPTY OR MISSING DATA!")
            
    porous_results = {}
            
    # 2. Iterate through the stages to perform the physics calculations
    for i, stage in enumerate(stages):
        if stage.get('calc_porous', False):
            
            ref_idx = 0 if 'Baseline' in stage['comp_name'] else 2
            
            df_porous = loaded_data[i].copy()
            df_ref = loaded_data[ref_idx].copy()
            
            if df_porous.empty or df_ref.empty:
                continue
            
            # --- THE SHARED OPERATING DOMAIN SAFEGUARD ---
            ref_min, ref_max = df_ref['Q_m3s'].min(), df_ref['Q_m3s'].max()
            por_min, por_max = df_porous['Q_m3s'].min(), df_porous['Q_m3s'].max()

            shared_min_flow = max(ref_min, por_min)
            shared_max_flow = min(ref_max, por_max)
            
            original_por_len = len(df_porous)
            df_porous = df_porous[(df_porous['Q_m3s'] >= shared_min_flow) & (df_porous['Q_m3s'] <= shared_max_flow)].copy()
            
            if df_porous.empty:
                print(f"❌ ERROR: Trimming removed all data for {stage['comp_name']}. Skipping calculation.")
                continue

            # --- INTERPOLATION & DELTA P CALCULATION ---
            f_ref = interp1d(df_ref['Q_m3s'], df_ref['P_Pa'], kind='linear', fill_value='extrapolate')
            p_ref_matched = f_ref(df_porous['Q_m3s'])
            
            delta_p = df_porous['P_Pa'] - p_ref_matched
            valid_mask = delta_p > 0
            
            q_valid = df_porous['Q_m3s'][valid_mask].values
            dp_valid = delta_p[valid_mask].values
            
            if len(q_valid) == 0:
                continue
                
            # --- VELOCITY CALCULATION ---
            D = stage['geom_D']
            L = stage['geom_L']
            A = np.pi * (D / 2.0)**2
            v_valid = q_valid / A
            
            # --- CURVE FIT & FLUENT COEFFICIENTS ---
            popt, _ = curve_fit(porous_fit, v_valid, dp_valid)
            a, b = popt[0], popt[1]
            
            # Apply the optional multiplier (defaults to 1.0 if not specified)
            mult = stage.get('coeff_mult', 1.0)
            
            C2 = ((2 * a) / (rho * L)) * mult
            inv_alpha = (b / (mu * L)) * mult
            
            mult_str = f" (x{mult} multiplier applied)" if mult != 1.0 else ""
            
            print(f"\n>> ANSYS FLUENT COEFFICIENTS: {stage['comp_name']}{mult_str}")
            print(f"   Delta P Equation: {a:.5f}v^2 + {b:.5f}v")
            print(f"   Inertial Resistance (C2) : {C2:.3e} m^-1")
            print(f"   Viscous Resistance (1/a) : {inv_alpha:.3e} m^-2\n")
            
            # Store formatted string for the Matplotlib text boxes
            plot_text = (f"CFD Porous Coefficients:\n"
                         f"Inertial ($C_2$): {C2:,.2f} m$^{{-1}}$\n"
                         f"Viscous ($1/\\alpha$): {inv_alpha:.2e} m$^{{-2}}$")
            porous_results[stage['comp_name']] = plot_text

    # 3. Generating visual polynomial fits specifically for smooth plotting
    stage_polys = []
    for df in loaded_data:
        if df is not None and not df.empty:
            df_fit = df[df['Q_Nm3h'] <= max_Q_Nm3h]
            # Fitting directly to standard units (Nm3/h & mbar) to draw the lines
            popt, _ = curve_fit(porous_fit, df_fit['Q_Nm3h'], df_fit['P_mbar'])
            stage_polys.append(popt)
        else:
            stage_polys.append(None)

    # -------------------------------------------------------------
    # Plotting on the provided row axes
    # -------------------------------------------------------------
    Q_range = np.linspace(0, max_Q_Nm3h, 100)

    # PANEL 1: Baseline TFA Effect
    if stage_polys[0] is not None and stage_polys[1] is not None:
        p_base = porous_fit(Q_range, *stage_polys[0])
        p_feed = porous_fit(Q_range, *stage_polys[1])
        
        ax1.plot(Q_range, p_base, label='FESTO Flow sensor SFAB', color='#666666', linewidth=1.5)
        ax1.fill_between(Q_range, 0, p_base, color='#E0E0E0', alpha=0.5)
        
        ax1.plot(Q_range, p_feed, label=stages[1]['comp_name'], color='#4A90E2', linewidth=1.5)
        ax1.fill_between(Q_range, p_base, p_feed, color='#D0E3F5', alpha=0.9, label='Pure Baseline Resistance')
        
        ax1.set_title(f'Effect of Baseline TFA (during {config_name} tests)', weight='bold')
        ax1.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
        ax1.set_ylabel('Total System Pressure [mbar]')
        ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
        ax1.legend(loc='upper left')
        
        feed_comp = stages[1]['comp_name']
        if feed_comp in porous_results:
            ax1.text(0.03, 0.77, porous_results[feed_comp], transform=ax1.transAxes, 
                     fontsize=9, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

    # PANEL 2: Silencer Effect
    if stage_polys[2] is not None and stage_polys[3] is not None:
        p_pre_sil = porous_fit(Q_range, *stage_polys[2])
        p_sil = porous_fit(Q_range, *stage_polys[3])
        
        c_line_pre, c_fill_pre, c_line_sil, c_fill_sil = colors
        
        ax2.plot(Q_range, p_pre_sil, label=f'System Pre-Silencer ({config_name})', color=c_line_pre, linewidth=1.5)
        ax2.fill_between(Q_range, 0, p_pre_sil, color=c_fill_pre, alpha=0.5)
        
        ax2.plot(Q_range, p_sil, label=stages[3]['comp_name'], color=c_line_sil, linewidth=1.5)
        ax2.fill_between(Q_range, p_pre_sil, p_sil, color=c_fill_sil, alpha=0.9, label='Pure Silencer Resistance')
        
        ax2.set_title(f'Effect of Exhaust Silencer ({config_name})', weight='bold')
        ax2.set_xlabel('Volumetric Flow Rate [Nm$^3$/h]')
        ax2.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
        ax2.legend(loc='upper left')
        
        sil_comp = stages[3]['comp_name']
        if sil_comp in porous_results:
            ax2.text(0.03, 0.77, porous_results[sil_comp], transform=ax2.transAxes, 
                     fontsize=9, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

def main():
    # Matplotlib Configuration Setup
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.frameon': True, 'legend.edgecolor': 'black', 'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in'
    })

    # ==========================================
    # Configuration Data Structures
    # ==========================================
    # Added 'coeff_mult': 5.5 to the Baseline TFA definitions
    
    stages_wc08 = [
        {'files': ['TFA_20250902_1514.csv', 'TFA_20250902_1004.csv'], 'comp_name': 'Base (SFAB)', 'calc_porous': False},
        {'files': ['TFA_20250902_1405.csv', 'TFA_20250902_1033.csv'], 'comp_name': 'Baseline TFA (during WC08 tests)', 'calc_porous': True, 'geom_D': 0.008, 'geom_L': 0.100, 'coeff_mult': 5.5},
        {'files': ['TFA_20250902_1316.csv', 'TFA_20250902_1219.csv'], 'comp_name': 'Pre-Silencer (WC08)', 'calc_porous': False},
        {'files': ['TFA_20250911_1152.csv', 'TFA_20250911_1144.csv'], 'comp_name': '+ Silencer (WC08)', 'calc_porous': True, 'geom_D': 0.025, 'geom_L': 0.020}
    ]
    colors_wc08 = ('#D35400', '#FAD7A0', '#1B4F72', '#A9CCE3') 

    stages_wc10 = [
        {'files': ['TFA_20250902_1514.csv', 'TFA_20250902_1004.csv'], 'comp_name': 'Base (SFAB)', 'calc_porous': False},
        {'files': ['TFA_20260723_1417.csv', 'TFA_20260723_1426.csv'], 'comp_name': 'Baseline TFA (during WC10 tests)', 'calc_porous': True, 'geom_D': 0.008, 'geom_L': 0.100, 'coeff_mult': 5.5},
        {'files': ['TFA_20260723_1213.csv', 'TFA_20260723_1227.csv'], 'comp_name': 'Pre-Silencer (WC10)', 'calc_porous': False},
        {'files': ['TFA_20260723_1309.csv', 'TFA_20260723_1256.csv'], 'comp_name': '+ Silencer (WC10)', 'calc_porous': True, 'geom_D': 0.025, 'geom_L': 0.020}
    ]
    colors_wc10 = ('#F5821F', '#FDE2CA', '#2E7D32', '#C8E6C9') 

    # ==========================================
    # Master Figure Setup (2x2 Grid)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')
    fig.canvas.manager.set_window_title('Performance Matrix - WC08 vs WC10')

    process_and_plot_config("WC08", stages_wc08, colors_wc08, axes[0])
    process_and_plot_config("WC10", stages_wc10, colors_wc10, axes[1])

    plt.tight_layout(h_pad=3.5, w_pad=2.0)
    
    plt.savefig('combined_resistance_matrix_2x2.png', dpi=600, bbox_inches='tight')
    print("\nSaved 'combined_resistance_matrix_2x2.png'")
    
    plt.show()

if __name__ == "__main__":
    main()