import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_tfa_data(filepath):
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
    sns.set_theme(style="ticks")
    
    # Define file pairs for the progressive build
    stages = {
        "1. Valve & Flowmeter": ["TFA_20250902_1514.csv", "TFA_20250902_1004.csv"],
        "2. Added 2.2m Tube": ["TFA_20250902_1405.csv", "TFA_20250902_1033.csv"],
        "3. Added Launching Station": ["TFA_20250902_1326.csv", "TFA_20250902_1103.csv"],
        "4. Added 4.3m Tube (Full Empty)": ["TFA_20250902_1316.csv", "TFA_20250902_1219.csv"]
    }

    colors = ['#6A8A22', '#34677D', '#F5821F', '#DC1E26'] # FAPS Palette
    
    plt.figure(figsize=(8, 5))
    
    for (label, files), color in zip(stages.items(), colors):
        df_1 = load_tfa_data(files[0])
        df_2 = load_tfa_data(files[1])
        df_combined = pd.concat([df_1, df_2]).sort_values('Q_Nm3h')
        
        if not df_combined.empty:
            plt.plot(df_combined['Q_Nm3h'], df_combined['P_mbar'], 
                     label=label, color=color, linewidth=2, alpha=0.9)

    plt.title('Progressive System Flow Resistance (Component Validation)', weight='bold')
    plt.xlabel('Volumetric Flow Rate [$\mathrm{Nm^3/h}$]')
    plt.ylabel('System Static Pressure [$\mathrm{mbar}$]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Test Setup Stage")
    plt.tight_layout()
    plt.savefig('progressive_validation_matrix.png', dpi=400)
    print("Saved 'progressive_validation_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()