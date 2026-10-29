import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comprehensive_analysis(input_csv_path, tube_column_name='Tube_Diameter'):
    """
    Reads the experimental dataset, merges with metadata, separates by Tube Diameter,
    generates summaries, sorts IDs sequentially, and produces IEEE-compliant charts.
    """
    # Set IEEE publication style aesthetics
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'figure.autolayout': True, 'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })

    print(f"Loading dataset from {input_csv_path}...")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"The file {input_csv_path} was not found in the directory.")
        
    df = pd.read_csv(input_csv_path)
    
    # Safety Check: Ensure the tube diameter column actually exists
    if tube_column_name not in df.columns:
        print(f"\nCRITICAL ERROR: The column '{tube_column_name}' was not found in your CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        print("Please pass the correct column name for the tube type in the function call.")
        return

    # -------------------------------------------------------------
    # Metadata Definition & Merge
    # -------------------------------------------------------------
    metadata = [
        {"WIRE_uid": 1, "Color": "Orange", "Section": 0.5, "Length": 0.5},
        {"WIRE_uid": 2, "Color": "Orange", "Section": 0.5, "Length": 1.0},
        {"WIRE_uid": 3, "Color": "Orange", "Section": 0.5, "Length": 2.0},
        {"WIRE_uid": 4, "Color": "Orange", "Section": 0.5, "Length": 3.5},
        {"WIRE_uid": 5, "Color": "Blue", "Section": 1.0, "Length": 0.5},
        {"WIRE_uid": 6, "Color": "Blue", "Section": 1.0, "Length": 1.0},
        {"WIRE_uid": 7, "Color": "Blue", "Section": 1.0, "Length": 2.0},
        {"WIRE_uid": 8, "Color": "Blue", "Section": 1.0, "Length": 3.5},
        {"WIRE_uid": 9, "Color": "Yellow", "Section": 1.5, "Length": 0.5},
        {"WIRE_uid": 10, "Color": "Yellow", "Section": 1.5, "Length": 1.0},
        {"WIRE_uid": 11, "Color": "Yellow", "Section": 1.5, "Length": 2.0},
        {"WIRE_uid": 12, "Color": "Yellow", "Section": 1.5, "Length": 3.5},
        {"WIRE_uid": 13, "Color": "-", "Section": None, "Length": None},
        {"WIRE_uid": 14, "Color": "-", "Section": None, "Length": None},
        {"WIRE_uid": 15, "Color": "-", "Section": None, "Length": None},
        {"WIRE_uid": 16, "Color": "-", "Section": None, "Length": None},
        {"WIRE_uid": 17, "Color": "Purple", "Section": 4.0, "Length": 0.5},
        {"WIRE_uid": 18, "Color": "Purple", "Section": 4.0, "Length": 1.0},
        {"WIRE_uid": 19, "Color": "Purple", "Section": 4.0, "Length": 2.0},
        {"WIRE_uid": 20, "Color": "Purple", "Section": 4.0, "Length": 3.5},
        {"WIRE_uid": 21, "Color": "Red", "Section": 6.0, "Length": 0.5},
        {"WIRE_uid": 22, "Color": "Red", "Section": 6.0, "Length": 1.0},
        {"WIRE_uid": 23, "Color": "Red", "Section": 6.0, "Length": 2.0},
        {"WIRE_uid": 24, "Color": "Red", "Section": 6.0, "Length": 3.5},
    ]
    df_meta = pd.DataFrame(metadata)
    df_merged = pd.merge(df, df_meta, on='WIRE_uid', how='left')
    
    # FORMAT FIX: Create the combined zero-padded ID (e.g., WC08_UID01)
    df_merged['ID'] = df_merged[tube_column_name].astype(str) + '_UID' + df_merged['WIRE_uid'].astype(str).str.zfill(2)
    
    df_merged['Section_str'] = df_merged['Section'].astype(str) + ' $\mathrm{mm^2}$'
    df_merged['Length_str'] = df_merged['Length'].apply(lambda x: f"{x} m" if pd.notnull(x) else "Unknown / None")

    # -------------------------------------------------------------
    # STEP 1: Generate Split-UID Summary Table with Exact Row Sorting
    # -------------------------------------------------------------
    print("Calculating mean values separated by Tube Diameter and UID...")
    
    summary_df = df_merged.groupby([tube_column_name, 'WIRE_uid']).agg({
        'ID': 'first',
        'Pressure_Stable_Average (mbar)': 'mean',
        'Flow_Stable_Average (Nm^3/h)': 'mean',
        'Temp_Stable_Average (C)': 'mean',
        'Color': 'first',
        'Section': 'first',
        'Length': 'first'
    }).reset_index()
    
    # Reorder columns to put ID first
    cols = ['ID', tube_column_name, 'WIRE_uid'] + [c for c in summary_df.columns if c not in ['ID', tube_column_name, 'WIRE_uid']]
    summary_df = summary_df[cols]
    
    # -------------------------------------------------------------
    # EXPLICIT SORTING: Interleave WC08 and WC10 sequentially by UID
    # -------------------------------------------------------------
    summary_df = summary_df.sort_values(by=['WIRE_uid', tube_column_name]).reset_index(drop=True)
    
    # Rename columns for clarity
    summary_df.rename(columns={
        'Pressure_Stable_Average (mbar)': 'Mean_Pressure_Stable_Average_mbar',
        'Flow_Stable_Average (Nm^3/h)': 'Mean_Flow_Stable_Average_Nm3h',
        'Temp_Stable_Average (C)': 'Mean_Temperature_Stable_Average_C' 
    }, inplace=True)
    
    output_summary_path = "uid_summary_experimental_data.csv"
    summary_df.to_csv(output_summary_path, index=False, sep=';', decimal=',')
    print(f"Summary table successfully saved to {output_summary_path} with paired interleaved IDs.")

    # -------------------------------------------------------------
    # Official FAPS Color Palette Mapping
    # -------------------------------------------------------------
    def rgb_norm(r, g, b): return (r / 255.0, g / 255.0, b / 255.0)

    faps_section_colors = [
        '#D55E00',  # 0.5 mm² - Deep Orange
        '#0072B2',  # 1.0 mm² - Deep Marine Blue
        '#8CB31D',  # 1.5 mm² - Yellow-Green
        '#5D3A9B',  # 4.0 mm² - Deep Muted Purple
        '#B22222'   # 6.0 mm² - Brick Red
    ]

    faps_length_colors = [
        rgb_norm(106, 138, 34),  # 0.5 m
        rgb_norm(52, 103, 125),  # 1.0 m
        rgb_norm(0, 47, 108),    # 2.0 m
        rgb_norm(245, 130, 31)   # 3.5 m
    ]

    # -------------------------------------------------------------
    # PLOT 1: Pressure vs. Flow Categorized by Section AND Tube Type
    # -------------------------------------------------------------
    print("Generating Plot 1...")
    plt.figure(figsize=(7.5, 4.5))
    unique_sections = sorted(df_merged['Section'].dropna().unique())
    sec_color_map = {f"{sec} $\mathrm{{mm^2}}$": faps_section_colors[i] for i, sec in enumerate(unique_sections)}

    ax1 = sns.scatterplot(
        data=df_merged.dropna(subset=['Section']),
        x='Flow_Stable_Average (Nm^3/h)',
        y='Pressure_Stable_Average (mbar)',
        hue='Section_str',
        style=tube_column_name, 
        palette=sec_color_map,
        s=60,
        alpha=0.85,
        edgecolor='k',
        linewidth=0.5
    )
    
    plt.title('System Pressure vs. Flow Rate Categorized by Cross-Section', weight='bold')
    plt.xlabel('Stable Average Flow Rate ($\mathrm{Nm^3/h}$)')
    plt.ylabel('Stable Average Static Pressure ($\mathrm{mbar}$)')
    ax1.tick_params(direction='in', length=4, width=1)
    ax1.grid(True, linestyle=':', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.savefig('plot_pressure_vs_flow_by_section_ieee.png', dpi=600, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # PLOT 2: Effect of Length & Cross-Section on Pressure
    # -------------------------------------------------------------
    print("Generating Plot 2...")
    plt.figure(figsize=(7.5, 4.5))
    df_cat_plot = df_merged.dropna(subset=['Length', 'Section', 'Pressure_Stable_Average (mbar)'])
    
  
    ax2 = sns.lineplot(
        data=df_cat_plot,
        x='Length',
        y='Pressure_Stable_Average (mbar)',
        hue='Section_str',
        style=tube_column_name, 
        palette=sec_color_map,
        marker='o',
        linewidth=1.8,
        markersize=6
    )
    
    plt.title('Effect of Wire Length and Cross-Section on Stable Pressure', weight='bold')
    plt.xlabel('Wire Length ($\mathrm{m}$)')
    plt.ylabel('Stable Average Static Pressure ($\mathrm{mbar}$)')
    ax2.tick_params(direction='in', length=4, width=1)
    ax2.grid(True, linestyle=':', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.savefig('plot_length_vs_pressure_by_section_ieee.png', dpi=600, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # PLOT 3: Split-Panel Boxplot (Section Color)
    # -------------------------------------------------------------
    print("Generating Plot 3: Multi-panel Box Plot (IEEE Style)...")
    
    g3 = sns.catplot(
        data=df_merged.dropna(subset=['Section']).sort_values('WIRE_uid'),
        x='WIRE_uid',
        y='Pressure_Stable_Average (mbar)',
        hue='Section_str',
        col=tube_column_name,  
        kind='box',
        palette=sec_color_map,
        dodge=False,
        linewidth=1.0,
        fliersize=3,
        height=4.5,
        aspect=1.2
    )
    
    g3.fig.suptitle('Experimental Pressure Dispersion Across UIDs Categorized by Tube Type', weight='bold', y=1.05)
    g3.set_axis_labels('WIRE UID Type', 'Stable Average Static Pressure ($\mathrm{mbar}$)')
    for ax in g3.axes.flat:
        ax.tick_params(direction='in', length=4, width=1)
        ax.grid(True, linestyle=':', alpha=0.5)
    
    g3.savefig('plot_uid_pressure_boxplot_split_section_ieee.png', dpi=600, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # PLOT 4: Split-Panel Boxplot (Length Color)
    # -------------------------------------------------------------
    print("Generating Plot 4: Multi-panel Box Plot (IEEE Style)...")
    unique_lengths = sorted([l for l in df_merged['Length'].dropna().unique()])
    len_color_map = {f"{l} m": faps_length_colors[i] for i, l in enumerate(unique_lengths)}

    g4 = sns.catplot(
        data=df_merged.dropna(subset=['Length']).sort_values('WIRE_uid'),
        x='WIRE_uid',
        y='Pressure_Stable_Average (mbar)',
        hue='Length_str',
        col=tube_column_name,
        kind='box',
        palette=len_color_map,
        dodge=False,
        linewidth=1.0,
        fliersize=3,
        height=4.5,
        aspect=1.2
    )
    
    g4.fig.suptitle('Experimental Pressure Dispersion Across UIDs Categorized by Tube Type (Length)', weight='bold', y=1.05)
    g4.set_axis_labels('WIRE UID Type', 'Stable Average Static Pressure ($\mathrm{mbar}$)')
    for ax in g4.axes.flat:
        ax.tick_params(direction='in', length=4, width=1)
        ax.grid(True, linestyle=':', alpha=0.5)
    
    g4.savefig('plot_uid_pressure_boxplot_split_length_ieee.png', dpi=600, bbox_inches='tight')
    plt.close()

    print("\nAll IEEE-compliant split-dimensional analyses generated successfully!")

if __name__ == "__main__":
    input_file = "Stable_Phase_Summary__SystemStable_TRUE.csv"
    
    # !!! IMPORTANT !!! 
    # Change 'Tube_Diameter' below to the EXACT column name in your CSV that holds the WC08 / WC10 labels.
    generate_comprehensive_analysis(input_file, tube_column_name='Tube_Diameter')