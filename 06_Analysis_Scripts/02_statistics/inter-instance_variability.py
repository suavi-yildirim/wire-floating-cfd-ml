import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_ieee_compliant_faps_analysis(input_csv_path):
    """
    Reads the experimental dataset, merges with metadata, and generates 
    an IEEE-compliant publication-quality faceted boxplot using official FAPS colors.
    """
    # Set IEEE publication style aesthetics
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.autolayout': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })

    print(f"Loading dataset from {input_csv_path}...")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"The file {input_csv_path} was not found in the directory.")
        
    df = pd.read_csv(input_csv_path)
    
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
        {"WIRE_uid": 7, "Color": "Blue", "Section": 1.0, "Length": None},
        {"WIRE_uid": 8, "Color": "Blue", "Section": 1.0, "Length": None},
        {"WIRE_uid": 9, "Color": "Yellow", "Section": 1.5, "Length": 0.5},
        {"WIRE_uid": 10, "Color": "Yellow", "Section": 1.5, "Length": 1.0},
        {"WIRE_uid": 11, "Color": "Yellow", "Section": 1.5, "Length": 2.0},
        {"WIRE_uid": 12, "Color": "Yellow", "Section": 1.5, "Length": 3.5},
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
    
    # Filter valid sections and format labels
    df_plot = df_merged.dropna(subset=['Section']).copy()
    df_plot['Section_str'] = 'Cross-Section: ' + df_plot['Section'].astype(str) + ' $\mathrm{mm^2}$'

    # -------------------------------------------------------------
    # Official FAPS Color Palette Mapping (Normalized RGB)
    # -------------------------------------------------------------
    def rgb_norm(r, g, b):
        return (r / 255.0, g / 255.0, b / 255.0)

    faps_palette = [
        rgb_norm(106, 138, 34),  # Instance 1: FAPS-Grün dunkel
        rgb_norm(52, 103, 125),  # Instance 2: Hausfarbe Türkis
        rgb_norm(0, 47, 108),    # Instance 3: FAU-Blau
        rgb_norm(245, 130, 31),  # Instance 4: Sonderfa. Orange
        rgb_norm(220, 30, 38)    # Instance 5: Sonderfa. Echtrot
    ]

    # -------------------------------------------------------------
    # IEEE-Compliant Faceted Plot Generation
    # -------------------------------------------------------------
    print("Generating IEEE-Compliant Faceted Inter-Instance Plot...")
    
    g = sns.catplot(
        data=df_plot,
        x='WIRE_uid',
        y='Pressure_Stable_Average (mbar)',
        hue='WIRE_instance',
        kind='box',
        col='Section_str',
        col_wrap=2,
        sharex=False,
        sharey=True,
        palette=faps_palette,
        height=3.2,
        aspect=1.6,
        linewidth=1.0,
        fliersize=2
    )
    
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("WIRE UID Type", "Stable Average Static Pressure ($\mathrm{mbar}$)")
    
    # Clean ticks and borders for IEEE style
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.tick_params(direction='in', length=4, width=1)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    # Adjust layout and legend
    g._legend.set_title("Wire Instance")
    g.fig.subplots_adjust(top=0.9)
    
    output_path = 'plot_ieee_faceted_faps.png'
    g.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"IEEE-compliant plot successfully generated and saved to {output_path}")

if __name__ == "__main__":
    input_file = "Stable_Phase_Summary__SystemStable_TRUE.csv"
    generate_ieee_compliant_faps_analysis(input_file)