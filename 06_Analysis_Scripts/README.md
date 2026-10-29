# WFE analysis scripts

Scripts behind the statistics in the thesis (Chapter 7). They read the data in
`../03_Experimental_Data` and write tables and figures to `output/`.

## Setup

Python 3 with `numpy`, `scipy`, `matplotlib` and `openpyxl`. Run from this folder:

```bash
cd 02_statistics
python3 01_run_inventory.py
python3 02_instance_analysis.py
python3 03_direction_effect.py
python3 04_friction_estimate.py
```

`wfe_common.py` (in the same folder) holds the shared paths and loaders; it is not run directly.

## Scripts and where their results appear

| Script | What it computes | Thesis location |
|---|---|---|
| `01_run_inventory.py` | Run files per tube, first and last run per tube, run counts per UID | Section 6 / 7 (run counts, WC08 vs WC10 timeline) |
| `02_instance_analysis.py` | Kruskal–Wallis test of stable pressure and flow between instances, per wire type and tube | Instance analysis; reason for reducing to three instances |
| `03_direction_effect.py` | Within-wire rising vs dropping medians, paired Wilcoxon, pooled Mann–Whitney on flow, slope figure | Section 7.4.6, Figure "Within-wire effect of the approach direction" |
| `04_friction_estimate.py` | Per-instance rising−dropping differences, pressure–weight calibration, half-band friction force | Section 7.4.6, exploratory friction paragraph and Figure "Exploratory friction estimate" |

## Verified results (run on the current data)

- 01: 813 run files (WC08 486, WC10 327); WC08 from 2026-03-03 08:51 to 2026-03-06 14:48, WC10 from 2026-03-24 11:20 to 2026-03-26 14:42.
- 02: significant instance differences in 14/20 WC08 and 13/20 WC10 wire types (p < 0.05).
- 03: WC08 16/16 wire types with rises > drops, median 58 mbar, p ≈ 3×10⁻⁵; WC10 11/17, median 125 mbar, p ≈ 0.035. Pooled flow WC08 8.9 vs 8.2 Nm³/h (p ≈ 2×10⁻⁹); WC10 21.0 vs 20.8 (n.s.).
- 04: WC08 n = 37 instances, median 40 mbar, slope 294 mbar/N (R² = 0.48), F_f ≈ 0.07 N (13 % of mean weight). WC10 n = 27, median 59 mbar, slope 821 mbar/N (R² = 0.77), F_f ≈ 0.04 N (7 %). Exploratory; see the caveats in Section 7.4.6.

## Notes

- The "lowest run counts" printed by `01_run_inventory.py` count raw run files per UID and tube. They do not match the lowest-count UIDs quoted earlier in the session (UID 8, 24, 4, 20), which were not reproduced by this script. Recheck those numbers before they are quoted in the thesis.
- The figure file `output/direction_effect_within_wire.pdf` is regenerated from the same data. The version in `figures/07_results/` was made in an earlier session with a different legend layout; use one of them consistently.
- `raw_commands/` contains the exploratory commands captured from the session transcript, including failed or superseded attempts. They are kept for provenance; the four scripts above are the clean versions.

## Folder layout (grouped by usage)

Copies of the scripts from the thesis folder. The originals were not moved or changed; the copies keep their folder structure inside each group. Their hard-coded paths (for example `D:\FAU\...` or absolute paths on the author's machine) may need editing before they run here.

| Folder | Usage | Contents |
|---|---|---|
| `01_data_preparation/` | Building the summary tables from the raw run files | `process_data.py`, `generate_uid_experimental_summary.py`, `TFA_Process_Data.py`, `detect_leak.py`, `make_5pct_net_mass_flow.py` (synthetic 5 % imbalance report file from WC08_UID01; not solver output) |
| `02_statistics/` | Statistics on the stable runs (this session and related analyses) | `01`–`04` analysis scripts, `wfe_common.py`, `inter-instance_variability.py`, `pressure_structure.py`, photo-to-pressure comparison and shape analyses |
| `03_figures_wfe_cfd/` | Thesis figures for WFE, TFA and CFD | `WFE_thesis_figures 2/code/make_figures.py` (Figures 1–12 of the WFE and CFD set), `Settling_Analysis/`, `mesh_stats.py`, CFD validation and helix plots, TFA calibration scripts; `session_2026-10-19/` (CFD, imbalance, convergence, air-volume and hardware figure scripts from the 2026-10-19 session, see its README) |
| `04_machine_learning/` | Surrogate models, physics baseline and ML figures | `thesis_ml_update/` (`ml_results_figures.py`, `piml_ablation.py`, `physics_cfd_constants.py`, MATLAB patch), `Scripts/` (MATLAB holdout, ablation and dataset scripts) |
| `05_photo_straightness/` | Photo-based straightness of the wires and its figures | `code/` and `scripts/` of the v3_improved and photo-pipeline folders |

Not copied: `03_PIML/Old/` and `Scripts - Copy` (older versions, kept only as reference in the original folder), `Thesis_Data/02_TFA_WFE_WTE/untitled.py` (scratch file), and data files.

## Photo straightness (05_photo_straightness)

Pipeline order used for the hand-annotated WC10 analysis (Section 7.1): `heic2jpg.py` (convert photos) → `make_wire_sheets.py` (clean sheets for annotation) → `fit_marked_wires.py` (spline through the magenta dots, scale from the cyan rule marks) → `analyse_marked.py` (deviation, bend, curvature, bending energy) → `pressure_flow_compare.py` / `explore_marked.py` (Spearman within wire type, permutation test, Benjamini–Hochberg) → `plot_marked.py`, `plot_true_scale_compact.py`, `make_thesis_figures.py` (thesis figures). `wire_pipeline.py`, `process_more.py`, `rerun_subset.py`, `qc_angle.py`, `qc_track_jumps.py`, `tip_sheet*.py` and `left_tiles.py` belong to the automatic tracker and its checks, which proved unreliable beyond about 1 m. `run_features.py` and `baseline_check.py` (in `02_statistics/`) build the per-run pressure features. Scripts take their data directory as an argument.
