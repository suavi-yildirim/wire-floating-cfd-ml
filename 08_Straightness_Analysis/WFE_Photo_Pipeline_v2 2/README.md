# WFE wire-straightness photo pipeline (v2: 0.5 m and 1.0 m wires)

Data set: 72 photos (IMG_5393–IMG_5511, 4–5 March 2026, WC08 campaign), 37 photo groups.
36 groups were matched to a run session; IMG_5415/5416 (12:24, 4 March) has no session after it.
70 photos were tracked. IMG_5434 and IMG_5472 have no wire found, and 8 groups have only one view.

## Run (macOS: `brew install ffmpeg`; `pip3 install numpy pandas opencv-python scipy matplotlib`)
    python3 -I code/run_log.py <01_WFE_Raw folder> results/wfe_runlog_local_time.csv
    python3 -I code/wire_pipeline.py --photos <photo folder> --runlog results/wfe_runlog_local_time.csv \
            --template code/ruler_template_source.jpg --out <out>
    python3 -I code/qc_track_jumps.py <out>
    python3 -I code/join_pressure.py <out> results/wfe_runlog_local_time.csv <Stable_Phase_Summary.csv>
    python3 -I code/straightness_pressure_analysis.py <out>
    python3 -I code/make_photo_figures.py <out>
Runtime is about 15 min for 72 photos. The far-away 1.0 m photos are upscaled and run about 10 s each.

## Method changes since v1
- **Matching:** groups and sessions are now assigned in one global step (Hungarian method). The cost is the lead time from the photo to the start of the session, with at most 60 min allowed. A colour contradiction adds 60 min to the cost. Leaving a group unmatched costs 25 min. Red and purple insulation photograph dark, and orange can read as red, so these confusions are tolerated.
- **Scale:** if the rule labels are not found at the standard scale, the image is retried at 1.5×, 1.75× and 2×. This covers the photos of the 1.0 m wires, which were taken from further away.
- **Geometry:** the search band is fixed in mm above the label line, using values measured on well-tracked photos. The rule top edge lies 4–12 mm and the board top edge 38–62 mm above the label line. Previously a thick dark wire's own edge could be taken as the board edge.
- **Full span** (`full_*` columns, `photo_centrelines_fullspan.csv`, `twoplane_full_*`): the track runs between the outermost tapes, so for 1.0 m wires it continues past the end of the rule. **Beyond 500 mm the scale is extrapolated** with the 1-D projective fit of the 100–400 mm labels (`full_extrapolated`). The track is cut at the first lateral jump of more than 3 mm within 10 px, which stops it at the converging board edge. For the two-plane result, only the range common to both views is used.
- **Automatic track QC** (`qc_track_jumps.py`): if the centreline moves more than 2.5 mm laterally within 3 mm along the rule, the tracker has left the wire. This happens when a wire crosses the steel rule. The check flagged exactly the cases found by eye: IMG_5396, 5402, 5469, 5470 and 5478. `status_final` combines all checks.

## Key result (results/straightness_pressure_stats.csv)
Within-type relative stable pressure deviation vs maximum two-plane deviation r_max, 90–490 mm window, QC-ok only:
- 1.0 m, WC08, 5 wire types, n = 16: Spearman ρ = −0.01 (p = 0.98).
- Full span: ρ = −0.22 (p = 0.43).
- 0.5 m: no wire type has ≥3 QC-ok instances, so no within-type test is possible.

There is no detectable effect of the flat-on-board straightness on levitation pressure. Limitations: the photo shows the shape lying flat on the board, not in the tube; there are 3–4 instances per type; instance order equals run order.

## Folders
- `results/`: metrics, matches and pressures (column definitions in the script docstrings)
- `figures/paper/`: thesis figures (PDF and PNG), including your two example figures
- `figures/qc_montages/`: all tracked overlays (appendix) and the contact sheet
- `qc/`: overlay per photo. Green is the 90–490 mm window; light green is the full span beyond 500 mm, where the scale is extrapolated.
