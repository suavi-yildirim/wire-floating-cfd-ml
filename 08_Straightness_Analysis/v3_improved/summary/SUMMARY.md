# WFE straightness: summary so far

## Question
Does the initial shape of a wire (straightness, bending) explain its stable levitation pressure?

## Earlier result (v2, thesis chapter 6)
- 25 QC-ok wires (72 photos, 4–5 March). No association: Spearman rho ~ 0 (1.0 m, n = 16).
- Limits: 3–4 instances per type, shape measured lying on the board, not in the tube.

## What changed in the tracker (v3)
- One clamp only: the top tape. The span runs to the free tip.
- Tip: hand labels (wire side) are used where they exist; otherwise an automatic ferrule search.
- Robustness fixes: centroid divided by zero (row 0 spikes), clamp chosen from the wrong tape, offset of the rule band, spikes from the 181-column median.
- Angle QC: flags bends above 15° over 20 mm (limit is a parameter); cuts the usable wire at the first violation; bridges gaps up to 30 mm with a straight line.

## Runs
1. **Labelled test set (16 photos, 5393–5478):** all tracked, jumps < 3 mm. Angle QC flags IMG_5402 and IMG_5472 inside 90–490 mm; IMG_5478 beyond 500 mm.
2. **Full set (5393–5511, 72 photos): not rerun** with the latest tracker. The earlier full run was stopped and deleted.
3. **More_Photos (56 labelled, UID + instance + plane given):** 40 tracked, 16 no wire found, 16 two-plane wires, 14 with stable pressure matched.
   - Angle QC flags 25 of the 40 tracked photos inside the window (bends 20–68°). Most are likely tracking errors, not wire shape.

## Next step
Hand-marked points on the 41 priority sheets (`more/to_mark_priority`): 10 done (6006, 6007, 6009, 6011, 6012, 6014, 6016, 6018, 6020, 6026).
