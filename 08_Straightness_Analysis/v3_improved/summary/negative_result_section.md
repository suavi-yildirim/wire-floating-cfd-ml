# Initial wire shape and levitation behaviour: negative result (draft text)

## Automated shape reconstruction and its limits

Automated reconstruction of the wire centreline from two-view photographs was attempted for both tube diameters. Segmentation performance depended strongly on illumination, background contrast, camera angle and perspective. For 0.5 m wires the automatic centreline was reliable in most photographs. For 1.0 m wires the rule scale covers only the first half of the free span, and the scale beyond it is extrapolated, which introduces systematic errors; in a number of photographs the wire could not be detected beyond the midpoint. The 2.0 m wires extended beyond the photographic frame and could not be evaluated in full. Automatic results were therefore retained only for the 0.5 m wires and the visible part of the 1.0 m wires, and were checked against the photographs in each case.

For the WC10 data set, the centrelines were annotated manually: the wire was marked with a sequence of points from the clamp to the free tip in each view, the millimetre scale was taken from two rule labels 300 mm apart, and a smooth curve was fitted through the points. Figure X shows the annotated photographs with the fitted trajectories overlaid.

## Statistical analysis

Shape descriptors were computed for each wire instance: maximum and RMS deviation from the chord, maximum change of slope over 20 mm, curvature magnitude and bending energy, arc excess, and the number of inflections. These were related to the stable levitation pressure, stable flow rate, lift-off pressure and time, time to stabilisation, overshoot and settling scatter of the matched run session. All values were expressed relative to the mean of the same wire type, so that between-type differences in weight and friction were removed. Nine shape descriptors and ten responses were tested (90 combinations) using Spearman rank correlation with a within-type permutation test, and p-values were corrected with the Benjamini–Hochberg procedure. No association reached significance; the smallest corrected q-value was 0.90. The strongest uncorrected correlation (maximum deviation against lift-off pressure, ρ = −0.45, p = 0.056) was negative, whereas a greater initial deviation would be expected to coincide with a higher pressure. Given the lack of significance, this sign carries no interpretive weight, and no conclusion is drawn about the reverse direction. The same absence of association held for the complete data and for subsets that excluded partially visible wires.

## Interpretation

The absence of a detectable effect may be attributed to four factors that the present design did not control.

1. **Standardised straightening.** All wires were passed through a two-plane roller straightener before each test series. This reduces the initial curvature to a narrow range (maximum chord deviation of roughly 5–15 mm over a 500 mm span, bend angles below 5° over 20 mm). A narrow range of the independent variable reduces the power to detect an effect, so the straightener is itself an important factor in this result.
2. **Limited number of instances per wire type.** Each wire type has two to five instances, and most analyses use three to four. With the observed variability, a moderate correlation would require on the order of 25–30 instances to reach conventional significance. The present data set contains 18 instances matched to pressure.
3. **Confounding between wire types.** Comparing wire types confounds shape with cross-section, length, weight and surface friction, which differ between types. The within-type analysis was therefore preferred, and it also showed no significant effect.
4. **Run order and rig drift.** The instance number coincides with run order within a day, so slow drift of the rig (for example temperature, see Table X) cannot be separated from instance-to-instance differences.

## Conclusion of this analysis

Within the range of initial shapes that remained after straightening, no significant influence of initial straightness on stable levitation pressure or flow rate was detected. This result does not exclude an effect for wires with larger initial curvature, or for a larger number of instances per type. The initial shape was characterised, no significant effect was found within the tested range, and any influence of straightness is below the detection limit of the present experiment.
