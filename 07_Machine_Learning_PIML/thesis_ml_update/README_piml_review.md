# PIML_03 review and thesis status, 17 Oct 2026

## Bottom line
Your PIML pipeline is in better shape than you probably feel it is. The grouped cross-validation is leakage-free, the calibration happens inside each fold, and for unseen wire types the physics-informed GPR lands at 1.7x (flow) and 1.4x (pressure) the experimental repeatability limit. That is a defensible RQ4 result. The remaining work is mostly consistency (three versions of the baseline and one outdated table are in circulation), one bug, and the missing ablation, which I have run. The bigger risk for submission is not ML: Ch. 7 has no text for the CFD validation (RQ1, RQ2) or the helical study (RQ3), Ch. 6 is empty and Ch. 8 is a skeleton.

## 1. PIML_03: findings, ranked
1. **Physics features help flow prediction, but not everywhere (new ablation, same folds).** GPR MAPE for flow, geometry only vs +physics: LOUO 13 -> 7 %, LOSO 54 -> 28 %, LOLO 10 -> 11 % (no gain). Wire weight alone gives 12 % (LOUO). Table tab:ml_ablation.
2. **The Darcy pressure features are redundant.** Replacing Darcy + dynamic pressure by the flow baseline alone gives equal results (LOUO 9 vs 8 %, LOLO 11 vs 11 %) or better (LOSO 20 vs 26 %).
3. **Pressure is the rig characteristic plus weight.** With measured flow: p = a_D Q^2 + b_D Q + c W/A_tube, c = 1.06, R^2 = 0.998, MAPE 1.7 % (5 coefficients, 794 runs). So the prediction problem is a flow-rate problem; pressure error is amplified flow error.
4. **LOSO fails for 4.0 and 6.0 mm^2.** Flow error +53 % / -30 % (WC08 / WC10) for 4.0; pressure +98 % in WC08 for 6.0. Ch. 5 currently says the 4.0 fold tests interpolation. Report it, it is a finding (blockage ratio > 0.4, gaps in the training range).
5. **Bug: gauge vs absolute.** `Pressure_Pa` is gauge (Build_MAT_Dataset.m l. 184), `physBaseline` adds P_amb. The standalone Darcy R^2 = -0.04 is a coincidence of means; consistent basis gives R^2 = -1.72 (MAPE 76 %). Surrogates are unaffected (constant offset).
6. **Baseline provenance.** Fitted C_D median 3.5 matches the CFD end-face coefficient (3.5 / 4.2), but c_f sits on its lower bound 0.8 in 27 of 29 folds (CFD: 1.16), so (C_D, c_f) are not identified. The fixed constants in the scripts (C_D = 0.88 in CrossVal, 1.168 in Build_MAT_Dataset, both commented "CFD") do not appear in CFD_Coefficients_per_case.csv. CFD constants without fit underestimate flow by 3.8 m^3/h (R^2 = 0.39).
7. **Three baseline numbers in circulation.** 1.5 mm^2 holdout: R^2 = 0.630 (Summary CSV, C_D = 0.88), 0.573 (RL benchmark CSV, C_D = 1.168), 0.558 (thesis Table, no matching file). The thesis table's best model (Medium tree, R^2 = 0.9779) is not in any v8 output (best there: Coarse tree 0.970). Both are replaced by the cross-validation section.
8. **Model selection on the test fold.** `Auto_Model_Holdout_Evaluation_LOSO_LOLO.m` fixes "Coarse Tree" for flow ("Top model, RMSE 1.08") after seeing the 1.5 mm^2 holdout; the CV pipeline uses GPR. Use one model family per claim and no selection on the test fold.
9. **LOSO GPR is implementation-sensitive.** MATLAB R^2 = 0.77, Python 0.47 (flow); LOUO/LOLO agree to 0.001. With 32 distinct training inputs, report LOSO as a range. The linear model with the physics feature is the stable extrapolator (flow R^2 0.83, pressure 0.92).
10. **Hysteresis is real.** LOUO residuals Rises vs Drops: +1.7 vs -5.8 kPa, p ~ 1e-30. Matches the friction argument in the stable-operating-point section of Ch. 5.
11. **RQ4 wording.** Ch. 1 said "drag forces", the models predict p_in and Q_N. Fixed.

## 2. Thesis status (updated zip)
| Chapter | State | Open |
|---|---|---|
| 1 Introduction | done | RQ4 reworded (patched) |
| 2 Foundations | done | 80/20 paragraph contradicted Ch. 5 (patched); no hybrid/physics-informed ML subsection |
| 3 State of the art | done | 1 TODO (cycle-time source) |
| 4 Need for action | done | ML scope bullet claimed CFD-derived coefficients as features (patched); mesh-independence decision (commented TODO) |
| 5 Methodology | experimental + numerical complete | ML part: baseline, GPR hyperparameters, LOLO, MAPE, ablation were missing or wrong (patched); several commented TODOs (GCI, y+ treatment, flow sensor reference, calibration factor) |
| 6 Implementation | one figure only | merge into 7 or write; title does not fit content |
| 7 Results | 7.1 straightness done; 7.2 CFD validation: one paragraph + placeholder bullets with example numbers; 7.3 energetics: placeholders; figs 4-12 without text; ML: replaced | helical (RQ3) absent; RQ2 absent |
| 8 Conclusion | skeleton, contains a Turkish note at the end | write after Ch. 7 |
| QA | 12 undefined refs (eq:blockage, eq:reynolds, eq:rmse, eq:equilibrium, sec:ml_validation, chap:need_for_action, ...), 2 duplicate labels (eq:drag_decomposition, fig:app_holdout_0_5m), FAPS-Thesis.pdf in the zip is the 2017 template, symbol drift Q_N vs \dot V_N | |

## 3. Next steps, in order of impact
1. **Ch. 7.2 CFD validation text (RQ1, RQ2).** The figures (07-12) exist, the text does not. This is the core of the thesis; ML is already ahead of it.
2. **Helical study (RQ3).** Needs Helical.zip (not visible to me in this session, see below).
3. **Run `MATLAB_ablation_patch.m`**, fix the gauge basis, rerun, and replace the Python ablation numbers (30 min). Then paste the patched chapters.
4. **Ch. 6/8 and QA list** (undefined refs, duplicates, \todo items, stale PDF).
5. Decide: SVR/ANN (Ch. 5 claimed them, they were never run): evaluate or drop.

## 4. Assumptions I made
- "Rises" = wire entered the window from below (from the column name; consistent with the sign of the residuals).
- C_D bound 0.4 and c_f bound 0.8 are yours; I did not change them.
- Python ablation uses the 40 distinct input vectors with replicate scatter as known noise; it reproduces MATLAB LOUO/LOLO.
